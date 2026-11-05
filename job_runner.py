"""Offline-testable durable-job worker with graceful cancellation draining.

No provider SDK imports. A running paid inference task is shielded from
cancellation so we do not advertise a safe retry while it is still running.
Hard process termination cannot be drained; a lease-expired job still requires
an explicit retry and may have incurred one prior paid call.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Callable
from durable_jobs import DurableJobStore

log = logging.getLogger(__name__)


async def run_job(store: DurableJobStore, job_id: str, processors: dict[str, Callable],
                  *, lease_seconds: int = 90, renew_every: float = 10) -> bool:
    """Claim, process, then persist. Safely drain in-flight work on cancellation.

    For graceful cancellation, shield underlying async/to_thread operations and
    keep the lease heartbeat running until the processor completes. If the
    processor finishes successfully, commit its result even when the caller has
    already cancelled this worker. Never automatically re-run a claimed job.
    """
    claimed = store.claim(job_id, lease_seconds=lease_seconds)
    if claimed is None:
        return False
    token, data, content_type, metadata = claimed
    kind = store.status(job_id, include_result=False)['kind']
    handler = processors.get(kind)
    if handler is None:
        store.fail(job_id, token)
        return False

    async def heartbeat():
        while True:
            await asyncio.sleep(renew_every)
            if not store.renew(job_id, token, lease_seconds=lease_seconds):
                return

    async def invoke_processor():
        if inspect.iscoroutinefunction(handler):
            return await handler(data, content_type, metadata)
        return await asyncio.to_thread(handler, data, content_type, metadata)

    hb = asyncio.create_task(heartbeat())
    work = asyncio.create_task(invoke_processor())
    cancelled_by_caller = False
    try:
        while True:
            try:
                # Shield keeps the coroutine + any to_thread work alive even
                # when the runner is cancelled by ASGI shutdown or a client.
                result = await asyncio.shield(work)
                break
            except asyncio.CancelledError:
                if work.done():
                    if work.cancelled():
                        # Processor itself cancelled; there is no work left to
                        # drain and we can now honestly report retry required.
                        raise
                    # It finished between cancellation and the await. Re-check
                    # its result on the next iteration, without restarting it.
                cancelled_by_caller = True
                # Repeated cancels also drain: don't release the lease early.
        if not isinstance(result, dict):
            raise TypeError('Processor must return a dict')
        committed = store.complete(job_id, token, result)
        if cancelled_by_caller:
            raise asyncio.CancelledError
        return committed
    except asyncio.CancelledError:
        # Only after `work` has finished/cancelled. We never report retry while
        # the processor is still able to finish an already-paid operation.
        if not work.done():
            # Defensive: no current branch should reach here. Keep the lease
            # alive and drain rather than exposing a premature retry.
            while not work.done():
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    if work.done():
                        break
                except Exception:
                    break
        # A completed result remains completed. Otherwise retry is required.
        if store.status(job_id, include_result=False)['status'] != 'done':
            store.fail(job_id, token, code='retry_required')
        raise
    except Exception as exc:
        log.error('Styligma job failed (job_id=%s, kind=%s, error_type=%s)',
                  job_id, kind, type(exc).__name__)
        store.fail(job_id, token)
        return False
    finally:
        hb.cancel()
        try:
            await hb
        except asyncio.CancelledError:
            pass


async def run_ready(store: DurableJobStore, processors: dict[str, Callable],
                    *, limit: int = 4) -> int:
    """Resume never-started queued jobs only; no paid retries for interrupted work."""
    store.cleanup()
    store.mark_interrupted()
    tasks = [run_job(store, jid, processors) for jid in store.ready_ids(limit=limit)]
    if not tasks:
        return 0
    outcome = await asyncio.gather(*tasks)
    return sum(outcome)
