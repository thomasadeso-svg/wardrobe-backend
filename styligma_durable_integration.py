"""Local-only opt-in wiring for the *real* Styligma processing callbacks.

This module does not enable itself, create Railway resources or deploy.
Apply the accompanying two-line backend-main.py integration only after review.
The production application is left exactly as it was unless the explicit
STYLIGMA_DURABLE_JOBS_ENABLED=1 flag is supplied and storage is verified.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from durable_jobs import DurableJobStore
from serverless_api_candidate import ServerlessJobApi
from durable_jobs import RateLimitExceeded
from fastapi import HTTPException

# The two status and two submit routes already exposed by current Styligma.
TARGET_PATHS = frozenset({
    '/process-clothing', '/process-clothing-status/{job_id}',
    '/scan-video', '/scan-status/{job_id}',
})


def _check_store_location(directory: str, environ: dict[str, str]) -> Path:
    """Fail closed on Railway if the job directory is not on a mounted volume.

    Local tests allow an ordinary temporary directory. A real Railway deployment
    must advertise RAILWAY_VOLUME_MOUNT_PATH and store all data inside it.
    """
    if not directory or not Path(directory).is_absolute():
        raise RuntimeError('Provide an absolute STYLIGMA_JOB_STORE_DIR')
    dest = Path(directory).resolve()
    running_on_railway = any(environ.get(v) for v in ('RAILWAY_PROJECT_ID', 'RAILWAY_ENVIRONMENT_ID'))
    if running_on_railway:
        mount = environ.get('RAILWAY_VOLUME_MOUNT_PATH')
        if not mount:
            raise RuntimeError('Durable jobs require an attached Railway volume')
        volume = Path(mount).resolve()
        if not volume.is_dir() or not dest.is_relative_to(volume):
            raise RuntimeError('Durable jobs path must be within Railway volume')
    # Caller remains responsible for volume capacity, privacy policy and access controls.
    return dest


def make_real_photo_processor(backend_module: Any):
    """Use exact current backend processing functions; do not create AI calls here."""
    async def process_photo(data: bytes, content_type: str, metadata: dict) -> dict:
        analysis, image = await asyncio.gather(
            backend_module._analyze_clothing_bytes(data, content_type),
            backend_module._remove_background_job_bytes(data),
        )
        if analysis.get('rejected'):
            return {'analysis': analysis}
        return {'analysis': analysis, 'image': image}
    return process_photo


def make_real_video_processor(video_module: Any):
    """Preserve video scanner's single-worker lock and cancellation rules."""
    async def process_video(data: bytes, content_type: str, metadata: dict) -> dict:
        path = None
        frame_issues: list[dict] = []
        cancel_event = threading.Event()
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                path = tmp.name
                tmp.write(data)
            async with video_module._scan_lock:
                # The current server's MULTI_ITEM_LIVE_ENABLED remains false.
                # A client flag alone must never enable new paid detection.
                can_detect_multi = bool(metadata.get('multi_item_capable')) and bool(
                    getattr(video_module, 'MULTI_ITEM_LIVE_ENABLED', False))
                detector = video_module._make_region_detector() if can_detect_multi else None
                work = asyncio.create_task(asyncio.to_thread(
                    video_module._process_video_sync, path,
                    region_detector=detector, frame_issues=frame_issues,
                    cancel_event=cancel_event))
                try:
                    items, frames_scanned, duplicates_removed = await asyncio.shield(work)
                except asyncio.CancelledError:
                    cancel_event.set()
                    # Do not release single-scan lock or unlink input while thread
                    # still works; mirrors the current video_scan.py behavior.
                    while not work.done():
                        try:
                            await asyncio.shield(work)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    with contextlib.suppress(Exception):
                        work.result()
                    raise
            if frames_scanned == 0:
                raise ValueError('No usable frames found')
            return {
                'items': [item.model_dump() if hasattr(item, 'model_dump') else
                          item.dict() if hasattr(item, 'dict') else item for item in items],
                'frames_scanned': frames_scanned,
                'duplicates_removed': duplicates_removed,
                'frame_issues': frame_issues,
            }
        finally:
            if path:
                with contextlib.suppress(OSError):
                    os.unlink(path)
    return process_video


def make_video_notification(video_module: Any):
    async def notify(push_token: str, job_id: str, record: dict | None):
        if not record or record.get('status') != 'done':
            await video_module._send_push_notification(
                push_token, 'Wardrobe scan failed',
                'Something went wrong processing your video. Please try again.',
                {'job_id': str(__import__('uuid').UUID(hex=job_id)), 'type': 'video_scan_error'})
            return
        n = len(record.get('result', {}).get('items') or ())
        msg = (f"Found {n} item{'s' if n != 1 else ''} — tap to review and add them"
               if n else 'Scan finished, but no items were clearly detected. Tap to try again.')
        from uuid import UUID
        await video_module._send_push_notification(
            push_token, 'Your wardrobe scan is ready ✨', msg,
            {'job_id': str(UUID(hex=job_id)), 'type': 'video_scan_complete'})
    return notify


def install_durable_jobs(app, backend_module, video_module, *, directory: str,
                         environ: dict[str, str] | None = None,
                         clock=None):
    """Opt-in replacement for four job endpoints. All other routes untouched.

    Before exposing the replacement, verify exact four old route patterns exist;
    this guards against silently changing the wrong version of Styligma.
    """
    env = dict(os.environ) if environ is None else environ
    root = _check_store_location(directory, env)
    existing = {getattr(route, 'path', '') for route in app.router.routes}
    missing = TARGET_PATHS - existing
    if missing:
        raise RuntimeError(f'Styligma endpoints changed; refusing install: {sorted(missing)}')

    # clock injection is for deterministic offline tests only.
    store = DurableJobStore(root, clock=clock or time.time)
    # Synchronous cleanup before accepting new requests after a cold boot.
    store.cleanup()
    api = ServerlessJobApi(store,
        photo_processor=make_real_photo_processor(backend_module),
        video_processor=make_real_video_processor(video_module),
        # Original in-RAM video cap resets every time Railway sleeps.
        # Persistent video cap uses the same 5-per-24h policy.
        video_rate_check=store.enforce_video_rate,
        on_video_complete=make_video_notification(video_module))
    # The legacy /scan-photo handler calls video_module._check_rate_limit at
    # request time. Make its quota durable and SHARED with /scan-video so a
    # sleep/restart cannot reset the original combined 5-per-day cap. This is
    # an opt-in quota-only change; its image processing code is untouched.
    original_limit = getattr(video_module, '_check_rate_limit', None)
    if not callable(original_limit):
        raise RuntimeError('Missing legacy scan-photo rate limit; refusing install')
    def _legacy_persistent_limit(network_key: str):
        try:
            store.enforce_video_rate(network_key)
        except RateLimitExceeded as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from None
    video_module._check_rate_limit = _legacy_persistent_limit

    # Swap only four routes, leaving outfit, gallery and other APIs in place.
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, 'path', '') not in TARGET_PATHS]
    app.include_router(api.router)

    # Queued jobs accepted just before a clean shutdown may start on next boot.
    # A processor that actually started is NEVER retried automatically.
    @app.on_event('startup')
    async def _start_resume():
        # No network calls. Already expired uploads are removed before serving.
        store.cleanup()
        app.state.styligma_resume_task = asyncio.create_task(api.resume_queued())
        app.state.styligma_maintenance_task = asyncio.create_task(api.maintenance_loop())

    @app.on_event('shutdown')
    async def _stop_maintenance():
        task = getattr(app.state, 'styligma_maintenance_task', None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app.state.styligma_durable_api = api
    return api


def enable_from_env(app, backend_module, video_module, environ: dict[str, str] | None = None):
    env = dict(os.environ) if environ is None else environ
    if env.get('STYLIGMA_DURABLE_JOBS_ENABLED') != '1':
        return None
    return install_durable_jobs(
        app, backend_module, video_module,
        directory=env.get('STYLIGMA_JOB_STORE_DIR', ''), environ=env)
