"""OFFLINE-TESTED FastAPI adapter candidate for Styligma's existing job URLs.

This deliberately is NOT automatically inserted in backend-main.py / video_scan.py.
It requires manual integration + full backend/mobile regression before deployment.
All processing callbacks are caller-supplied; there are NO AI calls here.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Callable

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from durable_jobs import DurableJobStore, RateLimitExceeded, JobCapacityExceeded, RequestConflict, RequestExpired
from job_runner import run_job, run_ready

log = logging.getLogger(__name__)

PHOTO_MAX = 20 * 1024 * 1024
VIDEO_MAX = 50 * 1024 * 1024
VIDEO_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/x-m4v"}


async def _read_bounded(upload: UploadFile, cap: int) -> bytes:
    # Avoid unbounded file.read(); reject oversized uploads before buffering all.
    data = bytearray()
    while chunk := await upload.read(min(1024 * 1024, cap + 1 - len(data))):
        data.extend(chunk)
        if len(data) > cap:
            raise HTTPException(status_code=413, detail="Upload is too large")
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    return bytes(data)


class ServerlessJobApi:
    def __init__(self, store: DurableJobStore, *, photo_processor: Callable,
                 video_processor: Callable, video_rate_check: Callable | None = None,
                 on_video_complete: Callable | None = None):
        self.store = store
        self.processors = {"photo": photo_processor, "video": video_processor}
        self.video_rate_check = video_rate_check
        self.on_video_complete = on_video_complete
        self.router = APIRouter()
        self._setup_routes()

    async def resume_queued(self):
        """Call on FastAPI startup in a non-blocking task.

        Only never-started (queued) work resumes. Interrupted jobs are NOT
        reprocessed automatically because a paid API call may have succeeded.
        """
        return await run_ready(self.store, self.processors)

    async def _background(self, kind: str, job_id: str, push_token: str | None = None):
        await run_job(self.store, job_id, self.processors)
        if kind == "video" and push_token and self.on_video_complete:
            try:
                # Caller sends the normal success OR failure notification.
                # Push tokens stay process-local, never persist with photo data.
                await self.on_video_complete(push_token, job_id, self.store.status(job_id))
            except Exception:
                log.warning("Video push failed; polling remains available", exc_info=False)

    async def maintenance_loop(self, interval_seconds: float = 60):
        """Prune expired uploads while the app is awake; never makes network calls."""
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await asyncio.to_thread(self.store.cleanup)
            except Exception:
                log.warning('Durable job maintenance failed', exc_info=False)

    def _setup_routes(self):
        r = self.router

        @r.post("/process-clothing")
        async def process_clothing(request: Request, background_tasks: BackgroundTasks,
                                   file: UploadFile = File(...)):
            self.store.cleanup()
            payload = await _read_bounded(file, PHOTO_MAX)
            try:
                jid = self.store.submit("photo", payload, file.content_type or "image/jpeg",
                                        request_id=request.headers.get("x-styligma-request-id"))
            except (RequestConflict, RequestExpired, ValueError) as exc:
                raise HTTPException(status_code=410 if isinstance(exc, RequestExpired) else 409,
                                    detail=str(exc)) from None
            background_tasks.add_task(self._background, "photo", jid)
            return {"job_id": jid, "status": "processing"}

        @r.get("/process-clothing-status/{job_id}")
        async def process_clothing_status(job_id: str, background_tasks: BackgroundTasks):
            record = self.store.status(job_id)
            if record is None or record.get("kind") != "photo":
                raise HTTPException(status_code=404, detail="Unknown or expired clothing job.")
            if job_id in self.store.ready_ids(limit=64):
                # A never-started queued job may survive a restart; safe to
                # dispatch because the worker lease prevents duplicate claims.
                background_tasks.add_task(self._background, "photo", job_id)
            if record["status"] == "done":
                return {"job_id": job_id, "status": "done", "result": record["result"]}
            if record["status"] == "error":
                return {"job_id": job_id, "status": "error", "error": record.get("error", "processing_failed")}
            return {"job_id": job_id, "status": "processing"}

        @r.post("/scan-video", status_code=202)
        async def scan_video(request: Request, background_tasks: BackgroundTasks,
                             file: UploadFile = File(...), push_token: str | None = Form(None),
                             scan_capability: str | None = Form(None)):
            req_id = request.headers.get("x-styligma-request-id")
            try:
                prior = self.store.find_request(req_id) if req_id else None
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None
            if prior and (prior["kind"] != "video" or prior["status"] == "expired"):
                raise HTTPException(status_code=409 if prior["kind"] != "video" else 410,
                                    detail="Original request is no longer recoverable")
            # Exact duplicate requests do not re-consume the five-scan quota.
            if not prior and self.video_rate_check:
                try:
                    self.video_rate_check(request.client.host if request.client else "unknown")
                except RateLimitExceeded as exc:
                    raise HTTPException(status_code=429, detail=str(exc)) from None
            self.store.cleanup()  # Free expired queued uploads before enforcing concurrency limits.
            self.store.mark_interrupted()
            if file.content_type not in VIDEO_CONTENT_TYPES:
                raise HTTPException(status_code=400, detail="Unsupported video format. Use mp4 or mov.")
            if not prior and self.store.count_active("video") >= 4:
                raise HTTPException(status_code=429, detail="Scanner is busy. Please try again shortly.")
            self.store.cleanup()
            payload = await _read_bounded(file, VIDEO_MAX)
            # Existing multi-item feature gate must ALSO be enforced by the injected
            # video processor; metadata is never authorization to turn it on.
            meta = {"multi_item_capable": scan_capability == "multi_item_regions_v1"}
            try:
                jid = self.store.submit("video", payload, file.content_type,
                                        metadata=meta, max_active=4, request_id=req_id)
            except JobCapacityExceeded as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from None
            except (RequestConflict, RequestExpired, ValueError) as exc:
                raise HTTPException(status_code=410 if isinstance(exc, RequestExpired) else 409,
                                    detail=str(exc)) from None
            background_tasks.add_task(self._background, "video", jid, push_token)
            # Preserve the hyphenated UUID representation of the existing video API.
            return {"job_id": str(uuid.UUID(hex=jid))}

        @r.get("/job-by-request/{request_id}")
        async def job_by_request(request_id: str):
            try:
                record = self.store.find_request(request_id)
            except ValueError:
                raise HTTPException(status_code=404, detail="Unknown request") from None
            if record is None:
                raise HTTPException(status_code=404, detail="Unknown request")
            if record["status"] == "expired":
                raise HTTPException(status_code=410, detail="Request expired. New explicit upload required")
            jid = record["job_id"]
            if record["kind"] == "video":
                jid = str(uuid.UUID(hex=jid))
            return {"job_id": jid, "kind": record["kind"], "status": record["status"],
                    "error": record.get("error")}

        @r.get("/scan-status/{job_id}")
        async def scan_status(job_id: str, background_tasks: BackgroundTasks):
            try:
                lookup_id = uuid.UUID(job_id).hex
            except (ValueError, AttributeError):
                raise HTTPException(status_code=404, detail="Job not found or expired.")
            record = self.store.status(lookup_id)
            if record is None or record.get("kind") != "video":
                raise HTTPException(status_code=404, detail="Job not found or expired.")
            if lookup_id in self.store.ready_ids(limit=64):
                background_tasks.add_task(self._background, "video", lookup_id)
            result = record.get("result", {})
            return {
                "status": record["status"],
                "items": result.get("items"),
                "frames_scanned": result.get("frames_scanned"),
                "duplicates_removed": result.get("duplicates_removed"),
                "error": record.get("error"),
                "frame_issues": result.get("frame_issues"),
            }
