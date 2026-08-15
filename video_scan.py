"""
Video Wardrobe Scan — backend module for Styligma

Feature: user films a pan across their closet; we extract frames, dedupe
near-identical/repeated garments, remove backgrounds, classify each item
with Claude vision, and return a review list for the app to confirm before
anything is added to the wardrobe.

ASYNC JOB MODEL: POST /scan-video returns a job_id immediately (202) and
processes in the background — the app doesn't block waiting for results,
and the user can record more scans while an earlier one is still running.
Completion is signaled via an Expo push notification (if a push_token was
supplied) with GET /scan-status/{job_id} as a polling fallback.

MERGE INSTRUCTIONS:
1. Add this router to backend-main.py:
     from video_scan import router as video_scan_router
     app.include_router(video_scan_router)
2. Add new dependencies to requirements.txt:
     opencv-python-headless
     imagehash
     httpx
     Pillow  (already present via rembg dependency, but pin explicitly)
3. Railway build: opencv-python-headless avoids needing system GUI libs
   that opencv-python requires and that Railway's build image lacks.

COST / SAFETY GUARDRAILS (do not remove without reconsidering):
- MAX_VIDEO_SECONDS caps processing time and Claude API spend per scan.
- MAX_ITEMS_RETURNED caps how many items get sent to the frontend, since
  a long slow pan could otherwise generate 40+ near-duplicate candidates.
- Frames are deduped BEFORE hitting Claude vision (not after), since the
  Claude call is the expensive step — hashing is done locally and free.
- Per-IP rate limit (5 scans / 24h) is a best-effort abuse deterrent —
  the app has no accounts, so this is not a real security boundary, just
  a cheap tripwire against scripted abuse. The actual free-tier cap
  (1 free scan, then Pro) lives client-side in WardrobeStore.
- Items are processed CONCURRENTLY (up to MAX_CONCURRENT_ITEM_PROCESSING
  at once via asyncio + a semaphore), not one at a time. A 20-item scan
  went from ~80s sequential to roughly however long the slowest single
  item takes (~10-15s). Raise the concurrency cap cautiously — it's
  bounded by Claude's rate limits and Railway's CPU, not just app logic.
"""

import io
import os
import time
import uuid
import asyncio
import tempfile
import base64
import json
from typing import List, Optional, Dict
from collections import defaultdict

import cv2
import imagehash
import httpx
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from rembg import remove as rembg_remove
import anthropic

router = APIRouter()

# ── Guardrails ──────────────────────────────────────────────────────────
MAX_VIDEO_SECONDS = 20
FRAME_SAMPLE_INTERVAL_SEC = 0.5   # check a candidate frame twice a second
DEDUP_HASH_THRESHOLD = 8          # lower = stricter dedup (0-64 scale, phash)
MAX_ITEMS_RETURNED = 20
MIN_FRAME_SHARPNESS = 40.0        # rejects motion-blurred frames (Laplacian variance)
MAX_CONCURRENT_ITEM_PROCESSING = 5  # cap parallel rembg+Claude calls — protects Claude rate limits and Railway CPU

# ── Abuse guard ──────────────────────────────────────────────────────────
# The app has no accounts (by design — see privacy policy), so we can't do
# real per-user rate limiting server-side. This is a blunt, best-effort
# per-IP limiter to stop scripted abuse from burning through Claude/Railway
# spend — it is NOT the primary limit. The real free-tier cap lives
# client-side in WardrobeStore (FREE_LIMITS.FREE_VIDEO_SCANS) since that's
# where we can actually distinguish "one honest user" from another.
# In-memory only: resets on deploy/restart, and won't catch someone
# rotating IPs — acceptable for a hobby-plan abuse deterrent, not a real
# security boundary. If this app scales up, replace with Redis + a real
# per-device identifier header instead of IP.
RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_SECONDS = 24 * 60 * 60
_request_log: dict = defaultdict(list)


def _check_rate_limit(ip: str):
    now = time.time()
    _request_log[ip] = [t for t in _request_log[ip] if now - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(_request_log[ip]) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many scans from this network today. Please try again tomorrow.",
        )
    _request_log[ip].append(now)


claude_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env, same as rest of backend

# ── Background job store ─────────────────────────────────────────────────
# Video scans now run as background jobs instead of blocking the request:
# the app uploads the video, immediately gets a job_id back, and is free
# to record more videos/photos while this one processes. When it's done,
# we push a notification (if the app registered a push token) and the app
# can also poll GET /scan-status/{job_id} as a fallback.
#
# IMPORTANT MVP LIMITATION: this job store is a plain in-memory dict.
# It does NOT survive a Railway restart/redeploy, and does NOT work if
# you ever scale to multiple instances (each instance has its own dict).
# That's an acceptable tradeoff for a single-instance Hobby-plan app —
# a job lost to a redeploy just means the user re-scans, no real harm.
# If this app scales up, replace with Redis or a small Postgres table.
JOB_RETENTION_SECONDS = 60 * 60  # drop job results after 1 hour so this dict doesn't grow forever
_jobs: Dict[str, dict] = {}


def _cleanup_old_jobs():
    now = time.time()
    expired = [jid for jid, job in _jobs.items() if now - job["created_at"] > JOB_RETENTION_SECONDS]
    for jid in expired:
        del _jobs[jid]


async def _send_push_notification(push_token: str, title: str, body: str, data: dict):
    """
    Fires an Expo push notification once a job finishes. Best-effort —
    if this fails (bad token, Expo API hiccup, etc.) we log and move on;
    the app's polling fallback (GET /scan-status) still works either way,
    so a failed push is a worse experience, not a broken one.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                "https://exp.host/--/api/v2/push/send",
                json={"to": push_token, "title": title, "body": body, "data": data, "sound": "default"},
            )
    except Exception as e:
        print(f"Push notification failed (non-fatal): {e}")


class DetectedItem(BaseModel):
    temp_id: str
    image_base64: str          # background-removed PNG, base64-encoded
    name: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    color: Optional[str] = None
    colors: Optional[List[str]] = None
    style: Optional[str] = None
    season: Optional[List[str]] = None
    fabric_guess: Optional[str] = None
    confidence: str = "medium"  # low/medium/high — surfaced in UI so low-confidence items are easy to spot/remove


class ScanResponse(BaseModel):
    items: List[DetectedItem]
    frames_scanned: int
    duplicates_removed: int


class ScanJobCreated(BaseModel):
    job_id: str


class ScanJobStatus(BaseModel):
    status: str  # "processing" | "done" | "error"
    items: Optional[List[DetectedItem]] = None
    frames_scanned: Optional[int] = None
    duplicates_removed: Optional[int] = None
    error: Optional[str] = None


def _sharpness(frame) -> float:
    """Laplacian variance — a simple, fast motion-blur detector. Low value = blurry frame, skip it."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _extract_candidate_frames(video_path: str) -> List["Image.Image"]:
    """Pull frames at a fixed interval, skip blurry ones. Returns PIL images."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(fps * FRAME_SAMPLE_INTERVAL_SEC))

    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps else 0
    if duration > MAX_VIDEO_SECONDS:
        cap.release()
        raise HTTPException(
            status_code=400,
            detail=f"Video too long ({duration:.0f}s). Max {MAX_VIDEO_SECONDS}s — keep the pan short and steady.",
        )

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            if _sharpness(frame) >= MIN_FRAME_SHARPNESS:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    return frames


def _dedupe_frames(frames: List["Image.Image"]) -> List["Image.Image"]:
    """
    Perceptual-hash dedup. A slow pan produces many frames of the SAME
    garment — we only want to keep one representative frame per distinct
    item. This runs before the expensive Claude call, not after.
    """
    kept = []
    kept_hashes = []
    for img in frames:
        h = imagehash.phash(img)
        if all(h - kh > DEDUP_HASH_THRESHOLD for kh in kept_hashes):
            kept.append(img)
            kept_hashes.append(h)
        if len(kept) >= MAX_ITEMS_RETURNED:
            break
    return kept


def _remove_background(img: "Image.Image") -> "Image.Image":
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result_bytes = rembg_remove(buf.getvalue())
    return Image.open(io.BytesIO(result_bytes))


def _classify_with_claude(img: "Image.Image") -> dict:
    """
    Same classification contract as the existing single-photo add flow —
    keeps WardrobeItem fields consistent whether an item came from a
    single photo or a video scan.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "You are classifying a single clothing item cropped from a wardrobe scan video. "
        "The background has been removed. Respond with ONLY valid JSON, no markdown, no preamble:\n"
        "{\n"
        '  "category": "top|bottom|dress|shoes|outerwear|accessory",\n'
        '  "subcategory": "short specific type, e.g. blazer, jeans, sneakers",\n'
        '  "color": "primary color name",\n'
        '  "colors": ["array", "of", "all", "visible", "colors"],\n'
        '  "style": "casual|formal|sporty|elegant|streetwear",\n'
        '  "season": ["spring","summer","fall","winter"] (all seasons this item suits),\n'
        '  "fabric_guess": "best guess at material",\n'
        '  "confidence": "low|medium|high" (low if item is unclear, cropped oddly, or ambiguous)\n'
        "}\n"
        "If the image does not contain a clear clothing item (e.g. it's a wall, hanger only, "
        'blurred nothing), respond with {"category": null} instead.'
    )

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text = response.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"category": None}


@router.post("/scan-photo", response_model=DetectedItem)
async def scan_photo(request: Request, file: UploadFile = File(...)):
    """
    Single-photo path — for adding one item quickly rather than a full
    closet pan. Reuses the same rembg + Claude classification pipeline
    as the video scan, just skips frame extraction/dedup since there's
    only one image. NOT subject to the free-video-scan cap (it's a
    single cheap Claude call, same cost class as the app's existing
    manual add-item flow) — only the per-IP abuse guard applies.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    if file.content_type not in ("image/jpeg", "image/png", "image/heic", "image/heif"):
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    cutout = _remove_background(img)
    classification = _classify_with_claude(cutout)

    if not classification.get("category"):
        raise HTTPException(
            status_code=422,
            detail="Couldn't identify a clothing item in this photo. Try a clearer, closer shot.",
        )

    buf = io.BytesIO()
    cutout.save(buf, format="PNG")
    return DetectedItem(
        temp_id="photo_0",
        image_base64=base64.b64encode(buf.getvalue()).decode(),
        category=classification.get("category"),
        subcategory=classification.get("subcategory"),
        color=classification.get("color"),
        colors=classification.get("colors"),
        style=classification.get("style"),
        season=classification.get("season"),
        fabric_guess=classification.get("fabric_guess"),
        name=classification.get("name") or f"{classification.get('color','')} {classification.get('subcategory','Item')}".strip(),
        confidence=classification.get("confidence", "medium"),
    )


def _process_single_frame_sync(frame: "Image.Image", temp_id: str) -> Optional[DetectedItem]:
    """
    Blocking work for ONE detected item: background removal + Claude
    classification. Runs inside a thread (via asyncio.to_thread) so
    multiple items can be processed concurrently instead of one at a
    time — this is what took a 20-item scan from ~80s down to roughly
    however long the slowest single item takes.
    """
    cutout = _remove_background(frame)
    classification = _classify_with_claude(cutout)

    if not classification.get("category"):
        return None

    buf = io.BytesIO()
    cutout.save(buf, format="PNG")
    return DetectedItem(
        temp_id=temp_id,
        image_base64=base64.b64encode(buf.getvalue()).decode(),
        category=classification.get("category"),
        subcategory=classification.get("subcategory"),
        color=classification.get("color"),
        colors=classification.get("colors"),
        style=classification.get("style"),
        season=classification.get("season"),
        fabric_guess=classification.get("fabric_guess"),
        name=classification.get("name") or f"{classification.get('color','')} {classification.get('subcategory','Item')}".strip(),
        confidence=classification.get("confidence", "medium"),
    )


async def _process_frames_concurrently(frames: List["Image.Image"]) -> List[DetectedItem]:
    """
    Processes all deduped frames in parallel, capped at
    MAX_CONCURRENT_ITEM_PROCESSING at a time via a semaphore — so a
    20-item scan doesn't fire 20 simultaneous Claude requests at once
    and trip rate limits, but also doesn't process them one by one.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ITEM_PROCESSING)

    async def _bounded(frame, temp_id):
        async with semaphore:
            return await asyncio.to_thread(_process_single_frame_sync, frame, temp_id)

    tasks = [_bounded(frame, f"scan_{i}") for i, frame in enumerate(frames)]
    results = await asyncio.gather(*tasks)
    return [item for item in results if item is not None]  # drop frames that weren't actually garments


async def _run_video_scan_job(job_id: str, tmp_path: str, push_token: Optional[str]):
    """
    The actual scan work, run in the background AFTER the app already
    got its job_id response back. Same extraction/dedupe/classify
    pipeline as before — just no longer blocking the HTTP request.
    """
    try:
        raw_frames = _extract_candidate_frames(tmp_path)
        if not raw_frames:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = "No usable frames found — video may be too dark, too blurry, or too short."
            return

        deduped = _dedupe_frames(raw_frames)
        duplicates_removed = len(raw_frames) - len(deduped)
        items = await _process_frames_concurrently(deduped)

        _jobs[job_id].update({
            "status": "done",
            "items": items,
            "frames_scanned": len(raw_frames),
            "duplicates_removed": duplicates_removed,
        })

        if push_token:
            count = len(items)
            body = (
                f"Found {count} item{'s' if count != 1 else ''} — tap to review and add them"
                if count > 0 else "Scan finished, but no items were clearly detected. Tap to try again."
            )
            await _send_push_notification(
                push_token,
                title="Your wardrobe scan is ready ✨",
                body=body,
                data={"job_id": job_id, "type": "video_scan_complete"},
            )
    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)
        if push_token:
            await _send_push_notification(
                push_token,
                title="Wardrobe scan failed",
                body="Something went wrong processing your video. Please try again.",
                data={"job_id": job_id, "type": "video_scan_error"},
            )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/scan-video", response_model=ScanJobCreated, status_code=202)
async def scan_video(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    push_token: Optional[str] = Form(None),
):
    """
    Returns immediately with a job_id — processing happens in the
    background. The app is free to record more videos/photos right
    away. Pass push_token (Expo push token) to get notified when it's
    done; otherwise poll GET /scan-status/{job_id}.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)
    _cleanup_old_jobs()

    if file.content_type not in ("video/mp4", "video/quicktime", "video/x-m4v"):
        raise HTTPException(status_code=400, detail="Unsupported video format. Use mp4 or mov.")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing", "created_at": time.time()}
    background_tasks.add_task(_run_video_scan_job, job_id, tmp_path, push_token)

    return ScanJobCreated(job_id=job_id)


@router.get("/scan-status/{job_id}", response_model=ScanJobStatus)
async def scan_status(job_id: str):
    """Polling fallback for when push notifications aren't granted/reliable."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired.")

    return ScanJobStatus(
        status=job["status"],
        items=job.get("items"),
        frames_scanned=job.get("frames_scanned"),
        duplicates_removed=job.get("duplicates_removed"),
        error=job.get("error"),
    )
