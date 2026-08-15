
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
FRAME_SAMPLE_INTERVAL_SEC = 0.5

# Stage 1: cheap whole-frame dedupe before expensive AI processing.
FRAME_DEDUP_HASH_THRESHOLD = 8

# Stage 2: stronger dedupe on the normalized garment cutout itself.
GARMENT_DEDUP_HASH_THRESHOLD = 12

MAX_ITEMS_RETURNED = 20
MIN_FRAME_SHARPNESS = 40.0
MAX_CONCURRENT_ITEM_PROCESSING = 5

# Final wardrobe image normalization.
NORMALIZED_CANVAS_SIZE = 900
NORMALIZED_PADDING = 90
MIN_ALPHA_BBOX_RATIO = 0.08

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
    """
    Pull frames at a fixed interval, apply phone-video orientation metadata
    when OpenCV exposes it, and skip blurry frames.
    """
    cap = cv2.VideoCapture(video_path)

    try:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass

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

        if idx % frame_interval == 0 and _sharpness(frame) >= MIN_FRAME_SHARPNESS:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))

        idx += 1

    cap.release()
    return frames

def _dedupe_frames(frames: List["Image.Image"]) -> List["Image.Image"]:
    """
    Cheap first-pass dedupe on the complete video frame.

    This only prevents obviously repeated neighboring frames from reaching
    Claude. A second garment-level dedupe runs later on the actual cutout.
    """
    kept = []
    kept_hashes = []

    for img in frames:
        h = imagehash.phash(img)

        if all(h - kh > FRAME_DEDUP_HASH_THRESHOLD for kh in kept_hashes):
            kept.append(img)
            kept_hashes.append(h)

        if len(kept) >= MAX_ITEMS_RETURNED:
            break

    return kept

def _remove_background(img: "Image.Image") -> "Image.Image":
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result_bytes = rembg_remove(buf.getvalue())
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")


def _normalize_cutout(img: "Image.Image") -> Optional["Image.Image"]:
    """
    Convert a background-removed result into a clean wardrobe asset.

    We only crop, scale and center the REAL detected garment. We do not
    generate a fake front-facing view, because that could alter the item.
    """
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()

    if not bbox:
        return None

    left, top, right, bottom = bbox
    garment_w = max(1, right - left)
    garment_h = max(1, bottom - top)
    bbox_ratio = (garment_w * garment_h) / max(1, rgba.width * rgba.height)

    if bbox_ratio < MIN_ALPHA_BBOX_RATIO:
        return None

    cropped = rgba.crop(bbox)

    inner_size = NORMALIZED_CANVAS_SIZE - (NORMALIZED_PADDING * 2)
    scale = min(inner_size / cropped.width, inner_size / cropped.height)

    new_w = max(1, int(cropped.width * scale))
    new_h = max(1, int(cropped.height * scale))
    cropped = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new(
        "RGBA",
        (NORMALIZED_CANVAS_SIZE, NORMALIZED_CANVAS_SIZE),
        (255, 255, 255, 0),
    )

    x = (NORMALIZED_CANVAS_SIZE - new_w) // 2
    y = (NORMALIZED_CANVAS_SIZE - new_h) // 2
    canvas.alpha_composite(cropped, (x, y))

    return canvas


def _cutout_quality_score(img: "Image.Image") -> float:
    """
    Score a normalized cutout so duplicate groups keep their best frame.
    Higher = sharper and more useful.
    """
    rgba = img.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.getbbox()

    if not bbox:
        return 0.0

    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba.convert("RGB"), mask=alpha)

    import numpy as np
    arr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    sharp = _sharpness(arr)

    left, top, right, bottom = bbox
    fill_ratio = ((right - left) * (bottom - top)) / max(1, rgba.width * rgba.height)

    return float(sharp) + (fill_ratio * 500.0)


def _same_garment_metadata(a: DetectedItem, b: DetectedItem) -> bool:
    """
    Conservative identity check used before garment-image hash comparison.
    """
    a_cat = (a.category or "").strip().lower()
    b_cat = (b.category or "").strip().lower()
    a_color = (a.color or "").strip().lower()
    b_color = (b.color or "").strip().lower()

    return a_cat == b_cat and (not a_color or not b_color or a_color == b_color)


def _garment_hash_threshold(a: DetectedItem, b: DetectedItem) -> int:
    """
    If Claude agrees on subcategory we can dedupe a little more aggressively.
    If labels differ, require a much closer visual match.
    """
    a_sub = (a.subcategory or "").strip().lower()
    b_sub = (b.subcategory or "").strip().lower()

    if a_sub and b_sub and a_sub == b_sub:
        return GARMENT_DEDUP_HASH_THRESHOLD

    return 7

def _classify_with_claude(img: "Image.Image") -> dict:
    """
    Strictly validate ONE garment candidate and classify it.

    Important: rembg can still leave several overlapping clothes in one
    cutout, so Claude acts as a quality gate instead of forcing every frame
    into the wardrobe.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "You are validating ONE garment candidate extracted from a wardrobe scan video. "
        "Be strict. A candidate is usable only if ONE dominant wearable fashion item is "
        "clearly visible and recognizable. Reject it when: several similarly dominant "
        "garments overlap, the garment is heavily cut off, it is mostly closet/background, "
        "the cutout is badly distorted, or it is too blurry/ambiguous. "
        "A hanger attached to one clear garment is okay. "
        "Respond with ONLY valid JSON, no markdown, no preamble:\n"
        "{\n"
        '  "usable": true,\n'
        '  "category": "top|bottom|dress|shoes|outerwear|accessory",\n'
        '  "subcategory": "short specific type, e.g. blazer, jeans, sneakers",\n'
        '  "color": "primary color name",\n'
        '  "colors": ["array", "of", "all", "visible", "colors"],\n'
        '  "style": "casual|formal|sporty|elegant|streetwear",\n'
        '  "season": ["spring","summer","fall","winter"],\n'
        '  "fabric_guess": "best guess at material",\n'
        '  "confidence": "low|medium|high"\n'
        "}\n"
        'If unusable, respond exactly like {"usable": false, "category": null}. '
        "Do not force a classification."
    )

    response = claude_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )

    text = response.content[0].text.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"usable": False, "category": None}


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
    normalized = _normalize_cutout(cutout)

    if normalized is None:
        raise HTTPException(
            status_code=422,
            detail="Couldn't isolate a clear clothing item in this photo.",
        )

    classification = _classify_with_claude(normalized)

    if classification.get("usable") is False or not classification.get("category"):
        raise HTTPException(
            status_code=422,
            detail="Couldn't identify a clothing item in this photo. Try a clearer, closer shot.",
        )

    buf = io.BytesIO()
    normalized.save(buf, format="PNG")
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


def _process_single_frame_sync(frame: "Image.Image", temp_id: str) -> Optional[dict]:
    """
    Process one frame into a normalized garment candidate.

    We retain the PIL cutout and its hash/quality score so a second pass can
    collapse repeated views of the same garment and keep the best one.
    """
    cutout = _remove_background(frame)
    normalized = _normalize_cutout(cutout)

    if normalized is None:
        return None

    classification = _classify_with_claude(normalized)

    if classification.get("usable") is False or not classification.get("category"):
        return None

    buf = io.BytesIO()
    normalized.save(buf, format="PNG")

    item = DetectedItem(
        temp_id=temp_id,
        image_base64=base64.b64encode(buf.getvalue()).decode(),
        category=classification.get("category"),
        subcategory=classification.get("subcategory"),
        color=classification.get("color"),
        colors=classification.get("colors"),
        style=classification.get("style"),
        season=classification.get("season"),
        fabric_guess=classification.get("fabric_guess"),
        name=classification.get("name")
        or f"{classification.get('color','')} {classification.get('subcategory','Item')}".strip(),
        confidence=classification.get("confidence", "medium"),
    )

    return {
        "item": item,
        "cutout": normalized,
        "hash": imagehash.phash(normalized.convert("RGB")),
        "quality": _cutout_quality_score(normalized),
    }


async def _process_frames_concurrently(frames: List["Image.Image"]) -> List[dict]:
    """
    Process candidate frames concurrently while retaining garment cutouts
    for the second dedupe pass.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ITEM_PROCESSING)

    async def _bounded(frame, temp_id):
        async with semaphore:
            return await asyncio.to_thread(_process_single_frame_sync, frame, temp_id)

    tasks = [_bounded(frame, f"scan_{i}") for i, frame in enumerate(frames)]
    results = await asyncio.gather(*tasks)

    return [result for result in results if result is not None]


def _dedupe_processed_garments(processed: List[dict]) -> tuple[List[DetectedItem], int]:
    """
    Second-pass duplicate removal on the ACTUAL normalized garment.

    Whole-frame dedupe alone is not enough because a slight camera move
    changes the wardrobe background. Here we compare garment cutouts.

    When two candidates are duplicates, keep the sharper/better candidate.
    """
    kept: List[dict] = []
    removed = 0

    for candidate in processed:
        duplicate_index = None

        for idx, existing in enumerate(kept):
            if not _same_garment_metadata(candidate["item"], existing["item"]):
                continue

            threshold = _garment_hash_threshold(candidate["item"], existing["item"])
            distance = candidate["hash"] - existing["hash"]

            if distance <= threshold:
                duplicate_index = idx
                break

        if duplicate_index is None:
            kept.append(candidate)
            continue

        removed += 1

        if candidate["quality"] > kept[duplicate_index]["quality"]:
            kept[duplicate_index] = candidate

    items: List[DetectedItem] = []

    for i, candidate in enumerate(kept[:MAX_ITEMS_RETURNED]):
        item = candidate["item"].model_copy(update={"temp_id": f"scan_{i}"})
        items.append(item)

    return items, removed


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

        deduped_frames = _dedupe_frames(raw_frames)
        frame_duplicates_removed = len(raw_frames) - len(deduped_frames)

        processed = await _process_frames_concurrently(deduped_frames)
        items, garment_duplicates_removed = _dedupe_processed_garments(processed)

        duplicates_removed = frame_duplicates_removed + garment_duplicates_removed

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
