"""
Video Wardrobe Scan — backend module for Styligma

Improved version:
- stronger frame dedupe (keeps sharper frame among near-duplicates)
- second-stage garment dedupe AFTER background removal + Claude classification
- rejects tiny / clipped / poor candidates
- normalizes final garment image onto a centered transparent square canvas
- does NOT invent a fake new angle; it only cleans and standardizes the real cutout
"""

import io
import os
import re
import time
import uuid
import asyncio
import tempfile
import base64
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
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

# ── Guardrails / tuning ────────────────────────────────────────────────
MAX_VIDEO_SECONDS = 20
FRAME_SAMPLE_INTERVAL_SEC = 0.6
DEDUP_HASH_THRESHOLD = 12            # pre-Claude frame dedupe
GARMENT_HASH_THRESHOLD = 10          # post-Claude garment dedupe
MAX_ITEMS_RETURNED = 12
MIN_FRAME_SHARPNESS = 55.0
MAX_CONCURRENT_ITEM_PROCESSING = 5

# Candidate quality filters
MIN_OBJECT_AREA_RATIO = 0.08         # reject if garment is too small in original frame
MAX_ALLOWED_EDGE_TOUCHES = 1         # reject if object is clipped by 2+ image edges
CANVAS_SIZE = 900
CANVAS_PADDING = 70

# ── Abuse guard ────────────────────────────────────────────────────────
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


claude_client = anthropic.Anthropic()

# ── Background job store ───────────────────────────────────────────────
JOB_RETENTION_SECONDS = 60 * 60
_jobs: Dict[str, dict] = {}


def _cleanup_old_jobs():
    now = time.time()
    expired = [jid for jid, job in _jobs.items() if now - job["created_at"] > JOB_RETENTION_SECONDS]
    for jid in expired:
        del _jobs[jid]


async def _send_push_notification(push_token: str, title: str, body: str, data: dict):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                "https://exp.host/--/api/v2/push/send",
                json={"to": push_token, "title": title, "body": body, "data": data, "sound": "default"},
            )
    except Exception as e:
        print(f"Push notification failed (non-fatal): {e}")


# ── API models ─────────────────────────────────────────────────────────
class DetectedItem(BaseModel):
    temp_id: str
    image_base64: str
    name: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    color: Optional[str] = None
    colors: Optional[List[str]] = None
    style: Optional[str] = None
    season: Optional[List[str]] = None
    fabric_guess: Optional[str] = None
    confidence: str = "medium"


class ScanResponse(BaseModel):
    items: List[DetectedItem]
    frames_scanned: int
    duplicates_removed: int


class ScanJobCreated(BaseModel):
    job_id: str


class ScanJobStatus(BaseModel):
    status: str
    items: Optional[List[DetectedItem]] = None
    frames_scanned: Optional[int] = None
    duplicates_removed: Optional[int] = None
    error: Optional[str] = None


# ── Internal models ────────────────────────────────────────────────────
@dataclass
class FrameCandidate:
    image: Image.Image
    sharpness: float


@dataclass
class ProcessedCandidate:
    item: DetectedItem
    garment_hash: imagehash.ImageHash
    score: float
    category: str
    subcategory: str
    color: str
    colors: List[str]


# ── Image quality helpers ──────────────────────────────────────────────
def _sharpness(frame) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _to_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA")


def _alpha_bbox(img: Image.Image):
    rgba = _to_rgba(img)
    alpha = rgba.getchannel("A")
    mask = alpha.point(lambda p: 255 if p > 10 else 0)
    return mask.getbbox()


def _nontransparent_ratio(img: Image.Image) -> float:
    rgba = _to_rgba(img)
    alpha = rgba.getchannel("A")
    hist = alpha.histogram()
    non_transparent = sum(hist[11:])
    total = img.size[0] * img.size[1]
    return non_transparent / total if total else 0.0


def _edge_touches(bbox, size: Tuple[int, int], margin: int = 6) -> int:
    if not bbox:
        return 4
    x1, y1, x2, y2 = bbox
    w, h = size
    touches = 0
    if x1 <= margin:
        touches += 1
    if y1 <= margin:
        touches += 1
    if x2 >= w - margin:
        touches += 1
    if y2 >= h - margin:
        touches += 1
    return touches


def _normalize_cutout(img: Image.Image):
    """
    Takes the background-removed garment, rejects poor candidates,
    crops tightly, then places it on a clean centered transparent square canvas.
    No fake angle generation — only cleanup/standardization.
    """
    rgba = _to_rgba(img)
    bbox = _alpha_bbox(rgba)
    if not bbox:
        return None, None

    x1, y1, x2, y2 = bbox
    full_w, full_h = rgba.size
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    full_area = max(1, full_w * full_h)
    object_area_ratio = bbox_area / full_area
    touches = _edge_touches(bbox, rgba.size)

    # Reject tiny distant detections / heavily clipped detections
    if object_area_ratio < MIN_OBJECT_AREA_RATIO:
        return None, None
    if touches > MAX_ALLOWED_EDGE_TOUCHES:
        return None, None

    cropped = rgba.crop(bbox)
    cropped_alpha_ratio = _nontransparent_ratio(cropped)
    if cropped_alpha_ratio < 0.18:
        return None, None

    inner_size = CANVAS_SIZE - (CANVAS_PADDING * 2)
    scale = min(inner_size / cropped.width, inner_size / cropped.height)
    new_w = max(1, int(cropped.width * scale))
    new_h = max(1, int(cropped.height * scale))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    paste_x = (CANVAS_SIZE - new_w) // 2
    paste_y = (CANVAS_SIZE - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)

    meta = {
        "object_area_ratio": object_area_ratio,
        "cropped_alpha_ratio": cropped_alpha_ratio,
        "edge_touches": touches,
    }
    return canvas, meta


# ── Frame extraction / dedupe ──────────────────────────────────────────
def _extract_candidate_frames(video_path: str) -> List[FrameCandidate]:
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

    frames: List[FrameCandidate] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % frame_interval == 0:
            sharp = _sharpness(frame)
            if sharp >= MIN_FRAME_SHARPNESS:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(FrameCandidate(image=Image.fromarray(rgb), sharpness=sharp))
        idx += 1

    cap.release()
    return frames


def _dedupe_frames(frames: List[FrameCandidate]) -> List[FrameCandidate]:
    """
    Fast pre-Claude dedupe using perceptual hash.
    If two frames are near-duplicates, keep the sharper one.
    """
    kept: List[FrameCandidate] = []
    kept_hashes = []

    for cand in frames:
        h = imagehash.phash(cand.image)
        matched_idx = None

        for i, kh in enumerate(kept_hashes):
            if h - kh <= DEDUP_HASH_THRESHOLD:
                matched_idx = i
                break

        if matched_idx is None:
            kept.append(cand)
            kept_hashes.append(h)
        else:
            if cand.sharpness > kept[matched_idx].sharpness:
                kept[matched_idx] = cand
                kept_hashes[matched_idx] = h

        if len(kept) >= MAX_ITEMS_RETURNED * 3:
            # still allow more than MAX_ITEMS_RETURNED through to the second-stage dedupe,
            # but stop ridiculous scan explosions
            break

    return kept


# ── Claude classification ───────────────────────────────────────────────
def _classify_with_claude(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "You are classifying ONE clothing item from a wardrobe-scan app. "
        "The image already has background removed and has been centered on a clean canvas. "
        "Respond with ONLY valid JSON, no markdown, no explanation.\n"
        "{\n"
        '  "name": "short natural item name, e.g. beige blazer",\n'
        '  "category": "top|bottom|dress|shoes|outerwear|accessory",\n'
        '  "subcategory": "short specific type, e.g. blazer, button-down shirt, mini skirt, jeans, sneakers",\n'
        '  "color": "primary color name",\n'
        '  "colors": ["array", "of", "visible", "colors"],\n'
        '  "style": "casual|formal|sporty|elegant|streetwear",\n'
        '  "season": ["spring","summer","fall","winter"],\n'
        '  "fabric_guess": "best guess at material",\n'
        '  "confidence": "low|medium|high"\n'
        "}\n"
        "Rules:\n"
        "- Only classify it if there is ONE clear dominant clothing item.\n"
        "- If the item is too partial, too unclear, too clipped, or not clearly a clothing item, return {\"category\": null}.\n"
        "- Use stable/common subcategory names.\n"
        "- Do NOT describe the background.\n"
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


# ── Garment-level dedupe helpers ────────────────────────────────────────
def _clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.lower().strip()
    value = value.replace("button up", "button-down")
    value = value.replace("button-up", "button-down")
    value = value.replace("button down", "button-down")
    value = re.sub(r"[^a-z0-9\s\-]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _subcategory_family(subcategory: Optional[str]) -> str:
    s = _clean_text(subcategory)

    mapping = [
        ("button-down shirt", "shirt"),
        ("dress shirt", "shirt"),
        ("shirt", "shirt"),
        ("blouse", "blouse"),
        ("t-shirt", "tshirt"),
        ("tee", "tshirt"),
        ("tank", "tank"),
        ("blazer", "blazer"),
        ("jacket", "jacket"),
        ("coat", "coat"),
        ("hoodie", "hoodie"),
        ("sweater", "sweater"),
        ("cardigan", "cardigan"),
        ("jeans", "jeans"),
        ("trouser", "pants"),
        ("pant", "pants"),
        ("shorts", "shorts"),
        ("skirt", "skirt"),
        ("dress", "dress"),
        ("sneaker", "sneakers"),
        ("boot", "boots"),
        ("heel", "heels"),
        ("loafer", "loafers"),
    ]

    for needle, family in mapping:
        if needle in s:
            return family

    return s


def _confidence_score(conf: str) -> float:
    return {
        "high": 30.0,
        "medium": 18.0,
        "low": 8.0,
    }.get((conf or "medium").lower(), 18.0)


def _build_detected_item(normalized_cutout: Image.Image, classification: dict, temp_id: str) -> DetectedItem:
    buf = io.BytesIO()
    normalized_cutout.save(buf, format="PNG")

    category = classification.get("category")
    subcategory = classification.get("subcategory")
    color = classification.get("color")
    colors = classification.get("colors") or []

    name = classification.get("name")
    if not name:
        name = f"{color or ''} {subcategory or 'Item'}".strip()

    return DetectedItem(
        temp_id=temp_id,
        image_base64=base64.b64encode(buf.getvalue()).decode(),
        category=category,
        subcategory=subcategory,
        color=color,
        colors=colors,
        style=classification.get("style"),
        season=classification.get("season"),
        fabric_guess=classification.get("fabric_guess"),
        name=name,
        confidence=classification.get("confidence", "medium"),
    )


def _candidates_are_duplicates(a: ProcessedCandidate, b: ProcessedCandidate) -> bool:
    if a.category != b.category:
        return False

    a_family = _subcategory_family(a.subcategory)
    b_family = _subcategory_family(b.subcategory)
    if a_family != b_family:
        return False

    hash_diff = a.garment_hash - b.garment_hash
    color_match = False

    if a.color and b.color and a.color == b.color:
        color_match = True
    elif a.color and a.color in b.colors:
        color_match = True
    elif b.color and b.color in a.colors:
        color_match = True
    elif set(a.colors).intersection(set(b.colors)):
        color_match = True

    # Same family + close garment hash + color compatibility
    if hash_diff <= GARMENT_HASH_THRESHOLD and (color_match or hash_diff <= GARMENT_HASH_THRESHOLD // 2):
        return True

    return False


def _dedupe_processed_items(candidates: List[ProcessedCandidate]) -> Tuple[List[DetectedItem], int]:
    """
    Second-stage dedupe AFTER background removal + classification.
    This is what fixes 'same shirt appears 5 times from slightly different frames.'
    """
    kept: List[ProcessedCandidate] = []
    duplicates_removed = 0

    for cand in candidates:
        matched_idx = None
        for i, existing in enumerate(kept):
            if _candidates_are_duplicates(cand, existing):
                matched_idx = i
                break

        if matched_idx is None:
            kept.append(cand)
        else:
            duplicates_removed += 1
            if cand.score > kept[matched_idx].score:
                kept[matched_idx] = cand

    kept.sort(key=lambda c: c.score, reverse=True)
    return [c.item for c in kept[:MAX_ITEMS_RETURNED]], duplicates_removed


# ── Processing pipeline ─────────────────────────────────────────────────
def _process_single_frame_sync(frame: FrameCandidate, temp_id: str) -> Optional[ProcessedCandidate]:
    """
    One frame -> rembg -> normalize -> Claude classify -> internal candidate.
    """
    cutout = _to_rgba(_remove_background(frame.image))
    normalized_cutout, meta = _normalize_cutout(cutout)
    if normalized_cutout is None:
        return None

    classification = _classify_with_claude(normalized_cutout)
    if not classification.get("category"):
        return None

    item = _build_detected_item(normalized_cutout, classification, temp_id)
    garment_hash = imagehash.phash(normalized_cutout.convert("RGB"))

    score = (
        _confidence_score(item.confidence)
        + min(frame.sharpness / 10.0, 25.0)
        + (meta["object_area_ratio"] * 60.0)
        + (meta["cropped_alpha_ratio"] * 20.0)
        - (meta["edge_touches"] * 6.0)
    )

    return ProcessedCandidate(
        item=item,
        garment_hash=garment_hash,
        score=score,
        category=_clean_text(item.category),
        subcategory=_clean_text(item.subcategory),
        color=_clean_text(item.color),
        colors=[_clean_text(c) for c in (item.colors or [])],
    )


async def _process_frames_concurrently(frames: List[FrameCandidate]) -> List[ProcessedCandidate]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ITEM_PROCESSING)

    async def _bounded(frame, temp_id):
        async with semaphore:
            return await asyncio.to_thread(_process_single_frame_sync, frame, temp_id)

    tasks = [_bounded(frame, f"scan_{i}") for i, frame in enumerate(frames)]
    results = await asyncio.gather(*tasks)
    return [item for item in results if item is not None]


def _remove_background(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result_bytes = rembg_remove(buf.getvalue())
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")


# ── Endpoints ───────────────────────────────────────────────────────────
@router.post("/scan-photo", response_model=DetectedItem)
async def scan_photo(request: Request, file: UploadFile = File(...)):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    if file.content_type not in ("image/jpeg", "image/png", "image/heic", "image/heif"):
        raise HTTPException(status_code=400, detail="Unsupported image format.")

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    sharp = 100.0  # single photo path doesn't need frame sharpness estimation like video

    processed = _process_single_frame_sync(FrameCandidate(image=img, sharpness=sharp), "photo_0")
    if not processed:
        raise HTTPException(
            status_code=422,
            detail="Couldn't identify one clear clothing item in this photo. Try a clearer, closer shot.",
        )

    return processed.item


async def _run_video_scan_job(job_id: str, tmp_path: str, push_token: Optional[str]):
    try:
        raw_frames = _extract_candidate_frames(tmp_path)
        if not raw_frames:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = "No usable frames found — video may be too dark, too blurry, or too short."
            return

        deduped_frames = _dedupe_frames(raw_frames)
        frame_duplicates_removed = len(raw_frames) - len(deduped_frames)

        processed_candidates = await _process_frames_concurrently(deduped_frames)
        final_items, garment_duplicates_removed = _dedupe_processed_items(processed_candidates)

        total_duplicates_removed = frame_duplicates_removed + garment_duplicates_removed

        _jobs[job_id].update({
            "status": "done",
            "items": final_items,
            "frames_scanned": len(raw_frames),
            "duplicates_removed": total_duplicates_removed,
        })

        if push_token:
            count = len(final_items)
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
