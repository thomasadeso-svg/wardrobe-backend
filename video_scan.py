"""Video pan backend: streamed frames, serialized scans, explicit reusable model,
exact consecutive-frame reuse before inference, and existing crop tracking.
Near-duplicate frames still require classification before the crop tracker.
Single process/replica only: job state remains in memory and is lost on restart.
"""

import io
import os
import time
import uuid
import hashlib
import logging
import math
import threading
from contextlib import suppress
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
from background_removal import remove_background_bytes
from scan_continuity import SourceContinuity, source_mask, reconcile_backward
import anthropic

router = APIRouter()
logger = logging.getLogger("uvicorn.error")
_scan_lock = asyncio.Lock()
_cleanup_task = None
MAX_PENDING_SCANS = 4
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# Explicit model avoids dependency upgrades silently changing the video model.
VIDEO_REMBG_MODEL = os.getenv("VIDEO_REMBG_MODEL", "u2net")
# 0 preserves input resolution; opt in to 1024 after comparing real scan quality.
MAX_FRAME_EDGE = int(os.getenv("SCAN_MAX_FRAME_EDGE", "0"))
# Deliberately not an environment/client option. Activation requires a separate
# reviewed change and spending authorization; offline tests inject a detector.
MULTI_ITEM_LIVE_ENABLED = False


def _make_region_detector():
    if not MULTI_ITEM_LIVE_ENABLED:return None
    from scan_detection import RegionDetector
    return RegionDetector(anthropic.Anthropic(max_retries=0, timeout=45))

@router.on_event("startup")
async def start_scan_cleanup():
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_cleanup_loop())

@router.on_event("shutdown")
async def stop_scan_cleanup():
    if _cleanup_task is not None:
        _cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cleanup_task

async def _cleanup_loop():
    while True:
        await asyncio.sleep(60)
        _cleanup_old_jobs()


# ── Guardrails ──────────────────────────────────────────────────────────
MAX_VIDEO_SECONDS = 20
FRAME_SAMPLE_INTERVAL_SEC = 0.5   # back to 2/sec — denser sampling is FINE now that temporal run-grouping collapses consecutive frames, and it makes run boundaries easier to detect
TRACK_MAX_FRAME_GAP = 6           # TRACKING: a candidate can join a track only if it's within this many sampled frames of that track's last appearance. Raised 3 -> 6: at 0.5s sampling a garment often stays in view several seconds, so 3 was cutting tracks mid-garment. Still bounded, so a similar item much later in the video stays separate (spec #6).
TRACK_SIMILARITY_THRESHOLD = 20   # TRACKING: max shape+color distance between GARMENT CROPS (not raw frames) for them to count as the same physical item. Raised 14 -> 20: rembg smearing/artifacts and angle changes on the SAME garment shift the crop signature more than initially estimated. Watch the TRACK_DEBUG logs before moving this again.
COLOR_HASH_WEIGHT = 1.5           # colorhash counts a bit more than phash in signature distance, since garments often differ mainly by color while keeping a similar silhouette on a rail.
TRACK_DEBUG = True                # logs measured candidate-to-track distances so thresholds can be tuned from real numbers instead of guesswork. Safe to leave on (stdout only); set False to quieten Railway logs.
MAX_ITEMS_RETURNED = 20
MIN_FRAME_SHARPNESS = 22.0        # relaxed from 40 — was rejecting clearly-identifiable garments for mild motion blur. Lower = more forgiving.


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


claude_client = None  # Created only for actual classification, never diagnostic-only runs.


def _scan_event(report, scan_id, stage, **fields):
    """Metadata only: never log images, tokens, prompts, paths or API responses."""
    event = {"scan_id": scan_id, "stage": stage, **fields}
    if report is not None:
        report.append(event)
    logger.info("[SCAN_DIAG] %s", json.dumps(event, allow_nan=False))

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
    expired = [jid for jid, job in _jobs.items()
               if job["status"] != "processing"
               and now - job.get("finished_at", job["created_at"]) > JOB_RETENTION_SECONDS]
    for jid in expired:
        del _jobs[jid]
    for ip, timestamps in list(_request_log.items()):
        remaining = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if remaining:
            _request_log[ip] = remaining
        else:
            del _request_log[ip]



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
    possible_duplicate_of: Optional[str] = None
    review_required: bool = False
    detection_diagnostics: Optional[List[str]] = None
    detection_notes: Optional[List[str]] = None
    detection_region: Optional[List[float]] = None
    image_kind: Optional[str] = None  # cutout or unverified_source_region


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
    frame_issues: Optional[List[dict]] = None


def _sharpness(frame) -> float:
    """Laplacian variance — a simple, fast motion-blur detector. Low value = blurry frame, skip it."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _crop_signature(crop: "Image.Image"):
    """
    Visual fingerprint of a CLEANED GARMENT CROP (post-rembg, post-framing)
    — NOT the raw video frame.

    This is the central fix for the "same sandals appear 3x" bug. Raw
    video frames of one garment differ by background, rail, wall, and the
    user's hand, so their hashes diverge and temporal runs got cut
    mid-garment. Once the background is stripped and the garment is
    normalized to a centered square, adjacent captures of the SAME item
    become near-identical, which is what dedup actually needs.

    Composited onto white first: phash/colorhash ignore alpha, so
    transparent regions would otherwise read as black and swamp the
    signal.
    """
    if crop.mode != "RGBA":
        crop = crop.convert("RGBA")
    white = Image.new("RGB", crop.size, (255, 255, 255))
    white.paste(crop, mask=crop.getchannel("A"))
    return imagehash.phash(white), imagehash.colorhash(white)


def _crop_distance(sig_a, sig_b) -> float:
    """Weighted shape+color distance between two garment-crop signatures."""
    return (sig_a[0] - sig_b[0]) + ((sig_a[1] - sig_b[1]) * COLOR_HASH_WEIGHT)


def _colors_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """
    Loose color compatibility. Claude's color wording drifts across frames
    of one garment ("beige"/"cream"/"tan"), so exact equality is too
    strict — but we still want black vs pink to block a merge. Treats
    colors as compatible if either is missing, they share a word, or both
    fall in the same coarse family.
    """
    if not a or not b:
        return True
    a, b = a.lower().strip(), b.lower().strip()
    if a == b or a in b or b in a:
        return True
    families = [
        {"black", "charcoal", "onyx", "jet"},
        {"white", "cream", "ivory", "off-white", "eggshell"},
        {"beige", "tan", "camel", "khaki", "sand", "nude", "taupe", "brown", "chocolate"},
        {"grey", "gray", "silver", "slate"},
        {"navy", "blue", "denim", "indigo", "cobalt"},
        {"pink", "rose", "blush", "salmon"},
        {"red", "burgundy", "maroon", "wine"},
        {"green", "olive", "khaki", "sage", "emerald"},
    ]
    for fam in families:
        if any(w in a for w in fam) and any(w in b for w in fam):
            return True
    return False


def _local_features(crop):
    """Bounded, foreground-only descriptors for geometric duplicate verification."""
    import numpy as np
    rgba = np.asarray(crop.convert("RGBA").resize((400, 400), Image.Resampling.LANCZOS))
    mask = (rgba[:, :, 3] >= 200).astype("uint8") * 255
    # Exclude cutout boundaries, where segmentation artifacts create false details.
    mask = cv2.erode(mask, np.ones((7, 7), dtype="uint8"))
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    points, descriptors = cv2.ORB_create(nfeatures=600).detectAndCompute(gray, mask)
    return (np.float32([point.pt for point in points]).reshape(-1, 2), descriptors)


def _geometric_duplicate(a, b):
    """Require distinctive, spatially distributed details with consistent geometry.

    This is supporting evidence, never proof of physical identity. Missing or
    textureless crops fail closed. No inference or network calls are made.
    """
    import numpy as np
    if a is None or b is None or a[1] is None or b[1] is None:
        return False
    if min(len(a[0]), len(b[0])) < 12:
        return False
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    def distinctive(source, target):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(source, target, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < 0.7 * n.distance}
    forward, reverse = distinctive(a[1], b[1]), distinctive(b[1], a[1])
    pairs = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    if len(pairs) < 12:
        return False
    src = np.float32([a[0][i] for i, _ in pairs])
    dst = np.float32([b[0][j] for _, j in pairs])
    transform, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if transform is None or inliers is None or not np.isfinite(transform).all():
        return False
    keep = inliers.ravel().astype(bool)
    if keep.sum() < 12 or keep.mean() < 0.75:
        return False
    # A matching logo or a small patch alone must not merge whole garments.
    for points, all_points in [(src[keep], a[0]), (dst[keep], b[0])]:
        area = cv2.contourArea(cv2.convexHull(points))
        total = cv2.contourArea(cv2.convexHull(all_points))
        if area < 1600 or area < 0.3 * max(total, 1):
            return False
    return True


def _foreground_palette(crop):
    """Small normalized colour histogram of opaque garment pixels, not background."""
    import numpy as np
    rgba = np.asarray(crop.convert("RGBA").resize((200, 200), Image.Resampling.LANCZOS))
    mask = (rgba[:, :, 3] >= 200).astype("uint8") * 255
    if np.count_nonzero(mask) < 100:
        return None
    lab = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2LAB)
    histogram = cv2.calcHist([lab], [0, 1, 2], mask, [8, 8, 8], [0, 256] * 3)
    return histogram / max(float(histogram.sum()), 1.0)


def _palette_matches(a, b):
    if a is None or b is None:
        return False
    distance = float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    return math.isfinite(distance) and distance <= 0.35


def _candidate_view(cand):
    return {"sig": cand["_crop_sig"], "features": cand.get("_local_features"),
            "palette": cand.get("_palette"), "frame_index": cand["_frame_index"]}


def _source_view(cand):
    appearance = cand.get('_source_appearance')
    if appearance is None or len(appearance['features'][0]) < 12:
        return []
    return [dict(appearance, frame_index=cand['_frame_index'], temp_id=cand['temp_id'])]


def _reconnect_source_tracks(tracks, better, report, scan_id):
    """Repair weak mask coverage only with surviving flow AND source geometry.

    No re-identification after complete point loss. At most three original-pixel
    references per track and forty eligible tracks; abstain on competing matches,
    overlapping observation intervals, or any missing identity evidence.
    """
    eligible = [t for t in tracks if t['category'] == 'shoes'
                and t['source_continuity_ids'] and t['source_views']]
    if len(eligible) > 40:
        _scan_event(report, scan_id, 'source_reconnection', decision='abstained', reason='track_limit')
        return tracks
    edges, degree = [], {}
    for i, a in enumerate(eligible):
        for b in eligible[i+1:]:
            family = _subtype_family(a['subcategory'])
            if not family or family != _subtype_family(b['subcategory']):
                continue
            # Co-visible pairs (even visually identical ones) cannot reconnect.
            if not (a['last_frame'] < b['first_frame'] or b['last_frame'] < a['first_frame']):
                continue
            witness = ((a['source_partial_ids'] & b['source_continuity_ids'])
                       | (b['source_partial_ids'] & a['source_continuity_ids']))
            if not witness:
                continue
            match = next(((u, v) for u in a['source_views'] for v in b['source_views']
                          if _palette_matches(u['palette'], v['palette'])
                          and _geometric_duplicate(u['features'], v['features'])), None)
            _scan_event(report, scan_id, 'source_reconnection', track_id=a['track_id'],
                        other_track_id=b['track_id'], decision='verified' if match else 'rejected',
                        reason='surviving_flow_and_source_geometry' if match else 'source_geometry_unconfirmed')
            if match:
                edges.append((a, b, match))
                for t in (a, b):
                    degree[t['track_id']] = degree.get(t['track_id'], 0) + 1
    removed = set()
    for a, b, (u, v) in edges:
        # No greedy choice or transitive appearance chain through another item.
        if degree[a['track_id']] != 1 or degree[b['track_id']] != 1:
            _scan_event(report, scan_id, 'source_reconnection', decision='abstained',
                        track_id=a['track_id'], other_track_id=b['track_id'], reason='ambiguous_matches')
            continue
        review_diagnostics=list(dict.fromkeys(a['best'].get('detection_diagnostics',[])+b['best'].get('detection_diagnostics',[])))
        a['best'] = better(a['best'], b['best'])
        if review_diagnostics:
            a['best'].update(detection_diagnostics=review_diagnostics,review_required=True,confidence='low')
        for field in ('observation_ids','source_exclusions'):
            a[field]=set(a.get(field,())) | set(b.get(field,()))
            a['best']['_'+field]=set(a[field])
        removed.add(b['track_id'])
        _scan_event(report, scan_id, 'tracking', decision='merged', method='source_foreground_geometry',
                    track_id=a['track_id'], absorbed_track_id=b['track_id'],
                    frame_index=v['frame_index'], reference_frame_index=u['frame_index'],
                    source_reference_ids=[u['temp_id'], v['temp_id']], selected_id=a['best']['temp_id'])
    return [t for t in tracks if t['track_id'] not in removed]


def _crop_visibility(crop):
    """Prefer retained opaque garment pixels over sharp but ghosted cutouts."""
    histogram = crop.convert('RGBA').getchannel('A').histogram()
    foreground = sum(histogram[16:])
    return sum(histogram[200:]) / max(foreground, 1)


def _subtype_family(value):
    value = str(value or '').strip().lower()
    # Classification wording can describe the same boot by shaft or construction.
    if value in {'boot', 'boots', 'ankle boots', 'sock boots', 'ankle boot', 'sock boot'}:
        return 'boots'
    return value


def _build_garment_tracks(candidates: List[dict], report=None, scan_id=None) -> List[dict]:
    """
    TEMPORAL GARMENT TRACKING (replaces the old frame-run grouping).

    Each candidate carries its frame_index, cleaned garment crop
    signature, classification, sharpness and confidence. We walk them in
    CAPTURE ORDER and attach each to an existing track when it is:
      - temporally NEARBY (within TRACK_MAX_FRAME_GAP of that track's last
        frame) — this is what stops a genuinely different but similar
        garment later in the video from being merged into an earlier one,
      - category-compatible,
      - color-compatible (loose matching, see _colors_compatible),
      - visually similar on the CROP signature.

    One track = one physical garment. We then keep the single BEST
    candidate per track (sharpest, preferring higher confidence).
    """
    tracks: List[dict] = []  # each: {last_frame, sig, category, color, best}

    def better(a: dict, b: dict) -> dict:
        """Pick the nicer candidate: confidence first, then sharpness."""
        rank = {"high": 2, "medium": 1, "low": 0}
        # A severe loss of foreground opacity is visible in the supplied boots
        # replay. Classifier confidence and edge sharpness cannot detect this.
        va, vb = a.get('_visibility', 1), b.get('_visibility', 1)
        if abs(va - vb) > 0.2:
            return a if va > vb else b
        if min(va, vb) >= .5 and a.get('_source_clipped', False) != b.get('_source_clipped', False):
            return b if a.get('_source_clipped', False) else a
        ra, rb = rank.get(a.get("confidence", "medium"), 1), rank.get(b.get("confidence", "medium"), 1)
        if ra != rb:
            return a if ra > rb else b
        return a if a["_sharpness"] >= b["_sharpness"] else b

    for cand in sorted(candidates, key=lambda c: c["_frame_index"]):
        best_match = None       # (distance, track) — the closest track this candidate legitimately belongs to
        best_rejection = None   # for logging: closest track we did NOT attach to, and why

        # IMPORTANT: evaluate EVERY track, don't stop at the first one that
        # rejects. An earlier-created track (e.g. the black shirt) will
        # reject a cream shirt on color — but the cream shirt's own track
        # may be further down the list. Bailing out early meant candidates
        # never reached their real match and spawned duplicate tracks.
        for tr in tracks:
            gap = cand["_frame_index"] - tr["last_frame"]
            anchor_distance = _crop_distance(cand["_crop_sig"], tr["sig"])
            dist = anchor_distance
            cat_ok = cand.get("category") == tr["category"]
            col_ok = _colors_compatible(cand.get("color"), tr["color"])
            subtype = cand.get("subcategory")
            subtype_ok = bool(subtype) and subtype == tr["subcategory"]
            timestamp = cand.get("_timestamp_seconds")
            last_timestamp = tr.get("last_timestamp")
            elapsed = timestamp - last_timestamp if timestamp is not None and last_timestamp is not None else None
            method = "crop_hash"
            reference_frame = tr["views"][0]["frame_index"]
            reason = None
            continuity = cand.get('_source_continuity')
            continuous_shoe = (cand.get('category') == tr['category'] == 'shoes'
                and bool(_subtype_family(subtype))
                and _subtype_family(subtype) == _subtype_family(tr['subcategory'])
                and continuity is not None and continuity in tr['source_continuity_ids']
                and elapsed is not None and math.isfinite(elapsed) and elapsed >= 0)
            co_visible = bool(set(cand.get('_source_exclusions', ())) & tr['observation_ids']
                              or tr['source_exclusions'] & set(cand.get('_observation_ids', (cand.get('temp_id'),))))
            if co_visible:
                reason = 'co_visible_distinct_items'
            elif cand.get('_identity_unverified') or tr['best'].get('_identity_unverified'):
                reason = 'unverified_source_region'
            elif continuous_shoe:
                # This identity was observed through EVERY intervening original
                # frame, including frames rejected for classification sharpness.
                method = 'source_foreground_flow'
            elif elapsed is not None and (not math.isfinite(elapsed) or elapsed < 0 or elapsed > 3.0):
                reason = "elapsed_time_outside_track_window"
            elif gap > TRACK_MAX_FRAME_GAP:
                reason = f"gap={gap}>{TRACK_MAX_FRAME_GAP}"
            elif not cat_ok:
                reason = f"category '{cand.get('category')}'!='{tr['category']}'"
            elif (cand.get("subcategory") and tr["subcategory"] and not subtype_ok
                  and _subtype_family(subtype) != _subtype_family(tr['subcategory'])):
                reason = "subcategory_mismatch"
            else:
                # Extra references must have been verified against the original
                # anchor. They cannot admit more references through chain matching.
                if subtype_ok and elapsed is not None and 0 <= elapsed <= 1.0:
                    for view in tr["views"][1:]:
                        d = _crop_distance(cand["_crop_sig"], view["sig"])
                        if d < dist and _palette_matches(cand.get("_palette"), view["palette"]):
                            dist, reference_frame = d, view["frame_index"]
                            method = "verified_view_hash"
                # Shoe silhouettes alone do not distinguish two similar pairs.
                # Real extracted candidates must also have distinctive details
                # or exact pixels. Ambiguous silhouettes go to review below.
                shoe_identity_required = cand.get('category') == 'shoes' and '_local_features' in cand
                exact_pixels = bool(cand.get('image_base64')) and cand.get('image_base64') == tr['best'].get('image_base64')
                if (col_ok and dist <= TRACK_SIMILARITY_THRESHOLD and (subtype_ok or not subtype or not tr['subcategory'])
                        and (not shoe_identity_required or exact_pixels)):
                    pass
                else:
                    # Colour words are fallible. Overriding them requires BOTH
                    # matching actual foreground colours and geometric evidence.
                    verified = False
                    if (_subtype_family(subtype) == _subtype_family(tr['subcategory'])
                            and elapsed is not None and 0 <= elapsed <= 1.0):
                        for view in tr["views"]:
                            if (_palette_matches(cand.get("_palette"), view["palette"])
                                    and _geometric_duplicate(cand.get("_local_features"), view["features"])):
                                verified = True
                                method = "local_features" if col_ok else "visual_colour_override"
                                reference_frame = view["frame_index"]
                                break
                    if not verified:
                        reason = (f"color '{cand.get('color')}' vs '{tr['color']}';visual_unconfirmed"
                                  if not col_ok else f"dist={dist:.1f}>{TRACK_SIMILARITY_THRESHOLD};visual_unconfirmed"
                                  if dist > TRACK_SIMILARITY_THRESHOLD else 'shoe_identity_unconfirmed')
            if reason is None:
                if best_match is None or dist < best_match[0]:
                    best_match = (dist, tr, gap, method, reference_frame)
                continue
            _scan_event(report, scan_id, "track_comparison", frame_index=cand["_frame_index"],
                        track_id=tr["track_id"], decision="rejected", reason=reason,
                        anchor_distance=float(anchor_distance), best_distance=float(dist),
                        colour_labels_compatible=col_ok, reference_frame_index=reference_frame,
                        elapsed_seconds=elapsed if elapsed is not None and math.isfinite(elapsed) else None,
                        geometry_window_eligible=elapsed is not None and math.isfinite(elapsed) and 0 <= elapsed <= 1.0,
                        subtype_family_compatible=_subtype_family(subtype) == _subtype_family(tr['subcategory']),
                        palette_available=cand.get('_palette') is not None and tr['views'][0]['palette'] is not None,
                        anchor_palette_matches=_palette_matches(cand.get('_palette'), tr['views'][0]['palette']))

            if best_rejection is None or dist < best_rejection[0]:
                best_rejection = (dist, reason)

        if best_match is not None:
            dist, tr, gap, method, reference_frame = best_match
            _scan_event(report, scan_id, "tracking", frame_index=cand["_frame_index"],
                        timestamp_seconds=cand.get("_timestamp_seconds"),
                        decision="merged", track_id=tr["track_id"], distance=float(dist), gap=gap, method=method,
                        reference_frame_index=reference_frame,
                        source_continuity_id=cand.get('_source_continuity') if method == 'source_foreground_flow' else None,
                        supported_source_points=cand.get('_source_continuity_evidence', {}).get('supported_points')
                            if method == 'source_foreground_flow' else None)
            review_diagnostics=list(dict.fromkeys(tr['best'].get('detection_diagnostics',[])+cand.get('detection_diagnostics',[])))
            tr["best"] = better(tr["best"], cand)
            if review_diagnostics:
                tr['best']['detection_diagnostics']=review_diagnostics
                tr['best']['review_required']=True
                tr['best']['confidence']='low'
            tr['observation_ids'].update(cand.get('_observation_ids', (cand.get('temp_id'),)))
            tr['source_exclusions'].update(cand.get('_source_exclusions', ()))
            tr['best']['_observation_ids']=set(tr['observation_ids'])
            tr['best']['_source_exclusions']=set(tr['source_exclusions'])
            if cand.get('_source_continuity') is not None:
                tr['source_continuity_ids'].add(cand['_source_continuity'])
            tr['source_partial_ids'].update(cand.get('_source_partial_continuity', set()))
            tr['source_views'] = sorted(tr['source_views'] + _source_view(cand),
                                        key=lambda v: len(v['features'][0]), reverse=True)[:3]
            tr["last_frame"] = cand["_frame_index"]
            tr["last_timestamp"] = cand.get("_timestamp_seconds")
            # Keep at most three views, always including the first. Admit new
            # references only by direct anchor hash + actual colour agreement.
            # A rescue via another view or geometry cannot extend the chain.
            anchor = tr["views"][0]
            anchor_distance = _crop_distance(cand["_crop_sig"], anchor["sig"])
            if (cand.get("subcategory") and cand.get("subcategory") == tr["subcategory"]
                    and 4 <= anchor_distance <= TRACK_SIMILARITY_THRESHOLD
                    and _palette_matches(cand.get("_palette"), anchor["palette"])):
                tr["views"].append(_candidate_view(cand))
                tr["views"] = tr["views"][:1] + tr["views"][-2:]
            # NOTE: tr["sig"] is deliberately NOT updated. The track's
            # signature stays anchored to the frame that created it.
            # Updating it per-frame let a single rembg-artifact frame poison
            # the track — the next clean frame of the same garment then
            # measured far from that corrupted signature and spawned a
            # duplicate. (Sharpness can't be trusted to reject artifact
            # frames either: smearing ADDS edges, so artifacts often score
            # as "sharper" than the clean capture.)
            if TRACK_DEBUG:
                print(f"[TRACK] frame {cand['_frame_index']} ({cand.get('subcategory')}) "
                      f"-> MERGED (gap={gap}, dist={dist:.1f})")
        else:
            tracks.append({
                "track_id": len(tracks),
                "source_continuity_ids": {cand['_source_continuity']} if cand.get('_source_continuity') is not None else set(),
                "source_partial_ids": set(cand.get('_source_partial_continuity', set())),
                "source_views": _source_view(cand),
                "first_frame": cand['_frame_index'],
                "observation_ids": set(cand.get('_observation_ids', (cand.get('temp_id'),))),
                "source_exclusions": set(cand.get('_source_exclusions', ())),
                "last_timestamp": cand.get("_timestamp_seconds"),
                "subcategory": cand.get("subcategory"),
                "views": [_candidate_view(cand)],
                "last_frame": cand["_frame_index"],
                "sig": cand["_crop_sig"],
                "category": cand.get("category"),
                "color": cand.get("color"),
                "best": cand,
            })
            _scan_event(report, scan_id, "tracking", frame_index=cand["_frame_index"],
                        timestamp_seconds=cand.get("_timestamp_seconds"),
                        decision="new_track", track_id=tracks[-1]["track_id"],
                        reason=best_rejection[1] if best_rejection else "first_track")
            if TRACK_DEBUG:
                if best_rejection:
                    d, reason = best_rejection
                    print(f"[TRACK] frame {cand['_frame_index']} ({cand.get('subcategory')}, "
                          f"{cand.get('color')}) -> NEW track. Closest rejected: dist={d:.1f}, "
                          f"blocked by: {reason}")
                else:
                    print(f"[TRACK] frame {cand['_frame_index']} ({cand.get('subcategory')}) -> first track")

    tracks = _reconnect_source_tracks(tracks, better, report, scan_id)
    if TRACK_DEBUG:
        print(f"[TRACK] === {len(candidates)} candidates -> {len(tracks)} garment tracks ===")

    # Keep ambiguous tracks reviewable. Do not erase similar shoes on colour/hash
    # alone. Compare actual palettes in the same short capture window; exact
    # duplicate bytes are safe to collapse even when label wording drifted.
    retained = []
    for tr in sorted(tracks, key=lambda t: (t['best'].get('_visibility', 1), t['best']['_sharpness']), reverse=True):
        candidate = tr['best']
        for previous in retained:
            if (candidate.get('_identity_unverified') or previous.get('_identity_unverified')
                    or set(candidate.get('_source_exclusions', ())) & set(previous.get('_observation_ids', ()))):
                continue
            a, b = candidate.get('_timestamp_seconds'), previous.get('_timestamp_seconds')
            if (a is None or b is None or abs(a-b) > 3.0
                    or candidate.get('category') != previous.get('category')
                    or not _subtype_family(candidate.get('subcategory'))
                    or _subtype_family(candidate.get('subcategory')) != _subtype_family(previous.get('subcategory'))):
                continue
            exact = candidate.get('image_base64') and candidate.get('image_base64') == previous.get('image_base64')
            if exact:
                candidate = None
                break
            palette_matches = _palette_matches(candidate.get('_palette'), previous.get('_palette'))
            _scan_event(report, scan_id, 'review_comparison',
                        frame_index=candidate['_frame_index'], reference_frame_index=previous['_frame_index'],
                        decision='flagged' if palette_matches else 'unconfirmed',
                        reason='matching_palette' if palette_matches else 'palette_missing_or_different',
                        palette_available=candidate.get('_palette') is not None and previous.get('_palette') is not None)
            if palette_matches:
                candidate['possible_duplicate_of'] = previous.get('temp_id')
                # Older apps understand low confidence and leave it unselected.
                candidate['confidence'] = 'low'
                break
        if candidate is not None:
            retained.append(candidate)
    for tr in tracks:
        _scan_event(report, scan_id, "track_result", track_id=tr["track_id"],
                    selected_frame_index=tr["best"]["_frame_index"],
                    returned=any(tr['best'] is item for item in retained))
    return sorted(retained, key=lambda item: item['_frame_index'])


def _sharpness_pil(img: "Image.Image") -> float:
    """Laplacian variance on a PIL image (stage-1 dedup uses this to pick the cleanest frame)."""
    import numpy as np
    arr = np.array(img.convert("L"))
    return cv2.Laplacian(arr, cv2.CV_64F).var()


def _remove_background(img: "Image.Image") -> "Image.Image":
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result_bytes = remove_background_bytes(buf.getvalue(), VIDEO_REMBG_MODEL)
    return Image.open(io.BytesIO(result_bytes))


def _frame_cutout(cutout: "Image.Image", canvas: int = 800, pad_ratio: float = 0.12) -> "Image.Image":
    """
    Normalizes a background-removed garment into a uniform, centered,
    padded square — so wardrobe cards all look consistent instead of the
    garment sitting at a random size/offset in the frame.

    This ONLY crops/centers/pads the REAL captured pixels — it does not
    invent, straighten, or hallucinate anything (per the spec's Goal 4).
    Steps:
      1. Ensure RGBA (rembg output has an alpha channel = the cutout mask).
      2. Find the bounding box of the actual garment (non-transparent pixels).
      3. Crop tight to that garment.
      4. Scale it to fit inside a padded square, preserving aspect ratio.
      5. Center it on a transparent square canvas.
    If the image has no detectable garment pixels, it's returned unchanged
    (the classifier will drop it anyway).
    """
    if cutout.mode != "RGBA":
        cutout = cutout.convert("RGBA")

    alpha = cutout.getchannel("A")
    bbox = alpha.getbbox()  # bounds of non-transparent content
    if bbox is None:
        return cutout  # nothing to frame

    garment = cutout.crop(bbox)
    gw, gh = garment.size

    inner = int(canvas * (1 - pad_ratio * 2))  # target content area inside padding
    scale = min(inner / gw, inner / gh)
    new_w, new_h = max(1, int(gw * scale)), max(1, int(gh * scale))
    garment = garment.resize((new_w, new_h), Image.LANCZOS)

    square = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    square.paste(garment, ((canvas - new_w) // 2, (canvas - new_h) // 2), garment)
    return square


def _classify_with_claude(img: "Image.Image") -> dict:
    """
    Same classification contract as the existing single-photo add flow —
    keeps WardrobeItem fields consistent whether an item came from a
    single photo or a video scan.
    """
    global claude_client
    if claude_client is None:
        claude_client = anthropic.Anthropic(max_retries=0, timeout=45)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    prompt = (
        "You are classifying a single clothing item cropped from a wardrobe scan video. "
        "The background has been removed. The garment may be partially cropped, touching an "
        "edge, at an angle, on a hanger, or not filling the whole frame — that is NORMAL and "
        "you should still identify it. Be forgiving: if you can tell what the garment is, "
        "classify it. Respond with ONLY valid JSON, no markdown, no preamble:\n"
        "{\n"
        '  "category": "top|bottom|dress|shoes|outerwear|bag|jewelry|accessory",\n'
        '  "subcategory": "ONE specific type, e.g. blazer, jeans, sneakers. Commit to a single '
        'answer — do NOT hedge with slashes or \'or\' (never \'poncho/cape\' or \'skirt or trousers\'). '
        'Pick the single most likely one.",\n'
        '  "color": "primary color name",\n'
        '  "colors": ["array", "of", "all", "visible", "colors"],\n'
        '  "style": "casual|formal|sporty|elegant|streetwear",\n'
        '  "season": ["spring","summer","fall","winter"] (all seasons this item suits),\n'
        '  "fabric_guess": "best guess at material",\n'
        '  "confidence": "low|medium|high" (use low only if you truly cannot tell what it is)\n'
        "}\n"
        "ONLY respond with {\"category\": null} if there is genuinely NO garment at all — "
        "e.g. an empty wall, a bare hanger, a hand, or a fully blurred frame with nothing "
        "identifiable. When in doubt and any garment is visible, classify it rather than reject it."
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

    cutout = await asyncio.to_thread(_remove_background, img)
    classification = await asyncio.to_thread(_classify_with_claude, cutout)

    if not classification.get("category"):
        raise HTTPException(
            status_code=422,
            detail="Couldn't identify a clothing item in this photo. Try a clearer, closer shot.",
        )

    cutout = _frame_cutout(cutout)  # normalize framing to match video-scan output
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


def _process_single_frame_sync(frame: "Image.Image", frame_index: int, temp_id: str, classification_cache=None):
    """
    Per-frame work: background removal -> normalized crop -> classification.

    Note the ORDER change: rembg + framing now run BEFORE dedup, because
    dedup fingerprints the cleaned garment crop (that's the whole fix).
    Exact repeated frames are reused by the caller; other frames still
    require classification before the existing category-aware tracker.

    Returns a candidate dict carrying _frame_index, _crop_sig, _sharpness
    for the tracker; those internal fields are stripped before returning
    to the client.
    """
    cutout = _remove_background(frame)
    # Empty segmentation is not a garment: avoid an unnecessary classifier call.
    if cutout.mode == 'RGBA' and cutout.getchannel('A').getbbox() is None:
        logger.info('scan_frame_cost rembg_calls=1 classification_calls=0 reason=empty_cutout')
        return None
    original_alpha = cutout.convert('RGBA').getchannel('A')
    cutout = _frame_cutout(cutout)  # normalize: crop tight, center, pad into uniform square

    digest = hashlib.sha256(cutout.tobytes()).digest()
    reused = classification_cache is not None and digest in classification_cache
    classification = classification_cache[digest] if reused else _classify_with_claude(cutout)
    if classification_cache is not None:
        classification_cache[digest] = classification
    logger.info('scan_frame_cost rembg_calls=1 classification_calls=%s', 0 if reused else 1)
    if classification.get("category") not in {'top', 'bottom', 'dress', 'shoes', 'outerwear', 'bag', 'jewelry', 'accessory'}:
        return None

    return _candidate_from_cutout(cutout, frame_index, temp_id, classification, original_alpha, frame)


def _candidate_from_cutout(cutout, frame_index, temp_id, classification, original_alpha=None, frame=None):
    """Shared candidate construction; caller has already obtained classification."""

    buf = io.BytesIO()
    cutout.save(buf, format="PNG")
    mask = source_mask(original_alpha) if original_alpha is not None and classification.get('category') == 'shoes' else None
    source_appearance = None
    if mask is not None and frame is not None:
        # Preserve original RGB detail; removal can erase texture in its RGB output.
        # The cached removal alpha still excludes the original background.
        original = frame.convert('RGBA')
        original.putalpha(original_alpha)
        original_crop = _frame_cutout(original)
        source_appearance = {'features': _local_features(original_crop),
                             'palette': _foreground_palette(original_crop)}
        original_crop.close()
        original.close()
    return {
        "temp_id": temp_id,
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "category": classification.get("category"),
        "subcategory": classification.get("subcategory"),
        "color": classification.get("color"),
        "colors": classification.get("colors"),
        "style": classification.get("style"),
        "season": classification.get("season"),
        "fabric_guess": classification.get("fabric_guess"),
        "name": classification.get("name") or f"{classification.get('color','')} {classification.get('subcategory','Item')}".strip(),
        "confidence": "low" if _crop_visibility(cutout) < 0.5 else classification.get("confidence", "medium"),
        "_frame_index": frame_index,             # internal: preserves capture order
        "_visibility": _crop_visibility(cutout),
        "_local_features": _local_features(cutout),
        "_palette": _foreground_palette(cutout),
        "_crop_sig": _crop_signature(cutout),    # internal: fingerprint of the CLEAN crop
        "_sharpness": _sharpness_pil(cutout),    # internal: for picking the best in a track
        "_source_mask": mask,
        "_source_appearance": source_appearance,
        "_source_clipped": bool(mask is not None and (mask[0].any() or mask[-1].any()
                                or mask[:, 0].any() or mask[:, -1].any())),
    }


def _process_detected_regions_sync(frame, removed, detection_text, frame_index, timestamp, source_id):
    """Local multi-item integration from an existing genuine response; no inference.

    Used by the gated video path and saved-response replay; never acquires responses.
    Invalid/unsegmented boxes keep a source preview and explicit review diagnostics.
    """
    import numpy as np
    from scan_regions import adapt, box_valid
    if frame.size!=removed.size:raise ValueError('Original/segmentation dimensions differ')
    result=adapt(detection_text,removed)
    candidates=[]
    ids={f'{source_id}:{record["index"]}' for record in result['records']}
    for record in result['records']:
        identity=f'{source_id}:{record["index"]}'
        raw=record['raw'] if isinstance(record['raw'],dict) else {}
        box=raw.get('bbox');mask=record['mask'];diagnostics=record['diagnostics']
        if mask is not None:
            piece=removed.convert('RGBA')
            alpha=np.asarray(piece.getchannel('A'))*mask.astype('uint8')
            piece.putalpha(Image.fromarray(alpha))
            cutout=_frame_cutout(piece)
            candidate=_candidate_from_cutout(cutout,frame_index,identity,record['metadata'],piece.getchannel('A'),frame)
            piece.close()
        else:
            # Show the proposed source area; never label it a recovered cutout.
            preview=frame.crop(tuple(box)) if box_valid(box,*frame.size) else frame.copy()
            cutout=_frame_cutout(preview.convert('RGBA'))
            candidate=_candidate_from_cutout(cutout,frame_index,identity,record['metadata'])
            preview.close()
        candidate.update(_timestamp_seconds=timestamp,_source_exclusions=ids-{identity},
            _observation_ids={identity},_identity_unverified=mask is None or any(e in diagnostics for e in
                ('ambiguous_subtype','invalid_category','missing_subtype','invalid_unit','invalid_item_id',
                 'shoe_count_contradiction','invalid_shoe_coordinates','shoe_outside_pair','invalid_confidence',
                 'missing_visible_shoe_count','contradictory_completeness','invalid_completeness')),
            review_required=bool(diagnostics),detection_diagnostics=diagnostics,
            detection_notes=record.get('notes',[]),
            detection_region=box if box_valid(box,*frame.size) else None,
            image_kind='cutout' if mask is not None else 'unverified_source_region')
        if diagnostics:candidate['confidence']='low'  # Older clients also require review.
        candidates.append(candidate);cutout.close()
    return candidates,{'diagnostics':result['diagnostics'],'raw_detection':result['raw'],
        'records':[{'index':r['index'],'diagnostics':r['diagnostics'],'notes':r.get('notes',[])} for r in result['records']]}


def _process_video_sync(video_path, diagnostic_only=False, report=None, scan_id=None, region_detector=None, frame_issues=None, cancel_event=None):
    """Stream sampled frames: only one raw frame is retained at a time.

    Exact repeated samples reuse the previous result before rembg/Claude.
    They still enter tracking at their original index, preserving track gaps.
    Near-duplicate and angle matching remain the existing crop tracker.
    """
    scan_id = scan_id or str(uuid.uuid4())
    cap = cv2.VideoCapture(video_path)
    candidates = []
    sampled = 0
    processed = 0
    repeated = 0
    opportunities = 0
    sharpness_rejected = 0
    previous_digest = None
    previous_result = None
    exact_results = {}  # Scan-local, bounded by the <=20-second sampled video.
    classification_cache = {}  # Exact normalized pixels only; never shared across jobs.
    continuity = SourceContinuity() if not diagnostic_only else None
    started = time.monotonic()
    issues=frame_issues if frame_issues is not None else []
    if region_detector is not None:
        from scan_detection import MAX_REQUESTS, MAX_SAMPLE_FRAMES, MAX_CANDIDATES, MAX_FRAME_EDGE as DETECTION_EDGE, DetectionLimit
    def issue(code, timestamp, **details):
        entry={'code':code,'timestamp_seconds':round(timestamp,4),**details}
        issues.append(entry)
        _scan_event(report,scan_id,'frame_issue',**entry)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not (0 < fps <= 240):
            raise ValueError("Could not read video frame rate.")
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if count / fps > MAX_VIDEO_SECONDS:
            raise ValueError(f"Video too long. Maximum {MAX_VIDEO_SECONDS} seconds.")
        interval = max(1, int(fps * FRAME_SAMPLE_INTERVAL_SEC))
        _scan_event(report, scan_id, "video", fps=float(fps),
                    reported_frame_count=float(count), sample_interval_frames=interval,
                    sharpness_threshold=MIN_FRAME_SHARPNESS, diagnostic_only=diagnostic_only)
        idx = 0
        while True:
            if cancel_event is not None and cancel_event.is_set():raise InterruptedError('Scan cancelled')
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= int(fps * MAX_VIDEO_SECONDS) + 1:
                raise ValueError("Video exceeds the scan duration limit.")
            if continuity is not None:
                continuity.advance(frame, idx / fps)
            if idx % interval == 0:
                opportunities += 1
                score = float(_sharpness(frame))
                accepted = score >= MIN_FRAME_SHARPNESS
                _scan_event(report, scan_id, "sampling", decoded_frame_index=idx,
                            timestamp_seconds=round(idx / fps, 4), sharpness=score if math.isfinite(score) else None,
                            decision="accepted" if accepted else "sharpness_rejected",
                            frame_index=sampled if accepted else None)
                if not accepted:
                    sharpness_rejected += 1
                    idx += 1
                    continue
                frame_index = sampled
                if region_detector is not None and (sampled>=MAX_SAMPLE_FRAMES or len(candidates)>=MAX_CANDIDATES):
                    issue('processing_limit_reached',idx/fps);break
                if region_detector is not None and region_detector.attempts>=MAX_REQUESTS and hashlib.sha256(frame.tobytes()).digest() not in exact_results:
                    issue('request_limit_reached',idx/fps);break
                sampled += 1
                if diagnostic_only:
                    idx += 1
                    continue
                digest = hashlib.sha256(frame.tobytes()).digest()
                is_reuse = digest in exact_results
                if is_reuse:
                    repeated += 1
                    cached = exact_results[digest]
                    result = [dict(r) for r in cached] if region_detector is not None else dict(cached) if cached is not None else None
                else:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    try:
                        if region_detector is not None:
                            img.thumbnail((DETECTION_EDGE,DETECTION_EDGE),Image.Resampling.LANCZOS)
                        elif MAX_FRAME_EDGE > 0:
                            img.thumbnail((MAX_FRAME_EDGE, MAX_FRAME_EDGE), Image.Resampling.LANCZOS)
                        try:
                            if region_detector is None:
                                result = _process_single_frame_sync(img, frame_index, f"scan_{frame_index}", classification_cache)
                            else:
                                removed=None
                                try:
                                    removed=_remove_background(img)  # exactly once per new original frame
                                    if cancel_event is not None and cancel_event.is_set():raise InterruptedError('Scan cancelled')
                                    text=region_detector.detect(img,removed)  # one request for ALL items, even with an empty mask
                                    result,detail=_process_detected_regions_sync(img,removed,text,frame_index,idx/fps,f'{scan_id}:{frame_index}')
                                    if detail['diagnostics']:issue('detection_incomplete',idx/fps,diagnostics=detail['diagnostics'])
                                    rejected=[r for r in result if r.get('category') is None]
                                    if rejected:issue('rejected_detections',idx/fps,records=[{'id':r['temp_id'],'diagnostics':r['detection_diagnostics']} for r in rejected])
                                    result=[r for r in result if r.get('category') is not None]
                                except InterruptedError:raise
                                except Exception as exc:
                                    issue(str(exc) if isinstance(exc,DetectionLimit) else 'detection_failed',idx/fps,error_type=type(exc).__name__)
                                    result=[]  # cached failure: no retry or scalar paid fallback
                                finally:
                                    if removed is not None:removed.close()
                        except Exception as exc:
                            _scan_event(report, scan_id, "processing", frame_index=frame_index,
                                        timestamp_seconds=round(idx / fps, 4),
                                        decision="error", error_type=type(exc).__name__)
                            raise
                    finally:
                        img.close()
                    processed += 1
                    previous_digest = digest
                    previous_result = [dict(r) for r in result] if region_detector is not None else dict(result) if result is not None else None
                    exact_results[digest] = previous_result
                if region_detector is not None:
                    if len(candidates)+len(result)>MAX_CANDIDATES:
                        issue('candidate_limit_reached',idx/fps,unprocessed_detections=len(result));break
                    results=result
                    ids={f'{scan_id}:{frame_index}:{i}' for i in range(len(results))}
                    for i,r in enumerate(results):
                        identity=f'{scan_id}:{frame_index}:{i}'
                        r.update(temp_id=identity,_observation_ids={identity},_source_exclusions=ids-{identity})
                    _scan_event(report,scan_id,'multi_detection',frame_index=frame_index,
                        timestamp_seconds=round(idx/fps,4),source='exact_reuse' if is_reuse else 'processed',candidate_count=len(results))
                else:
                    results=[result] if result is not None else []
                if region_detector is None:
                    _scan_event(report, scan_id, "classification", frame_index=frame_index,
                            timestamp_seconds=round(idx / fps, 4),
                            source="exact_reuse" if is_reuse else "processed",
                            decision="accepted" if result is not None else "no_category_or_invalid_json",
                            category=result.get("category") if result else None,
                            subcategory=result.get("subcategory") if result else None)
                for result in results:
                    mask = result.get('_source_mask')
                    if result.get('category') == 'shoes' and mask is not None and not result.get('_identity_unverified'):
                        result['_decoded_frame_index'] = idx
                        result['_source_family'] = _subtype_family(result.get('subcategory'))
                        identity, evidence = continuity.identify(mask, _subtype_family(result.get('subcategory')))
                        result['_source_continuity'] = identity
                        result['_source_continuity_evidence'] = evidence
                        if evidence.get('observed_fraction', 1) < .75:
                            result['_source_clipped'] = True
                        _scan_event(report, scan_id, 'source_continuity', frame_index=frame_index,
                                    timestamp_seconds=round(idx / fps, 4), identity=identity, **evidence)
                    result["_frame_index"] = frame_index
                    result["_timestamp_seconds"] = round(idx / fps, 4)
                    if region_detector is None:result["temp_id"] = f"scan_{frame_index}"
                    candidates.append(result)
            idx += 1
    finally:
        cap.release()
    _scan_event(report, scan_id, "sampling_summary", decoded_frames=idx,
                sampling_opportunities=opportunities, accepted_frames=sampled,
                sharpness_rejected=sharpness_rejected, diagnostic_only=diagnostic_only)
    if diagnostic_only:
        return [], sampled, 0
    if cancel_event is not None and cancel_event.is_set():raise InterruptedError('Scan cancelled')
    reconcile_backward(video_path, candidates, idx, fps)
    if cancel_event is not None and cancel_event.is_set():raise InterruptedError('Scan cancelled')
    tracks = _build_garment_tracks(candidates, report=report, scan_id=scan_id)
    duplicates = len(candidates) - len(tracks)
    final = []
    for result in tracks:
        public = {k: value for k, value in result.items() if not k.startswith("_")}
        final.append(DetectedItem(**public))
    logger.info("[SCAN] sampled=%s processed=%s exact_reused=%s tracks=%s returned=%s seconds=%.1f",
                sampled, processed, repeated, len(tracks), len(final), time.monotonic() - started)
    return final, sampled, duplicates


async def _run_video_scan_job(job_id: str, tmp_path: str, push_token: Optional[str]):
    """
    The actual scan work, run in the background AFTER the app already
    got its job_id response back. Same extraction/dedupe/classify
    pipeline as before — just no longer blocking the HTTP request.
    """
    cancel_event=threading.Event()
    frame_issues=[]
    try:
        # One video runs at a time per process, including extraction.
        # Shield + await prevents cancellation from releasing the lock while
        # its underlying thread is still processing.
        async with _scan_lock:
            # Old clients cannot display frame-level failures. Require both the
            # new client capability and the separately authorized server gate.
            detector=_make_region_detector() if _jobs[job_id].get('multi_item_capable') else None
            work = asyncio.create_task(asyncio.to_thread(_process_video_sync, tmp_path, scan_id=job_id,
                region_detector=detector,frame_issues=frame_issues,cancel_event=cancel_event))
            try:
                items, frames_scanned, duplicates_removed = await asyncio.shield(work)
            except asyncio.CancelledError:
                cancel_event.set()  # Stop after in-flight work; never start another frame/request.
                while not work.done():
                    try:await asyncio.shield(work)
                    except asyncio.CancelledError:continue  # repeated cancellation must not release the worker lock
                    except Exception:break
                with suppress(Exception):
                    work.result()
                raise
        if frames_scanned == 0:
            raise ValueError("No usable frames found. Try a brighter, steadier scan.")

        _jobs[job_id].update({
            "status": "done",
            "items": items,
            "frames_scanned": frames_scanned,
            "duplicates_removed": duplicates_removed,
            "frame_issues": frame_issues,
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
    except asyncio.CancelledError:
        if job_id in _jobs:_jobs[job_id].update(status='error',error='Scan cancelled')
        raise
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
        if job_id in _jobs:
            _jobs[job_id]["finished_at"] = time.time()
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
    scan_capability: Optional[str] = Form(None),
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

    if sum(job["status"] == "processing" for job in _jobs.values()) >= MAX_PENDING_SCANS:
        raise HTTPException(status_code=429, detail="Scanner is busy. Please try again shortly.")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing", "created_at": time.time(),
                     "multi_item_capable": scan_capability == 'multi_item_regions_v1'}
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video too large. Maximum 50 MB.")
                tmp.write(chunk)
        background_tasks.add_task(_run_video_scan_job, job_id, tmp_path, push_token)
    except BaseException:
        _jobs.pop(job_id, None)
        if tmp_path:
            with suppress(OSError):
                os.unlink(tmp_path)
        raise

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
        frame_issues=job.get("frame_issues"),
    )
   
