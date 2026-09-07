# Video pan diagnostics (review branch only)

This branch adds scan diagnostics and a conservative shoe duplicate fallback.
Sharpness remains 22, sampling remains every 0.5 seconds, and the ordinary
crop-hash threshold remains 20. If that hash rejects a nearby shoe candidate,
matching foreground ORB features plus RANSAC geometry can confirm the merge.
It requires the same nonempty shoe subtype, compatible nonempty color, and a
last-seen timestamp within one second. At least 12 mutual distinctive matches,
75% geometric inliers, and broad spatial coverage are required. Features stay
anchored to the first crop, avoiding frame-to-frame identity drift.
All tracks also expire after a three-second nominal timestamp gap, including
when blur rejection compresses accepted-frame numbering.

No extra Claude calls are added. This fixes result grouping after classification;
it does not save those existing API calls. Small additional local OpenCV work is
required. Same-model, similar-looking shoes may still be merged; textureless or
heavily occluded items may still duplicate. This is a draft, not a verified
production resolution. Endpoint response shapes and dependencies are unchanged.

## Local diagnostic-only mode

In a backend environment with the existing requirements installed, run:

```sh
python diagnose_scan.py original-camera-video.mp4
```

The original camera video must be at most 20 seconds. This command calls the same
decoder, sampler and sharpness gate used by normal scans. It stops before hashing,
background removal, classification and tracking. No Anthropic API key is needed;
the Anthropic client is now initialized lazily only for actual classification.
No model session is created and no inference/API calls are made in diagnostic mode.
Existing backend packages are still required to import the module.

JSON on stdout contains timestamps, sharpness scores and acceptance decisions.
It contains no images or extracted thumbnails, and does not preserve the input
video: keep your own original. Screen recordings are not equivalent to the
original camera file because scaling, overlays and compression affect sharpness.
This is a local command, not a new public endpoint or an option in the mobile app.
If run on paid hosting, CPU/RAM usage can still cost money; no-cost here means no
Claude/rembg inference, not a promise of free hosting.

## Normal-scan logs (only after an approved deployment)

Filter for `[SCAN_DIAG]` and correlate by `scan_id` (the job ID). Events include:

- `video`: reported FPS/frame count and active sampler settings.
- `sampling`: each scheduled frame, timestamp, sharpness and gate decision.
- `sampling_summary`: decoded frames, sampling opportunities, accepted/rejected.
- `classification`: accepted/no-category-or-invalid-JSON and exact reuse status.
- `processing`: exception type on failure, without exception messages or responses.
- `tracking`: merged/new track, distance/gap when merged and nearest rejection
  reason when starting a new track.
- `track_result`: chosen frame for each track and whether it survived the item cap.

Ordinary `[TRACK]` and `[SCAN]` logs remain available. Detailed events do not log
pixels, base64, API keys, prompts or full API responses. Category/subcategory text
is included in normal-scan classification metadata. Timestamp is decoded index/FPS
(nominal time, not exact presentation time for variable-frame-rate videos).

For compatibility, frame_index still numbers sharpness-accepted samples rather
than all scheduled frames. decoded_frame_index and timestamp_seconds disambiguate
missing samples. Tracking additionally checks nominal timestamp gaps.

These diagnostics cannot independently prove whether rembg removed a garment
incorrectly or Claude misclassified it. Comparing the original video with sample
timestamps is the first step; further controlled cutout inspection may be needed.

## Tests

```sh
python -m unittest discover -s . -p 'test_scan_diagnostics.py' -v
```

Ten offline pipeline/tracker tests and four real OpenCV geometry tests pass.
Geometry tests cover rotation, unrelated textures, blank crops, and a small
shared patch. The first ten tests isolate functions with fake inference and
decoder dependencies; they are not an end-to-end backend import test.

Local fixture check, 2026-09-07: five shoe crops at 10.5, 11, 12, 13, and 14 seconds
from the supplied IMG_5223.mp4 matched in all ten pairwise geometry comparisons.
All fifteen shoe-to-top comparisons were rejected (tops at 1.5, 6.5, 7.5 seconds).
Crops used official u2net weights through OpenCV DNN with rembg preprocessing
and app framing. This is not Railway's ONNX Runtime engine. These checks validate
the geometry helper; they do not run live Claude classification or prove the
full tracking result. Two distinct similar shoes remain an outstanding fixture.
No paid APIs, production scan, or CI workflow were run for this update.

Do not merge until reviewed. The production branch is main; merging may trigger
Railway deployment. A draft PR or branch may trigger separately configured preview
or CI automation; this change does not alter or disable repository automation.
