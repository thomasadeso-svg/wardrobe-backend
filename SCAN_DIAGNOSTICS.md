# Video pan diagnostics (review branch only)

This change adds observability, not a recognition fix. Sharpness remains 22,
sampling remains every 0.5 seconds, and the garment-track threshold remains 20.
Normal endpoint response shapes and item-selection logic are unchanged. No
frontend, deployment configuration, requirements or billing settings are changed.

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
missing samples without changing existing tracking behavior.

These diagnostics cannot independently prove whether rembg removed a garment
incorrectly or Claude misclassified it. Comparing the original video with sample
timestamps is the first step; further controlled cutout inspection may be needed.

## Tests

```sh
python -m unittest discover -s . -p 'test_scan_diagnostics.py' -v
```

Eight offline tests exercise functions compiled directly from video_scan.py with
fake decoder/model dependencies. They cover no-inference diagnostic mode, exact
reuse labels, timestamps, unchanged accepted-frame numbering, null classifications,
error redaction/capture cleanup, duration rejection, threshold behavior and NaN
JSON safety. Syntax compilation also passes. No live API or real OpenCV/model
integration test was run in the preparation environment. Real-video validation
remains pending. No CI workflow was added or run.

Do not merge until reviewed. The production branch is main; merging may trigger
Railway deployment. A draft PR or branch may trigger separately configured preview
or CI automation; this change does not alter or disable repository automation.
