# Video pan matching revision 2

Production logs from scan 7bb16d9d-e289-487f-bf13-8601e573de3e show two
polo splits at crop distance 27 (threshold 20) and a sneaker split at distance
11 caused by Claude's grey/white colour-label disagreement. Nineteen accepted
frames became six tracks. No exact reuse occurred; processing took 103.6 seconds.

This revision:
- Checks actual foreground Lab colour histograms plus geometric evidence before
  overriding a colour-label disagreement. It never treats grey and white labels
  alone as equivalent.
- Computes local features for all garment categories, including polos.
- Keeps at most three reference views per track. Additional references must match
  the original anchor at hash distance <=20 and agree on actual foreground colour.
  Candidates rescued only through other views cannot become new references.
- Limits supplemental matching to one second since last appearance and matching
  nonempty subtype. Different known subtypes cannot merge through the base hash.
- Logs each rejected track comparison, including anchor and best-view distances.

The main hash threshold remains 20. API contracts, sampling, and the number of
classification calls remain unchanged. Added local feature work can increase CPU
usage; API or Railway cost savings are not established.

## Validation and limits

Initial validation: 19 tests passed (14 isolated pipeline/tracker tests and five real OpenCV geometry/
colour tests). The distance-27 polo test uses synthetic signatures to verify the
bounded-reference rule; it does not reproduce the actual latest polo cutouts.
The grey/white test verifies colour override needs both pixel colour and geometry.
Tests also reject different colours, different subtypes, blank crops, small
shared patches, and unbounded reference drift.

Saved original-video cutouts: four adjacent shoe comparisons (10.5/11, 11/12,
12/13, 13/14 seconds) pass both geometry and colour checks. The two saved polo
views (6.5/7.5 seconds) pass colour comparison but FAIL geometry. The tank/polo
comparison fails both. Thus a reliable polo fix is not established; multiple
reference hashes may help, but the original video from the latest production
scan is needed to validate that failure. Similar distinct shoes remain untested.

The saved cutouts used official u2net weights through OpenCV DNN with rembg
preprocessing, not Railway ONNX Runtime. No paid APIs were used in these checks.
No full Claude-to-app scan was performed for this revision.

On the September 8 resume, all 14 isolated tests passed again. The local OpenCV
binary crashed on import (bus error), including in an isolated import command;
the five geometry/colour tests could not be rerun in that environment. The prior
19-test result remains historical evidence, not a fresh complete test pass.

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

Do not merge until reviewed. The production branch is main; merging may trigger
Railway deployment. A draft PR or branch may trigger separately configured preview
or CI automation; this change does not alter or disable repository automation.
