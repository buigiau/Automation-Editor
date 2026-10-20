# Golden: fresh CPU solo-person output — 2026-10-18

The final fresh run produced **14/14 fills and 14 sampler WAVs in 216.560 seconds
(3:37 rounded)**. The previous output at `D:\golden\191026` took 1676.295 seconds
(27:56), so the measured fresh-output speedup on this source is **7.74×**.

Both runs use the same Golden project pair and source:
`YTSave_YouTube_Media_UuyUdjBy-LM_Kids-Turn-Into-Superheroes-in-a-Superhero-School_001_1080p.mp4`.
The source duration is 1256.589 seconds (20:56.6). This is a different source
from the earlier 29:51 Talking Tom benchmark.

| Measurement | Previous output | Final fresh output |
| --- | --- | --- |
| Wall time | 1676.295 s | 216.560 s |
| AI/discovery stage | 1671.053 s | 205.497 s |
| Candidate classifications | 57 | 16 |
| Qwen candidate classifications | 57 (4B CPU) | 0 |
| Completed fills / sampler scenes | 14 / 14 | 14 / 14 |
| Solo scenes | Not measured in old format | 14 |
| Group-scene fallback selections | Not measured in old format | 0 |
| Identity unresolved | Old color/species labels are estimates | 11 |

The final run uses `--cold --phases first` and a newly created application cache.
The command refuses to reuse a nonempty cache. Voice analysis, discovery,
selected-frame audits and video-only assets were generated for this run;
`ai_cache_hit=false`. No cached rebuild timing is included. Model weights are
installed beforehand, and OS disk caching is not cleared. Two preceding fresh
trials measured 234.107 s and 267.310 s; they did not supply the final cache.
An interim 202.618-second output contained one leg-only crop and was rejected.
The final version rejects person detections cut off above the frame when no
face is visible; replaying that exact crop confirms that it is now rejected.

The CPU path uses [OpenCV Zoo's YOLOX detector](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox),
MediaPipe face landmarks and the existing aligned ArcFace model. Preview
detection uses 320-pixel model input; proposed cuts are checked at 640 pixels.
All three samples (15%, 50%, 85%) must contain one detected person and no
detected neighbour. At least two agreeing readable face samples are required
to recognize an identity. Masks, profiles and small faces can remain useful
solo footage while retaining cut-local `person-unresolved-*` labels and review
flags. No Qwen vision runner was loaded for this run; GPU inference was disabled.

Discovery seeks across the source and stops as soon as the slots are valid.
The first pass skips non-reference decoder frames. If short, it retries native
cadence before permitting group scenes. Every allocated cut still goes through
native-frame transition checks and visual duplicate checks. Duplicate-check
samples share one seek instead of decoding repeatedly from a keyframe.
Group scenes are deferred during the solo search and flagged when used as a
shortage fallback. Same-shot fallback retains source gaps and duplicate checks.

Validation:

- The complete plan has no validation errors, 14 distinct slot IDs, and 14 nonempty
  sampler files. All rows have `subject_count=1` and `group_scene_fallback=false`.
- Manual inspection uses three additional source frames per cut at 10%, 50%,
  and 90%, recorded beside the contact sheet. This checks sampled appearance,
  not every frame or speech synchronization.
- 154 focused Python tests pass, including deferred groups, a native-cadence retry
  before group fallback, extra people in a middle sample, changing/hidden face
  identities, single-seek duplicate samples and Premiere export metadata.
- All 49 mocked Premiere panel tests pass. Real Premiere/Cubase host application,
  final playback and lip synchronization are outside the timing measurement.

Artifacts:

- `output/golden-fast-solo-cold-20261018f/report.json`
- `output/golden-fast-solo-cold-20261018f/first/ai_review.html`
- `output/golden-fast-solo-cold-20261018f/first/edit-plan.json`
- `output/golden-fast-solo-cold-20261018f/selected-frames-audit.jpg`
- `output/golden-fast-solo-cold-20261018f/manual-visual-review.json`

The configured Qwen fallback and saved local model now default to 2B, CPU with
two inference threads. An already-open GUI must be restarted to read the new
saved setting. References and footage outside the fast-person route can still
need Qwen and take longer; this result is not a universal runtime guarantee.
