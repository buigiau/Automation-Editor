# Performance update — 2026-10-11

## Fast export with manual review

`video.verification_mode: review` retains character identities, exposure-ranked
cast preference, complete slot durations, source-gap rules and sparse face
eligibility. It uses up to four image features per track, transcribes only
the provisional selected cuts with surrounding context, and exports without
the dense face/lip/articulation rejection loop. All scenes and voice choices
are labelled for manual review; no dense-verification flag is asserted.
The GUI exposes this option; the local CPU configuration now enables it.
The default/library and example configuration remain `verified`.

The completed fresh-analysis-cache Golden run took **366.25 seconds (6m 6s)**
on CPU, producing 14 nested scenes, 14 sampler WAVs, the Premiere apply payload,
timing audio preview and listening review. It stopped at 270 seconds of the
1781-second source. Rankings therefore describe analyzed footage, not the full
video. The minimum source gap is 5.02 seconds. This is a draft export for manual
verification, not a densely verified edit or a rendered Premiere/Cubase project.

| Fresh-cache stage | Seconds |
| --- | ---: |
| Character identity/ranking | 115.27 |
| Recurring compositions | 16.76 |
| Allocation attempts | 14.00 |
| Speech around selected cuts | 43.93 |
| Video-only stream copy | 0.89 |
| Sampler files and listening review | 0.32 |

The inclusive source-processing time is 364.37 seconds; subtracting the named
character/composition/allocation/speech stages leaves about 174.4 seconds for
decoding, sparse face/pose analysis and other preparation. These stages overlap
the inclusive total and must not be summed with it. System peak RAM use was
7.81 GB, including other applications; this is not the job's own memory usage.
Character runtime reported only `CPUExecutionProvider`; CUDA was hidden.
Artifacts and measurements: `output/speed-111026/review-cold-final`.
Three-frame visual review identified scene 9 as a distant dark shot and scene
10 as a printed ID-card face. Both need replacement or explicit user acceptance;
all scenes and voice choices remain marked for manual review.

The previous cold probe selected an allocation but its benchmark console callback
failed to encode a Vietnamese filename under Windows cp1252, stopping export.
The reproduction script now writes UTF-8; the 366.25-second result is a separate
successful run with a new analysis-cache directory.

The first completed review export reused the prior cancelled run's frame and
crop caches: 14 nested scenes and 14 sampler tracks in 64.90 seconds on CPU.
Character inference took 7.62 seconds, compositions 10.94 seconds, selected-range
speech 43.03 seconds; dense verification was deferred. This is a warm-cache
measurement, not a fresh-video time. Artifacts are in
`output/speed-111026/cpu-only`; `review-1.jpg` and `review-2.jpg` show three frames
per cut, and `audio_review.html` compares original audio with selected WAVs.

## Strict verification diagnosis

The cancelled Golden run in `D:\golden\111026` spent 5942.1 seconds
(99.0 minutes) without producing a plan: 2365.6 seconds in video analysis,
3574.9 seconds in object/face queries. It was processing a 1781-second source
at 6 fps across the entire source before allocation.

Changes for automatic cast jobs:

- `until_filled` now honors early stopping. Cast exposure is ranked within the
  analyzed footage; choose `full` for ranking across every configured source.
- Incremental preparation uses 2 fps by default. Optional-lip animation jobs
  with the installed detector use independent actor/face/prop queries before
  allocation and defer the initial face/pose pass. Selected cuts still undergo
  dense face/lip verification, audio checks, composition limits and source gaps.
- Physical tracks allow one native frame of timestamp rounding. A 25 fps
  source sampled at 2 fps has alternating .48/.52-second gaps; a strict .50
  limit previously split valid tracks. Missing samples and shot changes still
  reset tracks. Embedding-track caches include the native frame rate.
- CPU is the default for speech, character embeddings and object detection.
  The local Golden configuration disables the heavy object detector, retaining
  face/pose evidence. OpenCV/BLAS/ONNX inference uses two compute threads;
  Whisper uses four threads and one worker on the 16-thread host. GUI workers
  have below-normal process priority; explicitly CPU jobs hide CUDA devices.
  `AutoEdit-CPU.cmd` starts the installed environment with CUDA hidden.
- Whisper allocations are released before dense verification.
- Dense selected-face queries inherit the configured device, batch size and
  detector enable flag; they previously hardcoded `auto` and batches of four.
- Decoder ETA excludes time spent checking allocations and running models.
- Recurring-composition signatures are cached per source timestamp. Growing a
  prefix decodes only newly requested frames; grouping retains the same evidence
  and composition limit as a complete scan.

The development GPU runs saturated VRAM and were stopped. They did not produce
a verified export and do not establish a 10–15 minute whole-project time.
Reproduction: `output/speed-111026/run_golden.py`, now using the CPU configuration.
Use a separate output/cache directory to measure a fresh run; the script does
not import into Premiere or Cubase.

A bounded CPU-only probe of the first 30 seconds of the same Golden source
prepared 60 observations in 17.39 seconds and measured character features in
15.79 seconds (33.18 seconds combined). Character inference reported only
`CPUExecutionProvider`. This is preparation time, not complete export time.
Report: `output/speed-111026/cpu-probe/measurement.json`.

Regression suite on CPU with CUDA hidden: 641 passed, 15 skipped. The final
review metadata change also passed all 32 subject-quality tests.

The bounded whole-project CPU run stopped at 899.98 seconds without an export.
It found a provisional 14-cut allocation at 270 source seconds, but dense face
verification rejected eight cuts; only six passed that allocation's face check.
The CPU path therefore has not demonstrated the requested 10–15 minute export.
Object detection took 0 seconds, character processing 204.11 seconds, source
speech 130.37 seconds, dense verification 173.94 seconds, allocation 62.46
seconds, and recurring compositions 104.74 seconds. Stage totals overlap with
the inclusive source-processing measurement. The composition-frame reuse fix
was written during this run and applies to newly started workers, so this run
still measured the older repeated-prefix composition decoding.
Progress and reusable analysis checkpoints are in `output/speed-111026/cpu-only`;
Strict timings are preserved as `strict-before-review-performance.json` and
`strict-before-review.log` in that directory. That strict run was stopped.

## Earlier measurements — 2026-10-09

The implementation uses source and reference evidence, with no character-name,
McQueen-specific, franchise or filename branch.

## Changes

- Discover candidates sparsely, verify only promising regions densely, and stop
  after the complete allocation and existing audio/lip checks succeed.
- Reuse frame detections and image features across growing search regions.
  Previously measured tracks are not reclustered across the whole prefix at
  each step. New image references reuse compatible source features.
- Explicit body references use image screening. When lip motion is optional,
  candidate preparation skips face/pose inference while retaining transitions
  detected from native frames. Face references, mandatory lips and full scans
  retain their respective analysis paths.
- Discovery is permissive; dense tile identity uses the original image's stricter
  CCIP limit. Alternate views cannot widen that acceptance limit.
- Imported reference types are not excluded by category. If references cannot
  fill the project after the configured source search, independent whole-cast
  analysis enables the existing exposure-ranked fallback. Independently detected
  and confidently ranked vehicle/robot/toy candidates remain eligible in that
  explicit fallback without being labelled as matches to the input image.
- GUI and CLI share a process lock. Concurrent projects wait before loading
  heavy models. Detector auto precision uses CUDA FP16 / CPU FP32, two-frame
  batches, a bounded frame cache and reduced batches on GPU memory errors.
- Cancellation handles queued and active jobs; finished evidence stays cached.

## Measurements

Candidate preparation on three restored local sources, at 6 samples/s for the
same four-second interval (30–34 s), 24 observations each:

| Source type | Face/pose preparation | Visual candidate preparation | Ratio |
| --- | ---: | ---: | ---: |
| Animated humanoids — Spidey | 4.511 s | 0.663 s | 6.80× |
| Animated nonhumans — Minions | 3.115 s | 0.494 s | 6.31× |
| Live action — Venom | 1.285 s | 0.467 s | 2.75× |

These measurements include decoder/model initialization and measure candidate
preparation only. They do not measure complete-project speed or recognition
accuracy, and are single runs rather than averages. Optional visual-body jobs
can use this preparation path; these numbers do not mean every processing mode
skips its required face/lip models.

Raw report: `output/general-speed-20261009/candidate-preparation.json`.
Verification: **591 tests passed, 15 skipped**.

An earlier development run produced 14 McQueen slots in 220.5 seconds with
partially populated caches. Visual review found a firetruck mistaken for the
reference, so that export was rejected. Its time is not an accepted final
benchmark. The final implementation retains strict original-image checks;
there is no completed fresh-cache project benchmark for this final revision.
No guarantee of a fixed project duration is made: source coverage, references,
hardware and required lip analysis still affect total time.

To repeat a whole-project benchmark on any source configuration:

```powershell
.venv-gpu\Scripts\python.exe scripts/benchmark_pipeline.py --config path/to/project.yaml --device cpu --modes until_filled --repeats 2
```

The first run uses a new output/cache directory; the second reuses that cache.
Repeat `--config` with another project and add `--concurrent` to measure queue
wait separately from processing time. Results include actual performance and
memory statistics. No source-specific configuration is required by the code.
