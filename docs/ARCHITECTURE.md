# Architecture

Mouth/expression matching for a Premiere template + Cubase template. This is **not** phoneme-level lip sync. Categories are broad: `A`, `E`, `O`, `HAHA`, `CRY`, plus `CLOSED` / `NEUTRAL`.

## Observed Soda Pop project

Completed example:

`D:\Editor\Soda Pop\soda pop\Adobe Premiere Pro Auto-Save\Soda pop_1--54ea6537-c9cb-4b62-45a0-a808f7e2364b-2026-09-21_20-49-47.prproj`

| Object | Role |
| --- | --- |
| `Sequence 04` | Source sequence. ~27 min master clip `YTSave_YouTube_Friendly-Monster-Stories-for-Kids_Media_N22ZKABi_Ws_001_1080p.mp4` |
| `PJ 5 - demo` | Template. V2 holds **nested sequences named `1`–`14`**, reused in a rhythmic pattern (174 instances). A1 holds the beat WAV (~65.8 s) |
| Nested sequences `1`–`14` | The actual **slots**. Each is a short take cut from the source. Editing a nested sequence updates every instance on `PJ 5 - demo` |
| `D:\Editor\Voice\*.wav` | Short mouth/expression reference audio (`a.wav`, `oh.WAV`, `heheha.WAV`, …) |
| `soda pop.cpr` | Cubase template with Sampler Track 01–16 |

So “place into the corresponding slot in PJ5-demo” means: fill nested sequence `N` with the matched source in/out. The template already sequences those nested sequences on the timeline.

## Data contract: `edit-plan.json`

Every module reads/writes this file. It is the only coupling between analysis and host adapters.

```
audio files ──► audio analyzer ──► categories + duration
source video ──► video analyzer ──► timed mouth/expression segments
template     ──► premiere inspector ──► slots (nested seqs, times, tracks)
                                          │
                                          ▼
                                       matcher
                                          │
                                          ▼
                                   edit-plan.json
                                      │        │
                                      ▼        ▼
                            Premiere UXP     Cubase adapter
                            (fill slots,     (stems + mixdown WAV)
                             import mix,
                             save as)
```

## Modules

| Package | Responsibility | Host needed? |
| --- | --- | --- |
| `autoedit.audio` | Classify each WAV by filename lexicon + acoustic shape | No |
| `autoedit.video` | Sample frames, detect face/mouth, emit category segments | No |
| `autoedit.match` | Four tiers: speaking face, closeup, body shot, stable shot with a person; full cut duration, source gaps and no overlap; face or independently confirmed body at every sample | No |
| `autoedit.plan` | Join matches with Premiere slots + Cubase tracks | No |
| `autoedit.premiere.prproj` | Offline gzip-XML inspector | No |
| `plugins/premiere-uxp` | Apply plan inside Premiere via official UXP | Premiere |
| `autoedit.cubase` | Sampler WAVs, verified Cubase 13 track-archive import, timing preview | Cubase needed to load samples and render MIDI/FX |

## Incremental completion

`until_filled` is the default pipeline mode. Candidate discovery visits 30-second
source blocks for reference jobs and jobs with automatic cast ranking disabled.
Automatic cast ranking uses 30-second prefixes at `incremental_fps` (default 2)
and stops after a complete allocation meets the verification policy. Explicit `full` scans every
configured source up to the analysis limit. Both rank individual identities by
exposure and never pool co-visible lookalikes. Optional-lip animation jobs with
the installed object detector defer face/pose landmarks to selected cuts;
independent face/prop queries still qualify candidates before allocation.
`verification_mode: review` retains this exposure ranking and sparse eligibility,
caps identity features at four per track, and transcribes only selected ranges.
It exports all slots without the dense rejection/reselection gate. Every cut and
audio choice is marked for manual review, with dense and onset verification
flags false. The following dense verification requirements describe `verified` mode.
Independent tight animation face queries supplement human landmarks. Contextual
props and measured stationary screen overlays are rejected before ranking; strong
face evidence can resolve weak plush-label confusion, while paper/logo rejection
remains mandatory. Clear-face jobs require readable face evidence across each cut
and repeat selected-actor verification densely before publishing a plan.
Spatial Lab frame signatures group recurring compositions with complete-link
distance, preventing gradual camera moves from bridging unrelated views. Frames
are decoded at actual analysis timestamps and cached by source metadata, sample
times and signature version. Greedy allocation, bounded search and dense
reselection share the configurable two-cut composition limit. Actor exposure
remains the preference within eligible distinct views; temporal distance alone
does not make repeated wide shots diverse. A stronger independent tight face
can resolve weak plush labels and near-full-frame poster/sheet ambiguity;
localized printed faces and logos remain excluded.
Remaining face/costume ambiguity schedules a separate person query. A localized
human body must contain the independently measured tight face in its head region;
this confirmation can override plush labels but cannot override localized paper
or printed posters. Dense faces inherit person-class evidence only from the
selected nearby actor, with fresh face measurements and spatial overlap.
Central watermark/logo regions corroborated by localized text panels are marked
as promotional content and excluded by every shot tier, including co-visible
face tracks. Subtitle bands and corner channel marks remain eligible content.
For incremental jobs, candidate discovery visits 30-second
source blocks at 1 fps. Explicit body references use bounded tiled CCIP batches,
cached by source, timestamps, model fingerprint and tile version. These features
can be compared with new references without decoding again. Face references and
automatic cast discovery use the object/face path. Body screening schedules
separated clip-sized verification windows; remaining positions are still searched
before exhaustion. Reference screening only requests
dense work; it never qualifies footage. The range analyzer verifies promising
regions at 6 fps with bracketing context. The callback enriches new frames with
object or dense imported-body identity evidence, updates identities and tries the full Premiere allocation,
including any empty intro. Only a verified allocation with valid sampler bindings
can stop analysis. Insufficient footage/dense lip rejection continues the scan;
invalid inputs, models and runtime errors abort it. Sparse misses trigger a 2 fps
discovery pass, then dense verification of remaining regions before source exhaustion.
Sources are visited in input order until enough footage is verified. `full` retains
whole-source analysis; `search_fps: 0` retains the legacy dense prefix callbacks.

Character exposure is measured over analyzed observations. `verified_ranges`
specifies continuous dense evidence; searched timestamps do not fill holes.
Matcher free ranges and source pooling preserve these restrictions, including
after onset refinement. Region boundaries reset tracks; chunk boundaries within
continuous evidence are not shot boundaries. Expansion reuses
independent object observations and crop embeddings. With references, each new
region is enriched once and namespaced track IDs preserve its completed evidence;
reference IDs remain shared. A final whole-search cast ranking occurs only for
exposure fallback after source exhaustion. The reference detector performs one
query, with FP16 on CUDA and bounded batches of two by default. Reference-crop
detections may shorten the first discovery query; unclear crops and later passes
use broad queries. Strict lip failures persist as source-local rejected intervals.

Explicit body references without mandatory lip motion use dense tiled CCIP
verification before loading the object detector. Each qualifying frame has a
measured identity feature; a consistent track needs at least two observations.
Unmatched frames remain holes. These tiles carry no measured face/mouth geometry.
Their candidate analyzer skips face and pose inference, preserves native-frame
transition detection and uses a separate cache identity. Dense tile qualification
retains the stricter original-image CCIP limit; permissive discovery and alternate
views cannot relax it. Face references and mandatory lip mode keep their models.
Only tracks confirmed against the original image seed at most eight alternate
views per target/model. These alternate views guide discovery; adaptation remains within the original-distance bound;
adapted tracks never seed further views. Cached dense regions are reconsidered
when this gallery changes. Model hashes are reused within a process while file
metadata is unchanged. Before exposure fallback, original face observations are
restored and independent object detection supplies the whole cast.
Native hard-shot events stay protected. Background-only safety events can be
suppressed when at least three contiguous dense samples bracket the event,
identify the same imported target and retain consistent appearance features.
Original events and suppressed timestamps remain in the evidence; exposure
fallback restores the original transition checks.

Optional reference images are analyzed once per content/model/region fingerprint and
cached independently of video embeddings. The GUI displays face suggestions and
allows explicit whole-character crops. Undetected group images require selection,
never a whole-image identity fallback. Head and body templates use matching crop
features. Imported targets bypass category/prop exclusions, including vehicles and
robots, but remain provisional until consistent reference identity is established.
Group images provide separate face
templates; complete-link grouping merges only strongly matching alternate views,
with reference-image co-occurrence constraints. Each video track must match one
target consistently across multiple crops. Ambiguous identities are excluded.
Cuts integrate identity evidence for one dominant reference target rather than
aggregating unrelated imported characters. Both greedy and bounded allocation
prefer least-used eligible targets, while retaining full duration, gap, speech
and transition validation. This is best-effort diversity, not a quota that extends
the sequential scan. Reference-only allocation runs before fallback; fallback
is enabled only after configured sources/limits are exhausted. Multi-source
pooling canonicalizes template provenance across models conservatively and does
not prefix reference IDs. Final plan metadata reports selected cuts and duration
per target; timeline slot order and project length remain unchanged.

Audio uses stable overlapping core windows with one owner for each word/event.
Dense lip caches belong to individual cuts and their geometry, independently of
cast rank labels. Raw analysis checkpoints are atomic and may be resumed after a
one-second state warmup. Final video-only assets are made only after verification
and only for used sources; their original packet timestamps remain unchanged.

## MVP vs later

MVP (this tree):

1. Inspect a `.prproj` and list `PJ 5 - demo` slots.
2. Classify a folder of voice WAVs.
3. Decode source video sequentially with PyAV at 6 samples/second by default. Animation searches overlapping image regions for missed faces. Use mandatory MediaPipe landmarks, face/mouth clarity and normalized lip movement in time windows. Confirm bodies with independent pose detection and torso/limb geometry. Require a face or confirmed body at every sample across each selected cut and its brackets, including intro. Stable scenery alone cannot qualify a cut. Account for frame timestamp quantization; reject missing samples and fail when there are too few qualifying cuts. Cache identity includes source kind and detector version, independently of source gap. OpenCV geometry is diagnostic only and cannot qualify a cut.
4. Write `edit-plan.json`.
   Before selection, form multiple face tracks bounded by transitions, extract ArcFace/CCIP ONNX features from representative source-resolution crops, and cluster tracks using complete-link cosine distances with co-occurrence constraints. Rank clusters by exposure and distinct abrupt shots; automatically select 75% face exposure. Retry full slot allocation with main, uncertain, then supporting subjects. Persist `character_analysis.json` and per-cut `character_selection`; propagate reviews into UXP template sequence markers. Base detection and embedding caches are independent of cast-ranking settings.
5. Prepare one sample per scene run and a Premiere timing preview. `Import Cubase` exports selected sampler tracks, imports only updated instrument/channel settings into the open project, and verifies MIDI/tempo/plugin state through read-back. It leaves the project unsaved for review. Export the real MIDI/FX mix from Cubase.
6. UXP panel: load plan, fill nested sequences and an optional empty intro gap, import mixdown. The intro adds no audio track or scene index. The user saves the project in Premiere.

Later increments (not in MVP):

- Live Cubase event placement if Steinberg ever ships a timeline API
- Per-instance trim of nested-sequence in-points on `PJ 5 - demo` V2
- Learned viseme classifier for a specific character
- Watch-folder batch jobs
