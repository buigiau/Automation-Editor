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
