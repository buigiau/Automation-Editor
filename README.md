# Automation-Editor

The default GUI now opens the **Preset + AI** workflow: choose Soda Pop,
Golden or Cosita, an output folder and source videos, with optional character
images. Set template pairs, Voice and the AI backend once in **Thiết lập**.
Use `AutoEdit-Local.cmd` for Qwen3-VL 4B Instruct locally without an API key,
or select Gemini and save its key. Local Voice estimates require playback review.
Local defaults to 2 CPU threads; GPU requires an explicit choice in settings.
AI shortages now default to a complete draft: preserve validated cuts and fill
remaining slots with explicitly marked footage for review. This also handles local
inference timeouts; `ai.shortage_policy: strict` restores failure on missing scenes.
See [AI workflow, review and cache](docs/AI_WORKFLOW.md) for setup and usage.
`autoedit gui --legacy` opens the previous local-analysis GUI described below.

See [performance changes and measured limits](docs/PERFORMANCE.md) for the
latest speed work and verification results.

Run `AutoEdit-CPU.cmd` for the CPU configuration. Speech and character models
default to CPU. GUI analysis workers limit compute threads and use below-normal
priority. Restart the GUI after changing `config.yaml` so it loads these settings.

Use **Fast export; review faces, lips and voice afterward** (or
`video.verification_mode: review`) to export the complete allocation before
dense face/lip/onset checks. Character exposure ranking and source gaps remain
active. This mode uses up to four identity images per track and transcribes
only the selected source ranges. Every scene and voice is marked for manual
review; `verification_mode: verified` restores the dense verification gate.

## Candidate search and queued jobs

The GUI and CLI now share an OS job lock. A second project waits automatically
before loading analysis models or touching its run outputs. A crashed process
releases the lock. **Cancel** removes a queued job or stops an active job; closing
the GUI also cancels its active analysis. Completed frame batches and analysis
regions remain cached for the next run. Cancel does not create a partial edit plan.

In verified mode, for imported character references, `until_filled` first discovers candidate character regions at
`video.search_fps: 1`. It verifies only candidate regions at the configured
`sample_fps` (normally 6), and stops after a complete allocation, intro and the
existing mouth/audio checks pass. Sparse observations never qualify a cut.
If necessary, search adds 2 fps observations, then verifies the remaining footage
densely before enabling exposure-ranked reference fallback. There is no automatic
10-minute cutoff. `video.search_fps: 0` restores the legacy dense prefix search;
`analysis_mode: full` still analyzes all selected sources. Automatic cast selection
now honors `until_filled`: it ranks the footage analyzed so far in 30-second
increments at `video.incremental_fps: 2` and stops when a complete allocation
meets the chosen verification policy. Choose `full` when cast exposure must cover every source.

Explicit body reference crops use tiled image embeddings for discovery and
dense reference verification. Localized whole-character crops use
`body_reference_cosine_distance: 0.30`. Tiles use the stricter original-image
identity limit (`ccip_cosine_distance: 0.20`); the looser discovery limit and
alternate views cannot certify a cut. The first pass prefers sustained sparse hits and defers isolated
flashes. Promising clip-sized regions are checked at the
configured full sample rate, with per-frame identity evidence and the existing
transition checks. Image tiles identify the imported target; they do not supply
face landmarks or lip motion. Body-only visual verification skips the face/pose
models while retaining native-frame transition detection. Mandatory lip mode
keeps the face/object path.
`video.reference_discovery: objects` retains object-based discovery and verification
for comparison; face references also keep that path. Weaker positions remain
available in later passes and dense exhaustion.

Confirmed body views can guide discovery of another angle, bounded by the original
reference and model fingerprint. Adapted matches cannot grow this gallery or
relax dense tile identity checks.
Earlier dense features are reconsidered only when confirmed views change.
If the reference search is exhausted, independent whole-cast detection runs
before exposure-ranked fallback.

Reference jobs use one detector query per verification region. Each new region is enriched
once; expanding the search reuses its identities instead of recomputing the whole
searched prefix. Exposure ranking is recomputed across all searched regions only
if reference fallback becomes necessary. Selected input crops can supply a shorter
first-pass query; unclear crops and later discovery passes retain the broad query.

The detector defaults to batch 2, a 64 MiB frame cache and two CPU threads.
`compute_type: auto` uses FP16 on CUDA and FP32 on CPU. Set `float32` explicitly
to retain that precision. GPU memory pressure still reduces the batch size.
Its weights are released between inference stages instead of being moved into
host RAM. Request IDs and explicit success/error states prevent a failed worker
from reading an older result; each completed detector batch is checkpointed.

`performance.json` separates processing time from queue wait and reports sampled
system RAM pressure. Detector metadata reports its peak allocated/reserved VRAM.
`video_analysis_meta.analyzed_sources` includes `verified_ranges`,
`verified_seconds`, `searched_until_sec` and `dense_region_count`; the last
searched timestamp does not imply all earlier footage was verified densely.

Compare actual project completion and cache reuse with:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/benchmark_pipeline.py --config soda.yaml --config golden.yaml --modes until_filled legacy --repeats 2 --concurrent
```

The two projects start together and analysis queues automatically. Reports include
queue wait, processing time, memory and completed slot counts. Run with recovered
original source/reference files to reproduce an earlier job. Unit tests and short
GPU smoke checks do not establish a sub-10-minute completion time or character
recognition accuracy on a full project.

## Optional character references and articulation checks

In the GUI, use **Target characters (optional) > Import images** to choose
portraits of different characters, alternate views, or a thumbnail with multiple
characters. **Review / crop** shows detected face suggestions and lets you drag
a separate box around each desired face or whole character, including an animated
vehicle. Click a box to remove it. **Clear** restores automatic exposure ranking.
The CLI accepts repeated `--character-reference actor-a.png --character-reference group.jpg`,
or set `video.characters.reference_images` in YAML. Human references need a
readable face. Each detected face in a group image gets its own identity template;
strongly matching alternate views across images are merged conservatively, but
co-visible faces remain separate. Unreadable detected regions are reported;
an image with no usable identity fails validation. A cartoon image with no
isolated face requires an explicitly selected character region; a group image is
never silently embedded as one identity. YAML can set `video.characters.reference_regions`
as a mapping from each image path to a list of `{box: [x, y, width, height], kind: body}`
or `kind: face` regions in original image pixels. Head references are compared with
head features, and whole-character references with body features, using the same
identity model and preprocessing. With imported references, detector labels such
as vehicle, robot, toy or prop do not exclude a target. Such detections remain
provisional until multiple consistent observations match the selected reference.
Multiple consistent track observations are required; one similar frame or a
lookalike ensemble does not establish a reference match. Evidence must consistently
match one target; switching between imported characters cannot qualify a track.

With references, incremental analysis stops early only after enough verified
reference-character cuts are available. Otherwise it searches every selected
source up to `video.max_seconds` (zero means no limit) before enabling the
exposure-ranked fallback. These cuts are labelled in
`video.character_selection.reference_fallback`. Without references, the
existing stop-when-filled behavior remains. Timeline order and slot durations
stay as configured in the project.

Among eligible reference cuts, selection prefers characters used fewer times,
then less selected duration, before ordinary exposure/quality preferences.
This increases variety without relaxing scene, mouth-motion or audio checks.
Distribution is best effort: missing footage, short appearances, insufficient
slots or failed articulation checks can leave a target unselected. Adding targets
expands eligible footage, not the project's slot count. Analysis still stops once
the project is filled; it does not scan further solely to give every target a cut.
Per-cut `character_selection.reference_id` identifies the chosen target.
`video_analysis_meta.character_reference_selection` and the log report selected
counts/durations per target and fallback cuts. Multi-source pooling keeps imported
identities shared, while ordinary video track IDs remain source-local.

Current voice taxonomy overrides stale catalogs in every matching path.
Laughter, coughs, groans and burps cannot enter ordinary speech/neutral fallback
through a matching mouth shape. Vocal events need source-event evidence; the
current detector does not detect burps, so normal source analysis never assigns
them automatically. Unknown labels remain uncertain rather than generic speech.

Selected readable mouths are refined with 350 ms context and original-frame
measurements around the opening. Articulation checks distinguish opening,
reshaping and closed-consonant release from preparation/closing. Cuts may move
to the measured onset; full-window person/character evidence, gaps, boundaries
and durations are then checked again. Visual fallback remains an estimate
(`phoneme_sync_verified: false`); unreadable mouths require review. These local
checks do not claim exact phoneme recognition. Cubase MIDI is preserved and
`cubase_timing_verified` remains false until actual playback is checked; the
Premiere preview does not simulate MIDI/FX.

Reference matching can reduce wrong-character choices, but searching for a
late character and stricter onset checks may increase runtime. Compare real
video results and `performance.json` before claiming an accuracy or speed gain.

## Stop when enough scenes are verified

`video.analysis_mode: until_filled` searches each source in 30-second
increments, then densely checks promising regions. After each verified region
it tries to fill every unique Premiere slot and any empty intro, then verifies
the selected mouths and prepares the scene/sample mapping. It stops source
analysis as soon as a complete allocation passes those checks. Repeated instances
of a nested sequence do not require additional source cuts.

Automatic cast selection ranks individual characters in the analyzed footage;
`full` ranks after scanning all sources up to `video.max_seconds`. Co-visible lookalikes and
geometric species tags are not pooled into one identity. Measured screen overlays,
independently detected props and uncorroborated cartoon face hypotheses are removed
before ranking. When clear faces are required, a cast identity needs repeated face
observations; body-only detections cannot promote furniture into the cast. With
references, early stopping still ranks only the analyzed footage. CLI accepts
`--analysis-mode full` and `--analysis-chunk-sec 30` on `autoedit run`.

Each dense region retains face/pose/tracking and transition state while decoding.
One second of measured context protects its boundaries. Failed dense mouth
checks exclude those cuts and continue searching. The slot lengths, source gaps,
person/prop checks, transition rules and strict speaking-mouth policy still apply.
If all selected sources or `video.max_seconds` are exhausted, the run fails rather
than publishing an incomplete plan. Existing plans remain in place on failure;
check `run.log` for the latest run status before applying a previous plan.

Completed analysis checkpoints, per-frame object detections, per-crop embeddings,
audio windows and individual dense cut checks are cached. A resumed analysis
warms detector/tracker state over the last second, then processes new frames.
The isolated object worker reuses weights and decoded frames between actor and
context queries. It selects CPU/CUDA, defaults to batch 1 (explicit `auto` uses
available VRAM), reduces larger batches on memory errors and reports the actual
runtime. Detector weights are released between stages so character/Whisper models
can use the GPU without copying the detector into host RAM.
Object `compute_type: float16` is opt-in pending real-video quality review;
float32 remains the default. Changing GPU alone does not establish accuracy.

`edit-plan.json` includes the analyzed ranges and stop reason. `performance.json`
records total wall time and stage timings; `source_processing_inclusive` includes
the allocation callbacks, so stage timings must not be summed. Video-only assets
are created for used sources after verification, using stream copy of the original
video; this final copy can still read the entire used file without analyzing it.

For real-input comparisons on each target machine:

```text
python scripts/benchmark_pipeline.py --config config.yaml --device auto
python scripts/benchmark_pipeline.py --config config.yaml --device cpu
```

Each invocation creates a fresh benchmark output directory. The first run of each
mode uses a cold cache; the second reuses it. Timing includes interpreter/model
startup and artifact preparation. The report records whether the run met the
10-minute target. It does not measure Premiere/Cubase import or final rendering,
and timings are not a recognition-quality evaluation. Review selected cuts and
audio before enabling float16; full and early-stop runs may select different
characters because they analyze different amounts of footage.

## Multiple source videos

In the GUI, use **Add videos…** to select several files at once or add more
files later. **Remove selected** removes highlighted files from the list.
The selector searches all selected videos to fill the existing Premiere slots.
Each cut stays inside one video, keeps its required duration, and respects
the source gap between cuts from the same file. Lip checks and audio review
use the corresponding original video; Premiere fills use its video-only copy.
Character identities and exposure ranks are analyzed independently per video.
Adding sources increases available footage; every selected cut still needs
to pass the configured person, transition and lip-motion checks.

CLI: repeat `--video`, for example
`autoedit run --video "film1.mp4" --video "film2.mp4"`.
YAML: `premiere.source_media: ["film1.mp4", "film2.mp4"]`.
Existing configurations with a single source path still work.

## Sound-led scene selection

The pipeline scans the original soundtrack across the analyzed part of the
source video before choosing cuts. Confident words that match the shared Voice
library supply candidate cut onsets. Each cut keeps the Premiere slot duration,
person/face requirements, transition checks, character filters and source gaps.
Character exposure rank takes priority among qualifying cuts. Within each cast
rank, qualifying sound onsets take priority; visual selection fills remaining slots.
Words with Whisper probability at least 0.8 rank first; supported words from
0.6 to 0.8 rank second and require listening review. Lower scores fall back.

English sample names and Vietnamese phonetic names share a pronunciation index.
CMU pronunciations support homophones: `know` can use `no.WAV` or `no2.WAV`,
`I`/`eye` can use `ai.WAV`/`aii.WAV`, and `my` can use `mai.WAV`/`mái.WAV`.
The Vietnamese vowel `a.wav` is distinct from the English article `a`.
Vietnamese spellings are curated in `audio/pronunciation.py`; stress and repeated
letters describe variants, while consonants and vowel identities must agree.
Matching compares complete word pronunciations, not arbitrary substrings.

Only the sound at the opening of a scene determines its WAV. A selected word
onset is carried into the plan with its supported sample paths; later words
cannot replace it. Visual fallback cuts may match their first sound within
250 ms of the cut. Mid-word cuts and unsupported openings use measured mouth
shapes and timing, marked for review. Each scene has one WAV, including later reprises.

The default requires readable faces (`video.require_clear_face: true`) while
`video.require_lip_motion: false` permits silent faces and unreadable lips. Facial
sharpness/size is measured separately from mouth clarity. At least 85% of each
cut must have a readable face, with face evidence at both brackets; final cuts
are checked again densely. A failed face check reselects footage with the same
durations and source gaps. Insufficient qualifying footage fails the job instead
of filling with body-only scenes. Explicitly disable **Require clear faces** or set
`video.require_clear_face: false` to permit body shots, including body-only reference
targets. Uncertain audio matches still need review.
Recurring camera compositions are measured from spatial image colour structure,
independently of actor identity and source time. `video.max_similar_cuts: 2`
limits each similar view to two cuts, including distant returns to the same
wide shot. Within that limit, exposure-ranked actors retain priority. When
primary views cannot fill the timeline, qualifying supporting views are used;
the limit also applies during alternative allocation and dense reselection.
Set the value to `0` to disable it. A source with too few distinct eligible
views fails allocation instead of silently repeating the same composition.
Central promotional text widgets corroborated by a localized panel are excluded;
ordinary subtitles and corner channel logos do not trigger this content filter.
Ambiguous costumes can receive a separate person query: an independently localized
face must sit in the confirmed person's head region. Body boxes alone never
establish faces, and localized printed props remain excluded.
The GUI's **Require speaking mouths** checkbox enables strict lip selection.
In that mode the opening must show speech and mouth shapes must change within
the selected cut, above pixel noise; a static open mouth or camera pan cannot qualify.
Animation analysis adds paired eyes/pupils, shared face colour and a mouth detector
for flat coloured cartoons, alongside MediaPipe. With the installed object model,
an independent tight face query also measures stylised faces missed by human
landmarks. Broad cartoon/body boxes cannot supply face evidence. Prop and screen
overlay checks apply to these faces, and selected cuts repeat the face query densely
against the selected actor. Face localizations never supply mouth geometry or speech.
Scenery and character embeddings alone never establish a face. Both
source types search overlapping face regions. Brief detection misses may use
reversible, distributed face feature tracks for up to one second after a measured
face; these tracks establish visibility, never mouth geometry or speech. Tracking
uses actual shot boundaries, rather than resetting at every motion safety event.
The unvoiced intro is reserved before voiced scenes and does not require lip motion.
New video, character and lip cache versions automatically replace stale analyses.
Motion-aware transition checks retain real cuts, flashes and dissolves. A detected
hard cut clears the half-second comparison history so it does not mark ordinary
frames in the next shot as more transitions.
The strict policy verifies each selected cut again at 20 samples/s
and reselects cuts that have static/unclear mouths. It fails if alternatives cannot
be verified, instead of silently assigning an unrelated neutral WAV. If greedy
selection fragments the available footage, a bounded search tries alternative
qualifying windows with unchanged lengths and gaps before reporting insufficient footage.
When dense verification rejects a candidate, its failed interval is excluded
without adding a source gap; only selected cuts and a reserved intro reserve gaps.

Source analysis uses Whisper word timestamps and a bundled CMU dictionary;
dictionary matching does not verify the actual waveform or accent. Existing
spectral detectors suggest cough, sigh and laugh events outside transcribed
speech. Strict speaking selection excludes event hints below 0.85 probability;
the bundled spectral hints are not confirmed laughter. Event and pronunciation
matches are marked for listening review.
Unknown words/pronunciations fall back instead of forcing a nearest match.
Phonetic substitutions also abstain when the source word's possible readings
do not agree on a sample, such as reduced `to` versus stressed `too`. Known
elongated interjections (`ohhhh`, `heyyy`, `whaaat`) normalize to their word.
Long soundtracks are decoded in bounded chunks with overlapping ASR context.

The default shared library is the user's downloaded Voice folder. The GUI or
`audio.directory` can select another folder. Runs write `source_transcript.json`,
`source_sound_matches.json`, the selected `video.source_sound`, and the audio
match evidence into the output directory. `audio_review.html` lets you compare
source audio and selected samples. Existing Cubase MIDI and FX remain in place;
the generated timing preview does not verify the final Cubase render.

Install for CPU with `pip install -e ".[cpu,dev]"` or `requirements.txt`;
run checks with `python -m pytest`.
`scripts/check_sound_cases.py` runs actual WAV sentences through Whisper and
the selector, using synthetic visual evidence to isolate audio/cut behavior.
Its output includes misses and event probes; passing unit tests does not
establish recognition recall or actual mouth synchronization on a film.

## Choosing sampler tracks

Leave **Sampler tracks** blank in the GUI to keep the existing mapping.
To skip sampler 1, enter `2-13` before Validate / Run. The field also accepts
individual numbers and ranges such as `1,3-5`, in increasing order. Select
exactly one sampler per Premiere scene run. Missing tracks or a count mismatch
stop the job before source analysis.

For Zoomally, the 12 scenes in `BEAT MAU` (footage sequences `4` through `15`)
map to `SAMPLER TRACK 2` through `SAMPLER TRACK 13`. Track names are read without
case sensitivity and preserved exactly for Cubase import. In Cubase, manually
select these same 12 tracks before clicking **Import Cubase**. The importer
checks the selected track names against the plan; sampler 1 is excluded from
the generated bundle and import. Existing MIDI and effects use the existing
import workflow.

CLI: add `--sampler-tracks 2-13` to `autoedit run`.
YAML: set `cubase.sampler_tracks: "2-13"` (a list such as `[2, 3, 4]` also works).
An empty string restores the default mapping. Regenerate the plan after
changing the sampler selection.

For the Dame Tu Cosita pair (`DAMTUCOSITA flip_3.prproj` and
`DAME TU COSITA new.cpr`), enter `2-15`. The 14 numbered tracks map to the
14 distinct Premiere nests. The later nest `3` run maps to
`SAMPLER TRACK 2 (D)` and reuses the sample chosen for nest `3` on
`SAMPLER TRACK 4`; nest `14` maps to `SAMPLER TRACK 15`. The `(D)` label's
number does not override Premiere UID identity. This extra case applies only
to an explicit range when native CPR track order, numbered tracks and `(D)`
companions exactly match first appearances and reprises. Blank selections and
ordinary one-track-per-run mappings retain their existing behavior. Before
**Import Cubase**, select tracks `2-15` **and** `SAMPLER TRACK 2 (D)`;
the generated import instructions list all required tracks. MIDI and effects
remain under the existing verified import workflow.

## Optional NVIDIA acceleration

Whisper and character embeddings default to `auto`: use CUDA when available,
otherwise use CPU and report the reason. `cpu` forces the original CPU path;
`cuda` requires working CUDA and fails explicitly instead of silently using CPU.
Set `video.speech_device`, `video.speech_compute_type`, `video.speech_device_index`,
and `video.characters.device` / `device_index` in `config.yaml`.

Character selection is automatic. For animation, one/two-eye Minion detection
runs alongside cartoon geometry and MediaPipe. Individual identities are ranked
by observed exposure across the configured source scan. Species/style tags and
co-visible lookalikes cannot merge different actors into one identity.
Cuts prefer the highest ranked eligible
actor before camera quality and sound bonuses. Recognized primary views retain
that preference when identity needs review; lower ranked actors and unidentified
views provide fallback. All duration, gap, transition and
lip requirements remain in force. Unreadable mouths still establish a face when
goggles/pupils are verified, but never establish speech. Unsupported views may be
missed. The previous `main_group: yellow_minions` preset now uses automatic cast
ranking; the GUI no longer asks the user to select the main character.
With `video.characters.embedding_model: auto`, stylised actors in live-action
sources also try the installed CCIP model when ArcFace leaves at least 45% of
tracks uncertain. CCIP is selected only if its uncertainty improves by at least
10 percentage points. Explicit `arcface` / `ccip` settings keep that model.
Automatic CCIP identity features use measured faces with narrow context, consistently
across close-ups and full-body detections. Face-verified tracks use facial observations
for their identity while retaining their full independently measured exposure.
Costume-dependent object labels cannot split a measured facial identity.
`video.characters.ccip_face_cosine_distance` controls automatic face comparison;
co-visible actors remain separate. Explicit
whole-character reference crops retain body comparison. Subject selection and cut
ranking prefer exposure rank before speech, face size and sound bonuses.
Feature tracking checks the intermediate decoded frames to bridge short
landmark misses; every step needs reversible matching image features and cuts
reset it. Tracked visibility never supplies unmeasured lips or speech.
With independent object analysis, an unmatched tracked/geometric face cannot
establish an actor. Localized posters and partially overlapping printed faces are
excluded, including a tracked box that drifts beyond the poster edge.

Whole-character recognition can additionally detect visible heads, bodies and
creatures without requiring human facial landmarks. Install the official
[Grounding DINO Tiny model](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
with `.venv-gpu/Scripts/python.exe scripts/setup_object_detector.py` (use `--cpu`
for a CPU runtime). It installs in `.venv-objects` and `models/objects`, separately
from the existing audio/ONNX environments. The model uses the Apache 2.0 license.
`video.object_detection.enabled: auto` uses it when installed; `true` requires
it and `false` retains face/pose-only analysis. Inference stays local and every
analysis sample gets independent actor and contextual detections. The actor-only
query preserves recall when more prop classes are added; contextual boxes reject
conflicting props after combining both passes. Each query has its own cache.
Brief missing observations between compatible detections can request a crop
from the full-resolution source. That frame still needs its own accepted model
box; bracketing geometry never establishes a subject. Cuts and long time gaps
disable this refinement, and contextual prop evidence still rejects false faces.
Cached boxes avoid repeated model
work. An encompassing body and its tight character region are deduplicated,
while separate foreground actors retain their own tracks. CCIP appearance
features rank the resulting subjects by measured exposure. Negative ball,
balloon, bag, vehicle, poster and paper detections help reject prop-only regions;
small props held by a creature do not erase the creature. Confirmed person/creature types
prevent similar whole-body embeddings from mixing people with monsters.
Unambiguous nested head/body boxes can continue the same physical track.
Confirmed creature appearances across shots can form an ensemble when every
view matches a fixed, most-observed appearance prototype. This does not merge
physical tracks or combine human identities using whole-body appearance.
Ranked primary views with uncertain identity remain eligible before known
secondary actors and keep their review flags. Full scenes are reserved before
the unvoiced intro, whose source gap otherwise wastes scarce primary footage.
A body/character box can qualify visual footage but never establishes a face, readable lips or
speech; measured face landmarks are retained only for one contained face.
This adds processing time on the first run and still needs visual review on
unsupported styles. It does not force any detected creature class to rank first.
`video.object_detection.prompt` can override the default queries for a different
cast; `positive_labels` can specify accepted actor words (default: person,
skeleton, monster, animal, character). Other query results never establish an
actor. Changing the prompt invalidates the detection cache; changing accepted
labels rebuilds identities from cached boxes. On the tested
109.58-second source, a CUDA query scan took about five minutes. The default uses
two scans on the first run; repeated plans reuse both. The detector's precision
setting is independent of Whisper precision; auto uses FP16 on CUDA and FP32 on CPU.
Auto precision is `int8_float16` on CUDA to limit VRAM use, and `int8` on CPU.
Character embeddings retain float32 math with TF32 disabled; CUDA convolution
workspace is limited to reduce memory pressure on the 4 GB laptop GPU.
MediaPipe face/pose detection and video decoding remain on their existing CPU paths.
Changing devices alone does not improve recognition accuracy.

On Windows, run `.venv/Scripts/python.exe scripts/setup_gpu.py` to install the
GPU requirements in the active virtual environment and replace CPU ONNX Runtime.
Restart the tool afterward. Do not install `requirements.txt` and
`requirements-gpu.txt` together: `onnxruntime` and `onnxruntime-gpu` share files.
Use the setup script rather than installing the GPU requirements directly:
faster-whisper's metadata requires the CPU distribution. The script removes it
after installing dependencies, then reinstalls GPU without dependency resolution.
This satisfies inference/VAD imports, although `pip check` may still report the
upstream CPU distribution requirement. Rerun setup after installing/upgrading
faster-whisper or CPU requirements, which can otherwise overwrite GPU files.
The GPU extra pins CUDA 12.4/cuDNN 9 and ONNX Runtime 1.20.0 for the existing
RTX 3050 driver; it does not update the driver or system environment variables.
The Windows loader discovers NVIDIA DLLs from this environment's pip packages.
On Linux, configure the CUDA loader search path as described by faster-whisper.
See the [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#gpu)
and [ONNX CUDA provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

An unavailable CUDA runtime or GPU memory error in auto mode retries the
affected audio chunk/character crop on CPU. A partial Whisper chunk is discarded
before retrying, avoiding duplicate words. Invalid models/inputs still raise errors.
Logs and output JSON record the actual device, precision/providers and fallback.
Caches distinguish explicit device/precision settings. Auto can reuse an existing
cache made on another device; cache hits perform no new model inference.

For a repeatable Whisper comparison, run:
`python scripts/benchmark_inference.py path/to/english.wav --seconds 60`.
It repeats/truncates the same audio, warms up both devices and reports timings
separately from startup, plus whether the final word sequences agree.
This measures Whisper throughput only, not full video processing or accuracy.
The optional hardware test
`tests/test_acceleration.py::test_real_cuda_character_backend_preserves_vectors_on_synthetic_model`
requires the `onnx` package and `AUTOEDIT_GPU_TESTS=1`. Run that test separately;
it executes a synthetic convolution model on real CUDA and compares CPU results,
without requiring character weights. It does not evaluate character accuracy.
