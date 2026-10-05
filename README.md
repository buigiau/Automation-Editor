# Automation-Editor

## Optional character references and articulation checks

In the GUI, use **Target characters (optional) > Import images** to choose
portraits of different characters, alternate views, or a thumbnail with multiple
characters; **Clear** restores automatic exposure ranking.
The CLI accepts repeated `--character-reference actor-a.png --character-reference group.jpg`,
or set `video.characters.reference_images` in YAML. Human references need a
readable face. Each detected face in a group image gets its own identity template;
strongly matching alternate views across images are merged conservatively, but
co-visible faces remain separate. Unreadable detected regions are reported;
an image with no usable identity fails validation. A cartoon image with no
detected faces is treated as one cropped character, so use individual crops
if the group detector misses faces. References and video use the same identity model and preprocessing.
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

The default `video.analysis_mode: until_filled` analyzes the beginning of each
source in 30-second increments, at the existing sample rate. After each increment
it tries to fill every unique Premiere slot and any empty intro, then verifies
the selected mouths and prepares the scene/sample mapping. It stops source
analysis as soon as a complete allocation passes those checks. Repeated instances
of a nested sequence do not require additional source cuts.

Cast exposure is ranked within the footage analyzed so far. It does not claim
to identify the most frequent character in the unseen remainder. In the GUI,
choose **Stop when all slots are verified** or **Analyze all sources**. YAML can
set `video.analysis_mode: full` for whole-source ranking; `video.analysis_chunk_sec`
controls the interval (default 30). CLI accepts `--analysis-mode full` and
`--analysis-chunk-sec 30` on `autoedit run`.

The scanner retains face/pose/tracking and transition state between increments.
One second of measured context protects their boundaries. Failed dense mouth
checks exclude those cuts and continue searching. The slot lengths, source gaps,
person/prop checks, transition rules and strict speaking-mouth policy still apply.
If all selected sources or `video.max_seconds` are exhausted, the run fails rather
than publishing an incomplete plan. Existing plans remain in place on failure;
check `run.log` for the latest run status before applying a previous plan.

Completed analysis checkpoints, per-frame object detections, per-crop embeddings,
audio windows and individual dense cut checks are cached. A resumed analysis
warms detector/tracker state over the last second, then processes new frames.
The isolated object worker reuses weights and decoded frames between actor and
context queries. It selects CPU/CUDA and a batch size using available VRAM,
reduces the batch on memory errors and reports the actual runtime. CUDA weights
are offloaded between object scans so character/Whisper models can use the GPU.
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

The default selects clear character shots (`video.require_lip_motion: false`),
including closeups and independently confirmed body shots. Unreadable mouths
do not discard otherwise eligible footage; uncertain audio matches need review.
The GUI's **Require speaking mouths** checkbox enables strict lip selection.
In that mode the opening must show speech and mouth shapes must change within
the selected cut, above pixel noise; a static open mouth or camera pan cannot qualify.
Animation analysis adds paired eyes/pupils, shared face colour and a mouth detector
for flat coloured cartoons, alongside MediaPipe. Unsupported styles still require
landmarks; scenery and character embeddings alone never establish a face. Both
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

## Optional NVIDIA acceleration

Whisper and character embeddings default to `auto`: use CUDA when available,
otherwise use CPU and report the reason. `cpu` forces the original CPU path;
`cuda` requires working CUDA and fails explicitly instead of silently using CPU.
Set `video.speech_device`, `video.speech_compute_type`, `video.speech_device_index`,
and `video.characters.device` / `device_index` in `config.yaml`.

Character selection is automatic. For animation, one/two-eye Minion detection
runs alongside cartoon geometry and MediaPipe. Verified yellow Minions form one
ensemble with separate face tracks; concurrent members count once toward its
observed screen time. The ensemble and other identities are ranked together by
exposure, without forcing Minions first. Cuts prefer the highest ranked eligible
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
Strongly matching CCIP appearances repeatedly seen as separate faces in the
same frames can form an ensemble. Their physical tracks stay separate and
simultaneous copies count once toward exposure. Subject selection and cut
ranking prefer exposure rank before speech, face size and sound bonuses.
Feature tracking checks the intermediate decoded frames to bridge short
landmark misses; every step needs reversible matching image features and cuts
reset it. Tracked visibility never supplies unmeasured lips or speech.

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
two scans on the first run; repeated plans reuse both. The object model uses
float32 independently of Whisper precision.
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
