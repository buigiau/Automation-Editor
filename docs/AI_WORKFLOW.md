# Preset + AI autofill

Open `AutoEdit-CPU.cmd` or `AutoEdit-GPU.cmd`. Set up the three
project pairs, Voice folder and API key once under **Thiết lập**. Select
**Soda Pop / Golden / Cosita**, output folder and one or more source videos,
optionally add character images, then **Chạy**. Each image has a thumbnail and
remove button. Removing all images restores automatic character selection.
Images belong to this session; they are not restored from settings or legacy YAML.

## Qwen local (no API key)

Open `AutoEdit-Local.cmd`, or select **Thiết lập → Cài / kiểm tra Qwen local**.
The setup downloads the official portable Windows NVIDIA Ollama runtime and
the saved local model (new default: `qwen3-vl:2b-instruct`). Downloads, runtime, model and server logs
stay in `output/local-ai`; allow several GB of disk space. Installation requires
internet; subsequent inference runs on the machine. Setup saves `backend: local`
without deleting the Gemini key or its model setting. You can switch back under
**Thiết lập → AI backend**. A new launch/worker reads saved settings.

For footage with people and no character references, `until_filled` first uses
a CPU person detector and face landmarks/embeddings. Setup also installs the
35.9 MB [OpenCV Zoo YOLOX detector](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox)
with its Apache-2.0 license and verifies the pinned SHA-256. It uses the existing
`models/face_landmarker.task` and `models/characters/arcface.onnx`; missing optional
models fall back to Qwen. To install just person detection, run
`python scripts/setup_fast_people.py`.

Preview screening uses 320-pixel model input; selected cut verification uses
640-pixel input at 15%, 50%, and 85%. A solo cut must have one detected person
and no detected neighbour in all three samples. Person counts include detected
animals and small bystanders. Aligned face embeddings identify a person only
when at least two readable samples agree. Masks, profiles, small or hidden faces
retain unique `person-unresolved-*` cut IDs with `identity_sample_consistent=false`;
they are never merged into a recognized cast identity. These cuts can still be
useful solo footage. The review page explicitly flags unresolved identities.
Counts and identities are sampled evidence, not a guarantee for every frame.
CPU detection failures are inconclusive: unknown/zero-human previews are retained
for Qwen, and failed selected-cut person checks use Qwen directly. One spurious
human detection cannot turn an animated source into human-only footage.

Solo candidates are allocated before group scenes. If fast discovery is short,
it retries discovery at native cadence before enabling group-scene fallback.
Group use is recorded as `group_scene_fallback` and displayed on the review page.
The initial discovery pass skips non-reference decoder frames, uses six-second
regions, and prefers clear solo previews without first indexing the full video.
Approximate shot bounds remain marked partial. Allocation and final selected-cut
audits still examine native frames for transitions and duplicated footage.
Eight duplicate-check samples now share one seek and sequential decode.

On suitable human footage this path makes no Qwen generation calls and does not
load its large vision runner. The configured Qwen model remains the fallback
for other footage or character references. Runtime evidence records separate
candidate/model counts and CPU person statistics in `analysis_scope`.
Voice matching remains an estimate requiring playback and lip-sync review.

Use `scripts/benchmark_ai.py ... --phases first --cold` for fresh-output timing.
`--cold` requires an empty application cache and refuses to reuse or delete an
existing cache. It never includes a cached rebuild in a fresh benchmark. Model
installation/download time is separate; operating-system file caching is not
cleared.

The Ollama catalog server binds only `127.0.0.1:11435`, starts automatically for local
jobs and disables Ollama cloud. Existing manually started loopback Ollama servers
can be configured under `local_url`. Cloud models and remote endpoints are
rejected. Fast `until_filled` uses the workspace-managed installation: a private
llama.cpp child loads the same installed weights on an ephemeral loopback port.
It encodes one image at a time with 64–256 image tokens instead of Ollama's
1024-token minimum. Its image encoder stays on CPU, inference batches are 32
tokens, the context starts at 1024 and grows only for larger reference requests,
and the default 8 GiB prompt cache is disabled. It terminates the owned process
on completion/error/cancel. Custom Ollama endpoints remain available in `full`
mode. Missing runtime/model errors direct you to setup; the job never
silently changes to a cloud provider. No extra Python inference package is needed.

Local now defaults to **CPU with 2 threads**, with no GPU layer offload. GPU
offload is an explicit choice under **Thiết lập → Tài nguyên local**; its default
limit is 8 layers, configurable under **Nâng cao**. This limits offloaded layers,
not GPU utilization, so GPU mode can still interfere with editing/display work.
The `full` mode uses 128-token inference batches. Managed runners run at below-normal
priority. Idle model retention is 30 seconds and normal completion/error/cancel
paths explicitly unload the model. A forcibly terminated worker cannot execute
cleanup; the idle timeout is the fallback.

Before bounded-image CPU inference, the app requires **6 GB free system RAM** for
the 4B model, or **4.5 GB** for the exact `qwen3-vl:2b-instruct` tag. The older
`full` workflow requires 8/5 GB respectively. These are admission checks, not a
guarantee against other applications allocating more memory. As a
conservative resource guard. It reports an actionable error before loading or
scanning when RAM is insufficient in the older full workflow. The fast workflow
waits for enough free RAM, with progress logs and cancellation. While local CPU inference is running, a
socket watchdog interrupts the request if free RAM falls below 1.5 GB. It also
honors cancellation and deadlines during image prefill, before the first token.
After memory pressure the fast workflow releases its owned runner, retains
completed observations and waits for at least 6 GiB free RAM across three samples
before retrying the pending image. Ordinary inference errors are not retried
indefinitely. An explicit hard runtime limit still applies while waiting.
CPU Qwen is substantially slower; do not
assume the GPU measurements apply. The October 15 full-source GPU trial was
stopped after it interfered with desktop responsiveness. The final CPU profile
has automated coverage. The measured 4B CPU request took 182.48 seconds for one
12-frame batch. The October 16 2B CPU trial of the original 28-image strategy
stopped at 270.758 seconds with only seven structurally accepted fills and
visibly incorrect character labels; it did not meet the five-minute target.

Local defaults to **`until_filled`**: seek to distributed regions of each source,
discover shot windows, and sample at `video_fps`. The fast CPU pass uses coarse
discovery; each proposed cut still passes native-frame transition checks. Distant
regions are visited early; subsequent passes fill the remaining gaps. Additional
windows from visited regions are retained for shortage fallback. Multiple sources
are interleaved. Durations, source gaps, transitions and duplicate frames are
screened before vision inference. Qwen sees each cut frame separately, never a
multi-scene collage. Two frames at 20% and 80% of the proposed cut must both show
a readable character and agree on species and main body color. A color palette
avoids rejecting tan/brown variations of one animal. Logos, text, scenery,
tiny/occluded actors and inconsistent pairs are rejected. A negative first frame
skips the second request. Generic or uncertain identities
are not usable character evidence. This strategy does not impose a three-minute
prefix limit, and does not treat the largest visited timestamp as full coverage.
Allocate the longest template fills first and stop source discovery immediately
when every slot passes duration, gap, transition and selected-frame duplicate
checks. There is no fixed first-three-minute restriction. If short of distinct
shots, the enabled shortage fallback tries different content within the same
shot; if still short, search expands. No source is decoded completely up front.
Long continuous takes also yield rolling candidate windows before the take ends;
their stable shot ID protects strict shot uniqueness and their bounds describe
the observed prefix (`shot_bounds_complete: false`), not an unseen camera cut.
Shots already in progress at different seek positions share an unknown shot ID
until an actual cut is observed; distinct segments remain available through the
explicit shortage fallback.
Search regions overlap by the longest requested fill plus safety margins, so a
long shot crossing a seek-region edge remains usable. Once short fills are
satisfied, person/face previews skip shots too short for the remaining slots.
With character references, unrelated actors cannot end discovery prematurely:
character substitution is considered after available sources are exhausted.

For human reference images, local `until_filled` searches clothing/costumes,
accessories and hair styling, without facial identity matching. Face regions in
both the reference and sampled video frames are masked before model requests;
source images/video remain intact. CPU wardrobe colours screen and rank candidates,
then one compact model request checks both cut samples independently. Both samples
must agree on the same reference look. A colour alone cannot establish a match.
The search immediately uses promising windows and then visits neighbouring regions
around recent matches. A requested outfit can appear in a group scene. These rows
carry `reference_match_scope=costume_appearance`, two appearance samples, and
unverified facial identity. Unrelated people never substitute for a missing outfit
in this mode, including at the deadline or source exhaustion. A different outfit
on the same person requires another reference; this is appearance search.
Reference discovery prioritises the longest continuous fill for at most 75 seconds
(or one quarter of the target duration), then reuses cached
regions for shorter fills. The final-block exception needs only opening evidence,
so it cannot consume the only continuous take needed by an earlier long block.

For nested fills, preflight also records the largest source-out actually used by
the parent scene instances as `visible_duration_sec`. This can be much shorter
than the nested overwrite extent: Cosita slot-03 needs a 5.68-second overwrite
but plays only its first 0.56 seconds. Character, transition, duplicate, source-gap
and review checks use that displayed prefix. The full fill length and original
source bounds remain enforced. The unused nested tail is never presented as
character evidence. The final-block exception still reserves its full source
interval and follows its separate opening-only evidence rule.

`local_target_seconds` defaults to 300 and is a desired completion time. It does
not stop a search that can still find footage. After the target, the progress log
says it is continuing. Solo scenes are preferred; at the target, deferred group
scenes can fill remaining slots without waiting for another whole-source pass.
These selections retain their group-fallback flags, character checks, native-frame
transition checks, source gaps and duplicate checks. Search stops when filled, sources are exhausted, Cancel
is requested, or an optional explicit `local_max_seconds` limit is reached.
The latter defaults to 0 (no hard maximum); when set, 30 seconds are reserved for
validation/export. Each individual model request remains bounded. Deterministic
allocation can use at most ten seconds of that reserve to retain a character
observation completed at the discovery deadline; model/discovery deadlines are
not extended, and twenty seconds remain for export. Advanced
settings and CLI expose the target and hard maximum independently:
`--local-analysis-mode until_filled --local-target-seconds 300 --local-max-seconds 0`.
These settings do not promise five-minute completion. By default,
`ai.shortage_policy: complete` preserves validated fills and completes missing
slots with decoded footage for a full, explicitly labelled draft. `strict`
retains the previous failure and partial review behavior.

RAM waits count toward an explicitly configured hard maximum. For example,
`local_max_seconds: 570` stops discovery at roughly 540 seconds to reserve export
time, even if the source still contains unsearched footage. Complete mode then
fills the missing slots without more model calls; strict mode reports the limit
and explains how to resume with a larger limit or zero. The
five-minute target remains independent. Reference search visits at most three
nearby regions before returning to the distributed region order, so matching
outfits in one block cannot keep distant footage unvisited indefinitely.

`ai.transition_tail_fallback_sec` defaults to 0. Setting it to 1 or 2 explicitly
allows scene/character changes in that many final seconds when a full-duration
cut cannot pass the usual checks. Discovery can extend a shot into its following
footage for this fallback. Native-frame transition checks still protect the
opening, and identity/count samples cover only that stable prefix. The full cut
retains its requested duration, source bounds, source gaps and duplicate checks.
The review page and selected row carry a tail-fallback flag; character exposure
does not include the unverified tail. The GUI exposes the setting under Advanced.

`ai.final_scene_transition_fallback: true` permits scene changes anywhere in the
final timeline block, with no tail-duration limit. The final scene is determined
from its timeline end, not a slot number or the longest duration. Earlier blocks
keep their normal checks. A final nest that is also used in an earlier scene is
not exempted, since replacing that nest would also change earlier blocks.
The final fill retains its full requested duration, available-source bounds,
source gaps and duplicate checks. Its character/count evidence describes only
the sampled opening; the rest can contain other shots or characters. Review and
export metadata mark this exception explicitly. With this policy, discovery
does not spend time searching for a continuous shot as long as the final fill.

Qwen character appearance/visibility in this mode is checked at two cut frames,
not certified across every frame; the plan records `local_character_verified:
false` and `analysis_scope` with searched regions and classified candidate counts.
The mouth score describes a still-image expression, not movement. It must not
be represented as verified talking or lip sync. Species/color agreement is not
face recognition and cannot establish unique identity among similar actors.
Appearance rankings
cover classified candidate windows, not the whole video. Short appearances can
still be missed. Candidate labels cache the actual source/window identities,
references and model digest; final validated selections are cached as before.

The optional **`full`** mode retains the earlier exhaustive workflow: every
configured source window is decoded and Qwen observes all sampled frame batches
(`local_batch_frames`, default 12). It also checks chosen character frames against
actual character images. Use it only when full analysis matters more than speed.
It is not appropriate for the five-minute target on CPU.

Qwen is used for **visual** recognition, reference matching, visibility and mouth
expression estimates. It does **not** listen to the source audio. All local
Voice matches use catalog metadata and remain `audio_uncertain: true` with
unverified lip sync. Listen in the review page before applying the result.
Duration, source-gap, distinct-shot/fallback and visual-duplicate validation,
Premiere/Cubase artifact export and manual corrections use the shared pipeline.
Candidates are audited for transitions at native cadence. These checks reduce
mistakes; they do not establish frame-perfect visibility or lip sync.
Plans record `qwen-local-visual`, the installed model's
weight digest and the visual-only Voice uncertainty.

CLI: `autoedit run --preset golden --video VIDEO --ai-backend local`.
Benchmark: `python scripts/benchmark_ai.py --preset golden --video VIDEO
--root output/qwen-benchmark --backend local --video-fps 3 --max-source-seconds 0`.
The first run scans and infers; the second output reuses the validated result.
To install the optional lighter CPU model without changing the saved model:
`python scripts/setup_local_ai.py --model qwen3-vl:2b-instruct`.
Add `--activate` to select it, or benchmark with `--local-model qwen3-vl:2b-instruct`.

Settings are in `autoedit.settings.json` in the project root (ignored by Git).
The API key is stored locally in that file; alternatively leave it empty and set
`GEMINI_API_KEY`. Worker configs travel over stdin. Keys are excluded from logs,
review snapshots, plans and caches. **Nâng cao** contains model, sequence, zero-based
video-track index, sampler selection, source gap and shared cache directory.

The model defaults to `gemini-3.8-flash`. The REST integration follows Google's
[video understanding](https://ai.google.dev/gemini-api/docs/video-understanding),
[Files API](https://ai.google.dev/gemini-api/docs/files), and
[structured output](https://ai.google.dev/gemini-api/docs/structured-output) contracts.
Original videos **with audio** and reference images are uploaded; the Voice
catalog supplies filenames, durations and phonetic metadata. Voice WAVs are not
uploaded. An API failure is reported directly and never starts the local scanner.

## Selection and validation

Template inspection runs before an API request. Unique nested sequence fills,
their source-duration requirements, timeline instances and scene runs come from
the `.prproj`; sampler names and indices come from the `.cpr`. Mismatched counts,
invalid offsets and unsupported retiming stop before upload. Local template
checks found Soda Pop with 14 unique fills / 16 scene runs, Golden 14 / 14,
and Cosita 14 / 15; these are observations, not hardcoded counts.

With a blank AI sampler selection, native `(D)` companions require a unique
mapping between CPR track order and Premiere UID reprises. For Cosita this
selects numbered tracks `2-15` plus `SAMPLER TRACK 2 (D)`, leaves track 1 out,
and shares nest 3's sample on track 4 with its `(D)` reprise. An explicit `2-15`
selection supports the same mapping. Unknown or ambiguous companion layouts
stop before analysis. Import errors list missing and unexpected track names.

Gemini receives explicit instructions to recognize humans, creatures, trees,
vehicles and personified objects, merge alternate reference views, prefer
reference characters and use appearance-duration-ranked fallback if needed.
Appearance intervals are validated and unioned per source to produce an
**estimated** duration ranking. Old human-face/prop filters and full-source
Whisper are not called in this branch.

Each chosen segment must have the exact slot duration, fit its source and full
continuous-shot interval, reference a catalog Voice and evidenced character,
and respect the configured gap. The first selection uses unique continuous shots.
If slots remain empty, `ai.shot_fallback: true` authorizes the single repair request
to select different non-overlapping segments within a used shot. Shot IDs and
long-footage duplicate groups can then repeat, but overlapping cuts and repeated
selected visual content remain blocked. The review marks `same_shot_fallback`.
Eight frames per selected segment are
compared using perceptual hashes and color differences to flag repeated footage
across files or timestamps. This is a conservative spot check, not proof that
arbitrary crops, speed changes or every possible duplicate will be detected.
AI shot boundaries and character identity still need human review.

**Nâng cao** now exposes Gemini video FPS, the analyzed prefix duration and the
shortage fallback checkbox. `ai.video_fps` defaults to 1; use 3 or 6 to test denser
evidence for short animated clips. `ai.max_source_seconds: 180` limits each source
to its first three minutes, while 0 analyzes the full source. Original source
timestamps and duration remain unchanged; selections outside the analyzed window
are rejected. FPS, window and fallback settings are part of the AI cache key.
CLI equivalents are `--ai-video-fps 3 --ai-max-source-seconds 180`.

The tool requests at most one correction from Gemini. If valid distinct shots
remain insufficient, complete mode exports all template fills, Premiere payload
and Cubase sampler WAVs. Draft fills use `character_id: unverified`, never invented
character appearances. The review and plan disclose `completion_fallback` and
`fallback_relaxations`; matching identity and Voice still require playback review.
It first tries distinct, continuous footage with the configured source gap, then
allows transitions, smaller gaps, visual duplicates and finally temporal overlap
only as needed. A source shorter than the template fill gets a separately rendered,
labelled held-frame asset. Exact duration, source bounds and decoding remain mandatory.
`ai-validation.json` distinguishes `complete` (usable full bundle) from
`selection_complete` (all fills passed the original AI requirements).
Cancellation and invalid templates/media remain errors, and drafts are not cached
as verified AI selections. No backend switches to a cloud provider automatically.
`ai.shortage_policy: strict` keeps partial review artifacts without publishing
an incomplete plan. Previous root plans/manifests are moved to `previous_runs` so a failed
new run cannot accidentally import an old result. That directory is an audit
archive; use the current output's review page and plan for playback/import.

Recover an existing failed run without repeating inference:
`python scripts/recover_ai_draft.py "D:\\golden\\201026\\ai-review.json" --output "D:\\golden\\201026"`.
Existing observations and all input identities are checked before recovery.

## Review and corrections

Open **Xem / nghe review**. Each unique fill includes its original video/sound,
selected Voice, reason and fallback/uncertainty flags. Play original sound or
video with the replacement Voice. Native browser codec support applies; for
unsupported source formats use a local video player at the displayed timestamps.
The replacement playback is an estimate, not a render of Cubase MIDI/FX.

Adjust a cut's start within its existing shot or select another Voice. Download
the corrections JSON and use **Áp dụng chỉnh sửa**. The worker verifies the
review revision and source/template/Voice fingerprints, revalidates all cuts and
duplicate checks, and rebuilds the bundle **without calling Gemini**. Invalid
corrections leave the current bundle intact and produce
`ai-correction-errors.json`. A new shot or different character requires a new run.

Successful output includes `edit-plan.json`, `premiere_apply.json`, sampler WAVs,
`sampler_mapping.csv`, `cubase_import.json`, a timing preview and the review page.
Template repetitions reuse their nested fill and Voice but map to the existing
scene-run sampler tracks. Premiere fills use cached silent video streams and keep
template audio. Cubase samples retain the established importer and preserve MIDI/FX.
Open the Premiere panel and **Apply**, then manually select the expected Cubase
tracks and **Import Cubase**. A completed AI job prepares artifacts; it does not
open either editor, import samples or export a final mix automatically.

## Cache, cancellation and measurement

Voice metadata, uploaded handles, validated AI results, selected-frame checks and
silent video assets live in an output-independent cache, by default
`output/.ai-cache`. AI cache identities include source content, paired templates,
slot requirements, preset, reference content, Voice content, gap, model and prompt
version. A new output folder reuses cached work. Remote handles are checked for
expiry and scoped by a credential hash. File uploads are chunked. API waits and
processing polls are bounded.

The existing cross-process lock admits one job at a time; the GUI uses a fresh
worker with two compute threads and cancellation. Video decoders and OpenCV use
two threads. Completed caches survive cancellation. `performance.json` records
total processing, queue wait, upload, AI and Voice stages. Correction rebuilds
also get their own run timings. Time manual review and Cubase import separately;
they occur after artifact generation and cannot be inferred from worker time.

Tests use fake Gemini responses, real generated WAV/video and optional local
Premiere/Cubase templates. These validate control flow and artifacts, not Gemini
recognition quality or Cubase playback. Benchmark the first run and cache reuse
with actual footage, include upload + review/corrections + import, and compare
with the 5-minute manual baseline before making speed claims.

```powershell
autoedit run --preset soda_pop --video source-a.mp4 --video source-b.mp4 -o output/job1
autoedit run --preset cosita --video source.mp4 --character-reference tree.png -o output/job2
autoedit ai-review autoedit-corrections.json -o output/job2
autoedit gui --legacy
```

CLI `run` without `--preset` or `--ai` retains the configured local pipeline.
`--ai` enables Gemini using explicit YAML paths, with `GEMINI_API_KEY` available
as a fallback. `--settings` selects a different preset registry file.
