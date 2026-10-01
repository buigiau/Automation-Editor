# Automation-Editor

## Sound-led scene selection

The pipeline scans the original soundtrack across the analyzed part of the
source video before choosing cuts. Confident words that match the shared Voice
library supply candidate cut onsets. Each cut keeps the Premiere slot duration,
person/face requirements, transition checks, character filters and source gaps.
Qualifying sound onsets take priority; visual selection fills remaining slots.
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
250 ms of the cut. Mid-word cuts and unsupported openings use the existing
visual/neutral fallback. Each scene has one WAV, including later reprises.

Source analysis uses Whisper word timestamps and a bundled CMU dictionary;
dictionary matching does not verify the actual waveform or accent. Existing
spectral detectors suggest cough, sigh and laugh events outside transcribed
speech. Event and pronunciation matches are marked for listening review.
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

## Optional NVIDIA acceleration

Whisper and character embeddings default to `auto`: use CUDA when available,
otherwise use CPU and report the reason. `cpu` forces the original CPU path;
`cuda` requires working CUDA and fails explicitly instead of silently using CPU.
Set `video.speech_device`, `video.speech_compute_type`, `video.speech_device_index`,
and `video.characters.device` / `device_index` in `config.yaml`.
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
