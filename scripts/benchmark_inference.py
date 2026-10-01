"""Compare CPU/CUDA Whisper with identical audio, model and decoding settings."""
import argparse
import gc
import json
import re
from pathlib import Path
import time

import av
import numpy as np

from autoedit.audio.source_speech import _model, _MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="small")
    parser.add_argument("--seconds", type=float, default=60,
                        help="Repeat/truncate test audio to this duration (benchmark only)")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("output/performance-probe/gpu-benchmark.json"))
    args = parser.parse_args()
    if args.seconds <= 0 or args.repeats < 1:
        parser.error("seconds and repeats must be positive")
    with av.open(str(args.audio)) as container:
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        frames = [converted.to_ndarray().reshape(-1) for frame in container.decode(audio=0)
                  for converted in resampler.resample(frame)]
        frames.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(None))
    audio = np.concatenate(frames)
    if not len(audio):
        raise ValueError("Empty audio")
    length = round(args.seconds * 16000)
    audio = np.tile(audio, int(np.ceil(length / len(audio))))[:length]
    results = []
    for device in ("cpu", "cuda"):
        started = time.perf_counter()
        runtime = _model(args.model, device=device, progress=lambda m: print(m, flush=True))
        load = time.perf_counter() - started
        settings = dict(language="en", beam_size=5, word_timestamps=True,
                        vad_filter=True, condition_on_previous_text=False, temperature=0.0)
        # Warmup excluded from timings; model startup is measured separately.
        list(runtime.transcribe(audio[:min(len(audio), 16000*5)], **settings)[0])
        times = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            segments = list(runtime.transcribe(audio, **settings)[0])
            times.append(round(time.perf_counter() - started, 3))
        words = [{"word": word.word.strip(), "start": round(word.start, 4),
                  "end": round(word.end, 4), "prob": word.probability}
                 for segment in segments for word in segment.words or []]
        results.append({"execution": runtime.execution, "load_seconds": round(load, 3),
                        "inference_seconds": times, "mean_seconds": round(float(np.mean(times)), 3),
                        "words": words})
        print(json.dumps({k: v for k, v in results[-1].items() if k != "words"}), flush=True)
        _MODELS.clear()
        del runtime
        gc.collect()
    cpu, gpu = results
    same_words = [w["word"] for w in cpu["words"]] == [w["word"] for w in gpu["words"]]
    normalize = lambda words: [re.sub(r"^\W+|\W+$", "", word["word"].casefold()) for word in words]
    same_normalized_words = normalize(cpu["words"]) == normalize(gpu["words"])
    timing_difference = (max(abs(a[key] - b[key]) for a, b in zip(cpu["words"], gpu["words"])
                             for key in ("start", "end"))
                         if same_normalized_words and cpu["words"] else None)
    report = {"model": args.model, "source_audio": str(args.audio.resolve()),
              "test_audio_seconds": args.seconds, "audio_repeated": True,
              "speedup": round(cpu["mean_seconds"] / gpu["mean_seconds"], 2),
              "identical_word_sequence": same_words,
              "identical_words_ignoring_case_and_edge_punctuation": same_normalized_words,
              "max_word_timestamp_difference_seconds": timing_difference, "results": results,
              "scope": "Whisper only; repeated test audio; does not measure full video pipeline or accuracy"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Whisper speedup: {report['speedup']}x; same normalized words: {same_normalized_words}; report: {args.output}")


if __name__ == "__main__":
    main()
