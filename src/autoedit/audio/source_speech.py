"""Whisper word timestamps from source-video ranges, including a full scan.

Decode the original soundtrack with PyAV, preserving its offset relative to
video time. Bounded overlapping context supports full-video scans. Word and
event timestamps drive cut-onset candidates; cache identity includes source
stat, ranges, detector version and ASR settings.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from autoedit.audio.detail import _is_hallucination

ASR_VERSION = 2
_MODELS = {}


def _model(name):
    if name not in _MODELS:
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        from huggingface_hub.errors import LocalEntryNotFoundError
        cache = str(Path(__file__).resolve().parents[3] / "models" / "whisper")
        resolved = name if Path(name).is_dir() else None
        if resolved is None:
            for root in (None, cache):
                try:
                    resolved = download_model(name, cache_dir=root, local_files_only=True)
                    break
                except LocalEntryNotFoundError:
                    pass
            if resolved is None:
                resolved = download_model(name, cache_dir=cache)
        _MODELS[name] = WhisperModel(resolved, device="cpu", compute_type="int8")
    return _MODELS[name]


def _windows(ranges, padding=1.0):
    windows = []
    for start, end in sorted(set(ranges)):
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError("Invalid source transcription range")
        start, end = max(0.0, start-padding), end+padding
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], end)
        else:
            windows.append([start, end])
    return windows


def _decode_window(container, stream, origin, start, end):
    import av
    rate = 16000
    audio = np.zeros(int(math.ceil((end-start)*rate)), dtype=np.float32)
    container.seek(int((start+origin) / float(stream.time_base)), stream=stream, backward=True)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)

    def copy(frame):
        if frame.time is None:
            raise ValueError("Source audio has no timestamps; cannot align it with video.")
        offset = int(round((float(frame.time)-origin-start)*rate))
        values = frame.to_ndarray().reshape(-1)
        lo, hi = max(0, offset), min(len(audio), offset+len(values))
        if hi > lo:
            audio[lo:hi] = values[lo-offset:hi-offset]

    for frame in container.decode(stream):
        if frame.time is not None and float(frame.time)-origin >= end:
            break
        for converted in resampler.resample(frame):
            copy(converted)
    for converted in resampler.resample(None):
        copy(converted)
    return audio


def _chunks(windows, size=60.0, context=1.0):
    """Bound memory for long videos; overlapping context has one owner per word."""
    for lo, hi in windows:
        if hi-lo <= size:
            yield lo, hi, lo, hi
            continue
        core = lo
        while core < hi:
            edge = min(core+size-2*context, hi)
            yield max(lo, core-context), min(hi, edge+context), core, edge
            core = edge


def transcribe_source(path, ranges, cache_dir, model="small", language="en", progress=None):
    """Return absolute source word timestamps; silent sources have no words.

    Missing/broken ASR is an explicit error, never a successful silent fallback.
    Set language=None to auto-detect for non-English source videos.
    """
    import av
    src = Path(path).resolve()
    stat = src.stat()
    windows = _windows(ranges)
    identity = {"path": str(src), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "windows": windows, "model": model, "language": language, "version": ASR_VERSION}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    dest = Path(cache_dir) / ("speech-" + key + ".json")
    if dest.is_file():
        try:
            cached = json.loads(dest.read_text(encoding="utf-8"))
            if cached.get("cache_identity") == identity:
                if progress:
                    progress("Source speech cache hit: reusing Whisper word timestamps.")
                return cached
        except (ValueError, OSError):
            pass
    result = {"has_audio": False, "words": [], "events": [], "method": "whisper-source-audio",
              "model": model, "language": language, "cache_identity": identity}
    with av.open(str(src)) as container:
        if container.streams.audio:
            result["has_audio"] = True
            stream = container.streams.audio[0]
            video = container.streams.video[0] if container.streams.video else stream
            origin = float(video.start_time * video.time_base) if video.start_time is not None else 0.0
            chunks = list(_chunks(windows))
            from autoedit.audio.source_events import detect_vocal_events
            for index, (start, end, core_start, core_end) in enumerate(chunks, 1):
                if progress:
                    progress(f"Whisper source audio {index}/{len(chunks)}: {start:.2f}-{end:.2f}s")
                samples = _decode_window(container, stream, origin, start, end)
                if not len(samples) or float(np.max(np.abs(samples))) < 1e-4:
                    continue
                segments, _ = _model(model).transcribe(
                    samples, language=language, beam_size=5, word_timestamps=True,
                    vad_filter=True, condition_on_previous_text=False, temperature=0.0)
                chunk_words = []
                for segment in segments:
                    if segment.no_speech_prob > 0.6 or _is_hallucination(segment.text):
                        continue
                    for word in segment.words or []:
                        a, b, prob = float(word.start), float(word.end), float(word.probability)
                        if not all(math.isfinite(v) for v in (a, b, prob)) or not 0 <= a < b <= end-start+0.05:
                            continue
                        chunk_words.append({"word": word.word.strip(), "start": round(start+a, 4),
                                            "end": round(start+b, 4), "prob": prob})
                result["words"].extend(w for w in chunk_words
                                       if core_start <= (w["start"]+w["end"])/2 < core_end)
                result["events"].extend(e for e in detect_vocal_events(samples, 16000, chunk_words, start)
                                        if core_start <= (e["start"]+e["end"])/2 < core_end)
    result["words"].sort(key=lambda word: word["start"])
    result["events"].sort(key=lambda event: event["start"])
    result["status"] = ("transcribed" if result["words"] else "vocal-events" if result["events"]
                        else "no-speech" if result["has_audio"] else "no-audio")
    if progress:
        progress(f"Source speech: {result['status']}, {len(result['words'])} words, {len(result['events'])} vocal-event hints")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temp.replace(dest)
    return result
