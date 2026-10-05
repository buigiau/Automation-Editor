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

from autoedit.acceleration import device_settings, is_accelerator_error, prepare_nvidia_runtime
from autoedit.audio.detail import _is_hallucination

ASR_VERSION = 3
_MODELS = {}


class _WhisperRuntime:
    def __init__(self, path, device, compute_type, device_index, progress=None):
        prepare_nvidia_runtime()
        from faster_whisper import WhisperModel
        import ctranslate2
        self.path = path
        self.requested_device = device
        self.requested_compute_type = compute_type
        self.device_index = device_index
        self.progress = progress
        self.fallback_reason = None
        self.factory = WhisperModel
        self.device = device
        if device != "cpu":
            try:
                available = ctranslate2.get_cuda_device_count() > device_index
            except RuntimeError as exc:
                if device == "cuda":
                    raise
                available = False
                self.fallback_reason = str(exc)
            if available:
                self.device = "cuda"
            elif device == "cuda":
                raise RuntimeError(f"CUDA device {device_index} is unavailable")
            else:
                self.device = "cpu"
                self.fallback_reason = self.fallback_reason or "No usable CUDA device"
        self.compute_type = self._precision(self.device)
        try:
            self.model = self._load()
        except Exception as exc:
            if self.device != "cuda" or device != "auto" or not is_accelerator_error(exc):
                raise
            self._fallback(exc)
        self._report()

    def _precision(self, device):
        if self.requested_compute_type != "auto":
            if device == "cpu" and self.requested_device == "auto":
                import ctranslate2
                if self.requested_compute_type not in ctranslate2.get_supported_compute_types("cpu"):
                    return "int8"
            return self.requested_compute_type
        return "int8_float16" if device == "cuda" else "int8"

    def _load(self):
        return self.factory(self.path, device=self.device, compute_type=self.compute_type,
                            device_index=self.device_index if self.device == "cuda" else 0)

    def _report(self):
        if self.progress:
            message = f"Whisper runtime: {self.device}, {self.compute_type}"
            if self.fallback_reason:
                message += f"; CPU fallback: {self.fallback_reason}"
            self.progress(message)

    def _fallback(self, error):
        import gc
        self.model = None
        gc.collect()
        self.device = "cpu"
        # A GPU-only precision requested in auto mode must not break CPU recovery.
        self.compute_type = self._precision("cpu")
        self.fallback_reason = str(error)
        self.model = self._load()

    @property
    def execution(self):
        return {"device": self.device, "compute_type": self.compute_type,
                "device_index": self.device_index if self.device == "cuda" else None,
                "fallback_reason": self.fallback_reason}

    def transcribe(self, samples, **kwargs):
        # faster-whisper defers GPU work until its segments iterator is consumed.
        # Discard a partial chunk and rerun it once on CPU; never duplicate words.
        try:
            segments, info = self.model.transcribe(samples, **kwargs)
            return iter(list(segments)), info
        except Exception as exc:
            if self.device != "cuda" or self.requested_device != "auto" or not is_accelerator_error(exc):
                raise
            self._fallback(exc)
            self._report()
            segments, info = self.model.transcribe(samples, **kwargs)
            return iter(list(segments)), info


def _model(name, device="auto", compute_type="auto", device_index=0, progress=None):
    device_settings(device, device_index)
    if compute_type not in {"auto", "int8", "int8_float16", "int8_float32", "float16", "float32", "bfloat16", "int8_bfloat16"}:
        raise ValueError("Unsupported Whisper compute_type")
    key = (name, device, compute_type, device_index)
    if key not in _MODELS:
        prepare_nvidia_runtime()
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
        _MODELS[key] = _WhisperRuntime(resolved, device, compute_type, device_index, progress)
    else:
        _MODELS[key].progress = progress
        _MODELS[key]._report()
    return _MODELS[key]


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


class IncrementalSpeech:
    """Stable core windows; overlap supplies context but each word has one owner."""

    def __init__(self, transcribe, chunk_sec=30):
        self.transcribe = transcribe
        self.chunk_sec = chunk_sec
        self.parts = {}

    def scan(self, path, end, cache_dir, **kwargs):
        rows = []
        start = 0.
        while start < end - 1e-8:
            edge = min(start + self.chunk_sec, end)
            key = (path, start, edge)
            if key not in self.parts:
                self.parts[key] = self.transcribe(path, [(start, edge)], cache_dir, **kwargs)
            rows.append((start, edge, self.parts[key]))
            start = edge
        result = {k: v for k, v in rows[0][2].items() if k not in ('words', 'events', 'cache_identity')}
        for field in ('words', 'events'):
            result[field] = sorted([item for lo, hi, part in rows for item in part.get(field, [])
                if lo <= (item['start'] + item['end'])/2 < hi], key=lambda item: item['start'])
        result['has_audio'] = any(part.get('has_audio') for _, _, part in rows)
        result['analyzed_seconds'] = end
        result['execution'] = rows[-1][2].get('execution')
        result['executions'] = [{'range': [lo, hi], 'execution': part.get('execution')}
                                for lo, hi, part in rows]
        result['status'] = ('transcribed' if result['words'] else 'vocal-events' if result['events']
                            else 'no-speech' if result['has_audio'] else 'no-audio')
        return result


def transcribe_source(path, ranges, cache_dir, model="small", language="en", progress=None,
                      device="auto", compute_type="auto", device_index=0):
    """Return absolute source word timestamps; silent sources have no words.

    Missing/broken ASR is an explicit error, never a successful silent fallback.
    Set language=None to auto-detect for non-English source videos.
    """
    import av
    device_settings(device, device_index)
    src = Path(path).resolve()
    stat = src.stat()
    windows = _windows(ranges)
    identity = {"path": str(src), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "windows": windows, "model": model, "language": language, "version": ASR_VERSION,
                "device": device, "compute_type": compute_type, "device_index": device_index}
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
                runtime = _model(model, device=device, compute_type=compute_type,
                                 device_index=device_index, progress=progress)
                segments, _ = runtime.transcribe(
                    samples, language=language, beam_size=5, word_timestamps=True,
                    vad_filter=True, condition_on_previous_text=False, temperature=0.0)
                result["execution"] = getattr(runtime, "execution", None)
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
