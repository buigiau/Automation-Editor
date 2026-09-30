"""Classify short voice files and measure duration."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any, Iterable

from autoedit.audio.categories import category_from_stem
from autoedit.audio.detail import classify_recording, index_segments
from autoedit.audio.segments import expand_match_units, load_mono
from autoedit.audio.phonetics import filename_phonetics
from autoedit.audio.taxonomy import lookup, filename_label


AUDIO_SUFFIXES = {".wav", ".wave", ".aif", ".aiff", ".mp3", ".flac"}


def classify_filename(path: str | Path) -> tuple[str, float]:
    return category_from_stem(Path(path).stem)


def _wav_duration_and_shape(path: Path) -> dict[str, Any]:
    """Read format metadata and energy without misinterpreting 24-bit PCM."""
    import soundfile as sf
    import numpy as np
    decoded, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = decoded.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
    zcr = float(np.mean(np.signbit(mono[1:]) != np.signbit(mono[:-1]))) if len(mono) > 1 else 0.0
    return dict(duration_sec=len(mono)/rate, sample_rate=rate, channels=decoded.shape[1],
                rms=round(rms, 5), zero_cross_rate=round(zcr, 5), acoustic_hint=None)


def analyze_file(path: str | Path) -> dict[str, Any]:
    """Use curated filename labels and measure the recording's vocal envelope."""
    p = Path(path)
    phonetics = filename_phonetics(p.stem)
    taxonomy = lookup(p.stem)
    label = filename_label(p.stem)
    extra = _wav_duration_and_shape(p) if p.suffix.lower() in {".wav", ".wave"} else {
        "duration_sec": 0.0,
        "sample_rate": 0,
        "channels": 0,
        "rms": 0.0,
        "zero_cross_rate": 0.0,
        "acoustic_hint": None,
    }
    segments: list[dict[str, Any]] = []
    envelope = []
    active_start, active_end = 0.0, float(extra["duration_sec"])
    if p.suffix.lower() in {".wav", ".wave"} and p.exists():
        try:
            samples, rate = load_mono(str(p))
            import numpy as np
            hop = max(1, int(rate * 0.02))
            padded = np.pad(samples, (0, (-len(samples)) % hop))
            energy = np.sqrt(np.mean(padded.reshape(-1, hop)**2, axis=1)) if len(padded) else np.zeros(0)
            envelope = np.round(energy / max(float(energy.max()) if len(energy) else 0, 1e-8), 4).tolist()
            active = np.flatnonzero(energy > max(0.005, float(energy.max()) * 0.08)) if len(energy) else []
            if len(active):
                active_start = max(0, float(active[0] * hop / rate) - 0.005)
                active_end = min(len(samples)/rate, float((active[-1]+1) * hop/rate) + 0.005)
            if label:
                category = label
                segments = index_segments([{"start_sec": active_start, "end_sec": active_end,
                    "duration_sec": active_end-active_start, "category": category,
                    "confidence": 1.0, "source": "user-filename", "phrase": p.stem}])
            else:
                # Unknown filenames are acoustic estimates, never curated neutral.
                segments = index_segments(classify_recording(samples, rate, filename_stem=p.stem))
        except (wave.Error, ValueError, EOFError, OSError):
            segments = []
    if segments:
        main = max(segments, key=lambda s: (s["duration_sec"], s["confidence"]))
        category = main["category"]
        confidence = main["confidence"]
    else:
        category, confidence = "UNCLEAR", 0.3
    return {
        "path": str(p.resolve()) if p.exists() else str(p),
        "name": p.name,
        "stem": p.stem,
        "category": category,
        "group": taxonomy[0] if taxonomy else "UNKNOWN",
        "confidence": confidence,
        "categories": sorted({s["category"] for s in segments}),
        "segment_count": len(segments),
        "segments": segments,
        "method": "filename-and-waveform" if phonetics["action"] != "UNCLEAR" else "content-segments",
        "filename_hint": classify_filename(p)[0],
        "phonetics": phonetics,
        "envelope": envelope, "envelope_step_sec": 0.02,
        "active_start_sec": active_start, "active_end_sec": active_end,
        **extra,
    }


def iter_audio_files(directory: str | Path, pattern: str = "*.wav") -> list[Path]:
    root = Path(directory)
    files = [p for p in root.glob(pattern) if p.is_file()]
    files += [p for p in root.glob(pattern.upper()) if p.is_file()]
    # Windows is case-insensitive; unique by resolved path
    uniq: dict[str, Path] = {}
    for p in files:
        uniq[str(p.resolve()).lower()] = p
    return sorted(uniq.values(), key=lambda p: p.name.lower())


def analyze_audio_dir(
    directory: str | Path,
    pattern: str = "*.wav",
    files: Iterable[str] | None = None,
    max_files: int = 0,
) -> list[dict[str, Any]]:
    if files:
        paths = [Path(f) if Path(f).is_absolute() else Path(directory) / f for f in files]
    else:
        paths = iter_audio_files(directory, pattern)
    if max_files and max_files > 0:
        paths = paths[: max_files]
    return [analyze_file(p) for p in paths]
