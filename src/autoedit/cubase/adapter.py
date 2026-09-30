"""Build Cubase-ready stems and a timeline-aligned mixdown WAV.

Cubase has no official API to drop a file onto a named track at a timestamp.
This adapter therefore:

1. Writes one silence-padded stem per slot/track.
2. Mixes those stems to mixdown.wav (the WAV Premiere will import).
3. Writes cubase_import.json describing track, start time, and source file.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


def _read_wav_mono_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sw * 8}-bit: {path}")
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    return samples, rate


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples
    duration = samples.size / float(src_rate)
    n = int(round(duration * dst_rate))
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def _write_wav_stereo(path: Path, stereo: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(stereo, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    interleaved = pcm.reshape(-1).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved)


def apply_cubase_plan(plan: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    if plan.get("audio_mode") == "sampler_per_scene_v1":
        from autoedit.cubase.sampler import render_sampler_bundle
        return render_sampler_bundle(plan, output_dir)
    out = Path(output_dir)
    stems_dir = out / "cubase_stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    mix_cfg = plan.get("mixdown") or {}
    sample_rate = int(mix_cfg.get("sample_rate") or 48000)
    slots = plan.get("cubase_slots", plan.get("slots")) or []

    ends = []
    for slot in slots:
        start = float((slot.get("cubase") or {}).get("start_sec") or 0.0)
        audio = slot.get("audio") or {}
        dur = float(audio.get("duration_sec") or (slot.get("premiere") or {}).get("duration_sec") or 0.0)
        ends.append(start + dur)
    total = max(ends) if ends else 0.0
    # Pad a little past last clip so Premiere has handles
    total = max(total, 1.0) + 0.25
    n_samples = int(math.ceil(total * sample_rate))
    mix = np.zeros((n_samples, 2), dtype=np.float32)

    import_list: list[dict[str, Any]] = []
    stems_by_track: dict[str, np.ndarray] = {}
    for slot in slots:
        audio = slot.get("audio")
        cubase = slot.get("cubase") or {}
        if not audio or not audio.get("path"):
            continue
        src = Path(audio["path"])
        track_name = str(cubase.get("track_name") or "Sampler Track 01")
        if not src.exists():
            import_list.append(
                {
                    "slot_id": slot.get("id"),
                    "track_name": track_name,
                    "error": f"missing audio {src}",
                }
            )
            continue
        try:
            samples, rate = _read_wav_mono_float(src)
        except Exception as exc:
            import_list.append(
                {
                    "slot_id": slot.get("id"),
                    "track_name": track_name,
                    "error": str(exc),
                }
            )
            continue
        seg_start = audio.get("segment_start_sec")
        seg_end = audio.get("segment_end_sec")
        if seg_start is not None or seg_end is not None:
            a = int(round(float(seg_start or 0) * rate))
            b = int(round(float(seg_end) * rate)) if seg_end is not None else samples.size
            a = max(0, min(a, samples.size))
            b = max(a, min(b, samples.size))
            samples = samples[a:b]
        samples = _resample(samples, rate, sample_rate)
        start = float(cubase.get("start_sec") or 0.0)
        offset = int(round(start * sample_rate))
        if track_name not in stems_by_track:
            stems_by_track[track_name] = np.zeros((n_samples, 2), dtype=np.float32)
        stem = stems_by_track[track_name]
        end = min(n_samples, offset + samples.size)
        if offset < n_samples and end > offset:
            chunk = samples[: end - offset]
            stem[offset:end, 0] += chunk
            stem[offset:end, 1] += chunk
            mix[offset:end, 0] += chunk
            mix[offset:end, 1] += chunk
        import_list.append(
            {
                "slot_id": slot.get("id"),
                "track_name": track_name,
                "track_index": cubase.get("track_index"),
                "start_sec": start,
                "source": str(src),
                "segment_start_sec": audio.get("segment_start_sec"),
                "segment_end_sec": audio.get("segment_end_sec"),
                "category": audio.get("category"),
                "duration_sec": round(samples.size / sample_rate, 3),
            }
        )

    stem_paths: dict[str, str] = {}
    for track_name, stem in stems_by_track.items():
        peak_t = float(np.max(np.abs(stem))) if stem.size else 0.0
        if peak_t > 1.0:
            stem = stem / peak_t * 0.95
            stems_by_track[track_name] = stem
        safe = track_name.replace(" ", "_")
        stem_path = stems_dir / f"{safe}.wav"
        _write_wav_stereo(stem_path, stem, sample_rate)
        stem_paths[track_name] = str(stem_path.resolve())
    for row in import_list:
        if "error" not in row:
            row["stem"] = stem_paths.get(str(row.get("track_name")), "")

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak * 0.95

    mix_path = Path(mix_cfg.get("path") or (out / "mixdown.wav"))
    _write_wav_stereo(mix_path, mix, sample_rate)
    plan["mixdown"] = {
        "path": str(mix_path.resolve()),
        "duration_sec": round(n_samples / sample_rate, 3),
        "sample_rate": sample_rate,
        "channels": 2,
        "peak": round(peak, 4),
    }

    manifest = {
        "cubase_project": (plan.get("project") or {}).get("cubase_project"),
        "mixdown": plan["mixdown"],
        "tracks": import_list,
        "instructions": [
            "Open the Cubase template listed in cubase_project.",
            "Import each cubase_stems/*.wav onto the matching Sampler Track (File > Import > Audio File) at 00:00:00:00.",
            "Each stem already contains every voice file assigned to that track, padded to slot times.",
            "Optionally bounce File > Export > Audio Mixdown, or use plugins/cubase-midi-remote/AutoEdit.js to trigger that command.",
            "The Python mixdown.wav is already timeline-aligned and is what Premiere should import.",
        ],
    }
    manifest_path = out / "cubase_import.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"mixdown": plan["mixdown"], "manifest": str(manifest_path.resolve()), "tracks": import_list}
