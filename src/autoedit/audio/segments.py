"""Split a WAV into timed expression segments using the waveform.

Classification uses short-time loudness, zero-crossings, and spectral
balance. The filename is not consulted.
"""

from __future__ import annotations

import wave
from array import array
from typing import Any

import numpy as np

# Broad mouth/expression labels. LAUGH is the audio-side name for a laugh.
CATEGORIES = ("A", "E", "O", "U", "LAUGH", "CRY", "CLOSED", "NEUTRAL")

_WIN_SEC = 0.030
_HOP_SEC = 0.010
_MIN_SEG_SEC = 0.08


def load_mono(path: str) -> tuple[np.ndarray, int]:
    """Decode integer/float WAV, including signed 24-bit samples."""
    import soundfile as sf
    decoded, rate = sf.read(path, dtype="float32", always_2d=True)
    return decoded.mean(axis=1), rate


def _frame_matrix(samples: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    win = max(8, int(rate * _WIN_SEC))
    hop = max(1, int(rate * _HOP_SEC))
    if samples.size < win:
        pad = np.zeros(win, dtype=np.float32)
        pad[: samples.size] = samples
        samples = pad
    n = 1 + max(0, (samples.size - win) // hop)
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n, win),
        strides=(samples.strides[0] * hop, samples.strides[0]),
        writeable=False,
    )
    window = np.hanning(win).astype(np.float32)
    return frames * window, frames, win, hop


def _spectral_features(windowed: np.ndarray, raw_frames: np.ndarray, rate: int) -> dict[str, np.ndarray]:
    n = windowed.shape[0]
    spec = np.abs(np.fft.rfft(windowed, axis=1))
    freqs = np.fft.rfftfreq(windowed.shape[1], d=1.0 / rate)
    mag_sum = spec.sum(axis=1) + 1e-12
    centroid = (spec * freqs).sum(axis=1) / mag_sum
    power = spec ** 2
    total = power.sum(axis=1) + 1e-12

    def band(lo: float, hi: float) -> np.ndarray:
        mask = (freqs >= lo) & (freqs < hi)
        return power[:, mask].sum(axis=1) / total

    sub = band(90, 400)
    low = band(400, 800)
    mid = band(800, 1600)
    high = band(1600, 4000)
    rms = np.sqrt(np.mean(raw_frames ** 2, axis=1) + 1e-12)
    signs = np.sign(raw_frames)
    signs[signs == 0] = 1
    zc = np.mean(signs[:, 1:] * signs[:, :-1] < 0, axis=1)
    # Spectral flatness: geometric / arithmetic mean of magnitude.
    log_mean = np.mean(np.log(spec + 1e-12), axis=1)
    flat = np.exp(log_mean) / (spec.mean(axis=1) + 1e-12)
    times = (np.arange(n) * int(rate * _HOP_SEC) + windowed.shape[1] / 2) / rate
    return {
        "time": times.astype(np.float32),
        "rms": rms.astype(np.float32),
        "zcr": zc.astype(np.float32),
        "centroid": centroid.astype(np.float32),
        "sub": sub.astype(np.float32),
        "low": low.astype(np.float32),
        "mid": mid.astype(np.float32),
        "high": high.astype(np.float32),
        "flat": np.clip(flat, 0, 1).astype(np.float32),
    }


def _scores(
    rms: float,
    zcr: float,
    centroid: float,
    sub: float,
    low: float,
    mid: float,
    high: float,
    flat: float,
    peaks: int,
    dur: float,
) -> dict[str, float]:
    brightness = min(1.0, centroid / 2500.0)
    # Laugh is decided by burst clustering, not by a smooth vowel's ripples.
    cry = 0.0
    if peaks <= 2 and dur >= 0.28 and zcr > 0.12 and flat > 0.2 and brightness < 0.55 and high < 0.25:
        cry = 0.5 + min(0.3, (zcr - 0.12) * 1.2)
    if rms < 0.012:
        return {
            "CLOSED": 1.0, "NEUTRAL": 0.2, "A": 0.0, "E": 0.0, "O": 0.0, "U": 0.0, "LAUGH": 0.0, "CRY": 0.0,
        }
    return {
        "U": sub * 1.8 + (1.0 - brightness) * 0.45 - high * 1.4 - mid * 0.15,
        "O": low * 1.45 + sub * 0.45 + (1.0 - abs(brightness - 0.38)) * 0.25 - high * 1.1,
        "A": mid * 1.35 + (1.0 - abs(brightness - 0.55)) * 0.35 - sub * 0.2,
        "E": high * 1.7 + brightness * 0.9 - sub * 0.5,
        "LAUGH": 0.0,
        "CRY": cry,
        "CLOSED": 0.15 if rms < 0.03 else 0.0,
        "NEUTRAL": 0.12,
    }


def _pick(scores: dict[str, float]) -> tuple[str, float]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_s = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = best_s - second
    if best_s < 0.22:
        return "NEUTRAL", 0.4
    conf = max(0.4, min(0.95, 0.5 + margin))
    return best, round(float(conf), 3)


def _peak_count(rms: np.ndarray) -> int:
    if rms.size < 3:
        return 1 if float(rms.max()) > 0.02 else 0
    smooth = np.convolve(rms, np.ones(3) / 3.0, mode="same")
    height = max(float(smooth.max()) * 0.45, 0.02)
    peaks = 0
    last = -99
    for i in range(1, smooth.size - 1):
        if smooth[i] >= height and smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1]:
            if i - last >= 3:
                peaks += 1
                last = i
    return max(peaks, 1)


def _active_mask(rms: np.ndarray) -> np.ndarray:
    if rms.size == 0:
        return rms.astype(bool)
    peak = float(np.max(rms))
    if peak < 0.008:
        return np.zeros(rms.shape, dtype=bool)
    floor = float(np.percentile(rms, 10))
    thresh = max(0.01, floor * 3.0, peak * 0.18)
    thresh = min(thresh, peak * 0.45)
    active = rms >= thresh
    # Bridge gaps shorter than ~40 ms so a vowel is not split by a dip.
    gap = 0
    start = None
    for i, flag in enumerate(active):
        if flag:
            if start is not None and 0 < gap <= 4:
                active[start:i] = True
            start = None
            gap = 0
        else:
            if start is None:
                start = i
            gap += 1
    return active


def _cluster_runs(runs: list[tuple[int, int]], hop_sec: float) -> list[tuple[int, int, str]]:
    """Group short, closely spaced bursts into one laugh. Leave sustained sounds alone."""
    if not runs:
        return []
    clusters: list[list[tuple[int, int]]] = [[runs[0]]]
    for run in runs[1:]:
        prev_a, prev_b = clusters[-1][-1]
        gap = (run[0] - prev_b) * hop_sec
        run_len = (run[1] - run[0]) * hop_sec
        prev_len = (prev_b - prev_a) * hop_sec
        if gap <= 0.18 and prev_len <= 0.28 and run_len <= 0.5:
            clusters[-1].append(run)
        else:
            clusters.append([run])
    spans: list[tuple[int, int, str]] = []
    for cluster in clusters:
        if len(cluster) >= 3:
            spans.append((cluster[0][0], cluster[-1][1], "laugh"))
        else:
            for a, b in cluster:
                spans.append((a, b, "sound"))
    return spans


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _label_span(feat: dict[str, np.ndarray], a: int, b: int, rate_hop: float) -> dict[str, Any]:
    sl = slice(a, b)
    rms = feat["rms"][sl]
    dur = max(_HOP_SEC, (b - a) * rate_hop)
    peaks = _peak_count(rms)
    scores = _scores(
        rms=float(np.mean(rms)),
        zcr=float(np.mean(feat["zcr"][sl])),
        centroid=float(np.mean(feat["centroid"][sl])),
        sub=float(np.mean(feat["sub"][sl])),
        low=float(np.mean(feat["low"][sl])),
        mid=float(np.mean(feat["mid"][sl])),
        high=float(np.mean(feat["high"][sl])),
        flat=float(np.mean(feat["flat"][sl])),
        peaks=peaks,
        dur=dur,
    )
    category, confidence = _pick(scores)
    start = float(feat["time"][a] - _WIN_SEC / 2)
    end = float(feat["time"][b - 1] + _WIN_SEC / 2)
    start = max(0.0, start)
    return {
        "start_sec": round(start, 3),
        "end_sec": round(end, 3),
        "duration_sec": round(max(0.0, end - start), 3),
        "category": category,
        "confidence": confidence,
        "features": {
            "rms": round(float(np.mean(rms)), 4),
            "zcr": round(float(np.mean(feat["zcr"][sl])), 4),
            "centroid_hz": round(float(np.mean(feat["centroid"][sl])), 1),
            "sub_ratio": round(float(np.mean(feat["sub"][sl])), 3),
            "low_ratio": round(float(np.mean(feat["low"][sl])), 3),
            "mid_ratio": round(float(np.mean(feat["mid"][sl])), 3),
            "high_ratio": round(float(np.mean(feat["high"][sl])), 3),
            "flatness": round(float(np.mean(feat["flat"][sl])), 3),
            "peak_count": peaks,
        },
    }


def _split_on_label_changes(feat: dict[str, np.ndarray], a: int, b: int) -> list[tuple[int, int]]:
    """Cut a voiced run where the frame label stays different for ~80 ms."""
    labels: list[str] = []
    for i in range(a, b):
        scores = _scores(
            rms=float(feat["rms"][i]),
            zcr=float(feat["zcr"][i]),
            centroid=float(feat["centroid"][i]),
            sub=float(feat["sub"][i]),
            low=float(feat["low"][i]),
            mid=float(feat["mid"][i]),
            high=float(feat["high"][i]),
            flat=float(feat["flat"][i]),
            peaks=1,
            dur=0.1,
        )
        labels.append(_pick(scores)[0])
    # Majority smooth over 5 frames.
    sm: list[str] = []
    for i in range(len(labels)):
        window = labels[max(0, i - 2) : i + 3]
        sm.append(max(window, key=window.count))  # Resolve ties in temporal order, not randomized set order.
    cuts = [a]
    hold = sm[0] if sm else "NEUTRAL"
    hold_at = 0
    for i, lab in enumerate(sm):
        if lab != hold:
            if i - hold_at >= 8:
                cuts.append(a + i)
                hold = lab
                hold_at = i
        else:
            hold_at = i
    spans = []
    edges = cuts + [b]
    for i in range(len(edges) - 1):
        if edges[i + 1] - edges[i] >= 4:
            spans.append((edges[i], edges[i + 1]))
    return spans or [(a, b)]


def segment_samples(samples: np.ndarray, rate: int) -> list[dict[str, Any]]:
    if samples.size == 0 or rate <= 0:
        return []
    windowed, raw_frames, _win, hop = _frame_matrix(samples, rate)
    feat = _spectral_features(windowed, raw_frames, rate)
    hop_sec = hop / float(rate)
    mask = _active_mask(feat["rms"])
    spans: list[tuple[int, int]] = []
    forced_laugh: set[tuple[int, int]] = set()
    for a, b, kind in _cluster_runs(_runs(mask), hop_sec):
        if (b - a) * hop_sec < 0.05:
            continue
        if kind == "laugh":
            spans.append((a, b))
            forced_laugh.add((a, b))
        else:
            spans.extend(_split_on_label_changes(feat, a, b))
    segments = []
    for a, b in spans:
        seg = _label_span(feat, a, b, hop_sec)
        if (a, b) in forced_laugh and seg["features"]["peak_count"] >= 3:
            seg["category"] = "LAUGH"
            seg["confidence"] = max(seg["confidence"], 0.75)
        if seg["duration_sec"] >= _MIN_SEG_SEC and seg["category"] != "CLOSED":
            segments.append(seg)
    if not segments:
        # Entire file is quiet or one short puff.
        whole = _label_span(feat, 0, len(feat["rms"]), hop_sec)
        whole["start_sec"] = 0.0
        whole["end_sec"] = round(samples.size / float(rate), 3)
        whole["duration_sec"] = whole["end_sec"]
        segments = [whole]
    for i, seg in enumerate(segments):
        seg["id"] = f"aseg-{i:03d}"
        seg["index"] = i
    return segments


def segment_wav(path: str) -> list[dict[str, Any]]:
    samples, rate = load_mono(path)
    return segment_samples(samples, rate)


def expand_match_units(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One match unit per detected segment, file order then time order."""
    units: list[dict[str, Any]] = []
    for item in files:
        segs = item.get("segments") or []
        if not segs:
            units.append(item)
            continue
        for seg in segs:
            units.append(
                {
                    **item,
                    "category": seg["category"],
                    "confidence": seg["confidence"],
                    "duration_sec": seg["duration_sec"],
                    "segment_id": seg["id"],
                    "segment_index": seg["index"],
                    "segment_start_sec": seg["start_sec"],
                    "segment_end_sec": seg["end_sec"],
                    "segment_features": seg.get("features") or {},
                    "segments": segs,
                }
            )
    return units
