"""Detailed labels for one recording.

Speech words come from a transcription of the waveform. A word is kept only
when its probability is high enough, or when a weaker transcript agrees with
the filename. Whistles, beeps, coughs, sighs, and similar sounds are labeled
from the spectrum and are not overwritten by a transcript.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from autoedit.audio.categories import _fold
from autoedit.audio.segments import (
    _active_mask,
    _frame_matrix,
    _peak_count,
    _spectral_features,
    segment_samples,
)

from autoedit.audio.taxonomy import TOKEN_CATEGORY as _TOKEN_CATEGORY, filename_label

_NUMBER_WORDS = {
    "0": "ZERO",
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "8": "EIGHT",
    "10": "TEN",
}

_HALLUCINATION_SNIPS = (
    "thanks for watching",
    "subscribe",
    "チャンネル",
    "登録",
    "구독",
    "좋아요",
    "부탁",
)

def _clean_token(word: str) -> str:
    word = word.strip().lower()
    word = word.replace("'", "")
    word = re.sub(r"[^0-9a-z\u00c0-\u024f]+", "", word)
    return word


def _category_for_token(token: str) -> str | None:
    if not token:
        return None
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    if token in _TOKEN_CATEGORY:
        return _TOKEN_CATEGORY[token]
    # Elongated "ahhh", "ohhh", "wooo", "booo", "rrrr".
    letters = re.sub(r"(.)\1+", r"\1", token)
    if letters in _TOKEN_CATEGORY:
        return _TOKEN_CATEGORY[letters]
    if set(token) <= set("r") and len(token) >= 3:
        return "RASPBERRY"
    if letters in {"arg", "urg", "ugh", "agr"} or (set(letters) <= set("arg") and "g" in letters and len(token) >= 6):
        return "GROAN"
    if set(token) <= set("o") and len(token) >= 3:
        return "OH"
    if set(token) <= set("ar") and len(token) >= 6:
        return "GROAN"
    return None


def _is_hallucination(text: str) -> bool:
    low = text.lower()
    if any(snip in low for snip in _HALLUCINATION_SNIPS):
        return True
    token = _clean_token(low)
    if _category_for_token(token) in {"GROAN", "RASPBERRY", "OH", "BOO"}:
        return False
    letters = re.sub(r"[^a-z]", "", low)
    if len(letters) >= 24 and len(set(letters)) <= 4:
        return True
    return False


def _file_features(samples: np.ndarray, rate: int) -> dict[str, float]:
    windowed, raw, _win, _hop = _frame_matrix(samples, rate)
    feat = _spectral_features(windowed, raw, rate)
    mask = _active_mask(feat["rms"])
    use = mask if mask.any() else np.ones(len(feat["rms"]), dtype=bool)
    return {
        "duration": float(samples.size / rate) if rate else 0.0,
        "rms": float(feat["rms"][use].mean()),
        "centroid": float(feat["centroid"][use].mean()),
        "zcr": float(feat["zcr"][use].mean()),
        "flat": float(feat["flat"][use].mean()),
        "sub": float(feat["sub"][use].mean()),
        "low": float(feat["low"][use].mean()),
        "mid": float(feat["mid"][use].mean()),
        "high": float(feat["high"][use].mean()),
        "peaks": float(_peak_count(feat["rms"])),
    }


def acoustic_category(feat: dict[str, float]) -> str | None:
    """Non-speech class when the spectrum is unambiguous. Otherwise None."""
    cent = feat["centroid"]
    flat = feat["flat"]
    zcr = feat["zcr"]
    sub, low, mid, high = feat["sub"], feat["low"], feat["mid"], feat["high"]
    dur = feat["duration"]
    peaks = feat["peaks"]
    if cent >= 4800 and zcr >= 0.15 and flat < 0.18 and high < 0.35:
        return "WHISTLE"
    if cent >= 4500 and flat >= 0.20 and zcr >= 0.15 and peaks >= 3:
        return "COUGH"
    if flat <= 0.01 and dur >= 3.0 and cent < 1400:
        return "BELL"
    if flat <= 0.015 and mid >= 0.85 and 700 <= cent <= 1800 and dur < 3.0:
        return "BEEP"
    if flat <= 0.03 and mid >= 0.9 and dur < 2.0 and cent < 1600 and zcr < 0.08 and peaks <= 2:
        return "CHIME"
    if (
        dur < 0.6
        and flat < 0.04
        and sub < 0.08
        and 0.2 <= low <= 0.5
        and 0.25 <= mid <= 0.55
        and 0.2 <= high <= 0.4
        and flat >= 0.018
    ):
        return "HORN"
    if sub >= 0.75 and zcr <= 0.06 and dur >= 0.5 and 0.015 <= flat <= 0.12 and cent < 2300:
        return "SIGH"
    return None


def _accept_word(category: str, prob: float, name_label: str | None) -> bool:
    if category is None:
        return False
    if prob >= 0.60:
        return True
    if prob >= 0.40 and name_label and name_label == category:
        return True
    return False


def _speech_segments(words: list[dict[str, Any]], name_label: str | None, duration: float) -> list[dict[str, Any]]:
    prepared = []
    for word in words:
        raw = str(word.get("word") or "")
        if _is_hallucination(raw):
            continue
        token = _clean_token(raw)
        category = _category_for_token(token)
        prob = float(word.get("prob") or 0.0)
        if not _accept_word(category, prob, name_label):
            continue
        prepared.append(
            {
                "category": category,
                "phrase": token,
                "start_sec": float(word.get("start") or 0.0),
                "end_sec": float(word.get("end") or 0.0),
                "confidence": round(prob, 3),
            }
        )
    if not prepared:
        return []
    # Collapse a run of laugh syllables into one bout.
    merged: list[dict[str, Any]] = []
    for item in prepared:
        if (
            merged
            and item["category"] == "LAUGH"
            and merged[-1]["category"] == "LAUGH"
            and item["start_sec"] - merged[-1]["end_sec"] <= 0.45
        ):
            prev = merged[-1]
            prev["end_sec"] = max(prev["end_sec"], item["end_sec"])
            prev["phrase"] = (prev["phrase"] + " " + item["phrase"]).strip()
            prev["confidence"] = round(max(prev["confidence"], item["confidence"]), 3)
        else:
            merged.append(dict(item))
    collapsed: list[dict[str, Any]] = []
    for item in merged:
        if (
            collapsed
            and collapsed[-1]["category"] == item["category"]
            and item["start_sec"] <= collapsed[-1]["end_sec"] + 0.05
        ):
            prev = collapsed[-1]
            prev["end_sec"] = max(prev["end_sec"], item["end_sec"])
            prev["confidence"] = round(max(prev["confidence"], item["confidence"]), 3)
            if item["phrase"] not in prev["phrase"]:
                prev["phrase"] = (prev["phrase"] + " " + item["phrase"]).strip()
        else:
            collapsed.append(item)
    merged = collapsed
    segments = []
    for item in merged:
        start = max(0.0, item["start_sec"])
        end = item["end_sec"] if item["end_sec"] > start else min(duration, start + 0.2)
        segments.append(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(max(0.0, end - start), 3),
                "category": item["category"],
                "confidence": item["confidence"],
                "phrase": item["phrase"],
                "word": item["phrase"],
                "source": "speech",
            }
        )
    return segments


def _vowel_segments(samples: np.ndarray, rate: int) -> list[dict[str, Any]]:
    raw = segment_samples(samples, rate)
    mapped = {"A": "VOWEL_A", "E": "VOWEL_E", "O": "VOWEL_O", "U": "VOWEL_U", "LAUGH": "LAUGH", "CRY": "CRY", "CLOSED": "CLOSED", "NEUTRAL": "UNCLEAR"}
    out = []
    for seg in raw:
        seg = dict(seg)
        seg["category"] = mapped.get(seg["category"], seg["category"])
        seg["phrase"] = None
        seg["word"] = None
        seg["source"] = "acoustic"
        out.append(seg)
    return out


def classify_recording(
    samples: np.ndarray,
    rate: int,
    words: list[dict[str, Any]] | None = None,
    filename_stem: str = "",
) -> list[dict[str, Any]]:
    if samples.size == 0 or rate <= 0:
        return []
    duration = float(samples.size / rate)
    feat = _file_features(samples, rate)
    name_label = filename_label(filename_stem)
    sound = acoustic_category(feat)
    if sound in {"BEEP", "BELL", "CHIME", "COUGH", "HORN"}:
        return [
            {
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "duration_sec": round(duration, 3),
                "category": sound,
                "confidence": 0.75,
                "phrase": None,
                "word": None,
                "source": "acoustic",
                "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in feat.items()},
            }
        ]
    speech = _speech_segments(words or [], name_label, duration)
    if speech:
        return speech
    if sound:
        return [
            {
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "duration_sec": round(duration, 3),
                "category": sound,
                "confidence": 0.75,
                "phrase": None,
                "word": None,
                "source": "acoustic",
                "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in feat.items()},
            }
        ]
    return _vowel_segments(samples, rate)


def index_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, seg in enumerate(segments):
        seg["id"] = f"aseg-{i:03d}"
        seg["index"] = i
    return segments
