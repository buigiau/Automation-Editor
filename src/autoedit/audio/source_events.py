"""Conservative vocal-event hints outside transcribed speech.

Reuse the existing spectral cough/sigh and burst laugh detectors. These are
heuristics, so every event assignment remains flagged for listening review.
Ordinary transcript words ('cough', 'laugh') are never treated as actual events.
"""
from autoedit.audio.detail import _file_features, acoustic_category
from autoedit.audio.segments import (
    _active_mask, _cluster_runs, _frame_matrix, _runs, _spectral_features,
)


def detect_vocal_events(samples, rate, words, offset=0.0):
    if len(samples) < int(rate*.08):
        return []
    windowed, raw, win, hop = _frame_matrix(samples, rate)
    features = _spectral_features(windowed, raw, rate)
    spans = _cluster_runs(_runs(_active_mask(features["rms"])), hop/rate)
    events = []
    for a, b, kind in spans:
        lo, hi = a*hop/rate, min(len(samples)/rate, ((b-1)*hop+win)/rate)
        if not .08 <= hi-lo <= 4:
            continue
        start, end = offset+lo, offset+hi
        if any(w["start"]-.08 < end and w["end"]+.08 > start for w in words):
            continue
        clip = samples[int(lo*rate):int(hi*rate)]
        action = acoustic_category(_file_features(clip, rate))
        if action not in {"COUGH", "SIGH"}:
            action = "LAUGH" if kind == "laugh" else None
        if action:
            events.append({"action": action, "start": round(start, 4), "end": round(end, 4),
                           "prob": .75, "method": "spectral-vocal-event-hint", "needs_review": True})
    return events
