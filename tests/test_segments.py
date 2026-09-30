import math
import wave
from array import array
from pathlib import Path

from autoedit.audio.detail import classify_recording
from autoedit.audio.segments import expand_match_units, segment_samples
from autoedit.match.matcher import match_audio_to_video
import numpy as np


def _write(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    pcm = np.clip(samples, -1, 1)
    ints = (pcm * 32767).astype(np.int16)
    buf = array("h", ints.tolist())
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(buf.tobytes())


def _tone(rate: int, seconds: float, hz: float, amp: float = 0.6) -> np.ndarray:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    return (amp * np.sin(2 * math.pi * hz * t)).astype(np.float32)


def test_two_vowels_are_separate_segments():
    rate = 16000
    dark = _tone(rate, 0.45, 160)
    gap = np.zeros(int(0.2 * rate), dtype=np.float32)
    bright = _tone(rate, 0.45, 2400)
    samples = np.concatenate([dark, gap, bright])
    segs = segment_samples(samples, rate)
    cats = [s["category"] for s in segs]
    assert len(segs) >= 2
    assert segs[0]["category"] in {"U", "O"}
    assert segs[-1]["category"] == "E"
    assert segs[0]["end_sec"] < segs[-1]["start_sec"]
    assert segs[0]["start_sec"] < 0.15
    assert segs[-1]["end_sec"] > 0.9
    assert "A" not in cats or cats[-1] != "A"


def test_laugh_bursts_are_one_laugh_segment():
    rate = 16000
    pieces = []
    for i in range(5):
        pieces.append(_tone(rate, 0.07, 420, amp=0.7))
        pieces.append(np.zeros(int(0.09 * rate), dtype=np.float32))
    samples = np.concatenate(pieces)
    segs = segment_samples(samples, rate)
    assert any(s["category"] == "LAUGH" for s in segs)
    laugh = next(s for s in segs if s["category"] == "LAUGH")
    assert laugh["features"]["peak_count"] >= 3
    assert laugh["end_sec"] - laugh["start_sec"] > 0.4


def test_filename_does_not_override_spectrum(tmp_path):
    from autoedit.audio.detail import classify_recording

    samples = _tone(16000, 0.5, 2400)
    segs = classify_recording(samples, 16000, words=[{"word": "bye", "start": 0.0, "end": 0.4, "prob": 0.9}], filename_stem="a")
    # A confident transcript is speech. A bright tone with no trusted words stays a vowel.
    vowels = classify_recording(samples, 16000, words=[], filename_stem="a")
    assert vowels[0]["category"] in {"VOWEL_E", "VOWEL_A", "WHISTLE"}
    assert segs[0]["category"] == "BYE"
    assert segs[0]["word"] == "bye"


def test_matcher_uses_segment_category_not_filename():
    audio = expand_match_units(
        [
            {
                "name": "a.wav",
                "path": "a.wav",
                "category": "E",
                "duration_sec": 1.0,
                "confidence": 0.9,
                "segments": [
                    {
                        "id": "aseg-000",
                        "index": 0,
                        "category": "U",
                        "confidence": 0.8,
                        "duration_sec": 0.4,
                        "start_sec": 0.0,
                        "end_sec": 0.4,
                        "features": {},
                    },
                    {
                        "id": "aseg-001",
                        "index": 1,
                        "category": "E",
                        "confidence": 0.9,
                        "duration_sec": 0.4,
                        "start_sec": 0.5,
                        "end_sec": 0.9,
                        "features": {},
                    },
                ],
            }
        ]
    )
    assert [a["category"] for a in audio] == ["U", "E"]
    assert audio[1]["segment_start_sec"] == 0.5
    video = [
        {"id": "v-o", "category": "O", "start_sec": 1, "end_sec": 2, "duration_sec": 1, "confidence": 0.9},
        {"id": "v-e", "category": "E", "start_sec": 3, "end_sec": 4, "duration_sec": 1, "confidence": 0.9},
    ]
    matches = match_audio_to_video(audio, video)
    assert matches[0]["status"] == "matched"
    assert matches[0]["video"]["id"] == "v-o"  # U matches the round-mouth family
    assert matches[1]["video"]["id"] == "v-e"
    assert matches[1]["audio"]["segment_start_sec"] == 0.5
