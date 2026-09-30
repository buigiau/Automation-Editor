import numpy as np

from autoedit.audio.detail import classify_recording


def _tone(rate, seconds, hz, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_what_and_why_stay_distinct():
    samples = _tone(16000, 0.4, 180)
    what = classify_recording(
        samples, 16000,
        words=[{"word": "What?", "start": 0.0, "end": 0.16, "prob": 0.67}],
        filename_stem="what",
    )
    why = classify_recording(
        samples, 16000,
        words=[{"word": "Why?", "start": 0.0, "end": 0.2, "prob": 0.67}],
        filename_stem="why",
    )
    assert what[0]["category"] == "WHAT"
    assert what[0]["word"] == "what"
    assert why[0]["category"] == "WHY"
    assert what[0]["start_sec"] == 0.0
    assert what[0]["end_sec"] == 0.16


def test_weak_transcript_needs_filename_agreement():
    samples = _tone(16000, 0.4, 200)
    rejected = classify_recording(
        samples, 16000,
        words=[{"word": "Bye.", "start": 0.0, "end": 0.2, "prob": 0.45}],
        filename_stem="love",
    )
    assert rejected[0]["category"] != "BYE"
    accepted = classify_recording(
        samples, 16000,
        words=[{"word": "Five", "start": 0.02, "end": 0.3, "prob": 0.42}],
        filename_stem="five",
    )
    assert accepted[0]["category"] == "FIVE"
    assert accepted[0]["phrase"] == "five"


def test_laugh_syllables_merge_and_keep_span():
    samples = _tone(16000, 1.2, 220)
    segs = classify_recording(
        samples, 16000,
        words=[
            {"word": "Ha", "start": 0.0, "end": 0.2, "prob": 0.8},
            {"word": "ha", "start": 0.3, "end": 0.5, "prob": 0.9},
            {"word": "What?", "start": 0.8, "end": 1.0, "prob": 0.7},
        ],
        filename_stem="clip",
    )
    assert [s["category"] for s in segs] == ["LAUGH", "WHAT"]
    assert segs[0]["start_sec"] == 0.0
    assert segs[0]["end_sec"] == 0.5
    assert segs[0]["phrase"] == "ha ha"


def test_beep_comes_from_spectrum_not_a_guessed_word():
    rate = 16000
    samples = _tone(rate, 0.8, 1000, amp=0.8)
    segs = classify_recording(
        samples, rate,
        words=[{"word": "You", "start": 0.0, "end": 0.4, "prob": 0.9}],
        filename_stem="beepp",
    )
    assert segs[0]["category"] == "BEEP"
    assert segs[0]["word"] is None
    assert segs[0]["source"] == "acoustic"


def test_hallucinated_outro_is_ignored():
    samples = _tone(16000, 0.4, 200)
    segs = classify_recording(
        samples, 16000,
        words=[{"word": "Thanks for watching!", "start": 0.0, "end": 0.3, "prob": 0.91}],
        filename_stem="ohhh",
    )
    assert segs[0]["category"] != "THANKS"
    assert segs[0]["source"] == "acoustic"
