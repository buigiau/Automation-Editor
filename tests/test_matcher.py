from autoedit.match.matcher import match_audio_to_video


def test_same_category_no_reuse():
    audio = [
        {"name": "a.wav", "category": "A", "duration_sec": 0.4, "confidence": 0.9},
        {"name": "a2.wav", "category": "A", "duration_sec": 0.4, "confidence": 0.9},
    ]
    segs = [
        {"id": "1", "category": "A", "start_sec": 1.0, "end_sec": 2.0, "duration_sec": 1.0, "confidence": 0.8},
        {"id": "2", "category": "A", "start_sec": 5.0, "end_sec": 6.5, "duration_sec": 1.5, "confidence": 0.8},
        {"id": "3", "category": "O", "start_sec": 8.0, "end_sec": 9.0, "duration_sec": 1.0, "confidence": 0.8},
    ]
    matches = match_audio_to_video(audio, segs)
    assert matches[0]["status"] == "matched"
    assert matches[1]["status"] == "matched"
    assert matches[0]["video"]["id"] != matches[1]["video"]["id"]


def test_fallback_when_category_missing():
    audio = [{"name": "haha.wav", "category": "HAHA", "duration_sec": 0.5, "confidence": 0.9}]
    segs = [
        {"id": "1", "category": "A", "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0, "confidence": 0.9},
    ]
    matches = match_audio_to_video(audio, segs)
    assert matches[0]["status"] in ("fallback", "matched")
    assert matches[0]["video"] is not None
