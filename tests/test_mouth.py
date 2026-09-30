from autoedit.video.mouth import classify_mouth


def test_closed():
    assert classify_mouth(0.05, 0.4, 0.4, 0.0)[0] == "CLOSED"


def test_o():
    assert classify_mouth(0.4, 0.45, 0.8, 0.0)[0] == "O"


def test_a():
    assert classify_mouth(0.7, 0.4, 0.3, 0.0)[0] == "A"


def test_e():
    assert classify_mouth(0.3, 0.8, 0.2, 0.1)[0] == "E"


def test_haha():
    assert classify_mouth(0.35, 0.7, 0.3, 0.6)[0] == "HAHA"


def test_cry():
    assert classify_mouth(0.4, 0.5, 0.3, -0.5)[0] == "CRY"


def test_merge_spaced_samples():
    from autoedit.video.analyzer import _merge_segments

    samples = [
        {"category": "A", "time_sec": i * 0.5, "confidence": 0.8, "backend": "t"}
        for i in range(6)
    ]
    samples[3]["category"] = "O"
    samples[4]["category"] = "O"
    segs = _merge_segments(samples, min_sec=0.2, gap=0.2, sample_interval=0.5)
    cats = [s["category"] for s in segs]
    assert "A" in cats and "O" in cats
    assert all(s["duration_sec"] >= 0.2 for s in segs)
