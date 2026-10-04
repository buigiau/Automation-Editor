from autoedit.video.mouth import classify_mouth


def test_closed():
    assert classify_mouth(0.05, 0.4, 0.4, 0.0)[0] == "CLOSED"


def test_face_continuity_accepts_motion_only_on_confirmed_physical_track():
    from autoedit.video.faces import _same_face
    a = {"face": [0, 0, 100, 100], "track_id": "track-1",
         "character_id": "actor-1", "identity_confidence": .8}
    b = {**a, "face": [40, 0, 100, 100]}
    assert _same_face(a, b)
    assert not _same_face(a, {**b, "track_id": "track-2"})
    assert not _same_face(a, {**b, "character_id": "actor-2"})
    assert not _same_face(a, {**b, "identity_confidence": .4})
    assert not _same_face({"face": a["face"]}, {"face": b["face"]})
    assert not _same_face(a, {**b, "face": [90, 0, 100, 100]})


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


def test_bounded_segments_apply_minimum_to_remaining_footage():
    from autoedit.video.pool import bounded_segments

    segments = [{"start_sec": .1, "end_sec": .3},
                {"start_sec": .9, "end_sec": 1.1},
                {"start_sec": 1.1, "end_sec": 1.3}]
    # The first segment lasts .2s despite floating-point subtraction; the tail
    # has only .1s inside the footage and the final segment is outside entirely.
    result = bounded_segments(segments, end_sec=1, min_sec=.2)
    assert len(result) == 1
    assert result[0]["start_sec"] == .1
    assert result[0]["end_sec"] == .3
