import copy
import numpy as np
import pytest

from autoedit.video.faces import annotate_speaking
from autoedit.match.matcher import match_slots_to_video


def sample(t, aperture=0.1, x=20):
    return {"time_sec": t, "backend": "mediapipe-landmarks", "face_detected": True,
            "face": [x, 20, 200, 200], "mouth_box": [x + 60, 120, 80, 20],
            "face_visible": 1, "closeup_score": 0.9, "mouth_clarity": 0.9,
            "frontal_score": 0.9, "lip_aperture": aperture, "lip_width_ratio": 0.6}


def test_camera_motion_and_static_open_mouth_are_not_talking():
    samples = [sample(i / 2, 0.25, x=20 + i * 5) for i in range(20)]
    for s in samples:
        s["activity_score"] = 1
    annotate_speaking(samples)
    assert all(s["clear_face"] for s in samples)
    assert not any(s["speaking"] for s in samples)


def test_repeated_lip_changes_on_clear_face_are_required():
    samples = [sample(i / 2, 0.05 if i % 2 else 0.2) for i in range(20)]
    annotate_speaking(samples)
    assert sum(s["speaking"] for s in samples) >= 16
    blurred = copy.deepcopy(samples)
    for s in blurred:
        s["mouth_clarity"] = 0.1
    annotate_speaking(blurred)
    assert not any(s["speaking"] for s in blurred)


def test_missing_face_or_different_face_does_not_fake_lip_movement():
    samples = [sample(i / 2, 0.05 if i % 2 else 0.2, x=20 if i % 2 else 500) for i in range(20)]
    annotate_speaking(samples)
    assert not any(s["speaking"] for s in samples)


def test_missing_samples_and_high_motion_without_face_never_fill_a_slot():
    for samples in ([], [{"time_sec": i / 2, "activity_score": 1} for i in range(30)]):
        annotate_speaking(samples)
        with pytest.raises(ValueError, match="Not enough qualifying footage"):
            match_slots_to_video([{"id": "1", "required_duration_sec": 3}], [],
                                 {"duration_sec": 15, "samples": samples})


def test_whole_cut_must_cover_face_not_just_a_good_instant():
    samples = [sample(i / 2, 0.05 if i % 2 else 0.2) for i in range(40)]
    annotate_speaking(samples)
    for s in samples:
        if s["time_sec"] > 2:
            s.update(clear_face=0, face_visible=0, speaking=0)
    with pytest.raises(ValueError, match="Not enough qualifying footage"):
        match_slots_to_video([{"id": "1", "required_duration_sec": 8}], [],
                             {"duration_sec": 20, "samples": samples})


def test_detected_talking_windows_keep_full_duration_and_gap():
    samples = [sample(i / 2, 0.05 if i % 2 else 0.2) for i in range(120)]
    annotate_speaking(samples)
    result = match_slots_to_video([{"id": str(i), "required_duration_sec": 5} for i in range(3)], [],
                                 {"duration_sec": 60, "samples": samples}, min_gap_sec=10)
    ranges = sorted((r["video"]["in_sec"], r["video"]["out_sec"]) for r in result)
    assert all(abs(b - a - 5) < 1e-8 for a, b in ranges)
    assert all(b[0] - a[1] >= 10 - 1e-8 for a, b in zip(ranges, ranges[1:]))


def test_unsampled_tail_cannot_cross_into_a_different_face():
    samples = [sample(i / 2, x=20 if i < 2 else 500) for i in range(3)]
    for s in samples:
        s.update(clear_face=1, speaking=1)
    # The cut ends at 0.96s; the sample at 1s reveals a change near its tail.
    # Neither of the previous clear samples is enough to approve that tail.
    with pytest.raises(ValueError, match="Not enough qualifying footage"):
        match_slots_to_video([{"id": "1", "required_duration_sec": 0.96}], [],
                             {"duration_sec": 1.1, "sample_fps": 2, "samples": samples})


def make_video(tmp_path):
    import cv2
    path = tmp_path / "test.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    assert writer.isOpened()
    for _ in range(40):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    return path


def test_cache_skips_decode_and_invalidates_on_setting_change(tmp_path, monkeypatch):
    from autoedit.video.analyzer import analyze_video
    import autoedit.video.reader as reader
    path = make_video(tmp_path)
    cache = tmp_path / "cache"
    log = []
    first = analyze_video(path, sample_fps=2, backend="opencv", cache_dir=cache)
    original = reader.sampled_frames
    def unexpected(*a, **kw):
        raise AssertionError("Cache hit should not decode video")
    monkeypatch.setattr(reader, "sampled_frames", unexpected)
    second = analyze_video(path, sample_fps=2, backend="opencv", cache_dir=cache, progress=log.append)
    assert first == second
    assert "cache hit" in log[0]
    monkeypatch.setattr(reader, "sampled_frames", original)
    third = analyze_video(path, sample_fps=4, backend="opencv", cache_dir=cache)
    assert third["sample_count"] > first["sample_count"]


def test_missing_mediapipe_does_not_fall_back_to_center(tmp_path, monkeypatch):
    import autoedit.video.faces as faces
    from autoedit.video.analyzer import analyze_video
    def missing():
        raise ImportError("missing")
    monkeypatch.setattr(faces, "FaceLandmarker", missing)
    with pytest.raises(RuntimeError, match="MediaPipe is required"):
        analyze_video(make_video(tmp_path))


def test_run_log_records_failure(tmp_path, monkeypatch):
    import autoedit.pipeline as pipeline
    def fail(config, log):
        log("Video 25%")
        raise ValueError("no speaking faces")
    monkeypatch.setattr(pipeline, "_run_pipeline", fail)
    with pytest.raises(ValueError):
        pipeline.run_pipeline({"job": {"output_dir": str(tmp_path)}})
    text = (tmp_path / "run.log").read_text()
    assert "Video 25%" in text and "FAILED" in text and "no speaking faces" in text


def test_crop_recovery_maps_landmarks_to_source_frame(monkeypatch):
    import autoedit.video.faces as faces

    created = []
    class Detector:
        def __init__(self):
            self.index = len(created)
            self.closed = False
            created.append(self)

        def detect(self, frame, time_sec):
            return sample(time_sec) if self.index == 2 else {"face_detected": False}

        def close(self):
            self.closed = True

    monkeypatch.setattr(faces, "FaceLandmarker", Detector)
    detector = faces.MultiRegionFaceLandmarker()
    try:
        frame = np.random.default_rng(3).integers(0, 255, (360, 640, 3), dtype=np.uint8)
        result = detector.detect(frame, 1)
        assert result["face_detected"]
        assert result["face"] == [148, 74, 200, 200]
        assert result["mouth_box"] == [208, 174, 80, 20]
        assert result["face_detection_region"] == 2
    finally:
        detector.close()
    assert all(d.closed for d in created)


def test_crop_recovery_cannot_invent_a_face(monkeypatch):
    import autoedit.video.faces as faces
    from types import SimpleNamespace
    monkeypatch.setattr(faces, "FaceLandmarker", lambda: SimpleNamespace(
        detect=lambda *a: {"face_detected": False}, close=lambda: None))
    detector = faces.MultiRegionFaceLandmarker()
    try:
        result = detector.detect(np.zeros((360, 640, 3), dtype=np.uint8), 1)
        assert not result["face_detected"]
        assert "face" not in result
    finally:
        detector.close()


@pytest.mark.parametrize("fps", [6, 20])
def test_speaking_window_uses_time_at_different_sampling_rates(fps):
    samples = [sample(i / fps, .05 if int((i / fps) / .4) % 2 else .2) for i in range(3 * fps)]
    annotate_speaking(samples)
    assert samples[fps]["speaking"] == 1


def test_animation_mouth_visibility_does_not_require_human_nose_alignment():
    samples = [sample(i / 6, .05 if i % 2 else .2) for i in range(18)]
    for s in samples:
        s["frontal_score"] = .4
    annotate_speaking(samples, source_kind="live_action")
    assert not any(s["clear_face"] for s in samples)
    annotate_speaking(samples, source_kind="animation")
    assert all(s["clear_face"] for s in samples)
    assert samples[6]["speaking"]


def test_source_kind_invalidates_analysis_cache(tmp_path):
    from autoedit.video.analyzer import analyze_video
    path = make_video(tmp_path)
    cache = tmp_path / "cache"
    first = analyze_video(path, backend="opencv", source_kind="live_action", cache_dir=cache)
    second = analyze_video(path, backend="opencv", source_kind="animation", cache_dir=cache)
    assert first["cache_identity"] != second["cache_identity"]
    assert second["cache_identity"]["source_kind"] == "animation"
