from types import SimpleNamespace as NS

import numpy as np
import pytest

from autoedit.match.matcher import match_slots_to_video, shot_tier
from autoedit.video.pose import pose_metrics


def sample(t, tier):
    face = tier <= 2 or (tier == 4 and round(t * 2) % 2 == 0)
    body = tier == 3 or (tier == 4 and not face)
    return {"time_sec": t, "face_visible": int(face), "face_detected": face,
            "closeup_score": .8 if tier <= 2 else .1 if face else 0, "mouth_clarity": .8 if tier == 1 else 0,
            "clear_face": int(tier == 1), "speaking": int(tier == 1),
            "person_detected": body, "person_visible": int(body), "pose_confirmed": body,
            "shot_scale": "full_body" if tier == 3 else "upper_body",
            "activity_score": .1 if tier >= 4 else .8}


@pytest.mark.parametrize("kind", ["animation", "live_action"])
def test_tiers_exhaust_in_order_with_full_cuts(kind):
    samples = [sample(i / 2, min(5, i // 24 + 1) if i % 24 < 20 else 5) for i in range(120)]
    matches = match_slots_to_video([{"id": str(i), "required_duration_sec": 8} for i in range(4)], [],
                                  {"duration_sec": 60, "samples": samples, "source_kind": kind})
    assert [m["video"]["selection_metrics"]["tier"] for m in matches] == [1, 2, 3, 4]
    ranges = sorted((m["video"]["in_sec"], m["video"]["out_sec"]) for m in matches)
    assert all(b-a == pytest.approx(8) for a, b in ranges)
    assert all(a[1] <= b[0]+1e-8 for a, b in zip(ranges, ranges[1:]))
    with pytest.raises(ValueError, match="Scenery and unconfirmed pose"):
        match_slots_to_video([{"id": str(i), "required_duration_sec": 8} for i in range(5)], [],
                             {"duration_sec": 60, "samples": samples, "source_kind": kind})


@pytest.mark.parametrize("tier", [1, 2, 3, 4])
def test_each_tier_accepts_only_its_evidence(tier):
    result = match_slots_to_video([{"id": "1", "required_duration_sec": 3}], [],
        {"duration_sec": 10, "samples": [sample(i/2, tier) for i in range(20)]})
    assert result[0]["video"]["selection_metrics"]["tier"] == tier


def test_missing_samples_are_rejected_and_min_tier_is_respected():
    samples = [sample(i/2, 3) for i in (0, 1, 18, 19)]
    with pytest.raises(ValueError, match="Not enough qualifying"):
        match_slots_to_video([{"id": "1", "required_duration_sec": 3}], [],
                             {"duration_sec": 10, "sample_fps": 2, "samples": samples})
    samples = [sample(i/2, 1 if i < 10 else 3) for i in range(20)]
    result = match_slots_to_video([{"id": "1", "required_duration_sec": 3}], [],
                                 {"duration_sec": 10, "samples": samples}, min_tier=3)
    assert result[0]["video"]["selection_metrics"]["tier"] == 3


@pytest.mark.parametrize("require_speaking", [True, False])
@pytest.mark.parametrize("pose", [False, True])
@pytest.mark.parametrize("kind", ["animation", "live_action"])
def test_scenery_and_false_pose_cannot_fill_clips_or_intro(require_speaking, pose, kind):
    # A still landscape and a false full-body pose on a screen both used to pass.
    samples = [{**sample(i / 2, 5), "person_detected": pose,
                "person_visible": int(pose), "shot_scale": "full_body" if pose else None}
               for i in range(20)]
    for slot_id in ("12", "intro"):
        with pytest.raises(ValueError, match="visible person throughout"):
            match_slots_to_video([{"id": slot_id, "required_duration_sec": 1}], [],
                                 {"duration_sec": 10, "samples": samples, "source_kind": kind},
                                 require_speaking=require_speaking)


@pytest.mark.parametrize("missing", [0, 1, 2])
def test_face_required_at_both_brackets_and_inside_cut(missing):
    samples = [sample(i / 2, 1) for i in range(3)]
    samples[missing].update(face_detected=False, face_visible=0)
    # Even strong aggregate scores cannot hide a missing subject at one sample.
    metrics = {"face_visible": .9995, "clear_face": .9995, "speaking": .9,
               "person_visible": 1, "full_body_visible": 1, "sample_coverage": 1}
    assert shot_tier(metrics, samples, 0, 2) is None


@pytest.mark.parametrize("min_tier", [0, 5])
def test_legacy_tier_cannot_disable_person_requirement(min_tier):
    with pytest.raises(ValueError, match="scenery is not allowed"):
        match_slots_to_video([{"id": "1", "required_duration_sec": 1}], [],
                             {"duration_sec": 10}, min_tier=min_tier)


def test_opencv_diagnostic_faces_cannot_confirm_people():
    samples = [{**sample(i / 2, 1), "backend": "opencv-haar"} for i in range(20)]
    with pytest.raises(ValueError, match="Not enough qualifying"):
        match_slots_to_video([{"id": "1", "required_duration_sec": 1}], [],
                             {"duration_sec": 10, "samples": samples})


def body_landmarks():
    points = [NS(x=.5, y=.5, visibility=0, presence=0) for _ in range(33)]
    positions = {0: (.5, .1), 11: (.35, .3), 12: (.65, .3),
                 13: (.25, .5), 14: (.75, .5), 15: (.2, .7), 16: (.8, .7),
                 23: (.4, .65), 24: (.6, .65), 25: (.4, .8), 26: (.6, .8),
                 27: (.4, .95), 28: (.6, .95)}
    for i, (x, y) in positions.items():
        points[i] = NS(x=x, y=y, visibility=.9, presence=.9)
    return points


@pytest.mark.parametrize("legs,scale", [(True, "full_body"), (False, "upper_body")])
def test_pose_scale_requires_actual_leg_evidence(legs, scale):
    points = body_landmarks()
    if not legs:
        for i in (25, 26, 27, 28):
            points[i].visibility = 0
    info = pose_metrics(points, 640, 400)
    assert info["person_detected"] and info["pose_confirmed"] and info["shot_scale"] == scale
    assert not pose_metrics([], 640, 400)["person_detected"]


@pytest.mark.parametrize("fault", ["flat_screen", "tiny_vehicle", "missing_torso", "four_points"])
def test_pose_rejects_scenery_like_geometry(fault):
    points = body_landmarks()
    if fault == "flat_screen":
        for p in points:
            p.y = .5 + (p.y - .5) * .04
    elif fault == "tiny_vehicle":
        for p in points:
            p.x = .5 + (p.x - .5) * .08
            p.y = .5 + (p.y - .5) * .08
    elif fault == "missing_torso":
        points[23].presence = .1
    else:
        points = points[:4]
    assert not pose_metrics(points, 640, 360)["pose_confirmed"]


def test_confirmed_upper_body_can_fill_without_readable_face():
    samples = [{**sample(i / 2, 3), "shot_scale": "upper_body"} for i in range(20)]
    result = match_slots_to_video([{"id": "1", "required_duration_sec": 3}], [],
                                 {"duration_sec": 10, "samples": samples})
    assert result[0]["video"]["selection_metrics"]["tier"] == 3


@pytest.mark.parametrize("sample_fps", [6, 10])
def test_real_frame_timestamps_do_not_look_like_missing_samples(sample_fps):
    fps = 24000 / 1001
    timestamps = np.ceil(np.arange(0, 3, 1 / sample_fps) * fps) / fps
    samples = [sample(float(t), 1) for t in timestamps]
    result = match_slots_to_video([{"id": "1", "required_duration_sec": 1.56}], [],
        {"duration_sec": 3, "fps": fps, "sample_fps": sample_fps, "samples": samples})
    assert result[0]["video"]["selection_metrics"]["sample_coverage"] == pytest.approx(1)


def test_frame_tolerance_does_not_fill_a_real_hole_in_analysis():
    fps = 24000 / 1001
    timestamps = np.ceil(np.arange(0, 3, .1) * fps) / fps
    samples = [sample(float(t), 1) for t in timestamps if not .7 < t < 1.3]
    with pytest.raises(ValueError, match="Not enough qualifying"):
        match_slots_to_video([{"id": "1", "required_duration_sec": 2.5}], [],
            {"duration_sec": 3, "fps": fps, "sample_fps": 10, "samples": samples})


def test_large_scene_change_is_not_marked_stable():
    import cv2
    from autoedit.video.analyzer import _visual_quality
    _, prev = _visual_quality(cv2, np.zeros((64, 64, 3), dtype=np.uint8), {})
    quality, _ = _visual_quality(cv2, np.full((64, 64, 3), 255, dtype=np.uint8), {}, prev)
    assert quality["activity_score"] > .25


def test_pose_model_change_invalidates_video_cache(tmp_path, monkeypatch):
    from test_talking_faces import make_video
    from autoedit.video.analyzer import analyze_video
    import autoedit.video.pose as pose
    model = tmp_path / "pose.task"
    monkeypatch.setattr(pose, "MODEL_PATH", model)
    path = make_video(tmp_path)
    first = analyze_video(path, backend="opencv", cache_dir=tmp_path / "cache")
    model.write_bytes(b"model-present")
    second = analyze_video(path, backend="opencv", cache_dir=tmp_path / "cache")
    from autoedit.video.analyzer import ANALYSIS_VERSION
    assert first["cache_identity"]["version"] == ANALYSIS_VERSION
    assert first["cache_identity"]["pose_model_mtime"] is None
    assert second["cache_identity"]["pose_model_mtime"] == model.stat().st_mtime_ns
    assert all("person_detected" in s for s in second["samples"])
