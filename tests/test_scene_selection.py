import numpy as np
import pytest

from autoedit.match.matcher import match_slots_to_video
from autoedit.audio.selection import select_audio_for_source
from autoedit.video.analyzer import _visual_quality, analyze_video


def slots(count, seconds=3):
    return [{"id": str(i), "required_duration_sec": seconds} for i in range(count)]


def test_prefers_clear_closeups_over_category_only_match():
    samples = [{"time_sec": t / 2, "face_visible": 1, "clear_face": 1, "speaking": 1, "closeup_score": 1 if t >= 30 else 0.1,
                "mouth_clarity": 1 if t >= 30 else 0.1} for t in range(60)]
    video = {"duration_sec": 30, "samples": samples, "segments": [
        {"start_sec": 0, "end_sec": 10, "category": "A"},
        {"start_sec": 15, "end_sec": 30, "category": "E"}]}
    match = match_slots_to_video(slots(1), [{"category": "A"}], video)[0]
    assert match["video"]["in_sec"] >= 15
    assert match["video"]["selection_metrics"]["mouth_clarity"] == 1


def test_whole_cut_quality_beats_single_good_frame():
    samples = [{"time_sec": t / 2, "face_visible": 1, "clear_face": 1, "speaking": 1, "mouth_clarity": 1 if t == 8 or t >= 40 else 0,
                "closeup_score": 1 if t == 8 or t >= 40 else 0} for t in range(60)]
    result = match_slots_to_video(slots(1, 5), [], {"duration_sec": 30, "samples": samples})
    assert result[0]["video"]["in_sec"] >= 20


def test_spreads_source_ranges_and_enforces_gap_without_changing_lengths():
    video = {"duration_sec": 100,
             "samples": [{"time_sec": i / 2, "face_visible": 1} for i in range(200)]}
    result = match_slots_to_video(slots(4), [], video, min_gap_sec=5, require_speaking=False)
    ranges = sorted((m["video"]["in_sec"], m["video"]["out_sec"]) for m in result)
    assert all(abs(b - a - 3) < 1e-8 for a, b in ranges)
    assert all(b[0] - a[1] >= 5 - 1e-8 for a, b in zip(ranges, ranges[1:]))
    assert ranges[-1][1] - ranges[0][0] > 60


def test_gap_is_never_silently_reduced_and_unanalyzed_footage_is_excluded():
    with pytest.raises(ValueError, match="Not enough analyzed"):
        match_slots_to_video(slots(3, 5), [], {"duration_sec": 100, "analyzed_seconds": 20}, min_gap_sec=5, require_speaking=False)
    video = {"duration_sec": 100, "analyzed_seconds": 20,
             "samples": [{"time_sec": i / 2, "face_visible": 1} for i in range(40)]}
    result = match_slots_to_video(slots(2), [], video, min_gap_sec=5, require_speaking=False)
    assert all(m["video"]["out_sec"] <= 20 for m in result)


def test_long_windows_are_reserved_before_short_slots_without_reordering_audio():
    from test_shot_tiers import sample
    video = {'duration_sec': 12, 'samples': [
        sample(i / 2, 1 if i <= 9 else 2 if i >= 18 else 5) for i in range(24)]}
    targets = [{'id': 'short', 'required_duration_sec': 1}, {'id': 'long', 'required_duration_sec': 4}]
    audio = [{'path': 'first.wav'}, {'path': 'second.wav'}]
    matches = match_slots_to_video(targets, audio, video, min_gap_sec=1)
    assert [m['audio'] for m in matches] == audio
    assert [m['video']['used_duration_sec'] for m in matches] == [1, 4]
    assert matches[0]['video']['in_sec'] >= 9
    assert matches[1]['video']['out_sec'] <= 4.5


def test_golden_sonic_reserves_all_lengths_with_original_gap_and_transitions():
    import json
    from pathlib import Path
    folder = Path(r'D:\golden\290926')
    if not (folder / 'video_analysis.json').is_file():
        pytest.skip('Local Golden regression analysis unavailable')
    info = json.loads((folder / 'premiere_inspect.json').read_text(encoding='utf-8'))
    video = json.loads((folder / 'video_analysis.json').read_text(encoding='utf-8'))
    from autoedit.video.analyzer import ANALYSIS_VERSION
    if (video.get('cache_identity') or {}).get('version') != ANALYSIS_VERSION:
        pytest.skip('Golden analysis must be regenerated with current multi-face detector')
    matches = match_slots_to_video(info['slots'], [], video, min_gap_sec=3)
    assert len(matches) == 14
    assert info['intro_slot'] is None
    for slot, match in zip(info['slots'], matches):
        cut = match['video']
        assert cut['out_sec'] - cut['in_sec'] == pytest.approx(slot['required_duration_sec'])
        assert cut['selection_metrics']['transition_safe']
        assert cut['selection_metrics']['transition_count'] == 0
    ranges = sorted((m['video']['in_sec'], m['video']['out_sec']) for m in matches)
    assert all(right[0] - left[1] >= 3 - 1e-8 for left, right in zip(ranges, ranges[1:]))


def test_center_fallback_is_not_scored_as_a_real_closeup():
    import cv2
    frame = np.random.default_rng(3).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    q, _ = _visual_quality(cv2, frame, {"face_detected": False, "face": [0, 0, 100, 100]})
    assert q["face_visible"] == q["closeup_score"] == q["mouth_clarity"] == 0
    det = {"face_detected": True, "face": [10, 10, 80, 80], "backend": "mediapipe"}
    sharp, _ = _visual_quality(cv2, frame, det)
    blurred, _ = _visual_quality(cv2, cv2.GaussianBlur(frame, (31, 31), 8), det)
    assert sharp["mouth_clarity"] > blurred["mouth_clarity"]


def test_video_analysis_emits_quality_samples(tmp_path):
    import cv2
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    if not writer.isOpened():
        pytest.skip("MJPG writer unavailable")
    for _ in range(20):
        writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    info = analyze_video(path, sample_fps=2, backend="opencv")
    assert len(info["samples"]) == 4
    assert all(s["face_visible"] == 0 for s in info["samples"])


def test_animation_neutral_policy_does_not_treat_unknown_as_neutral():
    items = [{"name": "cry.wav", "category": "CRY"},
             {"name": "voice.wav", "category": "UNCLEAR"},
             {"name": "hi.wav", "category": "HI"},
             {"name": "calm.wav", "category": "NEUTRAL"}]
    chosen = select_audio_for_source(items, "animation")
    assert [i["name"] for i in chosen] == ["hi.wav"]
    assert all(i["audio_style"] == "NEUTRAL" for i in chosen)
    assert select_audio_for_source(items, "live_action") == items
    with pytest.raises(ValueError, match="requires neutral"):
        select_audio_for_source(items[:2], "animation")


@pytest.mark.parametrize("source_word,stem", [(None, "what"), ("What", "what"), ("I", "ai"), ("know", "no2")])
def test_animation_pipeline_prioritizes_source_word_then_neutral(tmp_path, monkeypatch, source_word, stem):
    import wave
    import autoedit.pipeline as pipeline
    from test_nested_fills import fixture_project
    neutral = tmp_path / "hi.wav"
    reaction = tmp_path / (stem+".wav")
    for path in (neutral, reaction):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(8000)
            wav.writeframes((np.sin(np.arange(1600) * 0.2) * 8000).astype(np.int16).tobytes())
    monkeypatch.setattr(pipeline, "require_inputs", lambda **kw: None)
    monkeypatch.setattr(pipeline, "refine_selected_cuts", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "write_audio_review", lambda *a, **kw: None)
    def source_speech(path, ranges, *args, **kwargs):
        assert ranges == [(0.0, 30.0)]  # Scan before choosing final cuts.
        return {"has_audio": True, "words": [{"word": source_word, "start": 6.13,
                 "end": 6.3, "prob": .95}] if source_word else []}
    monkeypatch.setattr(pipeline, "transcribe_source", source_speech)
    monkeypatch.setattr(pipeline, "prepare_video_only_source", lambda *a, **kw: "video-only.mov")
    monkeypatch.setattr(pipeline, "inspect_sampler_tracks", lambda path: {
        "path": path, "tracks": [{"index": 1, "name": "Sampler Track 01"}]})
    monkeypatch.setattr(pipeline, "analyze_audio_dir", lambda *a, **kw: [
        {"path": str(neutral), "duration_sec": 0.2, "category": "VOWEL_A"},
        {"path": str(reaction), "duration_sec": 0.2, "category": "WHAT"}])
    monkeypatch.setattr(pipeline, "analyze_video", lambda *a, **kw: {"duration_sec": 30, "segments": [], "samples": [{"time_sec": i / 2, "face_visible": 1, "clear_face": 1, "speaking": 1} for i in range(60)]})
    result = pipeline.run_pipeline({"job": {"output_dir": str(tmp_path / "out")},
        "cubase": {"project": "test.cpr"},
        "premiere": {"project": str(fixture_project(tmp_path)), "source_media": "source.mp4"},
        "video": {"source_kind": "animation", "characters": {"enabled": False}}})
    plan = result["plan"]
    assert len(plan["slots"]) == 1
    assert len(plan["cubase_slots"]) == 1
    assert all(s["audio"]["path"] == str(reaction if source_word else neutral) for s in plan["cubase_slots"])
    expected = ("source-audio-pronunciation" if source_word in {"know", "I"} else
                "source-audio-whisper-exact-word" if source_word else "neutral-no-readable-viseme")
    assert all(s["audio_match"]["method"] == expected for s in plan["cubase_slots"])
    if source_word:
        assert plan["slots"][0]["video"]["in_sec"] == 6.13
        assert all(s["audio"]["placement_offset_sec"] == 0 for s in plan["cubase_slots"])
    if not source_word:
        assert all(s["audio"]["audio_style"] == "NEUTRAL" for s in plan["cubase_slots"])
    assert plan["project"]["import_mixdown"] is False
    with wave.open(plan["cubase_slots"][0]["cubase"]["sample_path"], "rb") as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        assert np.abs(pcm).max() > 100
