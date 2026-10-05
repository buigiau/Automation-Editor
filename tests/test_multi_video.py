from copy import deepcopy

import pytest

from autoedit.config import apply_inputs, load_config, source_paths
from autoedit.match.matcher import match_slots_to_video
from autoedit.premiere.adapter import apply_payload
from autoedit.video.pool import combine_video_infos, local_cut, localize_plan, span_for_cut


def footage(path, duration=5, analyzed=None):
    length = analyzed or duration
    return {"path": path, "duration_sec": duration, "analyzed_seconds": length,
            "sample_fps": 2, "fps": 25, "segments": [],
            "samples": [{"time_sec": i / 2, "face_visible": 1}
                        for i in range(int(length * 2))]}


def targets(count=2, duration=4):
    return [{"id": str(i), "nested_sequence": str(i), "nested_sequence_uid": f"uid-{i}",
             "required_duration_sec": duration} for i in range(count)]


def test_runtime_accepts_multiple_sources_and_deduplicates_physical_paths(tmp_path):
    first, second = tmp_path / "first film.mp4", tmp_path / "second.mp4"
    cfg = load_config(None)
    apply_inputs(cfg, video=[first, second, first.parent / "." / first.name])
    assert cfg["premiere"]["source_media"] == [str(first), str(second)]
    assert source_paths(str(first)) == [str(first)]
    from autoedit.cli import build_parser
    assert build_parser().parse_args(["run", "--video", str(first), "--video", str(second)]).video == [str(first), str(second)]


def test_validation_checks_every_selected_video(tmp_path):
    from autoedit.validate import validate_inputs
    good = tmp_path / "good.mp4"
    good.touch()
    bad = tmp_path / "bad.txt"
    bad.touch()
    errors = validate_inputs(source_video=[good, bad, tmp_path / "missing.mov"])
    assert any("got .txt" in e for e in errors)
    assert any("missing.mov" in e for e in errors)
    assert not any("Source video is not selected" in e for e in errors)


def test_pool_fills_slots_when_each_video_alone_is_insufficient():
    infos = [footage("first.mp4"), footage("second.mp4")]
    originals = deepcopy(infos)
    for info in infos:
        with pytest.raises(ValueError, match="Not enough"):
            match_slots_to_video(targets(), [], info, min_gap_sec=5)
    pool = combine_video_infos(infos, 5)
    matches = match_slots_to_video(targets(), [], pool, min_gap_sec=5)
    cuts = [local_cut(pool, m["video"]) for m in matches]
    assert {c["source_video"] for c in cuts} == {"first.mp4", "second.mp4"}
    assert all(0 <= c["in_sec"] < c["out_sec"] <= 5 for c in cuts)
    assert all(c["out_sec"] - c["in_sec"] == pytest.approx(4) for c in cuts)
    assert infos == originals
    with pytest.raises(ValueError, match="Not enough"):
        match_slots_to_video(targets(1, 7), [], pool)


def test_pool_respects_gaps_per_source_and_analysis_limit():
    pool = combine_video_infos([footage("a.mp4", 100, 12), footage("b.mp4", 100, 12)], 3)
    matches = match_slots_to_video(targets(4), [], pool, min_gap_sec=3)
    cuts = [local_cut(pool, m["video"]) for m in matches]
    for source in ("a.mp4", "b.mp4"):
        ranges = sorted((c["in_sec"], c["out_sec"]) for c in cuts if c["source_video"] == source)
        assert len(ranges) == 2
        assert all(end <= 12 for start, end in ranges)
        assert ranges[1][0] - ranges[0][1] >= 3 - 1e-8


@pytest.mark.parametrize("duration,analyzed,segment_end", [
    (188 + 1 / 24, None, 188.167),  # Last sample interval exceeds the real video end.
    (10, 2.0, 2.167),  # A configured analysis limit also bounds metadata.
    (10, 5 / 6, 0.834),  # Millisecond rounding must not expand the source span.
])
@pytest.mark.parametrize("count", [1, 3])
def test_cached_tail_segments_localize_within_each_source(duration, analyzed, segment_end, count):
    infos = [footage(f"source-{i}.mp4", duration, analyzed) for i in range(count)]
    limit = analyzed or duration
    for info in infos:
        info["segments"] = [{"id": "tail", "category": "NEUTRAL",
                             "start_sec": limit - 0.5, "end_sec": segment_end,
                             "duration_sec": segment_end - limit + 0.5}]
    originals = deepcopy(infos)
    pool = combine_video_infos(infos, 2)
    plan = {"project": {}, "video_analysis_meta": {}, "video_segments": pool["segments"]}
    localize_plan(plan, pool, {info["path"]: info["path"] + ".mov" for info in infos})
    assert len(plan["video_segments"]) == count
    for i, segment in enumerate(plan["video_segments"]):
        assert segment["end_sec"] <= limit + 1e-8
        assert segment["end_sec"] == pytest.approx(limit)
        assert segment["duration_sec"] == pytest.approx(segment["end_sec"] - segment["start_sec"])
        if count > 1:
            assert segment["source_video"] == infos[i]["path"]
    assert infos == originals


def test_segment_clamping_does_not_allow_invalid_selected_cuts():
    pool = combine_video_infos([footage("a.mp4"), footage("b.mp4")], 2)
    for cut in ({"in_sec": 4, "out_sec": 5.1},
                {"in_sec": 4, "out_sec": pool["source_spans"][1]["offset_sec"] + 1}):
        with pytest.raises(ValueError, match="crosses source videos"):
            span_for_cut(pool, cut)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("word_matched", [False, True])
def test_pipeline_preserves_sources_through_lip_checks_audio_and_premiere(tmp_path, monkeypatch, strict, word_matched):
    import autoedit.pipeline as pipeline
    slots = targets()
    scenes = [{**s, "index": i, "first_start_sec": i * 4 + 1, "last_end_sec": i * 4 + 5,
               "instances": [{"start_sec": i * 4 + 1, "end_sec": i * 4 + 5}]}
              for i, s in enumerate(slots)]
    inspect = {"slots": slots, "scene_slots": scenes, "intro_slot": None}
    monkeypatch.setattr(pipeline, "require_inputs", lambda **kw: None)
    monkeypatch.setattr(pipeline, "inspect_prproj", lambda *a, **kw: inspect)
    monkeypatch.setattr(pipeline, "inspect_sampler_tracks", lambda path: {
        "path": path, "tracks": [{"index": i+1, "name": f"Sampler Track {i+1}"} for i in range(2)]})
    audio = [{"path": "hi.wav", "name": "hi.wav", "category": "HI", "group": "NEUTRAL",
              "duration_sec": 1, "phonetics": {"action": "SPEECH", "visemes": ["A", "E"]}}]
    if word_matched:
        audio = [{"path": name+".wav", "name": name+".wav", "category": name.upper(),
                  "group": "NEUTRAL", "duration_sec": .3,
                  "phonetics": {"action": "SPEECH", "visemes": ["A", "E"]}}
                 for name in ("ai", "what")]
    monkeypatch.setattr(pipeline, "analyze_audio_dir", lambda *a, **kw: audio)
    monkeypatch.setattr(pipeline, "expand_match_units", lambda *a: [])

    def analyze(path, **kwargs):
        info = footage(path)
        info["segments"] = [{"category": "NEUTRAL", "start_sec": 4,
                             "end_sec": 5.167, "duration_sec": 1.167}]
        for i, sample in enumerate(info["samples"]):
            sample.update(clear_face=1, speaking=1, lip_aperture=.05 if i % 2 else .3,
                          lip_width_ratio=.5)
        return info

    monkeypatch.setattr(pipeline, "analyze_video", analyze)
    transcribed, refined = [], []

    def transcribe(path, ranges, *a, **kw):
        transcribed.append((path, ranges))
        if word_matched:
            return {"has_audio": True, "words": [{"word": "I" if path == "first.mp4" else "What",
                                                    "start": .5, "end": .8, "prob": .95}]}
        return {"words": [], "has_audio": False}

    def refine(path, selected, *a, **kw):
        refined.append((path, selected))
        assert all(0 <= s["video"]["in_sec"] < s["video"]["out_sec"] <= 5 for s in selected)
        from test_audio_onsets import lip_samples
        return {'samples': [sample for s in selected for sample in lip_samples(s['video']['in_sec'],
            s['video']['out_sec'], 'O' if word_matched and path == 'second.mp4' else 'A')]}

    monkeypatch.setattr(pipeline, "transcribe_source", transcribe)
    monkeypatch.setattr(pipeline, "refine_selected_cuts", refine)
    monkeypatch.setattr(pipeline, "prepare_video_only_source", lambda path, *a, **kw: path+".mov")
    monkeypatch.setattr(pipeline, "apply_cubase_plan", lambda *a, **kw: {})
    reviews = []
    monkeypatch.setattr(pipeline, "write_audio_review", lambda plan, out, speech: reviews.append(deepcopy(plan)))
    result = pipeline.run_pipeline({"job": {"output_dir": str(tmp_path)},
        "premiere": {"project": "template.prproj", "source_media": ["first.mp4", "second.mp4"]},
        "cubase": {"project": "template.cpr"},
        "video": {"require_lip_motion": strict, "source_gap_sec": 5, "characters": {"enabled": False}}})
    plan = result["plan"]
    assert transcribed == [("first.mp4", [(0, 5)]), ("second.mp4", [(0, 5)])]
    assert {path for path, selected in refined} == {"first.mp4", "second.mp4"}
    assert {s["source_video"] for s in plan["slots"]} == {"first.mp4", "second.mp4"}
    assert {s["video"]["source_video"] for s in plan["cubase_slots"]} == {"first.mp4", "second.mp4"}
    assert {s["source_video"] for s in plan["video_segments"]} == {"first.mp4", "second.mp4"}
    assert all(s["end_sec"] <= 5 for s in plan["video_segments"])
    assert all(s["video"]["out_sec"] <= 5 for s in reviews[0]["cubase_slots"])
    if word_matched:
        for slot in plan["cubase_slots"]:
            expected = "I" if slot["video"]["source_video"] == "first.mp4" else "What"
            assert slot["audio_match"]["source_word"] == expected
            assert slot["audio_match"]["source_word_start_sec"] == pytest.approx(.5)
            assert slot["video"]["source_sound"]["start"] == pytest.approx(.5)
    payload = apply_payload(plan)
    for action in payload["fill_nested_sequences"]:
        assert action["video_only_source"] == action["source_media"]+".mov"
        assert action["source_duration_sec"] == 5
        assert action["out_sec"] <= 5


def test_intro_and_word_times_localize_without_changing_timeline():
    pool = combine_video_infos([footage("a.mp4"), footage("b.mp4")], 5)
    offset = pool["source_spans"][1]["offset_sec"]
    cut = {"in_sec": offset+1, "out_sec": offset+4,
           "source_sound": {"start": offset+1, "end": offset+1.2}}
    slot = {"video": cut, "premiere": {"timeline_start_sec": 20},
            "audio_match": {"source_word_start_sec": offset+1}}
    plan = {"project": {}, "video_analysis_meta": {}, "slots": [deepcopy(slot)],
            "cubase_slots": [deepcopy(slot)], "intro_fill": {
                "in_sec": offset, "out_sec": offset+1, "end_sec": 1, "overwrite_at_sec": 0}}
    localize_plan(plan, pool, {"a.mp4": "a.mov", "b.mp4": "b.mov"})
    assert plan["slots"][0]["video"]["source_sound"]["start"] == 1
    assert plan["slots"][0]["premiere"]["timeline_start_sec"] == 20
    assert plan["cubase_slots"][0]["audio_match"]["source_word_start_sec"] == 1
    assert plan["intro_fill"]["in_sec"] == 0
    assert plan["intro_fill"]["end_sec"] == 1
    assert plan["intro_fill"]["video_only_source"] == "b.mov"


def test_rematch_saved_pool_uses_correct_source_word_and_keeps_local_cuts(monkeypatch):
    from autoedit.audio.rematch import rematch_fixed_slots
    pool = combine_video_infos([footage("a.mp4"), footage("b.mp4")], 5)
    cut = {"source_video": "b.mp4", "in_sec": 1, "out_sec": 4}
    plan = {"cubase_slots": [{"premiere": {"nested_sequence_uid": "uid"}, "video": cut}]}
    received = []

    def choose(candidates, video, info, used, recent):
        received.append(video)
        return {"path": "hi.wav"}, {"source_word_start_sec": video["in_sec"]}

    monkeypatch.setattr("autoedit.audio.rematch.choose_audio", choose)
    result = rematch_fixed_slots(plan, [], pool)
    assert received[0]["in_sec"] == pool["source_spans"][1]["offset_sec"]+1
    assert result["cubase_slots"][0]["video"] == cut
    assert result["cubase_slots"][0]["audio_match"]["source_word_start_sec"] == 1


def test_pooled_face_track_identifiers_do_not_collide():
    infos = [footage("a.mp4"), footage("b.mp4")]
    for info in infos:
        for sample in info["samples"]:
            sample.update(track_id="track-1", character_id="character-1", shot_id=0,
                          faces=[{"track_id": "track-1", "character_id": "character-1", "shot_id": 0}])
    pool = combine_video_infos(infos, 0)
    first, second = pool["samples"][0], pool["samples"][10]
    assert first["track_id"] != second["track_id"]
    assert first["faces"][0]["track_id"] == first["track_id"]
    assert second["faces"][0]["character_id"] == second["character_id"]
