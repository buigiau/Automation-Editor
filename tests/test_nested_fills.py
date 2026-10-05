import gzip
from pathlib import Path

import pytest

from autoedit.match.matcher import match_slots_to_video
from autoedit.plan.generator import build_edit_plan
from autoedit.premiere.adapter import apply_payload
from autoedit.premiere.prproj import inspect_prproj


def fixture_project(tmp_path):
    # Two sequences share a name. Only the one referenced through VideoSequenceSource is a slot.
    xml = '''<Project>
      <Sequence ObjectUID="main"><Name>PJ 5 - demo</Name><TrackGroups><Pair><Second ObjectRef="1"/></Pair></TrackGroups></Sequence>
      <VideoTrackGroup ObjectID="1"><Track ObjectRef="2"/></VideoTrackGroup>
      <VideoClipTrack ObjectID="2"><Index>1</Index><TrackItems><Item ObjectRef="3"/><Item ObjectRef="4"/></TrackItems></VideoClipTrack>
      <VideoClipTrackItem ObjectID="3"><TrackItem><End>254016000000</End></TrackItem><SubClip ObjectRef="5"/></VideoClipTrackItem>
      <VideoClipTrackItem ObjectID="4"><TrackItem><Start>254016000000</Start><End>508032000000</End></TrackItem><SubClip ObjectRef="5"/></VideoClipTrackItem>
      <SubClip ObjectID="5"><Clip ObjectRef="6"/><MasterClip ObjectURef="master"/></SubClip>
      <VideoClip ObjectID="6"><Clip><Source ObjectRef="7"/><InPoint>508032000000</InPoint><OutPoint>762048000000</OutPoint></Clip></VideoClip>
      <VideoSequenceSource ObjectID="7"><SequenceSource><Sequence ObjectURef="real"/></SequenceSource></VideoSequenceSource>
      <MasterClip ObjectUID="master"><Name>1</Name></MasterClip>
      <Sequence ObjectUID="real"><Name>1</Name><TrackGroups><Pair><Second ObjectRef="8"/></Pair></TrackGroups></Sequence>
      <VideoTrackGroup ObjectID="8"><Track ObjectRef="9"/></VideoTrackGroup>
      <VideoClipTrack ObjectID="9"><Index>0</Index><TrackItems><Item ObjectRef="10"/></TrackItems></VideoClipTrack>
      <VideoClipTrackItem ObjectID="10"><TrackItem><End>1016064000000</End></TrackItem></VideoClipTrackItem>
      <Sequence ObjectUID="unused"><Name>1</Name></Sequence>
    </Project>'''
    path = tmp_path / "test.prproj"
    path.write_bytes(gzip.compress(xml.encode()))
    return path


def test_resolves_real_nest_and_preserves_repeat_offsets(tmp_path):
    info = inspect_prproj(fixture_project(tmp_path))
    assert len(info["slots"]) == 1
    slot = info["slots"][0]
    assert slot["nested_sequence_uid"] == "real"
    assert slot["first_start_sec"] == 0
    assert slot["max_instance_duration_sec"] == 1
    assert slot["required_duration_sec"] == 4
    assert slot["occurrences"] == 2
    assert [i["source_in_sec"] for i in slot["instances"]] == [2, 2]
    assert inspect_prproj(fixture_project(tmp_path), video_track_index=0)["slots"] == []


def test_one_full_cut_per_nest_even_with_extra_or_missing_audio(tmp_path):
    slots = inspect_prproj(fixture_project(tmp_path))["slots"]
    video = {"path": "source.mp4", "duration_sec": 20,
             "samples": [{"time_sec": i / 2, "face_visible": 1} for i in range(40)], "segments": [
        {"start_sec": 19.5, "end_sec": 20, "duration_sec": 0.5, "category": "A"}]}
    for count in (0, 100):
        audio = [{"category": "A", "duration_sec": 0.1}] * count
        matches = match_slots_to_video(slots, audio, video, require_speaking=False)
        plan = build_edit_plan(config={}, slots=slots, matches=matches, video_info=video)
        payload = apply_payload(plan)
        assert len(plan["slots"]) == len(payload["fill_nested_sequences"]) == 1
        action = payload["fill_nested_sequences"][0]
        assert action["out_sec"] - action["in_sec"] == 4
        assert action["out_sec"] <= 20
        assert action["overwrite_at_sec"] == action["video_track_index"] == 0
        assert action["nested_sequence_uid"] == "real"
        assert payload["overwrite_template_clips"] == []


def test_no_short_or_overlapping_cuts():
    slots = [{"id": str(i), "required_duration_sec": 3.333333333} for i in range(3)]
    video = {"duration_sec": 10, "segments": [],
             "samples": [{"time_sec": i / 2, "face_visible": 1} for i in range(20)]}
    matches = match_slots_to_video(slots, [], video, require_speaking=False)
    ordered = sorted(matches, key=lambda m: m["video"]["in_sec"])
    for a, b in zip(ordered, ordered[1:]):
        assert a["video"]["out_sec"] <= b["video"]["in_sec"]
    with pytest.raises(ValueError, match="Not enough"):
        match_slots_to_video(slots, [], {"duration_sec": 9, "segments": []}, require_speaking=False)


def test_old_destructive_plans_rejected():
    with pytest.raises(ValueError, match="Regenerate"):
        apply_payload({"slots": [{"video": {"in_sec": 0, "out_sec": 1},
                                  "premiere": {"mode": "timeline_clip"}}]})


@pytest.mark.parametrize('intro_state', [None, 'occupied', 'empty'])
def test_pipeline_audio_count_never_changes_video_slot_count(tmp_path, monkeypatch, intro_state):
    import autoedit.pipeline as pipeline

    inspected = inspect_prproj(fixture_project(tmp_path))
    inspected['intro_slot'] = ({'id': 'intro', 'status': intro_state, 'start_sec': 0, 'end_sec': 1,
                               'required_duration_sec': 1, 'video_track_index': 1,
                               'reason': 'Existing opening content'} if intro_state else None)
    monkeypatch.setattr(pipeline, 'inspect_prproj', lambda *a, **kw: inspected)
    selections = []
    selection_arguments = []
    def select(targets, *args, **kwargs):
        selections.append(targets)
        selection_arguments.append(kwargs)
        return match_slots_to_video(targets, *args, **kwargs)
    monkeypatch.setattr(pipeline, 'match_slots_to_video', select)
    audio = [{"id": str(i), "path": f"{i}.wav", "duration_sec": 0.1, "category": "HI", "group": "NEUTRAL",
              'phonetics':{'action':'SPEECH','visemes':['A','E']}} for i in range(100)]
    monkeypatch.setattr(pipeline, "inspect_sampler_tracks", lambda path: {
        "path": path, "tracks": [{"index": 1, "name": "Sampler Track 01"}]})
    monkeypatch.setattr(pipeline, "require_inputs", lambda **kwargs: None)
    monkeypatch.setattr(pipeline, "analyze_audio_dir", lambda *args, **kwargs: audio)
    monkeypatch.setattr(pipeline, "expand_match_units", lambda items: items)
    monkeypatch.setattr(pipeline, "analyze_video", lambda *args, **kwargs: {
        "path": "source.mp4", "duration_sec": 20, "segments": [],
        "samples": [{"time_sec": i / 2, "face_visible": 1, "clear_face": 1, "speaking": 1} for i in range(40)]})
    received = []
    monkeypatch.setattr(pipeline, "refine_selected_cuts", lambda *a, **kw: {})
    monkeypatch.setattr(pipeline, "write_audio_review", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "transcribe_source", lambda *a, **kw: {"has_audio": False, "words": []})
    monkeypatch.setattr(pipeline, "prepare_video_only_source", lambda *a, **kw: "video-only.mov")
    monkeypatch.setattr(pipeline, "apply_cubase_plan", lambda plan, out: received.append(plan) or {})
    result = pipeline.run_pipeline({
        "job": {"output_dir": str(tmp_path / "output")},
        "video": {"require_lip_motion": False, "characters": {"enabled": False}},
        "cubase": {"project": "test.cpr"},
        "premiere": {"project": str(fixture_project(tmp_path)), "source_media": "source.mp4",
                     "slot_mode": "timeline_clips"},
    })
    assert len(result["plan"]["slots"]) == 1
    assert len(received[0]["cubase_slots"]) == 1
    assert len(result["plan"]["audio"]) == 1
    assert len(received[0]["cubase_slots"][0]["premiere"]["instances"]) == 2
    if intro_state == 'empty':
        assert len(selections) == 2
        assert selections[1][0]['id'] == 'intro'
        assert selection_arguments[1]['require_speaking'] is False
        fill = result['plan']['intro_fill']
        main = result['plan']['slots'][0]['video']
        assert selection_arguments[1]['excluded_ranges'] == [(main['in_sec'],main['out_sec'])]
        assert max(main['in_sec']-fill['out_sec'], fill['in_sec']-main['out_sec']) >= 5
        assert apply_payload(result['plan'])['intro_fill']
    else:
        assert len(selections) == 1  # No extra source range reserved for an intro.
        assert not result['plan'].get('intro_fill')
        assert not apply_payload(result['plan']).get('intro_fill')
        assert 'no intro will be added' in (tmp_path / 'output/run.log').read_text(encoding='utf-8')


def test_pipeline_reselects_static_dense_cut_before_assigning_any_voice(tmp_path, monkeypatch):
    import autoedit.pipeline as pipeline
    from test_characters import face
    inspected = inspect_prproj(fixture_project(tmp_path))
    monkeypatch.setattr(pipeline, 'inspect_prproj', lambda *a, **kw: inspected)
    monkeypatch.setattr(pipeline, 'require_inputs', lambda **kw: None)
    monkeypatch.setattr(pipeline, 'inspect_sampler_tracks', lambda path: {
        'path':path, 'tracks':[{'index':1,'name':'Sampler Track 01'}]})
    monkeypatch.setattr(pipeline, 'analyze_audio_dir', lambda *a, **kw: [
        {'id':'hi','path':'hi.wav','name':'hi.wav','duration_sec':1,'category':'HI',
         'group':'NEUTRAL','phonetics':{'action':'SPEECH','visemes':['A','E']}}])
    monkeypatch.setattr(pipeline, 'expand_match_units', lambda items: [])
    monkeypatch.setattr(pipeline, 'analyze_video', lambda *a, **kw: {
        'path':'source.mp4','duration_sec':30,'sample_fps':6,
        'samples':[{**face(),'time_sec':i/6,'face_visible':1,'clear_face':1,'speaking':1,
                    'lip_aperture':.05 if i%2 else .3,'lip_width_ratio':.5} for i in range(180)]})
    monkeypatch.setattr(pipeline, 'transcribe_source', lambda *a, **kw: {'has_audio':False,'words':[]})
    monkeypatch.setattr(pipeline, 'prepare_video_only_source', lambda *a, **kw: 'video-only.mov')
    monkeypatch.setattr(pipeline, 'write_audio_review', lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, 'apply_cubase_plan', lambda *a, **kw: {})
    refined = []
    def refine(path, slots, *args, **kwargs):
        cuts = [s['video'] for s in slots]
        refined.append(cuts)
        from test_audio_onsets import lip_samples
        samples = [sample for cut in cuts for sample in lip_samples(cut['in_sec'], cut['out_sec'])]
        if len(refined) == 1:
            for sample in samples:
                sample['lip_aperture'] = .2
        return {'samples':samples}
    monkeypatch.setattr(pipeline, 'refine_selected_cuts', refine)
    result = pipeline.run_pipeline({'job':{'output_dir':str(tmp_path/'output')},
        'premiere':{'project':'template.prproj','source_media':'source.mp4'},
        'cubase':{'project':'template.cpr'},
        'video':{'source_gap_sec':0,'require_lip_motion':True,'characters':{'enabled':False}}})
    assert len(refined) >= 2  # the alternative may need a further onset alignment check
    old, final = refined[0][0], result['plan']['slots'][0]['video']
    assert final['out_sec'] <= old['in_sec'] or final['in_sec'] >= old['out_sec']
    assert final['used_duration_sec'] == old['used_duration_sec'] == 4
    assert final['selection_metrics']['dense_lip_motion_verified']
    assert result['plan']['cubase_slots'][0]['audio_match']['method'] == 'filename-visemes-and-dense-lip-rhythm'


@pytest.mark.parametrize("has_main_cut", [False, True])
def test_pipeline_rejects_scenery_before_publishing_plan_or_exporting(tmp_path, monkeypatch, has_main_cut):
    import autoedit.pipeline as pipeline

    info = inspect_prproj(fixture_project(tmp_path))
    info["intro_slot"] = {"id": "intro", "status": "empty", "required_duration_sec": 1,
                          "start_sec": 0, "end_sec": 1, "video_track_index": 1}
    monkeypatch.setattr(pipeline, "inspect_prproj", lambda *a, **kw: info)
    monkeypatch.setattr(pipeline, "require_inputs", lambda **kw: None)
    monkeypatch.setattr(pipeline, "inspect_sampler_tracks", lambda *a: {
        "tracks": [{"index": 1, "name": "Sampler Track 01"}]})
    monkeypatch.setattr(pipeline, "analyze_audio_dir", lambda *a, **kw: [
        {"path": "hi.wav", "name": "hi.wav", "duration_sec": 1, "category": "HI", "group": "NEUTRAL"}])
    monkeypatch.setattr(pipeline, "expand_match_units", lambda items: [])
    monkeypatch.setattr(pipeline, "analyze_video", lambda *a, **kw: {
        "path": "source.mp4", "duration_sec": 10,
        "samples": [{"time_sec": i / 2, "face_visible": int(has_main_cut and i <= 8),
                     "person_visible": 1, "person_detected": True, "shot_scale": "full_body",
                     "activity_score": 0} for i in range(20)]})

    def unexpected(*a, **kw):
        pytest.fail("Rejected main/intro footage must not reach media export or transcription")

    monkeypatch.setattr(pipeline, "prepare_video_only_source", unexpected)
    monkeypatch.setattr(pipeline, "transcribe_source", unexpected)
    monkeypatch.setattr(pipeline, "apply_cubase_plan", unexpected)
    plan_path = tmp_path / "edit-plan.json"
    payload_path = tmp_path / "premiere_apply.json"
    plan_path.write_bytes(b"previous plan")
    payload_path.write_bytes(b"previous payload")
    with pytest.raises(ValueError, match="visible person throughout"):
        pipeline.run_pipeline({"job": {"output_dir": str(tmp_path)},
            "premiere": {"project": "template.prproj", "source_media": "source.mp4"},
            "cubase": {"project": "template.cpr"}, "video": {"source_gap_sec": 0, "characters": {"enabled": False}}})
    assert plan_path.read_bytes() == b"previous plan"
    assert payload_path.read_bytes() == b"previous payload"
    assert "FAILED" in (tmp_path / "run.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [
    r"D:\Editor\Soda Pop\soda pop\Adobe Premiere Pro Auto-Save\Soda pop_1_1--559037c5-69e4-5f0b-8300-3b611d2ce6b7-2026-09-25_06-47-09.prproj",
    r"D:\Editor\Golden\Adobe Premiere Pro Auto-Save\golden_1--f80fc78b-3e33-59a8-be37-bfe00ee618bc-2026-09-23_22-05-49.prproj",
])
def test_user_reference_projects(path):
    if not Path(path).exists():
        pytest.skip("Local reference project not installed")
    before = Path(path).read_bytes()
    info = inspect_prproj(path)
    slots = info["slots"]
    assert len(slots) == 14
    assert len({s["nested_sequence_uid"] for s in slots}) == 14
    assert [s["nested_sequence"] for s in slots] == [str(i) for i in range(1, 15)]
    for slot in slots:
        assert slot["required_duration_sec"] >= max(i["source_out_sec"] for i in slot["instances"])
    video = {"duration_sec": 1000, "segments": [],
             "samples": [{"time_sec": i / 2, "face_visible": 1} for i in range(2000)]}
    matches = match_slots_to_video(slots, [{"duration_sec": 0.1}] * 200, video, require_speaking=False)
    plan = build_edit_plan(config={}, slots=slots, matches=matches, video_info=video)
    assert len(apply_payload(plan)["fill_nested_sequences"]) == 14
    assert Path(path).read_bytes() == before
