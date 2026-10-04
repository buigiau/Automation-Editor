from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from autoedit.video.characters import (
    analyze_characters, build_tracks, cluster_tracks, cut_character_metrics,
    exposure, prominent_face, rank_characters, settings,
)
from autoedit.match.matcher import match_slots_to_video
from autoedit.premiere.adapter import apply_payload
from autoedit.video.refine import select_reference_face


def face(box=None):
    return {"face": box or [30, 20, 80, 80], "mouth_box": [50, 65, 40, 20],
            "backend": "mediapipe-landmarks", "face_detected": True,
            "mouth_clarity": 1, "frontal_score": 1, "closeup_score": 1,
            "category": "A", "confidence": 1}


def track(tid, vector, start, seconds, shot, indices=None):
    return {"id": tid, "vectors": [vector, vector], "shot_id": shot,
            "intervals": [[start, start + seconds]], "observations": [[i, 0] for i in (indices or [int(start * 10)])]}


def sample(time, state, rank=1):
    return {**face(), "time_sec": time, "face_visible": 1, "clear_face": 1,
            "speaking": 1, "character_status": state,
            "character_id": f"character-{rank}", "character_rank": rank,
            "identity_confidence": 1}


def video(states):
    samples = [sample(i / 2, state, 1 if state == "main" else 3) for i, state in enumerate(states)]
    return {"duration_sec": len(states) / 2, "sample_fps": 2, "samples": samples,
            "character_analysis": {"min_main_fraction": .85}}


def test_tracks_are_one_to_one_and_reset_at_transition():
    samples = [{"time_sec": i / 2, "faces": [face(), face([160, 20, 80, 80])]} for i in range(6)]
    tracks = build_tracks(samples, [1.5], 3, 2, settings())
    assert len(tracks) == 4
    assert len({f["track_id"] for f in samples[0]["faces"]}) == 2
    assert samples[0]["faces"][0]["track_id"] == samples[2]["faces"][0]["track_id"]
    assert samples[2]["faces"][0]["track_id"] != samples[3]["faces"][0]["track_id"]
    assert sum(exposure(t["intervals"]) for t in tracks) == pytest.approx(6)


def test_animation_region_duplicates_are_removed_without_losing_other_faces(monkeypatch):
    import autoedit.video.faces as faces
    created = []
    class Detector:
        def __init__(self):
            self.index = len(created)
            created.append(self)
        def detect_all(self, frame, time):
            offsets = [(0, 0), (0, 54), (128, 54), (256, 54)]
            dx, dy = offsets[self.index]
            # The same physical face appears in the whole-frame and middle view.
            return [face([200 - dx, 100 - dy, 80, 80]), face([400 - dx, 100 - dy, 80, 80])]
        def close(self):
            pass
    monkeypatch.setattr(faces, "FaceLandmarker", Detector)
    detector = faces.MultiRegionFaceLandmarker()
    try:
        found = detector.detect(np.zeros((360, 640, 3), np.uint8), 1)
        assert len(found["faces"]) == 2
        assert sorted(d["face"] for d in found["faces"]) == [[200, 100, 80, 80], [400, 100, 80, 80]]
    finally:
        detector.close()


def test_motion_safety_events_do_not_inflate_shot_count():
    samples = [{"time_sec": i / 2, "faces": [face()]} for i in range(6)]
    tracks = build_tracks(samples, [.5, 1, 1.5], 3, 2, settings(), shot_times=[1.5])
    assert len(tracks) == 4
    assert [t["shot_id"] for t in tracks] == [0, 0, 0, 1]


def test_clustering_merges_returning_identity_but_never_co_visible_tracks():
    tracks = [track("a", [1, 0], 0, 4, 0, [0]), track("b", [1, 0], 6, 4, 1, [60]),
              track("c", [1, 0], 0, 4, 0, [0])]
    groups = cluster_tracks(tracks, .2)
    assert sorted(map(len, groups)) == [1, 2]
    assert not any({"a", "c"}.issubset({t["id"] for t in g}) for g in groups)


def test_complete_link_does_not_chain_distinct_people():
    def vector(degrees):
        angle = np.deg2rad(degrees)
        return [np.cos(angle), np.sin(angle)]
    tracks = [track(str(i), vector(i * 30), i * 4, 3, i) for i in range(3)]
    groups = cluster_tracks(tracks, .2)
    assert sorted(map(len, groups)) == [1, 2]


def test_rank_uses_75_percent_and_ignores_noise_in_denominator():
    tracks = [track("a", [1, 0, 0], 0, 60, 0), track("b", [0, 1, 0], 70, 20, 1),
              track("c", [0, 0, 1], 100, 20, 2),
              {**track("bad", [1, 0, 0], 130, 30, 3), "vectors": [[1, 0, 0], [0, 1, 0]]}]
    clusters, _, achieved = rank_characters(tracks, settings(), .2)
    assert [c["duration_sec"] for c in clusters] == [60, 20, 20]
    assert [c["is_main"] for c in clusters] == [True, True, False]
    assert achieved == pytest.approx(.8)
    assert tracks[-1]["identity_status"] == "noise"


def test_short_single_shot_identity_is_not_promoted():
    tracks = [track("rare", [1, 0], 0, .5, 0)]
    clusters, _, achieved = rank_characters(tracks, settings(), .2)
    assert not clusters[0]["is_main"]
    assert achieved == 0


def test_overlapping_exposure_is_counted_once():
    assert exposure([[0, 2], [1, 3], [3, 4], [6, 7]]) == 5


def test_appearance_ensemble_counts_simultaneous_copies_once_and_keeps_tracks():
    samples=[{'faces':[face([0,0,30,30]),face([60,0,30,30])]} for _ in range(4)]
    tracks=[track('left',[1,0],0,4,0,[0,1,2,3]),
            {**track('right',[1,0],0,4,0,[0,1,2,3]),
             'observations':[[i,1] for i in range(4)]},
            track('human',[0,1],5,5,1,[])]
    clusters,_,_=rank_characters(tracks,settings(),.2,appearance_samples=samples)
    assert clusters[0]['track_ids'] == ['human']  # 5s beats two simultaneous copies lasting 4s.
    ensemble=clusters[1]
    assert set(ensemble['track_ids']) == {'left','right'}
    assert ensemble['duration_sec'] == 4
    assert ensemble['character_group'].startswith('appearance-ensemble-')
    # Ranking a cached/reused set must not discard previously grouped tracks.
    again,_,_=rank_characters(tracks,settings(),.2,appearance_samples=samples)
    assert [c['duration_sec'] for c in again] == [5,4]


def test_appearance_ensemble_rejects_duplicate_boxes_and_distinct_looks():
    for right_vector,right_box in [([1,0],[0,0,30,30]),([0,1],[60,0,30,30])]:
        samples=[{'faces':[face([0,0,30,30]),face(right_box)]} for _ in range(4)]
        tracks=[track('left',[1,0],0,4,0,[0,1,2,3]),
                {**track('right',right_vector,0,4,0,[0,1,2,3]),
                 'observations':[[i,1] for i in range(4)]}]
        clusters,_,_=rank_characters(tracks,settings(),.2,appearance_samples=samples)
        assert len(clusters) == 2
        assert not any(c.get('character_group') for c in clusters)


def test_typed_whole_character_ensemble_bridges_views_but_never_people_or_distinct_creatures():
    from autoedit.video.characters import appearance_ensembles
    samples=[{'faces':[face()]} for _ in range(4)]
    def member(tid, vector, shot, family='creature'):
        return {**track(tid,vector,shot*2,2,shot,[shot*2,shot*2+1]),
                'subject_kind':'object','subject_family':family}
    head=member('head',[1,0],0)
    body=member('body',[.9,.43589],1)
    person=member('human',[1,0],2,'person')
    distinct=member('different',[0,1],3)
    result=appearance_ensembles([[head],[body],[person],[distinct]],samples)
    assert sorted(map(len,result))==[1,1,2]
    ensemble=next(g for g in result if len(g)==2)
    assert {t['id'] for t in ensemble}=={'head','body'}
    assert head['group_evidence']=='typed-whole-character-lookalikes-ccip'
    # Pairwise bounds prevent a chain of mildly different creatures merging.
    third=member('third',[.62,.7846],4)
    result=appearance_ensembles([[head],[body],[third]],samples)
    assert sorted(map(len,result))==[1,2]
    alternate=member('alternate',[.9,-.43589],5)
    result=appearance_ensembles([[head],[body],[alternate]],samples)
    assert len(result)==1  # Both views independently match the fixed head prototype.


def test_auto_model_uses_consistency_evidence_and_respects_explicit_policy(tmp_path,monkeypatch):
    import autoedit.video.characters as chars
    import autoedit.video.embeddings as embeddings
    (tmp_path/'ccip.onnx').write_bytes(b'available appearance model')
    monkeypatch.setattr(embeddings,'MODEL_ROOT',tmp_path)
    info={'source_kind':'live_action','sample_fps':2,'duration_sec':20,
          'samples':[{'time_sec':i/2,'frame_width':640,'frame_height':360,'faces':[face()]}
                     for i in range(40)]}
    calls=[]
    def load(info,cache,config,kind,progress):
        calls.append(kind)
        tracks=[track(str(i),[1,0],i*2,2,i,list(range(i*4,i*4+4))) for i in range(8)]
        if kind == 'live_action':
            for t in tracks: t['vectors']=[[1,0],[0,1]]
        return tracks,None,{'model_sha256':'fake'},tmp_path/'model.onnx'
    monkeypatch.setattr(chars,'_embedding_tracks',load)
    auto=chars.analyze_characters(info,tmp_path/'cache')
    assert calls == ['live_action','animation']
    assert auto['source_kind'] == 'live_action'
    assert auto['character_analysis']['embedding_model'] == 'ccip'
    assert auto['character_analysis']['auto_decision']['ccip_uncertain_fraction'] == 0
    calls.clear()
    forced=chars.analyze_characters(info,tmp_path/'cache',{'embedding_model':'arcface'})
    assert calls == ['live_action']
    assert forced['character_analysis']['embedding_model'] == 'arcface'


def test_prominence_prefers_large_near_center_face():
    small_main, large_extra = face([180, 120, 40, 40]), face([100, 50, 160, 160])
    assert prominent_face([small_main, large_extra], 400, 300) is large_extra


def test_primary_subject_prefers_exposure_rank_before_other_actors_speech(monkeypatch):
    from autoedit.video.characters import _annotate_subjects
    # This isolates subject priority; temporal speech evidence has separate tests.
    monkeypatch.setattr('autoedit.video.faces.annotate_speaking', lambda *args, **kwargs: None)
    main = {**sample(0,'main',1), 'face': [0,0,60,60], 'speaking': 0}
    other = {**sample(0,'main',2), 'face': [70,0,120,120], 'speaking': 1}
    info = {'samples': [{'time_sec':0,'faces':[other,main],
                         'frame_width':400,'frame_height':300}]}
    _annotate_subjects(info,'live_action')
    assert info['samples'][0]['character_rank'] == 1
    assert not info['samples'][0]['speaking']
    assert len(info['samples'][0]['faces']) == 2


def test_main_subject_has_priority_over_better_looking_extra():
    info = video(["supporting"] * 12 + ["main"] * 12)
    for s in info["samples"][12:]:
        s.update(clear_face=0, speaking=0, mouth_clarity=.3, closeup_score=.3)
    match = match_slots_to_video([{"id": "1", "required_duration_sec": 3}], [], info)[0]["video"]
    assert match["in_sec"] >= 6
    assert match["character_selection"]["decision"] == "main"
    assert not match["character_selection"]["needs_review"]


def test_exposure_rank_precedes_closeup_tier_for_qualifying_speakers():
    info=video(['main']*24)
    for index,s in enumerate(info['samples']):
        s.update(character_id='most' if index < 12 else 'next', character_rank=1 if index < 12 else 2,
                 clear_face=.6 if index < 12 else 1., mouth_clarity=.5 if index < 12 else 1.,
                 lip_aperture=.05 if index%2 else .3, lip_width_ratio=.6)
    cut=match_slots_to_video([{'id':'1','required_duration_sec':2}],[],info,require_lip_motion=True)[0]['video']
    assert cut['character_selection']['best_rank'] == 1
    assert cut['in_sec'] < 6


def test_known_supporting_actor_precedes_unidentified_face():
    info=video(['unknown']*12+['supporting']*12)
    for s in info['samples'][12:]:
        s.update(character_rank=3, clear_face=.6, mouth_clarity=.5)
    cut=match_slots_to_video([{'id':'1','required_duration_sec':2}],[],info)[0]['video']
    assert cut['character_selection']['decision'] == 'supporting'
    assert cut['in_sec'] >= 6


def test_lower_ranked_co_visible_track_can_fill_a_cut_when_primary_keeps_switching():
    samples=[]
    for i in range(24):
        primary={**sample(i/2,'main'), 'track_id':f'switch-{i}', 'character_rank':1}
        secondary={**sample(i/2,'supporting',2),'face':[150,20,80,80],'track_id':'steady',
                   'lip_aperture':.05 if i%2 else .3,'lip_width_ratio':.6,'category':'E'}
        primary['faces']=[{**primary},secondary]
        samples.append(primary)
    info={'duration_sec':12,'sample_fps':2,'samples':samples,
          'character_analysis':{'min_main_fraction':.85}}
    cut=match_slots_to_video([{'id':'cut','required_duration_sec':3}],[],info,require_lip_motion=True)[0]['video']
    assert cut['selection_metrics']['subject_track_id'] == 'steady'
    assert cut['character_selection']['best_rank'] == 2
    assert cut['selection_metrics']['lip_motion_within_cut']
    assert cut['category'] == 'E'


def test_alternative_track_must_cover_every_frame_and_cannot_be_stitched_across_missing_observations():
    info=video(['unknown']*16)
    for i,s in enumerate(info['samples']):
        s.update(face_detected=False,face_visible=0)
        s['faces']=[] if i%3 == 2 else [{**sample(s['time_sec'],'main'),'track_id':'holes'}]
    with pytest.raises(ValueError,match='Not enough'):
        match_slots_to_video([{'id':'cut','required_duration_sec':2}],[],info)


def test_dense_refinement_uses_selected_secondary_track_instead_of_primary_face():
    from autoedit.video.refine import cut_references
    primary={**sample(1,'main'),'track_id':'primary','frame_width':100,'frame_height':100}
    selected={**face([10,10,30,40]),'track_id':'secondary','character_id':'second'}
    primary['faces']=[selected]
    slot={'video':{'in_sec':.5,'out_sec':1.5,'selection_metrics':{'subject_track_id':'secondary'}}}
    references=cut_references([primary],slot)
    found=select_reference_face([face([20,20,60,80]),face([100,100,80,80])],200,200,1,references,[1])
    assert found['track_id'] == 'secondary'
    assert found['character_id'] == 'second'
    assert primary['track_id'] == 'primary'


@pytest.mark.parametrize("state", ["unknown", "supporting"])
def test_fallback_fills_full_slots_and_marks_review(state):
    info = video(["main"] * 10 + [state] * 14)
    matches = match_slots_to_video([{"id": str(i), "required_duration_sec": 3} for i in range(2)], [], info, min_gap_sec=1)
    assert len(matches) == 2
    assert matches[0]["video"]["character_selection"]["decision"] == "main"
    assert matches[1]["video"]["character_selection"]["needs_review"]
    ranges = sorted((m["video"]["in_sec"], m["video"]["out_sec"]) for m in matches)
    assert ranges[1][0] - ranges[0][1] >= 1
    assert all(b - a == pytest.approx(3) for a, b in ranges)


def test_noise_only_cannot_fill_even_last_fallback():
    with pytest.raises(ValueError, match="Noise-only"):
        match_slots_to_video([{"id": "1", "required_duration_sec": 3}], [], video(["noise"] * 12))


def test_mixed_identity_and_body_only_are_uncertain():
    mixed = cut_character_metrics([sample(0, "main"), sample(.5, "supporting")], 0, 1, .5, .85)
    assert mixed["decision"] == "uncertain"
    assert cut_character_metrics([{"time_sec": 0}], 0, .5, .5, .85)["decision"] == "uncertain"


def test_low_embedding_confidence_is_not_accepted_as_main():
    s = sample(0, "main")
    s["identity_confidence"] = .1
    assert cut_character_metrics([s], 0, .5, .5, .85)["decision"] == "uncertain"


def test_recognized_primary_view_precedes_certain_secondary_and_keeps_review_flag():
    info=video(['main']*12+['main']*12)
    for s in info['samples'][:12]:
        s.update(identity_confidence=.4)
    for s in info['samples'][12:]:
        s.update(character_id='secondary',character_rank=2)
    cut=match_slots_to_video([{'id':'cut','required_duration_sec':2}],[],info)[0]['video']
    assert cut['in_sec'] < 6 and cut['character_selection']['best_rank']==1
    assert cut['character_selection']['decision']=='uncertain'
    assert cut['character_selection']['needs_review']


def test_primary_allocation_requires_rank_throughout_not_one_prominent_frame():
    from autoedit.match.matcher import _match_slots_to_video
    info=video(['main']*24)
    for s in info['samples']:
        s.update(character_id='secondary',character_rank=2)
    info['samples'][2].update(character_id='primary',character_rank=1)
    metrics=cut_character_metrics(info['samples'][:7],0,3,.5,.85)
    assert metrics['rank_fractions'][1]==pytest.approx(1/6)
    with pytest.raises(ValueError,match='Not enough'):
        _match_slots_to_video([{'id':'cut','required_duration_sec':3}],[],info,
                             character_level=0,character_rank_limit=1)


def test_conflicting_track_embedding_remains_uncertain():
    t = track("mixed", [1, 0], 0, 10, 0)
    t["vectors"] = [[1, 0], [1, 0], [1, 0], [0, 1]]
    cluster_tracks([t], .2)
    assert t["identity_status"] == "clustered"
    assert t["identity_confidence"] <= .4


def test_embedding_cache_reuses_decode_but_recomputes_cast_selection(tmp_path, monkeypatch):
    import autoedit.video.characters as chars
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake model fingerprint")
    config = {"arcface_model": str(model)}
    info = {"path": "source.mp4", "source_kind": "live_action", "cache_identity": {"version": 10},
            "sample_fps": 2, "duration_sec": 10,
            "samples": [{"time_sec": i / 2, "faces": [face()], "frame_width": 640, "frame_height": 360}
                        for i in range(20)]}
    calls = []
    monkeypatch.setattr(chars, "EmbeddingModel", lambda *a: None)
    def extract(*a):
        calls.append(1)
        return [track("first", [1, 0], 0, 6, 0, list(range(12))),
                track("second", [0, 1], 6, 4, 1, list(range(12, 20)))]
    monkeypatch.setattr(chars, "_extract_tracks", extract)
    first = analyze_characters(info, tmp_path / "cache", config)
    second = analyze_characters(info, tmp_path / "cache", {**config, "coverage_target": .5})
    assert len(calls) == 1
    assert [c["is_main"] for c in first["character_analysis"]["clusters"]] == [True, True]
    assert [c["is_main"] for c in second["character_analysis"]["clusters"]] == [True, False]
    assert "character_status" not in info["samples"][0]
    model.write_bytes(b"changed weights")
    analyze_characters(info, tmp_path / "cache", config)
    assert len(calls) == 2


def test_missing_model_fails_explicitly_and_disabled_filter_is_supported(tmp_path):
    info = {"sample_fps": 2, "samples": [], "source_kind": "live_action"}
    with pytest.raises(FileNotFoundError, match="Character embedding model missing"):
        analyze_characters(info, tmp_path, {"arcface_model": str(tmp_path / "absent.onnx")})
    assert analyze_characters(info, tmp_path, {"enabled": False}) is info


def test_dense_lip_refinement_tracks_chosen_face_at_new_resolution():
    ref = {"time_sec": 1, "face": [20, 30, 40, 50], "frame_width": 100, "frame_height": 100,
           "character_id": "main", "track_id": "t", "shot_id": 1}
    candidates = [face([100, 100, 90, 90]), face([40, 60, 80, 100])]
    chosen = select_reference_face(candidates, 200, 200, 1.05, [ref], [1])
    assert chosen["face"] == [40, 60, 80, 100]
    assert chosen["character_id"] == "main"
    assert not select_reference_face(candidates[:1], 200, 200, 1.05, [ref], [1])["face_detected"]


def test_payload_preserves_character_review_metadata():
    character = {"decision": "supporting", "needs_review": True, "reason": "supporting-character-fallback"}
    plan = {"project": {}, "slots": [{"id": "s", "video": {"in_sec": 0, "out_sec": 3,
        "character_selection": character}, "premiere": {"nested_sequence_uid": "u", "duration_sec": 3}}]}
    assert apply_payload(plan)["fill_nested_sequences"][0]["character_selection"] == character


@pytest.mark.parametrize("config", [{"coverage_target": 0}, {"coverage_target": float("nan")},
                                     {"ccip_cosine_distance": 2}, {"enabled": "false"},
                                     {"embeddings_per_track": 1.5}])
def test_invalid_settings_are_rejected(config):
    with pytest.raises(ValueError):
        settings(config)


def test_full_resolution_crop_extraction_with_real_video_decode(tmp_path):
    import cv2
    from autoedit.video.characters import _extract_tracks
    from autoedit.video.analyzer import analyze_video
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 96))
    assert writer.isOpened()
    for _ in range(40):
        writer.write(np.full((96, 128, 3), 90, np.uint8))
    writer.release()
    info = analyze_video(path, sample_fps=2, backend="opencv")
    for s in info["samples"]:
        s["faces"] = [face()]
    seen = []
    class Model:
        def extract(self, image, *args):
            seen.append(image.shape)
            return np.array([1, 0], np.float32)
    tracks = _extract_tracks(info, settings(), Model(), None)
    assert len(tracks) == 1
    assert len(tracks[0]["vectors"]) == 8
    assert set(seen) == {(96, 128, 3)}


def test_pipeline_runs_character_filter_and_publishes_metadata(tmp_path, monkeypatch):
    import autoedit.pipeline as pipeline
    import autoedit.video.characters as chars
    from test_nested_fills import fixture_project
    model = tmp_path / "arcface.onnx"
    model.write_bytes(b"test-model")
    info = {"path": "source.mp4", "source_kind": "live_action", "sample_fps": 2,
        "duration_sec": 20, "analyzed_seconds": 20, "segments": [],
        "samples": [{**face(), "time_sec": i / 2, "faces": [face()],
                     "frame_width": 640, "frame_height": 360, "face_visible": 1}
                    for i in range(40)]}
    monkeypatch.setattr(pipeline, "require_inputs", lambda **kw: None)
    monkeypatch.setattr(pipeline, "analyze_video", lambda *a, **kw: deepcopy(info))
    monkeypatch.setattr(chars, "EmbeddingModel", lambda *a: SimpleNamespace(
        extract=lambda *a: np.array([1, 0], np.float32)))
    monkeypatch.setattr(chars, "_selected_source_frames", lambda path, times:
                        ((t, np.zeros((360, 640, 3), np.uint8)) for t in times))
    monkeypatch.setattr(pipeline, "inspect_sampler_tracks", lambda path: {"path": path,
        "tracks": [{"index": 1, "name": "Sampler Track 01"}]})
    monkeypatch.setattr(pipeline, "analyze_audio_dir", lambda *a, **kw: [{
        "path": "hi.wav", "duration_sec": .1, "category": "HI", "group": "NEUTRAL"}])
    monkeypatch.setattr(pipeline, "prepare_video_only_source", lambda *a, **kw: "video-only.mov")
    monkeypatch.setattr(pipeline, "transcribe_source", lambda *a, **kw: {"has_audio": False, "words": []})
    monkeypatch.setattr(pipeline, "write_audio_review", lambda *a: None)
    monkeypatch.setattr(pipeline, "apply_cubase_plan", lambda *a: {})
    references = []
    monkeypatch.setattr(pipeline, "refine_selected_cuts", lambda *a, **kw:
                        references.append(kw["character_samples"]) or {"samples": []})
    out = tmp_path / "out"
    result = pipeline.run_pipeline({"job": {"output_dir": str(out)},
        "premiere": {"project": str(fixture_project(tmp_path)), "source_media": "source.mp4"},
        "cubase": {"project": "test.cpr"},
        "video": {"require_lip_motion": False, "characters": {"arcface_model": str(model)}}})
    assert result["plan"]["slots"][0]["video"]["character_selection"]["decision"] == "main"
    assert references[0][0]["character_id"] == "character-001"
    assert (out / "character_analysis.json").is_file()
    assert result["plan"]["video_analysis_meta"]["character_analysis"]["clusters"][0]["is_main"]
    import json
    payload = json.loads((out / "premiere_apply.json").read_text(encoding="utf-8"))
    assert not payload["fill_nested_sequences"][0]["character_selection"]["needs_review"]
