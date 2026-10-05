from copy import deepcopy

import numpy as np
import pytest

from autoedit.video.characters import settings, cut_character_metrics
from autoedit.video.references import match_reference_tracks, reference_vectors
from autoedit.match.matcher import match_slots_to_video, InsufficientFootageError
from test_characters import track, video
from test_incremental_analysis import job


def test_optional_references_enable_identity_without_changing_auto_default():
    assert settings({'enabled':False})['enabled'] is False
    assert settings({'enabled':False, 'reference_images':['actor.png']})['enabled'] is True
    assert settings()['reference_images'] == []
    with pytest.raises(ValueError, match='list of image paths'):
        settings({'reference_images':'actor.png'})


def test_reference_matching_needs_multiple_consistent_crops_and_not_an_ensemble(monkeypatch, tmp_path):
    monkeypatch.setattr('autoedit.video.references.reference_vectors', lambda *a: ([[1.,0.]], ['actor.png']))
    tracks = [track('target',[1,0],0,2,0), track('other',[0,1],0,2,0),
              {**track('weak',[1,0],3,2,1), 'vectors':[[1,0],[0,1]]}]
    tracks[0]['character_group'] = tracks[1]['character_group'] = 'lookalike-ensemble'
    info = match_reference_tracks(tracks,'live_action',settings({'reference_images':['actor.png']}),tmp_path,
                                  {'model_sha256':'model'})
    assert info['matched_tracks'] == 1
    assert [t['reference_match'] for t in tracks] == [True,False,False]


def test_co_visible_lookalikes_are_ambiguous(monkeypatch, tmp_path):
    monkeypatch.setattr('autoedit.video.references.reference_vectors', lambda *a: ([[1.,0.]], ['actor.png']))
    tracks = [track('a',[1,0],0,2,0), track('b',[1,0],0,2,0), track('c',[1,0],0,2,0)]
    match_reference_tracks(tracks,'animation',settings({'reference_images':['actor.png']}),tmp_path,
                           {'model_sha256':'model'})
    assert all(not t['reference_match'] and t['reference_ambiguous'] for t in tracks)


def test_reference_cache_is_content_and_model_specific(tmp_path, monkeypatch):
    import cv2
    from test_characters import face
    path = tmp_path/'ảnh.png'
    cv2.imencode('.png', np.zeros((120,120,3),np.uint8))[1].tofile(str(path))
    calls = []
    class Model:
        def __init__(self,*a):
            pass
        def extract(self,*a):
            calls.append(1)
            return np.array([1.,0.],np.float32)
    class Detector:
        def __init__(self,**kw):
            pass
        def detect(self,*a):
            return {'faces':[face()]}
        def close(self):
            pass
    monkeypatch.setattr('autoedit.video.references.EmbeddingModel',Model)
    monkeypatch.setattr('autoedit.video.faces.MultiRegionFaceLandmarker',Detector)
    cfg = settings({'reference_images':[str(path)]})
    for _ in range(2):
        reference_vectors('live_action',cfg,tmp_path,{'model_sha256':'one'})
    assert len(calls) == 1
    reference_vectors('live_action',cfg,tmp_path,{'model_sha256':'two'})
    cv2.imencode('.png',np.ones((120,120,3),np.uint8))[1].tofile(str(path))
    reference_vectors('live_action',cfg,tmp_path,{'model_sha256':'two'})
    assert len(calls) == 3


def test_invalid_reference_is_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError, match='Invalid character reference'):
        reference_vectors('live_action',settings({'reference_images':['missing.png']}),tmp_path,
                          {'model_sha256':'model'})


def test_reference_priority_is_independent_of_exposure_rank_and_cannot_be_momentary():
    info = video(['main']*20)
    info['character_analysis']['reference'] = {'images':['actor.png']}
    for sample in info['samples']:
        is_target = sample['time_sec'] >= 5
        sample.update(reference_match=is_target,reference_confidence=float(is_target),
                      character_rank=3 if is_target else 1)
    slot = [{'id':'one','required_duration_sec':2}]
    info['reference_only'] = True
    cut = match_slots_to_video(slot,[],info)[0]['video']
    assert cut['in_sec'] >= 5
    assert cut['character_selection']['reference_fraction'] >= .85
    assert cut['character_selection']['best_rank'] == 3
    for sample in info['samples']:
        sample.update(reference_match=sample['time_sec']==5,reference_confidence=1.)
    with pytest.raises(InsufficientFootageError):
        match_slots_to_video(slot,[],info)
    info['reference_only'] = False
    assert match_slots_to_video(slot,[],info)[0]['video']['character_selection']['best_rank'] == 1


@pytest.mark.parametrize('present', [False, True])
def test_pipeline_waits_for_reference_then_stops_or_falls_back_after_all_sources(job, monkeypatch, present):
    import autoedit.pipeline as pipeline
    cfg, inspect, install, scanned, speech, exports, refined = job
    cfg['video']['characters']['reference_images'] = ['actor.png']
    monkeypatch.setattr('autoedit.video.objects.augment_with_objects',lambda raw,*a,**kw:raw)
    def analyze(raw,*a,**kw):
        info = deepcopy(raw)
        info['character_analysis'] = {'min_main_fraction':.85,'reference':{'images':['actor.png']}}
        for sample in info['samples']:
            match = present and sample['time_sec'] >= 6
            properties = {'character_status':'main','character_rank':1,'character_id':'actor',
                          'identity_confidence':1.,'reference_match':match,'reference_confidence':float(match)}
            sample.update(properties)
            for face in sample['faces']:
                face.update(properties)
        return info
    monkeypatch.setattr(pipeline,'analyze_characters',analyze)
    install([6,12,18])
    plan = pipeline.run_pipeline(cfg)['plan']
    fallback = plan['slots'][0]['video']['character_selection']['reference_fallback']
    if present:
        assert scanned == [('first.mp4',6),('first.mp4',12)]
        assert not fallback
        assert plan['slots'][0]['video']['character_selection']['reference_fraction'] >= .85
        assert plan['slots'][0]['video']['character_selection']['reason'] == 'reference-character'
    else:
        assert len(scanned) == 6 and scanned[-1] == ('unused.mp4',18)
        assert fallback


def test_cli_supports_multiple_characters_or_alternate_views():
    from autoedit.cli import build_parser
    assert build_parser().parse_args(['run']).character_reference is None
    args = build_parser().parse_args(['run','--character-reference','front.png','--character-reference','side.jpg'])
    assert args.character_reference == ['front.png','side.jpg']


def test_distinct_co_visible_references_are_not_lookalike_ambiguity(monkeypatch, tmp_path):
    monkeypatch.setattr('autoedit.video.references.reference_vectors',
                        lambda *a: ([[1.,0.],[0.,1.]], ['a.png','b.png']))
    tracks = [track('a',[1,0],0,2,0), track('b',[0,1],0,2,0)]
    info = match_reference_tracks(tracks,'animation',settings({'reference_images':['a.png','b.png']}),
                                  tmp_path,{'model_sha256':'model'})
    assert info['matched_tracks'] == 2
    assert len({t['reference_id'] for t in tracks}) == 2
    assert all(t['reference_match'] and not t['reference_ambiguous'] for t in tracks)


def test_track_cannot_combine_evidence_from_different_targets(monkeypatch, tmp_path):
    monkeypatch.setattr('autoedit.video.references.reference_vectors',
                        lambda *a: ([[1.,0.],[0.,1.]], ['a.png','b.png']))
    mixed = {**track('mixed',[1,0],0,2,0), 'vectors':[[1,0],[0,1]]}
    info = match_reference_tracks([mixed],'animation',settings({'reference_images':['a.png','b.png']}),
                                  tmp_path,{'model_sha256':'model'})
    assert info['matched_tracks'] == 0
    assert mixed['reference_id'] is None


def test_alternate_views_merge_but_group_image_faces_never_merge():
    from autoedit.video.references import _reference_groups
    vectors = np.asarray([[1.,0.],[1.,0.],[1.,0.]])
    catalog = {'templates':[
        {'id':'a','image_index':0,'image_sha256':'group'},
        {'id':'b','image_index':0,'image_sha256':'group'},
        {'id':'c','image_index':1,'image_sha256':'portrait'}]}
    groups = _reference_groups(vectors,catalog,.2)
    assert sorted(len(g['templates']) for g in groups) == [1,2]
    assert not any({'a','b'} <= {t['id'] for t in g['templates']} for g in groups)
    # Importing the same group through another path cannot bypass co-occurrence.
    catalog['templates'].append({'id':'b','image_index':2,'image_sha256':'group'})
    groups = _reference_groups(np.asarray([[1.,0.]]*4),catalog,.2)
    assert not any({'a','b'} <= {t['id'] for t in g['templates']} for g in groups)


def test_group_image_extracts_separate_crops_and_caches_all_targets(tmp_path, monkeypatch):
    import cv2
    from test_characters import face
    path = tmp_path/'group.png'
    cv2.imencode('.png',np.zeros((180,240,3),np.uint8))[1].tofile(str(path))
    faces = [{**face([20,40,80,80]),'target':'a'}, {**face([110,40,80,80]),'target':'b'}]
    extracted = []
    class Model:
        def __init__(self,*a):
            pass
        def extract(self,frame,subject,width,height,other_faces):
            if len(other_faces) > 1:
                return None  # Expanded head crop would include the neighbour.
            extracted.append((subject['target'],width,subject['face']))
            return np.asarray([1.,0.] if subject['target']=='a' else [0.,1.])
    class Detector:
        def __init__(self,**kw):
            pass
        def detect(self,*a):
            return {'faces':faces}
        def close(self):
            pass
    monkeypatch.setattr('autoedit.video.references.EmbeddingModel',Model)
    monkeypatch.setattr('autoedit.video.faces.MultiRegionFaceLandmarker',Detector)
    cfg = settings({'reference_images':[str(path)]})
    vectors, catalog = reference_vectors('animation',cfg,tmp_path,{'model_sha256':'one'})
    assert len(vectors) == len(catalog['templates']) == 2
    assert len({t['id'] for t in catalog['templates']}) == 2
    assert all(width < 240 for _,width,_ in extracted)
    assert catalog['skipped_regions'] == []
    warm_vectors, warm_catalog = reference_vectors('animation',cfg,tmp_path,{'model_sha256':'one'})
    assert vectors == warm_vectors and catalog == warm_catalog
    assert len(extracted) == 2


def test_unreadable_face_in_group_is_reported_not_silently_dropped(tmp_path, monkeypatch):
    import cv2
    from test_characters import face
    path = tmp_path/'group.png'
    cv2.imencode('.png',np.zeros((180,240,3),np.uint8))[1].tofile(str(path))
    class Model:
        def __init__(self,*a):
            pass
        def extract(self,frame,subject,*a):
            return np.asarray([1.,0.]) if subject['face'][0] < 100 else None
    class Detector:
        def __init__(self,**kw):
            pass
        def detect(self,*a):
            return {'faces':[face([20,40,80,80]),face([110,40,80,80])]}
        def close(self):
            pass
    monkeypatch.setattr('autoedit.video.references.EmbeddingModel',Model)
    monkeypatch.setattr('autoedit.video.faces.MultiRegionFaceLandmarker',Detector)
    cfg = settings({'reference_images':[str(path)]})
    vectors, catalog = reference_vectors('live_action',cfg,tmp_path,{'model_sha256':'one'})
    assert len(vectors) == 1
    assert catalog['skipped_regions'][0]['face_index'] == 1
    assert reference_vectors('live_action',cfg,tmp_path,{'model_sha256':'one'})[1] == catalog
    messages = []
    match_reference_tracks([track('a',[1,0],0,2,0)],'live_action',cfg,tmp_path,
                           {'model_sha256':'one'},messages.append)
    assert any('face 2 unreadable' in message for message in messages)


def test_reference_ids_follow_crop_provenance_when_models_detect_different_faces(tmp_path, monkeypatch):
    import cv2
    from test_characters import face
    path = tmp_path/'group.png'
    cv2.imencode('.png',np.zeros((180,240,3),np.uint8))[1].tofile(str(path))
    left, right = face([20,40,80,80]), face([110,40,80,80])
    class Model:
        def __init__(self,*a):
            pass
        def extract(self,*a):
            return np.asarray([1.,0.])
    class Detector:
        def __init__(self,include_animation):
            self.animation = include_animation
        def detect(self,*a):
            return {'faces':[left,right] if self.animation else [right]}
        def close(self):
            pass
    monkeypatch.setattr('autoedit.video.references.EmbeddingModel',Model)
    monkeypatch.setattr('autoedit.video.faces.MultiRegionFaceLandmarker',Detector)
    cfg = settings({'reference_images':[str(path)]})
    _, human = reference_vectors('live_action',cfg,tmp_path,{'model_sha256':'human'})
    _, animation = reference_vectors('animation',cfg,tmp_path,{'model_sha256':'cartoon'})
    assert human['templates'][0]['id'] == animation['templates'][1]['id']
    assert human['templates'][0]['id'] != animation['templates'][0]['id']


def test_cut_requires_one_dominant_target_not_combined_imported_exposure():
    info = video(['main']*8)
    for sample in info['samples']:
        sample.update(reference_match=True,reference_id='a' if sample['time_sec'] < 2 else 'b',
                      reference_confidence=1.)
    metrics = cut_character_metrics(info['samples'],0,4,.5,.85)
    assert metrics['reference_any_fraction'] == 1.
    assert metrics['reference_fraction'] == .5 and metrics['reference_id'] is None
    info['character_analysis']['reference'] = {'images':['group.png']}
    info['reference_only'] = True
    with pytest.raises(InsufficientFootageError):
        match_slots_to_video([{'id':'long','required_duration_sec':4}],[],info)


def multi_target_footage():
    info = video(['main']*96)
    info['character_analysis']['reference'] = {'images':['group.png']}
    info['reference_only'] = True
    for sample in info['samples']:
        time = sample['time_sec']
        rid, rank = ('a',1) if time < 24 else ('b',2) if time < 36 else ('c',3)
        sample.update(reference_match=True,reference_id=rid,reference_confidence=1.,
                      character_id=rid,character_rank=rank)
    return info


@pytest.mark.parametrize('bounded',[False,True])
def test_allocation_prefers_less_used_eligible_targets_and_keeps_slot_order(bounded):
    from autoedit.match.matcher import _match_slots_to_video
    info = multi_target_footage()
    slots = [{'id':str(i),'required_duration_sec':2} for i in range(6)]
    matches = _match_slots_to_video(slots,[],info,min_gap_sec=1,allocation_search=bounded)
    refs = [match['video']['character_selection']['reference_id'] for match in matches]
    assert {rid:refs.count(rid) for rid in set(refs)} == {'a':2,'b':2,'c':2}
    cuts = sorted((m['video']['in_sec'],m['video']['out_sec']) for m in matches)
    assert all(end+1 <= start+1e-8 for (_,end),(start,_) in zip(cuts,cuts[1:]))
    assert all(m['video']['used_duration_sec'] == slots[i]['required_duration_sec']
               for i,m in enumerate(matches))


def test_unavailable_reference_is_not_a_quota_that_blocks_early_completion():
    info = multi_target_footage()
    info['character_analysis']['reference']['characters'] = [{'id':'never-seen'}]
    for sample in info['samples']:
        sample.update(reference_id='a')
    slots = [{'id':str(i),'required_duration_sec':2} for i in range(4)]
    matches = match_slots_to_video(slots,[],info,min_gap_sec=1)
    assert len(matches) == 4
    assert all(m['video']['character_selection']['reference_id'] == 'a' for m in matches)


def test_automatic_selection_retains_exposure_order_without_references():
    info = multi_target_footage()
    del info['character_analysis']['reference']
    del info['reference_only']
    for sample in info['samples']:
        sample.update(reference_match=False,reference_id=None)
    matches = match_slots_to_video([{'id':str(i),'required_duration_sec':2} for i in range(4)],[],info)
    assert all(m['video']['character_selection']['best_rank'] == 1 for m in matches)


def test_report_shows_selected_and_unselected_targets_and_fallback():
    from autoedit.video.references import reference_selection_report
    reference = {'characters':[{'id':rid,'templates':[]} for rid in ['a','b','missing']]}
    slots = [{'video':{'in_sec':0,'out_sec':3,
                      'character_selection':{'reference_id':rid,'reference_fallback':rid is None}}}
             for rid in ['a','a','b',None]]
    report = reference_selection_report(reference,slots)
    assert report['selected_target_count'] == 2 and report['fallback_cuts'] == 1
    assert not report['all_targets_selected']
    assert [(c['selected_cuts'],c['selected_seconds'],c['status']) for c in report['characters']] == [
        (2,6,'selected'),(1,3,'selected'),(0,0,'not-selected')]


def test_multi_source_pool_shares_reference_ids_but_keeps_video_tracks_distinct():
    from autoedit.video.pool import combine_video_infos
    infos = []
    for path in ['first.mp4','second.mp4']:
        info = multi_target_footage()
        info['path'] = path
        info['character_analysis']['reference']['characters'] = [
            {'id':rid,'templates':[{'id':rid,'image_index':0}]} for rid in ['a','b','c']]
        for sample in info['samples']:
            sample['reference_template_id'] = sample['reference_id']
            sample['track_id'] = 'track'
        infos.append(info)
    pool = combine_video_infos(infos,1)
    assert {s['reference_id'] for s in pool['samples']} == {'a','b','c'}
    assert {s['track_id'] for s in pool['samples']} == {'source-0:track','source-1:track'}
    assert len(pool['character_analysis']['reference']['characters']) == 3


def test_pool_does_not_merge_targets_when_source_models_disagree():
    from autoedit.video.references import pooled_reference_catalog
    a, b = {'id':'a','image_index':0}, {'id':'b','image_index':1}
    common = {'images':['a.png','b.png'],'matched_tracks':0}
    merged = {**common,'characters':[{'id':'a','templates':[a,b]}]}
    separate = {**common,'characters':[{'id':'a','templates':[a]},{'id':'b','templates':[b]}]}
    catalog, owners = pooled_reference_catalog([merged,separate])
    assert len(catalog['characters']) == 2 and owners == {'a':'a','b':'b'}


def test_co_visible_targets_get_separate_cuts_and_refine_the_selected_mouth():
    from test_characters import sample, face
    from autoedit.video.refine import cut_references, select_reference_face
    samples = []
    for i in range(24):
        primary = {**sample(i/2,'main',1),'track_id':'track-a','reference_match':True,
                   'reference_id':'a','reference_template_id':'a','reference_confidence':1.,
                   'lip_aperture':.05 if i%2 else .3,'lip_width_ratio':.6,
                   'frame_width':640,'frame_height':360}
        secondary = {**sample(i/2,'main',2),'track_id':'track-b','reference_match':True,
                     'reference_id':'b','reference_template_id':'b','reference_confidence':1.,
                     'face':[160,20,80,80],'lip_aperture':.05 if i%2 else .3,'lip_width_ratio':.6}
        primary['faces'] = [dict(primary), secondary]
        samples.append(primary)
    info = {'duration_sec':12,'sample_fps':2,'samples':samples,'reference_only':True,
            'character_analysis':{'min_main_fraction':.85,'reference':{'images':['group.png']}}}
    matches = match_slots_to_video([{'id':str(i),'required_duration_sec':3} for i in range(2)],
                                  [],info,min_gap_sec=1,require_lip_motion=True)
    assert {m['video']['character_selection']['reference_id'] for m in matches} == {'a','b'}
    secondary = next(m for m in matches if m['video']['character_selection']['reference_id'] == 'b')
    assert secondary['video']['selection_metrics']['subject_track_id'] == 'track-b'
    references = cut_references(samples,secondary)
    time = secondary['video']['in_sec']
    refined = select_reference_face([face(),face([160,20,80,80])],640,360,time,
                                    references,[s['time_sec'] for s in references])
    assert refined['track_id'] == 'track-b' and refined['reference_id'] == 'b'
    assert refined['reference_template_id'] == 'b'


def test_pipeline_reports_multiple_targets_and_stops_after_verified_completion(job, monkeypatch):
    import autoedit.pipeline as pipeline
    cfg, inspect, install, scanned, speech, exports, refined = job
    second = {**deepcopy(inspect['slots'][0]),'id':'two','nested_sequence':'2',
              'nested_sequence_uid':'uid-2','instances':[{'start_sec':4,'end_sec':8}]}
    inspect['slots'].append(second)
    inspect['scene_slots'].append({**second,'index':1,'first_start_sec':4,'last_end_sec':8})
    monkeypatch.setattr(pipeline,'inspect_sampler_tracks',lambda *a: {
        'path':'template.cpr','tracks':[{'index':i,'name':f'Sampler Track {i:02}'} for i in [1,2]]})
    cfg['video']['characters']['reference_images'] = ['group.png']
    monkeypatch.setattr('autoedit.video.objects.augment_with_objects',lambda raw,*a,**kw:raw)
    def analyze(raw,*a,**kw):
        info = deepcopy(raw)
        info['character_analysis'] = {'min_main_fraction':.85,'reference':{
            'images':['group.png'],'characters':[{'id':rid,'templates':[]} for rid in ['a','b']]}}
        for sample in info['samples']:
            rid = 'a' if sample['time_sec'] < 6 else 'b'
            properties = {'character_status':'main','character_rank':1 if rid=='a' else 2,
                          'character_id':rid,'identity_confidence':1.,'reference_match':True,
                          'reference_id':rid,'reference_confidence':1.}
            sample.update(properties)
            for face in sample['faces']:
                face.update(properties)
        return info
    monkeypatch.setattr(pipeline,'analyze_characters',analyze)
    install([6,12,18])
    plan = pipeline.run_pipeline(cfg)['plan']
    assert scanned == [('first.mp4',6),('first.mp4',12)]
    assert exports == ['first.mp4']
    assert {s['video']['character_selection']['reference_id'] for s in plan['slots']} == {'a','b'}
    report = plan['video_analysis_meta']['character_reference_selection']
    assert report['all_targets_selected'] and report['fallback_cuts'] == 0
    assert [c['selected_cuts'] for c in report['characters']] == [1,1]
    assert all(s['video']['selection_metrics']['dense_lip_motion_verified'] for s in plan['slots'])
