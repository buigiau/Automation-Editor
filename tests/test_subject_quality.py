from copy import deepcopy

import numpy as np
import pytest

from autoedit.match.matcher import InsufficientFootageError, match_slots_to_video
from autoedit.video.characters import _annotate_subjects, rank_individual_characters, settings
from autoedit.video.objects import merge_subjects, normalize_objects
from autoedit.video.subjects import exclude_screen_overlays
from test_characters import face, sample, track
from test_incremental_analysis import job


@pytest.fixture(autouse=True)
def isolate_synthetic_job_lock(tmp_path, monkeypatch):
    # These jobs use fake media/models; they must not queue behind a real job.
    monkeypatch.setattr('autoedit.runtime.ROOT', tmp_path)


def test_transparent_logo_is_excluded_across_shots_but_actor_remains(tmp_path, monkeypatch):
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'source')
    rng = np.random.default_rng(4)
    logo = rng.integers(0, 255, (48, 54, 3), dtype=np.uint8)
    def frames(path, times):
        for time in times:
            image = np.full((360,640,3), 40 if time < 3 else 210, np.uint8)
            image[14:62,16:70] = logo
            yield time, image
    monkeypatch.setattr('autoedit.video.characters._selected_source_frames', frames)
    corner = face([8,8,70,60])
    actor = face([280,70,90,110])
    info = {'path': str(path), 'shot_times_sec': [3], 'samples': [
        {'time_sec': i/2, 'frame_width':640, 'frame_height':360,
         'faces': [deepcopy(corner), deepcopy(actor)], **deepcopy(corner)} for i in range(12)]}
    exclude_screen_overlays(info)
    assert len(info['subject_filter']['screen_overlays']) == 1
    assert all(len(s['faces']) == 1 and s['faces'][0]['face'] == actor['face'] for s in info['samples'])
    assert all(not s['face_detected'] for s in info['samples'])
    # Stationarity in one shot cannot establish a screen overlay.
    info['shot_times_sec'] = []
    for s in info['samples']:
        s['faces'] = [deepcopy(corner), deepcopy(actor)]
    exclude_screen_overlays(info)
    assert all(len(s['faces']) == 2 for s in info['samples'])


def test_moving_corner_actor_is_not_a_watermark(tmp_path, monkeypatch):
    path = tmp_path/'source.mp4'
    path.write_bytes(b'source')
    def frames(path, times):
        for time in times:
            yield time, np.random.default_rng(int(time*100)+1).integers(0,255,(360,640,3),dtype=np.uint8)
    monkeypatch.setattr('autoedit.video.characters._selected_source_frames', frames)
    info = {'path': str(path), 'shot_times_sec': [3], 'samples': [
        {'time_sec':i/2, 'frame_width':640, 'frame_height':360, 'faces':[face([8,8,70,60])]}
        for i in range(12)]}
    exclude_screen_overlays(info)
    assert info['subject_filter']['screen_overlays'] == []
    assert all(s['faces'] for s in info['samples'])


@pytest.mark.parametrize('prop', ['a teddy bear', 'a stuffed toy', 'a basket', 'a logo'])
def test_props_cannot_confirm_actor_even_with_higher_actor_score(prop):
    raw = [{'box':[20,20,100,160], 'score':.35, 'label':prop},
           {'box':[20,20,100,160], 'score':.8, 'label':'a skeleton an animated character'}]
    assert not normalize_objects(raw,640,360)
    assert not merge_subjects({'faces':[face([30,30,50,60])]}, [], [raw[0]])


def test_cartoon_geometry_on_furniture_needs_independent_actor():
    hypothesis = {**face(), 'backend':'animation-eyes-mouth', 'face_evidence':'goggle-pupil-yellow-mouth'}
    assert not merge_subjects({'faces':[hypothesis]}, [])
    confirmed = merge_subjects({'faces':[hypothesis]}, [{'box':[20,10,120,200], 'score':.8, 'label':'a person'}])
    assert len(confirmed) == 1 and confirmed[0]['face_detected']


def test_clear_faces_do_not_require_speaking_or_readable_lips():
    samples = [dict(sample(i/2,'main'), readable_face=1., clear_face=0., speaking=0., mouth_clarity=0.)
               for i in range(20)]
    video = {'duration_sec':10, 'sample_fps':2, 'samples':samples, 'require_clear_face':True,
             'character_analysis':{'min_main_fraction':.85}}
    cut = match_slots_to_video([{'id':'1','required_duration_sec':2}],[],video)[0]['video']
    assert cut['selection_metrics']['readable_face'] == 1
    assert cut['selection_metrics']['clear_face'] == 0
    assert cut['character_selection']['best_rank'] == 1


def test_facial_readability_measures_face_sharpness_independently_of_mouth():
    import cv2
    from autoedit.video.analyzer import _visual_quality
    frame = np.random.default_rng(6).integers(0,255,(360,640,3),dtype=np.uint8)
    detected = {**face([200,80,120,150]), 'mouth_box':[235,180,40,15], 'frontal_score':0.}
    sharp,_ = _visual_quality(cv2,frame,detected)
    assert sharp['readable_face'] == 1 and sharp['mouth_clarity'] == 0
    blurred,_ = _visual_quality(cv2,cv2.GaussianBlur(frame,(61,61),10),detected)
    assert blurred['readable_face'] == 0


def test_clear_face_requirement_never_falls_back_to_body_to_fill_slots():
    subjects = merge_subjects({'faces':[]}, [{'box':[20,20,90,180],'score':.9,'label':'a person'}])
    samples = [dict(subjects[0], time_sec=i/2, character_id='main', character_rank=1,
                    character_status='main', identity_confidence=1., faces=deepcopy(subjects)) for i in range(20)]
    video = {'duration_sec':10,'sample_fps':2,'samples':samples,'require_clear_face':True,
             'character_analysis':{'min_main_fraction':.85}}
    with pytest.raises(InsufficientFootageError):
        match_slots_to_video([{'id':'1','required_duration_sec':2}],[],video)
    video['require_clear_face'] = False
    assert match_slots_to_video([{'id':'1','required_duration_sec':2}],[],video)


def test_individual_ranking_keeps_similar_covisible_actors_separate():
    left = track('left',[1,0],0,4,0,[0,1,2])
    right = track('right',[1,0],0,4,0,[0,1,2])
    for t in [left,right]:
        t.update(character_group='yellow_minions',group_evidence='goggle-pupil-yellow-mouth')
    longer = track('longer',[0,1],5,5,1,[3,4,5])
    clusters,_,_ = rank_individual_characters([left,right,longer],settings(),.2)
    assert clusters[0]['track_ids'] == ['longer']
    assert len(clusters) == 3
    assert all(not c.get('character_group') for c in clusters)


def test_cast_exposure_excludes_body_only_prop_when_faces_required():
    tracks = [track('basket',[1,0],0,50,0,[0,1]),track('actor',[0,1],50,10,1,[2,3])]
    samples = [{'faces':[dict(face(),face_detected=False)]} for _ in range(2)]
    samples += [{'faces':[face()]},{'faces':[face()]}]
    clusters,_,_ = rank_individual_characters(tracks,{**settings(),'_require_actor_face':True},.2,samples)
    assert len(clusters) == 1 and clusters[0]['track_ids'] == ['actor']
    assert tracks[0]['identity_status'] == 'unknown'


def test_removed_primary_subject_cannot_leave_object_evidence_on_empty_sample():
    subject = merge_subjects({'faces':[]},[{'box':[10,10,60,60],'score':.9,'label':'an animated character'}])[0]
    info = {'samples':[dict(subject,time_sec=0.,frame_width=640,frame_height=360,faces=[])]}
    _annotate_subjects(info,'animation')
    assert not info['samples'][0].get('object_confirmed')
    assert info['samples'][0]['body_visible'] == 0


def test_dense_readable_face_check_rejects_missing_time_and_body_only_frames():
    from autoedit.video.refine import readable_face_fraction
    samples = [dict(sample(i/20,'main'),readable_face=1.,speaking=0.) for i in range(40)]
    assert readable_face_fraction(samples,0,2)['readable_face_fraction'] == pytest.approx(1)
    assert readable_face_fraction(samples[:10],0,2)['sample_coverage'] < .3
    for s in samples[10:30]:
        s['face_detected'] = False
    assert readable_face_fraction(samples,0,2)['readable_face_fraction'] < .6


@pytest.mark.parametrize('mode', ['until_filled', 'full'])
def test_automatic_cast_respects_requested_analysis_scope(job, monkeypatch, mode):
    import autoedit.pipeline as pipeline
    config,inspect,install,scanned,speech,exports,refined = job
    config['video'].update(analysis_mode=mode, require_lip_motion=False,require_clear_face=False,characters={'enabled':True})
    def characters(info,*args,**kwargs):
        info['character_analysis'] = {'min_main_fraction':.85}
        for s in info['samples']:
            s.update(character_id='main',character_rank=1,character_status='main',identity_confidence=1.)
        return info
    monkeypatch.setattr(pipeline,'analyze_characters',characters)
    install([6])
    result = pipeline.run_pipeline(config)
    assert scanned == ([('first.mp4',6),('unused.mp4',6)] if mode == 'full' else [('first.mp4',6)])
    assert result['plan']['video_analysis_meta']['analysis_mode'] == mode
    assert result['plan']['video_analysis_meta']['character_ranking_scope'] == (
        'configured_sources' if mode == 'full' else 'analyzed_footage')


def test_dense_unclear_face_is_reselected_and_never_published(job, monkeypatch):
    import autoedit.pipeline as pipeline
    config,inspect,install,scanned,speech,exports,refined = job
    config['video']['require_clear_face'] = True
    original = pipeline.refine_selected_cuts
    rejected = []
    def refine(path, slots,*args,**kwargs):
        info = original(path, slots,*args,**kwargs)
        if not rejected:
            rejected.append((slots[0]['video']['in_sec'],slots[0]['video']['out_sec']))
            for s in info['samples']:
                s.update(readable_face=0.,face_detected=False)
        return info
    monkeypatch.setattr(pipeline,'refine_selected_cuts',refine)
    install([6,18])
    result = pipeline.run_pipeline(config)
    cut = result['plan']['slots'][0]['video']
    assert cut['selection_metrics']['dense_clear_face_verified']
    assert cut['out_sec'] <= rejected[0][0] or cut['in_sec'] >= rejected[0][1]


def test_fast_review_keeps_ranked_characters_and_defers_dense_checks(job, monkeypatch):
    import autoedit.pipeline as pipeline
    config, inspect, install, scanned, speech, exports, refined = job
    config['video'].update(verification_mode='review', require_lip_motion=True,
                          characters={'enabled': True, 'embeddings_per_track': 8})
    measured = []
    def characters(info, cache_dir, character_config, **kwargs):
        measured.append(character_config)
        info['character_analysis'] = {'min_main_fraction': .85, 'clusters': [
            {'id': 'main', 'rank': 1, 'observed_seconds': 6, 'is_main': True}]}
        for s in info['samples']:
            s.update(character_id='main', character_rank=1, character_status='main', identity_confidence=1.)
        return info
    monkeypatch.setattr(pipeline, 'analyze_characters', characters)
    monkeypatch.setattr(pipeline, 'refine_selected_cuts', lambda *a, **kw: pytest.fail('Review export ran dense checks'))
    install([6, 12])
    plan = pipeline.run_pipeline(config)['plan']
    assert scanned == [('first.mp4', 6)]
    assert measured[0]['enabled'] and measured[0]['embeddings_per_track'] == 4
    assert config['video']['characters']['embeddings_per_track'] == 8
    assert plan['review_status'] == 'manual_review_required'
    assert plan['video_analysis_meta']['stop_reason'] == 'all_slots_selected_for_review'
    assert plan['video_analysis_meta']['verification_mode'] == 'review'
    cut = plan['slots'][0]['video']
    assert cut['character_selection']['best_rank'] == 1
    assert cut['out_sec']-cut['in_sec'] == pytest.approx(4)
    assert not cut['selection_metrics']['dense_clear_face_verified']
    assert not cut['selection_metrics']['dense_lip_motion_verified']
    assert plan['cubase_slots'][0]['audio_match']['needs_review']
    assert not plan['cubase_slots'][0]['audio_match']['onset_verified']
    assert plan['cubase_slots'][0]['audio_match']['method'] != 'filename-visemes-and-dense-lip-rhythm'
    assert not refined and speech and exports
    assert speech[0][1] == [(max(0., cut['in_sec']-1), min(6., cut['out_sec']+1))]


def test_review_audio_ranges_use_source_local_times_and_skip_unused_source(job, monkeypatch):
    import autoedit.pipeline as pipeline
    config, inspect, install, scanned, speech, exports, refined = job
    config['video'].update(verification_mode='review', analysis_mode='full', require_lip_motion=False)
    install([12])
    plan = pipeline.run_pipeline(config)['plan']
    assert len(speech) == 1  # Selected cuts in one source; unused source has no ASR work.
    assert speech[0][0] == 'first.mp4'
    cut = plan['slots'][0]['video']
    assert speech[0][1] == [(max(0., cut['in_sec']-1), min(12., cut['out_sec']+1))]
    assert not refined and plan['review_status'] == 'manual_review_required'


def test_tight_animation_face_query_never_promotes_cartoon_body_or_speech():
    from autoedit.video.objects import localized_faces
    from autoedit.match.matcher import _has_face
    localized = localized_faces([
        {'label':'a cartoon face','score':.9,'box':[10,10,180,300],'clarity':1.},
        {'label':'a face','score':.6,'box':[40,30,120,100],'clarity':1.},
        {'label':'a face','score':.7,'box':[45,35,118,98],'clarity':1.}],640,360)
    assert len(localized) == 1 and _has_face(localized[0])
    assert localized[0]['face'] == [45,35,73,63]
    assert localized[0]['speaking'] == 0 and not localized[0]['lip_geometry_measured']
    localized[0].pop('face_evidence')
    assert not _has_face(localized[0])


def test_dense_localized_face_follows_actor_and_rejects_printed_faces():
    from autoedit.video.objects import localized_faces
    from autoedit.video.refine import select_localized_reference_face
    faces = localized_faces([{'label':'a face','score':.6,'box':[100,60,170,140],'clarity':1.},
                            {'label':'a face','score':.7,'box':[300,60,370,140],'clarity':1.}],640,360)
    reference = dict(sample(1.,'main'),face=[100,60,70,80],frame_width=640,frame_height=360)
    selected = select_localized_reference_face(faces,640,360,1.,[reference],[1.])
    assert selected['face'] == [100,60,70,80] and selected['character_id'] == reference['character_id']
    reference['prop_regions'] = [{'box':[90,50,100,110],'label':'a sheet of paper'}]
    assert not select_localized_reference_face(faces,640,360,1.,[reference],[1.])['face_detected']


def test_original_actor_face_coordinates_and_cached_prop_rejection(tmp_path, monkeypatch):
    from autoedit.video.subjects import refine_actor_faces
    path = tmp_path/'source.mp4'
    model = tmp_path/'face.task'
    path.write_bytes(b'source'); model.write_bytes(b'model')
    monkeypatch.setattr('autoedit.video.faces.MODEL_PATH',model)
    class Detector:
        def detect_all(self,image,time):
            assert image.shape[:2] == (360,200)
            return [{**face([50,40,60,80]),'mouth_box':[65,95,30,10],
                     'face_keypoints':[[55,55]]*7}]
        def close(self): pass
    monkeypatch.setattr('autoedit.video.faces.FaceLandmarker',Detector)
    monkeypatch.setattr('autoedit.video.characters._selected_source_frames',
        lambda path,times: iter((t,np.random.default_rng(2).integers(0,255,(720,1280,3),dtype=np.uint8)) for t in times))
    subject = merge_subjects({'faces':[]},[{'box':[100,40,100,180],'score':.8,'label':'a person'}])[0]
    sample_data = dict(time_sec=1.,frame_width=640,frame_height=360,faces=[subject],
        prop_regions=[{'box':[120,55,40,50],'label':'a poster'}])
    info = {'path':str(path),'samples':[sample_data]}
    refine_actor_faces(info,tmp_path)
    assert not subject['face_detected'] and info['actor_face_refinement']['measured_faces'] == 0
    sample_data['prop_regions'] = []
    monkeypatch.setattr('autoedit.video.faces.FaceLandmarker',lambda:pytest.fail('actual measurements should be cached'))
    refine_actor_faces(info,tmp_path)
    assert subject['face'] == [125,60,30,40] and subject['actor_face_measured']
    assert subject['face_keypoints'][0] == [127.5,67.5]


def test_readable_face_size_is_independent_of_sampling_resolution():
    import cv2
    from autoedit.video.analyzer import _visual_quality
    image = np.random.default_rng(8).integers(0,255,(360,640,3),dtype=np.uint8)
    detected = {**face([100,60,40,35]),'mouth_box':[110,80,10,5]}
    assert _visual_quality(cv2,image,detected)[0]['readable_face'] == 1
    tiny = {**detected,'face':[100,60,20,15]}
    assert _visual_quality(cv2,image,tiny)[0]['readable_face'] == 0


def test_confident_animated_face_resolves_weak_plush_label_but_never_paper():
    from autoedit.video.objects import face_on_prop, localized_faces
    actor = localized_faces([{'label':'a face','score':.63,'box':[100,60,180,140]}],640,360)[0]
    prop = {'label':'a teddy bear','box':[80,20,150,200],'score':.5}
    assert not face_on_prop(actor,prop)
    assert face_on_prop(actor,{**prop,'score':.8})
    assert face_on_prop(actor,{**prop,'label':'a sheet of paper','score':.3})


def test_shark_costume_face_can_resolve_weaker_plush_and_full_frame_poster_labels():
    from autoedit.video.objects import face_on_prop, localized_faces
    actor = localized_faces([{'label':'a face','score':.475,'box':[274,107,344,161]}],640,360)[0]
    plush = {'label':'a teddy bear','box':[235,39,138,213],'score':.397}
    background = {'label':'a poster a sheet','box':[5,6,629,349],
                  'score':.417, 'full_frame':True, 'frame_fraction':.953}
    assert not face_on_prop(actor,plush)
    assert not face_on_prop(actor,background)
    assert face_on_prop(actor,{**background,'score':.7})
    assert face_on_prop(actor,{**background,'frame_fraction':.4,'full_frame':False})
    assert face_on_prop(actor,{**plush,'label':'a sheet of paper'})
    assert face_on_prop(actor,{**plush,'score':.6})
    sample = {'frame_width':640,'frame_height':360,'faces':[actor]}
    raw = {'label':'a poster a sheet','box':[5,6,634,355],'score':.417}
    assert merge_subjects(sample,[],[raw])[0]['face_detected']


def test_person_confirmation_resolves_costume_ambiguity_without_inventing_faces(monkeypatch,tmp_path):
    from autoedit.video.subjects import confirm_animation_people
    from autoedit.video.objects import localized_faces,face_on_prop
    sample = {'time_sec':1.,'frame_width':640,'frame_height':360}
    face_rows = {'samples':[{'time_sec':1.,'objects':[
        {'box':[254,198,374,300],'label':'a face','score':.4,'clarity':1.}]}]}
    props = {'samples':[{'objects':[
        {'box':[215,75,426,356],'label':'a teddy bear','score':.54}]}]}
    def scan(info,*args):
        return {'samples':[{'time_sec':1.,'objects':[
            {'box':[212,75,428,356],'label':'a child','score':.52}]}]}, {'source':'test'}
    monkeypatch.setattr('autoedit.video.objects._cached_object_scan',scan)
    result,meta=confirm_animation_people({'path':'video.mp4','samples':[sample]},props,face_rows,tmp_path, {})
    assert meta['confirmed_faces'] == 1
    face=localized_faces(face_rows['samples'][0]['objects'],640,360)[0]
    face['person_costume_confirmation']=result['samples'][0]['person_faces'][0]['person']
    assert not face_on_prop(face,{'box':[215,75,211,281],'label':'a teddy bear','score':.54})
    assert face_on_prop(face,{'box':[215,75,211,281],'label':'a sheet of paper','score':.54})
    props['samples'][0]['objects'].append({'box':[215,75,426,356],'label':'a sheet of paper','score':.4})
    face_rows['samples'][0].pop('person_faces')
    result,meta=confirm_animation_people({'path':'video.mp4','samples':[sample]},props,face_rows,tmp_path, {})
    assert meta is None
    assert not result['samples'][0].get('person_faces')


def test_native_actor_queries_measure_profile_face_and_reject_paper(monkeypatch,tmp_path):
    from autoedit.video.subjects import refine_animation_actor_faces
    subjects = merge_subjects({'faces':[]},[{'box':[100,40,100,180],'score':.8,'label':'a person'}])
    info = {'path':'source.mp4','samples':[
        dict(time_sec=i,frame_width=640,frame_height=360,faces=deepcopy(subjects),
             prop_regions=[] if i == 0 else [{'box':[120,55,60,70],'label':'a sheet of paper','score':.4}])
        for i in range(2)]}
    def scan(request,*args):
        assert all(s['roi'] == [100,40,200,220] for s in request['samples'])
        return {'samples':[{'time_sec':s['time_sec'],'objects':[
            {'label':'a face','score':.8,'box':[125,60,175,115],'clarity':1.}]} for s in request['samples']]}, {'measured':True}
    monkeypatch.setattr('autoedit.video.objects._cached_object_scan',scan)
    refine_animation_actor_faces(info,tmp_path,{})
    measured = info['samples'][0]['faces'][0]
    assert measured['face'] == [125,60,50,55] and measured['readable_face']
    assert measured['face_detection_region'] == 'native-actor-face-query'
    assert not measured['lip_geometry_measured'] and not measured['speaking']
    assert not info['samples'][1]['faces'][0]['face_detected']
    assert info['actor_face_queries']['measured_faces'] == 1


def test_tracking_drift_onto_poster_cannot_establish_actor_or_readable_face():
    drifting = {**face([135,107,106,121]),'backend':'feature-tracked-face',
                'face_evidence':'reversible-face-features','verified_feature_count':30}
    assert not merge_subjects({'faces':[drifting]},[])
    poster = {'label':'a poster','box':[170,100,260,205],'score':.5}
    assert not merge_subjects({'faces':[drifting]},[],[poster])


def test_localized_poster_cannot_be_overridden_by_confident_face_query():
    from autoedit.video.objects import face_on_prop,localized_faces
    detected = localized_faces([{'label':'a face','score':.8,'box':[100,60,180,140]}],640,360)[0]
    poster = {'label':'a poster','box':[80,20,150,200],'score':.4}
    assert face_on_prop(detected,poster)
    assert not face_on_prop(detected,{**poster,'box':[0,0,640,360],'full_frame':True})


def test_face_identity_tolerance_preserves_covisible_individuals():
    left = track('left',[1,0],0,5,0,[0,1])
    costume = track('costume',[.75,np.sqrt(1-.75**2)],5,5,1,[2,3])
    neighbour = track('neighbour',[1,0],0,5,0,[0,1])
    for t in [left,costume,neighbour]:t['identity_region'] = 'measured-face'
    clusters,_,_ = rank_individual_characters([left,costume,neighbour],settings(),.2)
    assert len(clusters) == 2
    assert set(clusters[0]['track_ids']) == {'left','costume'}
    assert clusters[0]['duration_sec'] == 10
    assert clusters[1]['track_ids'] == ['neighbour']


def test_automatic_face_crops_ignore_costume_but_body_references_retain_it():
    from autoedit.video.embeddings import EmbeddingModel
    model = EmbeddingModel.__new__(EmbeddingModel)
    model.kind,model.size = 'animation',32
    red = np.full((240,320,3),(30,30,200),np.uint8)
    blue = np.full_like(red,(200,30,30))
    shared = np.random.default_rng(9).integers(0,255,(84,72,3),dtype=np.uint8)
    red[73:157,94:166] = shared
    blue[73:157,94:166] = shared
    detected = {**face([100,80,60,70]),'subject_kind':'face','identity_face_crop':True}
    assert np.array_equal(model.prepare(red,detected,320,240),model.prepare(blue,detected,320,240))
    body = {**detected,'subject_kind':'object','subject_box':[80,40,110,180]}
    assert not np.array_equal(model.prepare(red,body,320,240),model.prepare(blue,body,320,240))


def test_dense_reference_cannot_inherit_body_box_of_another_covisible_actor():
    from autoedit.video.refine import cut_references
    primary = merge_subjects({'faces':[]},[{'box':[100,40,100,180],'score':.8,'label':'a person'}])[0]
    chosen = {**face([300,80,60,70]),'track_id':'chosen','subject_kind':'face','character_id':'second'}
    samples = [{**primary,'time_sec':1.,'frame_width':640,'frame_height':360,
                'faces':[primary,chosen],'prop_regions':[{'box':[100,100,40,50],'label':'a paper'}]}]
    refs = cut_references(samples,{'video':{'in_sec':1.,'out_sec':2.,'selection_metrics':{'subject_track_id':'chosen'}}})
    assert refs[0]['face'] == chosen['face'] and refs[0]['character_id'] == 'second'
    assert 'subject_box' not in refs[0] and 'object_confirmed' not in refs[0]
    assert refs[0]['prop_regions'] == samples[0]['prop_regions']
