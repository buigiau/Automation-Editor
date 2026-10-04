import math

from autoedit.video.objects import merge_subjects, normalize_objects, object_evidence


def face(box):
    return {'face':box,'face_detected':True,'backend':'mediapipe-landmarks',
            'lip_aperture':.2,'mouth_box':[box[0]+5,box[1]+20,20,10]}


def test_independent_object_detection_establishes_body_without_inventing_a_face():
    subjects=merge_subjects({'faces':[]},[{'box':[20,20,90,180],'score':.65,'label':'skeleton'}])
    assert len(subjects)==1 and object_evidence(subjects[0])
    assert not subjects[0]['face_detected'] and not subjects[0]['speaking']
    assert not subjects[0]['clear_face'] and 'lip_aperture' not in subjects[0]
    assert subjects[0]['subject_box']==[20,20,90,180]


def test_lips_belong_to_one_contained_face_not_another_covisible_actor():
    actor=face([30,30,30,40])
    other=face([150,30,30,40])
    result=merge_subjects({'faces':[actor,other]},[{'box':[20,20,90,180],'score':.65,'label':'skeleton'}])
    assert len(result)==2
    assert result[0]['lip_aperture']==actor['lip_aperture']
    assert result[0]['face']==actor['face'] and result[0]['track_box']==[20,20,90,180]
    assert result[1]['face']==other['face'] and not result[1].get('object_confirmed')
    crowded=merge_subjects({'faces':[actor,other]},[{'box':[20,20,180,180],'score':.65,'label':'skeleton'}])
    assert not crowded[0]['face_detected'] and 'lip_aperture' not in crowded[0]


def test_object_boxes_reject_nonfinite_small_and_unconfirmed_detections():
    raw=[{'box':[0,0,90,100],'score':.7,'label':'skeleton'},
         {'box':[1,1,91,101],'score':.6,'label':'monster'},
         {'box':[math.nan,0,100,100],'score':.9},
         {'box':[0,0,10,10],'score':.9},
         {'box':[140,0,180,80],'score':.2}]
    assert len(normalize_objects(raw,200,200))==1
    assert not object_evidence({'subject_box':[0,0,90,100],'object_score':1})
    assert not object_evidence({'object_detector':'grounding-dino','object_confirmed':True,
                               'subject_box':[0,0,90,100],'object_score':.2})


def test_head_and_body_are_one_subject_but_foreground_actor_stays_separate():
    raw=[{'box':[20,0,120,180],'score':.6,'label':'skeleton'},
         {'box':[30,10,80,60],'score':.5,'label':'monster'},
         {'box':[0,0,100,180],'score':.7,'label':'person'}]
    result=normalize_objects(raw,200,200)
    assert len(result)==2
    assert {o['label'] for o in result}=={'person','monster'}
    # A second label for a nested region must not create a false co-visible
    # identity constraint that splits one creature's head and body forever.
    raw[1]['label']='animal'
    result=normalize_objects(raw,200,200)
    assert len(result)==2 and {o['label'] for o in result}=={'person','animal'}


def test_prop_classification_rejects_smiley_ball_but_keeps_people_inside_a_vehicle():
    raw=[{'box':[20,20,100,100],'score':.75,'label':'a ball a balloon'},
         {'box':[20,20,100,100],'score':.4,'label':'an animal'},
         {'box':[0,0,240,200],'score':.9,'label':'a vehicle'},
         {'box':[120,20,180,180],'score':.65,'label':'a person'},
         {'box':[180,20,220,80],'score':.45,'label':'a monster'}]
    result=normalize_objects(raw,240,200)
    assert {o['label'] for o in result}=={'a person','a monster'}


def test_ball_face_and_a_crop_of_several_props_abstain_despite_higher_character_score():
    raw=[{'box':[30,30,70,80],'score':.32,'label':'ball'},
         {'box':[80,30,120,80],'score':.31,'label':'ball balloon'},
         {'box':[30,50,70,70],'score':.48,'label':'monster'},
         {'box':[20,10,140,100],'score':.55,'label':'skeleton'}]
    assert not normalize_objects(raw,200,200)


def test_creature_holding_a_small_ball_keeps_its_independent_head_detection():
    raw=[{'box':[30,30,150,160],'score':.5,'label':'monster'},
         {'box':[70,110,90,130],'score':.4,'label':'ball'}]
    assert [o['label'] for o in normalize_objects(raw,200,200)]==['monster']
    obscured=[{'box':[30,30,150,190],'score':.5,'label':'skeleton'},
              {'box':[40,40,140,140],'score':.4,'label':'ball'}]
    assert not normalize_objects(obscured,200,200)
    printed=[{'box':[30,30,150,160],'score':.5,'label':'monster'},
             {'box':[20,20,160,180],'score':.4,'label':'a sheet'}]
    assert not normalize_objects(printed,200,200)
    assert not normalize_objects([{'box':[0,0,100,100],'score':.9,'label':'a sheet'}],200,200)


def test_negative_props_also_reject_false_human_landmarks_without_erasing_held_props():
    prop={'box':[20,20,100,100],'score':.4,'label':'ball'}
    false_face=face([30,30,60,60])
    person=face([120,20,40,60])
    result=merge_subjects({'faces':[false_face,person]},[],[prop])
    assert len(result)==1 and result[0]['face']==person['face']
    creature=face([0,0,160,180])
    assert len(merge_subjects({'faces':[creature]},[],[prop]))==1


def test_head_body_tracks_bridge_nested_regions_without_crossing_character_types():
    from autoedit.video.characters import build_tracks, settings
    def subject(box,label):
        return merge_subjects({'faces':[]},[{'box':box,'label':label,'score':.6}])[0]
    samples=[{'time_sec':i/6,'faces':[
        subject([30,20,40,40] if i%2 else [20,10,80,180],'monster' if i%2 else 'skeleton'),
        subject([120,10,80,180],'person')]} for i in range(8)]
    tracks=build_tracks(samples,[],2,6,settings())
    assert len(tracks)==2
    assert len({s['faces'][0]['track_id'] for s in samples})==1
    assert {t['subject_family'] for t in tracks}=={'person','creature'}
    from autoedit.video.faces import _same_face
    assert all(_same_face(a['faces'][0],b['faces'][0]) for a,b in zip(samples,samples[1:]))
    assert not _same_face(samples[0]['faces'][0],samples[1]['faces'][1])
    # Identical boxes and similar image features cannot turn a human into a skeleton.
    conflict=[{'time_sec':i/6,'faces':[subject([20,10,80,180],label)]}
              for i,label in enumerate(['person','skeleton'])]
    assert len(build_tracks(conflict,[],1,6,settings()))==2


def test_type_evidence_prevents_similar_human_and_creature_embeddings_from_merging():
    from autoedit.video.characters import cluster_tracks
    tracks=[{'id':str(i),'vectors':[[1,0],[1,0]],'observations':[[i,0]],
             'subject_family':family} for i,family in enumerate(['person','creature','creature'])]
    groups=cluster_tracks(tracks,.2)
    assert sorted(map(len,groups))==[1,2]
    assert not any({t['subject_family'] for t in g}=={'person','creature'} for g in groups)


def test_nested_track_continuity_survives_embedding_cache(tmp_path,monkeypatch):
    import autoedit.video.characters as chars
    from autoedit.video.faces import _same_face
    model=tmp_path/'ccip.onnx'
    model.write_bytes(b'appearance model')
    info={'path':'unused.mp4','cache_identity':{'version':15},'source_kind':'live_action',
          'sample_fps':6,'duration_sec':6,'samples':[]}
    for i in range(36):
        box=[30,20,40,40] if i%2 else [20,10,80,180]
        subject=merge_subjects({'faces':[]},[{'box':box,'label':'skeleton','score':.6}])[0]
        info['samples'].append({'time_sec':i/6,'frame_width':200,'frame_height':200,'faces':[subject]})
    monkeypatch.setattr(chars,'EmbeddingModel',lambda *a:None)
    calls=[]
    def extract(video,config,*args):
        calls.append(True)
        tracks=chars.build_tracks(video['samples'],[],6,6,config)
        for track in tracks:
            track['vectors']=[[1,0],[1,0]]
        return tracks
    monkeypatch.setattr(chars,'_extract_tracks',extract)
    cfg={'embedding_model':'ccip','ccip_model':str(model)}
    first=chars.analyze_characters(info,tmp_path/'cache',cfg)
    cached=chars.analyze_characters(info,tmp_path/'cache',cfg)
    assert len(calls)==1
    for a,b in zip(cached['samples'],cached['samples'][1:]):
        assert _same_face(a,b)
    assert [(s.get('track_association'),s.get('track_previous_box')) for s in first['samples']]==[
        (s.get('track_association'),s.get('track_previous_box')) for s in cached['samples']]


def test_body_track_can_fill_a_cut_with_no_face_or_mouth():
    from autoedit.match.matcher import match_slots_to_video
    samples=[]
    for i in range(30):
        subject=merge_subjects({'faces':[]},[{'box':[20,20,90,180],'score':.65,'label':'skeleton'}])[0]
        subject.update(track_id='body-track',character_id='main-creature',character_rank=1,
                       character_status='main',identity_confidence=.8,shot_id=1)
        samples.append({**subject,'time_sec':i/6,'faces':[subject]})
    info={'sample_fps':6,'fps':24,'duration_sec':5,'samples':samples,
          'character_analysis':{'min_main_fraction':.85},'transition_times_sec':[]}
    cut=match_slots_to_video([{'id':'long','required_duration_sec':2.36}],[],info)[0]['video']
    assert cut['used_duration_sec']==2.36 and cut['selection_metrics']['tier']==3
    assert cut['selection_metrics']['face_visible']==0
    assert cut['character_selection']['best_rank']==1
    from autoedit.match.matcher import InsufficientFootageError
    import pytest
    with pytest.raises(InsufficientFootageError):
        match_slots_to_video([{'id':'long','required_duration_sec':2.36}],[],info,require_lip_motion=True)


def test_object_cache_reuses_model_work_and_rejects_incomplete_results(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import autoedit.video.objects as objects
    source=tmp_path/'video.mp4'
    source.write_bytes(b'input video validated by pipeline')
    runtime=tmp_path/'python.exe'
    runtime.touch()
    model=tmp_path/'model'
    model.mkdir()
    (model/'model.safetensors').write_bytes(b'test model')
    monkeypatch.setattr(objects,'RUNTIME',runtime)
    monkeypatch.setattr(objects,'MODEL_DIR',model)
    info={'path':str(source),'cache_identity':{'version':15},'samples':[
        {'time_sec':0,'frame_width':200,'frame_height':200,'faces':[]}]}
    calls=[]
    def worker(command,**kwargs):
        request=json.loads(__import__('pathlib').Path(command[-1]).read_text(encoding='utf-8'))
        calls.append(request)
        result={'samples':[{'time_sec':0,'objects':[{'box':[20,20,110,190],
                'label':'skeleton','score':.65}]}]}
        __import__('pathlib').Path(request['result']).write_text(json.dumps(result),encoding='utf-8')
        return SimpleNamespace(stdout=iter(['OBJECT: scanned\n']),wait=lambda:0)
    monkeypatch.setattr(objects.subprocess,'Popen',worker)
    first=objects.augment_with_objects(info,tmp_path/'cache')
    second=objects.augment_with_objects(info,tmp_path/'cache')
    assert len(calls)==2 and object_evidence(first['samples'][0]['faces'][0])
    assert first==second and info['samples'][0]['faces']==[]
    objects.augment_with_objects(info,tmp_path/'cache',{'threshold':.4})
    assert len(calls)==4
    incomplete={**info,'samples':info['samples']*2}
    import pytest
    with pytest.raises(RuntimeError,match='incomplete'):
        objects.augment_with_objects(incomplete,tmp_path/'cache')


def test_regional_requests_only_cover_bracketed_holes_without_cuts_or_time_gaps():
    from autoedit.video.objects import _regional_requests
    samples=[{'time_sec':i/6,'frame_width':200,'frame_height':200} for i in range(3)]
    actor={'box':[20,20,110,190],'score':.65,'label':'skeleton'}
    rows=[{'objects':[actor]},{'objects':[]},{'objects':[actor]}]
    requests=_regional_requests(samples,rows,[],.3)
    assert len(requests)==1 and requests[0]['time_sec']==1/6
    assert requests[0]['expected_family']=='creature'
    assert not _regional_requests(samples,rows,[.2],.3)
    assert not _regional_requests([samples[0],samples[1],{**samples[2],'time_sec':1}],rows,[],.3)


def test_regional_geometry_cannot_supply_a_subject_without_a_new_model_detection(tmp_path,monkeypatch):
    import autoedit.video.objects as objects
    source=tmp_path/'source.mp4'; source.touch()
    runtime=tmp_path/'python.exe'; runtime.touch()
    model=tmp_path/'model'; model.mkdir()
    (model/'model.safetensors').touch()
    monkeypatch.setattr(objects,'RUNTIME',runtime)
    monkeypatch.setattr(objects,'MODEL_DIR',model)
    info={'path':str(source),'cache_identity':{'version':15},'samples':[
        {'time_sec':i/6,'frame_width':200,'frame_height':200,'faces':[]} for i in range(3)]}
    actor={'box':[20,20,110,190],'score':.65,'label':'skeleton'}
    found=[]
    calls=[]
    def scan(video,*args):
        regional=bool(video['samples'][0].get('roi'))
        calls.append(regional)
        return {'samples':[{'time_sec':s['time_sec'],'objects':found if regional else
                 ([] if s['time_sec']==1/6 else [actor])} for s in video['samples']]}, {'regional':regional}
    monkeypatch.setattr(objects,'_cached_object_scan',scan)
    missing=objects.augment_with_objects(info,tmp_path,{'prompt':objects.ACTOR_PROMPT})
    assert calls==[False,True] and missing['samples'][1]['faces']==[]
    found.append(actor)
    confirmed=objects.augment_with_objects(info,tmp_path,{'prompt':objects.ACTOR_PROMPT})
    assert object_evidence(confirmed['samples'][1]['faces'][0])
