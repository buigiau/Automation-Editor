from copy import deepcopy
from types import SimpleNamespace

import pytest

from autoedit.video import search
from autoedit.video.analyzer import ANALYSIS_VERSION, analyze_video
from autoedit.video.pool import combine_video_infos, span_for_cut
from autoedit.match.matcher import match_slots_to_video, InsufficientFootageError
from test_incremental_analysis import snapshot, video, job


@pytest.fixture
def search_inputs(tmp_path,monkeypatch):
    path=video(tmp_path,seconds=90)
    scanned, dense=[],[]
    def discover(path,start,end,fps,*args):
        scanned.append((start,end,fps))
        return {'start':start,'fps':fps}
    def dense_scan(path,**kwargs):
        start,end=kwargs['start_seconds'],kwargs['max_seconds']
        dense.append((start,end))
        info=snapshot(str(path),end,duration=90)
        info['samples']=[s for s in info['samples'] if s['time_sec'] >= start]
        info.update(decoded_seconds=end,cache_identity={'version':ANALYSIS_VERSION},fps=12)
        return info
    monkeypatch.setattr(search,'discovery_chunk',discover)
    monkeypatch.setattr(search,'analyze_video',dense_scan)
    return path,scanned,dense


def test_search_checks_candidates_and_stops_before_unneeded_source(search_inputs,tmp_path,monkeypatch):
    path,scanned,dense=search_inputs
    monkeypatch.setattr(search,'candidate_times',lambda info,*a: [2] if info['start']==0 else [45])
    checked=[]
    def verify(info):
        checked.append(deepcopy(info))
        return len(checked)==2
    info=search.search_video(path,{'search_fps':1,'sample_fps':6},
        [{'required_duration_sec':4}],tmp_path,SimpleNamespace(offload=lambda:None),verify)
    assert scanned==[(0,30,1),(30,60,1)]
    assert dense==[(0,8),(39,51)]
    assert info['verified_ranges']==[[0,7],[40,50]]
    assert info['verified_seconds']==17
    assert info['searched_until_sec']==60
    assert info['dense_region_count']==2
    assert all(s['time_sec'] < 51 for s in info['samples'])


def test_search_enriches_only_new_regions_and_preserves_prior_reference_evidence(search_inputs,tmp_path,monkeypatch):
    path,scanned,dense=search_inputs
    monkeypatch.setattr(search,'candidate_times',lambda info,*a:[2] if info['start']==0 else [45])
    enriched=[]
    checked=[]
    def enrich(raw):
        enriched.append(deepcopy(raw['verified_ranges']))
        raw['character_analysis']={'min_main_fraction':.85,'reference':{'characters':[
            {'id':'target','templates':[{'id':'shared-template','image_index':0}]}]},'tracks':[],'clusters':[]}
        for sample in raw['samples']:
            sample.update(reference_match=True,reference_id='target',reference_template_id='shared-template',track_id='track-local')
        return raw
    def check(info):
        checked.append(deepcopy(info))
        return len(checked)==2
    info=search.search_video(path,{'search_fps':1},[{'required_duration_sec':4}],tmp_path,
        SimpleNamespace(offload=lambda:None),check,enrich_region=enrich)
    assert enriched==[[[0,7]],[[40,50]]]
    assert info['_regions_enriched']
    assert {s['reference_id'] for s in info['samples']}=={'shared-template'}
    assert info['samples'][0]['track_id']=='region-0:track-local'
    assert info['samples'][-1]['track_id']=='region-1:track-local'
    assert checked[0]['samples'][0]==info['samples'][0]


def test_sparse_misses_get_denser_search_before_exhaustion(search_inputs,tmp_path,monkeypatch):
    path,scanned,dense=search_inputs
    monkeypatch.setattr(search,'candidate_times',lambda info,*a:[])
    info=search.search_video(path,{'search_fps':1},[{'required_duration_sec':4}],
        tmp_path,SimpleNamespace(offload=lambda:None),lambda _:False)
    assert [fps for _,_,fps in scanned]==[1,1,1,2,2,2]
    assert dense==[(0,31),(29,61),(59,90)]
    assert info['analysis_complete']
    assert info['verified_ranges']==[[0,90]]


def test_holes_and_context_cannot_supply_qualifying_cuts():
    info=snapshot('source',12)
    info.update(fps=24,verified_ranges=[[1,3],[6,9]])
    with pytest.raises(InsufficientFootageError):
        match_slots_to_video([{'id':'too-long','required_duration_sec':4}],[],info)
    cut=match_slots_to_video([{'id':'short','required_duration_sec':2}],[],info)[0]['video']
    assert any(a <= cut['in_sec'] and cut['out_sec'] <= b for a,b in info['verified_ranges'])


def test_pool_preserves_verified_holes_and_rejects_crossing_them():
    first=snapshot('first',12)
    first.update(verified_ranges=[[1,3],[6,9]])
    second=snapshot('second',12)
    pool=combine_video_infos([first,second],2)
    assert pool['verified_ranges']==[[1,3],[6,9],[15,27]]
    with pytest.raises(ValueError,match='unverified'):
        span_for_cut(pool,{'in_sec':2,'out_sec':7})
    assert span_for_cut(pool,{'in_sec':16,'out_sec':18})['path']=='second'


def test_pipeline_transcribes_verified_ranges_and_keeps_original_source_times(job,monkeypatch):
    import autoedit.pipeline as pipeline
    config,inspect,install,scanned,speech,exports,refined=job
    config['video']['search_fps']=1
    def find(path,config,slots,cache,session,on_chunk,progress):
        scanned.append(path)
        raw=snapshot(path,14)
        raw.update(verified_ranges=[[6,14]],verified_seconds=8,searched_until_sec=30,
                   analysis_complete=False,search_fps=1,dense_region_count=1)
        assert on_chunk(raw)
        return raw
    monkeypatch.setattr(search,'search_video',find)
    plan=pipeline.run_pipeline(config)['plan']
    assert scanned==['first.mp4']
    assert speech==[('first.mp4',[(0.,30.)])]  # Stable audio core is shared across visual windows.
    cut=plan['slots'][0]['video']
    assert 6 <= cut['in_sec'] and cut['out_sec'] <= 14
    assert plan['video_analysis_meta']['analyzed_sources'][0]['verified_seconds']==8


def test_dense_range_decoder_keeps_absolute_timestamps(tmp_path,monkeypatch):
    from test_characters import face
    class Detector:
        def __init__(self,**kwargs):
            pass
        def detect(self,frame,t):
            return {**face(), 'faces':[face()]}
        def close(self):
            pass
    class Pose:
        def detect(self,*a):
            return {'person_detected':False}
        def close(self):
            pass
    monkeypatch.setattr('autoedit.video.faces.MultiRegionFaceLandmarker',Detector)
    monkeypatch.setattr('autoedit.video.pose.PoseLandmarker',Pose)
    path=video(tmp_path)
    info=analyze_video(path,start_seconds=3,max_seconds=6,cache_dir=tmp_path/'cache')
    assert info['samples'][0]['time_sec']==3
    assert info['samples'][-1]['time_sec'] < 6
    assert info['verified_ranges']==[[3,6]]
    warm=analyze_video(path,start_seconds=3,max_seconds=6,cache_dir=tmp_path/'cache')
    assert info==warm


def test_cancelled_dense_region_resumes_from_checkpoint(tmp_path,monkeypatch):
    from autoedit.runtime import cancellation,JobCancelled
    import autoedit.video.analyzer as analyzer
    from test_characters import face
    token=tmp_path/'cancel'
    measured=[]
    def detect(cv2,np,frame):
        measured.append(1)
        if len(measured)==10:
            token.touch()
        return {**face(),'faces':[face()]}
    monkeypatch.setattr(analyzer,'_detect_opencv',detect)
    path=video(tmp_path)
    with cancellation(str(token)),pytest.raises(JobCancelled):
        analyze_video(path,backend='opencv',start_seconds=3,max_seconds=6,cache_dir=tmp_path/'cache')
    measured.clear()
    resumed=analyze_video(path,backend='opencv',start_seconds=3,max_seconds=6,cache_dir=tmp_path/'cache')
    assert resumed['analysis_complete']
    assert len(measured)<18
    assert len({s['time_sec'] for s in resumed['samples']})==len(resumed['samples'])


def test_vehicle_candidates_need_consistent_reference_identity():
    from autoedit.video.objects import normalize_objects,merge_subjects,object_evidence
    raw=[{'box':[20,20,100,100],'label':'a vehicle','score':.8}]
    assert normalize_objects(raw,200,200)==[]
    candidates=normalize_objects(raw,200,200,allow_vehicle=True)
    subject=merge_subjects({'faces':[]},candidates)[0]
    assert subject['target_candidate_only']
    assert not object_evidence(subject)
    subject['reference_match']=True
    assert object_evidence(subject)


@pytest.mark.parametrize('label',['a vehicle','a robot','a toy','a ball','a balloon','a bag','a poster','a sheet of paper','an unknown object'])
def test_imported_targets_are_not_excluded_by_object_category(label):
    from autoedit.video.objects import normalize_objects,merge_subjects,object_evidence
    raw=[{'box':[20,20,100,100],'label':label,'score':.8}]
    candidates=normalize_objects(raw,200,200,positive_labels=['person'],allow_reference_targets=True)
    assert len(candidates)==1
    subject=merge_subjects({'faces':[]},candidates)[0]
    assert subject['target_candidate_only']
    assert not object_evidence(subject)
    subject['reference_match']=True
    assert object_evidence(subject)


def test_reference_search_keeps_independently_detected_actors_for_eventual_fallback():
    from autoedit.video.objects import normalize_objects,merge_subjects,object_evidence
    raw=[{'box':[20,20,100,100],'label':'an animated character','score':.8}]
    subject=merge_subjects({'faces':[]},normalize_objects(raw,200,200,allow_reference_targets=True))[0]
    assert object_evidence(subject)
    assert not subject.get('reference_match')
