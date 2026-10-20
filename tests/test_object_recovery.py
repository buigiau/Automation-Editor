import gzip
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autoedit.video import objects, object_worker


@pytest.fixture
def request_input(tmp_path,monkeypatch):
    model = tmp_path/'model'
    model.mkdir()
    (model/'model.safetensors').write_bytes(b'model')
    monkeypatch.setattr(objects,'MODEL_DIR',model)
    source = tmp_path/'source.mp4'
    source.touch()
    return {'path':str(source),'cache_identity':{'path':str(source),'size':0,'mtime_ns':1},
            'samples':[{'time_sec':i,'frame_width':100,'frame_height':100} for i in range(3)]}


def test_empty_memory_error_is_failure_and_not_success(tmp_path,monkeypatch):
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'request_id':'current'}))
    response = {'ok':False,'request_id':'current','error':'','error_type':'MemoryError'}
    process = SimpleNamespace(stdin=io.StringIO(),stdout=iter(['RESULT: '+json.dumps(response)+'\n']))
    monkeypatch.setattr(objects.subprocess,'Popen',lambda *a,**k:process)
    with pytest.raises(RuntimeError,match='MemoryError'):
        objects.ObjectScanSession().run(['python','worker'],request,None,tmp_path/'worker.log')


def test_stale_request_response_aborts_before_result_read(tmp_path,monkeypatch):
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'request_id':'current'}))
    process = SimpleNamespace(stdin=io.StringIO(),stdout=iter(['RESULT: {"ok":true,"request_id":"previous"}\n']))
    monkeypatch.setattr(objects.subprocess,'Popen',lambda *a,**k:process)
    with pytest.raises(RuntimeError,match='stale request'):
        objects.ObjectScanSession().run(['python','worker'],request,None,tmp_path/'worker.log')


def test_partial_batches_survive_failure_and_only_missing_frames_are_retried(request_input,tmp_path):
    requests = []
    class Session:
        def run(self,command,path,*args):
            request=json.loads(path.read_text())
            requests.append(request)
            rows=[{'time_sec':s['time_sec'],'objects':[]} for s in request['samples']]
            data={'request_id':request['request_id'],'samples':rows[:1] if len(requests)==1 else rows,
                  'complete':len(requests)>1,'elapsed_sec':1,'runtime':{'device':'cuda'}}
            Path(request['result']).write_text(json.dumps(data))
            if len(requests)==1:
                raise RuntimeError('MemoryError')
    session=Session()
    with pytest.raises(RuntimeError,match='MemoryError'):
        objects._cached_object_scan(request_input,tmp_path,{},None,session)
    result,_=objects._cached_object_scan(request_input,tmp_path,{},None,session)
    assert [s['time_sec'] for s in requests[1]['samples']] == [1,2]
    assert [s['time_sec'] for s in result['samples']] == [0,1,2]
    assert result['elapsed_sec'] == 2


def test_old_result_cannot_populate_cache_after_worker_failure(request_input,tmp_path):
    class Session:
        def run(self,command,path,*args):
            request=json.loads(path.read_text())
            Path(request['result']).write_text(json.dumps({'request_id':'old', 'samples':[
                {'time_sec':0,'objects':[]}], 'complete':True}))
            raise RuntimeError('MemoryError')
    with pytest.raises(RuntimeError):
        objects._cached_object_scan(request_input,tmp_path,{},None,Session())
    assert not list(tmp_path.glob('*-frames.json.gz'))


def test_frame_cache_bound_is_in_bytes_and_default_batch_is_one():
    from collections import OrderedDict
    cache=OrderedDict((i,np.zeros((10,10,3),np.uint8)) for i in range(5))
    object_worker.trim_frames(cache,601)
    assert list(cache) == [3,4]
    assert sum(frame.nbytes for frame in cache.values()) <= 601
    assert object_worker.batch_size({},'cuda') == 1


def test_server_reports_memory_error_type_even_when_message_is_empty(tmp_path,monkeypatch,capsys):
    import sys
    path=tmp_path/'request.json'
    path.write_text(json.dumps({'request_id':'current'}))
    def fail(*a):
        raise MemoryError()
    monkeypatch.setattr(object_worker,'scan',fail)
    monkeypatch.setattr(sys,'argv',['worker','--serve'])
    monkeypatch.setattr(sys,'stdin',io.StringIO(json.dumps({'action':'scan','path':str(path),'request_id':'current'})+'\n'))
    object_worker.main()
    line=next(line for line in capsys.readouterr().out.splitlines() if line.startswith('RESULT:'))
    response=json.loads(line[8:])
    assert response == {'ok':False,'request_id':'current','error_type':'MemoryError','error':''}


def test_reference_jobs_use_one_query_without_contextual_prop_exclusions(request_input,tmp_path,monkeypatch):
    runtime=tmp_path/'python'
    runtime.touch()
    monkeypatch.setattr(objects,'RUNTIME',runtime)
    calls=[]
    class Session:
        def run(self,command,path,*args):
            request=json.loads(path.read_text())
            calls.append(request)
            rows=[{'time_sec':s['time_sec'],'objects':[{'box':[20,20,80,80],'label':'vehicle','score':.8}]}
                  for s in request['samples']]
            Path(request['result']).write_text(json.dumps({'request_id':request['request_id'],'complete':True,'samples':rows}))
    result=objects.augment_with_objects(request_input,tmp_path,{'target_references':True,'target_prompt':'a vehicle.'},session=Session())
    assert len(calls)==1
    assert calls[0]['settings']['prompt']=='a vehicle.'
    assert result['object_analysis']['inference_count']==3
    assert all(sample['faces'][0]['target_candidate_only'] for sample in result['samples'])


def test_reference_hint_is_derived_from_all_selected_regions_and_unknowns_keep_broad_query(tmp_path,monkeypatch):
    from PIL import Image
    source=tmp_path/'group.jpg'
    Image.new('RGB',(200,100),'red').save(source)
    config={'reference_images':[str(source)],'reference_regions':{str(source):[
        {'box':[0,0,100,100],'kind':'body'},{'box':[100,0,100,100],'kind':'body'}]}}
    labels=['vehicle','robot']
    def scan(info,*a):
        return {'samples':[{'objects':[{'box':s['roi'],'label':label,'score':.8}]}
                           for s,label in zip(info['samples'],labels)]},{}
    monkeypatch.setattr(objects,'_cached_object_scan',scan)
    assert objects.reference_search_prompt(config,{},tmp_path,None)=='a robot. a vehicle.'
    labels[1]='an unknown object'
    assert objects.reference_search_prompt(config,{},tmp_path,None)==objects.REFERENCE_PROMPT


def test_static_reference_scan_crops_native_pixels_and_keeps_region_coordinates(tmp_path):
    from PIL import Image
    source=tmp_path/'reference.png'
    Image.new('RGB',(200,100),'red').save(source)
    entries=list(object_worker.requested_frames({'path':str(source),'samples':[
        {'time_sec':0,'width':100,'height':50,'roi':[10,5,70,40]}]},{}))
    sample,crop,frame,transform=entries[0]
    assert crop.size==(120,70)
    assert frame.shape==(50,100,3)
    assert transform==(2.,2.,20,10)
