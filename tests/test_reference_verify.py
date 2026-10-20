from copy import deepcopy

import numpy as np

from autoedit.video.characters import settings
from autoedit.video.objects import object_evidence
from autoedit.video.reference_verify import qualify_body_references
from autoedit.video.reference_search import screening_tiles
from test_incremental_analysis import snapshot


def test_dense_identity_samples_require_measured_features_and_keep_unmatched_holes(monkeypatch, tmp_path):
    raw = snapshot('source', 4)
    raw['verified_ranges'] = [[0,4]]
    count = len(raw['samples'])
    values = np.tile([1.,0.],(count,len(screening_tiles(640,360)),1)).astype(np.float32)
    values[count//2] = [0.,1.]
    catalog = {'images':['body.png'], 'templates':[{'id':'target','image_index':0,'subject_kind':'object'}],
               'skipped_regions':[]}
    monkeypatch.setattr('autoedit.video.references.reference_vectors',lambda *a: ([[1.,0.]],deepcopy(catalog)))
    groups = [{'id':'target','template_indices':[0],'templates':catalog['templates']}]
    info = qualify_body_references(raw, values, np.asarray([[1.,0.]],np.float32), groups,
        settings({'reference_images':['body.png']}),tmp_path,{'model_sha256':'model'}, {})
    assert not info['samples'][count//2]['reference_match']
    measured = [s for s in info['samples'] if s.get('backend') == 'dense-reference-body']
    assert len(measured) == count-1
    assert all(object_evidence(s) for s in measured)
    assert all(not s['face_detected'] and not s['lip_geometry_measured'] and not s['speaking'] for s in measured)
    assert not object_evidence({**measured[0], 'reference_model_sha256':None})
    assert not object_evidence({**measured[0], 'reference_match':False})
    assert not object_evidence({**measured[0], 'reference_sample_fps':1})
    assert not object_evidence({**measured[0], 'reference_distance':.5})
    assert info['samples'][0]['_raw_faces'] == raw['samples'][0]['faces']
    assert raw['samples'][0].get('backend') != 'dense-reference-body'


def test_body_template_keeps_body_box_and_does_not_swallow_two_separate_subjects():
    from autoedit.video.objects import normalize_objects
    nested = [{'box':[20,20,100,140],'score':.8,'label':'a character'},
              {'box':[40,25,80,60],'score':.7,'label':'a character'}]
    result = normalize_objects(nested,200,200,allow_reference_targets=True,prefer_body_boxes=True)
    assert [obj['box'] for obj in result] == [[20,20,80,120]]
    crowd = [{'box':[0,0,200,200],'score':.9,'label':'an object'},
             {'box':[10,10,80,150],'score':.8,'label':'a character'},
             {'box':[110,10,190,150],'score':.8,'label':'a character'}]
    result = normalize_objects(crowd,200,200,allow_reference_targets=True,prefer_body_boxes=True)
    assert len(result) == 2
    assert all(obj['box'][2] < 100 for obj in result)


def test_loose_discovery_and_confirmed_views_cannot_certify_a_similar_other_character(monkeypatch,tmp_path):
    raw = snapshot('source',4)
    raw['verified_ranges'] = [[0,4]]
    # Similar colours/class fit the broad body limit, but fail original identity.
    vector = np.asarray([.73,(1-.73**2)**.5],np.float32)
    values = np.tile(vector,(len(raw['samples']),len(screening_tiles(640,360)),1))
    catalog = {'images':['body.png'],'templates':[{'id':'target','image_index':0,'subject_kind':'object'}],
               'skipped_regions':[]}
    monkeypatch.setattr('autoedit.video.references.reference_vectors',lambda *a:([[1.,0.]],deepcopy(catalog)))
    groups = [{'id':'target','template_indices':[0],'templates':catalog['templates']}]
    config = settings({'reference_images':['body.png'],'body_reference_cosine_distance':.30})
    gallery = {('model',('target',),'object'):[vector.tolist()]}
    info = qualify_body_references(raw,values,np.asarray([[1.,0.]],np.float32),groups,
        config,tmp_path,{'model_sha256':'model'},gallery)
    assert not any(s.get('reference_match') for s in info['samples'])


def test_background_change_requires_consistent_dense_identity_and_keeps_hard_cuts(monkeypatch,tmp_path):
    raw = snapshot('source',4)
    raw.update(verified_ranges=[[0,4]],transition_times_sec=[1.,2.],shot_times_sec=[2.])
    count = len(raw['samples'])
    values = np.tile([1.,0.],(count,len(screening_tiles(640,360)),1)).astype(np.float32)
    catalog = {'images':['body.png'],'templates':[{'id':'target','image_index':0,'subject_kind':'object'}],'skipped_regions':[]}
    monkeypatch.setattr('autoedit.video.references.reference_vectors',lambda *a:([[1.,0.]],deepcopy(catalog)))
    groups = [{'id':'target','template_indices':[0],'templates':catalog['templates']}]
    config = settings({'reference_images':['body.png']})
    def qualify(features):
        return qualify_body_references(raw,features,np.asarray([[1.,0.]],np.float32),groups,
            config,tmp_path,{'model_sha256':'model'}, {})
    info = qualify(values)
    assert info['transition_times_sec'] == [2.]
    assert info['unfiltered_transition_times_sec'] == [1.,2.]
    assert info['reference_background_events_ignored_sec'] == [1.]
    # A missed target in either bracket must retain the uncertainty event.
    missing = values.copy()
    index = min(range(count),key=lambda i:abs(raw['samples'][i]['time_sec']-1))
    missing[index] = [0.,1.]
    assert qualify(missing)['transition_times_sec'] == [1.,2.]
    # Two individually similar views still cannot suppress a visual jump.
    jump = values.copy()
    for i,sample in enumerate(raw['samples']):
        jump[i] = [.9, (1-.9**2)**.5 * (1 if sample['time_sec'] < 1 else -1)]
    assert qualify(jump)['transition_times_sec'] == [1.,2.]
