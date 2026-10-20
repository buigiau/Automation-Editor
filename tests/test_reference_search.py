from types import SimpleNamespace
from copy import deepcopy

import numpy as np

from autoedit.video.embeddings import EmbeddingModel
from autoedit.video.reference_search import BodyReferenceSearch, body_screening_enabled, clip_windows, ranked_clip_windows
from autoedit.video.search import candidate_times
from test_candidate_search import search_inputs


def test_only_explicit_body_crops_use_image_screening(tmp_path):
    image = str(tmp_path/'input.jpg')
    config = {'characters': {'reference_images': [image],
                            'reference_regions': {image: [{'kind': 'body', 'box': [0, 0, 100, 100]}]}}}
    assert body_screening_enabled(config)
    assert not body_screening_enabled({**config, 'reference_discovery': 'objects'})
    config['characters']['reference_regions'][image].append({'kind': 'face', 'box': [0, 0, 32, 32]})
    assert not body_screening_enabled(config)
    config['characters']['reference_regions'] = {}
    assert not body_screening_enabled(config)


def test_embedding_batches_are_bounded_and_keep_missing_crops():
    for dimension, expected in [(1, [1]*10), ('batch', [4, 4, 2])]:
        model = object.__new__(EmbeddingModel)
        model.input = SimpleNamespace(shape=[dimension, 3, 384, 384])
        model.prepare = lambda frame, face, *args: None if face < 0 else np.asarray([[[[face]]]], np.float32)
        sizes = []
        def run(tensor):
            sizes.append(len(tensor))
            return [np.tile([[1., 0.]], (len(tensor), 1))]
        model._run = run
        result = model.extract_many(None, list(range(10)) + [-1], 640, 360)
        assert sizes == expected
        assert result[-1] is None
        assert all(np.array_equal(vector, [1, 0]) for vector in result[:-1])


def test_screening_cache_is_reusable_with_new_reference_but_never_certifies_evidence(tmp_path, monkeypatch):
    import autoedit.video.reference_search as search
    path = tmp_path/'source.mp4'
    path.write_bytes(b'source identity')
    screen = object.__new__(BodyReferenceSearch)
    screen.config = {'ccip_cosine_distance': .2}
    screen.cache_dir = tmp_path
    screen.identity = {'model_sha256': 'model'}
    screen.key = ('key',)
    class Model:
        def extract_many(self, *args):
            return [[1., 0.]]*len(args[1])
    screen.runtime = {screen.key: Model()}
    screen.references = np.asarray([[1., 0.]], np.float32)
    def frames(*args, **kwargs):
        yield 6.25, np.zeros((360, 640, 3), np.uint8)
        yield 7.25, np.zeros((360, 640, 3), np.uint8)
    monkeypatch.setattr(search, 'sampled_frames', frames)
    info = screen.discover(path, 6, 8, 1, 10, {}, tmp_path, None, None)
    assert candidate_times(info, {}, tmp_path, None) == [6.25, 7.25]
    assert info['samples'] == []
    assert 'verified_ranges' not in info
    screen.references = np.asarray([[0., 1.]], np.float32)
    monkeypatch.setattr(search, 'sampled_frames', lambda *a, **k: (_ for _ in ()).throw(AssertionError('cache must avoid decoding')))
    assert screen.discover(path, 6, 8, 1, 10, {}, tmp_path, None, None)['screening_times'] == []


def test_candidate_windows_are_clip_sized_and_separated():
    windows = clip_windows([3, 4, 5, 6, 7, 8, 12, 16, 89.5], 4, 5, 90)
    assert windows == [(2.5, 7.5), (15.5, 20.5), (85., 90)]
    assert all(right-left == 5 for left, right in windows)


def test_body_crop_removes_neighbour_padding_but_rejects_a_second_subject_inside():
    model = object.__new__(EmbeddingModel)
    model.kind, model.size = 'animation', 32
    model._run = lambda tensor: [np.asarray([[1., 0.]], np.float32)]
    frame = np.zeros((200, 200, 3), np.uint8)
    body = {'face': [40, 40, 80, 80], 'subject_box': [40, 40, 80, 80], 'subject_kind': 'object'}
    neighbour = {'face': [65, 20, 30, 30]}  # centre is in padding, outside the body
    assert model.extract(frame, body, 200, 200, [body, neighbour]) is not None
    overlapping = {'face': [65, 60, 30, 30]}
    assert model.extract(frame, body, 200, 200, [body, overlapping]) is None


def test_body_search_verifies_only_clip_windows_and_releases_models(search_inputs, tmp_path, monkeypatch):
    from autoedit.video import reference_search, search
    path, scanned, dense = search_inputs
    releases = []
    class Screening:
        def __init__(self, *args):
            pass
        def discover(self, path, start, end, fps, *args):
            scanned.append((start, end, fps))
            return {'screening_times': [2, 3, 4, 20]}
        def close(self):
            releases.append(1)
    monkeypatch.setattr(reference_search, 'body_screening_enabled', lambda config: True)
    monkeypatch.setattr(reference_search, 'BodyReferenceSearch', Screening)
    checked = []
    info = search.search_video(path, {'search_fps': 1, 'source_gap_sec': 5},
        [{'required_duration_sec': 4}], tmp_path, SimpleNamespace(offload=lambda: None),
        lambda info: checked.append(info) or len(checked) == 2)
    assert scanned == [(0, 30, 1)]
    assert info['verified_ranges'] == [[1.5, 6.5], [19.5, 24.5]]
    assert dense == [(0.5, 7.5), (18.5, 25.5)]
    assert releases == [1, 1, 1]  # before both heavy stages and on exit


def test_ranked_windows_prioritize_strong_hits_and_defer_weak_ones():
    candidates = [{'time_sec': 3, 'distance': .34}, {'time_sec': 9, 'distance': .18},
                  {'time_sec': 11, 'distance': .2}, {'time_sec': 25, 'distance': .16}]
    assert ranked_clip_windows(candidates, 4, 5, 90, .23) == [(22.5, 27.5), (6.5, 11.5)]
    assert ranked_clip_windows(candidates[:1], 4, 5, 90, .36) == [(.5, 5.5)]
    isolated = [{'time_sec':10,'distance':.1}]
    assert ranked_clip_windows(isolated,1.5,5,90,.3,sample_fps=1) == []
    sustained = [{'time_sec':t,'distance':.1} for t in [9,10,11]]
    assert ranked_clip_windows(sustained,1.5,5,90,.3,sample_fps=1) == [(8.75,11.25)]


def test_fast_reference_exhaustion_restores_whole_cast_detection_before_fallback(search_inputs,tmp_path,monkeypatch):
    from autoedit.video import reference_search, search
    path, _, _ = search_inputs
    calls = []
    class Screening:
        def __init__(self,*args):
            pass
        def discover(self,*args):
            return {'screening_times':[]}
        def close(self):
            pass
    monkeypatch.setattr(reference_search,'body_screening_enabled',lambda config:True)
    monkeypatch.setattr(reference_search,'BodyReferenceSearch',Screening)
    def enrich(raw):
        fast = bool(raw.get('_body_reference_verification'))
        calls.append(fast)
        if fast:
            for sample in raw['samples']:
                sample['_raw_faces'] = deepcopy(sample['faces'])
                sample['faces'] = [{'backend':'dense-reference-body'}]
        else:
            assert all(s['faces'] and s['faces'][0].get('backend') != 'dense-reference-body' for s in raw['samples'])
        raw['character_analysis'] = {'clusters':[], 'tracks':[]}
        return raw
    info = search.search_video(path,{'search_fps':1},[{'required_duration_sec':4}],tmp_path,
        SimpleNamespace(offload=lambda:None),lambda _:False,enrich_region=enrich)
    assert info['analysis_complete']
    assert not info.get('_body_reference_verification')
    assert calls == [True,True,True,False]


def test_mandatory_lip_mode_keeps_object_face_verification(search_inputs,tmp_path,monkeypatch):
    from autoedit.video import reference_search, search
    path, _, _ = search_inputs
    class Screening:
        def __init__(self,*args):
            pass
        def discover(self,*args):
            return {'screening_times':[3]}
        def close(self):
            pass
    monkeypatch.setattr(reference_search,'body_screening_enabled',lambda config:True)
    monkeypatch.setattr(reference_search,'BodyReferenceSearch',Screening)
    def enrich(raw):
        assert not raw.get('_body_reference_verification')
        return raw
    search.search_video(path,{'search_fps':1,'require_lip_motion':True},[{'required_duration_sec':4}],tmp_path,
        SimpleNamespace(offload=lambda:None),lambda _:True,enrich_region=enrich)


def test_body_search_gathers_capacity_then_checks_strong_regions_without_early_dense_work(search_inputs,tmp_path,monkeypatch):
    from autoedit.video import reference_search, search
    path, scanned, dense = search_inputs
    class Screening:
        def __init__(self,*args):
            pass
        def discover(self,path,start,end,fps,*args):
            scanned.append((start,end,fps))
            rows = [{'time_sec':start+time,'distance':.1} for time in range(9,14)]
            return {'screening_times':[row['time_sec'] for row in rows],'screening_candidates':rows}
        def close(self):
            pass
    monkeypatch.setattr(reference_search,'body_screening_enabled',lambda config:True)
    monkeypatch.setattr(reference_search,'BodyReferenceSearch',Screening)
    info = search.search_video(path,{'search_fps':1},[{'required_duration_sec':4}],tmp_path,
        SimpleNamespace(offload=lambda:None),lambda _:True)
    assert scanned == [(0,30,1),(30,60,1),(60,90,1)]
    assert dense == [(7.5,14.5)]
    assert info['verified_ranges'] == [[8.5,13.5]]
    assert info['searched_until_sec'] == 90
