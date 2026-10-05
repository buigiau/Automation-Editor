"""Incremental completion must preserve the existing footage eligibility rules."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from test_characters import face


def snapshot(path, end, duration=1800):
    samples = []
    for i in range(round(end * 6)):
        measured = {**face(), 'lip_aperture': .05 if i % 2 else .3,
                    'lip_width_ratio': .5, 'face_visible': 1., 'clear_face': 1., 'speaking': 1.}
        samples.append({**measured, 'time_sec': i/6, 'faces': [dict(measured)],
                        'frame_width': 640, 'frame_height': 360})
    return {'path': path, 'duration_sec': duration, 'analyzed_seconds': end, 'sample_fps': 6,
            'samples': samples, 'segments': [], 'transition_times_sec': [], 'shot_times_sec': []}


@pytest.fixture
def job(tmp_path, monkeypatch):
    import autoedit.pipeline as pipeline
    slot = {'id': 'one', 'nested_sequence': '1', 'nested_sequence_uid': 'uid-1',
            'required_duration_sec': 4, 'instances': [{'start_sec': 0, 'end_sec': 4}]}
    scene = {**slot, 'index': 0, 'first_start_sec': 0, 'last_end_sec': 4}
    inspect = {'slots': [slot], 'scene_slots': [scene], 'intro_slot': None}
    monkeypatch.setattr(pipeline, 'require_inputs', lambda **kw: None)
    monkeypatch.setattr(pipeline, 'inspect_prproj', lambda *a, **kw: inspect)
    monkeypatch.setattr(pipeline, 'inspect_sampler_tracks', lambda *a: {
        'path': 'template.cpr', 'tracks': [{'index': 1, 'name': 'Sampler Track 01'}]})
    monkeypatch.setattr(pipeline, 'analyze_audio_dir', lambda *a, **kw: [
        {'id': 'hi', 'path': 'hi.wav', 'name': 'hi.wav', 'duration_sec': 1,
         'category': 'HI', 'group': 'NEUTRAL', 'phonetics': {'action': 'SPEECH', 'visemes': ['A', 'E']}}])
    monkeypatch.setattr(pipeline, 'expand_match_units', lambda *a: [])
    transcribed, exported, refined, scanned = [], [], [], []
    def speech(path, ranges, *a, **kw):
        transcribed.append((path, ranges))
        return {'has_audio': False, 'words': []}
    def refine(path, slots, *a, **kw):
        refined.extend((s['video']['in_sec'], s['video']['out_sec']) for s in slots)
        from test_audio_onsets import lip_samples
        return {'samples': [sample for s in slots for sample in lip_samples(s['video']['in_sec'], s['video']['out_sec'])]}
    monkeypatch.setattr(pipeline, 'transcribe_source', speech)
    monkeypatch.setattr(pipeline, 'refine_selected_cuts', refine)
    monkeypatch.setattr(pipeline, 'prepare_video_only_source', lambda path, *a, **kw: exported.append(path) or path+'.mov')
    monkeypatch.setattr(pipeline, 'apply_cubase_plan', lambda *a: {})
    monkeypatch.setattr(pipeline, 'write_audio_review', lambda *a: None)
    config = {'job': {'output_dir': str(tmp_path)},
        'premiere': {'project': 'template.prproj', 'source_media': ['first.mp4', 'unused.mp4']},
        'cubase': {'project': 'template.cpr'},
        'video': {'source_gap_sec': 0, 'require_lip_motion': True, 'characters': {'enabled': False}}}
    def install(ends):
        def analyze(path, **kwargs):
            for end in ends:
                scanned.append((path, end))
                raw = snapshot(path, end)
                if 'on_chunk' not in kwargs or kwargs['on_chunk'](raw):
                    return raw
            return raw
        monkeypatch.setattr(pipeline, 'analyze_video', analyze)
    return config, inspect, install, scanned, transcribed, exported, refined


def test_stops_only_after_verified_allocation_and_skips_later_sources(job):
    from autoedit.pipeline import run_pipeline
    config, inspect, install, scanned, speech, exports, refined = job
    inspect['slots'][0]['instances'] *= 174
    install([2, 6, 10])
    result = run_pipeline(config)
    assert scanned == [('first.mp4', 2), ('first.mp4', 6)]
    assert speech == [('first.mp4', [(0., 6)])]
    assert exports == ['first.mp4']
    assert len(result['plan']['slots']) == 1
    meta = result['plan']['video_analysis_meta']
    assert meta['stop_reason'] == 'all_slots_verified'
    assert meta['analyzed_sources'][0]['analyzed_seconds'] == 6
    assert result['plan']['slots'][0]['video']['selection_metrics']['dense_lip_motion_verified']
    assert json.loads((Path(config['job']['output_dir'])/'performance.json').read_text())['status'] == 'completed'


def test_missing_intro_continues_without_exporting_or_transcribing(job):
    from autoedit.pipeline import run_pipeline
    config, inspect, install, scanned, speech, exports, refined = job
    inspect['intro_slot'] = {'id': 'intro', 'status': 'empty', 'start_sec': 0, 'end_sec': 1,
                            'required_duration_sec': 1, 'video_track_index': 1}
    install([4, 8, 12])
    result = run_pipeline(config)
    assert scanned == [('first.mp4', 4), ('first.mp4', 8)]
    assert speech == [('first.mp4', [(0., 8)])]
    cut, intro = result['plan']['slots'][0]['video'], result['plan']['intro_fill']
    assert intro['out_sec'] <= cut['in_sec'] or cut['out_sec'] <= intro['in_sec']


def test_dense_failures_continue_and_are_not_retried_after_extension(job, monkeypatch):
    import autoedit.pipeline as pipeline
    config, inspect, install, scanned, speech, exports, refined = job
    original = pipeline.refine_selected_cuts
    def verify(path, slots, *a, **kw):
        assert not exports
        result = original(path, slots, *a, **kw)
        for sample in result['samples']:
            if slots[0]['video']['in_sec'] < 8:
                sample['lip_aperture'] = .2
        return result
    monkeypatch.setattr(pipeline, 'refine_selected_cuts', verify)
    install([6, 18, 24])
    result = pipeline.run_pipeline(config)
    assert scanned == [('first.mp4', 6), ('first.mp4', 18)]
    assert len(refined) == len(set(refined))
    assert result['plan']['slots'][0]['video']['in_sec'] >= 8


def test_full_mode_processes_every_selected_source(job):
    from autoedit.pipeline import run_pipeline
    config, inspect, install, scanned, speech, exports, refined = job
    config['video']['analysis_mode'] = 'full'
    install([12])
    result = run_pipeline(config)
    assert scanned == [('first.mp4', 12), ('unused.mp4', 12)]
    assert result['plan']['video_analysis_meta']['stop_reason'] == 'full_scan'
    assert len(exports) == 1  # Unused sources do not need a video-only asset.


def test_exhausted_sources_preserve_previous_plan(job):
    from autoedit.pipeline import run_pipeline
    from autoedit.match.matcher import InsufficientFootageError
    config, inspect, install, scanned, speech, exports, refined = job
    config['premiere']['source_media'] = ['first.mp4']
    install([2])
    dest = Path(config['job']['output_dir'])/'edit-plan.json'
    dest.write_text('previous plan')
    with pytest.raises(InsufficientFootageError):
        run_pipeline(config)
    assert not speech and not exports
    assert dest.read_text() == 'previous plan'


def test_model_errors_abort_instead_of_reading_more_video(job, monkeypatch):
    import autoedit.pipeline as pipeline
    config, inspect, install, scanned, speech, exports, refined = job
    config['video']['characters']['enabled'] = True
    def broken(*a, **kw):
        raise ValueError('Invalid model weights')
    monkeypatch.setattr('autoedit.video.objects.augment_with_objects', broken)
    install([6, 12])
    with pytest.raises(ValueError, match='model weights'):
        pipeline.run_pipeline(config)
    assert scanned == [('first.mp4', 6)]
    assert not exports


def video(tmp_path, seconds=8):
    import cv2
    path = tmp_path/'source.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 12, (96, 64))
    assert writer.isOpened()
    for _ in range(seconds*12):
        writer.write(np.zeros((64, 96, 3), np.uint8))
    writer.release()
    return path


def test_decoder_stops_and_resumes_with_context_without_repeating_prefix(tmp_path, monkeypatch):
    import autoedit.video.analyzer as analyzer
    source = video(tmp_path)
    detected = []
    def detect(*args):
        detected.append(1)
        return {**face(), 'face_visible': 1, 'faces': [face()]}
    monkeypatch.setattr(analyzer, '_detect_opencv', detect)
    checkpoints = []
    first = analyzer.analyze_video(source, backend='opencv', cache_dir=tmp_path/'cache', chunk_sec=2,
        on_chunk=lambda info: checkpoints.append(info['analyzed_seconds']) or True)
    assert checkpoints == [2]
    assert first['samples'][-1]['time_sec'] == pytest.approx(3)
    assert len(detected) == 19
    detected.clear()
    checkpoints.clear()
    resumed = analyzer.analyze_video(source, backend='opencv', cache_dir=tmp_path/'cache', chunk_sec=2,
        on_chunk=lambda info: checkpoints.append(info['analyzed_seconds']) or info['analyzed_seconds'] >= 4)
    assert checkpoints == [2, 4]
    assert len(detected) == 19  # Warm up 2-3s, then only new frames through 5s.
    times = [s['time_sec'] for s in resumed['samples']]
    assert len(times) == len(set(times))
    assert not resumed['shot_times_sec']
    assert resumed['samples'][:19] == first['samples']


def test_complete_cache_replays_prefixes_and_propagates_callback_errors(tmp_path, monkeypatch):
    import autoedit.video.analyzer as analyzer
    source = video(tmp_path)
    monkeypatch.setattr(analyzer, '_detect_opencv', lambda *a: {**face(), 'faces': [face()]})
    analyzer.analyze_video(source, backend='opencv', cache_dir=tmp_path/'cache')
    def unexpected(*a):
        pytest.fail('Complete cache must not decode/detect again')
    monkeypatch.setattr(analyzer, '_detect_opencv', unexpected)
    def invalid(info):
        raise ValueError('Broken downstream model')
    with pytest.raises(ValueError, match='downstream model'):
        analyzer.analyze_video(source, backend='opencv', cache_dir=tmp_path/'cache', on_chunk=invalid)
    result = analyzer.analyze_video(source, backend='opencv', cache_dir=tmp_path/'cache', chunk_sec=2,
                                    on_chunk=lambda info: True)
    assert result['analyzed_seconds'] == 2
    assert result['samples'][-1]['time_sec'] <= 3


def test_source_words_have_one_owner_and_old_audio_windows_are_reused():
    from autoedit.audio.source_speech import IncrementalSpeech
    calls = []
    def transcribe(path, ranges, *a, **kw):
        calls.append(ranges)
        return {'has_audio': True, 'words': [{'word': 'hello', 'start': 29.8, 'end': 30.2}], 'events': []}
    session = IncrementalSpeech(transcribe, 30)
    first = session.scan('video', 30, 'cache')
    second = session.scan('video', 60, 'cache')
    assert not first['words']
    assert len(second['words']) == 1
    assert calls == [[(0., 30.)], [(30., 60.)]]


def test_crop_cache_reuses_closed_tracks_and_model_between_prefixes(tmp_path, monkeypatch):
    import autoedit.video.characters as chars
    path = tmp_path/'model.onnx'
    path.write_bytes(b'fake')
    extracted, created = [], []
    class Model:
        execution = {'device': 'cpu'}
        def __init__(self, *a):
            created.append(1)
        def extract(self, frame, *a):
            extracted.append(float(frame[0, 0, 0]))
            return np.array([1., 0.], np.float32)
    monkeypatch.setattr(chars, 'EmbeddingModel', Model)
    monkeypatch.setattr(chars, '_selected_source_frames', lambda path, times:
        iter((t, np.full((2, 2, 3), t)) for t in times))
    config = {'arcface_model': str(path), 'embeddings_per_track': 2}
    runtime = {}
    for end in (2, 4):
        info = snapshot('video', end)
        info['shot_times_sec'] = info['transition_times_sec'] = [2.]
        chars.analyze_characters(info, tmp_path/'cache', config, runtime_cache=runtime)
    assert created == [1]
    assert len(extracted) == len(set(extracted)) == 4


def test_object_cache_infers_only_new_timestamps(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import autoedit.video.objects as objects
    source = tmp_path/'video.mp4'; source.touch()
    runtime = tmp_path/'python.exe'; runtime.touch()
    model = tmp_path/'model'; model.mkdir(); (model/'model.safetensors').touch()
    monkeypatch.setattr(objects, 'RUNTIME', runtime)
    monkeypatch.setattr(objects, 'MODEL_DIR', model)
    requests = []
    def worker(command, **kw):
        request = json.loads(Path(command[-1]).read_text())
        requests.append(request)
        data = {'samples': [{'time_sec': s['time_sec'], 'objects': []} for s in request['samples']]}
        Path(request['result']).write_text(json.dumps(data))
        return SimpleNamespace(stdout=iter([]), wait=lambda: 0)
    monkeypatch.setattr(objects.subprocess, 'Popen', worker)
    for count in (1, 2, 2):
        info = {'path': str(source), 'cache_identity': {'source': 'stable'}, 'samples': [
            {'time_sec': i, 'frame_width': 96, 'frame_height': 64, 'faces': []} for i in range(count)]}
        objects.augment_with_objects(info, tmp_path/'cache')
    assert [r['samples'][0]['time_sec'] for r in requests] == [0, 0, 1, 1]
    assert all(len(r['samples']) == 1 for r in requests)


def test_two_object_queries_share_decoded_frames(tmp_path, monkeypatch):
    from collections import OrderedDict
    import av
    from autoedit.video.object_worker import requested_frames
    source = video(tmp_path)
    request = {'path': str(source), 'samples': [{'time_sec': i, 'width': 96, 'height': 64} for i in range(3)]}
    cache = OrderedDict()
    first = list(requested_frames(request, cache))
    def unexpected(*a, **kw):
        pytest.fail('Second query must use decoded frames')
    monkeypatch.setattr(av, 'open', unexpected)
    second = list(requested_frames(request, cache))
    assert [entry[0]['time_sec'] for entry in first] == [entry[0]['time_sec'] for entry in second]
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(first, second))


def test_object_batch_oom_retries_without_duplicate_rows(monkeypatch):
    import sys
    from types import SimpleNamespace
    from collections import OrderedDict
    import autoedit.video.object_worker as worker
    class OOM(RuntimeError):
        pass
    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None,
        is_available=lambda: True, mem_get_info=lambda *a: (8*1024**3, 8*1024**3)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(worker, 'load_detector', lambda *a: ('processor', 'model', 'cuda'))
    samples = [{'time_sec': i, 'width': 96, 'height': 64} for i in range(3)]
    monkeypatch.setattr(worker, 'requested_frames', lambda *a: (
        (s, 'image', np.zeros((64, 96, 3), np.uint8), None) for s in samples))
    calls = []
    def detect(processor, model, device, images, settings):
        calls.append(len(images))
        if len(calls) == 1:
            raise OOM()
        return [[] for _ in images]
    monkeypatch.setattr(worker, 'detect_batch', detect)
    result = worker.scan({'path': 'video', 'model_dir': 'model', 'samples': samples,
        'settings': {'device': 'auto', 'prompt': 'a person.', 'threshold': .3}}, {}, OrderedDict())
    assert calls == [3, 2, 1]
    assert [row['time_sec'] for row in result['samples']] == [0, 1, 2]


def test_dense_cut_cache_survives_changing_other_selected_cuts(tmp_path, monkeypatch):
    import autoedit.video.refine as refine
    source = video(tmp_path)
    measured = []
    class Detector:
        def __init__(self, **kw):
            pass
        def detect(self, frame, time):
            measured.append(time)
            return {**face(), 'faces': [face()]}
        def close(self):
            pass
    monkeypatch.setattr('autoedit.video.faces.MultiRegionFaceLandmarker', Detector)
    slots = [{'video': {'in_sec': start, 'out_sec': start+1}} for start in (0, 2)]
    refine.refine_selected_cuts(source, slots[:1], tmp_path/'cache')
    count = len(measured)
    refine.refine_selected_cuts(source, slots, tmp_path/'cache')
    assert len(measured) > count
    assert all(t >= 1.65 for t in measured[count:])  # includes onset context, never redecodes the first cut


def test_object_service_offloads_and_restores_weights_without_loading_again(tmp_path, monkeypatch, capsys):
    import io
    import sys
    from types import SimpleNamespace
    import autoedit.video.object_worker as worker
    class OOM(RuntimeError):
        pass
    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None,
        is_available=lambda: True, device_count=lambda: 1))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    loaded, moved = [], []
    class Model:
        def to(self, device):
            moved.append(device)
            return self
    def load(*a):
        loaded.append(1)
        return 'processor', Model(), 'cuda'
    monkeypatch.setattr(worker, 'load_detector', load)
    expected = {'time_sec': 0, 'width': 96, 'height': 64}
    monkeypatch.setattr(worker, 'requested_frames', lambda *a: (
        (expected, 'image', np.zeros((64, 96, 3), np.uint8), None) for _ in range(1)))
    monkeypatch.setattr(worker, 'detect_batch', lambda *a: [[]])
    request = tmp_path/'request.json'
    request.write_text(json.dumps({'path': 'video', 'model_dir': 'model', 'samples': [expected],
        'result': str(tmp_path/'result.json'),
        'settings': {'device': 'auto', 'batch_size': 1, 'prompt': 'a person.', 'threshold': .3}}))
    monkeypatch.setattr(sys, 'argv', ['worker', '--serve'])
    monkeypatch.setattr(sys, 'stdin', io.StringIO('\n'.join(map(json.dumps,
        [str(request), {'action': 'offload'}, str(request)])) + '\n'))
    worker.main()
    assert loaded == [1]
    assert moved == ['cpu', 'cuda:0']
    assert capsys.readouterr().out.count('RESULT: {}') == 3


def test_cli_exposes_analysis_settings():
    from autoedit.cli import build_parser
    args = build_parser().parse_args(['run', '--analysis-mode', 'full', '--analysis-chunk-sec', '45'])
    assert args.analysis_mode == 'full' and args.analysis_chunk_sec == 45


@pytest.mark.parametrize('device,error,should_retry', [
    ('auto', 'CUDA error: no kernel image is available', True),
    ('cuda', 'CUDA error: no kernel image is available', False),
    ('auto', 'Invalid model tensor shape', False),
])
def test_object_gpu_compatibility_fallback_does_not_hide_model_errors(monkeypatch, device, error, should_retry):
    import sys
    from types import SimpleNamespace
    from collections import OrderedDict
    import autoedit.video.object_worker as worker
    class OOM(RuntimeError):
        pass
    torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=OOM, empty_cache=lambda: None,
        is_available=lambda: True, mem_get_info=lambda *a: (8*1024**3, 8*1024**3)))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    class Model:
        def to(self, device):
            return self
    monkeypatch.setattr(worker, 'load_detector', lambda *a: ('processor', Model(), 'cuda'))
    expected = {'time_sec': 0, 'width': 96, 'height': 64}
    monkeypatch.setattr(worker, 'requested_frames', lambda *a: (
        (expected, 'image', np.zeros((64, 96, 3), np.uint8), None) for _ in range(1)))
    devices = []
    def detect(processor, model, selected, images, settings):
        devices.append(selected)
        if selected.startswith('cuda'):
            raise RuntimeError(error)
        return [[] for _ in images]
    monkeypatch.setattr(worker, 'detect_batch', detect)
    request = {'path': 'video', 'model_dir': 'model', 'samples': [expected],
        'settings': {'device': device, 'prompt': 'a person.', 'threshold': .3}}
    if should_retry:
        result = worker.scan(request, {}, OrderedDict())
        assert devices == ['cuda', 'cpu']
        assert result['runtime']['device'] == 'cpu'
        assert result['runtime']['fallback_reason'] == error
    else:
        with pytest.raises(RuntimeError, match=error):
            worker.scan(request, {}, OrderedDict())


def test_context_cannot_supply_a_cut_beyond_the_committed_prefix():
    from autoedit.match.matcher import match_slots_to_video, InsufficientFootageError
    info = snapshot('video', 7)
    info['analyzed_seconds'] = 6
    with pytest.raises(InsufficientFootageError):
        match_slots_to_video([{'id': 'too-long', 'required_duration_sec': 6.5}], [], info, min_gap_sec=0)
