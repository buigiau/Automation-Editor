from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import re

import av
import numpy as np
import pytest

from autoedit.ai.cache import identity
from autoedit.ai.local import (allocate, analyze_sources, appearance_runs, local_settings,
                               scan_source, validate_observations)
from autoedit.ai.ollama import LocalAIError, OllamaClient, validate_endpoint
from autoedit.ai.ollama import LocalOutputError
from autoedit.ai.selection import validate_selection
from autoedit.runtime import cancellation, JobCancelled


def test_tail_transition_fallback_keeps_the_opening_stable(tmp_path, monkeypatch):
    from autoedit.ai import local
    evidence = {'safe': False, 'events': [8., 8.5, 9.28]}
    monkeypatch.setattr(local, 'read', lambda _: evidence)
    source = {'identity': {}}
    row = {'in_sec': 0., 'out_sec': 9.28}
    assert not local.transition_safe(source, row, tmp_path)
    assert local.transition_safe(source, row, tmp_path, tail_tolerance_sec=2)
    evidence['events'].append(7.)
    assert not local.transition_safe(source, row, tmp_path, tail_tolerance_sec=2)


def test_native_dissolve_events_are_sorted_before_bisection(media, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoedit.ai import local
    from autoedit.video import reader, transitions
    monkeypatch.setattr(transitions, 'TransitionDetector', lambda: SimpleNamespace(events=[.6, .2, .4, .2]))
    monkeypatch.setattr(reader, 'sampled_frames', lambda *a, **kw: iter(()))
    original = transitions.transition_metrics
    def metrics(events, start, end):
        assert events == [.2, .4, .6]
        return original(events, start, end)
    monkeypatch.setattr(transitions, 'transition_metrics', metrics)
    assert not local.transition_safe(media, {'in_sec': .1, 'out_sec': 1.}, tmp_path)


@pytest.mark.parametrize('url', ['https://example.com', 'http://192.168.1.2:11434',
    'http://localhost.evil.test:11434', 'http://key@localhost:11434',
    'http://127.0.0.1:11434/api', 'http://127.0.0.1:11434?x=1'])
def test_local_backend_never_accepts_a_remote_endpoint(url):
    with pytest.raises(ValueError):
        validate_endpoint(url)


def test_cloud_and_nonvision_models_are_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        OllamaClient('qwen3-vl:235b-cloud', tmp_path)
    client = OllamaClient('qwen3-vl:4b-instruct', tmp_path)
    monkeypatch.setattr(client, 'ensure_server', lambda: None)
    def request(route, body=None):
        return {'models': [{'name': client.model, 'digest': 'abc'}]} if route == '/api/tags' else {'capabilities': ['completion']}
    monkeypatch.setattr(client, 'request', request)
    with pytest.raises(LocalAIError, match='vision'):
        client.identity()
    monkeypatch.setattr(client, 'request', lambda route, body=None:
        request(route, body) if route == '/api/tags' else {'capabilities': ['vision'], 'remote_host': 'cloud.test'})
    with pytest.raises(LocalAIError, match='Cloud'):
        client.identity()


@pytest.fixture
def media(tmp_path):
    path = tmp_path / 'two-shots.mkv'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=24)
        stream.width, stream.height, stream.pix_fmt = 160, 90, 'bgr0'
        for i in range(144):
            array = np.zeros((90, 160, 3), dtype=np.uint8)
            array[:, :, 0 if i < 72 else 2] = 230
            array[20:60, 20:60] = 100
            frame = av.VideoFrame.from_ndarray(array, format='rgb24')
            frame.pts, frame.time_base = i, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return {'id': 's', 'path': str(path), 'identity': identity(path), 'duration_sec': 6.,
            'analyzed_start_sec': 0., 'analyzed_end_sec': 6.}


def test_native_cuts_and_requested_fps_cover_full_source_and_cache(media, tmp_path, monkeypatch):
    scan, folder = scan_source(media, 3, tmp_path/'cache', lambda _: None)
    assert len(scan['frames']) == 18
    assert scan['frames'][-1]['time'] > 5.6
    assert scan['cuts'] == pytest.approx([0, 3, 6])
    assert all((folder/f['image']).is_file() for f in scan['frames'])
    monkeypatch.setattr(av, 'open', lambda *a, **k: pytest.fail('Resumed scan decoded again'))
    cached, _ = scan_source(media, 3, tmp_path/'cache', lambda _: None)
    assert cached == scan


class Observer:
    def __init__(self):
        self.calls = []

    def chat(self, prompt, images, schema):
        count = int(re.search(r'#0\.\.#(\d+)', prompt).group(1))+1
        self.calls.append(images)
        return {'characters': [{'id': 'red-car', 'description': 'a talking car', 'reference_ids': [],
            'ranges': [{'first': 0, 'last': count-1, 'mouth': 2, 'voice_hint': 'speech'}]}]}


def contract_for(media):
    return {'backend_identity': {'model': 'test', 'digest': 'abc'},
        'local_analysis': local_settings({'local_batch_frames': 6}),
        'sources': [media], 'references': [], 'video_sampling': {'video_fps': 3},
        'slots': [{'id': 'a', 'required_duration_sec': 1}, {'id': 'b', 'required_duration_sec': 1}],
        'voices': [{'voice_id': 'v', 'name': 'Hi', 'duration_sec': 1}], 'source_gap_sec': .5}


def test_every_batch_is_observed_and_reused_with_safe_native_shot_bounds(media, tmp_path):
    contract = contract_for(media)
    observer = Observer()
    analysis = analyze_sources(contract, observer, tmp_path/'cache', lambda _: None)
    assert len(observer.calls) == 3  # all six seconds, despite only two fills
    again = analyze_sources(contract, observer, tmp_path/'cache', lambda _: None)
    assert len(observer.calls) == 3 and analysis == again
    characters, runs = appearance_runs(*analysis)
    result = allocate(contract, characters, runs, tmp_path/'cache')
    rows, errors, _ = validate_selection(result, contract['slots'], [media], contract['voices'], gap=.5)
    assert not errors and len(rows) == 2
    assert all(row['audio_uncertain'] for row in rows)
    assert all(not row['in_sec'] < 3 < row['out_sec'] for row in rows)
    assert len({row['shot_id'] for row in rows}) == 2


def test_incomplete_batches_split_without_losing_frames_or_cached_progress(media, tmp_path):
    class SplittingObserver(Observer):
        def chat(self, prompt, images, schema):
            count = int(re.search(r'#0\.\.#(\d+)', prompt).group(1))+1
            if count > 3:
                self.calls.append(images)
                raise LocalOutputError('Incomplete response')
            return super().chat(prompt, images, schema)
    observer = SplittingObserver()
    contract = contract_for(media)
    registry, evidence, scans = analyze_sources(contract, observer, tmp_path/'cache', lambda _: None)
    assert len(observer.calls) == 9
    assert sorted(e['frame_index'] for e in evidence) == list(range(18))
    analyze_sources(contract, observer, tmp_path/'cache', lambda _: None)
    assert len(observer.calls) == 9


def test_duplicate_character_ids_merge_ranges_without_inventing_visibility():
    one = {'id': 'car', 'description': 'car', 'reference_ids': [],
           'ranges': [{'first': 0, 'last': 1, 'mouth': 2, 'voice_hint': 'speech'}]}
    two = deepcopy(one)
    two['ranges'][0].update(first=4, last=5)
    result = validate_observations({'characters': [one, two]}, 6, [])
    assert len(result['characters']) == 1
    assert [(s['first'], s['last']) for s in result['characters'][0]['ranges']] == [(0, 1), (4, 5)]


def test_fallback_keeps_source_gap_and_never_repeats_selected_interval(tmp_path, monkeypatch):
    import autoedit.ai.visual as visual
    monkeypatch.setattr('autoedit.ai.local.transition_safe', lambda *a: True)
    monkeypatch.setattr(visual, 'fingerprints', lambda *a: [])
    monkeypatch.setattr(visual, 'similar', lambda *a: False)
    media = {'id': 's', 'duration_sec': 20, 'analyzed_end_sec': 20}
    contract = contract_for(media)
    contract['source_gap_sec'] = 5
    characters = [{'id': 'car', 'description': 'car', 'reference_ids': [],
        'appearances': [{'source_id': 's', 'start_sec': 0, 'end_sec': 20}]}]
    runs = [{'character_id': 'car', 'source_id': 's', 'start_sec': 0, 'end_sec': 20,
        'shot': 0, 'shot_start_sec': 0, 'shot_end_sec': 20, 'mouth': 3, 'voice_hint': 'speech'}]
    first = allocate(contract, characters, runs, tmp_path)
    assert len(first['selections']) == 1
    result = allocate(contract, characters, runs, tmp_path, allow_same_shot=True)
    rows, errors, _ = validate_selection(result, contract['slots'], [media], contract['voices'], gap=5, allow_same_shot=True)
    assert not errors and len(rows) == 2 and rows[1]['same_shot_fallback']
    a, b = sorted(rows, key=lambda r: r['in_sec'])
    assert b['in_sec'] - a['out_sec'] >= 5


def test_invalid_or_cancelled_analysis_is_not_cached(media, tmp_path):
    with pytest.raises(LocalAIError):
        validate_observations({'characters': [{'id': 'car', 'description': 'car', 'reference_ids': [],
            'ranges': [{'first': 0, 'last': 99, 'mouth': 2, 'voice_hint': 'speech'}]}]}, 6, [])
    token = tmp_path/'cancel.request'
    token.touch()
    observer = Observer()
    with pytest.raises(JobCancelled), cancellation(token):
        analyze_sources(contract_for(media), observer, tmp_path/'cache', lambda _: None)
    assert not observer.calls


class Stream:
    def __init__(self, items):
        self.items = items
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __iter__(self):
        return iter(json.dumps(i).encode() for i in self.items)
    def read(self):
        return b'{}'


def test_local_transport_sends_only_images_and_checks_stream_completion(tmp_path, monkeypatch):
    image = tmp_path/'frame.jpg'
    image.write_bytes(b'local-image')
    client = OllamaClient('qwen3-vl:4b-instruct', tmp_path)
    monkeypatch.setattr(client, 'check_resources', lambda: None)
    requests = []
    class Opener:
        def open(self, request, **kwargs):
            requests.append(request)
            return Stream([{'message': {'content': '{"characters":[]}'}, 'done': True, 'done_reason': 'stop'}])
    client.opener = Opener()
    assert client.chat('inspect', [image], {}) == {'characters': []}
    body = json.loads(requests[0].data)
    assert body['stream'] and not body['think'] and body['messages'][0]['images']
    assert body['options']['num_gpu'] == 0 and body['options']['num_thread'] == 2
    assert body['keep_alive'] == '30s'
    assert not any('key' in k.lower() for k in requests[0].headers)
    client.opener.open = lambda *a, **k: Stream([{'message': {'content': '{}'}}])
    with pytest.raises(LocalAIError, match='interrupted'):
        client.chat('inspect', [image], {})


def test_low_free_ram_stops_cpu_before_loading_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr('autoedit.runtime.memory_snapshot', lambda: {'ram_available_bytes': 6 * 1024**3})
    client = OllamaClient('qwen3-vl:4b-instruct', tmp_path)
    with pytest.raises(LocalAIError, match='8 GB'):
        client.chat('inspect', [], {})


def test_ram_guard_allows_the_memory_consumed_by_its_own_loaded_model(tmp_path, monkeypatch):
    available = {'ram_available_bytes': 10 * 1024**3}
    monkeypatch.setattr('autoedit.runtime.memory_snapshot', lambda: available)
    client = OllamaClient('qwen3-vl:4b-instruct', tmp_path)
    class Opener:
        def open(self, *a, **k):
            return Stream([{'message': {'content': '{}'}, 'done': True}])
    client.opener = Opener()
    client.chat('inspect', [], {})
    available['ram_available_bytes'] = 5 * 1024**3
    assert client.chat('inspect again', [], {}) == {}
    client.unload()
    with pytest.raises(LocalAIError, match='8 GB'):
        client.check_resources()


def test_unload_runs_even_after_cancel_and_does_not_start_a_server(tmp_path):
    client = OllamaClient('qwen3-vl:4b-instruct', tmp_path)
    requests = []
    class Opener:
        def open(self, request, **kwargs):
            requests.append(json.loads(request.data))
            return Stream([])
    client.opener = Opener()
    token = tmp_path/'cancel.request'
    with cancellation(token):
        token.touch()
        client.unload()
    assert requests == [{'model': client.model, 'keep_alive': 0}]


@pytest.mark.parametrize('cause', ['memory', 'timeout', 'cancel'])
def test_prefill_socket_is_interrupted_without_waiting_for_a_generated_token(tmp_path, monkeypatch, cause):
    import threading
    import time
    import autoedit.ai.ollama as ollama
    monkeypatch.setattr(ollama, 'STREAM_POLL_SECONDS', .005)
    monkeypatch.setattr('autoedit.runtime.memory_snapshot', lambda:
        {'ram_available_bytes': (1 if cause == 'memory' else 10)*1024**3})
    client = OllamaClient('qwen3-vl:2b-instruct', tmp_path, timeout=.01 if cause == 'timeout' else 10)
    interrupted = threading.Event()
    class Socket:
        def shutdown(self, how):
            interrupted.set()
    cancel = tmp_path/'cancel.request'
    with cancellation(cancel):
        expected = JobCancelled if cause == 'cancel' else LocalAIError
        with pytest.raises(expected):
            with client._stream_guard(time.monotonic()):
                client._stream_socket = Socket()
                if cause == 'cancel':
                    cancel.touch()
                assert interrupted.wait(1), 'A stalled image prefill was not interrupted'
    assert client._stream_socket is None
