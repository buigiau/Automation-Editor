import json
from pathlib import Path
import subprocess

import pytest

from autoedit.ai.native import NativeVisionClient
from autoedit.ai.ollama import OllamaClient, LocalAIError, LocalOutputError


@pytest.fixture
def installation(tmp_path, monkeypatch):
    import autoedit.presets
    monkeypatch.setattr(autoedit.presets, 'ROOT', tmp_path)
    digest = 'f'*64
    folder = tmp_path/'output/local-ai'
    weights = folder/'models/blobs'/('sha256-'+digest)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'local-weights')
    manifest = folder/'models/manifests/registry.ollama.ai/library/qwen3-vl/2b-instruct'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'layers': [{'mediaType': 'application/vnd.ollama.image.model',
        'digest': 'sha256:'+digest, 'size': weights.stat().st_size}]}))
    executable = folder/'runtime/lib/ollama/llama-server.exe'
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(OllamaClient, 'identity', lambda self: {'model': self.model, 'digest': digest})
    return folder, weights, manifest


def test_native_identity_uses_installed_weights_and_bounds_image_resolution(installation, tmp_path):
    _, weights, _ = installation
    client = NativeVisionClient('qwen3-vl:2b-instruct', tmp_path)
    identity = client.identity()
    assert client.weights == weights
    assert identity['backend'] == 'llama.cpp'
    assert identity['image_min_tokens'] == 64 and identity['image_max_tokens'] == 256


def test_incomplete_installed_weights_are_rejected(installation, tmp_path):
    _, weights, _ = installation
    weights.write_bytes(b'truncated')
    with pytest.raises(LocalAIError, match='incomplete'):
        NativeVisionClient('qwen3-vl:2b-instruct', tmp_path).identity()


def test_native_runner_is_owned_loopback_cpu_and_has_no_large_prompt_cache(installation, tmp_path, monkeypatch):
    import autoedit.ai.native as native
    calls = []
    class Process:
        def poll(self):
            return None
    monkeypatch.setattr(native.subprocess, 'Popen', lambda cmd, **kw: calls.append((cmd, kw)) or Process())
    monkeypatch.setattr(native, 'memory_snapshot', lambda: {'ram_available_bytes': 10*1024**3})
    client = NativeVisionClient('qwen3-vl:2b-instruct', tmp_path)
    client.identity()
    monkeypatch.setattr(client, 'request', lambda *a, **k: {'status': 'ok'})
    client.ensure_server()
    cmd, kwargs = calls[0]
    assert cmd[cmd.index('--host')+1] == '127.0.0.1'
    assert cmd[cmd.index('--n-gpu-layers')+1] == '0'
    assert '--no-mmproj-offload' in cmd and '--offline' in cmd
    assert cmd[cmd.index('--ctx-size')+1] == '1024'
    assert cmd[cmd.index('--cache-ram')+1] == '0'
    assert cmd[cmd.index('--threads')+1] == cmd[cmd.index('--threads-batch')+1] == '2'
    assert kwargs['stdin'] == subprocess.DEVNULL


def test_unload_terminates_only_this_clients_owned_runner(tmp_path):
    calls = []
    class Process:
        def poll(self): return None
        def terminate(self): calls.append('terminate')
        def wait(self, **kw): calls.append('wait')
    client = NativeVisionClient('qwen3-vl:2b-instruct', tmp_path)
    client.process = Process()
    client.unload()
    client.unload()
    assert calls == ['terminate', 'wait'] and client.process is None


@pytest.mark.parametrize('finish', ['stop', 'length', None])
def test_native_stream_requires_complete_json_and_reports_truncation(tmp_path, monkeypatch, finish):
    path = tmp_path/'image.jpg'
    path.write_bytes(b'image-data')
    client = NativeVisionClient('qwen3-vl:2b-instruct', tmp_path)
    monkeypatch.setattr(client, 'ensure_server', lambda: None)
    bodies = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __iter__(self):
            yield b'data: '+json.dumps({'choices': [{'delta': {'content': '{"present":false}'},
                'finish_reason': finish}]}).encode()+b'\n'
            yield b'data: [DONE]\n'
    class Opener:
        def open(self, request, **kw):
            bodies.append(json.loads(request.data))
            return Response()
    client.opener = Opener()
    if finish == 'stop':
        assert client.chat('Inspect', [path], {}) == {'present': False}
        assert bodies[0]['messages'][0]['content'][0]['type'] == 'image_url'
        assert bodies[0]['response_format']['type'] == 'json_schema'
    else:
        with pytest.raises(LocalOutputError):
            client.chat('Inspect', [path], {})


def test_memory_pressure_releases_owned_model_and_resumes_without_discarding_the_request(tmp_path, monkeypatch):
    client = NativeVisionClient('qwen3-vl:2b-instruct', tmp_path)
    calls = []
    def generate(*args):
        calls.append('request')
        if len(calls) == 1:
            raise LocalAIError('Local AI stopped: free RAM fell below 1.5 GB')
        return {'present': True}
    monkeypatch.setattr(client, '_chat_once', generate)
    monkeypatch.setattr(client, 'unload', lambda: calls.append('unload'))
    monkeypatch.setattr(client, 'wait_for_memory', lambda minimum, **kw: calls.append(('wait', minimum, kw)))
    assert client.chat('Inspect', [], {}) == {'present': True}
    assert calls == ['request', 'unload', ('wait', 6, {'stable_samples': 3}), 'request']


def test_nonresource_errors_are_not_retried_forever(tmp_path, monkeypatch):
    client = NativeVisionClient('qwen3-vl:2b-instruct', tmp_path)
    monkeypatch.setattr(client, '_chat_once', lambda *a: (_ for _ in ()).throw(LocalAIError('Invalid model')))
    monkeypatch.setattr(client, 'wait_for_memory', lambda *a, **k: pytest.fail('Retried an unrelated error'))
    with pytest.raises(LocalAIError, match='Invalid model'):
        client.chat('Inspect', [], {})
