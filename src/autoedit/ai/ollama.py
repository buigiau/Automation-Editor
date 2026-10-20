"""Loopback-only Ollama transport; no cloud fallback or API credential is used."""
import base64
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, ProxyHandler, HTTPHandler, build_opener

from autoedit.runtime import check_cancelled

LOCAL_URL = 'http://127.0.0.1:11435'
STREAM_POLL_SECONDS = .5


class _TrackedHTTPHandler(HTTPHandler):
    """Keep the socket interruptible while image prefill sends no response yet."""
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def http_open(self, request):
        owner = self.owner
        class Connection(http.client.HTTPConnection):
            def connect(self):
                super().connect()
                owner._stream_socket = self.sock
        return self.do_open(Connection, request)


def resource_settings(ai):
    device = ai.get('local_device', 'cpu')
    if device not in ('cpu', 'gpu'):
        raise ValueError('ai.local_device must be cpu or gpu')
    threads = ai.get('local_threads', 2)
    layers = ai.get('local_gpu_layers', 8)
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError('ai.local_threads must be an integer from 1 to 8')
    if type(layers) is not int or not 0 <= layers <= 64:
        raise ValueError('ai.local_gpu_layers must be an integer from 0 to 64')
    return {'device': device, 'threads': threads, 'gpu_layers': layers}


def validate_endpoint(url):
    parsed = urlparse(url)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username or parsed.password or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment):
        raise ValueError('Local AI URL must be a loopback HTTP address, e.g. ' + LOCAL_URL)
    # Force validation of malformed ports before constructing any request.
    parsed.port
    return url.rstrip('/')


class LocalAIError(RuntimeError):
    pass


class LocalOutputError(LocalAIError):
    """An incomplete/invalid model answer can be retried with smaller image batches."""


class OllamaClient:
    def __init__(self, model, cache, log=None, url=LOCAL_URL, timeout=300,
                 device='cpu', threads=2, gpu_layers=8):
        self.url = validate_endpoint(url)
        if not re.fullmatch(r'[\w.:-]+', model) or 'cloud' in model.lower():
            raise ValueError('Choose a locally installed model, e.g. qwen3-vl:4b-instruct')
        self.model, self.cache = model, Path(cache)
        self.log = log or (lambda message: None)
        self.timeout = timeout
        self._inference_started = False
        self.max_output_tokens = 1800
        self.resources = resource_settings({'local_device': device, 'local_threads': threads,
                                            'local_gpu_layers': gpu_layers})
        self._stream_socket = None
        self.opener = build_opener(ProxyHandler({}), _TrackedHTTPHandler(self))

    @contextmanager
    def _stream_guard(self, started):
        from autoedit.runtime import cancellation_file, memory_snapshot
        cancel = cancellation_file()
        stopped = threading.Event()
        reason = []
        self._stream_socket = None
        timeout = min(self.timeout, getattr(self, 'job_deadline', float('inf'))-started)
        def watch():
            while not stopped.wait(STREAM_POLL_SECONDS):
                available = memory_snapshot().get('ram_available_bytes')
                if cancel and Path(cancel).exists():
                    reason.append('cancelled')
                elif self.resources['device'] == 'cpu' and available is not None and available < 1.5*1024**3:
                    reason.append('Local AI stopped: free RAM fell below 1.5 GB; completed batches are cached.')
                elif time.monotonic()-started >= timeout:
                    reason.append('Local generation exceeded the request timeout; completed batches are cached.')
                if reason:
                    active = self._stream_socket
                    if active is not None:
                        try:
                            active.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            yield
        finally:
            stopped.set()
            watcher.join(timeout=2)
            self._stream_socket = None
            if reason:
                check_cancelled()
                raise LocalAIError(reason[0]) from None

    def request(self, route, body=None, timeout=None):
        check_cancelled()
        remaining = getattr(self, 'job_deadline', float('inf'))-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Source matching search budget exhausted')
        timeout = min(timeout or self.timeout, remaining)
        payload = None if body is None else json.dumps(body).encode('utf-8')
        try:
            with self.opener.open(Request(self.url + route, data=payload,
                    headers={'Content-Type': 'application/json'}), timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise LocalAIError(f'Ollama HTTP {code}. Check the installed local model and available GPU memory.') from None
        except (URLError, TimeoutError, OSError):
            raise LocalAIError('Cannot reach local Ollama. Run scripts/setup_local_ai.py first.') from None
        except ValueError:
            raise LocalAIError('Ollama returned invalid JSON') from None

    def ensure_server(self):
        try:
            self.request('/api/version', timeout=2)
            return
        except LocalAIError:
            pass
        if self.url != LOCAL_URL:
            raise LocalAIError('Start Ollama at the configured local URL first.')
        from autoedit.presets import ROOT
        folder = ROOT / 'output/local-ai'
        executable = folder / 'runtime/ollama.exe'
        if not executable.is_file():
            raise LocalAIError('Local runtime is missing. Run scripts/setup_local_ai.py first.')
        folder.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, 'OLLAMA_HOST': LOCAL_URL, 'OLLAMA_NO_CLOUD': '1',
            'OLLAMA_MODELS': str(folder / 'models'), 'OLLAMA_NUM_PARALLEL': '1',
            'OLLAMA_MAX_LOADED_MODELS': '1', 'OLLAMA_CONTEXT_LENGTH': '4096',
            'OLLAMA_KEEP_ALIVE': '30s'}
        self.log('Starting local Ollama on ' + LOCAL_URL)
        with (folder / 'server.log').open('ab') as output:
            subprocess.Popen([str(executable), 'serve'], env=environment, stdin=subprocess.DEVNULL,
                stdout=output, stderr=output, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
        deadline = min(time.monotonic() + 30, getattr(self, 'job_deadline', float('inf')))
        while time.monotonic() < deadline:
            check_cancelled()
            try:
                self.request('/api/version', timeout=1)
                return
            except LocalAIError:
                time.sleep(.25)
        raise LocalAIError('Local Ollama did not start. See output/local-ai/server.log.')

    def identity(self):
        self.ensure_server()
        models = self.request('/api/tags').get('models', [])
        entry = next((m for m in models if self.model in (m.get('name'), m.get('model'))), None)
        if not entry:
            raise LocalAIError(f'Model {self.model} is not installed. Run scripts/setup_local_ai.py.')
        details = self.request('/api/show', {'model': self.model})
        if details.get('remote_model') or details.get('remote_host'):
            raise LocalAIError('Cloud models are not allowed in the local backend.')
        if 'vision' not in details.get('capabilities', []):
            raise LocalAIError('The local backend requires a vision model.')
        return {'model': self.model, 'digest': entry['digest'], 'backend': 'ollama',
                'analysis_version': 'qwen-local-1'}

    def chat(self, prompt, images, schema):
        check_cancelled()
        self.check_resources()
        context = max(4096, ((len(prompt)//3 + len(images)*1200 + self.max_output_tokens + 1024 + 4095)//4096)*4096)
        if context > 32768:
            raise LocalAIError('Local request is too large. Reduce the number of reference images.')
        body = {'model': self.model, 'stream': True, 'think': False, 'keep_alive': '30s',
            'messages': [{'role': 'user', 'content': prompt,
                          'images': [base64.b64encode(Path(p).read_bytes()).decode('ascii') for p in images]}],
            'format': schema, 'options': {'temperature': 0, 'seed': 0, 'num_ctx': context,
                                          'num_predict': self.max_output_tokens, 'num_thread': self.resources['threads'],
                                          'num_gpu': 0 if self.resources['device'] == 'cpu' else self.resources['gpu_layers'],
                                          'num_batch': 128}}
        started = time.monotonic()
        chunks, completed = [], False
        self._inference_started = True
        try:
            with self._stream_guard(started), self.opener.open(Request(self.url + '/api/chat', data=json.dumps(body).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}), timeout=self.timeout) as response:
                for line in response:
                    check_cancelled()
                    if time.monotonic() - started > self.timeout:
                        raise LocalAIError('Local generation exceeded the request timeout; completed batches are cached.')
                    item = json.loads(line)
                    if item.get('error'):
                        raise LocalAIError('Ollama inference failed. Check output/local-ai/server.log and GPU memory.')
                    chunks.append(item.get('message', {}).get('content', ''))
                    if item.get('done'):
                        if item.get('done_reason', 'stop') != 'stop':
                            from .cache import write, digest
                            write(self.cache / 'local/failed-response' / (digest([self.model, prompt]) + '.json'),
                                  {'model': self.model, 'partial_response': ''.join(chunks)})
                            raise LocalOutputError('Local response hit its token limit; split the image batch.')
                        completed = True
                        break
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise LocalAIError(f'Ollama inference HTTP {code}. Check the local runtime/model.') from None
        except (URLError, TimeoutError, OSError):
            raise LocalAIError('Local inference disconnected or timed out; completed batches are cached.') from None
        if not completed:
            raise LocalOutputError('Local response was interrupted; rerun to resume completed batches.')
        try:
            result = json.loads(''.join(chunks))
            json.dumps(result, allow_nan=False)
        except ValueError:
            raise LocalOutputError('Local model returned invalid JSON; no result was cached.') from None
        return result

    def check_resources(self):
        if self._inference_started:
            return
        if self.resources['device'] == 'cpu':
            from autoedit.runtime import memory_snapshot
            available = memory_snapshot().get('ram_available_bytes')
            minimum = 5 if self.model == 'qwen3-vl:2b-instruct' else 8
            if available is not None and available < minimum * 1024**3:
                raise LocalAIError(f'Qwen CPU paused: app requires at least {minimum} GB free RAM '
                    f'({available/1024**3:.1f} GB available). Close other apps or choose Gemini; '
                    'no model was loaded.')

    def unload(self):
        """Cleanup must work after cancellation and must never start the server."""
        try:
            with self.opener.open(Request(self.url + '/api/generate',
                    data=json.dumps({'model': self.model, 'keep_alive': 0}).encode(),
                    headers={'Content-Type': 'application/json'}), timeout=5) as response:
                response.read()
        except (URLError, OSError, TimeoutError):
            pass
        finally:
            self._inference_started = False
