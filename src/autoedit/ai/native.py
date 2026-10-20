"""Owned local llama.cpp runner for bounded-resolution, single-image vision."""
import base64
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request

from autoedit.runtime import check_cancelled, memory_snapshot
from .ollama import OllamaClient, LocalAIError, LocalOutputError, LOCAL_URL


class NativeVisionClient(OllamaClient):
    single_image = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog_url = self.url
        self.process = None
        self.weights = None
        self.context_size = 1024

    def check_resources(self):
        if self._inference_started or self.resources['device'] != 'cpu':
            return
        minimum = 4.5 if self.model == 'qwen3-vl:2b-instruct' else 6.
        self.wait_for_memory(minimum)

    def wait_for_memory(self, minimum, stable_samples=1):
        stable, last_log = 0, -float('inf')
        while True:
            check_cancelled()
            if time.monotonic() >= getattr(self, 'job_deadline', float('inf')):
                raise LocalAIError('Hard local deadline reached while waiting for RAM.')
            available = memory_snapshot().get('ram_available_bytes')
            if available is None:
                return
            stable = stable+1 if available >= minimum*1024**3 else 0
            if stable >= stable_samples:
                return
            if time.monotonic()-last_log >= 30:
                self.log(f'Local AI waiting for RAM: {available/1024**3:.1f} GB free; '
                    f'need {minimum:g} GB. Completed observations are preserved; Cancel remains available.')
                last_log = time.monotonic()
            time.sleep(1)

    def identity(self):
        if self.catalog_url != LOCAL_URL:
            raise LocalAIError('Fast local vision requires the workspace-managed model installation.')
        catalog = OllamaClient(self.model, self.cache, self.log, url=self.catalog_url)
        info = catalog.identity()
        from autoedit.presets import ROOT
        name, _, tag = self.model.partition(':')
        library = (ROOT/'output/local-ai/models/manifests/registry.ollama.ai/library').resolve()
        manifest = (library/name/(tag or 'latest')).resolve()
        try:
            if not manifest.is_relative_to(library):
                raise ValueError()
            data = json.loads(manifest.read_text(encoding='utf-8'))
            layer = next(item for item in data['layers']
                if item['mediaType'] == 'application/vnd.ollama.image.model')
            if not re.fullmatch(r'sha256:[a-f0-9]{64}', layer['digest']):
                raise ValueError()
            self.weights = ROOT/'output/local-ai/models/blobs'/layer['digest'].replace(':', '-')
            if self.weights.stat().st_size != layer['size']:
                raise ValueError()
        except (OSError, KeyError, StopIteration, TypeError, ValueError):
            raise LocalAIError('Installed Qwen weights are incomplete; run local model setup again.') from None
        from .people import VERSION as people_version, MODEL_SHA256
        return {**info, 'backend': 'llama.cpp', 'vision_version': 'single-image-2',
                'people_version': people_version, 'people_weights_sha256': MODEL_SHA256,
                'image_min_tokens': 64, 'image_max_tokens': 256}

    def ensure_server(self):
        if self.process is not None:
            if self.process.poll() is not None:
                raise LocalAIError('Local vision runner exited; inspect output/local-ai/vision-server.log.')
            return
        self.check_resources()
        if self.weights is None:
            self.identity()
        from autoedit.presets import ROOT
        folder = ROOT/'output/local-ai'
        executable = folder/'runtime/lib/ollama/llama-server.exe'
        if not executable.is_file():
            raise LocalAIError('Portable vision runner is missing; run scripts/setup_local_ai.py.')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        layers = 0 if self.resources['device'] == 'cpu' else self.resources['gpu_layers']
        command = [str(executable), '--model', str(self.weights), '--mmproj', str(self.weights),
            '--host', '127.0.0.1', '--port', str(port), '--offline', '--no-webui',
            '--no-mmproj-offload', '--image-min-tokens', '64', '--image-max-tokens', '256',
            '--ctx-size', str(self.context_size), '--parallel', '1', '--threads', str(self.resources['threads']),
            '--threads-batch', str(self.resources['threads']), '--n-gpu-layers', str(layers),
            '--batch-size', '32', '--ubatch-size', '32', '--cache-ram', '0', '--no-cache-idle-slots',
            '--no-jinja', '--chat-template', 'chatml']
        self.log('Starting owned local vision runner: CPU image encoder, 64–256 image tokens.')
        with (folder/'vision-server.log').open('ab') as output:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                env={**os.environ, 'OMP_NUM_THREADS': str(self.resources['threads'])},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                    | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
        self._inference_started = True
        deadline = min(time.monotonic()+45, getattr(self, 'job_deadline', float('inf')))
        while time.monotonic() < deadline:
            check_cancelled()
            available = memory_snapshot().get('ram_available_bytes')
            if available is not None and available < 1.5*1024**3:
                raise LocalAIError('Local AI stopped: free RAM fell below 1.5 GB while loading the model.')
            if self.process.poll() is not None:
                raise LocalAIError('Local vision runner failed to load; inspect output/local-ai/vision-server.log.')
            try:
                if self.request('/health', timeout=1).get('status') == 'ok':
                    return
            except LocalAIError:
                pass
            time.sleep(.2)
        raise LocalAIError('Local vision model did not become ready within the loading deadline.')

    def chat(self, prompt, images, schema):
        while True:
            try:
                return self._chat_once(prompt, images, schema)
            except LocalAIError as exc:
                if 'free RAM fell below' not in str(exc):
                    raise
                self.unload()
                self.log('Local AI paused and released the model after memory pressure; waiting to resume.')
                self.wait_for_memory(6, stable_samples=3)

    def _chat_once(self, prompt, images, schema):
        check_cancelled()
        required_context = max(1024, ((len(images)*256+len(prompt)//3+self.max_output_tokens+256+511)//512)*512)
        if required_context > self.context_size:
            self.unload()
            self.context_size = required_context
        self.ensure_server()
        started = time.monotonic()
        parts = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+
            base64.b64encode(Path(path).read_bytes()).decode('ascii')}} for path in images]
        parts.append({'type': 'text', 'text': prompt})
        body = {'messages': [{'role': 'user', 'content': parts}], 'stream': True,
            'temperature': 0, 'seed': 0, 'max_tokens': self.max_output_tokens,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'vision', 'schema': schema}}}
        chunks, complete = [], False
        try:
            with self._stream_guard(started), self.opener.open(Request(self.url+'/v1/chat/completions',
                    data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'}),
                    timeout=self.timeout) as response:
                for line in response:
                    check_cancelled()
                    if not line.startswith(b'data: '):
                        continue
                    payload = line[6:].strip()
                    if payload == b'[DONE]':
                        break
                    item = json.loads(payload)
                    choices = item.get('choices', [])
                    if not choices:
                        continue
                    choice = choices[0]
                    chunks.append(choice.get('delta', {}).get('content', '') or '')
                    if choice.get('finish_reason') == 'length':
                        raise LocalOutputError('Local vision response hit its token limit.')
                    if choice.get('finish_reason') == 'stop':
                        complete = True
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise LocalAIError(f'Local vision inference HTTP {code}; inspect vision-server.log.') from None
        except (URLError, TimeoutError, OSError):
            raise LocalAIError('Local vision request disconnected or timed out; completed observations are cached.') from None
        if not complete:
            raise LocalOutputError('Local vision response was interrupted.')
        try:
            result = json.loads(''.join(chunks))
            json.dumps(result, allow_nan=False)
            return result
        except ValueError:
            raise LocalOutputError('Local vision returned invalid JSON.') from None

    def unload(self):
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._inference_started = False

    def close(self):
        self.unload()
        people = getattr(self, 'people', None)
        if people is not None:
            people.close()
            self.people = None
