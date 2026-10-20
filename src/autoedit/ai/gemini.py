"""Gemini REST transport with bounded waits, chunked uploads and safe errors."""
import json
import mimetypes
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from autoedit.runtime import check_cancelled
from .cache import digest, read, write

BASE = 'https://generativelanguage.googleapis.com'


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key, model, cache, log=None):
        if not api_key:
            raise GeminiError('Set Gemini API key in Thiết lập or GEMINI_API_KEY.')
        if not re.fullmatch(r'[a-zA-Z0-9._-]+', model):
            raise GeminiError('Invalid Gemini model name')
        self.key, self.model, self.cache = api_key, model, Path(cache)
        self.log = log or (lambda message: None)
        self.request_timeout_sec = 300

    def request(self, url, body=None, headers=None, method=None, timeout=60):
        check_cancelled()
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname != 'generativelanguage.googleapis.com':
            raise GeminiError('Unexpected Gemini endpoint')
        hdr = {'x-goog-api-key': self.key, **(headers or {})}
        if isinstance(body, dict):
            body = json.dumps(body).encode()
            hdr['Content-Type'] = 'application/json'
        try:
            with urlopen(Request(url, data=body, headers=hdr, method=method), timeout=timeout) as response:
                payload = response.read()
                return (json.loads(payload) if payload else {}), response.headers
        except HTTPError as exc:
            # Never log server response, request headers, URLs or an exception containing the key.
            code = exc.code
            exc.close()
            raise GeminiError(f'Gemini HTTP {code}; check API key, model, quota and file limits. No local fallback.') from None
        except (URLError, TimeoutError, OSError):
            raise GeminiError('Gemini connection failed or timed out. Retry the job; no local fallback.') from None
        except ValueError:
            raise GeminiError('Gemini returned an invalid JSON response') from None

    def upload(self, path, stamp):
        path = Path(path)
        if path.suffix.lower() == '.bmp':
            # Gemini accepts PNG; retain BMP support in the local image picker.
            from PIL import Image
            from .cache import identity
            converted = self.cache / 'reference_images' / (digest(stamp) + '.png')
            if not converted.is_file():
                converted.parent.mkdir(parents=True, exist_ok=True)
                temporary = converted.with_suffix('.tmp')
                with Image.open(path) as picture:
                    picture.convert('RGB').save(temporary, format='PNG')
                temporary.replace(converted)
            path, stamp = converted, identity(converted)
        # Remote file handles are account scoped. Only a key hash is persisted.
        record = self.cache / 'uploads' / (digest([stamp, digest(self.key)]) + '.json')
        saved = read(record)
        info = None
        if saved:
            try:
                info, _ = self.request(BASE + '/v1beta/' + saved['name'])
            except GeminiError as exc:
                if 'HTTP 404' not in str(exc):
                    raise
        if not info:
            mime = mimetypes.guess_type(path.name)[0] or 'video/mp4'
            _, headers = self.request(BASE + '/upload/v1beta/files',
                {'file': {'display_name': path.name}}, headers={
                'X-Goog-Upload-Protocol': 'resumable', 'X-Goog-Upload-Command': 'start',
                'X-Goog-Upload-Header-Content-Length': str(path.stat().st_size),
                'X-Goog-Upload-Header-Content-Type': mime})
            url = headers.get('X-Goog-Upload-URL')
            if not url:
                raise GeminiError('Gemini did not return an upload URL')
            offset, total = 0, path.stat().st_size
            with path.open('rb') as handle:
                while chunk := handle.read(8 * 1024 * 1024):
                    final = offset + len(chunk) == total
                    result, _ = self.request(url, chunk, headers={
                        'X-Goog-Upload-Offset': str(offset), 'Content-Type': mime,
                        'X-Goog-Upload-Command': 'upload, finalize' if final else 'upload'})
                    offset += len(chunk)
                    self.log(f'Upload {path.name}: {offset * 100 // total}%')
            info = result.get('file', {})
            if not info.get('name'):
                raise GeminiError('Gemini upload has no file handle')
            write(record, {'name': info['name']})
        deadline = time.monotonic() + 600
        while info.get('state') != 'ACTIVE':
            if info.get('state') == 'FAILED' or time.monotonic() >= deadline:
                raise GeminiError('Gemini file processing failed or exceeded 10 minutes')
            self.log(f'Gemini processing {path.name}…')
            for _ in range(10):
                check_cancelled()
                time.sleep(.5)
            info, _ = self.request(BASE + '/v1beta/' + info['name'])
        return {'fileData': {'fileUri': info['uri'], 'mimeType': info.get('mimeType') or mimetypes.guess_type(path.name)[0]}}

    def generate(self, parts, schema):
        self.log('Gemini: analyzing characters, shots and Voice…')
        result, _ = self.request(BASE + '/v1beta/models/' + self.model + ':generateContent', {
            'contents': [{'role': 'user', 'parts': parts}],
            'generationConfig': {'responseMimeType': 'application/json', 'responseJsonSchema': schema}}, timeout=self.request_timeout_sec)
        candidates = result.get('candidates') or []
        if not candidates or candidates[0].get('finishReason') != 'STOP':
            raise GeminiError('Gemini did not complete the selection (blocked, empty or token limit).')
        raw = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', []) if not p.get('thought'))
        # Defend logs/cache against an accidentally echoed credential.
        raw = raw.replace(self.key, '[REDACTED]')
        try:
            parsed = json.loads(raw)
            json.dumps(parsed, allow_nan=False)
            return parsed
        except ValueError:
            return {'invalid_response': True, 'selections': [], 'characters': []}
