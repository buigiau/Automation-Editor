import io
import json
from urllib.error import HTTPError
import pytest
from autoedit.ai import gemini
from autoedit.ai.cache import identity, read


def test_api_key_only_in_header_and_http_errors_redacted(tmp_path, monkeypatch):
    seen = []
    def fail(request, timeout):
        seen.append(request)
        raise HTTPError(request.full_url+'?key=secret', 403, 'secret', {}, io.BytesIO(b'secret'))
    monkeypatch.setattr(gemini, 'urlopen', fail)
    client = gemini.GeminiClient('secret', 'gemini-3.8-flash', tmp_path)
    with pytest.raises(gemini.GeminiError) as error:
        client.generate([], {})
    assert 'secret' not in str(error.value)
    assert 'secret' not in seen[0].full_url
    assert seen[0].get_header('X-goog-api-key') == 'secret'
    assert error.value.__suppress_context__


def test_chunks_and_active_upload_cache(tmp_path, monkeypatch):
    source = tmp_path/'test.mp4'
    source.write_bytes(b'1'*(9*1024*1024))
    requests = []
    active = {'name': 'files/x', 'state': 'ACTIVE', 'uri': gemini.BASE+'/files/x', 'mimeType': 'video/mp4'}
    def request(self, url, body=None, headers=None, **kwargs):
        requests.append((url, body, headers))
        if url.endswith('/upload/v1beta/files'):
            return {}, {'X-Goog-Upload-URL': gemini.BASE+'/upload/session'}
        if url.endswith('/upload/session'):
            return ({'file': active} if headers['X-Goog-Upload-Command'].endswith('finalize') else {}), {}
        return active, {}
    monkeypatch.setattr(gemini.GeminiClient, 'request', request)
    client = gemini.GeminiClient('secret', 'gemini-3.8-flash', tmp_path/'cache')
    assert client.upload(source, identity(source))['fileData']['mimeType'] == 'video/mp4'
    chunks = [r for r in requests if r[0].endswith('/session')]
    assert [len(r[1]) for r in chunks] == [8*1024*1024, 1024*1024]
    assert [r[2]['X-Goog-Upload-Offset'] for r in chunks] == ['0', str(8*1024*1024)]
    requests.clear()
    client.upload(source, identity(source))
    assert len(requests) == 1 and requests[0][0].endswith('files/x')
    assert all('secret' not in p.read_text() for p in (tmp_path/'cache').rglob('*.json'))


def test_upload_does_not_forward_key_to_unexpected_host(tmp_path):
    client = gemini.GeminiClient('secret', 'gemini-3.8-flash', tmp_path)
    with pytest.raises(gemini.GeminiError, match='endpoint'):
        client.request('https://example.com/upload')


def test_blocked_or_truncated_response_is_explicit_failure(tmp_path, monkeypatch):
    client = gemini.GeminiClient('secret', 'gemini-3.8-flash', tmp_path)
    monkeypatch.setattr(client, 'request', lambda *a, **k: ({'candidates': [{'finishReason': 'MAX_TOKENS'}]}, {}))
    with pytest.raises(gemini.GeminiError, match='did not complete'):
        client.generate([], {})


def test_generation_uses_configured_timeout(tmp_path, monkeypatch):
    client = gemini.GeminiClient('secret', 'gemini-3.8-flash', tmp_path)
    client.request_timeout_sec = 900
    def request(*args, **kwargs):
        assert kwargs['timeout'] == 900
        return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{}'}]}}]}, {}
    monkeypatch.setattr(client, 'request', request)
    assert client.generate([], {}) == {}


def test_echoed_key_and_thoughts_cannot_enter_result(tmp_path, monkeypatch):
    client = gemini.GeminiClient('secret', 'gemini-3.8-flash', tmp_path)
    monkeypatch.setattr(client, 'request', lambda *a, **k: ({'candidates': [{'finishReason': 'STOP',
        'content': {'parts': [{'thought': True, 'text': 'internal'}, {'text': '{"reason":"secret"}'}]}}]}, {}))
    assert client.generate([], {}) == {'reason': '[REDACTED]'}


def test_nonfinite_json_is_returned_as_repairable_invalid_result(tmp_path, monkeypatch):
    client = gemini.GeminiClient('secret', 'gemini-3.8-flash', tmp_path)
    monkeypatch.setattr(client, 'request', lambda *a, **k: ({'candidates': [{'finishReason': 'STOP',
        'content': {'parts': [{'text': '{"in_sec": NaN}'}]}}]}, {}))
    assert client.generate([], {})['invalid_response']


def test_bmp_reference_uploads_as_png(tmp_path, monkeypatch):
    from PIL import Image
    source = tmp_path/'tree.bmp'
    Image.new('RGB', (12, 12), 'green').save(source)
    seen = []
    active = {'name': 'files/image', 'state': 'ACTIVE', 'uri': gemini.BASE+'/files/image', 'mimeType': 'image/png'}
    def request(self, url, body=None, headers=None, **kwargs):
        seen.append((body, headers))
        if isinstance(body, dict):
            return {}, {'X-Goog-Upload-URL': gemini.BASE+'/upload/session'}
        return {'file': active}, {}
    monkeypatch.setattr(gemini.GeminiClient, 'request', request)
    client = gemini.GeminiClient('secret', 'gemini-3.8-flash', tmp_path/'cache')
    assert client.upload(source, identity(source))['fileData']['mimeType'] == 'image/png'
    assert seen[0][1]['X-Goog-Upload-Header-Content-Type'] == 'image/png'
    assert seen[1][0].startswith(b'\x89PNG')
