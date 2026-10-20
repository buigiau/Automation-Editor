"""Install the official portable Ollama runtime and Qwen inside this workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate', action='store_true', help='Save Qwen local as the selected backend')
    parser.add_argument('--model', choices=['qwen3-vl:4b-instruct', 'qwen3-vl:2b-instruct'],
                        help='Defaults to the saved local model; launching the app preserves your choice')
    args = parser.parse_args()
    folder = ROOT / 'output/local-ai'
    runtime = folder / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    if not (runtime / 'ollama.exe').exists() or not (folder / 'runtime-version.json').exists():
        print('Downloading official portable Ollama (Windows NVIDIA runtime)…', flush=True)
        with urllib.request.urlopen('https://api.github.com/repos/ollama/ollama/releases/latest', timeout=60) as response:
            release = json.load(response)
        asset = next(a for a in release['assets'] if a['name'] == 'ollama-windows-amd64.zip')
        archive = folder / (release['tag_name'] + '.zip')
        temporary = archive.with_suffix('.download')
        if not archive.exists():
            with urllib.request.urlopen(asset['browser_download_url'], timeout=120) as response, temporary.open('wb') as target:
                total, last = 0, -1
                while block := response.read(4 * 1024 * 1024):
                    target.write(block)
                    total += len(block)
                    percent = total * 100 // asset['size']
                    if percent // 10 != last:
                        print(f'Ollama download: {percent}%', flush=True)
                        last = percent // 10
            if total != asset['size']:
                raise RuntimeError('Incomplete Ollama download; run setup again')
            temporary.replace(archive)
        expected = asset.get('digest')
        if expected:
            with archive.open('rb') as handle:
                actual = 'sha256:' + hashlib.file_digest(handle, 'sha256').hexdigest()
            if actual != expected:
                raise RuntimeError('Ollama archive checksum mismatch')
        print('Extracting Ollama runtime…', flush=True)
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                target = (runtime / item.filename).resolve()
                if not target.is_relative_to(runtime.resolve()):
                    raise RuntimeError('Unsafe archive entry')
            bundle.extractall(runtime)
        (folder / 'runtime-version.json').write_text(json.dumps({'version': release['tag_name'],
            'asset': asset['name'], 'digest': expected}), encoding='utf-8')
    from autoedit.ai.ollama import OllamaClient, LOCAL_URL
    from autoedit.presets import load_settings, save_settings
    selected_model = args.model or load_settings().get('local_model', 'qwen3-vl:2b-instruct')
    client = OllamaClient(selected_model, ROOT / 'output/.ai-cache')
    client.ensure_server()
    installed = client.request('/api/tags').get('models', [])
    if not any(client.model in (m.get('name'), m.get('model')) for m in installed):
        print('Downloading '+selected_model+' (Q4_K_M)…', flush=True)
        request = urllib.request.Request(LOCAL_URL + '/api/pull',
            data=json.dumps({'model': client.model, 'stream': True}).encode(),
            headers={'Content-Type': 'application/json'})
        last = None
        with client.opener.open(request, timeout=300) as response:
            for line in response:
                item = json.loads(line)
                if item.get('error'):
                    raise RuntimeError('Qwen download failed: ' + item['error'])
                status = item.get('status', '')
                if item.get('total'):
                    status += f" {item.get('completed', 0)*100//item['total']//10*10}%"
                if status != last:
                    print(status, flush=True)
                    last = status
    info = client.identity()
    from setup_fast_people import main as setup_people
    setup_people()
    print('Qwen ready: ' + json.dumps(info), flush=True)
    if args.activate:
        settings = load_settings()
        settings.update(backend='local', local_model=client.model, local_url=LOCAL_URL)
        save_settings(settings)
        print('Saved backend=local. No Gemini key is required for local runs.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
