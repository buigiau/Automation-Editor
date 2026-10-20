"""Output-independent, atomic caches. Credentials are never part of cache records."""
import hashlib
import json
from pathlib import Path
from autoedit.runtime import check_cancelled


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    # Hash the content too: copied footage and restored timestamps must invalidate correctly.
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            check_cancelled()
            sha.update(chunk)
    return {'path': str(path), 'size': stat.st_size, 'sha256': sha.hexdigest()}


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def voice_catalog(config, cache):
    from autoedit.audio.analyzer import analyze_file, iter_audio_files
    from autoedit.cubase.sampler import inspect_sample
    paths = ([Path(config['directory']) / p for p in config['files']] if config.get('files')
             else iter_audio_files(config['directory'], config.get('glob', '*.wav')))
    if config.get('max_files', 0) > 0:
        paths = paths[:config['max_files']]
    result = []
    for path in paths:
        check_cancelled()
        stamp = identity(path)
        key = digest(['voice-v1', stamp])
        dest = Path(cache) / 'voice' / (key + '.json')
        item = read(dest)
        if item is None:
            inspect_sample(str(path))
            item = analyze_file(path)
            item['voice_id'] = 'voice-' + key[:20]
            item['identity'] = stamp
            write(dest, item)
        result.append(item)
    if not result:
        raise ValueError('No usable WAV files in Voice library')
    return result
