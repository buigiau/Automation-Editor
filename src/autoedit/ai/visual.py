"""Compare only selected footage; no full-video recognition or transcription."""
from pathlib import Path
import math
from .cache import digest, read, write
from autoedit.runtime import check_cancelled


def probe(path):
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f'No video stream: {path}')
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / av.time_base
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'Cannot determine source duration: {path}')
        return {'duration_sec': duration, 'has_audio': bool(container.streams.audio)}


def fingerprints(source, row, cache):
    import av
    import cv2
    import numpy as np
    cv2.setNumThreads(2)
    if row.get('timeline_visible_prefix'):
        row = {**row, 'out_sec': row['identity_end_sec']}
    key = digest(['frames-v2', source['identity'], round(row['in_sec'], 9), round(row['out_sec'], 9)])
    dest = Path(cache) / 'frames' / (key + '.json')
    saved = read(dest)
    if saved is not None:
        return saved
    frames = []
    from .frames import cut_frames
    times = [row['in_sec']+(row['out_sec']-row['in_sec'])*f for f in (.04, .18, .32, .46, .60, .74, .88, .96)]
    for _, rgb in cut_frames(source['path'], times, 160):
        small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(small.astype(np.float32))[:8, :8].flatten()[1:]
        bits = (dct > np.median(dct)).astype(int).tolist()
        color = cv2.resize(rgb, (16, 16), interpolation=cv2.INTER_AREA)
        frames.append({'hash': bits, 'rgb': color.flatten().tolist(), 'std': float(small.std())})
    if len(frames) < 4:
        raise ValueError('Cannot decode enough frames to check selected footage')
    write(dest, frames)
    return frames


def similar(a, b):
    import numpy as np
    # Multiple color-aware frame correspondences guard against false matches from silhouettes.
    matches = set()
    for i, first in enumerate(a):
        for j, second in enumerate(b):
            hamming = sum(x != y for x, y in zip(first['hash'], second['hash']))
            error = float(np.mean(np.abs(np.array(first['rgb'], dtype=float) - second['rgb'])))
            if hamming <= 8 and error <= 16:
                matches.add((i, j))
    return len({i for i, _ in matches}) >= 3 and len({j for _, j in matches}) >= 3


def check_visual_duplicates(rows, sources, cache):
    source_map = {s['id']: s for s in sources}
    accepted, issues, samples = [], [], []
    for row in rows:
        check_cancelled()
        try:
            frames = fingerprints(source_map[row['source_id']], row, cache)
            duplicate = next((other for other, previous in samples if similar(frames, previous)), None)
            disclosed_repeat = row.get('completion_fallback') and 'visual_duplicate' in row.get('fallback_relaxations', [])
            if duplicate and not disclosed_repeat:
                issues.append({'slot_id': row['slot_id'], 'error': f"Frames repeat footage from {duplicate['slot_id']}; choose a different shot"})
            else:
                accepted.append(row)
                samples.append((row, frames))
        except (OSError, ValueError) as exc:
            issues.append({'slot_id': row['slot_id'], 'error': str(exc)})
    return accepted, issues
