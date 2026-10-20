"""Measure recurring camera compositions independently of actors and source time."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

COMPOSITION_VERSION = 1
COMPOSITION_DISTANCE = .075


def promotional_overlay(sample):
    """A central floating text widget, corroborated by a localized panel box."""
    from autoedit.video.subjects import contained_face
    width, height = sample.get('frame_width', 0), sample.get('frame_height', 0)
    if not width or not height:
        return False
    props = sample.get('prop_regions') or []
    for prop in props:
        words = set(prop.get('label', '').split())
        if not words.intersection({'logo', 'watermark'}) or prop.get('score', 0) < .3:
            continue
        x,y,w,h = prop['box']
        if not (.18 <= w/width <= .65 and w/max(1,h) >= 3
                and .35 <= (x+w/2)/width <= .65 and .2 <= (y+h/2)/height <= .7):
            continue
        if any(set(panel.get('label', '').split()).intersection({'poster','paper','sheet'})
               and panel.get('score', 0) >= .3 and panel.get('frame_fraction', 1) < .3
               and contained_face(prop['box'], panel['box']) for panel in props):
            return True
    return False


def frame_signature(frame):
    import cv2
    small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
    # Exclude the usual subtitle band and corner watermark. Keep spatial
    # colour structure: a close-up and a wide view of the same room differ.
    small = small[3:45].copy()
    small[:9, :10] = small[:9, 10:20].mean(axis=(0, 1))
    small = cv2.GaussianBlur(small, (9, 9), 0)
    return cv2.cvtColor(cv2.resize(small, (20, 12)), cv2.COLOR_BGR2LAB).reshape(-1)


def group_compositions(signatures, distance=COMPOSITION_DISTANCE):
    """Complete-link groups prevent a gradual zoom from bridging distinct views."""
    features = np.asarray(signatures, dtype=np.float32) / 255.
    if not len(features):
        return []
    groups, representatives, labels = [], [], []
    limit = distance * distance
    for index, feature in enumerate(features):
        candidates = np.argsort(np.mean((np.asarray(representatives)-feature)**2, axis=1)) if groups else []
        chosen = None
        for group in candidates:
            if np.mean((representatives[group]-feature)**2) > limit:
                break
            if np.max(np.mean((features[groups[group]]-feature)**2, axis=1)) <= limit:
                chosen = int(group)
                break
        if chosen is None:
            chosen = len(groups)
            groups.append([])
            representatives.append(feature)
        groups[chosen].append(index)
        labels.append(chosen)
    return labels


def annotate_compositions(info, cache_dir, progress=None):
    """Reuse source-frame signatures when an incremental search grows."""
    path = Path(info.get('path') or '')
    samples = info.get('samples') or []
    for sample in samples:
        sample['promotional_overlay'] = float(promotional_overlay(sample))
    # Synthetic/custom analyzers without physical media have no visual evidence.
    if not samples or not path.is_file() or not info.get('fps'):
        return info
    times = sorted({float(s['time_sec']) for s in samples})
    stat = path.stat()
    identity = {'version':COMPOSITION_VERSION, 'path':str(path.resolve()),
                'size':stat.st_size, 'mtime_ns':stat.st_mtime_ns}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    cache = Path(cache_dir) / f'composition-frames-{key}.npz'
    by_frame = {}
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as saved:
                saved_times, saved_signatures = saved['times'], saved['signatures']
                if len(saved_times) == len(saved_signatures):
                    by_frame = dict(zip(saved_times.tolist(), saved_signatures))
        except (OSError, ValueError, KeyError, EOFError):
            pass
    missing = [t for t in times if t not in by_frame]
    if missing:
        from autoedit.video.characters import _selected_source_frames
        if progress:
            progress(f'Measuring recurring camera compositions on {len(missing)} new source frames; '
                     f'reusing {len(times)-len(missing)}')
        frames = _selected_source_frames(path, missing)
        try:
            for t, frame in frames:
                by_frame[float(t)] = frame_signature(frame)
        finally:
            frames.close()
        if any(t not in by_frame for t in missing):
            raise RuntimeError('Video decoder did not return all requested composition frames')
        cache.parent.mkdir(parents=True, exist_ok=True)
        saved_times = sorted(by_frame)
        temporary = cache.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, times=np.asarray(saved_times),
                                signatures=np.asarray([by_frame[t] for t in saved_times], dtype=np.uint8))
        temporary.replace(cache)
    signatures = np.asarray([by_frame[t] for t in times], dtype=np.uint8)
    labels = group_compositions(signatures)
    by_time = dict(zip(times, labels))
    for sample in samples:
        sample['composition_id'] = f'view-{by_time[float(sample["time_sec"])]}'
    info['composition_analysis'] = {'version':COMPOSITION_VERSION,
        'method':'spatial-lab-complete-link', 'distance':COMPOSITION_DISTANCE,
        'sample_count':len(times), 'group_count':len(set(labels))}
    return info
