"""Cheap, deliberately permissive body-reference screening. Never cut evidence."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from autoedit.runtime import check_cancelled
from autoedit.video.characters import settings
from autoedit.video.embeddings import EmbeddingModel, fingerprint, model_path
from autoedit.video.reader import sampled_frames
from autoedit.video.references import reference_vectors

SCREEN_VERSION = 2


def body_screening_enabled(config):
    if config.get('reference_discovery', 'auto') == 'objects':
        return False
    character = settings(config.get('characters'))
    regions = {str(Path(p).resolve()): value for p, value in character['reference_regions'].items()}
    return bool(character['enabled'] and character['reference_images']) and all(
        regions.get(str(Path(p).resolve())) and all(r.get('kind') == 'body' for r in regions[str(Path(p).resolve())])
        for p in character['reference_images'])


def screening_tiles(width, height):
    boxes = [[0, 0, width, height], [.15*width, .05*height, .7*width, .9*height],
             [0, .1*height, .6*width, .8*height], [.4*width, .1*height, .6*width, .8*height]]
    # Small/distant characters need comparable crops rather than an embedding
    # dominated by the road, stands or another foreground character.
    boxes.extend([[(cx-.2)*width,(cy-.275)*height,.4*width,.55*height]
                  for cy in (.35,.65) for cx in (.2,.5,.8)])
    return [{'face': box, 'subject_box': box, 'subject_kind': 'object'} for box in boxes]


class BodyReferenceSearch:
    def __init__(self, config, cache_dir):
        self.config = settings(config.get('characters'))
        self.cache_dir = Path(cache_dir)
        path = model_path('animation', self.config)
        self.identity = {'model_sha256': fingerprint(path)}
        self.runtime = {}
        self.config['_runtime_cache'] = self.runtime
        refs, _ = reference_vectors('animation', self.config, self.cache_dir, self.identity)
        self.references = np.asarray(refs, np.float32)
        self.base_references = self.references.copy()
        self.key = ('animation', self.identity['model_sha256'], self.config['device'], self.config['device_index'])

    def use_confirmed_views(self, views):
        self.references = np.concatenate([self.base_references,np.asarray(views,np.float32)]) if views else self.base_references

    def close(self):
        from autoedit.runtime import release_inference
        release_inference(self.runtime)

    def discover(self, path, start, end, fps, duration, config, cache_dir, session, progress):
        stat = Path(path).stat()
        key = [str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns, start, end, fps,
               self.identity, SCREEN_VERSION]
        dest = self.cache_dir / ('body-screen-' + hashlib.sha256(json.dumps(key).encode()).hexdigest() + '.npz')
        if dest.is_file():
            with np.load(dest, allow_pickle=False) as data:
                times, vectors = data['times'], data['vectors']
        else:
            model = self.runtime.get(self.key)
            if model is None:
                model = self.runtime[self.key] = EmbeddingModel('animation', self.config)
            times, vectors = [], []
            frames = sampled_frames(path, fps, end, start_sec=start)
            try:
                for t, frame in frames:
                    check_cancelled()
                    height, width = frame.shape[:2]
                    values = model.extract_many(frame, screening_tiles(width, height), width, height)
                    times.append(t)
                    vectors.append(np.asarray([v if v is not None else np.zeros_like(self.references[0]) for v in values], np.float32))
            finally:
                frames.close()
            times, vectors = np.asarray(times), np.asarray(vectors, np.float32)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, times=times, vectors=vectors)
            temp.replace(dest)
        threshold = min(.45, self.config.get('body_reference_cosine_distance',self.config['ccip_cosine_distance']) + .16)
        distances = np.min(1-vectors @ self.references.T, axis=(1, 2)) if len(times) else np.asarray([])
        selected = times[distances <= threshold]
        if progress:
            progress(f'Image screening: {len(selected)}/{len(times)} candidate frames; dense identity checks still required')
        return {'screening_times': list(map(float, selected)),
                'screening_candidates': [{'time_sec':float(t), 'distance':float(d)}
                                         for t,d in zip(times,distances) if d <= threshold], 'sample_fps': fps,
                'source_kind': 'animation', 'samples': [], 'duration_sec': duration}


def clip_windows(times, longest, gap, limit):
    """Prefer separated clip-sized regions; deferred positions remain searchable."""
    length = longest + 1.
    windows = []
    for t in times:
        start = max(0., t - .5)
        end = min(limit, start + length)
        if end-start < min(longest, limit):
            start = max(0., end-length)
        if windows and start < windows[-1][1] + gap:
            continue
        windows.append((start, end))
    return windows


def ranked_clip_windows(candidates, longest, gap, limit, threshold, sample_fps=None):
    """Verify strongest matches first; weak hits remain for the second pass."""
    length = longest + 1.
    windows = []
    supported = sorted(row['time_sec'] for row in candidates if row['distance'] <= threshold)
    for candidate in sorted(candidates, key=lambda row: (row['distance'], row['time_sec'])):
        if candidate['distance'] > threshold:
            continue
        start = max(0., min(limit-length, candidate['time_sec']-length/2))
        end = min(limit, start+length)
        if sample_fps:
            hits = [time for time in supported if start <= time <= end]
            if (len(hits) < math.ceil(longest*sample_fps)+1
                    or any(b-a > 1.1/sample_fps for a,b in zip(hits,hits[1:]))):
                continue
        if any(start < right+gap and left < end+gap for left,right in windows):
            continue
        windows.append((start, end))
    return windows
