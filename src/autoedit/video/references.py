"""User-labelled reference images, independent of automatic exposure ranks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from autoedit.video.embeddings import EmbeddingModel

REFERENCE_VERSION = 4


def _isolated_animation_reference(model, frame, face, faces):
    """Keep head context where possible without embedding a second character."""
    height, width = frame.shape[:2]
    x, y, w, h = face['face']
    left, top = max(0, int(x-.35*w)), max(0, int(y-.55*h))
    right, bottom = min(width, int(x+1.35*w)), min(height, int(y+1.2*h))
    for other in faces:
        if other is face:
            continue
        ox, oy, ow, oh = other['face']
        if ox >= x+w:
            right = min(right, int(ox))
        elif ox+ow <= x:
            left = max(left, int(np.ceil(ox+ow)))
        elif oy >= y+h:
            bottom = min(bottom, int(oy))
        elif oy+oh <= y:
            top = max(top, int(np.ceil(oy+oh)))
        else:
            return None  # Overlapping face boxes cannot be safely isolated.
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    local = {**face, 'face': [x-left, y-top, w, h]}
    if face.get('subject_box'):
        bx, by, bw, bh = face['subject_box']
        local['subject_box'] = [bx-left, by-top, bw, bh]
    return model.extract(crop, local, crop.shape[1], crop.shape[0], [local])


def reference_vectors(kind, config, cache_dir, identity):
    paths = [Path(p).resolve() for p in config.get('reference_images', [])]
    if not paths:
        return [], []
    hashes = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
            raise ValueError(f'Invalid character reference image: {path}')
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    key = hashlib.sha256(json.dumps([hashes, kind, identity['model_sha256'], REFERENCE_VERSION]).encode()).hexdigest()
    cache = Path(cache_dir) / f'references-{key}.json'
    if cache.is_file():
        try:
            saved = json.loads(cache.read_text(encoding='utf-8'))
            vectors = np.asarray(saved['vectors'], dtype=np.float32)
            catalog = saved['catalog']
            if (vectors.ndim == 2 and len(vectors) == len(catalog['templates'])
                    and len(vectors) and np.isfinite(vectors).all()):
                return saved['vectors'], {**catalog, 'images': [str(p) for p in paths]}
        except (OSError, ValueError, KeyError):
            pass
    import cv2
    from autoedit.video.faces import MultiRegionFaceLandmarker
    runtime = config.get('_runtime_cache')
    model_key = (kind, identity['model_sha256'], config['device'], config['device_index'])
    model = runtime.get(model_key) if runtime is not None else None
    if model is None:
        model = EmbeddingModel(kind, config)
        if runtime is not None:
            runtime[model_key] = model
    detector = MultiRegionFaceLandmarker(include_animation=kind == 'animation')
    vectors = []
    catalog = {'images': [str(p) for p in paths], 'templates': [], 'skipped_regions': []}
    try:
        for index, path in enumerate(paths):
            frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None or min(frame.shape[:2]) < 32:
                raise ValueError(f'Unreadable or too small character reference: {path}')
            height, width = frame.shape[:2]
            faces = detector.detect(frame, float(index + 1)).get('faces', [])
            faces = sorted(faces, key=lambda f: tuple(f['face']))
            whole_image = not faces and kind == 'animation'
            if not faces and kind == 'animation':
                # A tightly cropped, user-labelled cartoon need not fit human landmarks.
                faces = [{'face': [0, 0, width, height], 'subject_kind': 'object',
                          'subject_box': [0, 0, width, height]}]
            if not faces:
                raise ValueError(f'No readable face in character reference; use a clearer image: {path}')
            before = len(vectors)
            for face_index, face in enumerate(faces):
                if kind == 'animation' and len(faces) > 1:
                    vector = _isolated_animation_reference(model, frame, face, faces)
                else:
                    vector = model.extract(frame, face, width, height, faces)
                box_key = hashlib.sha256(json.dumps(
                    [round(float(value), 2) for value in face['face']]).encode()).hexdigest()[:12]
                region = {'id': f'reference-{hashes[index][:16]}-{box_key}',
                          'image_index': index, 'image_sha256': hashes[index], 'face_index': face_index,
                          'box': face['face'], 'whole_image': whole_image}
                if vector is None:
                    catalog['skipped_regions'].append({**region, 'reason': 'unreadable-identity-crop'})
                    continue
                vectors.append(vector.tolist())
                catalog['templates'].append(region)
            if len(vectors) == before:
                raise ValueError(f'Cannot extract character identity from reference: {path}')
    finally:
        detector.close()
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix('.tmp')
    temporary.write_text(json.dumps({'vectors': vectors, 'catalog': catalog}), encoding='utf-8')
    temporary.replace(cache)
    return vectors, catalog


def _reference_groups(vectors, catalog, threshold):
    """Merge strong alternate views, never two faces from the same image."""
    distances = 1 - vectors @ vectors.T
    groups = []
    for index, template in enumerate(catalog['templates']):
        compatible = [group for group in groups if all(
            (template['id'] == catalog['templates'][j]['id'] or
             template.get('image_sha256', template['image_index']) !=
             catalog['templates'][j].get('image_sha256', catalog['templates'][j]['image_index']))
            and distances[index, j] <= threshold*.5 for j in group)]
        if compatible:
            min(compatible, key=lambda group: max(distances[index, j] for j in group)).append(index)
        else:
            groups.append([index])
    return [{'id': min(catalog['templates'][i]['id'] for i in group),
             'template_indices': group, 'templates': [catalog['templates'][i] for i in group]}
            for group in groups]


def match_reference_tracks(tracks, kind, config, cache_dir, identity, progress=None):
    if not config.get('reference_images'):
        return None
    vectors, catalog = reference_vectors(kind, config, cache_dir, identity)
    # Older integrations supplied one vector per image.
    if isinstance(catalog, list):
        catalog = {'images': catalog, 'templates': [
            {'id': f'reference-{i+1}', 'image_index': i, 'whole_image': False}
            for i in range(len(vectors))], 'skipped_regions': []}
    references = np.asarray(vectors, np.float32)
    threshold = config['ccip_cosine_distance' if kind == 'animation' else 'arcface_cosine_distance']
    groups = _reference_groups(references, catalog, threshold)
    for track in tracks:
        observations = np.asarray(track.get('vectors', []), np.float32)
        track.update(reference_match=False, reference_confidence=0., reference_id=None,
                     reference_template_id=None, reference_ambiguous=False)
        if len(observations) < 2:
            continue
        distances = 1 - observations @ references.T
        candidates = []
        for group in groups:
            nearest = distances[:, group['template_indices']].min(axis=1)
            fraction = float(np.mean(nearest <= threshold))
            if fraction >= .75 and int(np.sum(nearest <= threshold)) >= 2:
                candidates.append((float(np.median(nearest)), group['id'], fraction))
        candidates.sort()
        if not candidates:
            continue
        distance, reference_id, fraction = candidates[0]
        if len(candidates) > 1 and candidates[1][0]-distance < .03:
            track['reference_ambiguous'] = True
            continue
        track.update(reference_match=True, reference_id=reference_id,
                     reference_confidence=fraction, reference_distance=distance)
        group = next(g for g in groups if g['id'] == reference_id)
        track['reference_template_id'] = catalog['templates'][min(
            group['template_indices'], key=lambda i: float(np.median(distances[:, i])))]['id']
    # Two co-visible identities with effectively equal evidence are ambiguous.
    matched = [t for t in tracks if t['reference_match']]
    ambiguous = set()
    for i, left in enumerate(matched):
        for j, right in enumerate(matched[i+1:], i+1):
            if (left['reference_id'] == right['reference_id']
                    and abs(left['reference_distance']-right['reference_distance']) < .03
                    and any(max(a, c) < min(b, d) for a,b in left['intervals'] for c,d in right['intervals'])):
                ambiguous.update((i, j))
    for index in ambiguous:
        matched[index].update(reference_match=False, reference_id=None, reference_template_id=None,
                              reference_confidence=0., reference_ambiguous=True)
    count = sum(t['reference_match'] for t in tracks)
    if progress:
        progress(f"Character references: {len(catalog['images'])} images, {len(groups)} targets; {count}/{len(tracks)} tracks matched")
        for region in catalog['skipped_regions']:
            progress(f"Reference face {region['face_index']+1} unreadable: {catalog['images'][region['image_index']]}; use a clearer crop")
        if any(t.get('whole_image') for t in catalog['templates']):
            progress('Reference without readable faces treated as one cartoon crop; use separate crops for multiple characters')
    for group in groups:
        group['matched_tracks'] = sum(t.get('reference_id') == group['id'] for t in tracks)
        del group['template_indices']
    return {'images': catalog['images'], 'characters': groups, 'skipped_regions': catalog['skipped_regions'],
            'matched_tracks': count, 'cosine_distance': threshold,
            'policy': 'reference-first-fallback-after-scan', 'model_sha256': identity['model_sha256']}


def pooled_reference_catalog(catalogs):
    """Use shared template provenance across sources, even with different models.

    If models disagree about merging alternate views, keep them separate.
    Co-visible reference faces must never collapse into one pooled target.
    """
    memberships, templates = [], {}
    for catalog in catalogs:
        members = {}
        for character in catalog.get('characters', []):
            for template in character['templates']:
                templates[template['id']] = template
                members[template['id']] = character['id']
        memberships.append(members)
    partitions = {}
    for tid in sorted(templates):
        signature = tuple(members.get(tid, tid) for members in memberships)
        partitions.setdefault(signature, []).append(tid)
    owners, characters = {}, []
    for members in partitions.values():
        rid = min(members)
        owners.update({tid: rid for tid in members})
        characters.append({'id': rid, 'templates': [templates[tid] for tid in members]})
    return {**catalogs[0], 'characters': characters,
            'matched_tracks': sum(c.get('matched_tracks', 0) for c in catalogs),
            'source_models': sorted({c['model_sha256'] for c in catalogs if c.get('model_sha256')}),
            'skipped_regions': [r for c in catalogs for r in c.get('skipped_regions', [])]}, owners


def reference_selection_report(reference, slots):
    counts, seconds = {}, {}
    fallback = 0
    for slot in slots:
        cut = slot['video']
        selection = cut.get('character_selection') or {}
        rid = selection.get('reference_id')
        if not rid or selection.get('reference_fallback'):
            fallback += 1
            continue
        counts[rid] = counts.get(rid, 0) + 1
        seconds[rid] = seconds.get(rid, 0.) + cut['out_sec']-cut['in_sec']
    characters = [{**character, 'selected_cuts': counts.get(character['id'], 0),
                   'selected_seconds': seconds.get(character['id'], 0.),
                   'status': 'selected' if counts.get(character['id']) else 'not-selected'}
                  for character in reference.get('characters', [])]
    return {'policy': 'prefer-least-used-eligible-reference', 'characters': characters,
            'fallback_cuts': fallback, 'selected_target_count': len(counts),
            'all_targets_selected': bool(characters) and all(c['selected_cuts'] for c in characters)}
