"""User-labelled reference images, independent of automatic exposure ranks."""
import hashlib
import json
from pathlib import Path

import numpy as np

from autoedit.video.embeddings import EmbeddingModel

REFERENCE_VERSION = 6


def preflight_references(config, kind):
    """Reject unreadable/ambiguous inputs before decoding source footage or loading DINO."""
    paths = config.get('reference_images', [])
    if not paths:
        return
    import cv2
    from autoedit.video.faces import MultiRegionFaceLandmarker
    regions = {str(Path(p).resolve()):value for p,value in config.get('reference_regions', {}).items()}
    detector = None
    try:
        for name in paths:
            path = Path(name)
            if not path.is_file():
                raise ValueError(f'Invalid character reference image: {path}')
            frame = cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            if frame is None or min(frame.shape[:2]) < 32:
                raise ValueError(f'Unreadable or too small character reference: {path}')
            selected = regions.get(str(path.resolve()))
            if selected:
                for region in selected:
                    if (not isinstance(region,dict) or region.get('kind') not in ('face','body')
                            or len(region.get('box',[])) != 4):
                        raise ValueError(f'Invalid selected character region: {path}')
                    x,y,w,h = region['box']
                    if (not all(isinstance(v,(int,float)) and np.isfinite(v) for v in (x,y,w,h)) or
                            min(w,h)<32 or min(x,y)<0 or x+w>frame.shape[1] or y+h>frame.shape[0]):
                        raise ValueError(f'Invalid selected character region: {path}')
                    if region['kind'] == 'face':
                        if detector is None:
                            detector = MultiRegionFaceLandmarker(include_animation=kind == 'animation')
                        crop = frame[int(y):int(y+h),int(x):int(x+w)]
                        if len(detector.detect(crop,float(paths.index(name)+1+selected.index(region)/100)).get('faces',[])) != 1:
                            raise ValueError(f'Selected face region must contain one readable face: {path}')
                continue
            if detector is None:
                detector = MultiRegionFaceLandmarker(include_animation=kind == 'animation')
            if not detector.detect(frame,float(paths.index(name)+1)).get('faces'):
                raise ValueError(f'No isolated character in reference: {path}. Open Review / crop and select each character before running.')
    finally:
        if detector:
            detector.close()


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
    regions = {str(Path(p).resolve()): value for p, value in config.get('reference_regions', {}).items()}
    selected_regions = [regions.get(str(p), []) for p in paths]
    key = hashlib.sha256(json.dumps([hashes, selected_regions, kind, identity['model_sha256'], REFERENCE_VERSION]).encode()).hexdigest()
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
    detector = None
    def detect(frame, timestamp):
        nonlocal detector
        if detector is None:
            detector = MultiRegionFaceLandmarker(include_animation=kind == 'animation')
        return detector.detect(frame,timestamp).get('faces', [])
    vectors = []
    catalog = {'images': [str(p) for p in paths], 'templates': [], 'skipped_regions': []}
    try:
        for index, path in enumerate(paths):
            frame = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None or min(frame.shape[:2]) < 32:
                raise ValueError(f'Unreadable or too small character reference: {path}')
            height, width = frame.shape[:2]
            chosen = selected_regions[index]
            faces = detect(frame, float(index + 1)) if not chosen else []
            if chosen:
                for region in chosen:
                    x,y,w,h = region['box']
                    if (not all(np.isfinite(v) for v in (x,y,w,h)) or min(w,h) < 32 or
                            min(x,y) < 0 or x+w > width or y+h > height):
                        raise ValueError(f'Invalid selected character region: {path}')
                    if region.get('kind') == 'body' and kind == 'animation':
                        faces.append({'face':[x,y,w,h], 'subject_kind':'object', 'subject_box':[x,y,w,h]})
                    else:
                        crop = frame[int(y):int(y+h),int(x):int(x+w)]
                        detected = detect(crop, float(index+1+len(faces)/100))
                        if len(detected) != 1:
                            raise ValueError(f'Selected face region must contain one readable face: {path}')
                        face = dict(detected[0])
                        bx,by,bw,bh = face['face']
                        face['face'] = [bx+x,by+y,bw,bh]
                        if face.get('face_keypoints'):
                            face['face_keypoints'] = (np.asarray(face['face_keypoints']) + [x,y]).tolist()
                        faces.append(face)
            faces = sorted(faces, key=lambda f: tuple(f['face']))
            if not faces and kind == 'animation':
                raise ValueError(f'No isolated character in reference: {path}. Open Review / crop and select each character before running.')
            if not faces:
                raise ValueError(f'No readable face in character reference; use a clearer image: {path}')
            before = len(vectors)
            for face_index, face in enumerate(faces):
                if kind == 'animation' and len(faces) > 1 and face.get('subject_kind') != 'object':
                    vector = _isolated_animation_reference(model, frame, face, faces)
                else:
                    vector = model.extract(frame, face, width, height, faces)
                box_key = hashlib.sha256(json.dumps(
                    [round(float(value), 2) for value in face['face']]).encode()).hexdigest()[:12]
                region = {'id': f'reference-{hashes[index][:16]}-{box_key}',
                          'image_index': index, 'image_sha256': hashes[index], 'face_index': face_index,
                          'box': face['face'], 'whole_image': False,
                          'selected_region': bool(chosen), 'subject_kind': face.get('subject_kind','face')}
                if vector is None:
                    catalog['skipped_regions'].append({**region, 'reason': 'unreadable-identity-crop'})
                    continue
                vectors.append(vector.tolist())
                catalog['templates'].append(region)
            if len(vectors) == before:
                raise ValueError(f'Cannot extract character identity from reference: {path}')
    finally:
        if detector:
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
    gallery = config.get('_reference_gallery')
    def gallery_key(group, mode):
        return (identity['model_sha256'], tuple(sorted(t['id'] for t in group['templates'])), mode)
    for track in tracks:
        observations = np.asarray(track.get('vectors', []), np.float32)
        track.update(reference_match=False, reference_confidence=0., reference_id=None,
                     reference_template_id=None, reference_ambiguous=False)
        track.pop('reference_method', None)
        modes = [(observations, track.get('subject_kind'))]
        face_vectors = np.asarray(track.get('reference_face_vectors', []), np.float32)
        if len(face_vectors):
            # Body and head features must match the same crop type in references.
            modes = [(observations,'object'), (face_vectors,'face')]
        if max(len(vectors) for vectors,_ in modes) < 2:
            continue
        candidates = []
        for group in groups:
            for vectors,mode in modes:
                if len(vectors) < 2:
                    continue
                templates = [i for i in group['template_indices'] if mode is None or
                             catalog['templates'][i].get('subject_kind','face') == mode]
                if not templates:
                    continue
                distances = 1 - vectors @ references.T
                nearest = distances[:, templates].min(axis=1)
                mode_threshold = config.get('body_reference_cosine_distance',threshold) if kind == 'animation' and mode == 'object' else threshold
                accepted = nearest <= mode_threshold
                method = 'original-reference'
                # Body views can change substantially with camera angle. Learn
                # only from tracks already confirmed against the original crop.
                # Adapted matches never extend the gallery or its original bound.
                prototypes = gallery.get(gallery_key(group, mode), []) if gallery is not None and mode == 'object' else []
                if prototypes and np.mean(accepted) < .75:
                    close = (1-vectors @ np.asarray(prototypes, np.float32).T).min(axis=1) <= min(.15, mode_threshold)
                    accepted = accepted | (close & (nearest <= min(.4, mode_threshold+.10)))
                    method = 'confirmed-reference-view'
                fraction = float(np.mean(accepted))
                if fraction >= .75 and int(np.sum(accepted)) >= 2:
                    template = min(templates, key=lambda i:float(np.median(distances[:,i])))
                    candidates.append((float(np.median(nearest)),group['id'],fraction,template,method,mode))
        # Two crop modes matching one identity are still one candidate identity.
        candidates = list({rid:min((item for item in candidates if item[1]==rid),key=lambda item:item[0])
                           for rid in {item[1] for item in candidates}}.values())
        candidates.sort()
        if not candidates:
            continue
        distance, reference_id, fraction, template, method, mode = candidates[0]
        if len(candidates) > 1 and candidates[1][0]-distance < .03:
            track['reference_ambiguous'] = True
            continue
        track.update(reference_match=True, reference_id=reference_id,
                     reference_confidence=fraction, reference_distance=distance, reference_method=method,
                     reference_crop_mode=mode)
        track['reference_template_id'] = catalog['templates'][template]['id']
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
    if gallery is not None:
        for track in tracks:
            if (not track.get('reference_match') or track.get('reference_method') != 'original-reference'
                    or track.get('reference_crop_mode') != 'object'):
                continue
            group = next(g for g in groups if g['id'] == track['reference_id'])
            values = np.asarray(track['vectors'], np.float32)
            distances = (1-values @ references[group['template_indices']].T).min(axis=1)
            accepted = values[distances <= config.get('body_reference_cosine_distance',threshold)]
            if len(accepted) < 2:
                continue
            key = gallery_key(group, 'object')
            prototypes = gallery.setdefault(key, [])
            for value in accepted:
                if len(prototypes) >= 8:
                    break
                if not prototypes or min(1-np.asarray(prototypes) @ value) > .015:
                    prototypes.append(value.tolist())
    count = sum(t['reference_match'] for t in tracks)
    if progress:
        progress(f"Character references: {len(catalog['images'])} images, {len(groups)} targets; {count}/{len(tracks)} tracks matched")
        for region in catalog['skipped_regions']:
            progress(f"Reference face {region['face_index']+1} unreadable: {catalog['images'][region['image_index']]}; use a clearer crop")
    for group in groups:
        group['matched_tracks'] = sum(t.get('reference_id') == group['id'] for t in tracks)
        del group['template_indices']
    return {'images': catalog['images'], 'characters': groups, 'skipped_regions': catalog['skipped_regions'],
            'matched_tracks': count, 'cosine_distance': threshold,
            'cosine_distance_by_crop_type':{'face':threshold,'body':config.get('body_reference_cosine_distance',threshold)},
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
