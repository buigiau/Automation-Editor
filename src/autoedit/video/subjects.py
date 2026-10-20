"""Reject screen overlays before they can become cast identities."""
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
import gzip
import hashlib
import json

import numpy as np

SUBJECT_FILTER_VERSION = 1

# Pose/frame evidence belongs to the sample; selected-subject evidence does not.
SUBJECT_FIELDS = {
    'backend', 'face', 'face_detected', 'face_visible', 'face_keypoints',
    'face_evidence', 'face_detection_region', 'feature_anchor_sec',
    'face_detector_score', 'actor_face_measured', 'person_costume_confirmation',
    'verified_feature_count', 'mouth_box', 'mouth_clarity', 'frontal_score',
    'closeup_score', 'clear_face', 'face_clarity', 'readable_face', 'speaking', 'openness', 'width', 'roundness',
    'smile', 'lip_aperture', 'lip_width_ratio', 'lip_geometry_measured',
    'category', 'confidence', 'subject_kind', 'subject_box', 'track_box',
    'track_id', 'track_association', 'track_previous_box', 'object_detector',
    'object_confirmed', 'object_score', 'object_label', 'object_family',
    'body_visible', 'full_body_visible', 'subject_clarity',
    'character_id', 'character_rank', 'character_status', 'identity_confidence',
    'character_group', 'group_evidence', 'target_candidate_only',
    'exposure_fallback_confirmed', 'reference_match', 'reference_id',
    'reference_confidence', 'reference_template_id', 'reference_ambiguous',
}


def reset_selected_subject(sample):
    for key in SUBJECT_FIELDS:
        sample.pop(key, None)
    sample.update(face_detected=False, face_visible=0., clear_face=0., speaking=0.,
                  face_clarity=0., readable_face=0.,
                  closeup_score=0., mouth_clarity=0., body_visible=0.,
                  person_visible=float(bool(sample.get('pose_confirmed') and sample.get('person_detected'))))


def exclude_screen_overlays(info, progress=None):
    """Measure small corner crops fixed across different shots, not actor stillness.

    A static actor inside one shot is retained. Geometry alone never rejects a
    corner subject: original pixels must also agree across separate shots.
    """
    from autoedit.video.characters import iou, _selected_source_frames
    samples = info.get('samples', [])
    shots = sorted(info.get('shot_times_sec') or info.get('transition_times_sec') or [])
    groups = []
    for si, sample in enumerate(samples):
        width, height = sample.get('frame_width', 0), sample.get('frame_height', 0)
        if not width or not height:
            continue
        for fi, subject in enumerate(sample.get('faces', [])):
            box = subject.get('subject_box', subject.get('face'))
            if not box or len(box) != 4:
                continue
            x, y, w, h = box
            near_corner = ((x <= width*.08 or x+w >= width*.92)
                           and (y <= height*.08 or y+h >= height*.92))
            if not near_corner or not (0 < w <= width*.18 and 0 < h <= height*.25):
                continue
            group = next((g for g in groups if g['size'] == (width, height)
                          and iou(g['box'], box) >= .75), None)
            if group is None:
                group = {'box': list(box), 'size': (width, height), 'observations': []}
                groups.append(group)
            group['observations'].append((si, fi))
    requests = defaultdict(list)
    candidates = []
    for group in groups:
        observations = group['observations']
        times = [samples[si]['time_sec'] for si, _ in observations]
        shot_indices = {bisect_right(shots, time) for time in times}
        if len(set(times)) < 6 or max(times)-min(times) < 3 or len(shot_indices) < 2:
            continue
        # Every examined shot supplies a crop, with a bounded decode budget.
        by_shot = defaultdict(list)
        for si, _ in observations:
            by_shot[bisect_right(shots, samples[si]['time_sec'])].append(si)
        indices = [items[len(items)//2] for items in by_shot.values()]
        indices = [indices[int(i)] for i in np.linspace(0, len(indices)-1, min(8, len(indices)), dtype=int)]
        group['pixels'] = []
        candidates.append(group)
        for si in indices:
            requests[samples[si]['time_sec']].append(group)
    if not requests or not Path(info.get('path', '')).is_file():
        return info
    import cv2
    for time, frame in _selected_source_frames(info['path'], sorted(requests)):
        for group in requests[time]:
            width, height = group['size']
            x, y, w, h = group['box']
            sx, sy = frame.shape[1]/width, frame.shape[0]/height
            crop = frame[max(0, round(y*sy)):round((y+h)*sy), max(0, round(x*sx)):round((x+w)*sx)]
            if crop.size:
                group['pixels'].append(cv2.resize(crop, (32,32), interpolation=cv2.INTER_AREA).astype(np.float32)/255)
    rejected = []
    for group in candidates:
        pixels = group['pixels']
        if len(pixels) < 2:
            continue
        # Transparent watermark margins reveal a different background in each
        # shot. Require a textured, unchanged foreground inside the fixed crop.
        values = np.asarray(pixels)
        stable = (values.max(axis=0)-values.min(axis=0)).mean(axis=-1) <= .05
        texture = float(values[0][stable].std(axis=0).mean()) if stable.any() else 0.
        group['stable_pixel_fraction'] = float(stable.mean())
        if stable.mean() >= .55 and texture >= .1:
            rejected.append(group)
    for sample in samples:
        size = (sample.get('frame_width'), sample.get('frame_height'))
        boxes = [g['box'] for g in rejected if g['size'] == size]
        def overlay(subject):
            box = subject.get('subject_box', subject.get('face'))
            if not box:
                return False
            return any(iou(box, candidate) >= .6 or (
                candidate[0] <= box[0]+box[2]/2 <= candidate[0]+candidate[2]
                and candidate[1] <= box[1]+box[3]/2 <= candidate[1]+candidate[3]
                and box[2]*box[3] <= candidate[2]*candidate[3]) for candidate in boxes)
        sample['faces'] = [subject for subject in sample.get('faces', []) if not overlay(subject)]
        if overlay(sample):
            reset_selected_subject(sample)
    info['subject_filter'] = {'version': SUBJECT_FILTER_VERSION,
        'screen_overlays': [{'box': g['box'], 'frame_size': list(g['size']),
                             'observation_count': len(g['observations']),
                             'stable_pixel_fraction': g['stable_pixel_fraction']} for g in rejected]}
    info['cache_identity'] = {**(info.get('cache_identity') or {}), 'subject_filter': info['subject_filter']}
    if progress and rejected:
        progress(f'Excluded {len(rejected)} measured screen overlays from actor evidence and cast ranking')
    return info


def refine_actor_faces(info, cache_dir, progress=None):
    """Measure missing faces inside independently confirmed, prop-filtered actors.

    Coordinates come from landmarks on original pixels, never a presumed head
    rectangle. An ambiguous face shared by two actor regions is not attached.
    """
    from autoedit.video.faces import FaceLandmarker, MODEL_PATH
    from autoedit.video.characters import _selected_source_frames
    from autoedit.video.analyzer import _visual_quality
    from autoedit.video.objects import object_evidence
    from autoedit.match.matcher import _has_face
    requests = []
    for si, sample in enumerate(info.get('samples', [])):
        for fi, subject in enumerate(sample.get('faces', [])):
            if object_evidence(subject) and not _has_face(subject):
                requests.append({'sample':si,'subject':fi,'time_sec':sample['time_sec'],
                    'box':subject['subject_box'], 'size':[sample['frame_width'],sample['frame_height']]})
    if not requests or not Path(info.get('path','')).is_file():
        return info
    source = Path(info['path']).stat()
    identity = {'version':1,'path':info['path'],'size':source.st_size,'mtime':source.st_mtime_ns,
                'model_mtime':MODEL_PATH.stat().st_mtime_ns,'requests':requests}
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir)/f'actor-faces-{digest}.json.gz'
    measured = None
    if cache.is_file():
        try:
            saved = json.loads(gzip.decompress(cache.read_bytes()))
            if saved.get('identity') == identity:
                measured = saved['faces']
        except (OSError,ValueError,EOFError,KeyError):
            pass
    if measured is None:
        import cv2
        by_time = defaultdict(list)
        for index, request in enumerate(requests):
            by_time[request['time_sec']].append((index,request))
        measured = [None]*len(requests)
        detectors = []
        if progress:
            progress(f'Measuring faces in {len(requests)} confirmed actor crops at original resolution')
        try:
            for count,(time,frame) in enumerate(_selected_source_frames(info['path'],sorted(by_time)),1):
                entries = by_time[time]
                while len(detectors) < len(entries):
                    detectors.append(FaceLandmarker())
                sample = info['samples'][entries[0][1]['sample']]
                width,height = sample['frame_width'],sample['frame_height']
                sx,sy = frame.shape[1]/width,frame.shape[0]/height
                preview = cv2.resize(frame,(width,height),interpolation=cv2.INTER_AREA)
                for slot,(index,request) in enumerate(entries):
                    x,y,w,h = request['box']
                    left,top = max(0,round(x*sx)),max(0,round(y*sy))
                    right,bottom = min(frame.shape[1],round((x+w)*sx)),min(frame.shape[0],round((y+h)*sy))
                    crop = frame[top:bottom,left:right]
                    if not crop.size or min(crop.shape[:2]) < 32:
                        continue
                    found = detectors[slot].detect_all(crop,time)
                    if len(found) != 1:
                        continue
                    face = found[0]
                    for key in ('face','mouth_box'):
                        bx,by,bw,bh = face[key]
                        face[key] = [round((bx+left)/sx),round((by+top)/sy),round(bw/sx),round(bh/sy)]
                    face['face_keypoints'] = [[(px+left)/sx,(py+top)/sy] for px,py in face['face_keypoints']]
                    owners = [subject for subject in sample['faces'] if object_evidence(subject)
                              and contained_face(face['face'],subject['subject_box'])]
                    if len(owners) != 1 or owners[0]['subject_box'] != request['box']:
                        continue
                    face.update(_visual_quality(cv2,preview,face)[0])
                    face.update(face_detection_region='original-actor-roi', actor_face_measured=True)
                    measured[index] = face
                if progress and count % 100 == 0:
                    progress(f'Actor face refinement: {count}/{len(by_time)} source frames')
        finally:
            for detector in detectors:
                detector.close()
        cache.parent.mkdir(parents=True,exist_ok=True)
        temporary = cache.with_suffix('.tmp')
        temporary.write_bytes(gzip.compress(json.dumps({'identity':identity,'faces':measured}).encode()))
        temporary.replace(cache)
    count = 0
    for request,face in zip(requests,measured):
        sample = info['samples'][request['sample']]
        if face and not any(contained_face(face['face'],prop['box']) for prop in sample.get('prop_regions', [])):
            sample['faces'][request['subject']].update(face)
            count += 1
    info['actor_face_refinement'] = {'version':1,'requested_crops':len(requests),'measured_faces':count}
    info['cache_identity'] = {**(info.get('cache_identity') or {}),'actor_face_refinement':digest}
    if progress:
        progress(f'Confirmed {count}/{len(requests)} actor faces from original-resolution landmarks')
    return info


def refine_animation_actor_faces(info, cache_dir, settings, progress=None, session=None):
    """Measure small/profile cartoon faces in independently detected actors.

    Body rectangles only schedule original-pixel queries. Each accepted face
    needs a new tight face detection, one owner and independent prop rejection.
    """
    from autoedit.video.objects import _cached_object_scan, localized_faces, object_evidence, face_on_prop, FACE_PROMPT
    from autoedit.match.matcher import _has_face
    groups = []
    for si,sample in enumerate(info['samples']):
        missing = [(fi,f) for fi,f in enumerate(sample['faces']) if object_evidence(f) and not _has_face(f)]
        for slot,(fi,subject) in enumerate(missing):
            while len(groups) <= slot:
                groups.append([])
            x,y,w,h = subject['subject_box']
            groups[slot].append((si,fi,{'time_sec':sample['time_sec'],
                'frame_width':sample['frame_width'],'frame_height':sample['frame_height'],
                'roi':[x,y,x+w,y+h],'query':FACE_PROMPT}))
    count = 0
    identities = []
    elapsed = []
    if groups and progress:
        progress(f'Native actor face queries: {sum(map(len,groups))} small/profile character regions')
    for entries in groups:
        requests = [request for _,_,request in entries]
        rows,identity = _cached_object_scan({**info,'samples':requests},cache_dir,
            {**settings,'prompt':FACE_PROMPT,'version':1},progress,session)
        identities.append(identity)
        elapsed.append(rows.get('elapsed_sec'))
        for (si,fi,request),row in zip(entries,rows['samples']):
            sample = info['samples'][si]
            faces = localized_faces(row['objects'],sample['frame_width'],sample['frame_height'])
            if len(faces) != 1:
                continue
            face = faces[0]
            owners = [f for f in sample['faces'] if object_evidence(f) and contained_face(face['face'],f['subject_box'])]
            if len(owners) != 1 or owners[0] is not sample['faces'][fi]:
                continue
            if any(face_on_prop(face,p) for p in sample.get('prop_regions',[])):
                continue
            w,h = face['face'][2:]
            face.update(closeup_score=min(1.,w*h/(sample['frame_width']*sample['frame_height'])/.22),
                readable_face=float(face['face_clarity'] >= .25
                    and w/sample['frame_width'] >= .05 and h/sample['frame_height'] >= .075),
                face_detection_region='native-actor-face-query')
            sample['faces'][fi].update(face)
            count += 1
    info['actor_face_queries'] = {'version':1,'queried_regions':sum(map(len,groups)),'measured_faces':count,
        'elapsed_sec':sum(elapsed) if all(t is not None for t in elapsed) else None}
    info['cache_identity'] = {**(info.get('cache_identity') or {}),'actor_face_queries':identities}
    if groups and progress:
        progress(f'Confirmed {count} additional small/profile animation faces')
    return info


def confirm_animation_people(info, context, face_data, cache_dir, settings, progress=None, session=None):
    """Resolve costume/plush ambiguity using a new person query and a tight face.

    A person box never invents a face. Localized printed props remain vetoes.
    """
    from autoedit.video.objects import _cached_object_scan, localized_faces, face_on_prop, PROP_WORDS
    requests, candidates = [], {}
    for index, (sample, row) in enumerate(zip(info['samples'], context['samples'])):
        width,height = sample['frame_width'],sample['frame_height']
        props = []
        for obj in row['objects']:
            words = set(str(obj.get('label','')).split())
            if (obj.get('score',0) < .3 or not PROP_WORDS.intersection(words)
                    or words.intersection({'person','skeleton','monster','character','vehicle'})):
                continue
            x,y,right,bottom = obj['box']
            fraction = (right-x)*(bottom-y)/(width*height)
            props.append({**obj,'box':[x,y,right-x,bottom-y],
                          'full_frame':fraction >= .6,'frame_fraction':fraction})
        for face in localized_faces(face_data['samples'][index]['objects'],width,height):
            blockers = [p for p in props if face_on_prop(face,p)]
            if not blockers:
                continue
            def ambiguous(prop):
                words = set(prop['label'].split())
                return ((words.intersection({'teddy','stuffed'}) and not words.intersection({'paper','sheet','poster','logo','watermark'}))
                    or ('poster' in words and prop['frame_fraction'] >= .9))
            if all(ambiguous(p) for p in blockers):
                candidates.setdefault(index, []).append(face)
        if index in candidates:
            requests.append({'time_sec':sample['time_sec'],'frame_width':width,'frame_height':height})
    if not requests:
        return face_data, None
    prompt = 'a person wearing a costume. a child in a costume. an animated person.'
    if progress:
        progress(f'Confirming people in {len(requests)} frames with costume/prop ambiguity')
    rows, identity = _cached_object_scan({**info,'samples':requests}, cache_dir,
        {**settings,'prompt':prompt,'version':1},progress,session)
    by_time = {row['time_sec']:row for row in rows['samples']}
    confirmed = 0
    for index, faces in candidates.items():
        sample = info['samples'][index]
        people = by_time[sample['time_sec']]['objects']
        matches = []
        for face in faces:
            x,y,w,h = face['face']
            owners = []
            for obj in people:
                if obj.get('score',0) < .45 or not {'person','child'}.intersection(obj.get('label','').split()):
                    continue
                bx,by,right,bottom = obj['box']
                box = [bx,by,right-bx,bottom-by]
                if (contained_face(face['face'],box) and h <= .6*box[3]
                        and w <= .85*box[2] and y+h/2 <= by+.65*box[3]):
                    owners.append({'box':box,'score':obj['score'],'method':'independent-person-costume-query'})
            if owners:
                matches.append({'face':face['face'],'person':max(owners,key=lambda p:p['score'])})
        face_data['samples'][index]['person_faces'] = matches
        confirmed += len(matches)
    if progress:
        progress(f'Confirmed {confirmed} independently localized faces on people in costumes')
    return face_data, {**identity,'queried_frames':len(requests),'confirmed_faces':confirmed,
                       'elapsed_sec':rows.get('elapsed_sec')}


def contained_face(face, body):
    x,y,w,h = face
    bx,by,bw,bh = body
    intersection = max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))
    return w > 0 and h > 0 and intersection/(w*h) >= .8
