"""Dense lip samples only for already selected cuts; no long-source rescan."""
from pathlib import Path
import hashlib
import json
from bisect import bisect_right


def refine_selected_cuts(path, video_slots, cache_dir, progress=None, sample_fps=20, source_kind="live_action",
                         character_samples=None, main_group="auto", object_config=None):
    """Cache each cut independently so changing an allocation reuses other cuts."""
    samples, ranges = [], []
    unique = {}
    for slot in video_slots:
        unique[(slot['video']['in_sec'], slot['video']['out_sec'],
                (slot['video'].get('selection_metrics') or {}).get('subject_track_id'))] = slot
    from autoedit.video.objects import ObjectScanSession
    with ObjectScanSession() as object_session:
        for slot in sorted(unique.values(), key=lambda s: s['video']['in_sec']):
            info = _refine_ranges(path, [slot], cache_dir, progress, sample_fps, source_kind,
                                  character_samples, main_group, object_session, object_config)
            samples.extend(info['samples'])
            ranges.extend(info['ranges'])
    return {'samples': samples, 'sample_fps': sample_fps, 'method': 'selected-lips-20fps',
            'ranges': ranges, 'path': str(Path(path).resolve())}


def _refine_ranges(path, video_slots, cache_dir, progress=None, sample_fps=20, source_kind="live_action",
                   character_samples=None, main_group="auto", object_session=None, object_config=None):
    import av
    from autoedit.video.faces import FaceLandmarker, MultiRegionFaceLandmarker, MODEL_PATH, annotate_speaking
    from autoedit.video.analyzer import _visual_quality, _try_cv2
    source = Path(path).resolve()
    object_config = object_config or {}
    from autoedit.video.objects import FACE_PROMPT
    face_query = {'threshold': object_config.get('threshold', .3), 'prompt': FACE_PROMPT,
                  'device': object_config.get('device', 'cpu'), 'version': 1,
                  'batch_size': object_config.get('batch_size', 1),
                  'compute_type': object_config.get('compute_type', 'auto')}
    ranges = sorted({(s["video"]["in_sec"], s["video"]["out_sec"]) for s in video_slots})
    references_by_range = {(start,end): cut_references(character_samples or [], slot)
                           for slot in video_slots for start,end in [(slot['video']['in_sec'],slot['video']['out_sec'])]}
    references = [s for refs in references_by_range.values() for s in refs]
    reference_identity = [{k: s.get(k) for k in ("time_sec", "face", "frame_width", "frame_height",
                                                "reference_range", "backend", "subject_box", "face_detection_region", "prop_regions", "person_costume_confirmation")}
                          for s in references]
    identity = [str(source), source.stat().st_size, source.stat().st_mtime_ns, ranges, sample_fps,
                source_kind, MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else None, reference_identity, 17,
                face_query, object_config.get('enabled', 'auto')]
    if main_group != "auto":
        identity.append({"main_group":main_group,"version":4})
    if source_kind == 'animation':
        identity.append({'animation_detector_version':7})
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    dest = Path(cache_dir) / ("lips-" + key + ".json")
    if dest.is_file():
        try:
            info = json.loads(dest.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            info = None
        if info is not None:
            if character_samples is not None:
                times = [s['time_sec'] for s in references]
                info['samples'] = [{**s, **select_reference_face([s] if s.get('face_detected') else [],
                    s['frame_width'], s['frame_height'], s['time_sec'], references, times)} for s in info['samples']]
            return info
    if progress:
        progress(f"Verifying faces and lips at {sample_fps} samples/s for {len(ranges)} selected cuts only")
    cv2, _ = _try_cv2()
    detector = ((MultiRegionFaceLandmarker() if main_group == "auto" else
                 MultiRegionFaceLandmarker(main_group=main_group))
                if source_kind == "animation" else MultiRegionFaceLandmarker(include_animation=False))
    samples = []
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 2
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0
            for start, end in ranges:
                range_references = references_by_range[(start,end)]
                reference_times = [s['time_sec'] for s in range_references]
                context_start, context_end = max(0., start-.35), end+.35
                container.seek(int((context_start + origin) / float(stream.time_base)), stream=stream, backward=True)
                next_time = context_start
                for frame in container.decode(stream):
                    from autoedit.runtime import check_cancelled
                    check_cancelled()
                    if frame.time is None:
                        continue
                    t = float(frame.time) - origin
                    if t >= context_end:
                        break
                    if t < context_start or (not start-.25 <= t <= start+.55 and t + 1e-6 < next_time):
                        continue
                    frame = frame.reformat(width=960, height=round(frame.height * 960/frame.width), format="bgr24")
                    image = frame.to_ndarray(format="bgr24")
                    det = detector.detect(image, t)
                    candidates = det.get('faces', [])
                    if character_samples is not None:
                        det = select_reference_face(det.get("faces", []), image.shape[1], image.shape[0],
                                                    t, range_references, reference_times)
                    quality, _ = _visual_quality(cv2, image, det)
                    det.update(quality)
                    det["time_sec"] = t
                    det['refinement_cut'] = {'in_sec': start, 'out_sec': end}
                    det['frame_width'], det['frame_height'] = image.shape[1], image.shape[0]
                    det['_face_candidates'] = candidates
                    samples.append(det)
                    next_time = t + 1/sample_fps
    finally:
        detector.close()
    if object_config.get('enabled', 'auto') is not False and source_kind == 'animation' and any(s.get('backend') == 'grounding-dino-face'
            or s.get('face_detection_region') == 'original-actor-roi' for s in references):
        from autoedit.video.objects import _cached_object_scan, FACE_PROMPT, localized_faces
        stat = source.stat()
        data,_ = _cached_object_scan({'path':str(source),'samples':samples,
            'cache_identity':{'path':str(source),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}},cache_dir,
            face_query,
            progress,object_session)
        for sample,row in zip(samples,data['samples']):
            bounds = sample['refinement_cut']
            refs = references_by_range[(bounds['in_sec'],bounds['out_sec'])]
            times = [s['time_sec'] for s in refs]
            faces = localized_faces(row['objects'],sample['frame_width'],sample['frame_height'])
            for face in faces:
                w,h = face['face'][2:]
                face.update(closeup_score=min(1.,w*h/(sample['frame_width']*sample['frame_height'])/.22),
                    readable_face=float(face['face_clarity'] >= .25
                        and w/sample['frame_width'] >= .05 and h/sample['frame_height'] >= .075))
            selected = select_localized_reference_face(sample['_face_candidates']+faces,
                sample['frame_width'],sample['frame_height'],sample['time_sec'],refs,times)
            if selected.get('face_detected'):
                sample.update(selected)
            else:
                sample.update(selected, face_visible=0., readable_face=0., clear_face=0., mouth_clarity=0.)
        from autoedit.match.matcher import _has_face
        requests,indices = [],[]
        for index,sample in enumerate(samples):
            if _has_face(sample) and sample.get('readable_face'):
                continue
            bounds = sample['refinement_cut']
            refs = references_by_range[(bounds['in_sec'],bounds['out_sec'])]
            if not refs:
                continue
            reference = min(refs,key=lambda s:abs(s['time_sec']-sample['time_sec']))
            if abs(reference['time_sec']-sample['time_sec']) > .5 or not reference.get('face'):
                continue
            x,y,w,h = reference.get('subject_box') or reference['face']
            if not reference.get('subject_box'):
                x,y,w,h = x-.35*w,y-.55*h,1.7*w,1.75*h
            sx,sy = sample['frame_width']/reference['frame_width'],sample['frame_height']/reference['frame_height']
            requests.append({**sample,'roi':[max(0,x*sx),max(0,y*sy),
                min(sample['frame_width'],(x+w)*sx),min(sample['frame_height'],(y+h)*sy)],'query':FACE_PROMPT})
            indices.append(index)
        if requests:
            rows,_ = _cached_object_scan({'path':str(source),'samples':requests,
                'cache_identity':{'path':str(source),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}},cache_dir,
                face_query,
                progress,object_session)
            for index,row in zip(indices,rows['samples']):
                sample = samples[index]
                bounds = sample['refinement_cut']
                refs = references_by_range[(bounds['in_sec'],bounds['out_sec'])]
                faces = localized_faces(row['objects'],sample['frame_width'],sample['frame_height'])
                for face in faces:
                    w,h = face['face'][2:]
                    face.update(closeup_score=min(1.,w*h/(sample['frame_width']*sample['frame_height'])/.22),
                        readable_face=float(face['face_clarity'] >= .25
                            and w/sample['frame_width'] >= .05 and h/sample['frame_height'] >= .075))
                selected = select_localized_reference_face(faces,sample['frame_width'],sample['frame_height'],
                    sample['time_sec'],refs,[s['time_sec'] for s in refs])
                if selected.get('face_detected'):
                    sample.update(selected)
    for sample in samples:
        sample.pop('_face_candidates',None)
    annotate_speaking(samples, source_kind=source_kind)
    info = {"samples": samples, "sample_fps": sample_fps, "method": "selected-lips-20fps",
            "ranges": ranges, "path": str(source)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(info), encoding="utf-8")
    return info


def select_localized_reference_face(faces, width, height, time, references, reference_times):
    """Dense independent faces must follow the selected actor and avoid props."""
    from autoedit.video.objects import face_on_prop
    index = bisect_right(reference_times,time)-1
    choices = [i for i in (index,index+1) if 0 <= i < len(references)]
    if not choices:
        return select_reference_face([],width,height,time,references,reference_times)
    reference = references[min(choices,key=lambda i:abs(reference_times[i]-time))]
    sx,sy = width/reference['frame_width'],height/reference['frame_height']
    props = [{**p,'box':[x*sx,y*sy,w*sx,h*sy]} for p in reference.get('prop_regions',[]) for x,y,w,h in [p['box']]]
    confirmation = reference.get('person_costume_confirmation')
    if confirmation and abs(reference['time_sec']-time) <= .5 and reference.get('face'):
        from autoedit.video.characters import iou
        x,y,w,h = reference['face']
        expected = [x*sx,y*sy,w*sx,h*sy]
        bx,by,bw,bh = confirmation['box']
        for face in faces:
            if iou(face['face'],expected) >= .25:
                face['person_costume_confirmation'] = {**confirmation,'box':[bx*sx,by*sy,bw*sx,bh*sy]}
    return select_reference_face([f for f in faces if not any(face_on_prop(f,p) for p in props)],
                                 width,height,time,references,reference_times)


def cut_references(samples, slot):
    """Dense verification follows the actual selected co-visible face track."""
    cut=slot['video']
    selected_track=(cut.get('selection_metrics') or {}).get('subject_track_id')
    result=[]
    for sample in samples:
        if not cut['in_sec']-.5 <= sample['time_sec'] <= cut['out_sec']+.5:
            continue
        reference=dict(sample)
        if selected_track:
            selected=next((face for face in sample.get('faces',[]) if face.get('track_id') == selected_track),None)
            if selected is None:
                continue
            from autoedit.video.subjects import reset_selected_subject
            reset_selected_subject(reference)
            reference.update(selected)
        reference['reference_range']=[cut['in_sec'],cut['out_sec']]
        result.append(reference)
    return sorted(result,key=lambda sample:sample['time_sec'])


def readable_face_fraction(samples, start, end, maximum_interval=.1):
    """Time-weighted dense face evidence; missing time is never counted as clear."""
    from autoedit.match.matcher import _has_face
    samples = sorted((s for s in samples if start-maximum_interval <= s['time_sec'] <= end+maximum_interval),
                     key=lambda s:s['time_sec'])
    readable = covered = 0.
    for index, sample in enumerate(samples):
        following = samples[index+1]['time_sec'] if index+1 < len(samples) else sample['time_sec']+maximum_interval
        weight = max(0., min(end, following, sample['time_sec']+maximum_interval)-max(start,sample['time_sec']))
        covered += weight
        if _has_face(sample) and sample.get('readable_face', sample.get('clear_face', 0)):
            readable += weight
    length = end-start
    return {'readable_face_fraction': readable/length, 'sample_coverage': covered/length}


def select_reference_face(faces, width, height, time, references, reference_times):
    """Keep dense lip analysis on the subject chosen in the source scan."""
    from autoedit.video.characters import iou
    missing = {"backend": "mediapipe-landmarks", "face_detected": False,
               "category": "NEUTRAL", "confidence": 0.0}
    index = bisect_right(reference_times, time) - 1
    choices = [i for i in (index, index + 1) if 0 <= i < len(references)]
    if not choices or not faces:
        return missing
    reference = references[min(choices, key=lambda i: abs(reference_times[i] - time))]
    if abs(reference["time_sec"] - time) > .5 or not reference.get("face"):
        return missing
    x, y, w, h = reference["face"]
    sx, sy = width / reference["frame_width"], height / reference["frame_height"]
    expected = [x * sx, y * sy, w * sx, h * sy]
    ordered = sorted(faces, key=lambda face: iou(face["face"], expected), reverse=True)
    selected = ordered[0]
    if iou(selected["face"], expected) < .25:
        return missing
    if len(ordered) > 1 and iou(selected["face"], expected) - iou(ordered[1]["face"], expected) < .1:
        return missing
    return {**selected, **{k: reference.get(k) for k in ("character_id", "track_id", "shot_id",
                                                      'reference_match', 'reference_confidence',
                                                      'reference_id', 'reference_template_id',
                                                      'reference_ambiguous')}}
