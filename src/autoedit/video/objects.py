"""Optional whole-character detection, independent of face and lip landmarks."""
from copy import deepcopy
from bisect import bisect_right
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / 'models/objects/grounding-dino-tiny'
RUNTIME = ROOT / '.venv-objects' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
DEFAULT_PROMPT = 'a person. a skeleton. a monster. an animal. an animated character. a ball. a balloon. a vehicle. a bag. a poster. a sheet of paper. a teddy bear. a stuffed toy. a logo. a watermark. a basket. a table. a chair.'
PROP_WORDS = {'ball', 'balloon', 'bag', 'poster', 'paper', 'sheet', 'teddy', 'stuffed', 'logo', 'watermark', 'basket', 'table', 'chair'}
ACTOR_PROMPT = 'a person. a skeleton. a monster. an animal. an animated character.'
FACE_PROMPT = 'a face. a cartoon face.'
REFERENCE_PROMPT = ACTOR_PROMPT + ' a vehicle. a car. a truck. a bus. a robot. a toy. an object.'
OBJECT_VERSION = 3
SUBJECT_MERGE_VERSION = 19


def localized_faces(objects, width, height):
    """Independent tight face localizations; broad cartoon/body labels abstain.

    This establishes visible faces only, never mouth landmarks or speech.
    Prop and stationary-overlay rejection is applied before cast ranking.
    """
    from autoedit.video.characters import iou
    faces = []
    for obj in sorted(objects, key=lambda o:o.get('score',0), reverse=True):
        if obj.get('label') != 'a face' or obj.get('score',0) < .3:
            continue
        values = obj.get('box', [])
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            continue
        x,y,right,bottom = values
        if not (0 <= x < right <= width and 0 <= y < bottom <= height):
            continue
        box = [round(x),round(y),round(right-x),round(bottom-y)]
        if min(box[2:]) < 16 or not .4 <= box[2]/box[3] <= 2:
            continue
        if any(iou(box, f['face']) >= .5 for f in faces):
            continue
        faces.append(dict(backend='grounding-dino-face', face_detected=True, face=box,
            face_evidence='independent-tight-face-query', face_detector_score=obj['score'],
            face_clarity=float(obj.get('clarity',0.)),
            face_visible=1., mouth_box=[0,0,0,0], mouth_clarity=0., clear_face=0.,
            speaking=0., lip_geometry_measured=False, category='NEUTRAL', confidence=obj['score']))
    return faces


def face_on_prop(face, prop):
    """Resolve weak plush/poster confusion without overriding printed text props."""
    from autoedit.video.characters import iou
    from autoedit.video.subjects import contained_face
    words = set(str(prop.get('label','')).split())
    person = face.get('person_costume_confirmation') or {}
    confirmed_person = (person.get('score', 0) >= .45 and person.get('box')
        and face.get('face_evidence') == 'independent-tight-face-query'
        and contained_face(face['face'], person['box']))
    # The contextual model can call an entire animated frame a poster,
    # sometimes "a poster a sheet". Only a stronger independently measured
    # face resolves this ambiguity; localized printed faces still veto it.
    if (face.get('face_evidence') == 'independent-tight-face-query'
            and 'poster' in words and prop.get('frame_fraction', 0) >= .9
            and not words.intersection({'logo','watermark'})
            and (confirmed_person or (face.get('face_detector_score', 0) >= .4
                 and face['face_detector_score'] > prop.get('score', 1.)))):
        return False
    if (confirmed_person and words.intersection({'teddy','stuffed'})
            and not words.intersection({'paper','sheet','poster','logo','watermark'})):
        return False
    if (face.get('face_evidence') == 'independent-tight-face-query'
            and (words.intersection({'teddy','stuffed'}) or ('poster' in words and prop.get('full_frame')))
            and not words.intersection({'paper','sheet','logo','watermark'})
            and face.get('face_detector_score',0) >= .4
            and face['face_detector_score'] > prop.get('score',1.)):
        return False
    x,y,w,h = face['face']
    px,py,pw,ph = prop['box']
    overlap = max(0,min(x+w,px+pw)-max(x,px))*max(0,min(y+h,py+ph)-max(y,py))
    return (iou(face['face'],prop['box']) >= .5 or contained_face(face['face'],prop['box'])
            or (px <= x+w/2 <= px+pw and py <= y+h/2 <= py+ph and overlap/max(1,w*h) >= .35))


def subject_family(subject):
    """Independent, unambiguous type evidence; generic labels stay unknown."""
    if subject.get('object_family') in {'person','creature','vehicle'}:
        return subject['object_family']
    words = set(str(subject.get('object_label', subject.get('label', ''))).split())
    person = 'person' in words
    creature = bool(words & {'skeleton', 'monster'})
    if words & {'vehicle','car','truck','bus'} and not person and not creature:
        return 'vehicle'
    if person != creature:
        return 'person' if person else 'creature'
    return None


def object_evidence(subject):
    if subject.get('backend') == 'dense-reference-body':
        box = subject.get('subject_box')
        return bool(subject.get('reference_match') and subject.get('reference_body_measured')
                    and subject.get('reference_method') == 'dense-body-embedding'
                    and subject.get('reference_region_kind') == 'body'
                    and subject.get('reference_confidence',0) >= .75
                    and subject.get('reference_observation_count',0) >= 2
                    and subject.get('reference_model_sha256') and subject.get('reference_sample_fps',0) >= 2
                    and 0 <= subject.get('reference_distance',1) <= subject.get('reference_distance_limit',0) <= .4
                    and box and len(box) == 4 and all(math.isfinite(x) for x in box) and min(box[2:]) >= 32)
    box = subject.get('subject_box')
    ranked_fallback = (subject.get('exposure_fallback_confirmed')
                       and subject.get('character_status') in ('main','supporting')
                       and subject.get('character_rank') is not None
                       and subject.get('identity_confidence',0) >= .5)
    return bool((not subject.get('target_candidate_only') or subject.get('reference_match') or ranked_fallback)
                and subject.get('object_detector') == 'grounding-dino' and subject.get('object_confirmed')
                and subject.get('object_score', 0) >= .3 and box and len(box) == 4
                and all(math.isfinite(x) for x in box) and min(box[2:]) >= 24)


def enable_ranked_exposure_fallback(info):
    """Explicit fallback can use independently detected, exposure-ranked inputs.

    Reference-only selection stays unchanged. This does not invent image matches,
    body detections or lip measurements for vehicles/robots/other input types.
    """
    result = deepcopy(info)
    for sample in result.get('samples',[]):
        for subject in [sample,*sample.get('faces',[])]:
            if (subject.get('target_candidate_only')
                    and subject.get('character_status') in ('main','supporting')
                    and subject.get('character_rank') is not None
                    and subject.get('identity_confidence',0) >= .5):
                subject['exposure_fallback_confirmed'] = True
    return result


def normalize_objects(objects, width, height, threshold=.3, positive_labels=None, allow_vehicle=False,
                      allow_reference_targets=False, prefer_body_boxes=False):
    from autoedit.video.characters import iou
    accepted = []
    prop_words = PROP_WORDS | (set() if allow_vehicle else {'vehicle'})
    props=[obj for obj in objects if prop_words.intersection(str(obj.get('label','')).split()) and obj.get('score',0) >= threshold
            and len(obj.get('box',[]))==4 and all(math.isfinite(v) for v in obj['box'])]
    if allow_reference_targets:
        props = []
    compact_props=[]
    for prop in sorted(props,key=lambda o:o['score'],reverse=True):
        if 'vehicle' in prop['label'].split():
            continue
        px0,py0,px1,py1=prop['box']
        pb=[px0,py0,px1-px0,py1-py0]
        if not any(iou(pb,[p['box'][0],p['box'][1],p['box'][2]-p['box'][0],p['box'][3]-p['box'][1]])>=.6
                   for p in compact_props):
            compact_props.append(prop)
    for obj in sorted(objects, key=lambda o:o.get('score',0), reverse=True):
        values = obj.get('box', [])
        if len(values) != 4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
            continue
        x0,y0,x1,y1 = values
        x0,y0,x1,y1 = max(0,x0),max(0,y0),min(width,x1),min(height,y1)
        box = [round(x0),round(y0),round(x1-x0),round(y1-y0)]
        if obj.get('score',0) < threshold or min(box[2:]) < 24:
            continue
        if obj in props:
            continue
        labels = str(obj.get('label','')).lower().split()
        if not allow_reference_targets and not any(word in labels for word in (positive_labels or
                (('person','skeleton','monster','animal','character','vehicle') if allow_vehicle else
                 ('person','skeleton','monster','animal','character')))):
            continue
        def prop_conflict(p):
            px0,py0,px1,py1=p['box']
            overlap=max(0,min(x1,px1)-max(x0,px0))*max(0,min(y1,py1)-max(y0,py0))
            if 'vehicle' in p['label'].split():
                return p['score']>obj['score']+.05 and iou(box,[px0,py0,px1-px0,py1-py0])>=.5
            # Smiley props can receive a higher "character" score than "ball".
            # Conflicting, independently located prop evidence must abstain.
            actor_area=(x1-x0)*(y1-y0)
            prop_area=(px1-px0)*(py1-py0)
            return (iou(box,[px0,py0,px1-px0,py1-py0])>=.5
                    or (actor_area <= prop_area and overlap/max(1,actor_area)>=.85)
                    or (overlap/max(1,actor_area)>=.35 and
                        x0 <= (px0+px1)/2 <= x1 and y0 <= (py0+py1)/2 <= y1))
        if any(prop_conflict(p) for p in props):
            continue
        inside=[p for p in compact_props if x0 <= (p['box'][0]+p['box'][2])/2 <= x1
                and y0 <= (p['box'][1]+p['box'][3])/2 <= y1]
        if len(inside)>=2 and sum((p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1])
                                  for p in inside) >= .2*(x1-x0)*(y1-y0):
            continue
        if any(iou(box,other['box']) >= .6 for other in accepted):
            continue
        accepted.append({'box':box,'label':str(obj.get('label','character')),'score':float(obj['score']),
                         'clarity':float(obj.get('clarity',0.)),
                         **({'target_candidate_only':True} if allow_reference_targets and not
                            any(word in labels for word in ('person','skeleton','monster','animal','character')) else {})})
    def family(label):
        if 'skeleton' in label or 'monster' in label:
            return 'creature'
        if 'person' in label:
            return 'person'
        # "animal" is often an alternative label for a stylised creature's
        # tight region. It cannot establish a second actor inside its body.
        if 'animal' in label:
            return 'creature'
        return 'character'
    # A tight character/head region and its encompassing body are alternative
    # localizations of one subject. Keep the tight region for appearance cues;
    # distinct foreground people and background creatures stay independent.
    result=[]
    if prefer_body_boxes:
        def crowd_box(obj):
            x,y,w,h = obj['box']
            inside = []
            for other in accepted:
                if other is obj:
                    continue
                ox,oy,ow,oh = other['box']
                overlap = max(0,min(x+w,ox+ow)-max(x,ox))*max(0,min(y+h,oy+oh)-max(y,oy))
                if ow*oh < .8*w*h and overlap/max(1,ow*oh) >= .85:
                    inside.append(other)
            return any(iou(a['box'],b['box']) < .05 for i,a in enumerate(inside) for b in inside[i+1:])
        accepted = [obj for obj in accepted if not crowd_box(obj)]
    for obj in sorted(accepted,key=lambda o:o['box'][2]*o['box'][3], reverse=prefer_body_boxes):
        x,y,w,h=obj['box']
        def duplicate(other):
            ox,oy,ow,oh=other['box']
            overlap=max(0,min(x+w,ox+ow)-max(x,ox))*max(0,min(y+h,oy+oh)-max(y,oy))
            compatible=(family(obj['label'])==family(other['label'])
                        or 'character' in (family(obj['label']),family(other['label'])))
            return compatible and overlap/max(1,min(w*h,ow*oh)) >= .85
        duplicates = [other for other in result if duplicate(other)]
        if not duplicates:
            obj['family_hint'] = subject_family(obj)
            result.append(obj)
        elif len(duplicates) == 1 and not duplicates[0].get('family_hint'):
            # Preserve the type established by the independently detected
            # body when retaining its tighter, generically labelled head.
            duplicates[0]['family_hint'] = subject_family(obj)
    return result


def merge_subjects(sample, objects, props=()):
    """Retain measured lips only when one face lies inside one detected body.

    The legacy `faces` list carries tracked subjects as well as faces. Body
    entries explicitly say face_detected=False; `track_box` stores their
    independent location, never an inferred head or mouth rectangle.
    """
    def prop_face(face):
        for prop in props:
            px0,py0,px1,py1 = prop['box']
            if face_on_prop(face,{**prop,'box':[px0,py0,px1-px0,py1-py0],
                    'full_frame':(px1-px0)*(py1-py0) >= .6*sample.get('frame_width',float('inf'))*
                        sample.get('frame_height',float('inf')),
                    'frame_fraction':(px1-px0)*(py1-py0) / max(1,
                        sample.get('frame_width',float('inf'))*sample.get('frame_height',float('inf')))}):
                return True
        return False
    # Human landmarks can falsely locate a face on smiley props too. Apply
    # independent prop evidence to both detector paths before identity ranking.
    faces = [f for f in sample.get('faces', []) if not prop_face(f)]
    assigned = set()
    result = []
    def contained(face, box):
        x,y,w,h = face['face']
        bx,by,bw,bh = box
        overlap=max(0,min(x+w,bx+bw)-max(x,bx))*max(0,min(y+h,by+bh)-max(y,by))
        return overlap/max(1,w*h) >= .8
    # Smaller boxes claim faces first, preventing a crowd-sized box from
    # taking the lips of a different foreground actor.
    for obj in sorted(objects,key=lambda o:o['box'][2]*o['box'][3]):
        members = [i for i,f in enumerate(faces) if i not in assigned and contained(f,obj['box'])]
        measured = faces[members[0]] if len(members) == 1 else None
        if measured is not None:
            assigned.add(members[0])
            subject = dict(measured)
        else:
            subject = {'backend':'grounding-dino-body','face_detected':False,'face':obj['box'],
                'face_visible':0.,'closeup_score':0.,'mouth_clarity':0.,'frontal_score':0.,
                'mouth_box':[0,0,0,0],'clear_face':0.,'speaking':0.,'category':'NEUTRAL',
                'confidence':obj['score'],'lip_geometry_measured':False}
        subject.update(subject_kind='object',object_detector='grounding-dino',object_confirmed=True,
            object_score=obj['score'],object_label=obj['label'],subject_box=obj['box'],track_box=obj['box'],
            body_visible=1.,person_visible=1.,full_body_visible=0.,subject_clarity=obj.get('clarity',0.))
        if obj.get('target_candidate_only') or obj.get('family_hint') == 'vehicle' or subject_family(obj) == 'vehicle':
            subject['target_candidate_only'] = True
        if obj.get('family_hint'):
            subject['object_family'] = obj['family_hint']
        result.append(subject)
    # Geometric cartoon hypotheses on furniture/subtitles need an independently
    # detected actor. Human landmarks remain independently measured evidence.
    result.extend(dict(f,subject_kind='face') for i,f in enumerate(faces) if i not in assigned
                  and f.get('backend') not in {'animation-eyes-mouth','feature-tracked-face'})
    return result


class ObjectScanSession:
    """One isolated worker per job; models and bounded decoded frames survive queries."""

    def __init__(self):
        self.process = None

    def __enter__(self):
        return self

    def run(self, command, request, progress, log_path):
        from autoedit.runtime import check_cancelled
        check_cancelled()
        request_id = json.loads(request.read_text(encoding='utf-8'))['request_id']
        if self.process is None:
            self.process = subprocess.Popen([command[0], command[1], '--serve'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
        self.process.stdin.write(json.dumps({'action': 'scan', 'path': str(request.resolve()),
                                            'request_id': request_id}) + '\n')
        self.process.stdin.flush()
        tail = []
        with log_path.open('w', encoding='utf-8') as logfile:
            for line in self.process.stdout:
                check_cancelled()
                logfile.write(line)
                tail = (tail + [line.strip()])[-12:]
                if line.startswith('OBJECT: ') and progress:
                    progress(line[8:].strip())
                if line.startswith('RESULT: '):
                    status = json.loads(line[8:])
                    if status.get('request_id') != request_id:
                        raise RuntimeError('Whole-character worker returned a stale request response')
                    if status.get('ok') is not True or 'error' in status:
                        raise RuntimeError('Whole-character detector failed: ' +
                            status.get('error_type', 'RuntimeError') + ': ' +
                            (status.get('error') or 'no error message') + '; log: ' + str(log_path))
                    return
        raise RuntimeError(f'Whole-character worker exited (code={self.process.wait()}); scan request={request_id}; log: '
                           + str(log_path) + '\n' + '\n'.join(tail))

    def __exit__(self, *args):
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait()
            finally:
                self.process.stdout.close()

    def offload(self):
        """Release weights instead of moving them into already pressured host RAM."""
        if self.process is None:
            return
        from autoedit.runtime import check_cancelled
        check_cancelled()
        request_id = uuid.uuid4().hex
        self.process.stdin.write(json.dumps({'action': 'release', 'request_id': request_id}) + '\n')
        self.process.stdin.flush()
        for line in self.process.stdout:
            check_cancelled()
            if line.startswith('RESULT: '):
                status = json.loads(line[8:])
                if status.get('request_id') != request_id or status.get('ok') is not True or 'error' in status:
                    raise RuntimeError('Whole-character release failed: ' + str(status))
                return
        raise RuntimeError(f'Whole-character worker exited during release (code={self.process.wait()})')


def _cached_object_scan(info, cache_dir, settings, progress, session=None):
    """Cache each query independently so adding prop checks preserves actor work."""
    source_identity = info.get('cache_identity') or {}
    if 'regional_requests' in source_identity:
        source_identity = source_identity['source']
    # Detection depends on the source/frame/query, not the discovery density or
    # requested window. Reuse it when refining overlapping candidate regions.
    source_identity = {k:source_identity.get(k) for k in ('path', 'size', 'mtime_ns')}
    identity = {'source':source_identity, 'path':str(Path(info['path']).resolve()), **settings,
                'model_mtime':(MODEL_DIR/'model.safetensors').stat().st_mtime_ns}
    samples = info['samples']
    if len({s['time_sec'] for s in samples}) != len(samples):
        raise RuntimeError('Whole-character detector received incomplete or duplicate timestamps')
    def sample_key(s):
        value = [s['time_sec'], s['frame_width'], s['frame_height'], s.get('roi'), s.get('query')]
        return hashlib.sha256(json.dumps(value).encode()).hexdigest()
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory=Path(cache_dir)
    directory.mkdir(parents=True,exist_ok=True)
    cache=directory/f'objects-{key}-frames.json.gz'
    saved = {'rows':{}, 'elapsed_sec':0., 'runtime':None}
    def validate(data):
        if len(data['samples']) != len(info['samples']):
            raise RuntimeError('Whole-character detector returned incomplete analysis')
        if any(abs(sample['time_sec']-row['time_sec']) > 1e-5
               for sample,row in zip(info['samples'],data['samples'])):
            raise RuntimeError('Whole-character detection timestamp mismatch')
    if cache.is_file():
        try:
            saved=json.loads(gzip.decompress(cache.read_bytes()))
        except (OSError, ValueError, EOFError):
            pass
    missing = [s for s in samples if sample_key(s) not in saved['rows']]
    if not missing:
        if progress:
            progress('Whole-character cache hit: reusing independent object detections')
    else:
        request=directory/f'objects-{key}.request.json'
        result=directory/f'objects-{key}.result.json'
        from autoedit.runtime import cancellation_file
        request_id = uuid.uuid4().hex
        request.write_text(json.dumps({'request_id': request_id, 'cancel_file': cancellation_file(),
            'path':info['path'],'samples':[
            {'time_sec':s['time_sec'],'width':s['frame_width'],'height':s['frame_height'],
             **({'roi':s['roi'],'query':s.get('query')} if s.get('roi') else {})}
            for s in missing], 'model_dir':str(MODEL_DIR),'settings':settings,
            'result':str(result.resolve())}),encoding='utf-8')
        command=[str(RUNTIME),str(Path(__file__).with_name('object_worker.py')),str(request.resolve())]
        log_path=directory/f'objects-{key}.log'
        def commit(data, partial=False):
            rows = data['samples']
            if data.get('request_id') != request_id:
                raise RuntimeError('Whole-character detector returned stale results from a different request')
            if len(rows) > len(missing) or (not partial and len(rows) != len(missing)) or any(
                    abs(s['time_sec']-r['time_sec']) > 1e-5 for s,r in zip(missing,rows)):
                raise RuntimeError('Whole-character detector returned incomplete analysis or mismatched timestamps')
            saved['rows'].update({sample_key(s):row for s,row in zip(missing,rows)})
            saved['runtime'] = data.get('runtime')
            saved['elapsed_sec'] = ((saved['elapsed_sec'] or 0) + data.get('elapsed_sec', 0))
            temporary=cache.with_suffix('.tmp')
            temporary.write_bytes(gzip.compress(json.dumps(saved).encode()))
            temporary.replace(cache)
        try:
            _run_object_request(session, command, request, progress, log_path)
        except BaseException:
            # A completed batch is independent evidence. Preserve it on error/cancel,
            # but never accept a result belonging to an earlier request.
            try:
                commit(json.loads(result.read_text(encoding='utf-8')), partial=True)
            except (OSError, ValueError, KeyError, RuntimeError):
                pass
            raise
        data=json.loads(result.read_text(encoding='utf-8'))
        if data.get('complete') is not True:
            raise RuntimeError('Whole-character detector did not complete this request')
        commit(data)
    data = {'samples':[saved['rows'][sample_key(s)] for s in samples],
            'runtime':saved['runtime'], 'elapsed_sec':saved['elapsed_sec']}
    validate(data)
    return data,{**identity, 'sample_keys':[sample_key(s) for s in samples]}


def _run_object_request(session, command, request, progress, log_path):
    from autoedit.runtime import check_cancelled
    if session is not None:
        session.run(command, request, progress, log_path)
    else:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace',
                                 creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        tail=[]
        with log_path.open('w',encoding='utf-8') as log:
            for line in process.stdout:
                try:
                    check_cancelled()
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
                log.write(line)
                tail=(tail+[line.strip()])[-12:]
                if progress and line.startswith('OBJECT: '):
                    progress(line[8:].strip())
        if process.wait():
            raise RuntimeError(f'Whole-character detector failed; log: {log_path}\n'+ '\n'.join(tail))


def _regional_requests(samples, observations, transitions, threshold, positive_labels=None, allow_vehicle=False,
                       allow_reference_targets=False):
    """Re-detect a brief hole inside independently located bracketing actors.

    Geometry only requests a full-resolution crop. It never supplies presence
    or identity; the missing frame still needs its own accepted model box.
    """
    from autoedit.video.characters import iou
    detected=[normalize_objects(row['objects'],s['frame_width'],s['frame_height'],threshold,positive_labels,allow_vehicle,
                                allow_reference_targets)
              for s,row in zip(samples,observations)]
    result=[]
    for index in range(1,len(samples)-1):
        before,current,after=samples[index-1:index+2]
        if after['time_sec']-before['time_sec'] > .45:
            continue
        if bisect_right(transitions,after['time_sec']) != bisect_right(transitions,before['time_sec']):
            continue
        choices=[]
        for left in detected[index-1]:
            for right in detected[index+1]:
                family=left.get('family_hint') or subject_family(left)
                if not family or family != (right.get('family_hint') or subject_family(right)):
                    continue
                if any((obj.get('family_hint') or subject_family(obj)) == family for obj in detected[index]):
                    continue
                x,y,w,h=left['box']; ox,oy,ow,oh=right['box']
                overlap=max(0,min(x+w,ox+ow)-max(x,ox))*max(0,min(y+h,oy+oh)-max(y,oy))
                if iou(left['box'],right['box']) < .25 and overlap/max(1,min(w*h,ow*oh)) < .85:
                    continue
                x0,y0,x1,y1=min(x,ox),min(y,oy),max(x+w,ox+ow),max(y+h,oy+oh)
                pad_x,pad_y=(x1-x0)*.15,(y1-y0)*.15
                roi=[max(0,x0-pad_x),max(0,y0-pad_y),min(current['frame_width'],x1+pad_x),
                     min(current['frame_height'],y1+pad_y)]
                choices.append((min(left['score'],right['score']),roi,family))
        if choices:
            _,roi,family=max(choices,key=lambda entry:entry[0])
            result.append({'time_sec':current['time_sec'],'frame_width':current['frame_width'],
                           'frame_height':current['frame_height'],'roi':roi,'expected_family':family,
                           'query':{'person':'a person.', 'vehicle':'a vehicle.'}.get(family,'a skeleton. a monster.')})
    return result


def reference_search_prompt(config, object_config, cache_dir, session, progress=None):
    """Use detections on each selected input crop to shorten the first-pass query.

    This is a search hint only. Any unclassified target retains the broad query;
    exhaustive search restores it too, so labels never exclude imported targets.
    """
    from PIL import Image
    regions = {str(Path(p).resolve()): boxes for p,boxes in config.get('reference_regions',{}).items()}
    labels = set()
    for name in config.get('reference_images', []):
        path = Path(name).resolve()
        selected = regions.get(str(path))
        if not selected:
            return REFERENCE_PROMPT
        stat = path.stat()
        with Image.open(path) as image:
            scale = min(1.,640/image.width)
            width, height = round(image.width*scale),round(image.height*scale)
        samples = []
        for index, region in enumerate(selected):
            x,y,w,h = region['box']
            samples.append({'time_sec':float(index),'frame_width':width,'frame_height':height,
                            'roi':[x*scale,y*scale,(x+w)*scale,(y+h)*scale]})
        info = {'path':str(path),'samples':samples,'cache_identity':{'path':str(path),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}}
        settings = {'device':object_config.get('device','auto'), 'threshold':.3, 'prompt':REFERENCE_PROMPT,
                    'batch_size':object_config.get('batch_size',2), 'compute_type':object_config.get('compute_type','auto'), 'version':3}
        if 'device_index' in object_config:
            settings['device_index'] = object_config['device_index']
        data,_ = _cached_object_scan(info,cache_dir,settings,progress,session)
        narrow_data = None
        for index,(sample,row) in enumerate(zip(samples,data['samples'])):
            x0,y0,x1,y1 = sample['roi']
            def score(obj):
                ox0,oy0,ox1,oy1 = obj['box']
                area = max(0,min(x1,ox1)-max(x0,ox0))*max(0,min(y1,oy1)-max(y0,oy0))
                union = (x1-x0)*(y1-y0)+max(0,ox1-ox0)*max(0,oy1-oy0)-area
                return (area/max(1,union),obj.get('score',0))
            accepted = [obj for obj in row['objects'] if score(obj)[0] >= .35]
            if not accepted:
                if narrow_data is None:
                    narrow_data,_ = _cached_object_scan(info,cache_dir,
                        {**settings,'prompt':'a car. a truck. a bus. a robot. a toy.'},progress,session)
                accepted = [obj for obj in narrow_data['samples'][index]['objects'] if score(obj)[0] >= .35]
                if not accepted:
                    return REFERENCE_PROMPT
            best = max(accepted,key=lambda obj:score(obj)[0]*obj['score'])
            words = set(str(best['label']).split()) & {'person','skeleton','monster','animal','character','vehicle','car','truck','bus','robot','toy'}
            if len(words) != 1:
                return REFERENCE_PROMPT
            labels.update(words)
    prompt = ' '.join(f'a {label}.' for label in sorted(labels)) if labels else REFERENCE_PROMPT
    if progress:
        progress('Reference search hint: ' + prompt + ' Broad queries remain available for missed targets.')
    return prompt


def augment_with_objects(info, cache_dir, config=None, progress=None, session=None):
    config = config or {}
    mode = config.get('enabled','auto')
    if mode != 'auto' and not isinstance(mode,bool):
        raise ValueError('video.object_detection.enabled must be auto, true or false')
    if mode is False:
        return info
    ready = RUNTIME.is_file() and (MODEL_DIR/'model.safetensors').is_file()
    if not ready:
        if mode is True:
            raise FileNotFoundError('Whole-character detector missing. Run scripts/setup_object_detector.py.')
        if progress:
            progress('Whole-character model not installed: retaining face/pose analysis')
        return info
    if not Path(info['path']).is_file():
        return info
    threshold = float(config.get('threshold',.3))
    if not math.isfinite(threshold) or not .3 <= threshold <= .9:
        raise ValueError('Object detection threshold must be between .3 and .9')
    reference_targets = bool(config.get('target_references'))
    target_prompt = config.get('target_prompt') or REFERENCE_PROMPT
    settings = {'threshold':threshold, 'prompt':config.get('prompt') or (target_prompt if reference_targets else DEFAULT_PROMPT),
                'device':config.get('device','auto'),'version':OBJECT_VERSION}
    for field in ('batch_size', 'compute_type', 'device_index'):
        if field in config:
            settings[field] = config[field]
    scan_args = (session,) if session is not None else ()
    data,identity = _cached_object_scan(info,cache_dir,settings,progress,*scan_args)
    face_data = face_identity = None
    if info.get('source_kind') == 'animation' and not reference_targets:
        if progress:
            progress('Independent animation face query: measuring faces missed by human landmarks')
        face_data,face_identity = _cached_object_scan(info,cache_dir,
            {**settings,'prompt':FACE_PROMPT,'version':1,
             'batch_size':config.get('face_batch_size',settings.get('batch_size','auto'))},progress,*scan_args)
    primary = None
    primary_identity = None
    if not config.get('prompt') and not reference_targets:
        if progress:
            progress('Independent actor query: preserving actor recall before prop rejection')
        # Actor-only inference uses the unchanged v1 raw-box schema. Its cache
        # is independent of later additions to contextual/negative labels.
        actor_settings={**settings,'prompt':REFERENCE_PROMPT if reference_targets else ACTOR_PROMPT + (' a vehicle.' if config.get('target_vehicles') else ''),'version':1}
        primary,primary_identity = _cached_object_scan(info,cache_dir,actor_settings,progress,*scan_args)
    observations=[{'objects':row['objects']+(primary['samples'][index]['objects'] if primary else [])}
                  for index,row in enumerate(data['samples'])]
    requests=_regional_requests(info['samples'],observations,sorted(info.get('transition_times_sec') or []),
                                threshold,config.get('positive_labels'),bool(config.get('target_vehicles')),reference_targets)
    regional_data = None
    regional_identity = None
    if requests:
        if progress:
            progress(f'Full-resolution object refinement: {len(requests)} brief detection holes; no inferred subjects')
        regional_info={**info,'samples':requests,'cache_identity':{'source':info.get('cache_identity'),
                      'regional_requests':requests,'regional_version':2}}
        regional_data,regional_identity=_cached_object_scan(regional_info,cache_dir,
            {**settings,'prompt':config.get('prompt') or (settings['prompt'] if reference_targets else ACTOR_PROMPT),'version':1},progress,*scan_args)
        by_time={sample['time_sec']:index for index,sample in enumerate(info['samples'])}
        for request,row in zip(requests,regional_data['samples']):
            observations[by_time[row['time_sec']]]['objects'].extend(
                obj for obj in row['objects'] if subject_family(obj)==request['expected_family'])
    merged=deepcopy(info)
    if face_data:
        from autoedit.video.subjects import confirm_animation_people
        face_data, person_identity = confirm_animation_people(info, data, face_data, cache_dir,
            {**settings,'batch_size':config.get('face_batch_size',settings.get('batch_size','auto'))},progress,session)
    else:
        person_identity = None
    object_samples = 0
    for index,(sample,row) in enumerate(zip(merged['samples'],data['samples'])):
        if abs(sample['time_sec']-row['time_sec']) > 1e-5:
            raise RuntimeError('Whole-character detection timestamp mismatch')
        objects=normalize_objects(observations[index]['objects'],sample['frame_width'],sample['frame_height'],threshold,
                                  config.get('positive_labels'), allow_vehicle=bool(config.get('target_vehicles')),
                                  allow_reference_targets=reference_targets,
                                  prefer_body_boxes=bool(config.get('target_body_references')))
        object_samples += bool(objects)
        props=[obj for obj in row['objects'] if obj.get('score',0) >= threshold
               and 'vehicle' not in str(obj.get('label','')).split()
               and not {'person','skeleton','monster','character'}.intersection(str(obj.get('label','')).split())
               and PROP_WORDS.intersection(str(obj.get('label','')).split())
               and len(obj.get('box',[])) == 4 and all(math.isfinite(v) for v in obj['box'])]
        if reference_targets:
            props = []
        sample['prop_regions'] = [{'box':[p['box'][0],p['box'][1],
            p['box'][2]-p['box'][0],p['box'][3]-p['box'][1]], 'label':p['label'],'score':p['score'],
            'full_frame':(p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1]) >=
                .6*sample['frame_width']*sample['frame_height'],
            'frame_fraction':(p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1]) /
                (sample['frame_width']*sample['frame_height'])} for p in props]
        if face_data:
            from autoedit.video.characters import iou
            # Discard conflicting older hypotheses before deduplication, so a
            # rejected geometry/landmark cannot hide a stronger independent face.
            sample['faces'] = [f for f in sample.get('faces',[]) if not any(
                face_on_prop(f,p) for p in sample['prop_regions'])]
            for face in localized_faces(face_data['samples'][index]['objects'],sample['frame_width'],sample['frame_height']):
                confirmation = next((p for p in face_data['samples'][index].get('person_faces', [])
                    if p['face'] == face['face']), None)
                if confirmation:
                    face['person_costume_confirmation'] = confirmation['person']
                # Prefer actual mouth landmarks when both detectors agree.
                if not any(f.get('face_detected') and iou(face['face'],f['face']) >= .35
                           for f in sample.get('faces',[])):
                    area = sample['frame_width']*sample['frame_height']
                    w,h = face['face'][2:]
                    face.update(closeup_score=min(1.,w*h/area/.22))
                    face['readable_face'] = float(face['face_clarity'] >= .25
                        and w/sample['frame_width'] >= .05 and h/sample['frame_height'] >= .075)
                    sample.setdefault('faces',[]).append(face)
        for face in sample.get('faces',[]):
            if face.get('face_detected') and 'face_clarity' in face:
                x,y,w,h = face['face']
                face['readable_face'] = float(face['face_clarity'] >= .25
                    and w/sample['frame_width'] >= .05 and h/sample['frame_height'] >= .075
                    and x >= 0 and y >= 0 and x+w <= sample['frame_width'] and y+h <= sample['frame_height'])
        sample['faces']=merge_subjects(sample,objects,props)
    merged['cache_identity']={**(info.get('cache_identity') or {}),'object_detection':identity,
                              'actor_detection':primary_identity,
                              'regional_detection':regional_identity,
                              'animation_face_detection':face_identity,
                              'person_costume_detection':person_identity,
                              'subject_merge_version':SUBJECT_MERGE_VERSION,
                              'positive_object_labels':config.get('positive_labels')}
    merged['cache_identity']['target_vehicles'] = bool(config.get('target_vehicles'))
    merged['cache_identity']['target_references'] = reference_targets
    merged['cache_identity']['target_body_references'] = bool(config.get('target_body_references'))
    pass_elapsed = [data.get('elapsed_sec')]+([face_data.get('elapsed_sec')] if face_data else [])+([primary.get('elapsed_sec')] if primary else [])+(
        [regional_data.get('elapsed_sec')] if regional_data else [])
    if person_identity:
        pass_elapsed.append(person_identity.get('elapsed_sec'))
    merged['object_analysis']={'model':'grounding-dino-tiny','runtime':data.get('runtime'),
        'sample_count':len(data['samples']),'object_samples':object_samples,
        'raw_object_samples':sum(bool(s['objects']) for s in data['samples']),
        'settings':settings,'actor_query':ACTOR_PROMPT if primary else None,
        'elapsed_sec':sum(pass_elapsed) if all(v is not None for v in pass_elapsed) else None,
        'pass_elapsed_sec':pass_elapsed,
        'regional_samples':len(requests),
        'inference_count':len(data['samples'])*(2 if primary else 1)+len(requests)+
            (len(face_data['samples']) if face_data else 0)+(person_identity.get('queried_frames',0) if person_identity else 0)}
    from autoedit.video.subjects import exclude_screen_overlays, refine_actor_faces, refine_animation_actor_faces
    merged = exclude_screen_overlays(merged, progress)
    if not reference_targets:
        if info.get('source_kind') == 'animation':
            merged = refine_animation_actor_faces(merged,cache_dir,
                {**settings,'batch_size':config.get('face_batch_size',settings.get('batch_size','auto'))},progress,session)
            measured = merged['actor_face_queries']
            merged['object_analysis']['inference_count'] += measured['queried_regions']
            merged['object_analysis']['pass_elapsed_sec'].append(measured['elapsed_sec'])
            times = merged['object_analysis']['pass_elapsed_sec']
            merged['object_analysis']['elapsed_sec'] = sum(times) if all(t is not None for t in times) else None
        merged = refine_actor_faces(merged, cache_dir, progress)
    return merged
