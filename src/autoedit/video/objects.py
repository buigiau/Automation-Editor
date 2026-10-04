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

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / 'models/objects/grounding-dino-tiny'
RUNTIME = ROOT / '.venv-objects' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
DEFAULT_PROMPT = 'a person. a skeleton. a monster. an animal. an animated character. a ball. a balloon. a vehicle. a bag. a poster. a sheet of paper.'
ACTOR_PROMPT = 'a person. a skeleton. a monster. an animal. an animated character.'
OBJECT_VERSION = 3
SUBJECT_MERGE_VERSION = 9


def subject_family(subject):
    """Independent, unambiguous type evidence; generic labels stay unknown."""
    if subject.get('object_family') in {'person','creature'}:
        return subject['object_family']
    words = set(str(subject.get('object_label', subject.get('label', ''))).split())
    person = 'person' in words
    creature = bool(words & {'skeleton', 'monster'})
    if person != creature:
        return 'person' if person else 'creature'
    return None


def object_evidence(subject):
    box = subject.get('subject_box')
    return bool(subject.get('object_detector') == 'grounding-dino' and subject.get('object_confirmed')
                and subject.get('object_score', 0) >= .3 and box and len(box) == 4
                and all(math.isfinite(x) for x in box) and min(box[2:]) >= 24)


def normalize_objects(objects, width, height, threshold=.3, positive_labels=None):
    from autoedit.video.characters import iou
    accepted = []
    props=[obj for obj in objects if any(word in str(obj.get('label','')).split()
            for word in ('ball','balloon','vehicle','bag','poster','paper','sheet')) and obj.get('score',0) >= threshold
            and len(obj.get('box',[]))==4 and all(math.isfinite(v) for v in obj['box'])]
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
        if not any(word in labels for word in (positive_labels or ('person','skeleton','monster','animal','character'))):
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
                         'clarity':float(obj.get('clarity',0.))})
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
    for obj in sorted(accepted,key=lambda o:o['box'][2]*o['box'][3]):
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
        from autoedit.video.characters import iou
        x,y,w,h = face['face']
        for prop in props:
            px0,py0,px1,py1 = prop['box']
            pw,ph = px1-px0,py1-py0
            overlap = max(0,min(x+w,px1)-max(x,px0))*max(0,min(y+h,py1)-max(y,py0))
            if iou(face['face'],[px0,py0,pw,ph]) >= .5 or (w*h <= pw*ph and overlap/max(1,w*h) >= .85):
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
        if obj.get('family_hint'):
            subject['object_family'] = obj['family_hint']
        result.append(subject)
    result.extend(dict(f,subject_kind='face') for i,f in enumerate(faces) if i not in assigned)
    return result


def _cached_object_scan(info, cache_dir, settings, progress):
    """Cache each query independently so adding prop checks preserves actor work."""
    identity = {'source':info.get('cache_identity'),**settings,
                'model_mtime':(MODEL_DIR/'model.safetensors').stat().st_mtime_ns}
    key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    directory=Path(cache_dir)
    directory.mkdir(parents=True,exist_ok=True)
    cache=directory/f'objects-{key}.json.gz'
    def validate(data):
        if len(data['samples']) != len(info['samples']):
            raise RuntimeError('Whole-character detector returned incomplete analysis')
        if any(abs(sample['time_sec']-row['time_sec']) > 1e-5
               for sample,row in zip(info['samples'],data['samples'])):
            raise RuntimeError('Whole-character detection timestamp mismatch')
    if cache.is_file():
        data=json.loads(gzip.decompress(cache.read_bytes()))
        if progress:
            progress('Whole-character cache hit: reusing independent object detections')
    else:
        request=directory/f'objects-{key}.request.json'
        result=directory/f'objects-{key}.result.json'
        request.write_text(json.dumps({'path':info['path'],'samples':[
            {'time_sec':s['time_sec'],'width':s['frame_width'],'height':s['frame_height'],
             **({'roi':s['roi'],'query':s.get('query')} if s.get('roi') else {})}
            for s in info['samples']], 'model_dir':str(MODEL_DIR),'settings':settings,
            'result':str(result.resolve())}),encoding='utf-8')
        command=[str(RUNTIME),str(Path(__file__).with_name('object_worker.py')),str(request.resolve())]
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        log_path=directory/f'objects-{key}.log'
        tail=[]
        with log_path.open('w',encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                tail=(tail+[line.strip()])[-12:]
                if progress and line.startswith('OBJECT: '):
                    progress(line[8:].strip())
        if process.wait():
            raise RuntimeError(f'Whole-character detector failed; log: {log_path}\n'+ '\n'.join(tail))
        data=json.loads(result.read_text(encoding='utf-8'))
        validate(data)
        temporary=cache.with_suffix('.tmp')
        temporary.write_bytes(gzip.compress(json.dumps(data).encode()))
        temporary.replace(cache)
    validate(data)
    return data,identity


def _regional_requests(samples, observations, transitions, threshold, positive_labels=None):
    """Re-detect a brief hole inside independently located bracketing actors.

    Geometry only requests a full-resolution crop. It never supplies presence
    or identity; the missing frame still needs its own accepted model box.
    """
    from autoedit.video.characters import iou
    detected=[normalize_objects(row['objects'],s['frame_width'],s['frame_height'],threshold,positive_labels)
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
                           'query':'a person.' if family=='person' else 'a skeleton. a monster.'})
    return result


def augment_with_objects(info, cache_dir, config=None, progress=None):
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
    settings = {'threshold':threshold, 'prompt':config.get('prompt') or DEFAULT_PROMPT,
                'device':config.get('device','auto'),'version':OBJECT_VERSION}
    data,identity = _cached_object_scan(info,cache_dir,settings,progress)
    primary = None
    primary_identity = None
    if not config.get('prompt'):
        if progress:
            progress('Independent actor query: preserving actor recall before prop rejection')
        # Actor-only inference uses the unchanged v1 raw-box schema. Its cache
        # is independent of later additions to contextual/negative labels.
        actor_settings={**settings,'prompt':ACTOR_PROMPT,'version':1}
        primary,primary_identity = _cached_object_scan(info,cache_dir,actor_settings,progress)
    observations=[{'objects':row['objects']+(primary['samples'][index]['objects'] if primary else [])}
                  for index,row in enumerate(data['samples'])]
    requests=_regional_requests(info['samples'],observations,sorted(info.get('transition_times_sec') or []),
                                threshold,config.get('positive_labels'))
    regional_data = None
    regional_identity = None
    if requests:
        if progress:
            progress(f'Full-resolution object refinement: {len(requests)} brief detection holes; no inferred subjects')
        regional_info={**info,'samples':requests,'cache_identity':{'source':info.get('cache_identity'),
                      'regional_requests':requests,'regional_version':2}}
        regional_data,regional_identity=_cached_object_scan(regional_info,cache_dir,
            {**settings,'prompt':config.get('prompt') or ACTOR_PROMPT,'version':1},progress)
        by_time={sample['time_sec']:index for index,sample in enumerate(info['samples'])}
        for request,row in zip(requests,regional_data['samples']):
            observations[by_time[row['time_sec']]]['objects'].extend(
                obj for obj in row['objects'] if subject_family(obj)==request['expected_family'])
    merged=deepcopy(info)
    object_samples = 0
    for index,(sample,row) in enumerate(zip(merged['samples'],data['samples'])):
        if abs(sample['time_sec']-row['time_sec']) > 1e-5:
            raise RuntimeError('Whole-character detection timestamp mismatch')
        objects=normalize_objects(observations[index]['objects'],sample['frame_width'],sample['frame_height'],threshold,
                                  config.get('positive_labels'))
        object_samples += bool(objects)
        props=[obj for obj in row['objects'] if obj.get('score',0) >= threshold
               and 'vehicle' not in str(obj.get('label','')).split()
               and any(word in str(obj.get('label','')).split() for word in ('ball','balloon','bag','poster','paper','sheet'))
               and len(obj.get('box',[])) == 4 and all(math.isfinite(v) for v in obj['box'])]
        sample['faces']=merge_subjects(sample,objects,props)
    merged['cache_identity']={**(info.get('cache_identity') or {}),'object_detection':identity,
                              'actor_detection':primary_identity,
                              'regional_detection':regional_identity,
                              'subject_merge_version':SUBJECT_MERGE_VERSION,
                              'positive_object_labels':config.get('positive_labels')}
    pass_elapsed = [data.get('elapsed_sec')]+([primary.get('elapsed_sec')] if primary else [])+(
        [regional_data.get('elapsed_sec')] if regional_data else [])
    merged['object_analysis']={'model':'grounding-dino-tiny','runtime':data.get('runtime'),
        'sample_count':len(data['samples']),'object_samples':object_samples,
        'raw_object_samples':sum(bool(s['objects']) for s in data['samples']),
        'settings':settings,'actor_query':ACTOR_PROMPT if primary else None,
        'elapsed_sec':sum(pass_elapsed) if all(v is not None for v in pass_elapsed) else None,
        'pass_elapsed_sec':pass_elapsed,
        'regional_samples':len(requests),
        'inference_count':len(data['samples'])*(2 if primary else 1)+len(requests)}
    return merged
