"""Dense image-identity evidence for explicitly selected whole-character crops."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from functools import lru_cache

import numpy as np

from autoedit.runtime import check_cancelled, release_inference
from autoedit.video.characters import settings
from autoedit.video.embeddings import EmbeddingModel, fingerprint, model_path
from autoedit.video.reader import sampled_frames
from autoedit.video.reference_search import screening_tiles
from autoedit.video.references import reference_vectors, match_reference_tracks, _reference_groups

VERIFY_VERSION = 2


@lru_cache(maxsize=8)
def _model_digest(path, size, mtime_ns):
    return fingerprint(path)


def verify_body_references(raw, cache_dir, config, gallery, progress=None):
    config = settings(config)
    path = model_path('animation',config)
    stat = path.stat()
    digest = _model_digest(path,stat.st_size,stat.st_mtime_ns)
    identity = {'model_sha256':digest}
    refs, catalog = reference_vectors('animation', config, cache_dir, identity)
    refs = np.asarray(refs, np.float32)
    groups = _reference_groups(refs, catalog, config['ccip_cosine_distance'])
    key = [raw.get('cache_identity'), digest, VERIFY_VERSION,
           [[s['time_sec'], s['frame_width'], s['frame_height']] for s in raw['samples']]]
    dest = Path(cache_dir)/('body-verified-'+hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()+'.npz')
    if dest.is_file():
        with np.load(dest, allow_pickle=False) as data:
            values = data['vectors']
    else:
        model = EmbeddingModel('animation', config)
        samples = raw['samples']
        frames = sampled_frames(raw['path'],raw['sample_fps'], raw.get('cache_identity',{}).get('max_seconds',
            samples[-1]['time_sec']+1/raw['sample_fps']), start_sec=samples[0]['time_sec'])
        values = []
        try:
            for index,(t,frame) in enumerate(frames):
                check_cancelled()
                if index >= len(samples) or abs(t-samples[index]['time_sec']) > 1e-5:
                    raise RuntimeError('Dense reference verification timestamp mismatch')
                height,width = frame.shape[:2]
                extracted = model.extract_many(frame,screening_tiles(width,height),width,height)
                values.append([v if v is not None else np.zeros_like(refs[0]) for v in extracted])
            if len(values) != len(samples):
                raise RuntimeError('Dense reference verification returned incomplete frames')
            values = np.asarray(values,np.float32)
            dest.parent.mkdir(parents=True,exist_ok=True)
            temp = dest.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream,vectors=values)
            temp.replace(dest)
        finally:
            frames.close()
            del model
            release_inference({})
    return qualify_body_references(raw,values,refs,groups,config,cache_dir,identity,gallery,progress)


def qualify_body_references(raw, values, refs, groups, config, cache_dir, identity, gallery, progress=None):
    """Every accepted sample has fresh dense features; holes remain non-evidence."""
    info = deepcopy(raw)
    samples = info['samples']
    tile_count = len(screening_tiles(samples[0]['frame_width'],samples[0]['frame_height']))
    if values.shape != (len(samples),tile_count,refs.shape[1]) or not np.isfinite(values).all():
        raise RuntimeError('Invalid dense reference feature sequence')
    # A tile also contains scenery and neighbours. Its deliberately broad
    # discovery threshold cannot certify a character's identity. Keep the
    # original-reference identity limit even for confirmed alternate views.
    threshold = min(config.get('body_reference_cosine_distance',config['ccip_cosine_distance']),
                    config['ccip_cosine_distance'])
    interval = 1/info['sample_fps']
    ranges = info['verified_ranges']
    tracks = []
    choices = []
    selected_vectors = {}
    for group_index,group in enumerate(groups):
        indices = group['template_indices']
        distances = np.clip(1-values @ refs[indices].T,0,2)
        best = np.argmin(distances.min(axis=2),axis=1)
        selected = values[np.arange(len(samples)),best]
        observed = [i for i,s in enumerate(samples) if any(a <= s['time_sec'] < b for a,b in ranges)]
        tracks.append({'id':f'reference-tile-{group_index}', 'subject_kind':'object', 'shot_id':0,
            'observations':[[i,group_index] for i in range(len(samples))],
            'intervals':[[samples[i]['time_sec'],samples[i]['time_sec']+interval] for i in observed],
            'vectors':selected[observed].tolist()})
        choices.append((best,selected,distances.min(axis=(1,2))))
        selected_vectors[group['id']] = selected
    reference = match_reference_tracks(tracks,'animation',
        {**config,'body_reference_cosine_distance':threshold,'_reference_gallery':gallery},cache_dir,identity)
    for sample in samples:
        sample.setdefault('_raw_faces',deepcopy(sample.get('faces',[])))
        sample['faces'] = deepcopy(sample['_raw_faces'])
        sample.update(reference_match=False,reference_id=None,reference_template_id=None,
                      reference_confidence=0.,reference_body_measured=False)
        for face in sample['faces']:
            face.update(reference_match=False,reference_id=None,reference_template_id=None,
                        reference_confidence=0.,reference_body_measured=False)
    for track,group,(best,vectors,distances) in zip(tracks,groups,choices):
        if not track.get('reference_match') or track.get('reference_id') != group['id']:
            continue
        accepted = distances <= threshold
        for index,sample in enumerate(samples):
            if not accepted[index]:
                continue
            tiles = screening_tiles(sample['frame_width'],sample['frame_height'])
            box = tiles[int(best[index])]['subject_box']
            face = {'backend':'dense-reference-body', 'subject_kind':'object', 'face':box,
                'subject_box':box,'track_box':box,'face_detected':False, 'face_visible':0.,
                'body_visible':1., 'person_visible':1., 'full_body_visible':0.,
                'clear_face':0., 'mouth_clarity':0., 'mouth_box':[0,0,0,0], 'closeup_score':0.,
                'frontal_score':0., 'speaking':0., 'category':'NEUTRAL', 'confidence':1.,
                'lip_geometry_measured':False, 'subject_clarity':0.,
                'character_status':'main','character_id':group['id'],'character_rank':1,
                'identity_confidence':1., 'track_id':track['id'],
                'reference_match':True,'reference_id':track['reference_id'],
                'reference_template_id':track['reference_template_id'],
                'reference_confidence':track['reference_confidence'],
                'reference_method':'dense-body-embedding','reference_body_measured':True,
                'reference_region_kind':'body','body_localization':'image-tile',
                'reference_observation_count':len(track['vectors']),
                'reference_model_sha256':identity['model_sha256'],
                'reference_distance':float(distances[index]),'reference_distance_limit':threshold,
                'reference_sample_fps':info['sample_fps']}
            sample['faces'].append(face)
    from autoedit.video.characters import _annotate_subjects
    _annotate_subjects(info,info.get('source_kind','animation'))
    original_events = raw.get('unfiltered_transition_times_sec',raw.get('transition_times_sec',[]))
    info['unfiltered_transition_times_sec'] = list(original_events)
    retained, ignored = [], []
    hard = raw.get('shot_times_sec',[])
    for event in original_events:
        nearby = [i for i,s in enumerate(samples) if abs(s['time_sec']-event) <= .3]
        cohort = [samples[i] for i in nearby]
        same_target = (len(cohort) >= 3 and cohort[0]['time_sec'] <= event <= cohort[-1]['time_sec']
            and all(s.get('reference_body_measured') and s.get('reference_match')
                    and s.get('reference_id') == cohort[0].get('reference_id') for s in cohort)
            and all(b['time_sec']-a['time_sec'] <= interval+1/max(info.get('fps') or 25,1)+1e-6
                    for a,b in zip(cohort,cohort[1:])))
        if same_target and not any(abs(event-t) <= .3 for t in hard):
            vectors = selected_vectors.get(cohort[0]['reference_id'])
            if vectors is not None:
                measured = vectors[nearby]
                if np.max(1-measured @ measured.T) <= .15:
                    ignored.append(event)
                    continue
        retained.append(event)
    info['transition_times_sec'] = retained
    info['reference_background_events_ignored_sec'] = ignored
    info['character_analysis'] = {'reference':reference,'min_main_fraction':config['min_main_fraction'],
        'selection_policy':'dense-imported-body-reference','model_sha256':identity['model_sha256'],
        'clusters':[], 'tracks':[{k:v for k,v in t.items() if k != 'vectors'} for t in tracks]}
    info['_body_reference_verification'] = True
    template_ids = {template['id'] for group in groups for template in group['templates']}
    info['_reference_search_views'] = [value for key,prototypes in gallery.items()
        if isinstance(key,tuple) and len(key) == 3 and key[0] == identity['model_sha256']
        and key[2] == 'object' and set(key[1]) <= template_ids for value in prototypes]
    info['_reference_view_generation'] = hashlib.sha256(np.asarray(info['_reference_search_views'],np.float32).tobytes()).hexdigest()
    info['object_analysis'] = {'model':'dense-reference-embedding','sample_count':len(samples),'inference_count':0}
    matched = sum(bool(s.get('reference_match')) for s in samples)
    if progress and matched != raw.get('_dense_reference_match_count'):
        progress(f'Dense input identity: {matched}/{len(samples)} frames matched')
    info['_dense_reference_match_count'] = matched
    return info
