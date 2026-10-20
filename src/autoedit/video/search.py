"""Cheap candidate discovery followed by unchanged dense evidence checks."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from autoedit.runtime import check_cancelled, release_inference
from autoedit.video.analyzer import ANALYSIS_VERSION, analyze_video, _merge_segments
from autoedit.video.characters import settings as character_settings, merged_intervals
from autoedit.video.reader import sampled_frames

SEARCH_VERSION = 4


def contains_range(ranges, start, end):
    return any(a - 1e-8 <= start and end <= b + 1e-8 for a, b in ranges)


def remaining_ranges(start, end, covered):
    ranges = [(start, end)]
    for a, b in merged_intervals(covered):
        ranges = [(lo, hi) for left, right in ranges
                  for lo, hi in ((left, min(right, a)), (max(left, b), right)) if hi > lo + 1e-8]
    return ranges


def merge_evidence(parts, duration, config):
    """Keep original timestamps. Unverified holes never become continuous evidence."""
    if not parts:
        return None
    info = deepcopy(parts[-1])
    by_time = {sample['time_sec']: sample for part in parts for sample in part['samples']}
    info['samples'] = [deepcopy(by_time[t]) for t in sorted(by_time)]
    info['verified_ranges'] = merged_intervals([r for part in parts for r in part['verified_ranges']])
    info['analyzed_seconds'] = max(b for a, b in info['verified_ranges'])
    info['verified_seconds'] = sum(b-a for a, b in info['verified_ranges'])
    info['duration_sec'] = duration
    info['sample_count'] = len(info['samples'])
    info['decoded_seconds'] = max(part['decoded_seconds'] for part in parts)
    for field in ('transition_times_sec', 'shot_times_sec', 'unfiltered_transition_times_sec', 'reference_background_events_ignored_sec'):
        info[field] = sorted({t for part in parts for t in part.get(field, [])})
    # Reset tracking at every separate evidence region, even for tiny gaps.
    boundaries = [a for a, _ in info['verified_ranges'][1:]]
    for field in ('transition_times_sec', 'shot_times_sec'):
        info[field] = sorted(set(info[field] + boundaries))
    info['segments'] = []
    for left, right in info['verified_ranges']:
        segments = _merge_segments([s for s in info['samples'] if left <= s['time_sec'] < right],
            config.get('min_segment_sec', .25), config.get('merge_gap_sec', .2), 1/info['sample_fps'])
        for segment in segments:
            segment.update(start_sec=max(left,segment['start_sec']), end_sec=min(right,segment['end_sec']))
            segment['duration_sec'] = segment['end_sec']-segment['start_sec']
            segment['id'] = f'vseg-{len(info["segments"]):04d}'
            info['segments'].append(segment)
    info['cache_identity'] = {**parts[0]['cache_identity'], 'verified_ranges':info['verified_ranges'],
                              'search_version':SEARCH_VERSION}
    return info


def merge_character_regions(info, parts):
    """Reuse region identities; original reference IDs stay shared across regions."""
    from autoedit.video.pool import shift_times
    from autoedit.video.references import pooled_reference_catalog
    samples = {}
    catalogs = []
    clusters, tracks = [], []
    for index, part in enumerate(parts):
        prefix = f'region-{index}'
        for sample in shift_times(part['samples'], 0, prefix):
            t = sample['time_sec']
            certified = any(a <= t < b for a, b in part['verified_ranges'])
            # Right/left context must not overwrite evidence from another region.
            if t not in samples or certified:
                samples[t] = sample
        cast = part.get('character_analysis') or {}
        if cast.get('reference'):
            catalogs.append(cast['reference'])
        clusters.extend({**cluster, 'id':f'{prefix}:{cluster["id"]}'} for cluster in cast.get('clusters', []))
        tracks.extend({**track, 'id':f'{prefix}:{track["id"]}'} for track in cast.get('tracks', []))
    info['samples'] = [samples[t] for t in sorted(samples)]
    cast = parts[-1].get('character_analysis') or {}
    info['character_analysis'] = {**cast, 'clusters':clusters, 'tracks':tracks,
                                  'selection_policy':'reference-first-verified-regions'}
    if catalogs:
        reference, owners = pooled_reference_catalog(catalogs)
        info['character_analysis']['reference'] = reference
        for sample in info['samples']:
            for face in [sample, *sample.get('faces', [])]:
                template = face.get('reference_template_id')
                if face.get('reference_match') and template in owners:
                    face['reference_id'] = owners[template]
    analyses = [part.get('object_analysis') or {} for part in parts]
    info['object_analysis'] = {**analyses[-1],
                              'inference_count':sum(a.get('inference_count', 0) for a in analyses),
                              'region_count':len(parts)}
    info['_regions_enriched'] = True
    return info


def discovery_chunk(path, start, end, fps, duration, config, cache_dir, session, progress):
    """One actor query; no pose, lip tracking or negative-prop inference here."""
    from autoedit.video.objects import _cached_object_scan, MODEL_DIR, RUNTIME, ACTOR_PROMPT, REFERENCE_PROMPT, normalize_objects, merge_subjects
    stat = Path(path).stat()
    identity = {'path':str(Path(path).resolve()), 'size':stat.st_size, 'mtime_ns':stat.st_mtime_ns,
                'version':ANALYSIS_VERSION}
    settings = character_settings(config.get('characters'))
    object_config = config.get('object_detection') or {}
    ready = (object_config.get('enabled', 'auto') is not False and RUNTIME.is_file()
             and (MODEL_DIR/'model.safetensors').is_file() and settings['enabled'])
    head_references = bool(settings['reference_images']) and (not settings['reference_regions'] or any(
        region.get('kind') == 'face' for regions in settings['reference_regions'].values() for region in regions))
    prompt = config.get('_reference_prompt') or (REFERENCE_PROMPT if settings.get('reference_images') else ACTOR_PROMPT)
    key_data = [identity, start, end, fps, config.get('source_kind', 'live_action'), head_references, prompt, SEARCH_VERSION]
    dest = Path(cache_dir) / ('search-' + hashlib.sha256(json.dumps(key_data).encode()).hexdigest() + '.json')
    if dest.is_file():
        info = json.loads(dest.read_text(encoding='utf-8'))
    else:
        detector = None
        if not ready or head_references:
            from autoedit.video.faces import MultiRegionFaceLandmarker
            detector = MultiRegionFaceLandmarker(include_animation=config.get('source_kind') == 'animation')
        info = {'path':str(path), 'cache_identity':identity, 'sample_fps':fps, 'fps':0,
                'duration_sec':duration, 'analyzed_seconds':end,
                'source_kind':config.get('source_kind', 'live_action'), 'samples':[],
                'transition_times_sec':[], 'shot_times_sec':[], 'segments':[]}
        frames = sampled_frames(path, fps, end, start_sec=start)
        try:
            for t, frame in frames:
                check_cancelled()
                det = (detector.detect(frame, t) if detector else
                       {'backend':'candidate-discovery', 'faces':[], 'category':'NEUTRAL', 'confidence':0})
                info['samples'].append({**det, 'time_sec':t, 'frame_width':frame.shape[1],
                                        'frame_height':frame.shape[0]})
        finally:
            frames.close()
            if detector:
                detector.close()
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.tmp')
        temp.write_text(json.dumps(info), encoding='utf-8')
        temp.replace(dest)
    if ready and info['samples']:
        # Imported targets are not rejected by object category; identity must qualify them.
        scan_settings = {'device':object_config.get('device', 'auto'), 'threshold':.3,
                         'prompt':prompt, 'version':3, 'batch_size':object_config.get('batch_size', 2),
                         'compute_type':object_config.get('compute_type', 'auto')}
        for field in ('compute_type', 'device_index'):
            if field in object_config:
                scan_settings[field] = object_config[field]
        data, _ = _cached_object_scan(info, cache_dir, scan_settings, progress, session)
        for sample, row in zip(info['samples'], data['samples']):
            objects = normalize_objects(row['objects'],sample['frame_width'],sample['frame_height'],.3,
                                        allow_reference_targets=bool(settings.get('reference_images')))
            sample['faces'] = merge_subjects(sample, objects)
    return info


def candidate_times(info, config, cache_dir, progress):
    """Loose screening can only request dense work, never qualify a final cut."""
    if 'screening_times' in info:
        return info['screening_times']
    from autoedit.video.characters import _embedding_tracks
    from autoedit.video.references import reference_vectors
    import numpy as np
    settings = character_settings(config.get('characters'))
    if not settings['reference_images']:
        return [s['time_sec'] for s in info['samples'] if s.get('faces')]
    if not any(s.get('faces') for s in info['samples']):
        return []
    kind = ('animation' if info['source_kind'] == 'animation' or any(
        f.get('subject_kind') == 'object' for s in info['samples'] for f in s['faces']) else 'live_action')
    if any(region.get('kind') == 'body' for regions in settings['reference_regions'].values() for region in regions):
        kind = 'animation'
    settings = {**settings, 'max_track_gap_sec':1.5/info['sample_fps'], 'embeddings_per_track':2}
    tracks, _, identity, _ = _embedding_tracks(info, cache_dir, settings, kind, progress)
    refs, _ = reference_vectors(kind, settings, cache_dir, identity)
    refs = np.asarray(refs, dtype=np.float32)
    threshold = settings['ccip_cosine_distance' if kind == 'animation' else 'arcface_cosine_distance']
    times = []
    for track in tracks:
        vectors = np.asarray(track.get('vectors', []), dtype=np.float32)
        face_vectors = np.asarray(track.get('reference_face_vectors', []), dtype=np.float32)
        candidates = [vectors]
        if face_vectors.size:
            candidates.append(face_vectors)
        if any(v.size and np.min(1-v @ refs.T) <= min(.5, threshold+.12) for v in candidates):
            times.extend(info['samples'][si]['time_sec'] for si, _ in track['observations'])
    return sorted(set(times))


def search_video(path, config, slots, cache_dir, session, on_chunk, progress=None, enrich_region=None):
    from autoedit.video.reference_search import BodyReferenceSearch, body_screening_enabled
    from autoedit.video.references import preflight_references
    preflight_references(character_settings(config.get('characters')), config.get('source_kind','live_action'))
    screening = BodyReferenceSearch(config, cache_dir) if body_screening_enabled(config) else None
    try:
        return _search_video(path, config, slots, cache_dir, session, on_chunk, progress, enrich_region, screening)
    finally:
        if screening:
            screening.close()


def _search_video(path, config, slots, cache_dir, session, on_chunk, progress=None, enrich_region=None, screening=None):
    import av
    settings = character_settings(config.get('characters'))
    object_config = config.get('object_detection') or {}
    from autoedit.video.objects import reference_search_prompt, MODEL_DIR, RUNTIME, REFERENCE_PROMPT
    if (not screening and settings['reference_images'] and object_config.get('enabled','auto') is not False
            and RUNTIME.is_file() and (MODEL_DIR/'model.safetensors').is_file()):
        config = {**config, '_reference_prompt':reference_search_prompt(settings,object_config,cache_dir,session,progress)}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration*stream.time_base) if stream.duration else float(container.duration)/1e6
        native_fps = float(stream.average_rate or 25)
    maximum = float(config.get('max_seconds') or 0)
    limit = duration if maximum <= 0 else min(duration,maximum)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Source video duration is missing or invalid')
    fps = float(config.get('search_fps', 1))
    if not math.isfinite(fps) or not 0 < fps <= 2:
        raise ValueError('video.search_fps must be between 0 and 2 (0 disables candidate search)')
    chunk = float(config.get('analysis_chunk_sec', 30))
    if not math.isfinite(chunk) or chunk <= 0:
        raise ValueError('video.analysis_chunk_sec must be a positive finite number')
    dense_fps = float(config.get('sample_fps') or 6)
    if not math.isfinite(dense_fps) or not 2 <= dense_fps <= 10:
        raise ValueError('Video sampling must be between 2 and 10 fps for temporal lip detection.')
    longest = max(float(s.get('required_duration_sec') or s.get('max_instance_duration_sec') or 0) for s in slots)
    if not math.isfinite(longest) or longest <= 0:
        raise ValueError('Invalid nested clip duration')
    radius = longest + 1
    parts, covered = [], []
    info = None
    requests = 0
    view_generation = None

    def verify(start, end, searched_until, pass_fps):
        nonlocal info, requests, view_generation
        for left, right in remaining_ranges(start, end, covered):
            check_cancelled()
            if screening:
                screening.close()
            if progress:
                progress(f'Verifying candidate {left:.1f}-{right:.1f}s at {config.get("sample_fps", 6):g} fps')
            raw = analyze_video(path, sample_fps=float(config.get('sample_fps') or 6),
                max_seconds=min(limit, right+1), start_seconds=max(0, left-1),
                min_segment_sec=float(config.get('min_segment_sec') or .25),
                merge_gap_sec=float(config.get('merge_gap_sec') or .2),
                backend=config.get('backend', 'auto'), source_kind=config.get('source_kind', 'live_action'),
                cache_dir=cache_dir, progress=progress,
                **({'identity_only':True} if screening and not config.get('require_lip_motion') else {}))
            # Context is measured but not allowed to supply a cut outside this range.
            raw['verified_ranges'] = [[left, right]]
            if config.get('_reference_prompt'):
                raw['_reference_prompt'] = config['_reference_prompt']
            if screening and not config.get('require_lip_motion'):
                raw['_body_reference_verification'] = True
            if enrich_region:
                raw = enrich_region(raw)
                if screening and raw.get('_reference_search_views'):
                    screening.use_confirmed_views(raw['_reference_search_views'])
                if raw.get('_body_reference_verification'):
                    # Cached dense features can gain a confirmed alternate view
                    # without another video decode or GPU invocation.
                    generation = raw.get('_reference_view_generation')
                    if generation != view_generation:
                        parts[:] = [enrich_region(part) if part.get('_body_reference_verification') else part for part in parts]
                        view_generation = generation
            parts.append(raw)
            covered[:] = merged_intervals(covered+[[left, right]])
            requests += 1
            info = merge_evidence(parts, duration, config)
            if enrich_region:
                info = merge_character_regions(info, parts)
            info.update(fps=native_fps, search_fps=fps, searched_until_sec=searched_until,
                        search_pass_fps=pass_fps, dense_region_count=requests,
                        analysis_complete=False)
            if on_chunk(info):
                return True
        return False

    for pass_fps in sorted(set([fps, 2.0])):
        pending_candidates = {}
        if pass_fps > fps and config.get('_reference_prompt'):
            config = {**config, '_reference_prompt':REFERENCE_PROMPT}
        start = 0.
        while start < limit:
            check_cancelled()
            end = min(limit, start+chunk)
            if contains_range(covered, start, end):
                start = end
                continue
            if progress:
                progress(f'Finding characters {start:.1f}-{end:.1f}/{limit:.1f}s at {pass_fps:g} fps')
            release_inference({})
            discover = screening.discover if screening else discovery_chunk
            candidates = discover(path,start,end,pass_fps,duration,config,cache_dir,session,progress)
            session.offload()
            times = candidate_times(candidates,config,cache_dir,progress)
            release_inference({})
            if screening:
                from autoedit.video.reference_search import clip_windows, ranked_clip_windows
                if 'screening_candidates' in candidates:
                    ceiling = min(.45, settings['body_reference_cosine_distance'] + (.16 if pass_fps > fps else .03))
                    pending_candidates.update({row['time_sec']:row for row in candidates['screening_candidates']})
                    windows = ranked_clip_windows(list(pending_candidates.values()), longest,
                        float(config.get('source_gap_sec', 5)), limit, ceiling,
                        sample_fps=pass_fps if pass_fps == fps else None)
                    windows = [(left,right) for left,right in windows if not any(
                        max(left,a) < min(right,b) for a,b in covered)]
                    # Gather enough possible cuts before expensive verification;
                    # strongest candidates can replace earlier isolated/weak hits.
                    # Capacity never qualifies a cut or ends the source search.
                    if end < limit and len(windows) < len(slots)+2:
                        start = end
                        continue
                else:
                    windows = clip_windows(times, longest, float(config.get('source_gap_sec', 5)), limit)
            else:
                windows = merged_intervals([[max(0,t-radius), min(limit,t+radius)] for t in times])
            for left, right in windows:
                if verify(left,right,end,pass_fps):
                    return info
            start = end
    # Sparse misses must not enable a wrong-character fallback. Exhaust remaining
    # dense regions before declaring the selected source searched completely.
    start = 0.
    while start < limit:
        end = min(limit, start+chunk)
        if verify(start,end,limit,float(config.get('sample_fps') or 6)):
            return info
        start = end
    if screening and enrich_region and info.get('_body_reference_verification'):
        # Exposure fallback must retain an independently detected whole cast.
        original = deepcopy(info)
        original.pop('_body_reference_verification',None)
        original.pop('_regions_enriched',None)
        original.pop('character_analysis',None)
        original['transition_times_sec'] = original.pop('unfiltered_transition_times_sec',original.get('transition_times_sec',[]))
        for sample in original['samples']:
            sample['faces'] = sample.pop('_raw_faces',sample.get('faces',[]))
            for field in ('reference_match','reference_id','reference_template_id','reference_body_measured'):
                sample.pop(field,None)
        info = enrich_region(original)
        info['searched_until_sec'] = limit
        if on_chunk(info):
            return info
    info['analysis_complete'] = True
    info['searched_until_sec'] = limit
    return info
