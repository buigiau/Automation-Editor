"""AI branch: inspect first, select once (+ one repair), validate, export autofill."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
from time import monotonic
import uuid

from autoedit.config import source_paths
from autoedit.runtime import check_cancelled
from autoedit.premiere.prproj import inspect_prproj
from autoedit.cubase.inspect import inspect_sampler_tracks, select_sampler_tracks
from autoedit.validate import require_inputs
from .cache import digest, identity, read, write, voice_catalog
from .gemini import GeminiClient, GeminiError
from .selection import PROMPT, PROMPT_VERSION, SCHEMA, validate_selection
from .visual import probe, check_visual_duplicates
from .sampling import sampling_settings, annotate_sources, video_part
from .ollama import OllamaClient, LOCAL_URL, resource_settings
from .native import NativeVisionClient
from .local import LocalSelector, local_settings
from .completion import complete_selection, shortage_policy, VERSION as COMPLETION_VERSION
from .ollama import LocalAIError


class SelectionIncomplete(ValueError):
    pass


def preflight(config):
    shortage_policy(config.get('ai', {}))
    sampling_settings(config.get('ai', {}))
    backend = config.get('ai', {}).get('backend', 'gemini')
    if backend not in ('gemini', 'local'):
        raise ValueError('ai.backend must be gemini or local')
    if backend == 'local':
        local_settings(config.get('ai', {}))
        resource_settings(config.get('ai', {}))
    prem, cub, audio = (config[k] for k in ('premiere', 'cubase', 'audio'))
    require_inputs(premiere_project=prem['project'], cubase_project=cub['project'],
        source_video=prem['source_media'], audio_directory=audio['directory'],
        template_sequence=prem['template_sequence'], video_track_index=int(prem['video_track_index']),
        sampler_tracks=cub.get('sampler_tracks'))
    inspect = inspect_prproj(prem['project'], template_sequence=prem['template_sequence'],
        video_track_index=int(prem['video_track_index']))
    scenes = inspect['scene_slots']
    inventory = select_sampler_tracks(inspect_sampler_tracks(cub['project']), cub.get('sampler_tracks'),
                                      len(scenes), scenes=scenes, infer_duplicates=True)
    if not inspect['slots'] or not scenes or len(scenes) != len(inventory['tracks']):
        raise ValueError(f"Template mismatch: {len(scenes)} scene runs, {len(inventory['tracks'])} sampler tracks")
    if (inventory.get('mapping_mode') != 'duplicate_scene_tracks'
            and len({t['index'] for t in inventory['tracks']}) != len(inventory['tracks'])):
        raise ValueError('Sampler track indices are not unique')
    slots = deepcopy(inspect['slots'])
    intro = inspect.get('intro_slot')
    if intro and intro.get('status') == 'empty':
        slots.append(deepcopy(intro))
    for slot in slots:
        duration = slot.get('required_duration_sec') or slot.get('max_instance_duration_sec')
        if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid template duration: {slot['id']}")
        slot['required_duration_sec'] = duration
    if config.get('ai', {}).get('final_scene_transition_fallback'):
        final_scene = max(scenes, key=lambda s: s['last_end_sec'])
        for slot in slots:
            if (slot.get('nested_sequence_uid') == final_scene['nested_sequence_uid']
                    and slot['first_start_sec'] >= final_scene['first_start_sec']-.001):
                # A reused nest would also relax an earlier timeline block.
                slot['allow_scene_changes'] = True
    durations = {s['nested_sequence_uid']: s['required_duration_sec'] for s in slots if s.get('nested_sequence_uid')}
    for scene in scenes:
        for item in scene['instances']:
            start = float(item.get('source_in_sec') or 0)
            length = float(item['end_sec']) - float(item['start_sec'])
            end = float(item.get('source_out_sec', start+length))
            if (not all(math.isfinite(v) for v in (start, end, length)) or start < 0 or length <= 0
                    or abs(end-start-length) > .002 or end > durations[scene['nested_sequence_uid']] + .002):
                raise ValueError(f"Unsupported retimed/invalid scene offsets: {scene['id']}")
    for slot in slots:
        used = [float(item.get('source_out_sec', float(item.get('source_in_sec') or 0)+
                    float(item['end_sec'])-float(item['start_sec'])))
                for scene in scenes if scene['nested_sequence_uid'] == slot.get('nested_sequence_uid')
                for item in scene['instances']]
        if used:
            # A stale clip at the end of a nest can require a long overwrite
            # even when the parent timeline plays only its opening frames.
            slot['visible_duration_sec'] = min(slot['required_duration_sec'], max(used))
    gap = float(config.get('video', {}).get('source_gap_sec', 5))
    if not math.isfinite(gap) or gap < 0:
        raise ValueError('Source gap must be finite and non-negative')
    timeout = float(config.get('ai', {}).get('request_timeout_sec', 300))
    if not math.isfinite(timeout) or not 1 <= timeout <= 1800:
        raise ValueError('AI request timeout must be between 1 and 1800 seconds')
    for path in config.get('video', {}).get('characters', {}).get('reference_images', []):
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
    return inspect, inventory, slots, gap


def invalidate_bundle(out):
    """An incomplete new run must never expose a previous run as its result."""
    names = ('edit-plan.json', 'premiere_apply.json', 'cubase_import.json', 'ai_review.html', 'ai-review.json', 'ai-validation.json', 'ai-result.json')
    existing = [out / name for name in names if (out / name).exists()]
    if existing:
        archive = out / 'previous_runs' / uuid.uuid4().hex
        archive.mkdir(parents=True)
        for path in existing:
            path.replace(archive / path.name)


def run_ai_pipeline(config, log=None):
    resources = []
    try:
        return _run_ai_pipeline(config, log, resources)
    finally:
        for client in resources:
            getattr(client, 'close', client.unload)()


def _run_ai_pipeline(config, log=None, resources=None):
    job_started = monotonic()
    log = log or (lambda message: None)
    if config.get('_ai_review'):
        return apply_review(config, log)
    timings = config.setdefault('_timings', {})
    started = time.perf_counter()
    log('AI 1/5: checking paired template and sampler tracks…')
    inspect, inventory, slots, gap = preflight(config)
    for slot in slots:
        if slot.get('allow_scene_changes'):
            log(f"Final timeline block {slot['id']}: scene changes allowed throughout {slot['required_duration_sec']:g}s; opening character evidence only.")
        elif slot.get('visible_duration_sec', slot['required_duration_sec']) < slot['required_duration_sec']-.002:
            log(f"{slot['id']}: fill {slot['required_duration_sec']:g}s; parent timeline uses only the first {slot['visible_duration_sec']:g}s. Auditing visible footage only.")
    timings['template_validation'] = round(time.perf_counter() - started, 3)
    ai = config.get('ai', {})
    policy = shortage_policy(ai)
    sampling = sampling_settings(ai)
    from autoedit.presets import ROOT
    cache = Path(ai.get('cache_dir') or ROOT / 'output/.ai-cache').resolve()
    out = Path(config['job']['output_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    invalidate_bundle(out)
    started = time.perf_counter()
    voices = voice_catalog(config['audio'], cache)
    timings['voice_library'] = round(time.perf_counter() - started, 3)
    sources = [{'id': f'source-{i+1}', 'path': str(Path(p).resolve()), 'identity': identity(p), **probe(p)}
               for i, p in enumerate(source_paths(config['premiere']['source_media']))]
    sources = annotate_sources(sources, sampling)
    references = [{'id': f'reference-{i+1}', 'path': str(Path(p).resolve()), 'identity': identity(p)}
        for i, p in enumerate(config.get('video', {}).get('characters', {}).get('reference_images', []))]
    backend = ai.get('backend', 'gemini')
    if backend not in ('gemini', 'local'):
        raise ValueError('ai.backend must be gemini or local')
    model = ai.get('local_model', 'qwen3-vl:2b-instruct') if backend == 'local' else ai.get('model', 'gemini-3.8-flash')
    local_client = None
    startup_error = None
    if backend == 'local':
        options = local_settings(ai)
        client_type = NativeVisionClient if options['mode'] == 'until_filled' else OllamaClient
        local_client = client_type(model, cache, log, url=ai.get('local_url', LOCAL_URL),
            timeout=float(ai.get('request_timeout_sec', 300)), **resource_settings(ai))
        resources.append(local_client)
        if options['mode'] == 'until_filled':
            local_client.target_deadline = job_started + options['target_seconds']
            log(f"Local search target: {options['target_seconds']:g}s; " +
                (f"hard maximum: {options['max_seconds']:g}s (30s reserved for validation/export). "
                 "RAM waits count toward this maximum." if options['max_seconds'] else
                 "no hard maximum; continue until filled or source exhausted. Cancel remains available."))
            # A desired completion time is not permission to discard available
            # footage. Only an explicit maximum imposes a hard search deadline.
            if options['max_seconds']:
                local_client.job_deadline = job_started + max(1., options['max_seconds']-30.)
                # Keep inference bounded, but let the last completed observation
                # finish deterministic allocation before using the export reserve.
                local_client.finalization_deadline = min(local_client.job_deadline+10.,
                    job_started+max(1., options['max_seconds']-20.))
        try:
            backend_identity = local_client.identity()
        except (LocalAIError, TimeoutError) as exc:
            if policy == 'strict':
                raise
            startup_error = type(exc).__name__
            backend_identity = {'model': model, 'unavailable': True, 'error_type': startup_error}
            log('Local model unavailable; exporting a draft from decoded source footage.')
        log('Local visual analysis: CPU person/face screening when available, Qwen fallback; all Voice matches require audio review.')
    config['_ai_backend_identity'] = backend_identity if local_client else {'model': model}
    contract = {'prompt_version': PROMPT_VERSION, 'backend': backend, 'model': model, 'preset': ai.get('preset', 'custom'),
        'shortage_policy': policy, 'completion_version': COMPLETION_VERSION,
        'video_sampling': sampling,
        'sources': sources, 'references': references, 'source_gap_sec': gap,
        'slots': [{k: s[k] for k in ('id', 'required_duration_sec', 'visible_duration_sec', 'allow_scene_changes') if k in s} for s in slots],
        'template': {'premiere': identity(config['premiere']['project']), 'cubase': identity(config['cubase']['project']),
                     'sequence': inspect['template_sequence'], 'video_track_index': config['premiere']['video_track_index'],
                     'sampler_tracks': config['cubase'].get('sampler_tracks')},
        'voices': [{k: v.get(k) for k in ('voice_id', 'name', 'duration_sec', 'category', 'phonetics', 'identity')} for v in voices]}
    if local_client is not None:
        contract.update(backend_identity=backend_identity, local_analysis=local_settings(ai))
    key = digest(contract)
    record = cache / 'results' / (key + '.json')
    result = read(record)
    cache_hit = result is not None
    config['_ai_cache_hit'] = cache_hit
    timings.update(upload=0., ai=0.)
    parts, client = None, None
    local_selector = LocalSelector(local_client, cache, log) if local_client else None
    def request_selection(previous=None, errors=None):
        nonlocal parts, client
        if startup_error:
            raise LocalAIError('Local model unavailable')
        if local_selector is not None:
            t = time.perf_counter()
            try:
                return local_selector.select(contract, previous, errors, allow_same_shot=allow_same_shot)
            finally:
                timings['ai'] = round(timings.get('ai', 0) + time.perf_counter() - t, 3)
        if client is None:
            client = GeminiClient(ai.get('api_key') or os.environ.get('GEMINI_API_KEY', ''), model, cache, log)
            client.request_timeout_sec = float(ai.get('request_timeout_sec', 300))
            upload_started = time.perf_counter()
            # Use original videos with sound; silent assets are for Premiere only.
            parts = [{'text': PROMPT + '\nREQUIREMENTS:\n' + json.dumps(contract, ensure_ascii=False)}]
            for item in sources + references:
                uploaded = client.upload(item['path'], item['identity'])
                if item in sources:
                    uploaded = video_part(uploaded, item, sampling)
                parts += [{'text': item['id']}, uploaded]
            timings['upload'] = round(time.perf_counter() - upload_started, 3)
        request = parts
        if previous is not None:
            fallback_instruction = (' SHORTAGE FALLBACK AUTHORIZED: prioritize unused shots, but if still short, '
                'take different non-overlapping segments from the same continuous shot. The shot interval and ID '
                'may repeat; the actual selected scene/footage must not repeat. Preserve source_gap_sec. '
                'A duplicate_group may describe repeated long footage; only different selected visual content '
                'can pass local frame checks. Never repeat an already selected segment to fill the count.'
                if allow_same_shot else '')
            request = parts + [{'text': 'Repair this result ONCE. Replace rejected shots; retain valid choices where possible. '
                'Return the complete JSON. If impossible, keep valid scenes and explain shortage.\n'
                + fallback_instruction + '\n'
                + json.dumps({'previous': previous, 'validation_errors': errors}, ensure_ascii=False)}]
        t = time.perf_counter()
        try:
            return client.generate(request, SCHEMA)
        finally:
            timings['ai'] = round(timings.get('ai', 0) + time.perf_counter() - t, 3)
    def ask(previous=None, errors=None):
        if previous and previous.get('analysis_scope', {}).get('stop_reason') == 'analysis_unavailable':
            return previous
        try:
            return request_selection(previous, errors)
        except (LocalAIError, GeminiError, TimeoutError) as exc:
            if policy == 'strict':
                raise
            check_cancelled()
            log(f'AI selection unavailable ({type(exc).__name__}); completing a reviewable draft from local footage.')
            progressive = getattr(local_selector, 'progressive', None)
            choices = [r for r in (previous, getattr(progressive, 'best', None)) if isinstance(r, dict)]
            partial = deepcopy(max(choices, key=lambda r: len(r.get('selections', []))) if choices else
                               {'characters': [], 'selections': []})
            partial.setdefault('analysis_scope', {}).update(stop_reason='analysis_unavailable',
                error_type=type(exc).__name__)
            return partial
    log(f'AI 2/5: {len(slots)} unique fills, {len(inspect["scene_slots"])} scene runs; '
        f"{sampling['video_fps']:g} FPS, evidence={sum(s['analyzed_end_sec'] for s in sources):.1f}s; "
        + ('cached selection' if cache_hit else 'selecting footage'))
    allow_same_shot = bool(cache_hit and sampling['shot_fallback'] and isinstance(result, dict)
                          and result.get('_autoedit_dedup_policy') == 'distinct_segments')
    if result is None:
        result = ask()
    if (local_selector and sampling['shot_fallback'] and isinstance(result, dict)
            and result.get('_autoedit_dedup_policy') == 'distinct_segments'):
        allow_same_shot = True
    best_rows, best_result = [], None
    for attempt in range(2):
        check_cancelled()
        if not isinstance(result, dict):
            result = {'invalid_response': True, 'selections': [], 'characters': []}
        result['_autoedit_dedup_policy'] = 'distinct_segments' if allow_same_shot else 'unique_shots'
        log(f'AI 3/5: validating timestamps, shots and local frames (pass {attempt+1}/2)…')
        rows, errors, ranking = validate_selection(result, slots, sources, voices, [r['id'] for r in references], gap,
            allow_same_shot=allow_same_shot, transition_tail_fallback_sec=sampling['transition_tail_fallback_sec'])
        started = time.perf_counter()
        rows, visual_errors = check_visual_duplicates(rows, sources, cache)
        timings['selected_frame_checks'] = round(timings.get('selected_frame_checks', 0) + time.perf_counter() - started, 3)
        errors += visual_errors
        if len(rows) > len(best_rows):
            best_rows, best_result = deepcopy(rows), deepcopy(result)
        write(out / 'ai-result.json', result)
        write(out / 'ai-validation.json', {'complete': not errors, 'errors': errors, 'accepted': rows,
              'video_sampling':sampling,'deduplication_policy':result['_autoedit_dedup_policy'],
              'analysis_scope': result.get('analysis_scope', {}),
              'character_ranking': ranking, 'cache_hit': cache_hit, 'repair_count': attempt})
        if not errors or attempt == 1:
            break
        log('Requesting one repair for invalid/missing scenes…')
        # Persist a reviewable partial result before a repair API call can fail.
        save_review_snapshot(config, out, cache, inspect, inventory, slots, sources, voices, references, gap, result, rows, errors, key)
        if sampling['shot_fallback'] and len(rows) < len(slots):
            allow_same_shot = True
            log('Shortage fallback: permit distinct segments within the same shot; keep overlap, source gap and frame-duplicate checks.')
        result = ask(result, errors)
    snapshot = save_review_snapshot(config, out, cache, inspect, inventory, slots, sources, voices, references, gap, result, rows, errors, key)
    if errors and policy == 'complete':
        selection_errors = deepcopy(errors)
        if len(best_rows) > len(rows):
            best_result['analysis_scope'] = deepcopy(result.get('analysis_scope', {}))
            rows, result = best_rows, best_result
        started = time.perf_counter()
        result = complete_selection(result, rows, slots, sources, voices, references, gap, cache, log)
        result['selection_errors'] = selection_errors
        rows, errors, ranking = validate_selection(result, slots, sources, voices, [r['id'] for r in references], gap,
            allow_same_shot=allow_same_shot, transition_tail_fallback_sec=sampling['transition_tail_fallback_sec'],
            allow_completion=True)
        rows, visual_errors = check_visual_duplicates(rows, sources, cache)
        errors += visual_errors
        timings['draft_completion'] = round(time.perf_counter()-started, 3)
        write(out / 'ai-result.json', result)
        write(out / 'ai-validation.json', {'complete': not errors, 'selection_complete': False,
            'errors': errors, 'selection_errors': selection_errors, 'accepted': rows,
            'completion': result['completion'], 'video_sampling': sampling,
            'analysis_scope': result.get('analysis_scope', {}), 'character_ranking': ranking,
            'deduplication_policy': result['_autoedit_dedup_policy'], 'cache_hit': cache_hit, 'repair_count': attempt})
        snapshot = save_review_snapshot(config, out, cache, inspect, inventory, slots, sources, voices, references, gap, result, rows, errors, key)
    if errors:
        scope = result.get('analysis_scope', {})
        explanation = (f" Search stopped: {scope['stop_reason']}; "
            f"{scope.get('candidate_count', 0)} candidates checked."
            if scope.get('stop_reason') else '')
        if scope.get('stop_reason') == 'time_budget':
            maximum = scope.get('hard_max_seconds') or ai.get('local_max_seconds', 0)
            explanation += (f" Hard local time limit: {maximum:g}s; 30s reserved for validation/export."
                            if maximum else " The local search deadline was reached.")
            explanation += (" This does not establish that the source has too few usable scenes. "
                            "To continue searching, increase Local hard maximum or set local_max_seconds=0 "
                            "in Settings, then run again. Completed candidate observations remain cached.")
        raise SelectionIncomplete(f"Only {len(rows)}/{len(slots)} valid distinct shots. See ai_review.html / ai-validation.json. "
                                  + '; '.join(f"{e['slot_id']}: {e['error']}" for e in errors[:5]) + explanation)
    # A draft is not cached as an AI-verified selection. Future runs may improve it.
    if not result.get('completion'):
        write(record, result)
    return export_bundle(snapshot, out, rows, ranking, log, timings)


def save_review_snapshot(config, out, cache, inspect, inventory, slots, sources, voices, references, gap, result, rows, errors, key):
    from .review import write_review
    # Only fields required by artifact generation are retained. No API credentials.
    clean = {k: deepcopy(config[k]) for k in ('premiere', 'cubase', 'job')}
    clean['premiere']['template_sequence'] = inspect['template_sequence']
    snapshot = {'config': clean, 'cache': str(cache), 'inspect': inspect, 'inventory': inventory,
        'backend': config.get('ai', {}).get('backend', 'gemini'),
        'backend_identity': deepcopy(config.get('_ai_backend_identity', {})),
        'slots': slots, 'sources': sources, 'voices': voices, 'references': references,
        'gap': gap, 'result': result, 'accepted': rows, 'errors': errors, 'revision': key,
        'video_sampling': sampling_settings(config.get('ai', {})),
        'deduplication_policy': result.get('_autoedit_dedup_policy', 'unique_shots'),
        'shortage_policy': shortage_policy(config.get('ai', {})),
        'template_identity': [identity(config['premiere']['project']), identity(config['cubase']['project'])]}
    write(out / 'ai-review.json', snapshot)
    write_review(snapshot, out)
    return snapshot


def export_bundle(snapshot, out, rows, ranking, log, timings=None):
    from autoedit.plan.generator import build_edit_plan
    from autoedit.premiere.adapter import write_apply_payload
    from autoedit.schema import save_plan, validate_plan
    from autoedit.cubase.sampler import render_sampler_bundle
    from autoedit.video.silent_source import prepare_video_only_source
    from .review import write_review
    cfg, inspect, inventory = (snapshot[k] for k in ('config', 'inspect', 'inventory'))
    backend = snapshot.get('backend', 'gemini')
    method = 'qwen-local-visual' if backend == 'local' else 'gemini'
    sources = {s['id']: s for s in snapshot['sources']}
    voices = {v['voice_id']: v for v in snapshot['voices']}
    selected = {r['slot_id']: r for r in rows}
    timings = timings if timings is not None else {}
    started = time.perf_counter()
    log('AI 4/5: preparing cached video-only assets and sampler WAVs…')
    silent = {sid: prepare_video_only_source(sources[sid]['path'],
              Path(snapshot['cache']) / 'video_only' / sources[sid]['identity']['sha256'],
              progress=log)
              for sid in dict.fromkeys(r['source_id'] for r in rows)}
    timings['video_only_copy'] = round(time.perf_counter() - started, 3)
    def video(row):
        source = sources[row['source_id']]
        return {**row, 'source_video': source['path'], 'source_duration_sec': source['duration_sec'],
            'video_only_source': silent[row['source_id']],
            'character_selection': {'method': row.get('visual_method', method), 'character_id': row['character_id'],
                'character_ids': [row['character_id']], 'decision': 'uncertain',
                'reason': row['reason'],
                'reference_fallback': row['reference_fallback'], 'needs_review': True}}
    matches = [{'audio': deepcopy(voices[selected[s['id']]['voice_id']]), 'video': video(selected[s['id']]),
                'status': 'ai_selected', 'score': None} for s in inspect['slots']]
    plan = build_edit_plan(config=cfg, slots=inspect['slots'], matches=matches,
        video_info={'path': snapshot['sources'][0]['path']}, notes=['AI scene/Voice estimates; review playback before final mix.'])
    for slot in plan['slots']:
        slot['source_video'] = slot['video']['source_video']
    plan['project'].update(import_mixdown=False, source_videos=[{**s, 'video_only_source': silent.get(s['id'])} for s in snapshot['sources']],
                           video_only_source=next(iter(silent.values())))
    plan['template_audio_tracks'] = inspect.get('template_audio_tracks', [])
    if 'nested_structure' in inspect:
        plan['nested_structure'] = inspect['nested_structure']
    intro = inspect.get('intro_slot')
    if intro and intro.get('status') == 'empty':
        plan['intro_fill'] = {**video(selected[intro['id']]), 'slot_id': intro['id'], 'mode': 'intro_gap',
            'overwrite_at_sec': intro['start_sec'], 'slot_duration_sec': intro['required_duration_sec'],
            'end_sec': intro['end_sec'], 'video_track_index': intro['video_track_index'], 'audio_track_index': -1}
    parents = {s['premiere']['nested_sequence_uid']: s for s in plan['slots']}
    sampler_slots = []
    for scene, track in zip(inspect['scene_slots'], inventory['tracks']):
        parent = parents[scene['nested_sequence_uid']]
        row = selected[parent['id']]
        voice = deepcopy(voices[row['voice_id']])
        voice.update(segment_start_sec=voice.get('active_start_sec', 0),
                     segment_end_sec=voice.get('active_end_sec', voice['duration_sec']), placement_offset_sec=0.)
        sampler_slots.append({'id': scene['id'], 'index': scene['index'], 'audio': voice,
            'source_video': parent['source_video'], 'video': deepcopy(parent['video']),
            'audio_match': {'method': row.get('visual_method', method) + '-voice', 'needs_review': True,
                'audio_uncertain': row['audio_uncertain'], 'reference_fallback': row['reference_fallback'],
                'reason': row['reason'], 'phoneme_sync_verified': False, 'cubase_timing_verified': False},
            'premiere': {'nested_sequence': scene['nested_sequence'], 'nested_sequence_uid': scene['nested_sequence_uid'],
                'mode': 'scene_run', 'timeline_start_sec': scene['first_start_sec'], 'timeline_end_sec': scene['last_end_sec'],
                'duration_sec': parent['premiere']['duration_sec'], 'instances': scene['instances']},
            'cubase': {'track_index': track['index'], 'track_name': track['name'], 'project': inventory['path'],
                'start_sec': scene['first_start_sec'], 'preserve_existing_midi': True,
                **({'duplicate_of': track['duplicate_of']} if track.get('is_duplicate') else {})},
            'match_status': 'sample_selected'})
    plan.update(audio_mode='sampler_per_scene_v1', cubase_slots=sampler_slots, review_status='manual_review_required',
        ai_selection={'prompt_version': PROMPT_VERSION, 'backend': backend, 'revision': snapshot['revision'],
                      'backend_identity': snapshot.get('backend_identity', {}),
                      'analysis_scope': snapshot['result'].get('analysis_scope'),
                      'video_sampling': snapshot.get('video_sampling'),
                      'deduplication_policy': snapshot.get('deduplication_policy', 'unique_shots'),
                      'character_ranking': ranking, 'appearance_ranking_is_estimate': True})
    if snapshot['result'].get('completion'):
        plan['ai_selection']['completion'] = deepcopy(snapshot['result']['completion'])
        plan['ai_selection']['selection_errors'] = deepcopy(snapshot['result'].get('selection_errors', []))
        plan['notes'].append('Complete draft includes unverified fallback footage; see per-cut completion flags in ai_review.html.')
    if snapshot.get('review_provenance'):
        plan['ai_selection']['review_provenance'] = deepcopy(snapshot['review_provenance'])
    plan['audio'] = [s['audio'] for s in sampler_slots]
    plan['video_analysis_meta'].update(quality_method=method + '-with-local-duplicate-check',
        needs_review=True, character_ranking_scope=snapshot['result'].get('analysis_scope', {}).get(
            'character_ranking_scope', 'supplied_sources'), cpu_threads=2)
    check_cancelled()
    started = time.perf_counter()
    result = render_sampler_bundle(plan, out)
    timings['sampler_artifacts'] = round(time.perf_counter() - started, 3)
    plan['cubase_manifest'] = result['manifest']
    errors = validate_plan(plan)
    if errors:
        raise ValueError('; '.join(errors))
    check_cancelled()
    started = time.perf_counter()
    write_review(snapshot, out, plan)
    timings['review_page'] = round(time.perf_counter() - started, 3)
    plan['ai_review'] = str(out / 'ai_review.html')
    write_apply_payload(plan, out / 'premiere_apply.json')
    path = save_plan(plan, out / 'edit-plan.json')
    completion = snapshot['result'].get('completion', {})
    if completion:
        log(f"AI 5/5: complete draft ready; {completion['fallback_count']} fallback fill(s) need review. Open ai_review.html; then Apply Premiere / Import Cubase.")
    else:
        log('AI 5/5: autofill ready. Open ai_review.html; then Apply Premiere / Import Cubase.')
    return {'plan_path': str(path), 'plan': plan}


def apply_review(config, log):
    """Revalidate every slot and regenerate artifacts without another API request."""
    out = Path(config['job']['output_dir']).resolve()
    snapshot = read(out / 'ai-review.json')
    edits = read(config['_ai_review'])
    if not snapshot or not edits or edits.get('revision') != snapshot['revision']:
        raise ValueError('Corrections belong to a different review revision')
    for item in snapshot['sources'] + snapshot['voices'] + snapshot['references']:
        if identity(item['path']) != item['identity']:
            raise ValueError('Source, Voice or reference changed; run AI selection again')
    for stamp in snapshot['template_identity']:
        if identity(stamp['path']) != stamp:
            raise ValueError('Template changed; run AI selection again')
    result = deepcopy(snapshot['result'])
    by_id = {r['slot_id']: r for r in result['selections']}
    durations = {s['id']: s['required_duration_sec'] for s in snapshot['slots']}
    visible_durations = {s['id']: s.get('visible_duration_sec', s['required_duration_sec']) for s in snapshot['slots']}
    seen = set()
    for edit in edits.get('edits', []):
        sid = edit['slot_id']
        if sid not in by_id or sid in seen:
            raise ValueError('Unknown or repeated correction slot')
        seen.add(sid)
        row = by_id[sid]
        if 'in_sec' in edit:
            start = float(edit['in_sec'])
            row.update(in_sec=start, out_sec=start + durations[sid])
            if row.get('completion_fallback'):
                row.update(shot_start_sec=start, shot_end_sec=start+durations[sid])
            if row.get('timeline_visible_prefix'):
                row['identity_end_sec'] = start+visible_durations[sid]
        if 'voice_id' in edit:
            row['voice_id'] = edit['voice_id']
        row['audio_uncertain'] = True
    rows, errors, ranking = validate_selection(result, snapshot['slots'], snapshot['sources'], snapshot['voices'],
        [r['id'] for r in snapshot['references']], snapshot['gap'],
        allow_same_shot=snapshot.get('deduplication_policy') == 'distinct_segments',
        transition_tail_fallback_sec=snapshot.get('video_sampling', {}).get('transition_tail_fallback_sec', 0),
        allow_completion=snapshot.get('shortage_policy') == 'complete')
    rows, visual_errors = check_visual_duplicates(rows, snapshot['sources'], snapshot['cache'])
    errors += visual_errors
    if errors:
        write(out / 'ai-correction-errors.json', errors)
        raise ValueError('Corrections rejected; existing bundle preserved. ' + '; '.join(f"{e['slot_id']}: {e['error']}" for e in errors))
    snapshot.update(result=result, accepted=rows, errors=[], revision=digest([snapshot['revision'], edits]))
    invalidate_bundle(out)
    write(out / 'ai-review.json', snapshot)
    write(out / 'ai-result.json', result)
    write(out / 'ai-validation.json', {'complete': True, 'selection_complete': not bool(result.get('completion')),
        'errors': [], 'accepted': rows, 'character_ranking': ranking, 'completion': result.get('completion'),
        'selection_errors': result.get('selection_errors', [])})
    return export_bundle(snapshot, out, rows, ranking, log, config.get('_timings'))


def recover_draft(review_path, output_dir, log=print):
    """Recover a failed job's saved observations without rerunning AI inference."""
    started = time.perf_counter()
    snapshot = read(review_path)
    if not snapshot:
        raise ValueError('Missing AI review snapshot')
    for item in snapshot['sources'] + snapshot['voices'] + snapshot['references']:
        if identity(item['path']) != item['identity']:
            raise ValueError('Source, Voice or reference changed; run selection again')
    for stamp in snapshot['template_identity']:
        if identity(stamp['path']) != stamp:
            raise ValueError('Template changed; run selection again')
    sources, slots, voices = (snapshot[k] for k in ('sources', 'slots', 'voices'))
    refs = [r['id'] for r in snapshot['references']]
    options = dict(allow_same_shot=snapshot.get('deduplication_policy') == 'distinct_segments',
                   transition_tail_fallback_sec=snapshot.get('video_sampling', {}).get('transition_tail_fallback_sec', 0))
    rows, errors, _ = validate_selection(snapshot['result'], slots, sources, voices, refs, snapshot['gap'], **options,
        allow_completion=snapshot.get('shortage_policy') == 'complete')
    rows, visual_errors = check_visual_duplicates(rows, sources, snapshot['cache'])
    errors += visual_errors
    result = complete_selection(snapshot['result'], rows, slots, sources, voices, snapshot['references'],
                                snapshot['gap'], snapshot['cache'], log)
    result['selection_errors'] = errors
    rows, errors, ranking = validate_selection(result, slots, sources, voices, refs, snapshot['gap'], **options, allow_completion=True)
    rows, visual_errors = check_visual_duplicates(rows, sources, snapshot['cache'])
    errors += visual_errors
    if errors:
        raise ValueError('Recovered draft validation failed: '+str(errors))
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    snapshot['config']['job']['output_dir'] = str(out)
    snapshot.update(result=result, accepted=rows, errors=[], shortage_policy='complete',
        revision=digest([snapshot['revision'], result]),
        review_provenance={'draft_recovery': True, 'original_review': str(Path(review_path).resolve())})
    invalidate_bundle(out)
    write(out/'ai-review.json', snapshot)
    write(out/'ai-result.json', result)
    write(out/'ai-validation.json', {'complete': True, 'selection_complete': not bool(result['completion']['fallback_count']),
        'errors': [], 'selection_errors': result['selection_errors'], 'accepted': rows,
        'completion': result['completion'], 'analysis_scope': result.get('analysis_scope', {}), 'character_ranking': ranking})
    result_bundle = export_bundle(snapshot, out, rows, ranking, log)
    previous_performance = read(Path(review_path).parent/'performance.json')
    write(out/'draft-recovery.json', {'status': 'completed', 'total_sec': round(time.perf_counter()-started, 3),
        'completion': result['completion'], 'original_review': str(Path(review_path).resolve()),
        'previous_performance': previous_performance})
    write(out/'performance.json', {'status': 'completed', 'total_sec': round(time.perf_counter()-started, 3),
        'recovered_from_partial': True, 'completion': result['completion']})
    return result_bundle
