"""Orchestrate inspect → analyze → match → plan → cubase mixdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from autoedit.audio.analyzer import analyze_audio_dir
from autoedit.audio.segments import expand_match_units
from autoedit.audio.taxonomy import item_group
from autoedit.audio.source_speech import transcribe_source
from autoedit.audio.pronunciation import build_sound_anchors
from autoedit.audio.review import write_audio_review
from autoedit.cubase.adapter import apply_cubase_plan
from autoedit.match.matcher import match_slots_to_video, InsufficientFootageError
from autoedit.plan.generator import build_edit_plan
from autoedit.premiere.adapter import apply_payload, write_apply_payload
from autoedit.premiere.prproj import inspect_prproj
from autoedit.schema import save_plan, validate_plan
from autoedit.validate import require_inputs
from autoedit.video.analyzer import analyze_video
from autoedit.audio.scene_match import build_sampler_slots
from autoedit.cubase.inspect import inspect_sampler_tracks, select_sampler_tracks
from autoedit.video.refine import refine_selected_cuts
from autoedit.audio.phonetics import write_catalog
from autoedit.video.silent_source import prepare_video_only_source
from autoedit.video.characters import analyze_characters, settings as character_settings
from autoedit.config import source_paths
from autoedit.video.pool import combine_video_infos, shift_times, local_cut, localize_plan, span_for_cut

Log = Callable[[str], None]


def _source_has_audio(path):
    """Cheap stream metadata; silent files never force a whole-source ASR scan."""
    import av
    with av.open(str(path)) as media:
        return bool(media.streams.audio)


def _log(log: Log | None, msg: str) -> None:
    if log:
        log(msg)


def run_pipeline(config: dict[str, Any], log: Log | None = None) -> dict[str, Any]:
    from autoedit.runtime import cancellation, job_lock, MemoryMonitor
    with cancellation(config.get('_cancel_file')), job_lock(log) as waited, MemoryMonitor() as memory:
        return _run_logged_pipeline({**config, '_queue_wait_sec': round(waited, 3), '_memory': memory}, log)


def _run_logged_pipeline(config: dict[str, Any], log: Log | None = None) -> dict[str, Any]:
    """Persist progress/errors even when launched from the GUI."""
    import time
    import traceback
    from datetime import datetime

    out = Path((config.get("job") or {}).get("output_dir") or "./output")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = {**config, '_timings': {}}
    def performance(status):
        import platform
        import os
        data = {'status':status, 'total_sec':round(time.perf_counter()-started,3),
                'queue_wait_sec':config.get('_queue_wait_sec', 0),
                'ai_cache_hit': config.get('_ai_cache_hit'),
                'memory':config['_memory'].report,
                'stage_seconds':config['_timings'], 'platform':platform.platform(),
                'processor':platform.processor(), 'cpu_threads':os.cpu_count()}
        (out/'performance.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    with (out / "run.log").open("w", encoding="utf-8", buffering=1) as logfile:
        def report(message):
            line = f"{datetime.now().isoformat(timespec='seconds')} | {message}"
            logfile.write(line + "\n")
            _log(log, message)

        report("Run started; progress is saved to " + str((out / "run.log").resolve()))
        from autoedit.video.analyzer import ANALYSIS_VERSION
        report(f"Video analysis version={ANALYSIS_VERSION}; frame and half-second transition checks enabled")
        try:
            result = _run_pipeline(config, report)
        except Exception as exc:
            report(f"FAILED after {time.perf_counter() - started:.1f}s: {exc}")
            logfile.write(traceback.format_exc())
            from autoedit.runtime import JobCancelled
            performance('cancelled' if isinstance(exc, JobCancelled) else 'failed')
            raise
        report(f"Run completed in {time.perf_counter() - started:.1f}s")
        performance('completed')
        report('Stage timings (seconds): ' + json.dumps(config['_timings']))
        return result


def _timed(config, name, function, *args, **kwargs):
    import time
    started = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        timings = config.get('_timings')
        if timings is not None:
            timings[name] = round(timings.get(name, 0.) + time.perf_counter()-started, 3)


def _run_pipeline(config: dict[str, Any], log: Log | None = None) -> dict[str, Any]:
    if config.get('ai', {}).get('enabled') or config.get('_ai_review'):
        from autoedit.ai.pipeline import run_ai_pipeline
        return run_ai_pipeline(config, log)
    resources = []
    try:
        return _run_standard_pipeline(config, log, resources)
    finally:
        for resource in resources:
            resource.close()


def _run_standard_pipeline(config, log, resources):
    import time
    from autoedit.presets import ROOT
    selection_started = time.monotonic()
    shared_cache = Path(config.get('ai', {}).get('cache_dir') or ROOT/'output/.ai-cache')
    job = config.get("job") or {}
    out = Path(job.get("output_dir") or "./output")
    out.mkdir(parents=True, exist_ok=True)

    premiere_cfg = config.get("premiere") or {}
    audio_cfg = config.get("audio") or {}
    video_cfg = config.get("video") or {}
    verification_mode = video_cfg.get('verification_mode', 'verified')
    if verification_mode not in {'verified', 'review'}:
        raise ValueError('video.verification_mode must be verified or review')
    review_mode = verification_mode == 'review'
    if review_mode:
        requested_characters = character_settings(video_cfg.get('characters'))
        video_cfg = {**video_cfg, 'require_lip_motion': False, 'characters': {
            **requested_characters,
            'embeddings_per_track': min(requested_characters.get('embeddings_per_track', 8), 4)}}
        config = {**config, 'video': video_cfg}
        _log(log, 'Fast review export: retain exposure-ranked characters; sample up to four identity images per track; '
                  'transcribe selected ranges only; defer dense face/lip/onset checks to manual review')
    character_cfg = character_settings(video_cfg.get("characters"))
    body_references = any(region.get('kind') == 'body'
        for regions in character_cfg.get('reference_regions', {}).values() for region in regions)
    character_cfg['_require_actor_face'] = video_cfg.get('require_clear_face', not body_references)
    if not isinstance(character_cfg['_require_actor_face'], bool):
        raise ValueError('video.require_clear_face must be true or false')
    cubase_cfg = config.get("cubase") or {}
    template_name = premiere_cfg.get("template_sequence") or "PJ 5 - demo"
    source_kind = str(video_cfg.get("source_kind") or "live_action")
    if source_kind not in {"live_action", "animation"}:
        raise ValueError("Source type must be live_action or animation.")
    audio_directory = audio_cfg.get("directory") or ""

    require_inputs(
        premiere_project=premiere_cfg.get("project") or "",
        cubase_project=cubase_cfg.get("project") or "",
        source_video=premiere_cfg.get("source_media") or "",
        audio_directory=audio_directory,
        template_sequence=template_name,
        video_track_index=int(premiere_cfg.get("video_track_index", 1)),
        sampler_tracks=cubase_cfg.get("sampler_tracks"),
    )

    notes: list[str] = []

    inspect: dict[str, Any] = {"slots": []}
    prproj = premiere_cfg.get("project")
    if prproj:
        _log(log, f"Inspecting Premiere project {prproj}")
        inspect = inspect_prproj(
            prproj,
            source_sequence=premiere_cfg.get("source_sequence") or "",
            template_sequence=template_name,
            video_track_index=int(premiere_cfg.get("video_track_index", 1)),
        )
        if inspect.get("template_sequence"):
            template_name = inspect["template_sequence"]
            premiere_cfg = {**premiere_cfg, "template_sequence": template_name}
            config = {**config, "premiere": premiere_cfg}
            _log(log, f"  template={template_name} selection={inspect.get('template_selection', 'configured')}")
        (out / "premiere_inspect.json").write_text(
            json.dumps(inspect, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _log(
            log,
            f"  sequences={len(inspect.get('sequences') or [])} "
            f"nested_slots={len(inspect.get('slots') or [])} "
            f"timeline_instances={len(inspect.get('instance_slots') or [])}",
        )
    else:
        notes.append("No premiere.project set; slots will be synthesized from audio order.")

    slots = inspect.get("slots") or []
    scenes = inspect.get("scene_slots") or []
    inventory = inspect_sampler_tracks(cubase_cfg["project"])
    inventory = select_sampler_tracks(inventory, cubase_cfg.get("sampler_tracks"), len(scenes), scenes=scenes)
    (out / "cubase_inspect.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    if len(scenes) != len(inventory["tracks"]):
        raise ValueError(f"Premiere scene runs={len(scenes)}, Cubase sampler tracks={len(inventory['tracks'])}. Select the paired template.")
    if inventory.get("mapping_mode") == "duplicate_scene_tracks":
        duplicates = sum(bool(t.get("is_duplicate")) for t in inventory["tracks"])
        _log(log, f"Audio mapping: {len(scenes)} scene runs -> "
                  f"{len(inventory['tracks']) - duplicates} numbered sampler tracks + {duplicates} (D) companions; "
                  "repeated Premiere UIDs reuse their selected sample")
    else:
        _log(log, f"Audio mapping: {len(scenes)} scene runs -> {len(inventory['tracks'])} numbered sampler tracks")
    if inventory.get("selected_track_indices"):
        _log(log, "Selected sampler tracks: " + ", ".join(t["name"] for t in inventory["tracks"]))

    _log(log, f"Analyzing audio in {audio_directory}")
    audio_items = _timed(config, 'voice_library', analyze_audio_dir,
        audio_directory,
        pattern=audio_cfg.get("glob") or "*.wav",
        files=audio_cfg.get("files") or None,
        max_files=int(audio_cfg.get("max_files") or 0),
        cache_dir=shared_cache,
    )
    (out / "audio_analysis.json").write_text(
        json.dumps(audio_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    file_count = len(audio_items)
    write_catalog(audio_items, out / "voice_catalog.csv")
    neutral_count = sum(item_group(item) == "NEUTRAL" for item in audio_items)
    notes.append(f"Shared Voice library: {file_count} files, {neutral_count} NEUTRAL fallbacks; source words take priority for every source type.")
    # Keep complete recordings as candidates as well as their timed expressions.
    audio_items = audio_items + expand_match_units(audio_items)
    if not audio_items:
        raise ValueError("No usable audio found in the selected folder.")
    _log(log, f"  audio files={file_count} expression segments={len(audio_items)}")

    if not slots:
        raise ValueError("Template has no nested sequence slots on the selected video track.")
    notes.append(f"Using exactly {len(slots)} nested sequences; repeated timeline clips remain unchanged.")

    sources = source_paths(premiere_cfg.get("source_media"))
    source_media = sources[0]
    gap = float(video_cfg.get("source_gap_sec", 5.0))
    source_infos = []
    selected = None
    last_failure = None
    mode = video_cfg.get('analysis_mode', 'until_filled')
    if mode not in {'until_filled', 'full'}:
        raise ValueError('video.analysis_mode must be until_filled or full')
    requested_mode = mode
    automatic_cast = character_cfg['enabled'] and not character_cfg.get('reference_images')
    if automatic_cast:
        _log(log, 'Automatic cast ranking: ' + ('rank the analyzed footage and stop after all slots pass verification'
            if mode == 'until_filled' else 'analyze every configured source before ranking individuals'))
    import math
    import time
    chunk_sec = float(video_cfg.get('analysis_chunk_sec', 30))
    if not math.isfinite(chunk_sec) or chunk_sec <= 0:
        raise ValueError('video.analysis_chunk_sec must be a positive finite number')
    scan_fps = float(video_cfg.get('sample_fps') or 6)
    if not math.isfinite(scan_fps) or not 2 <= scan_fps <= 10:
        raise ValueError('Video sampling must be between 2 and 10 fps for temporal lip detection.')
    fast_animation = False
    if mode == 'until_filled' and automatic_cast and float(video_cfg.get('search_fps', 1)) != 0:
        incremental_fps = float(video_cfg.get('incremental_fps', 2))
        if not math.isfinite(incremental_fps) or not 2 <= incremental_fps <= 10:
            raise ValueError('video.incremental_fps must be between 2 and 10')
        scan_fps = min(scan_fps, incremental_fps)
        from autoedit.video.objects import RUNTIME, MODEL_DIR
        fast_animation = (source_kind == 'animation' and not video_cfg.get('require_lip_motion')
            and (video_cfg.get('object_detection') or {}).get('enabled', 'auto') is not False
            and RUNTIME.is_file() and (MODEL_DIR / 'model.safetensors').is_file())
        _log(log, f'Incremental cast scan at {scan_fps:g} fps; selected cuts retain dense face/lip verification')
    config = {**config, 'job': {**job, 'output_dir': str(out)}}
    from autoedit.audio.search import SourceMatchSession
    from autoedit.ai.cache import identity, digest
    # Source metadata is supplied by the video reader; no second whole-video
    # probe/hash is needed for the standard pipeline.
    audio_sources = []
    source_session = SourceMatchSession(config, audio_sources, audio_items,
        shared_cache,
        lambda m: _log(log, m), started=selection_started)
    resources.append(source_session)
    config['_source_match_session'] = source_session

    def source_audio(raw):
        path = str(raw['path'])
        source = next((s for s in audio_sources if s['path'] == path), None)
        if source is None:
            limit = min(raw['duration_sec'], float(video_cfg.get('max_seconds') or raw['duration_sec']))
            source = {'id': f'source-{len(audio_sources)+1}', 'path': path, 'duration_sec': raw['duration_sec'],
                      'analyzed_end_sec': limit, 'identity': identity(path),
                      'source_speech': {'words': [], 'events': [], 'status': 'not-scanned'}}
            audio_sources.append(source)
        return source

    def speech_scan(path, analyzed, cache_dir, **kwargs):
        source = next(s for s in audio_sources if s['path'] == str(path))
        ranges = analyzed if isinstance(analyzed, list) else [(0., float(analyzed))]
        for a, b in ranges:
            cursor = (int(float(a)//30))*30.
            while cursor < b:
                edge = min(cursor+30., source['analyzed_end_sec'])
                if edge <= cursor:
                    break
                try:
                    source_session.scan(source, cursor, edge)
                except TimeoutError:
                    source_session.stats['stop_reason'] = 'time_budget'
                    break
                cursor = edge
        return source['source_speech']
    from autoedit.video.objects import ObjectScanSession
    runtime = {}
    reference_gallery = {}
    rejected_by_source = {}
    latest_pool = []
    scope = 'all configured source footage' if mode == 'full' else 'analyzed footage'
    _log(log, f'Analysis mode={mode}; chunk={chunk_sec:g}s; cast ranking uses {scope}')
    from autoedit.runtime import selection_deadline, SelectionTimeout
    try:
        with selection_deadline(source_session.deadline):
            with ObjectScanSession() as object_session:
                for source_media in sources:
                    _log(log, f'Analyzing selected source video {source_media}')
                    current = None
                    attempts = 0
                    def prepare_region(raw):
                        from autoedit.runtime import check_cancelled, release_inference
                        check_cancelled()
                        source_audio(raw)
                        ranges = raw.get('verified_ranges') or [(0., raw.get('analyzed_seconds', raw['duration_sec']))]
                        speech_scan(raw['path'], ranges, out/'video_cache')
                        source_session.release_for_vision()
                        release_inference(runtime)
                        if raw.get('_body_reference_verification'):
                            from autoedit.video.reference_verify import verify_body_references
                            return _timed(config,'reference_verification',verify_body_references,raw,out / 'video_cache',
                                character_cfg,reference_gallery,progress=lambda m:_log(log,'  '+m))
                        enriched = raw
                        if character_cfg['enabled']:
                            from autoedit.video.objects import augment_with_objects
                            from autoedit.video.reference_search import body_screening_enabled
                            object_cfg = {**(video_cfg.get('object_detection') or {}),
                                          'target_references':bool(character_cfg.get('reference_images')),
                                          'target_body_references':body_screening_enabled({**video_cfg,'reference_discovery':'auto'}),
                                          'target_prompt':raw.get('_reference_prompt')}
                            enriched = _timed(config, 'objects', augment_with_objects, raw, out / 'video_cache', object_cfg,
                                progress=lambda m: _log(log, '  ' + m), session=object_session)
                        object_session.offload()
                        region_characters = {**character_cfg, '_reference_gallery':reference_gallery}
                        enriched = _timed(config, 'characters', analyze_characters, enriched, out / 'video_cache', region_characters,
                            progress=lambda m: _log(log, '  ' + m), runtime_cache=runtime)
                        release_inference(runtime)
                        return enriched

                    def evaluate(raw):
                        nonlocal selected, last_failure, current, attempts, latest_pool
                        attempts += 1
                        started = time.perf_counter()
                        raw.setdefault('path', source_media)
                        source_audio(raw)
                        from autoedit.runtime import check_cancelled, release_inference
                        check_cancelled()
                        enriched = raw if raw.get('_regions_enriched') else prepare_region(raw)
                        from autoedit.video.composition import annotate_compositions
                        enriched = _timed(config, 'compositions', annotate_compositions, enriched, out / 'video_cache',
                            progress=lambda m: _log(log, '  ' + m))
                        current = enriched
                        pool = source_infos + [enriched]
                        latest_pool = pool
                        _log(log, f'  Evidence ready through {enriched.get("analyzed_seconds", enriched["duration_sec"]):.1f}s '
                                  f'in {time.perf_counter()-started:.1f}s; checking {len(slots)} slots')
                        while True:
                            failed_before = sum(len(ranges) for ranges in rejected_by_source.values())
                            try:
                                selected = _select_candidate(config, pool, slots, scenes, inspect, inventory,
                                    audio_items, list(notes), log, speech_scan, rejected_by_source)
                                break
                            except InsufficientFootageError as exc:
                                last_failure = exc
                                if sum(len(ranges) for ranges in rejected_by_source.values()) > failed_before:
                                    # Dense rejections can require a different intro reservation.
                                    continue
                                _log(log, f'  Allocation not complete yet: {exc}; continue source analysis')
                                return False
                        _log(log, f'  {"Selected for manual review" if review_mode else "Verified"} '
                                  f'slots={len(slots)}/{len(slots)}; source analysis can stop')
                        return True
                    if mode == 'until_filled' and not automatic_cast and float(video_cfg.get('search_fps', 1)) != 0:
                        from autoedit.video.search import search_video
                        raw = _timed(config, 'source_processing_inclusive', search_video, source_media,
                            video_cfg, slots + ([inspect['intro_slot']] if (inspect.get('intro_slot') or {}).get('status') == 'empty' else []),
                            out / 'video_cache', object_session, evaluate, progress=lambda m: _log(log, '  ' + m),
                            **({'enrich_region':prepare_region} if character_cfg.get('reference_images') else {}))
                    else:
                        raw = _timed(config, 'source_processing_inclusive', analyze_video,
                            source_media,
                            sample_fps=scan_fps,
                            max_seconds=float(video_cfg.get('max_seconds') or 0),
                            min_segment_sec=float(video_cfg.get('min_segment_sec') or 0.25),
                            merge_gap_sec=float(video_cfg.get('merge_gap_sec') or 0.20),
                            backend=str(video_cfg.get('backend') or 'auto'),
                            progress=lambda m: _log(log, '  ' + m), cache_dir=out / 'video_cache',
                            source_kind=source_kind, main_group=character_cfg['main_group'],
                            **({'identity_only': True} if fast_animation else {}),
                            **({'on_chunk': evaluate, 'chunk_sec': chunk_sec} if mode == 'until_filled' else {}))
                    if mode == 'full':
                        raw.setdefault('path', source_media)
                        source_audio(raw)
                        speech_scan(source_media, float(raw.get('analyzed_seconds', raw['duration_sec'])), out/'video_cache')
                        if character_cfg['enabled']:
                            from autoedit.video.objects import augment_with_objects
                            raw = _timed(config, 'objects', augment_with_objects,raw, out / 'video_cache',
                                {**(video_cfg.get('object_detection') or {}), 'target_references':bool(character_cfg.get('reference_images'))},
                                progress=lambda m: _log(log, '  ' + m), session=object_session)
                        object_session.offload()
                        current = _timed(config, 'characters', analyze_characters,raw, out / 'video_cache', character_cfg,
                            progress=lambda m: _log(log, '  ' + m), runtime_cache=runtime)
                        from autoedit.video.composition import annotate_compositions
                        current = _timed(config, 'compositions', annotate_compositions, current, out / 'video_cache',
                            progress=lambda m: _log(log, '  ' + m))
                        from autoedit.runtime import release_inference
                        release_inference(runtime)
                    elif not attempts:  # Also supports custom analyzers returning a complete snapshot.
                        evaluate(raw)
                    if current is not None and raw.get('searched_until_sec') is not None:
                        for field in ('analysis_complete', 'searched_until_sec'):
                            current[field] = raw[field]
                    source_infos.append(current)
                    if mode == 'until_filled' and selected:
                        break
                if mode == 'full':
                    try:
                        selected = _select_candidate(config, source_infos, slots, scenes, inspect, inventory,
                            audio_items, list(notes), log, speech_scan, rejected_by_source)
                    except InsufficientFootageError as exc:
                        last_failure = exc
                if selected is None and character_cfg.get('reference_images'):
                    _log(log, 'Reference search exhausted configured sources/limits; enabling exposure-ranked fallback')
                    for index, info in enumerate(source_infos):
                        if info.get('_regions_enriched'):
                            _log(log, '  Ranking the complete searched cast once for reference fallback')
                            source_infos[index] = _timed(config, 'characters', analyze_characters, info,
                                out / 'video_cache', character_cfg, progress=lambda m: _log(log, '  ' + m), runtime_cache=runtime)
                            from autoedit.runtime import release_inference
                            release_inference(runtime)
                    selected = _select_candidate({**config, '_allow_reference_fallback': True}, source_infos,
                        slots, scenes, inspect, inventory, audio_items, list(notes), log, speech_scan, rejected_by_source)
    except SelectionTimeout:
        source_session.stats['stop_reason'] = 'time_budget'
        source_infos = latest_pool or source_infos
        if not source_infos:
            raise InsufficientFootageError('Search budget exhausted before usable character evidence was available') from None
        review_mode = True
        verification_mode = 'review'
        video_cfg = {**video_cfg, 'verification_mode': 'review'}
        config = {**config, 'video': video_cfg}
        _log(log, 'Shared search deadline reached; exporting completed evidence for manual review.')
        with selection_deadline(source_session.final_deadline):
            selected = _select_candidate(config, source_infos, slots, scenes, inspect, inventory,
                audio_items, list(notes), log, speech_scan, rejected_by_source)
    runtime.clear()
    if selected is None:
        raise last_failure or InsufficientFootageError('Not enough qualifying footage to fill all slots')
    plan, video_info, speech, lip_info = selected
    if source_session.stats['stop_reason'] == 'searching':
        source_session.stats['stop_reason'] = 'enough_slots' if all(
            s['audio_match'].get('cut_onset_matched') for s in plan['cubase_slots']) else 'neutral_fallback'
    source_session.stats.update(matched_slots=sum(bool(s['audio_match'].get('cut_onset_matched')) for s in plan['cubase_slots']),
        fallback_slots=sum(not s['audio_match'].get('cut_onset_matched') for s in plan['cubase_slots']),
        selection_total_sec=round(time.monotonic()-selection_started, 3))
    plan['source_audio_search'] = dict(source_session.stats)
    source_session.close()
    config.setdefault('_timings', {}).update(source_audio=round(source_session.stats['asr_sec'], 3),
        source_audio_regions=source_session.stats['regions'], source_audio_cache_hits=source_session.stats['cache_hits'])
    plan['video_analysis_meta'].update(
        analysis_mode=mode, stop_reason=('all_slots_selected_for_review' if review_mode else 'all_slots_verified')
            if mode == 'until_filled' else 'full_scan',
        verification_mode=verification_mode, needs_review=review_mode,
        requested_analysis_mode=requested_mode,
        character_ranking_scope='configured_sources' if mode == 'full' else 'analyzed_footage',
        subject_filters=[{'path':i['path'], **i.get('subject_filter',{})} for i in source_infos],
        compositions=[{'path':i['path'], **i.get('composition_analysis',{})} for i in source_infos],
        inference=[{'path': i['path'], 'objects':i.get('object_analysis', {}).get('runtime'),
            'characters':i.get('character_analysis', {}).get('execution')} for i in source_infos],
        analyzed_sources=[{'path': i['path'], 'duration_sec': i['duration_sec'],
            'sample_fps': i.get('sample_fps'),
            'analyzed_seconds': i.get('analyzed_seconds', i['duration_sec']),
            'decoded_seconds': i.get('decoded_seconds', i.get('analyzed_seconds', i['duration_sec'])),
            'verified_ranges': i.get('verified_ranges'), 'verified_seconds':i.get('verified_seconds'),
            'searched_until_sec':i.get('searched_until_sec'), 'search_fps':i.get('search_fps'),
            'dense_region_count':i.get('dense_region_count')}
            for i in source_infos])
    if video_info.get('character_analysis'):
        (out / 'character_analysis.json').write_text(json.dumps(video_info['character_analysis'], indent=2), encoding='utf-8')
    for name, data in [('video_analysis', video_info), ('source_transcript', speech),
                       ('source_sound_matches', video_info.get('sound_anchors', [])), ('lip_analysis', lip_info)]:
        (out / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    used = {span_for_cut(video_info, slot['video'])['path'] for slot in plan['slots']} if video_info.get('source_spans') else {source_infos[0]['path']}
    if plan.get('intro_fill') and video_info.get('source_spans'):
        used.add(span_for_cut(video_info, plan['intro_fill'])['path'])
    stamps = {s['path']: s['identity'] for s in audio_sources}
    video_only_sources = {path: _timed(config, 'video_only_copy', prepare_video_only_source, path,
        shared_cache/'video_only'/(stamps[path].get('sha256') or digest(stamps[path])),
        progress=lambda m: _log(log, '  ' + m)) for path in [i['path'] for i in source_infos] if path in used}
    plan['project']['video_only_source'] = next(iter(video_only_sources.values()))
    localize_plan(plan, video_info, video_only_sources)
    plan["audio"] = [s["audio"] for s in plan["cubase_slots"]]
    review = sorted({s["premiere"]["nested_sequence"] for s in plan["cubase_slots"]
                     if s["audio_match"].get("needs_review")}, key=lambda s: (len(s), s))
    if review:
        _log(log, "Audio matches need review for scenes: " + ", ".join(review))
    # Automatic import must not mistake a Python timing preview for a Cubase mix.
    plan["project"]["import_mixdown"] = False
    for slot in plan["cubase_slots"]:
        _log(log, f"  {slot['cubase']['track_name']} <- scene {slot['premiere']['nested_sequence']} "
                  f"<- {slot['audio'].get('name', slot['audio']['path'])} ({slot['audio']['category']}) "
                  f"method={slot['audio_match']['method']}")
    _log(log, "Preparing sampler WAVs and a Premiere timing preview; Cubase MIDI/FX render remains in Cubase")
    cubase_result = _timed(config, 'sampler_artifacts', apply_cubase_plan, plan, out)
    # Listening review uses the actual source soundtrack and generated samples.
    _timed(config, 'audio_review', write_audio_review, plan, out, speech)
    _log(log, "Listening review: " + str(out / "audio_review.html"))
    audio_errors = [row["error"] for row in cubase_result.get("tracks", []) if row.get("error")]
    if audio_errors:
        raise ValueError("Audio mixdown incomplete: " + "; ".join(audio_errors[:5]))
    plan["cubase_manifest"] = cubase_result.get("manifest")
    errors = validate_plan(plan)
    if errors:
        raise ValueError("Invalid edit plan: " + "; ".join(errors))
    plan_path = save_plan(plan, out / "edit-plan.json")
    write_apply_payload(plan, out / "premiere_apply.json")
    _log(log, f"Wrote {plan_path}")
    return {"plan_path": str(plan_path), "plan": plan, "inspect": inspect}


def _select_candidate(config, source_infos, slots, scenes, inspect, inventory, audio_items, notes, log, speech_scan, rejected_by_source):
    """Accept an entire allocation under the configured verification policy."""
    out = Path(config['job']['output_dir'])
    sources = [i['path'] for i in source_infos]
    video_cfg = config.get('video') or {}
    review_mode = video_cfg.get('verification_mode', 'verified') == 'review'
    character_cfg = character_settings(video_cfg.get('characters'))
    source_kind = video_cfg.get('source_kind') or 'live_action'
    def select(*args, **kwargs):
        return _timed(config, 'allocation', match_slots_to_video, *args, **kwargs)
    gap = float(video_cfg.get('source_gap_sec', 5.0))
    source_media = sources[0]
    video_info = combine_video_infos(source_infos, gap)
    if config.get('_allow_reference_fallback') and character_cfg.get('reference_images'):
        from autoedit.video.objects import enable_ranked_exposure_fallback
        video_info = enable_ranked_exposure_fallback(video_info)
    video_info['reference_only'] = bool(character_cfg.get('reference_images')) and not config.get('_allow_reference_fallback', False)
    video_info["source_kind"] = source_kind
    body_references = any(region.get('kind') == 'body'
        for regions in character_cfg.get('reference_regions', {}).values() for region in regions)
    require_clear_face = video_cfg.get('require_clear_face', not body_references)
    if not isinstance(require_clear_face, bool):
        raise ValueError('video.require_clear_face must be true or false')
    video_info['require_clear_face'] = require_clear_face
    max_similar = video_cfg.get('max_similar_cuts', 2)
    if isinstance(max_similar, bool) or not isinstance(max_similar, int) or max_similar < 0:
        raise ValueError('video.max_similar_cuts must be a non-negative integer (0 disables the limit)')
    video_info['max_similar_cuts'] = max_similar
    rejected_ranges = []
    for info in source_infos:
        offset = next((s['offset_sec'] for s in video_info.get('source_spans', []) if s['path'] == info['path']), 0.)
        rejected_ranges.extend((a+offset,b+offset) for a,b in rejected_by_source.get(info['path'], []))
    intro = inspect.get("intro_slot")
    if not intro:
        _log(log, "Intro: no empty lead-in before the first scene; no intro will be added.")
    elif intro["status"] != "empty":
        _log(log, "Intro: existing opening content is preserved; no intro will be added.")
    else:
        _log(log, f"Intro: empty lead-in of {intro['required_duration_sec']:.3f}s will be filled.")
    require_lip_motion = video_cfg.get("require_lip_motion", False)
    if not isinstance(require_lip_motion, bool):
        raise ValueError("video.require_lip_motion must be true or false")
    # Protect full scene windows before reserving a tiny unvoiced intro.
    # Its source gap can otherwise consume an entire primary-character shot.
    # Reserve it before sound-led allocation, keeping later onsets stable.
    extra = None
    excluded = []
    speech_by_source = {}
    for source_info in source_infos:
        analyzed = min(float(source_info["duration_sec"]),
                       float(source_info.get("analyzed_seconds") or source_info["duration_sec"]))
        ranges = source_info.get('verified_ranges')
        _log(log, f"Scanning source audio {source_info['path']} in {ranges or [(0, analyzed)]} before selecting sound-led cuts.")
        speech_by_source[source_info["path"]] = speech_scan(source_info["path"], ranges if ranges is not None else analyzed, out / "video_cache",
                               model=video_cfg.get("speech_model") or "small",
                               language=video_cfg.get("speech_language", "en"),
                               device=video_cfg.get("speech_device", "auto"),
                               compute_type=video_cfg.get("speech_compute_type", "auto"),
                               device_index=video_cfg.get("speech_device_index", 0),
                               progress=lambda m: _log(log, "  " + m))
        if review_mode:
            speech_by_source[source_info['path']] = {**speech_by_source[source_info['path']],
                'scope': 'analyzed-source-regions', 'requested_ranges': [list(r) for r in (ranges or [(0., analyzed)])]}
    speech = speech_by_source[source_infos[0]["path"]]
    if video_info.get("source_spans"):
        speech = {"model": video_cfg.get("speech_model") or "small", "words": [], "events": [],
                  "sources": speech_by_source}
        if review_mode:
            speech['scope'] = 'analyzed-source-regions'
        for span in video_info["source_spans"]:
            shifted = shift_times(speech_by_source[span["path"]], span["offset_sec"])
            for field in ("words", "events"):
                speech[field].extend({**item, "source_video": span["path"]} for item in shifted.get(field, []))
    video_info["source_speech"] = speech
    video_info["sound_anchors"] = build_sound_anchors(audio_items, speech)
    from autoedit.audio.matching import visual_event_confirmed
    video_info['sound_anchors'] = [a for a in video_info['sound_anchors']
        if not a.get('action') or visual_event_confirmed(a, samples=video_info.get('samples', []))]
    from autoedit.runtime import release_inference
    release_inference({})  # The transcript is plain data; free Whisper before dense GPU face queries.
    _log(log, f"  library-matched sound onsets={len(video_info['sound_anchors'])}; retaining slot lengths and source gaps")
    _log(log, f"  high-confidence onsets={sum(a['confidence_tier'] == 1 for a in video_info['sound_anchors'])}; "
              f"remaining supported onsets need listening review")
    visual_matches = None
    if intro and intro["status"] == "empty":
        visual_matches = select(slots, [], video_info, min_gap_sec=gap,
                                             require_lip_motion=require_lip_motion, rejected_ranges=rejected_ranges)
        try:
            extra = select([intro], [], video_info, min_gap_sec=gap,
                require_speaking=False, excluded_ranges=[(m['video']['in_sec'],m['video']['out_sec'])
                                                         for m in visual_matches])[0]['video']
        except InsufficientFootageError:
            # An alternative initial reservation can still make the full job
            # feasible; do not discard it merely because this layout ran out.
            extra = select([intro], [], video_info, min_gap_sec=gap,
                                        require_speaking=False)[0]['video']
            visual_matches = None
        excluded = [(extra["in_sec"], extra["out_sec"])]
        _log(log, f"Intro reserved at {extra['in_sec']:.3f}-{extra['out_sec']:.3f}s before sound-led cuts.")
    _log(log, "Selection mode: " + ("verified moving mouths" if require_lip_motion else
         "clear faces required throughout cuts" if require_clear_face else "visible character shots; body-only fallback enabled"))
    _log(log, "Checking full-duration person evidence around source-supported sound onsets.")
    if visual_matches is None:
        visual_matches = select(slots, [], video_info, min_gap_sec=gap,
                                             excluded_ranges=excluded, require_lip_motion=require_lip_motion, rejected_ranges=rejected_ranges)
    matches = visual_matches  # This allocation already used the opening-sound anchors.
    session = config.get('_source_match_session')
    unsearched = [p for p in source_paths(config['premiere']['source_media']) if p not in sources]
    presence = config.setdefault('_source_audio_presence', {})
    for path in unsearched:
        if path not in presence:
            presence[path] = _source_has_audio(path)
    audio_expected = any(s.get('has_audio') for s in speech_by_source.values()) or any(presence[p] for p in unsearched)
    if session and audio_expected and not session.expired and any(not m['video'].get('source_sound') for m in matches):
        limits = {s['path']: s['analyzed_end_sec'] for s in session.sources}
        complete = len(source_infos) == len(source_paths(config['premiere']['source_media'])) and all(
            float(i.get('analyzed_seconds') or i['duration_sec']) >= limits.get(str(i['path']), i['duration_sec'])-1e-6
            or i.get('analysis_complete') for i in source_infos)
        if not complete:
            raise InsufficientFootageError('More source audio must be searched before assigning neutral fallback voices')
    notes.append(f"Cuts limit recurring camera compositions to {max_similar or 'unlimited'} uses, then prefer exposure-ranked actors and library-matched sound onsets; each scene uses the WAV matching its opening sound. Unmatched speaking cuts use measured visemes/rhythm and require review. Legacy unvoiced selection may use neutral fallback.")
    for slot, match in zip(slots, matches):
        selected = match["video"]
        metrics = selected.get("selection_metrics") or {}
        _log(log, f"  clip {slot.get('nested_sequence')}: source {selected['in_sec']:.3f}-{selected['out_sec']:.3f}s "
                  f"tier={metrics.get('tier')} readable_face={metrics.get('readable_face', 0):.2f} "
                  f"mouth_clear={metrics.get('clear_face', 0):.2f} speaking={metrics.get('speaking', 0):.2f}")
        if metrics.get('composition_ids'):
            _log(log, '    composition=' + ','.join(metrics['composition_ids']))
        if selected.get("source_sound"):
            sound = selected["source_sound"]
            _log(log, f"    opening sound={sound.get('word') or sound.get('action')} at {sound['start']:.3f}s")
        if selected.get("character_selection"):
            character = selected["character_selection"]
            _log(log, f"    character={character['decision']} ids={character['character_ids']} "
                      f"rank={character['best_rank']} review={character['needs_review']}")
    notes.append(f"Source gap: {video_cfg.get('source_gap_sec', 5.0)}s; four-tier selection requires a face or independently confirmed body throughout full cuts. Stable scenery without a person is excluded.")

    plan = build_edit_plan(
        config=config,
        slots=slots,
        matches=matches,
        video_info=video_info,
        notes=notes,
    )
    plan["template_audio_tracks"] = inspect.get("template_audio_tracks", [])
    if inspect.get("nested_structure") is not None:
        plan["nested_structure"] = inspect["nested_structure"]
    if intro and intro["status"] == "empty":
        plan["intro_fill"] = {"slot_id": "intro", "mode": "intro_gap",
            "in_sec": extra["in_sec"], "out_sec": extra["out_sec"],
            "overwrite_at_sec": intro["start_sec"], "slot_duration_sec": intro["required_duration_sec"],
            "end_sec": intro["end_sec"], "video_track_index": intro["video_track_index"],
            "audio_track_index": -1, "selection_metrics": extra["selection_metrics"],
            "character_selection": extra.get("character_selection")}
        _log(log, f"Intro: extra video cut {extra['in_sec']:.3f}-{extra['out_sec']:.3f}s tier={extra['selection_metrics']['tier']} -> timeline 0-{intro['end_sec']:.3f}s; configured timeline audio keeps its original start and source in/out")
    elif intro:
        _log(log, "Intro already contains content; preserving it: " + intro["reason"])
    plan["audio_mode"] = "sampler_per_scene_v1"
    from autoedit.video.faces import lip_motion_evidence
    from autoedit.audio.onsets import AudioOnsetError
    alignment_attempts = 0
    lip_info = {'samples': video_info.get('samples', []), 'source_kind': source_kind,
                'source_speech': speech, 'verify_onsets': False}
    if video_info.get('source_spans'):
        lip_info['source_spans'] = video_info['source_spans']
    if review_mode:
        lip_info.update(method='sparse-review-only', dense_verification_deferred=True)
        plan['review_status'] = 'manual_review_required'
        plan['notes'].append('Fast review export: character exposure ranking is retained. '
            'Source words are measured around provisional cuts only; dense faces, lips and '
            'articulation onsets are not verified. Review every video cut and voice before final use.')
        for slot in plan['slots']:
            slot['video']['selection_metrics'].update(dense_clear_face_verified=False,
                dense_lip_motion_verified=False, needs_review=True, verification_status='manual_review_required')
        if plan.get('intro_fill'):
            plan['intro_fill']['selection_metrics'].update(needs_review=True,
                verification_status='manual_review_required', dense_clear_face_verified=False)
        plan['cubase_slots'] = build_sampler_slots(scenes, plan['slots'], audio_items, lip_info, inventory)
        for slot in plan['cubase_slots']:
            if slot['audio_match'].get('method') == 'filename-visemes-and-dense-lip-rhythm':
                slot['audio_match']['method'] = 'filename-visemes-and-sparse-lip-rhythm'
            slot['audio_match'].update(needs_review=True, onset_verified=False, phoneme_sync_verified=False,
                review_reason='Fast export: dense face/lip/onset verification deferred to manual review')
        _log(log, 'Exporting the complete ranked allocation for manual review; dense verification is deferred')
    while not review_mode:
        readable_slots = [s for s in plan["slots"] if s["video"]["selection_metrics"]["tier"] <= 2
                          or s["video"]["selection_metrics"].get("require_clear_face")
                          or s["video"]["selection_metrics"].get("require_lip_motion")]
        lip_info = {"samples": []}
        for index, source_info in enumerate(source_infos):
            source_slots = readable_slots
            offset = 0.0
            if video_info.get("source_spans"):
                span = video_info["source_spans"][index]
                offset = span["offset_sec"]
                source_slots = [{**slot, "video": local_cut(video_info, slot["video"])} for slot in readable_slots
                                if span_for_cut(video_info, slot["video"])["path"] == span["path"]]
                # Refinement references keep the namespaced track identifiers.
                refs = [shift_times(sample, -offset) for sample in video_info["samples"]
                        if sample.get("source_video") == span["path"]
                        or span["offset_sec"] <= sample["time_sec"] < span["end_sec"]]
            else:
                refs = video_info.get("samples")
            refined = (_timed(config, 'dense_lips', refine_selected_cuts,source_info["path"], source_slots, out / "video_cache", source_kind=source_kind,
                                       main_group=character_cfg["main_group"],
                                       object_config=video_cfg.get('object_detection'),
                                       character_samples=refs if source_info.get("character_analysis") else None,
                                       progress=lambda m: _log(log, "  " + m))
                    if source_slots and video_cfg.get("backend") != "opencv" else {"samples": []})
            lip_info["samples"].extend(shift_times(refined.get("samples", []), offset))
        rejected = []
        onset_rejections = set()
        for slot in plan["slots"]:
            cut = slot["video"]
            if cut['selection_metrics'].get('require_clear_face'):
                from autoedit.video.refine import readable_face_fraction
                dense_samples = [s for s in lip_info.get('samples', [])
                    if not s.get('refinement_cut') or (
                        abs(s['refinement_cut']['in_sec']-cut['in_sec']) < 1e-6
                        and abs(s['refinement_cut']['out_sec']-cut['out_sec']) < 1e-6)]
                dense = readable_face_fraction(dense_samples, cut['in_sec'], cut['out_sec'])
                cut['selection_metrics']['dense_readable_face_fraction'] = dense['readable_face_fraction']
                if dense['sample_coverage'] < .99 or dense['readable_face_fraction'] < .85:
                    _log(log, f"  {slot['id']}: dense readable faces={dense['readable_face_fraction']:.1%}; "
                              f"sample coverage={dense['sample_coverage']:.1%}; reselecting")
                    rejected.append(slot)
                    continue
                cut['selection_metrics']['dense_clear_face_verified'] = True
            if cut["selection_metrics"].get("require_lip_motion"):
                window = [s for s in lip_info.get("samples", [])
                          if cut["in_sec"] <= s["time_sec"] < cut["out_sec"] and s.get("clear_face")]
                if not lip_motion_evidence(window):
                    rejected.append(slot)
                else:
                    cut["selection_metrics"]["dense_lip_motion_verified"] = True
        lip_info.update(source_kind=source_kind, source_speech=speech, verify_onsets=True)
        if video_info.get('source_spans'):
            lip_info['source_spans'] = video_info['source_spans']
        if not rejected:
            try:
                sampler_slots = build_sampler_slots(scenes, plan['slots'], audio_items, lip_info, inventory)
                starts = {s['premiere']['nested_sequence_uid']: s['audio_match'].get('suggested_cut_in_sec')
                          for s in sampler_slots if s['audio_match'].get('suggested_cut_in_sec') is not None}
                moving = [s for s in plan['slots'] if s['premiere']['nested_sequence_uid'] in starts
                          and abs(starts[s['premiere']['nested_sequence_uid']]-s['video']['in_sec']) > .02]
                if moving:
                    alignment_attempts += 1
                    if alignment_attempts > 4:
                        rejected = moving
                        onset_rejections = {s['id'] for s in moving}
                    else:
                        fixed = {s['id']: starts.get(s['premiere']['nested_sequence_uid'], s['video']['in_sec'])
                                 for s in plan['slots']}
                        try:
                            corrected = select(slots, [], {**video_info, 'fixed_starts': fixed}, min_gap_sec=gap,
                                excluded_ranges=excluded, require_lip_motion=require_lip_motion,
                                rejected_ranges=rejected_ranges)
                        except InsufficientFootageError:
                            rejected = moving
                            onset_rejections = {s['id'] for s in moving}
                        else:
                            for old, match in zip(plan['slots'], corrected):
                                if old['video'].get('source_sound'):
                                    match['video']['source_sound'] = old['video']['source_sound']
                            plan['slots'] = build_edit_plan(config=config, slots=slots, matches=corrected,
                                video_info=video_info, notes=notes)['slots']
                            _log(log, 'Aligned cuts to measured articulation onsets; rechecking full windows and lips')
                            continue
                if not moving:
                    plan['cubase_slots'] = sampler_slots
            except AudioOnsetError as exc:
                rejected = [s for s in plan['slots'] if s['video'] == exc.cut]
                onset_rejections = {s['id'] for s in rejected}
                if not rejected:
                    raise
        if not rejected:
            break
        _log(log, "Face/lip/onset check rejected " + ", ".join(s["id"] for s in rejected)
             + "; selecting alternative cuts with the same slot lengths and source gap.")
        alignment_attempts = 0
        for slot in rejected:
            cut = slot['video']
            global_start = cut['in_sec']
            path = source_infos[0]['path']
            if video_info.get('source_spans'):
                cut = local_cut(video_info, cut)
                path = cut['source_video']
            failed = (cut['in_sec'], cut['out_sec'])
            if slot['id'] in onset_rejections:
                # An invalid beginning does not disqualify the rest of a long cut.
                failed = (max(0., cut['in_sec']-.25), min(cut['out_sec'], cut['in_sec']+.35))
            known = rejected_by_source.setdefault(path, [])
            if failed in known:
                raise InsufficientFootageError('No new verified face/lip/onset candidates available')
            known.append(failed)
            offset = global_start-cut['in_sec']
            rejected_ranges.append((failed[0]+offset, failed[1]+offset))
        matches = select(slots, [], video_info, min_gap_sec=gap,
                                       excluded_ranges=excluded, require_lip_motion=require_lip_motion,
                                       rejected_ranges=rejected_ranges)
        plan["slots"] = build_edit_plan(config=config, slots=slots, matches=matches,
                                       video_info=video_info, notes=notes)["slots"]
    for slot in plan["slots"]:
        cut = slot["video"]
        face_metric = ('sparse_readable_face=' + f"{cut['selection_metrics'].get('readable_face', 0):.2f}"
                       if review_mode else 'dense_readable_face=' +
                       f"{cut['selection_metrics'].get('dense_readable_face_fraction', 0):.2f}")
        _log(log, f"  final clip {slot['premiere']['nested_sequence']}: "
                  f"{cut['in_sec']:.3f}-{cut['out_sec']:.3f}s "
                  f"{face_metric} "
                  f"dense_face_verified={cut['selection_metrics'].get('dense_clear_face_verified', False)} "
                  f"dense_lip_motion={cut['selection_metrics'].get('dense_lip_motion_verified', False)}")
    lip_info.update(source_kind=source_kind, source_speech=speech)
    if video_info.get("source_spans"):
        lip_info["source_spans"] = video_info["source_spans"]
    for slot in plan['slots']:
        metrics = slot['video'].get('character_selection') or slot['video']['selection_metrics'].get('character_selection') or {}
        if character_cfg.get('reference_images'):
            fallback = metrics.get('reference_fraction', 0) < character_cfg['min_main_fraction']
            metrics.update(reference_fallback=fallback,
                           reference_reason='reference-search-exhausted' if fallback else 'reference-character')
            if not fallback:
                metrics.update(needs_review=metrics.get('reference_confidence', 0) < .5,
                               reason='reference-character')
            slot['video']['character_selection'] = metrics
    reference = (video_info.get('character_analysis') or {}).get('reference')
    if reference:
        from autoedit.video.references import reference_selection_report
        report = reference_selection_report(reference, plan['slots'])
        plan['video_analysis_meta']['character_reference_selection'] = report
        _log(log, f"Reference selection: {report['selected_target_count']}/{len(report['characters'])} targets used; "
                  f"{report['fallback_cuts']} fallback cuts")
        for character in report['characters']:
            _log(log, f"  {character['id']}: {character['selected_cuts']} cuts, "
                      f"{character['selected_seconds']:.2f}s ({character['status']})")
    for slot in plan['cubase_slots']:
        uid = slot['premiere']['nested_sequence_uid']
        parent = next(s for s in plan['slots'] if s['premiere']['nested_sequence_uid'] == uid)
        slot['video'] = dict(parent['video'])
    errors = validate_plan(plan)
    if errors:
        raise ValueError('Invalid edit plan: ' + '; '.join(errors))
    apply_payload(plan)  # Validate identified nested targets before any media export.
    return plan, video_info, speech, lip_info
