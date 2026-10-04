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
from autoedit.premiere.adapter import write_apply_payload
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


def _log(log: Log | None, msg: str) -> None:
    if log:
        log(msg)


def run_pipeline(config: dict[str, Any], log: Log | None = None) -> dict[str, Any]:
    """Persist progress/errors even when launched from the GUI."""
    import time
    import traceback
    from datetime import datetime

    out = Path((config.get("job") or {}).get("output_dir") or "./output")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
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
            raise
        report(f"Run completed in {time.perf_counter() - started:.1f}s")
        return result


def _run_pipeline(config: dict[str, Any], log: Log | None = None) -> dict[str, Any]:
    job = config.get("job") or {}
    out = Path(job.get("output_dir") or "./output")
    out.mkdir(parents=True, exist_ok=True)

    premiere_cfg = config.get("premiere") or {}
    audio_cfg = config.get("audio") or {}
    video_cfg = config.get("video") or {}
    character_cfg = character_settings(video_cfg.get("characters"))
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
    inventory = select_sampler_tracks(inventory, cubase_cfg.get("sampler_tracks"), len(scenes))
    (out / "cubase_inspect.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    if len(scenes) != len(inventory["tracks"]):
        raise ValueError(f"Premiere scene runs={len(scenes)}, Cubase sampler tracks={len(inventory['tracks'])}. Select the paired template.")
    _log(log, f"Audio mapping: {len(scenes)} scene runs -> {len(inventory['tracks'])} numbered sampler tracks")
    if inventory.get("selected_track_indices"):
        _log(log, "Selected sampler tracks: " + ", ".join(t["name"] for t in inventory["tracks"]))

    _log(log, f"Analyzing audio in {audio_directory}")
    audio_items = analyze_audio_dir(
        audio_directory,
        pattern=audio_cfg.get("glob") or "*.wav",
        files=audio_cfg.get("files") or None,
        max_files=int(audio_cfg.get("max_files") or 0),
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
    video_info: dict[str, Any] = {"segments": [], "path": source_media}
    for source_media in sources:
        _log(log, f"Analyzing selected source video {source_media}")
        video_info = analyze_video(
            source_media,
            sample_fps=float(video_cfg.get("sample_fps") or 6),
            max_seconds=float(video_cfg.get("max_seconds") or 0),
            min_segment_sec=float(video_cfg.get("min_segment_sec") or 0.25),
            merge_gap_sec=float(video_cfg.get("merge_gap_sec") or 0.20),
            backend=str(video_cfg.get("backend") or "auto"),
            progress=lambda m: _log(log, "  " + m),
            cache_dir=out / "video_cache",
            source_kind=source_kind,
            main_group=character_cfg["main_group"],
        )
        video_info.setdefault('path',source_media)
        if character_cfg['enabled']:
            from autoedit.video.objects import augment_with_objects
            video_info = augment_with_objects(video_info,out / 'video_cache',video_cfg.get('object_detection'),
                                              progress=lambda m: _log(log,'  '+m))
        video_info = analyze_characters(video_info, out / "video_cache", character_cfg,
                                        progress=lambda m: _log(log, "  " + m))
        video_info.setdefault("path", source_media)
        source_infos.append(video_info)
        _log(log, f"  segments={len(video_info.get('segments') or [])} backends={video_info.get('backend_counts')}")
    source_media = sources[0]
    video_info = combine_video_infos(source_infos, gap)
    if video_info.get("character_analysis"):
        (out / "character_analysis.json").write_text(
            json.dumps(video_info["character_analysis"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "video_analysis.json").write_text(
        json.dumps(video_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if len(sources) > 1:
        notes.append(f"Selecting from {len(sources)} videos; source gaps apply within each video. Cast ranks are analyzed independently per source.")

    video_info["source_kind"] = source_kind
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
    visual_matches = None
    if intro and intro["status"] == "empty":
        visual_matches = match_slots_to_video(slots, [], video_info, min_gap_sec=gap,
                                             require_lip_motion=require_lip_motion)
        try:
            extra = match_slots_to_video([intro], [], video_info, min_gap_sec=gap,
                require_speaking=False, excluded_ranges=[(m['video']['in_sec'],m['video']['out_sec'])
                                                         for m in visual_matches])[0]['video']
        except InsufficientFootageError:
            # An alternative initial reservation can still make the full job
            # feasible; do not discard it merely because this layout ran out.
            extra = match_slots_to_video([intro], [], video_info, min_gap_sec=gap,
                                        require_speaking=False)[0]['video']
            visual_matches = None
        excluded = [(extra["in_sec"], extra["out_sec"])]
        _log(log, f"Intro reserved at {extra['in_sec']:.3f}-{extra['out_sec']:.3f}s before sound-led cuts.")
    _log(log, "Selection mode: " + ("verified moving mouths" if require_lip_motion else
         "clear character shots; mouth evidence controls audio review, not footage eligibility"))
    _log(log, "Checking full-duration person evidence before source-audio analysis.")
    if visual_matches is None:
        visual_matches = match_slots_to_video(slots, [], video_info, min_gap_sec=gap,
                                             excluded_ranges=excluded, require_lip_motion=require_lip_motion)
    speech_by_source = {}
    for source_info in source_infos:
        analyzed = min(float(source_info["duration_sec"]),
                       float(source_info.get("analyzed_seconds") or source_info["duration_sec"]))
        _log(log, f"Scanning source audio {source_info['path']} across 0-{analyzed:.3f}s before selecting sound-led cuts.")
        speech_by_source[source_info["path"]] = transcribe_source(source_info["path"], [(0.0, analyzed)], out / "video_cache",
                               model=video_cfg.get("speech_model") or "small",
                               language=video_cfg.get("speech_language", "en"),
                               device=video_cfg.get("speech_device", "auto"),
                               compute_type=video_cfg.get("speech_compute_type", "auto"),
                               device_index=video_cfg.get("speech_device_index", 0),
                               progress=lambda m: _log(log, "  " + m))
    speech = speech_by_source[source_infos[0]["path"]]
    if video_info.get("source_spans"):
        speech = {"model": video_cfg.get("speech_model") or "small", "words": [], "events": [],
                  "sources": speech_by_source}
        for span in video_info["source_spans"]:
            shifted = shift_times(speech_by_source[span["path"]], span["offset_sec"])
            for field in ("words", "events"):
                speech[field].extend({**item, "source_video": span["path"]} for item in shifted.get(field, []))
    (out / "source_transcript.json").write_text(json.dumps(speech, indent=2, ensure_ascii=False), encoding="utf-8")
    video_info["source_speech"] = speech
    video_info["sound_anchors"] = build_sound_anchors(audio_items, speech)
    (out / "source_sound_matches.json").write_text(
        json.dumps(video_info["sound_anchors"], indent=2, ensure_ascii=False), encoding="utf-8")
    _log(log, f"  library-matched sound onsets={len(video_info['sound_anchors'])}; retaining slot lengths and source gaps")
    _log(log, f"  high-confidence onsets={sum(a['confidence_tier'] == 1 for a in video_info['sound_anchors'])}; "
              f"remaining supported onsets need listening review")
    matches = (match_slots_to_video(slots, [], video_info, min_gap_sec=gap,
                                    excluded_ranges=excluded, require_lip_motion=require_lip_motion)
               if video_info["sound_anchors"] else visual_matches)
    notes.append("Cuts prefer exposure-ranked actors, then library-matched sound onsets within each rank; each scene uses the WAV matching its opening sound. Unmatched speaking cuts use measured visemes/rhythm and require review. Legacy unvoiced selection may use neutral fallback.")
    for slot, match in zip(slots, matches):
        selected = match["video"]
        metrics = selected.get("selection_metrics") or {}
        _log(log, f"  clip {slot.get('nested_sequence')}: source {selected['in_sec']:.3f}-{selected['out_sec']:.3f}s "
                  f"tier={metrics.get('tier')} clear_face={metrics.get('clear_face', 0):.2f} speaking={metrics.get('speaking', 0):.2f}")
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
    video_only_sources = {info["path"]: prepare_video_only_source(
        info["path"], out / "video_only", progress=lambda m: _log(log, "  " + m)) for info in source_infos}
    plan["project"]["video_only_source"] = video_only_sources[source_infos[0]["path"]]
    plan["audio_mode"] = "sampler_per_scene_v1"
    from autoedit.video.faces import lip_motion_evidence
    rejected_ranges=[]
    for attempt in range(4):
        readable_slots = [s for s in plan["slots"] if s["video"]["selection_metrics"]["tier"] <= 2
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
            refined = (refine_selected_cuts(source_info["path"], source_slots, out / "video_cache", source_kind=source_kind,
                                       main_group=character_cfg["main_group"],
                                       character_samples=refs if source_info.get("character_analysis") else None,
                                       progress=lambda m: _log(log, "  " + m))
                    if source_slots and video_cfg.get("backend") != "opencv" else {"samples": []})
            lip_info["samples"].extend(shift_times(refined.get("samples", []), offset))
        rejected = []
        for slot in plan["slots"]:
            cut = slot["video"]
            if cut["selection_metrics"].get("require_lip_motion"):
                window = [s for s in lip_info.get("samples", [])
                          if cut["in_sec"] <= s["time_sec"] < cut["out_sec"] and s.get("clear_face")]
                if not lip_motion_evidence(window):
                    rejected.append(slot)
                else:
                    cut["selection_metrics"]["dense_lip_motion_verified"] = True
        if not rejected:
            break
        if attempt == 3:
            raise ValueError("Selected cuts have no verified lip movement in dense analysis; "
                             "cannot assign voice samples to these static/unclear faces.")
        _log(log, "Dense lip check rejected " + ", ".join(s["id"] for s in rejected)
             + "; selecting alternative moving-mouth cuts with the same slot lengths and source gap.")
        rejected_ranges.extend((s["video"]["in_sec"], s["video"]["out_sec"]) for s in rejected)
        matches = match_slots_to_video(slots, [], video_info, min_gap_sec=gap,
                                       excluded_ranges=excluded, require_lip_motion=require_lip_motion,
                                       rejected_ranges=rejected_ranges)
        plan["slots"] = build_edit_plan(config=config, slots=slots, matches=matches,
                                       video_info=video_info, notes=notes)["slots"]
    for slot in plan["slots"]:
        cut = slot["video"]
        _log(log, f"  final clip {slot['premiere']['nested_sequence']}: "
                  f"{cut['in_sec']:.3f}-{cut['out_sec']:.3f}s "
                  f"dense_lip_motion={cut['selection_metrics'].get('dense_lip_motion_verified', False)}")
    lip_info.update(source_kind=source_kind, source_speech=speech)
    if video_info.get("source_spans"):
        lip_info["source_spans"] = video_info["source_spans"]
    (out / "lip_analysis.json").write_text(json.dumps(lip_info), encoding="utf-8")
    plan["cubase_slots"] = build_sampler_slots(scenes, plan["slots"], audio_items, lip_info, inventory)
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
    cubase_result = apply_cubase_plan(plan, out)
    # Listening review uses the actual source soundtrack and generated samples.
    write_audio_review(plan, out, speech)
    _log(log, "Listening review: " + str(out / "audio_review.html"))
    audio_errors = [row["error"] for row in cubase_result.get("tracks", []) if row.get("error")]
    if audio_errors:
        raise ValueError("Audio mixdown incomplete: " + "; ".join(audio_errors[:5]))
    plan["cubase_manifest"] = cubase_result.get("manifest")
    errors = validate_plan(plan)
    if errors:
        plan.setdefault("notes", []).extend(f"validate: {e}" for e in errors)
    plan_path = save_plan(plan, out / "edit-plan.json")
    write_apply_payload(plan, out / "premiere_apply.json")
    _log(log, f"Wrote {plan_path}")
    return {"plan_path": str(plan_path), "plan": plan, "inspect": inspect}
