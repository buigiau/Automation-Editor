"""Join matches, Premiere slots, and Cubase tracks into edit-plan.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autoedit.schema import empty_plan, utc_now


def build_edit_plan(
    *,
    config: dict[str, Any],
    slots: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    video_info: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    premiere_cfg = config.get("premiere") or {}
    cubase_cfg = config.get("cubase") or {}
    job = config.get("job") or {}
    pattern = cubase_cfg.get("track_name_pattern") or "Sampler Track {index:02d}"
    track_count = int(cubase_cfg.get("sampler_track_count") or 16)

    plan_slots: list[dict[str, Any]] = []
    n = len(slots)
    for i in range(n):
        slot = slots[i] if i < len(slots) else {}
        match = matches[i] if i < len(matches) else {}
        audio = match.get("audio")
        video = match.get("video")
        track_index = (i % track_count) + 1
        try:
            track_name = pattern.format(index=track_index)
        except Exception:
            track_name = f"Sampler Track {track_index:02d}"
        start = slot.get("first_start_sec") or 0.0
        slot_dur = slot.get("required_duration_sec") or slot.get("max_instance_duration_sec")
        if video and slot_dur:
            inn = float(video.get("in_sec") or 0)
            out = float(video.get("out_sec") or inn)
            if out - inn < float(slot_dur) - 1e-6:
                raise ValueError(f"Source range is shorter than nested clip {slot.get('id')}")
            video = {**video, "out_sec": inn + float(slot_dur), "used_duration_sec": float(slot_dur)}
        source_video = (video_info or {}).get("path") or premiere_cfg.get("source_media") or ""
        target_sequence = premiere_cfg.get("template_sequence") or "PJ 5 - demo"
        source_start = (video or {}).get("in_sec") if video else None
        source_end = (video or {}).get("out_sec") if video else None
        category = (audio or {}).get("category") or (video or {}).get("category")
        plan_slots.append(
            {
                "index": i,
                "order": i,
                "id": slot.get("id") or f"slot-{i+1:03d}",
                "source_video": source_video,
                "source_start_sec": source_start,
                "source_end_sec": source_end,
                "category": category,
                "target_sequence": target_sequence,
                "target_slot": slot.get("id") or f"slot-{i+1:03d}",
                "cubase_track": track_name,
                "premiere": {
                    "nested_sequence": slot.get("nested_sequence"),
                    "nested_sequence_uid": slot.get("nested_sequence_uid"),
                    "fill_video_track_index": slot.get("fill_video_track_index", 0),
                    "timeline_start_sec": slot.get("first_start_sec"),
                    "timeline_end_sec": slot.get("last_end_sec"),
                    "duration_sec": slot.get("required_duration_sec") or slot.get("max_instance_duration_sec"),
                    "occurrences": slot.get("occurrences", 0),
                    "video_track_index": premiere_cfg.get("video_track_index", 1),
                    "instances": slot.get("instances") or [],
                    "mode": slot.get("mode") or "nested_sequence",
                },
                "cubase": {
                    "track_name": track_name,
                    "track_index": track_index,
                    "start_sec": start,
                    "project": cubase_cfg.get("project") or "",
                },
                "audio": audio,
                "video": video,
                "match_status": match.get("status", "empty"),
                "match_score": match.get("score"),
            }
        )

    output_dir = Path(job.get("output_dir") or "./output")
    mix_path = str((output_dir / "mixdown.wav").resolve())
    source_video = (video_info or {}).get("path") or premiere_cfg.get("source_media") or ""

    plan = empty_plan(
        created_at=utc_now(),
        project={
            "premiere_project": premiere_cfg.get("project") or "",
            "source_sequence": premiere_cfg.get("source_sequence") or "",
            "template_sequence": premiere_cfg.get("template_sequence") or "PJ 5 - demo",
            "source_video": source_video,
            "source_media": source_video,
            "cubase_project": cubase_cfg.get("project") or "",
            "import_source_video": True,
            "save_project": False,
            "import_mixdown": bool(premiere_cfg.get("import_mixdown", True)),
        },
        audio=[m.get("audio") for m in matches if m.get("audio")],
        video_segments=(video_info or {}).get("segments") or [],
        slots=plan_slots,
        mixdown={
            "path": mix_path,
            "duration_sec": 0.0,
            "sample_rate": int(cubase_cfg.get("sample_rate") or 48000),
            "channels": int(cubase_cfg.get("channels") or 2),
        },
        notes=list(notes or []),
        video_analysis_meta={
            "duration_sec": (video_info or {}).get("duration_sec"),
            "analyzed_seconds": (video_info or {}).get("analyzed_seconds"),
            "backend_counts": (video_info or {}).get("backend_counts"),
            "sample_count": (video_info or {}).get("sample_count"),
            "source_kind": (video_info or {}).get("source_kind"),
            "quality_method": (video_info or {}).get("quality_method"),
            "analysis_version": ((video_info or {}).get("cache_identity") or {}).get("version"),
            "transition_method": (video_info or {}).get("transition_method"),
            "character_analysis": {k: v for k, v in ((video_info or {}).get("character_analysis") or {}).items()
                                   if k not in ("tracks",)},
        },
    )
    return plan
