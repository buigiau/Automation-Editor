"""Prepare the payload the UXP plugin applies inside Premiere.

Python does not talk to Premiere directly. This module writes
premiere_apply.json next to edit-plan.json so the panel can load a
smaller, host-specific document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def apply_payload(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("operation_scope") == "sample_assignment_only" or plan.get("superseded_by"):
        raise ValueError("Sample-only or superseded plans cannot modify Premiere. Use Import Cubase only.")
    project = plan.get("project") or {}
    actions = []
    for slot in plan.get("slots") or []:
        video = slot.get("video") or {}
        premiere = slot.get("premiere") or {}
        if not video.get("in_sec") and video.get("in_sec") != 0:
            continue
        mode = premiere.get("mode") or "nested_sequence"
        if mode != "nested_sequence" or not premiere.get("nested_sequence_uid"):
            raise ValueError("Regenerate the plan: only identified nested sequences can be filled.")
        at = 0.0 if mode == "nested_sequence" else float(premiere.get("timeline_start_sec") or 0.0)
        vtrack = 0 if mode == "nested_sequence" else int(premiere.get("video_track_index") or 1)
        actions.append(
            {
                "slot_id": slot.get("id"),
                "mode": mode,
                "nested_sequence": premiere.get("nested_sequence"),
                "nested_sequence_uid": premiere.get("nested_sequence_uid"),
                "slot_duration_sec": premiere.get("duration_sec"),
                "instances": premiere.get("instances") or [],
                "template_video_track_index": premiere.get("video_track_index", 1),
                "template_sequence": project.get("template_sequence"),
                "source_media": project.get("source_media"),
                "in_sec": video.get("in_sec"),
                "out_sec": video.get("out_sec"),
                "overwrite_at_sec": at,
                "video_track_index": vtrack,
                "audio_track_index": -1,
                "character_selection": video.get("character_selection"),
            }
        )
    mix = plan.get("mixdown") or {}
    source_video = project.get("source_video") or project.get("source_media") or ""
    return {
        "fill_policy": "nested_sequences_v2",
        "source_duration_sec": (plan.get("video_analysis_meta") or {}).get("duration_sec"),
        "template_sequence": project.get("template_sequence") or "PJ 5 - demo",
        "source_video": source_video,
        "source_media": source_video,
        "video_only_source": project.get("video_only_source"),
        "template_audio_tracks": plan.get("template_audio_tracks"),
        "import_source_video": bool(project.get("import_source_video", True)),
        "save_project": False,
        "mixdown_wav": mix.get("path"),
        "import_mixdown": bool(project.get("import_mixdown", True)),
        "mixdown_duration_sec": mix.get("duration_sec"),
        "cubase_project": project.get("cubase_project") or "",
        "premiere_project": project.get("premiere_project") or "",
        "fill_nested_sequences": actions,
        "intro_fill": plan.get("intro_fill"),
        "overwrite_template_clips": [a for a in actions if a.get("mode") == "timeline_clip"],
    }


def write_apply_payload(plan: dict[str, Any], path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(apply_payload(plan), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest
