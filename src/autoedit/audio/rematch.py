"""Replace voice assignments while keeping all existing scene/track bindings."""
import copy
from collections import Counter

from autoedit.audio.scene_match import choose_audio
from autoedit.video.pool import shift_times


def rematch_fixed_slots(plan, candidates, video_info):
    if plan.get("superseded_by") or plan.get("source_word_cut_alignment"):
        raise ValueError("Use the original saved run with unchanged cuts, not the superseded cut-aligned plan.")
    slots = plan.get("cubase_slots") or []
    if not slots:
        raise ValueError("Existing scene-to-track assignments are required; no tracks will be created.")
    result = copy.deepcopy(plan)
    selected, used, recent = {}, Counter(), []
    for slot in result["cubase_slots"]:
        uid = slot["premiere"]["nested_sequence_uid"]
        if uid not in selected:
            cut = slot["video"]
            offset = 0.0
            if video_info.get("source_spans"):
                span = next((s for s in video_info["source_spans"] if s["path"] == cut.get("source_video")), None)
                if span is None:
                    raise ValueError("Saved cut source is missing from the source pool; run the full pipeline again.")
                offset = span["offset_sec"]
                cut = shift_times(cut, offset)
            audio, evidence = choose_audio(candidates, cut, video_info, used, recent=recent)
            selected[uid] = audio, shift_times(evidence, -offset)
            used[selected[uid][0]["path"]] += 1
        audio, evidence = selected[uid]
        slot["audio"], slot["audio_match"] = copy.deepcopy(audio), copy.deepcopy(evidence)
        recent.append(audio)
    result["audio"] = [slot["audio"] for slot in result["cubase_slots"]]
    result["operation_scope"] = "sample_assignment_only"
    return result


def verify_fixed_layout(before, after):
    for field in ("project", "slots", "intro_fill", "template_audio_tracks", "video_segments"):
        if before.get(field) != after.get(field):
            raise ValueError(f"Sample-only operation changed protected plan field: {field}")
    old, new = before["cubase_slots"], after["cubase_slots"]
    if len(old) != len(new):
        raise ValueError("Sample-only operation changed track count")
    generated = {"sample_path", "sample_validation", "sample_render_version", "timing_mode",
                 "source_onset_applied_to_sampler"}
    for a, b in zip(old, new):
        for field in ("id", "index", "premiere", "video"):
            if a.get(field) != b.get(field):
                raise ValueError(f"Sample-only operation changed track order or scene timing: {field}")
        if ({k: v for k, v in a["cubase"].items() if k not in generated} !=
                {k: v for k, v in b["cubase"].items() if k not in generated}):
            raise ValueError("Sample-only operation changed Cubase track configuration")
    return {"premiere_plan_unchanged": True, "source_cuts_unchanged": True,
            "scene_timing_unchanged": True, "cubase_track_bindings_unchanged": True,
            "verified_tracks": len(old)}
