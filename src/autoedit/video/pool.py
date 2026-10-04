"""Pool independent sources for selection, preserving their original time bases."""
import math


TIME_KEYS = {"time_sec", "start_sec", "end_sec", "in_sec", "out_sec", "start", "end",
             "source_word_start_sec", "source_word_end_sec", "feature_anchor_sec"}
ID_KEYS = {"track_id", "character_id", "shot_id", "subject_track_id"}


def bounded_segments(segments, end_sec, min_sec=0.0):
    """Limit analysis metadata to decoded footage, including older cached segments."""
    result = []
    for segment in segments:
        start = max(0.0, float(segment["start_sec"]))
        end = min(float(segment["end_sec"]), end_sec)
        duration = end - start
        if duration > 0 and duration + 1e-8 >= min_sec:
            result.append({**segment, "start_sec": start, "end_sec": end,
                           "duration_sec": duration})
    return result


def shift_times(value, offset, prefix=None):
    if isinstance(value, list):
        return [shift_times(item, offset, prefix) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in TIME_KEYS and isinstance(item, (int, float)):
            result[key] = item + offset
        elif prefix and key in ID_KEYS and item is not None:
            result[key] = f"{prefix}:{item}"
        else:
            result[key] = shift_times(item, offset, prefix)
    return result


def combine_video_infos(infos, gap):
    if len(infos) == 1:
        info = infos[0]
        duration = min(float(info["duration_sec"]),
                       float(info.get("analyzed_seconds") or info["duration_sec"]))
        return {**info, "segments": bounded_segments(info.get("segments", []), duration)}
    pool = {**infos[0], "samples": [], "segments": [], "transition_times_sec": [],
            "shot_times_sec": [], "source_spans": [], "backend_counts": {}}
    # The separator keeps source-gap reservations within the originating file.
    separator = max(1.0, float(gap) + 1.0)
    offset = 0.0
    for index, info in enumerate(infos):
        duration = min(float(info["duration_sec"]),
                       float(info.get("analyzed_seconds") or info["duration_sec"]))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid source duration: {info['path']}")
        pool["source_spans"].append({"path": info["path"], "offset_sec": offset,
                                     "end_sec": offset + duration,
                                     "duration_sec": info["duration_sec"], "analyzed_seconds": duration})
        pool["samples"].extend(shift_times(info.get("samples", []), offset, f"source-{index}"))
        pool["segments"].extend(shift_times(
            bounded_segments(info.get("segments", []), duration), offset, f"source-{index}"))
        for field in ("transition_times_sec", "shot_times_sec"):
            pool[field].extend(t + offset for t in info.get(field, []))
        for backend, count in info.get("backend_counts", {}).items():
            pool["backend_counts"][backend] = pool["backend_counts"].get(backend, 0) + count
        offset += duration + separator
    pool.update(duration_sec=offset-separator, analyzed_seconds=offset-separator,
                sample_count=len(pool["samples"]), fps=min(float(i.get("fps") or 0) for i in infos))
    if any(i.get("character_analysis") for i in infos):
        # Each video's cast is ranked independently; identities never collide.
        pool["character_analysis"] = {
            "selection_policy": "per-source-exposure-ranked-cast",
            "min_main_fraction": next(i["character_analysis"]["min_main_fraction"]
                                      for i in infos if i.get("character_analysis")),
            "sources": [i.get("character_analysis") for i in infos]}
    return pool


def span_for_cut(pool, cut):
    for span in pool.get("source_spans", []):
        if span["offset_sec"]-1e-8 <= cut["in_sec"] and cut["out_sec"] <= span["end_sec"]+1e-8:
            return span
    raise ValueError("Selected cut crosses source videos or unanalyzed footage")


def local_cut(pool, cut):
    span = span_for_cut(pool, cut)
    return {**shift_times(cut, -span["offset_sec"]), "source_video": span["path"],
            "source_duration_sec": span["duration_sec"]}


def localize_plan(plan, pool, video_only_sources):
    """Convert only source times; Premiere timeline times stay unchanged."""
    if not pool.get("source_spans"):
        return
    for group in ("slots", "cubase_slots"):
        for slot in plan.get(group, []):
            cut = slot.get("video")
            if not cut:
                continue
            span = span_for_cut(pool, cut)
            slot["video"] = local_cut(pool, cut)
            slot["video"]["video_only_source"] = video_only_sources[span["path"]]
            slot.update(source_video=span["path"], source_start_sec=slot["video"]["in_sec"],
                        source_end_sec=slot["video"]["out_sec"])
            if slot.get("audio_match"):
                slot["audio_match"] = shift_times(slot["audio_match"], -span["offset_sec"])
    if plan.get("intro_fill"):
        intro = plan["intro_fill"]
        cut = local_cut(pool, {"in_sec": intro["in_sec"], "out_sec": intro["out_sec"]})
        intro.update(cut, video_only_source=video_only_sources[cut["source_video"]])
    plan["project"]["source_videos"] = [
        {**span, "video_only_source": video_only_sources[span["path"]]} for span in pool["source_spans"]]
    plan["video_analysis_meta"].update(
        duration_sec=sum(s["duration_sec"] for s in pool["source_spans"]),
        analyzed_seconds=sum(s["analyzed_seconds"] for s in pool["source_spans"]))
    segments = []
    for segment in plan.get("video_segments", []):
        span = span_for_cut(pool, {"in_sec": segment["start_sec"], "out_sec": segment["end_sec"]})
        segments.append({**shift_times(segment, -span["offset_sec"]), "source_video": span["path"]})
    plan["video_segments"] = segments
