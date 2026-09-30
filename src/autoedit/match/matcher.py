"""Match each audio segment to one unused video segment of the same category."""

from __future__ import annotations

from typing import Any
import math
from bisect import bisect_right


# Video analysis still emits HAHA for a laugh. Audio analysis emits LAUGH.
# U has no dedicated video class, so O is the nearest mouth shape.
FAMILY: dict[str, tuple[str, ...]] = {
    "A": ("A",),
    "E": ("E",),
    "O": ("O", "U"),
    "U": ("U", "O"),
    "LAUGH": ("LAUGH", "HAHA"),
    "HAHA": ("HAHA", "LAUGH"),
    "CRY": ("CRY",),
    "CLOSED": ("CLOSED",),
    "NEUTRAL": ("NEUTRAL",),
}

FALLBACK_ORDER = ("A", "E", "O", "U", "HAHA", "LAUGH", "NEUTRAL", "CLOSED", "CRY")


def _has_face(sample):
    return bool(sample.get("face_detected", sample.get("face_visible", 0))
                and sample.get("backend", "mediapipe-landmarks") == "mediapipe-landmarks")


def _has_body(sample):
    # Legacy pose boxes have no independent detection/anatomy confirmation.
    return bool(sample.get("pose_confirmed") and sample.get("person_detected"))


def shot_tier(metrics, samples, first, last):
    """Best evidenced tier for the whole cut, including its bracketing samples."""
    from autoedit.video.faces import _same_face
    if not samples or metrics.get("sample_coverage", 1) < 0.999:
        return None
    window = samples[first:last + 1]
    # All four tiers require a person at every sampled point and both brackets.
    # Bodies need independent pose confirmation; low motion alone is never enough.
    if not window or not all(_has_face(s) or _has_body(s) for s in window):
        return None
    continuous = metrics.get("face_continuous")
    if continuous is None:
        continuous = all(not (a.get("face") and b.get("face")) or _same_face(a, b)
                         for a, b in zip(window, window[1:]))
    if (continuous and metrics.get("face_visible", 0) >= .9
            and metrics.get("clear_face", 0) >= 0.85 and metrics.get("speaking", 0) >= 0.35
            and samples[first].get("clear_face") and samples[last].get("clear_face")):
        return 1
    face_throughout = metrics.get("face_visible", 0) >= .999 and all(_has_face(s) for s in window)
    if continuous and face_throughout and metrics.get("closeup_score", 0) >= 0.25:
        return 2
    # Body shots include upper bodies. Small but continuous faces also confirm
    # a wider character shot without requiring readable lips.
    if metrics.get("body_visible", 0) >= .8 or (continuous and face_throughout):
        return 3
    if (metrics.get("activity_score", 1) <= .25
            and all(s.get("activity_score", 1) <= .25 for s in window)):
        return 4
    return None


def match_slots_to_video(slots, audio_items, video_info, min_gap_sec=0.0, require_speaking=True, excluded_ranges=(), min_tier: int = 1):
    """Retry the entire allocation with main, uncertain, then supporting footage."""
    levels = (0, 1, 2) if video_info.get("character_analysis") else (None,)
    last_error = None
    for level in levels:
        try:
            return _match_slots_to_video(slots, audio_items, video_info, min_gap_sec, require_speaking,
                                         excluded_ranges, min_tier, level)
        except InsufficientFootageError as exc:
            last_error = exc
    raise last_error


class InsufficientFootageError(ValueError):
    """A valid allocation could not fill every slot with the available candidates."""


def _match_slots_to_video(slots, audio_items, video_info, min_gap_sec=0.0, require_speaking=True,
                          excluded_ranges=(), min_tier: int = 1, character_level=None):
    """Rank speaking faces, closeups, body shots, then stable shots with people.

    require_speaking=False skips the speaking preference, never person evidence.
    """
    import numpy as np
    from autoedit.video.faces import _same_face
    from autoedit.video.characters import cut_character_metrics

    duration = float(video_info.get("duration_sec") or 0)
    from autoedit.video.transitions import transition_metrics
    if video_info.get("cache_identity") is not None or video_info.get("transition_method"):
        from autoedit.video.analyzer import ANALYSIS_VERSION
        version = (video_info.get("cache_identity") or {}).get("version", ANALYSIS_VERSION)
        if version != ANALYSIS_VERSION or "transition_times_sec" not in video_info:
            raise ValueError("Video analysis has no current transition check. Restart AutoEdit and regenerate the plan.")
    transitions = sorted(video_info.get("transition_times_sec") or [])
    analyzed = float(video_info.get("analyzed_seconds") or duration)
    duration = min(duration, analyzed)
    gap = float(min_gap_sec)
    if min_tier not in (1, 2, 3, 4):
        raise ValueError("min_tier must be between 1 and 4; scenery is not allowed.")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Source video duration is missing or invalid.")
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("Source gap must be a non-negative number of seconds.")
    needs = [float(s.get("required_duration_sec") or s.get("max_instance_duration_sec") or 0) for s in slots]
    if any(not math.isfinite(n) or n <= 0 for n in needs):
        raise ValueError("Invalid nested clip duration.")
    if sum(needs) + gap * max(0, len(slots) - 1) > duration + 1e-8:
        raise ValueError(f"Not enough analyzed source footage for full cuts with {gap:g}s gaps. Analyze more video or reduce Source gap.")

    samples = sorted(video_info.get("samples") or [], key=lambda s: s["time_sec"])
    times = np.array([s["time_sec"] for s in samples])
    fields = ("face_visible", "closeup_score", "mouth_clarity", "clear_face", "speaking",
              "person_visible", "full_body_visible", "activity_score", "sample_coverage", "body_visible")
    def sample_value(s, key):
        if key == "sample_coverage":
            return 1.0
        if key == "face_visible":
            return float(s.get("face_visible", s.get("face_detected", 0)))
        if key == "person_visible":
            return float(s.get("person_visible", s.get("person_detected", 0)))
        if key == "body_visible":
            return float(_has_body(s))
        if key == "full_body_visible":
            return float(bool(s.get("person_detected", s.get("person_visible"))) and s.get("shot_scale") == "full_body")
        return float(s.get(key, 1 if key == "activity_score" else 0) or 0)
    values = np.array([[sample_value(s, k) for k in fields] for s in samples]).reshape(-1, len(fields))
    sample_interval = 1 / float(video_info.get("sample_fps") or 2)
    if len(times) > 1 and not video_info.get("sample_fps"):
        sample_interval = float(np.median(np.diff(times)))
    deltas = np.maximum(0, np.diff(np.append(times, duration))) if len(times) else np.array([])
    video_fps = float(video_info.get("fps") or 0)
    frame_slack = 1 / video_fps if math.isfinite(video_fps) and video_fps > 0 else 0
    # Requested sample times land on real frames: at 23.976 fps, a 10 Hz
    # sampler alternates 0.0834/0.1251s gaps. Those are not missing samples.
    # Longer holes still only carry one nominal sample's worth of evidence.
    weights = np.where(deltas <= sample_interval + frame_slack + 1e-8,
                       deltas, np.minimum(deltas, sample_interval))
    prefix = np.vstack([np.zeros(len(fields)), np.cumsum(values * weights[:, None], axis=0)])
    # Bracket the end of a cut too: a last good sample must not mask a shot
    # change in the unsampled tail. Large face jumps also break continuity.
    breaks = [0] + [int(bool(a.get("face") and b.get("face")) and (
                       not _same_face(a, b) or a.get("track_id") != b.get("track_id")))
                    for a, b in zip(samples, samples[1:])]
    break_prefix = np.cumsum(breaks)

    def integrated_quality(time):
        j = int(np.searchsorted(times, time, side="right")) - 1
        if j < 0:
            return prefix[0]
        return prefix[j] + values[j] * min(max(0, time - times[j]), weights[j])
    segments = sorted(video_info.get("segments") or [], key=lambda s: s["start_sec"])
    segment_starts = [s["start_sec"] for s in segments]
    # Explore starts across the source; strict qualification below gates every cut.
    anchors = {float(s["time_sec"]) for s in samples}
    anchors.update(float(s["start_sec"]) for s in segments)
    anchors.update(t + .16 for t in transitions)
    anchors.update(float(t) for t in np.linspace(0, duration, min(2001, int(duration) + 2)))
    free = [(0.0, duration)]
    used = []
    matches = [None] * len(slots)
    # Reserve long uninterrupted windows before short cuts consume them.
    # Keep original indices for audio, source spread and returned timeline order.
    selection_order = sorted(range(len(slots)), key=lambda index: (-needs[index], index))

    def subtract(intervals, start, end):
        result = []
        for left, right in intervals:
            if end <= left or start >= right:
                result.append((left, right))
            else:
                if start > left:
                    result.append((left, start))
                if end < right:
                    result.append((end, right))
        return result

    def can_fit(intervals, remaining):
        lengths = [right - left for left, right in intervals]
        for need in sorted(remaining, reverse=True):
            choices = [i for i, length in enumerate(lengths) if length + 1e-8 >= need]
            if not choices:
                return False
            j = min(choices, key=lambda i: lengths[i])
            lengths[j] = max(0, lengths[j] - need - gap)
        return True

    for start, end in excluded_ranges:
        free = subtract(free, float(start) - gap, float(end) + gap)
        used.append((float(start), float(end)))

    for step, i in enumerate(selection_order):
        slot = slots[i]
        need = needs[i]
        audio = audio_items[i] if i < len(audio_items) else None
        target = (i + 0.5) * duration / max(1, len(slots))
        candidates = set()
        for left, right in free:
            if right - left + 1e-8 < need:
                continue
            candidates.update((left, max(left, right - need), max(left, min(target - need / 2, right - need))))
            candidates.update(max(left, min(t, right - need)) for t in anchors if left <= t < right)
        ranked = []
        for start in candidates:
            end = start + need
            continuity = transition_metrics(transitions, start, end)
            if not continuity["transition_safe"]:
                continue
            metrics = (integrated_quality(end) - integrated_quality(start)) / need
            measured = dict(zip(fields, map(float, metrics)))
            measured.update(continuity)
            first = min(len(samples) - 1, max(0, bisect_right(times, start) - 1))
            last = min(len(samples) - 1, max(0, bisect_right(times, end - 1e-8) - 1))
            after = min(len(samples) - 1, last + 1)
            if samples:
                measured["face_continuous"] = bool(break_prefix[after] == break_prefix[first])
            tier = shot_tier(measured, samples, first, after)
            if tier is None or tier < min_tier:
                continue
            if character_level is not None:
                cast = video_info["character_analysis"]
                character = cut_character_metrics(samples[first:after + 1], start, end,
                                                   sample_interval + frame_slack, cast["min_main_fraction"])
                level = {"main": 0, "uncertain": 1, "supporting": 2, "reject": 3}[character["decision"]]
                if level > character_level:
                    continue
                measured["character_selection"] = character
            measured["tier"] = tier
            measured["subject_evidence"] = ("face-throughout" if all(_has_face(s) for s in samples[first:after + 1])
                                            else "face-or-confirmed-body-throughout")
            quality = float(np.dot(metrics[:5], [1.5, 2.5, 3.0, 1.0, 2.0]))
            segment_index = bisect_right(segment_starts, (start + end) / 2) - 1
            seg = segments[segment_index] if segment_index >= 0 else {}
            overlaps = seg and start < seg["end_sec"] and end > seg["start_sec"]
            if not overlaps:
                seg = {}
            category_bonus = 0.3 if same_family((audio or {}).get("category"), seg.get("category")) else 0
            # Prefer different parts of the source, rather than neighbouring good frames.
            spread = 1 - abs((start + end) / 2 - target) / duration
            separation = min((max(a - end, start - b, 0) for a, b in used), default=duration)
            diversity = min(1, separation / max(duration / max(1, len(slots)), 1))
            score = quality + category_bonus + 0.8 * spread + 0.8 * diversity
            ranked.append((score, start, seg, measured))
        ranked.sort(key=lambda row: (
            {"main": 0, "uncertain": 1, "supporting": 2}.get(
                row[3].get("character_selection", {}).get("decision"), 0),
            row[3]["transition_count"], row[3]["tier"] if require_speaking else max(2, row[3]["tier"]),
            row[3].get("character_selection", {}).get("best_rank") or 1000000, -row[0], row[1]))
        chosen = None
        for score, start, seg, metrics in ranked:
            remaining = subtract(free, start - gap, start + need + gap)
            if can_fit(remaining, [needs[j] for j in selection_order[step + 1:]]):
                chosen = (score, start, seg, metrics, remaining)
                break
        if chosen is None:
            raise InsufficientFootageError(f"Not enough qualifying footage with a visible person throughout (tiers {min_tier}-4): selected {step}/{len(slots)} clips; {slot['id']} needs {need:.3f}s with a {gap:g}s source gap and a stable opening without transitions. Longest cuts were reserved first. Scenery and unconfirmed pose detections are excluded. Noise-only character cuts are excluded.")
        score, start, seg, metrics, free = chosen
        end = start + need
        used.append((start, end))
        matches[i] = {"audio": audio, "video": {**seg, "in_sec": start, "out_sec": end,
                        "used_duration_sec": need, "selection_metrics": metrics,
                        **({"character_selection": metrics["character_selection"]}
                           if "character_selection" in metrics else {}),
                        "source_gap_sec": gap}, "score": round(score, 4),
                        "status": "matched" if same_family((audio or {}).get("category"), seg.get("category")) else "fallback"}
    return matches


def same_family(audio_cat: str | None, video_cat: str | None) -> bool:
    if not audio_cat or not video_cat:
        return False
    audio_cat = audio_cat.removeprefix("VOWEL_")
    video_cat = video_cat.removeprefix("VOWEL_")
    if audio_cat == video_cat:
        return True
    return video_cat in FAMILY.get(audio_cat, ()) or audio_cat in FAMILY.get(video_cat, ())


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def _score(audio: dict[str, Any], seg: dict[str, Any]) -> float:
    cat_bonus = 1.0 if same_family(audio.get("category"), seg.get("category")) else 0.15
    need = float(audio.get("duration_sec") or 0.4)
    have = float(seg.get("duration_sec") or 0.0)
    if have <= 0:
        return 0.0
    # Prefer a segment at least as long as the audio, not wildly longer
    if have < need * 0.6:
        length = have / max(need, 1e-6) * 0.5
    else:
        length = max(0.2, 1.0 - abs(have - need) / max(need * 3.0, 1.0))
    conf = float(seg.get("confidence") or 0.0) * 0.4 + float(audio.get("confidence") or 0.0) * 0.1
    return cat_bonus * 2.0 + length + conf


def match_audio_to_video(
    audio_items: list[dict[str, Any]],
    video_segments: list[dict[str, Any]],
    min_score: float = 0.4,
) -> list[dict[str, Any]]:
    """Greedy one-to-one matching. Video time ranges are not reused."""
    remaining = [dict(s) for s in video_segments]
    matches: list[dict[str, Any]] = []
    used_ranges: list[tuple[float, float]] = []

    for audio in audio_items:
        best_i = -1
        best_s = -1.0
        want = audio.get("category")
        preferred = list(FAMILY.get(want or "", (want,)))
        candidates = list(enumerate(remaining))
        ordered_cats = preferred + [c for c in FALLBACK_ORDER if c not in preferred]
        for cat in ordered_cats:
            pool = [(i, s) for i, s in candidates if s.get("category") == cat]
            if not pool:
                continue
            for i, seg in pool:
                a0, a1 = float(seg["start_sec"]), float(seg["end_sec"])
                if any(_overlap(a0, a1, u0, u1) for u0, u1 in used_ranges):
                    continue
                sc = _score(audio, seg)
                if not same_family(want, cat):
                    sc *= 0.45
                if sc > best_s:
                    best_s, best_i = sc, i
            if best_i >= 0 and same_family(want, cat):
                break
        if best_i < 0 or best_s < min_score:
            matches.append(
                {
                    "audio": audio,
                    "video": None,
                    "score": round(best_s, 3),
                    "status": "unmatched",
                }
            )
            continue
        seg = remaining.pop(best_i)
        need = float(audio.get("duration_sec") or 0.4)
        # Trim the chosen window to audio length, centered in the segment when possible
        span = float(seg["end_sec"]) - float(seg["start_sec"])
        take = min(span, max(need, 0.2))
        extra = span - take
        in_sec = float(seg["start_sec"]) + extra * 0.25
        out_sec = in_sec + take
        used_ranges.append((in_sec, out_sec))
        video = {
            **seg,
            "in_sec": round(in_sec, 3),
            "out_sec": round(out_sec, 3),
            "used_duration_sec": round(take, 3),
        }
        matches.append(
            {
                "audio": audio,
                "video": video,
                "score": round(best_s, 3),
                "status": "matched" if same_family(want, video.get("category")) else "fallback",
            }
        )
    return matches
