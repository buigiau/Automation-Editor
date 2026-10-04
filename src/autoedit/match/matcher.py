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
                and (sample.get("backend", "mediapipe-landmarks") == "mediapipe-landmarks"
                     or (sample.get('backend') == 'feature-tracked-face'
                         and sample.get('face_evidence') == 'reversible-face-features'
                         and sample.get('verified_feature_count',0) >= 12)
                     or (sample.get("backend") == "animation-eyes-mouth" and
                         sample.get("face_evidence") in
                         {"paired-pupils-shared-colour-mouth", "goggle-pupil-yellow-mouth", "goggle-pupil-yellow-face"})))


def _has_body(sample):
    # Legacy pose boxes have no independent detection/anatomy confirmation.
    from autoedit.video.objects import object_evidence
    return bool(sample.get("pose_confirmed") and sample.get("person_detected")) or object_evidence(sample)


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
    mixed_moving_subject = (metrics.get('body_visible',0) >= .2
                            and metrics.get('face_visible',0) >= .2
                            and metrics.get('activity_score',0) > .25)
    if metrics.get("body_visible", 0) >= .8 or (continuous and face_throughout) or mixed_moving_subject:
        return 3
    if (metrics.get("activity_score", 1) <= .25
            and all(s.get("activity_score", 1) <= .25 for s in window)):
        return 4
    return None


def match_slots_to_video(slots, audio_items, video_info, min_gap_sec=0.0, require_speaking=True, excluded_ranges=(), min_tier: int = 1,
                         require_lip_motion=False, rejected_ranges=()):
    """Try a complete primary allocation, then retain exposure order on fallback."""
    character_analysis = video_info.get("character_analysis") or {}
    levels = ((0, 1), (2, None)) if character_analysis else ((None, None),)
    track_candidates = _face_track_candidates(slots, video_info, require_lip_motion, min_tier)
    last_error = None
    for level, rank_limit in levels:
        try:
            return _match_slots_to_video(slots, audio_items, video_info, min_gap_sec, require_speaking,
                                         excluded_ranges, min_tier, level, require_lip_motion,
                                         track_candidates=track_candidates, rejected_ranges=rejected_ranges,
                                         character_rank_limit=rank_limit)
        except InsufficientFootageError as exc:
            last_error = exc
            # Greedy quality ranking may fragment the qualifying windows even
            # when raw source time remains. Search those same verified windows
            # before allowing another character tier or losing sound onsets.
            try:
                return _match_slots_to_video(slots, audio_items, video_info, min_gap_sec, require_speaking,
                                             excluded_ranges, min_tier, level, require_lip_motion,
                                             allocation_search=True, track_candidates=track_candidates, rejected_ranges=rejected_ranges,
                                             character_rank_limit=rank_limit)
            except InsufficientFootageError as exc:
                last_error = exc
    if video_info.get("sound_anchors"):
        # A preferred onset can fragment valid footage. Retry the original
        # visual allocation rather than fail a job that can still be filled.
        return match_slots_to_video(slots, audio_items, dict(video_info, sound_anchors=[]),
                                    min_gap_sec, require_speaking, excluded_ranges, min_tier, require_lip_motion,
                                    rejected_ranges=rejected_ranges)
    raise last_error


class InsufficientFootageError(ValueError):
    """A valid allocation could not fill every slot with the available candidates."""


def _face_track_candidates(slots, video_info, require_lip_motion, min_tier):
    """Evaluate co-visible actors on their own uninterrupted, measured tracks.

    The prominent face is a preference, not the only face that can fill a cut.
    Missing observations, face jumps and unmeasured mouths are never repaired.
    """
    import numpy as np
    from collections import defaultdict
    from autoedit.video.faces import _same_face, lip_motion_evidence
    from autoedit.video.characters import cut_character_metrics
    from autoedit.video.transitions import transition_metrics
    samples = sorted(video_info.get('samples') or [], key=lambda s:s['time_sec'])
    if not samples or not any(s.get('faces') for s in samples):
        return {}
    needs = {float(s.get('required_duration_sec') or s.get('max_instance_duration_sec') or 0) for s in slots}
    if not needs or min(needs) <= 0:
        return {}
    interval = 1 / float(video_info.get('sample_fps') or 2)
    fps = float(video_info.get('fps') or 0)
    slack = 1 / fps if math.isfinite(fps) and fps > 0 else 0
    maximum_gap = interval + slack + 1e-8
    tracks = defaultdict(list)
    for index, sample in enumerate(samples):
        for face in sample.get('faces', []):
            if face.get('track_id') and (_has_face(face) or _has_body(face)):
                tracks[face['track_id']].append((index, {**face,'time_sec':sample['time_sec'],
                    'activity_score':sample.get('activity_score', 0)}))
    result = defaultdict(list)
    fields = ('face_visible','closeup_score','mouth_clarity','clear_face','speaking',
              'person_visible','full_body_visible','activity_score','sample_coverage','body_visible','subject_clarity')
    transitions = sorted(video_info.get('transition_times_sec') or [])
    sound_starts = [float(s['start']) for s in video_info.get('sound_anchors', [])]
    def evaluate(run):
        if len(run) < 2 or run[-1]['time_sec']-run[0]['time_sec'] < min(needs)-1e-8:
            return
        times = np.array([s['time_sec'] for s in run])
        values = np.array([[float(_has_face(s)),s.get('closeup_score',0),s.get('mouth_clarity',0),
                            s.get('clear_face',0),s.get('speaking',0),1,0,
                            s.get('activity_score',0),1,float(_has_body(s)),s.get('subject_clarity',0)] for s in run])
        weights = np.diff(np.append(times,times[-1]))
        prefix = np.vstack([np.zeros(len(fields)),np.cumsum(values*weights[:,None],axis=0)])
        def integrated(t):
            index=max(0,int(np.searchsorted(times,t,side='right'))-1)
            return prefix[index]+values[index]*max(0,min(t-times[index],weights[index]))
        for need in needs:
            starts=set(float(t) for t in times if t+need <= times[-1]+1e-8)
            starts.update(t for t in sound_starts if times[0] <= t and t+need <= times[-1]+1e-8)
            for start in starts:
                end=start+need
                first=max(0,int(np.searchsorted(times,start,side='right'))-1)
                after=min(len(run)-1,int(np.searchsorted(times,end,side='left')))
                window=run[first:after+1]
                measured=dict(zip(fields,map(float,(integrated(end)-integrated(start))/need)))
                measured.update(transition_metrics(transitions,start,end),face_continuous=True)
                if not measured['transition_safe']:
                    continue
                if require_lip_motion:
                    opening_end=min(end,start+.4)
                    opening=(integrated(opening_end)-integrated(start))/(opening_end-start)
                    measured.update(require_lip_motion=True,opening_speaking_fraction=float(opening[4]))
                    if (measured['clear_face'] < .5 or measured['speaking'] < .35 or opening[4] < .35
                            or not lip_motion_evidence([s for s in window if start <= s['time_sec'] < end and s.get('clear_face')])):
                        continue
                    measured['lip_motion_within_cut']=True
                tier=shot_tier(measured,window,0,len(window)-1)
                if tier is None or tier < min_tier:
                    continue
                if video_info.get('character_analysis'):
                    measured['character_selection']=cut_character_metrics(window,start,end,maximum_gap,
                        video_info['character_analysis']['min_main_fraction'])
                    if measured['character_selection']['decision'] == 'reject':
                        continue
                measured.update(tier=tier,subject_evidence=('face-throughout' if all(_has_face(s) for s in window)
                    else 'independently-detected-character-throughout'),subject_track_id=window[0]['track_id'])
                categories=defaultdict(float)
                confidence=0.
                for offset, face in enumerate(window[:-1]):
                    weight=max(0,min(end,window[offset+1]['time_sec'])-max(start,face['time_sec']))
                    categories[face.get('category','NEUTRAL')]+=weight
                    confidence+=weight*float(face.get('confidence',0))
                measured['subject_category']=max(categories,key=categories.get,default='NEUTRAL')
                measured['subject_confidence']=confidence/need
                result[need].append((start,measured))
    for observations in tracks.values():
        run=[]
        previous_index=None
        for index,face in observations:
            if run and (index != previous_index+1 or face['time_sec']-run[-1]['time_sec'] > maximum_gap
                        or not _same_face(run[-1],face) or face.get('shot_id') != run[-1].get('shot_id')):
                evaluate(run)
                run=[]
            run.append(face)
            previous_index=index
        evaluate(run)
    return dict(result)


def _match_slots_to_video(slots, audio_items, video_info, min_gap_sec=0.0, require_speaking=True,
                          excluded_ranges=(), min_tier: int = 1, character_level=None, require_lip_motion=False,
                          allocation_search=False, track_candidates=None, rejected_ranges=(), character_rank_limit=None):
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
              "person_visible", "full_body_visible", "activity_score", "sample_coverage", "body_visible", "subject_clarity")
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
    sound_by_start = {float(s["start"]): s for s in video_info.get("sound_anchors", [])
                      if not (require_lip_motion and s.get("action") and s.get("prob", 0) < .85)}
    source_spans = video_info.get("source_spans") or []
    free = ([(s["offset_sec"], s["end_sec"]) for s in source_spans]
            if source_spans else [(0.0, duration)])
    used = []
    matches = [None] * len(slots)
    candidate_options = {}
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
    for start, end in rejected_ranges:
        # Failed candidates were never selected. Only their failed interval is
        # unavailable; they create neither a source gap nor diversity credit.
        free = subtract(free, float(start), float(end))

    def make_match(index, score, start, seg, metrics):
        need = needs[index]
        audio = audio_items[index] if index < len(audio_items) else None
        end = start + need
        if source_spans:
            from autoedit.video.pool import span_for_cut
            span = span_for_cut(video_info, {"in_sec": start, "out_sec": end})
            seg = {**seg, "source_video": span["path"]}
        sound = sound_by_start.get(start)
        if sound and sound["end"] > end:
            sound = None
        return {"audio": audio, "video": {**seg, "in_sec": start, "out_sec": end,
                    **({"source_sound": sound} if sound else {}),
                    "used_duration_sec": need, "selection_metrics": metrics,
                    **({"character_selection": metrics["character_selection"]}
                       if "character_selection" in metrics else {}),
                    "source_gap_sec": gap}, "score": round(score, 4),
                "status": "sound_matched" if sound else "matched" if same_family((audio or {}).get("category"), seg.get("category")) else "fallback"}

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
            # Never clamp a sound onset to fit a window: that would cut into
            # another sound while keeping the original WAV assignment.
            candidates.update(t for t, sound in sound_by_start.items()
                              if left <= t and t+need <= right+1e-8 and sound["end"] <= t+need)
        ranked = []
        for start in candidates:
            end = start + need
            continuity = transition_metrics(transitions, start, end)
            if not continuity["transition_safe"]:
                continue
            metrics = (integrated_quality(end) - integrated_quality(start)) / need
            measured = dict(zip(fields, map(float, metrics)))
            measured.update(continuity)
            if require_lip_motion:
                opening_end = min(end, start + .4)
                opening = (integrated_quality(opening_end) - integrated_quality(start)) / (opening_end-start)
                measured["opening_speaking_fraction"] = float(opening[fields.index("speaking")])
                measured["require_lip_motion"] = True
                if (measured["clear_face"] < .5 or measured["speaking"] < .35
                        or measured["opening_speaking_fraction"] < .35):
                    continue
            first = min(len(samples) - 1, max(0, bisect_right(times, start) - 1))
            last = min(len(samples) - 1, max(0, bisect_right(times, end - 1e-8) - 1))
            after = min(len(samples) - 1, last + 1)
            if samples:
                measured["face_continuous"] = bool(break_prefix[after] == break_prefix[first])
            if require_lip_motion:
                from autoedit.video.faces import lip_motion_evidence
                window = [s for s in samples[first:after+1]
                          if start <= s["time_sec"] < end and s.get("clear_face")]
                if not measured.get("face_continuous") or not lip_motion_evidence(window):
                    continue
                measured["lip_motion_within_cut"] = True
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
                if character_rank_limit is not None and sum(fraction for rank,fraction in character['rank_fractions'].items()
                        if rank <= character_rank_limit) < cast['min_main_fraction']:
                    continue
                measured["character_selection"] = character
            measured["tier"] = tier
            measured["subject_evidence"] = ("face-throughout" if all(_has_face(s) for s in samples[first:after + 1])
                                            else "face-or-confirmed-body-throughout")
            quality = float(np.dot(metrics[:5], [1.5, 2.5, 3.0, 1.0, 2.0])) + 2.5*measured['subject_clarity']
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
        for start, measured in (track_candidates or {}).get(need, []):
            end=start+need
            if not any(left-1e-8 <= start and end <= right+1e-8 for left,right in free):
                continue
            character=measured.get('character_selection')
            if character_level is not None and character and {
                    'main':0,'uncertain':1,'supporting':2,'reject':3}[character['decision']] > character_level:
                continue
            if character_rank_limit is not None and (not character or sum(fraction for rank,fraction in
                    character['rank_fractions'].items() if rank <= character_rank_limit)
                    < video_info['character_analysis']['min_main_fraction']):
                continue
            quality=sum(measured[k]*v for k,v in zip(fields[:5],[1.5,2.5,3.,1.,2.]))+2.5*measured['subject_clarity']
            segment_index=bisect_right(segment_starts,(start+end)/2)-1
            seg=segments[segment_index] if segment_index >= 0 else {}
            if not (seg and start < seg['end_sec'] and end > seg['start_sec']):
                seg={}
            seg={**seg,'category':measured['subject_category'],'confidence':measured['subject_confidence'],
                 'backend':'selected-face-track','start_sec':start,'end_sec':end,'duration_sec':need}
            category_bonus=.3 if same_family((audio or {}).get('category'),seg.get('category')) else 0
            spread=1-abs((start+end)/2-target)/duration
            separation=min((max(a-end,start-b,0) for a,b in used),default=duration)
            diversity=min(1,separation/max(duration/max(1,len(slots)),1))
            score=quality+category_bonus+.8*spread+.8*diversity
            ranked.append((score,start,seg,measured))
        def exposure_rank(row):
            character = row[3].get('character_selection', {})
            minimum = (video_info.get('character_analysis') or {}).get('min_main_fraction', .85)
            ranks = [int(rank) for rank,fraction in character.get('rank_fractions', {}).items() if fraction >= minimum]
            # A momentary appearance cannot promote the entire cut's rank.
            return min(ranks, default=1000000)
        ranked.sort(key=lambda row: (
            # A recognized primary view with uncertain identity still precedes
            # a known secondary actor; its uncertainty stays visible for review.
            exposure_rank(row),
            {"main": 0, "supporting": 1, "uncertain": 2}.get(
                row[3].get("character_selection", {}).get("decision"), 0),
            sound_by_start[row[1]].get("confidence_tier", 1)
            if row[1] in sound_by_start and sound_by_start[row[1]]["end"] <= row[1]+need else 3,
            row[3]["transition_count"], row[3]["tier"] if require_speaking else max(2, row[3]["tier"]),
            # Preserve moving-mouth footage for voiced scenes when filling an
            # explicitly unvoiced opening.
            (-1 if require_speaking else 1) * round(row[3].get("speaking", 0), 2),
            -row[0], row[1]))
        if allocation_search:
            candidate_options[i] = ranked
            continue
        chosen = None
        for score, start, seg, metrics in ranked:
            remaining = subtract(free, start - gap, start + need + gap)
            if can_fit(remaining, [needs[j] for j in selection_order[step + 1:]]):
                chosen = (score, start, seg, metrics, remaining)
                break
        if chosen is None:
            speech_note = " Visible lip movement is required in the opening and across the cut; silent/static faces are excluded." if require_lip_motion else ""
            raise InsufficientFootageError(f"Not enough qualifying footage with a visible person throughout (tiers {min_tier}-4): selected {step}/{len(slots)} clips; {slot['id']} needs {need:.3f}s with a {gap:g}s source gap and a stable opening without transitions. Longest cuts were reserved first. Scenery and unconfirmed pose detections are excluded. Noise-only character cuts are excluded." + speech_note)
        score, start, seg, metrics, free = chosen
        end = start + need
        used.append((start, end))
        matches[i] = make_match(i, score, start, seg, metrics)
    if allocation_search:
        # Forward checking starts with the slot with fewest available choices.
        # Cache failed states and bound exploration on highly ambiguous inputs.
        failed, nodes = set(), 0
        def search(remaining, intervals):
            nonlocal nodes
            if not remaining:
                return {}
            nodes += 1
            if nodes > 20000 or not can_fit(intervals, [needs[j] for j in remaining]):
                return None
            state = (remaining, tuple((round(a, 8), round(b, 8)) for a,b in intervals))
            if state in failed:
                return None
            available = {}
            for index in remaining:
                available[index] = [row for row in candidate_options[index]
                                    if any(left-1e-8 <= row[1] and row[1]+needs[index] <= right+1e-8
                                           for left,right in intervals)]
                if not available[index]:
                    failed.add(state)
                    return None
            # Raw free time overestimates capacity when only a few windows
            # contain speech. Interval scheduling gives an upper bound on how
            # many remaining verified cuts can fit, including their source gap.
            for minimum in sorted({needs[j] for j in remaining}):
                group = [j for j in remaining if needs[j] >= minimum]
                starts = sorted({row[1] for j in group for row in available[j]})
                count, previous_end = 0, -math.inf
                for start in starts:
                    if start >= previous_end+gap-1e-8:
                        count += 1
                        previous_end = start+minimum
                if count < len(group):
                    failed.add(state)
                    return None
            index = min(remaining, key=lambda j: (len(available[j]), -needs[j], j))
            following = tuple(j for j in remaining if j != index)
            for row in available[index]:
                start = row[1]
                result = search(following, subtract(intervals, start-gap, start+needs[index]+gap))
                if result is not None:
                    result[index] = row
                    return result
            failed.add(state)
            return None
        allocation = search(tuple(selection_order), free)
        if allocation is None:
            speech_note = " Visible lip movement is required; static faces are excluded." if require_lip_motion else ""
            raise InsufficientFootageError("Not enough qualifying footage with a visible person throughout: "
                f"no verified allocation for {len(slots)} full cuts with a stable opening and a {gap:g}s source gap "
                "after checking alternative windows. Scenery and unconfirmed pose detections "
                "are excluded. Noise-only character cuts are excluded." + speech_note)
        for index, (score, start, seg, metrics) in allocation.items():
            matches[index] = make_match(index, score, start, seg, metrics)
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
