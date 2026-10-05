"""Choose one sample for each Premiere scene run, after source cuts are fixed."""
from collections import Counter
import math

from autoedit.match.matcher import same_family
from autoedit.audio.selection import select_audio_for_source
from autoedit.audio.taxonomy import TAXONOMY, item_group, item_phonetics, sound_type
from autoedit.audio.onsets import AudioOnsetError, articulation_onsets, source_onset
from autoedit.audio.pronunciation import VoiceIndex, source_sounds, valid_sound
from autoedit.cubase.inspect import parse_sampler_tracks

WORD_SHAPES = {sound.label: sound.visemes[0] for _, sound in TAXONOMY.values()
               if sound.visemes and sound.action == "SPEECH" and sound.label not in {"A", "E", "O", "U"}}


def _shape(category):
    category = (category or "").removeprefix("VOWEL_")
    return WORD_SHAPES.get(category, category)


def _active_bounds(item):
    lo = float(item.get("active_start_sec", item.get("segment_start_sec", 0)) or 0)
    hi = float(item.get("active_end_sec", item.get("segment_end_sec", item.get("duration_sec", 0))) or 0)
    return lo, hi


def _cut_samples(video_info, video, padding=0.):
    """Overlapping refinement windows must not mix actors or duplicate frames."""
    selected_track = (video.get('selection_metrics') or {}).get('subject_track_id')
    samples = {}
    for sample in video_info.get('samples', []):
        if not video['in_sec']-padding <= sample['time_sec'] < video['out_sec']+padding or not sample.get('clear_face'):
            continue
        boundary = sample.get('refinement_cut')
        if boundary and (abs(boundary['in_sec']-video['in_sec']) > .001 or abs(boundary['out_sec']-video['out_sec']) > .001):
            continue
        if selected_track and sample.get('track_id') != selected_track:
            continue
        samples.setdefault(round(sample['time_sec'], 7), sample)
    return sorted(samples.values(), key=lambda s: s['time_sec'])


def _source_word_audio(candidates, video, video_info, used):
    start, end = video["in_sec"], video["out_sec"]
    onset_slack = .25 if video_info.get('verify_onsets') else .001
    anchor = video.get("source_sound")
    if anchor:
        if not valid_sound(anchor) or abs(anchor["start"]-start) > onset_slack or anchor["end"] > end:
            raise ValueError("Selected source sound no longer agrees with the cut onset; regenerate the plan.")
        sound = anchor
        paths = {m['path'] for m in anchor['matches']}
        # Saved anchors cannot bypass a corrected taxonomy or event/speech gate.
        supported = [m for m in VoiceIndex(candidates).match(sound) if m['path'] in paths]
    else:
        # The first sound blocks all later words, even if it is unsupported or
        # uncertain. A cut through a word must not match the following word.
        sounds = [s for s in source_sounds(video_info.get("source_speech", {}))
                  if s["start"] < end and s["end"] > start
                  and not ((video.get("selection_metrics") or {}).get("require_lip_motion")
                           and s.get("action") and s.get("prob", 0) < .85)]
        if not sounds:
            return None
        sound = sounds[0]
        threshold = .75 if sound.get("action") else .6
        if not start-(.25 if video_info.get('verify_onsets') else 0) <= sound["start"] <= start+.25 or sound["end"] > end or sound["prob"] < threshold:
            return None
        supported = VoiceIndex(candidates).match(sound)
    matches = {m["path"]: m for m in supported}
    ranked, seen = [], set()
    for item in candidates:
        if item["path"] in seen:
            continue
        seen.add(item["path"])
        lo, hi = _active_bounds(item)
        length = hi-lo
        if item["path"] not in matches or not math.isfinite(length) or length <= 0:
            continue
        word_length = sound["end"]-sound["start"]
        fit = min(length, word_length)/max(length, word_length)
        retained = min(length, end-sound["start"])/length
        score = 3*sound["prob"] + fit + retained - .08*used[item["path"]]
        ranked.append((score, fit, retained, item, sound, lo, hi))
    if not ranked:
        if anchor:
            raise ValueError("WAV matched to the cut onset is missing from the Voice library; regenerate the plan.")
        return None
    score, fit, retained, item, word, lo, hi = max(ranked, key=lambda row: row[0])
    phon = item_phonetics(item)
    readable = [s for s in _cut_samples(video_info, video, .35)
                if s['time_sec'] <= min(end, start+.6)]
    onset = source_onset(readable, phon.get('visemes'), word['start'], video) if phon.get('action') == 'SPEECH' else None
    verify = video_info.get('verify_onsets', False)
    if verify and len(readable) >= 3 and phon.get('action') == 'SPEECH' and not onset:
        raise AudioOnsetError('Source sound has no matching articulation onset; select another cut', video)
    chosen = dict(item, segment_start_sec=lo, segment_end_sec=min(hi, lo+end-word["start"]),
                  placement_offset_sec=max(0., word["start"]-start))
    match = matches[item["path"]]
    evidence = {"method": match["kind"], "score": round(score, 4),
                "source_word": word.get("word"), "source_action": word.get("action"),
                "source_pronunciation": match.get("phones", []),
                "cut_onset_matched": True,
                "source_word_start_sec": word["start"],
                "source_word_end_sec": word["end"], "word_probability": word["prob"],
                "duration_fit": round(fit, 4), "retained_fraction": round(retained, 4),
                "transcript_model": video_info.get("source_speech", {}).get("model"),
                "phoneme_sync_verified": False,
                "needs_review": match["kind"] != "source-audio-whisper-exact-word"
                                or word["prob"] < .8 or fit < .5 or retained < .9
                                or (video.get("selection_metrics") or {}).get("tier", 1) > 1,
                "alternatives": [{"name": row[3].get("name", row[3]["path"]),
                                  "source_word": row[4].get("word"), "score": round(row[0], 4)}
                                 for row in sorted(ranked, key=lambda row: -row[0])[:3]]}
    evidence.update(sound_type=sound_type(phon.get('action', 'UNCLEAR')), onset_verified=False)
    if onset:
        evidence.update(onset, suggested_cut_in_sec=onset['visual_onset_sec'],
                        onset_error_sec=onset['visual_onset_sec']-word['start'])
        chosen['placement_offset_sec'] = max(0., onset['visual_onset_sec']-start)
    else:
        evidence.update(needs_review=True, timing_review='No measured articulation onset; source timestamp alone is not lip-sync verification')
    return chosen, evidence


def choose_audio(candidates, video, video_info, used, recent=()):
    """Prefer source words, then measured visemes, then neutral recordings.

    Onset search moves the recording, without stretching or pitching it. This
    is visual matching, not a claim to recover words from silent mouth motion.
    """
    import numpy as np

    samples = _cut_samples(video_info, video)
    strict = (video.get("selection_metrics") or {}).get("require_lip_motion")
    if strict:
        from autoedit.video.faces import lip_motion_evidence
        if not samples:
            raise ValueError("Selected speaking cut has no readable lips in dense analysis; regenerate/review this cut instead of assigning an unrelated neutral WAV.")
        if not lip_motion_evidence(samples):
            raise ValueError("Selected speaking cut has no visible lip movement in dense analysis; cannot assign a voice to a static face.")
    exact = _source_word_audio(candidates, video, video_info, used)
    if exact:
        return exact
    candidates = select_audio_for_source(candidates, video_info.get("source_kind", "live_action"))
    if not strict and (not samples or (video.get("selection_metrics") or {}).get("tier", 1) >= 3):
        return _coarse_audio(candidates, video, video_info, used)
    duration = video["out_sec"] - video["in_sec"]
    # Closed/preparatory and closing samples are essential negative evidence.
    times = np.array([s["time_sec"] - video["in_sec"] for s in samples])
    aperture = np.array([s.get("lip_aperture", s.get("openness", 0) / 2) for s in samples])
    opened = np.clip(aperture / max(float(np.quantile(aperture, .9)), .08), 0, 1)
    shapes = ["M" if a < .035 else _shape(s.get("category")) for a, s in zip(aperture, samples)]
    ranking = []
    seen = set()
    for item in candidates:
        # Whole recordings already contain the labelled active range. Expanded
        # duplicate segments must not bias the ranking or reuse penalty.
        if item["path"] in seen:
            continue
        seen.add(item["path"])
        phon = item_phonetics(item)
        action = phon.get("action", "SPEECH")
        # Events require source-event evidence, never expression similarity.
        if action != 'SPEECH':
            continue
        lo = float(item.get("active_start_sec", item.get("segment_start_sec", 0)) or 0)
        hi = float(item.get("active_end_sec", item.get("segment_end_sec", item.get("duration_sec", 0))) or 0)
        length = hi - lo
        if not math.isfinite(length) or length <= 0:
            continue
        visemes = phon.get("visemes") or [_shape(item.get("category"))]
        env = np.array(item.get("envelope") or [1., 1.])
        step = float(item.get("envelope_step_sec") or length)
        # Legacy callers may shift/trim their preview. The verified pipeline
        # moves the video cut instead, preserving the recording's initial sound.
        shifts = np.arange(-min(max(0, length-duration), .3), min(.3, max(0, duration-length)) + .001, .025)
        onset_samples = _cut_samples(video_info, video, .35)
        onsets = articulation_onsets(onset_samples, visemes, max(0, video['in_sec']-.25),
                                    min(video['out_sec']-.05, video['in_sec']+.3))
        if video_info.get('verify_onsets'):
            shifts = np.array([p['visual_onset_sec']-video['in_sec'] for p in onsets])
        for shift in shifts:
            local = times - shift
            active = (local >= 0) & (local < length)
            energy = np.interp(local + lo, np.arange(len(env))*step, env, left=0, right=0) * active
            progress = np.clip(local / length, 0, .999999)
            sound_shapes = [visemes[int(p * len(visemes))] if on else "M" for p, on in zip(progress, active)]
            overlap = float(np.mean([same_family(a, b) for a, b in zip(shapes, sound_shapes)]))
            rhythm = float(1 - np.mean(np.abs(energy - opened)))
            fit = min(length, duration) / max(length, duration)
            retained = max(0, min(duration, shift+length)-max(0, shift))/length
            score = 3*overlap + 2*rhythm + fit + .5*retained
            if video_info.get('verify_onsets'):
                score -= 4*abs(float(shift))
            # Moving the cut preserves the recording's initial articulation.
            ranking.append((score, overlap, rhythm, fit, shift, lo, hi, item))
    if not ranking:
        if video_info.get('verify_onsets'):
            raise AudioOnsetError('No labelled speech recording has a compatible articulation onset', video)
        raise ValueError("No labelled speech samples match this scene; review the Voice library labels.")
    # Avoid cycling back to the same word via another filename (ai/aii).
    # Exact transcript matches above always retain priority over variety.
    def label(item):
        return item.get("category") or item["path"]
    previous = {label(item) for item in recent[-2:]}
    fresh = [r for r in ranking if label(r[-1]) not in previous]
    if fresh:
        ranking = fresh
    ranking.sort(key=lambda r: (-(r[0] - .6*used[r[-1]["path"]]), r[-1]["path"]))
    score, overlap, rhythm, fit, shift, lo, hi, item = ranking[0]
    chosen = dict(item, segment_start_sec=lo if video_info.get('verify_onsets') else lo + max(0, -shift), segment_end_sec=hi,
                  placement_offset_sec=max(0, float(shift)))
    alternatives = []
    for row in ranking:
        name = row[-1].get("name", row[-1]["path"])
        if name not in [x["name"] for x in alternatives]:
            alternatives.append({"name": name, "score": round(row[0], 4)})
        if len(alternatives) == 3:
            break
    evidence = {"score": round(score, 4), "shape_overlap": round(overlap, 4),
        "rhythm_fit": round(rhythm, 4), "duration_fit": round(fit, 4),
        "onset_shift_sec": round(float(shift), 4), "lip_sample_count": len(samples),
        "method": "filename-visemes-and-dense-lip-rhythm", "phoneme_sync_verified": False,
        "needs_review": True, "review_reason": "No supported source word; visual fallback only",
        "reuse_count": used[item["path"]], "alternatives": alternatives}
    evidence.update(sound_type='speech', onset_verified=False)
    if video_info.get('verify_onsets'):
        proof = next(p for p in articulation_onsets(_cut_samples(video_info, video, .35),
            item_phonetics(item).get('visemes'), max(0, video['in_sec']-.25), video['in_sec']+.3)
            if abs(p['visual_onset_sec']-video['in_sec']-float(shift)) < 1e-6)
        evidence.update(proof, suggested_cut_in_sec=video['in_sec']+float(shift))
    return chosen, evidence


def _coarse_audio(candidates, video, video_info, used):
    """Unreadable lips use curated neutral recordings, with variety by file."""
    duration = video["out_sec"] - video["in_sec"]
    ranked, seen = [], set()
    for item in candidates:
        if item["path"] in seen or item_group(item) != "NEUTRAL" or item_phonetics(item).get('action') != 'SPEECH':
            continue
        seen.add(item["path"])
        lo, hi = _active_bounds(item)
        length = hi-lo
        if not math.isfinite(length) or length <= 0:
            continue
        fit = min(length, duration) / max(length, duration)
        score = 2*fit + .1*float(item.get("confidence") or 0)
        ranked.append((used[item["path"]], -score, fit, lo, hi, item))
    if not ranked:
        raise ValueError("No usable NEUTRAL taxonomy recordings for scenes without readable visemes.")
    # Use each file before cycling; expanded segments do not create extra votes.
    reuse, negative_score, fit, lo, hi, item = min(ranked, key=lambda row: row[:2])
    chosen = dict(item, segment_start_sec=lo, segment_end_sec=min(hi, lo+duration),
                  placement_offset_sec=0.0, audio_style="NEUTRAL")
    return chosen, {"score": round(-negative_score, 4), "duration_fit": round(fit, 4),
                    "method": "neutral-no-readable-viseme", "phoneme_sync_verified": False,
                    'sound_type': 'speech', 'onset_verified': False,
                    "needs_review": True, "reuse_count": reuse}


def build_sampler_slots(scenes, video_slots, candidates, video_info, inventory):
    tracks = inventory["tracks"]
    expected = parse_sampler_tracks(inventory.get("selected_track_indices"))
    if expected is None:
        expected = list(range(1, len(scenes)+1))
    if len(tracks) != len(scenes) or [t["index"] for t in tracks] != expected:
        raise ValueError(f"Premiere has {len(scenes)} scene runs but Cubase sampler tracks are "
                         f"{[t['index'] for t in tracks]}; exact ordered mapping is required (no wraparound).")
    parents = {s["premiere"]["nested_sequence_uid"]: s for s in video_slots}
    selected, used, slots = {}, Counter(), []
    for scene, track in zip(scenes, tracks):
        uid = scene["nested_sequence_uid"]
        parent = parents[uid]
        video = parent["video"]
        if uid not in selected:
            selected[uid] = choose_audio(candidates, video, video_info, used,
                                         recent=[s["audio"] for s in slots])
            used[selected[uid][0]["path"]] += 1
        audio, evidence = selected[uid]
        # A later reprise uses the same sample on its own numbered track.
        slots.append({"id": scene["id"], "index": scene["index"], "audio": dict(audio),
                      "video": dict(video), "audio_match": dict(evidence),
                      "premiere": {"nested_sequence": scene["nested_sequence"],
                                   "nested_sequence_uid": uid, "mode": "scene_run",
                                   "timeline_start_sec": scene["first_start_sec"],
                                   "timeline_end_sec": scene["last_end_sec"],
                                   "duration_sec": parent["premiere"]["duration_sec"],
                                   "instances": scene["instances"]},
                      "cubase": {"track_index": track["index"], "track_name": track["name"],
                                 "project": inventory["path"], "start_sec": scene["first_start_sec"],
                                 "preserve_existing_midi": True},
                      "match_status": "sample_selected"})
    return slots
