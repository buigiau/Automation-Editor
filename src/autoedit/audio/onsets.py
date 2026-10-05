"""Measured articulation onsets; moving nearby is not an onset certificate."""
import math

from autoedit.match.matcher import InsufficientFootageError
from autoedit.audio.taxonomy import TAXONOMY

_WORD_SHAPES = {s.label: s.visemes[0] for _,s in TAXONOMY.values()
                if s.action == 'SPEECH' and s.visemes and s.label not in {'A','E','O','U'}}


class AudioOnsetError(InsufficientFootageError):
    def __init__(self, message, cut=None):
        super().__init__(message)
        self.cut = cut


def articulation_onsets(samples, visemes, low, high):
    samples = sorted((s for s in samples if s.get('clear_face')), key=lambda s: s['time_sec'])
    if not visemes:
        return []
    initial = visemes[0]
    result = []
    for index in range(1, len(samples)-1):
        before, current, after = samples[index-1:index+2]
        time = float(current['time_sec'])
        if not low <= time <= high:
            continue
        if any(not 0 < b['time_sec']-a['time_sec'] <= .12 for a,b in ((before,current),(current,after))):
            continue
        if any(before.get(k) != s.get(k) for k in ('track_id', 'shot_id') for s in (current, after)):
            continue
        aperture = [float(s.get('lip_aperture', s.get('openness', 0)/2)) for s in (before,current,after)]
        widths = [float(s.get('lip_width_ratio', 0)) for s in (before,current,after)]
        if not all(math.isfinite(v) for v in aperture+widths):
            continue
        cartoon = current.get('backend') == 'animation-eyes-mouth'
        mouth_width = max(1, current.get('mouth_box', [0,0,18,0])[2])
        change = max(.06, 1.5/mouth_width) if cartoon else .015
        closed = .035
        opening = (aperture[1]-aperture[0] >= change
                   and aperture[2] >= aperture[1]-change/2 and aperture[1] >= closed)
        reshaping = (abs(widths[1]-widths[0]) >= change
                     and aperture[1] >= closed and aperture[1] >= aperture[0]-change/2
                     and aperture[2] >= aperture[1]-change/2)
        if initial == 'M':
            # Final closure before release, rather than the earlier silent preparation.
            valid = aperture[1] < closed and aperture[2]-aperture[1] >= change
            phase = 'closed-consonant-release'
        else:
            valid = opening or reshaping
            phase = 'opening' if opening else 'reshaping'
        category = current.get('category', '').removeprefix('VOWEL_')
        measured_shape = _WORD_SHAPES.get(category, category)
        expected = visemes[1] if initial == 'M' and len(visemes) > 1 else initial
        observed = after.get('category', measured_shape) if initial == 'M' else measured_shape
        observed = _WORD_SHAPES.get(observed, observed)
        if observed in {'A', 'E', 'O', 'U', 'M', 'F'}:
            valid = valid and (observed == expected or {observed, expected} <= {'O', 'U'})
        if valid:
            result.append({'visual_onset_sec': time, 'mouth_phase': phase,
                           'initial_viseme': initial, 'onset_verified': True,
                           'onset_resolution_sec': max(time-before['time_sec'], after['time_sec']-time)})
    return result


def source_onset(samples, visemes, word_start, cut):
    candidates = articulation_onsets(samples, visemes, max(0, word_start-.25),
                                    min(cut['out_sec']-.05, word_start+.25))
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs(p['visual_onset_sec']-word_start))
