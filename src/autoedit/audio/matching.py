"""Backend-independent opening-sound matching and conservative fallback selection.

ASR text and curated Voice labels establish a word/pronunciation match, not an
acoustic identity or verified lip sync. Visual hints alone never enable events.
"""
from collections import Counter
import math

from .pronunciation import VoiceIndex, active_bounds, source_sounds, valid_sound
from .taxonomy import item_group, item_phonetics, sound_type

POLICY_VERSION = 'source-first-1'


def search_budget(audio):
    value = audio.get('source_match_search_budget_sec', 300)
    if isinstance(value, bool):
        raise ValueError('audio.source_match_search_budget_sec must be finite and positive')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('audio.source_match_search_budget_sec must be finite and positive') from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError('audio.source_match_search_budget_sec must be finite and positive')
    return value


def neutral_pool(items):
    """A short file or a neutral visual expression does not make a WAV neutral."""
    return [v for v in items if item_group(v) == 'NEUTRAL'
            and item_phonetics(v).get('action') == 'SPEECH']


def choose_neutral(items, duration, used=()):
    pool = neutral_pool(items)
    counts = used if isinstance(used, Counter) else Counter(used)
    ranked = []
    for v in pool:
        lo, hi = active_bounds(v)
        if not all(math.isfinite(n) for n in (lo, hi)) or not 0 <= lo < hi:
            continue
        key = v.get('voice_id', v.get('path'))
        fit = min(hi-lo, duration) / max(hi-lo, duration)
        ranked.append(((counts[key], -fit, str(key)), v))
    if not ranked:
        raise ValueError('No usable NEUTRAL speech recordings in Voice; unmatched scenes cannot be assigned laughter or effects.')
    return min(ranked, key=lambda r: r[0])[1]


def visual_event_confirmed(sound, cut=None, samples=()):
    action = sound.get('action')
    if not action:
        return True
    cut = cut or {}
    if cut.get('visual_action') == action:
        return True
    # Legacy face analysis only has an explicit visual class for laughter.
    # Coughs/sighs need an explicit backend observation, never an open mouth.
    window = [s for s in samples if sound['start'] <= s.get('time_sec', -1) < sound['end']
              and s.get('clear_face')]
    return action == 'LAUGH' and len(window) >= 2 and all(
        s.get('category') in {'HAHA', 'LAUGH'} for s in window)


class OpeningSoundIndex:
    def __init__(self, items):
        self.items = {}
        for v in items:
            if v.get('path'):
                self.items.setdefault(v['path'], v)
        self.index = VoiceIndex(list(self.items.values()))

    def anchors(self, speech):
        rows = []
        for s in source_sounds(speech):
            if s['prob'] < (.75 if s.get('action') else .6):
                continue
            matches = self.index.match(s)
            if matches:
                rows.append({**s, 'matches': matches,
                             'confidence_tier': 1 if s['prob'] >= .8 and not s.get('action') else 2})
        return rows

    def opening(self, cut, speech, *, samples=(), allow_saved=False):
        start, end = cut['in_sec'], cut.get('identity_end_sec', cut['out_sec'])
        # One video frame of timestamp quantization is allowed; never .25s of
        # an unrelated opening word. ASR timestamps remain in source seconds.
        slack = 1 / max(1., float(cut.get('source_fps') or 30.)) + 1e-6
        sounds = [s for s in source_sounds(speech) if s['start'] < end and s['end'] > start]
        saved = cut.get('source_sound')
        if not sounds and saved and allow_saved and valid_sound(saved):
            sounds = [saved]
        if not sounds:
            return None
        sound = sounds[0]
        if (abs(sound['start']-start) > slack or sound['end'] > end+1e-6
                or sound['prob'] < (.75 if sound.get('action') else .6)
                or not visual_event_confirmed(sound, cut, samples)):
            return None
        matches = self.index.match(sound)
        return {**sound, 'matches': matches} if matches else None

    def select(self, cut, speech, used=(), *, preferred=None, samples=(), allow_saved=False, honor_preferred=False):
        counts = used if isinstance(used, Counter) else Counter(used)
        sound = self.opening(cut, speech, samples=samples, allow_saved=allow_saved)
        if not sound:
            v = choose_neutral(list(self.items.values()), cut['out_sec']-cut['in_sec'], counts)
            return v, fallback_evidence('No supported opening sound with visual event evidence')
        ranked = []
        for match in sound['matches']:
            v = self.items[match['path']]
            lo, hi = active_bounds(v)
            duration = sound['end']-sound['start']
            fit = min(hi-lo, duration)/max(hi-lo, duration)
            key = v.get('voice_id', v['path'])
            ranked.append(((honor_preferred and preferred is not None and key != preferred,
                            match['kind'] != 'source-audio-whisper-exact-word',
                            preferred is not None and key != preferred, -fit, counts[key], str(key)), v, match, fit))
        _, v, match, fit = min(ranked, key=lambda r: r[0])
        evidence = {'policy_version': POLICY_VERSION, 'method': match['kind'],
                    'source_word': sound.get('word'), 'source_action': sound.get('action'),
                    'source_word_start_sec': sound['start'], 'source_word_end_sec': sound['end'],
                    'source_pronunciation': match.get('phones', []), 'word_probability': sound['prob'],
                    'source_sound': {k: v for k, v in sound.items() if k != 'matches'},
                    'cut_onset_matched': True, 'duration_fit': round(fit, 4),
                    'sound_type': sound_type(sound.get('action', 'SPEECH')),
                    'needs_review': True, 'phoneme_sync_verified': False,
                    'onset_verified': False, 'cubase_timing_verified': False}
        if sound.get('action'):
            evidence.update(visual_event_confirmed=True, confidence_kind='heuristic-not-calibrated',
                            review_reason='Source vocal-event hint and visual action; listen to confirm')
        else:
            evidence['review_reason'] = 'Source timestamp/label match; articulation and MIDI timing need review'
        return v, evidence


def fallback_evidence(reason):
    return {'policy_version': POLICY_VERSION, 'method': 'neutral-source-unmatched',
            'cut_onset_matched': False, 'needs_review': True, 'sound_type': 'speech',
            'review_reason': reason, 'phoneme_sync_verified': False,
            'onset_verified': False, 'cubase_timing_verified': False}


def assign_rows(rows, sources, voices, *, manual=False):
    """Recompute evidence after selection, recovery or manual edits; no inference."""
    if not isinstance(rows, list):
        return []
    index = OpeningSoundIndex(voices)
    source_map = {s['id']: s for s in sources}
    voice_map = {v['voice_id']: v for v in voices}
    counts = Counter()
    for row in rows:
        if (not isinstance(row, dict) or row.get('source_id') not in source_map
                or row.get('voice_id') not in voice_map
                or not all(isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool)
                           and math.isfinite(row[k]) for k in ('in_sec', 'out_sec'))
                or row['out_sec'] <= row['in_sec']):
            continue  # The structural validator reports malformed backend output.
        speech = source_map[row['source_id']].get('source_speech', {})
        chosen, evidence = index.select(row, speech, counts, preferred=row.get('voice_id'))
        if manual:
            requested = voice_map[row['voice_id']]
            if requested['path'] != chosen['path']:
                supported = index.opening(row, speech)
                if supported and requested['path'] in {m['path'] for m in supported['matches']}:
                    chosen, evidence = index.select(row, speech, counts, preferred=row['voice_id'], honor_preferred=True)
                else:
                    chosen = requested
                    evidence = fallback_evidence('Manual override: opening sound does not match this Voice')
                    evidence['manual_override'] = True
            if not evidence.get('cut_onset_matched'):
                evidence['manual_override'] = True
        row.update(voice_id=chosen['voice_id'], audio_match=evidence, audio_uncertain=True)
        if evidence.get('cut_onset_matched'):
            row['source_sound'] = evidence['source_sound']
        else:
            row.pop('source_sound', None)
        counts[chosen['voice_id']] += 1
    return rows
