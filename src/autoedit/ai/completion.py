"""Finish a reviewable draft when recognition cannot fill the whole template.

These cuts are deliberately NOT character observations. Source decoding and exact
slot lengths remain mandatory; relaxed editorial checks are recorded per cut.
"""
from copy import deepcopy
from itertools import islice
import math
from pathlib import Path

from autoedit.runtime import check_cancelled
from .selection import number, used_end

VERSION = 'draft-completion-2-neutral-speech'
RELAXATIONS = {'character_unverified', 'scene_changes', 'source_gap',
               'visual_duplicate', 'overlap', 'low_detail', 'freeze_frame'}


def shortage_policy(ai):
    value = ai.get('shortage_policy', 'complete')
    if value not in ('complete', 'strict'):
        raise ValueError('ai.shortage_policy must be complete or strict')
    return value


def validate_draft(row, slot, source, voices, accepted, gap):
    """Validate actual media/timing without inventing identity or shot evidence."""
    a, b = row.get('in_sec'), row.get('out_sec')
    lo = source.get('analyzed_start_sec', 0.)
    hi = min(source['duration_sec'], source.get('analyzed_end_sec', source['duration_sec']))
    if not number(a) or not number(b) or not lo <= a < b <= hi+1e-6:
        raise ValueError('Draft source bounds are invalid')
    if abs(b-a-slot['required_duration_sec']) > .002:
        raise ValueError('Draft duration differs from template')
    row['out_sec'] = a+slot['required_duration_sec']
    if row['out_sec'] > hi+1e-6:
        raise ValueError('Exact draft duration exceeds source')
    if row.get('voice_id') not in voices:
        raise ValueError('Unknown draft Voice ID')
    flags = row.get('fallback_relaxations')
    if (not isinstance(flags, list) or not all(isinstance(f, str) and f in RELAXATIONS for f in flags)
            or 'character_unverified' not in flags):
        raise ValueError('Invalid draft relaxations')
    if (row.get('character_id') != 'unverified' or row.get('identity_sample_consistent') is not False
            or row.get('audio_uncertain') is not True or not isinstance(row.get('reference_fallback'), bool)
            or not isinstance(row.get('reason'), str) or not row['reason'].strip()):
        raise ValueError('Draft must disclose unverified character/audio')
    if (row.get('shot_start_sec') != a or row.get('shot_end_sec') != b
            or not isinstance(row.get('shot_id'), str) or not row['shot_id']):
        raise ValueError('Invalid draft interval')
    visible = slot.get('visible_duration_sec', slot['required_duration_sec'])
    if row.get('timeline_visible_prefix'):
        if not 0 < visible < slot['required_duration_sec']-.002 or row.get('identity_end_sec') != a+visible:
            raise ValueError('Invalid draft visible prefix')
    for previous in accepted:
        if previous['source_id'] != row['source_id']:
            continue
        distance = max(a-used_end(previous), previous['in_sec']-used_end(row))
        if distance < -1e-6 and 'overlap' not in flags:
            raise ValueError('Draft overlaps an existing selection')
        if distance < gap-1e-6 and 'source_gap' not in flags:
            raise ValueError('Draft source gap was not disclosed')
    row.update(local_character_verified=False, character_sample_frames=0,
               visual_method=VERSION, shot_bounds_complete=False)
    return row


def _positions(source, duration, selected, gap, overlap=False):
    lo = source.get('analyzed_start_sec', 0.)
    hi = min(source['duration_sec'], source.get('analyzed_end_sec', source['duration_sec']))
    spans = [(lo, hi)]
    if not overlap:
        for row in selected:
            if row['source_id'] != source['id']:
                continue
            left, right = row['in_sec']-gap, used_end(row)+gap
            spans = [(x, y) for a, b in spans for x, y in ((a, min(b, left)), (max(a, right), b)) if y > x]
    # Try wide free intervals first, and several points away from titles/credits.
    for a, b in sorted(spans, key=lambda span: span[0]-span[1]):
        if b-a < duration-1e-6:
            continue
        for fraction in (.37, .63, .19, .81, .5, .08, .92, 0., 1.):
            yield max(a, min(b-duration, a+(b-a-duration)*fraction))


def _freeze_source(sources, duration, cache, log):
    """A genuinely short input gets an explicit held-frame asset, never bad out-points."""
    import av
    from .cache import digest, identity
    from .frames import cut_frames
    from .visual import probe
    for source in list(sources):
        check_cancelled()
        lo = source.get('analyzed_start_sec', 0.)
        hi = source.get('analyzed_end_sec', source['duration_sec'])
        try:
            _, rgb = next(cut_frames(source['path'], [lo+(hi-lo)*.5], 1280))
        except (OSError, ValueError, StopIteration):
            continue
        dest = Path(cache)/'draft_assets'/(digest([VERSION, source['identity'], duration])+'.mp4')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temp = dest.with_suffix('.tmp.mp4')
            try:
                with av.open(str(temp), 'w') as target:
                    stream = target.add_stream('libx264', rate=25)
                    stream.width, stream.height = rgb.shape[1]//2*2, rgb.shape[0]//2*2
                    stream.pix_fmt = 'yuv420p'
                    stream.options = {'crf': '18', 'preset': 'veryfast'}
                    for index in range(math.ceil(duration*25)+2):
                        check_cancelled()
                        frame = av.VideoFrame.from_ndarray(rgb[:stream.height, :stream.width], format='rgb24')
                        frame.pts = index
                        for packet in stream.encode(frame):
                            target.mux(packet)
                    for packet in stream.encode():
                        target.mux(packet)
                temp.replace(dest)
            finally:
                temp.unlink(missing_ok=True)
        result = {'id': 'draft-hold-'+digest(str(dest))[:12], 'path': str(dest.resolve()),
                  'identity': identity(dest), **probe(dest), 'derived_from': source['id'],
                  'completion_asset': 'freeze_frame'}
        sources.append(result)
        log('Draft completion: source too short; generated a disclosed held-frame fill.')
        return result
    raise ValueError('Cannot decode any source frame to complete the draft')


def complete_selection(result, accepted, slots, sources, voices, references, gap, cache, log, *, deadline=float('inf')):
    from autoedit.runtime import selection_deadline
    with selection_deadline(deadline):
        return _complete_selection(result, accepted, slots, sources, voices, references, gap, cache, log)


def _complete_selection(result, accepted, slots, sources, voices, references, gap, cache, log):
    from .local import choose_voice, transition_safe
    from .visual import fingerprints, similar
    result = deepcopy(result)
    selected = deepcopy(accepted)
    from .selection import validate_selection
    _, _, valid_cast = validate_selection(result, slots, sources, voices, [r['id'] for r in references], gap,
        allow_same_shot=True, transition_tail_fallback_sec=2, allow_completion=True)
    used_characters = {r['character_id'] for r in accepted}
    result['characters'] = [c for c in valid_cast if c['id'] in used_characters]
    used_voices = [r['voice_id'] for r in selected]
    samples = [(r, fingerprints(next(s for s in sources if s['id'] == r['source_id']), r, cache)) for r in selected]
    missing = [s for s in slots if s['id'] not in {r['slot_id'] for r in selected}]
    log(f'Draft completion: preserving {len(selected)}/{len(slots)} validated fills; completing {len(missing)} without identity requirements.')
    for slot in sorted(missing, key=lambda s: -s['required_duration_sec']):
        check_cancelled()
        duration = slot['required_duration_sec']
        if not any(s.get('analyzed_end_sec', s['duration_sec'])-s.get('analyzed_start_sec', 0.) >= duration for s in sources):
            _freeze_source(sources, duration, cache, log)
        chosen = None
        # Only relax a constraint after trying the preceding tier. Decoding and
        # full duration are never relaxed, even for the most permissive tier.
        for tier in range(5):
            for source in sources:
                for a in islice(_positions(source, duration, selected, gap if tier < 2 else 0., overlap=tier == 4), 24):
                    check_cancelled()
                    flags = ['character_unverified']
                    row = {'slot_id': slot['id'], 'source_id': source['id'], 'in_sec': a, 'out_sec': a+duration,
                           'shot_id': f'draft-{slot["id"]}', 'shot_start_sec': a, 'shot_end_sec': a+duration,
                           'duplicate_group': '', 'character_id': 'unverified', 'reference_fallback': bool(references),
                           'audio_uncertain': True, 'identity_sample_consistent': False, 'completion_fallback': True,
                           'local_character_verified': False, 'character_sample_frames': 0, 'visual_method': VERSION,
                           'shot_bounds_complete': False}
                    visible = slot.get('visible_duration_sec', duration)
                    if visible < duration-.002:
                        row.update(timeline_visible_prefix=True, identity_end_sec=a+visible)
                    try:
                        frames = fingerprints(source, row, cache)
                        low_detail = max((f['std'] for f in frames), default=0) < 4
                        duplicate = any(similar(frames, previous) for _, previous in samples)
                        if tier < 3 and (low_detail or duplicate):
                            continue
                        if low_detail:
                            flags.append('low_detail')
                        if duplicate:
                            flags.append('visual_duplicate')
                        audit = {**row, 'out_sec': used_end(row)}
                        if not transition_safe(source, audit, cache):
                            if tier == 0:
                                continue
                            flags.append('scene_changes')
                    except (OSError, ValueError):
                        continue
                    for previous in selected:
                        if previous['source_id'] == row['source_id']:
                            distance = max(a-used_end(previous), previous['in_sec']-used_end(row))
                            if distance < gap-1e-6:
                                flags.append('source_gap')
                            if distance < -1e-6:
                                flags.append('overlap')
                    if source.get('completion_asset'):
                        flags.append(source['completion_asset'])
                    row.update(voice_id=choose_voice(voices, 'neutral', duration, used_voices)['voice_id'],
                               fallback_relaxations=sorted(set(flags)),
                               reason='Draft completion after AI shortage. Character and Voice match unverified. '
                                      'Review: '+', '.join(sorted(set(flags)))+'.')
                    from autoedit.audio.matching import fallback_evidence
                    row['audio_match'] = fallback_evidence('Draft completion after source-matched candidates were insufficient')
                    chosen = row, frames
                    break
                if chosen:
                    break
            if chosen:
                break
        if not chosen:
            raise ValueError(f'No decodable full-duration footage for {slot["id"]}; check source media')
        row, frames = chosen
        selected.append(row)
        samples.append((row, frames))
        used_voices.append(row['voice_id'])
        log(f'Draft fill {slot["id"]}: {row["source_id"]} {row["in_sec"]:.2f}–{row["out_sec"]:.2f}s; '+', '.join(row['fallback_relaxations']))
    result.update(selections=selected, completion={'version': VERSION, 'policy': 'complete',
                  'validated_count': len(accepted), 'fallback_count': len(missing), 'needs_review': True})
    return result
