from copy import deepcopy

import numpy as np
import pytest

from autoedit.ai import completion, visual
from autoedit.ai.selection import validate_selection
from autoedit.runtime import JobCancelled


@pytest.fixture
def draft(monkeypatch, tmp_path):
    import autoedit.ai.local as local
    monkeypatch.setattr(local, 'transition_safe', lambda *a: True)
    monkeypatch.setattr(visual, 'fingerprints', lambda *a: [{'std': 20}]*8)
    monkeypatch.setattr(visual, 'similar', lambda *a: False)
    return dict(result={'characters': [], 'selections': [], 'analysis_scope': {'stop_reason': 'time_budget'}},
        accepted=[], slots=[{'id': f'slot-{i}', 'required_duration_sec': 1.} for i in range(14)],
        sources=[{'id': 's', 'path': 'not-used', 'duration_sec': 1800.}],
        voices=[{'voice_id': 'v', 'duration_sec': 1}], references=[], gap=5., cache=tmp_path, log=lambda s: None)


def validate(result, args, allowed=True):
    return validate_selection(result, args['slots'], args['sources'], args['voices'], [], args['gap'], allow_completion=allowed)


def test_thirty_minute_source_completes_all_slots_without_fake_character_evidence(draft):
    result = completion.complete_selection(**draft)
    rows, errors, ranking = validate(result, draft)
    assert not errors and len(rows) == 14
    assert ranking == [] and result['characters'] == []
    assert all(r['identity_sample_consistent'] is False for r in rows)
    assert all(r['fallback_relaxations'] == ['character_unverified'] for r in rows)
    assert result['analysis_scope']['stop_reason'] == 'time_budget'
    assert result['completion']['fallback_count'] == 14
    assert validate(result, draft, allowed=False)[1]


def test_tiers_disclose_cuts_duplicates_and_overlap_only_when_required(draft, monkeypatch):
    import autoedit.ai.local as local
    monkeypatch.setattr(local, 'transition_safe', lambda *a: False)
    monkeypatch.setattr(visual, 'similar', lambda *a: True)
    draft['sources'][0]['duration_sec'] = 1.2
    draft['slots'] = draft['slots'][:2]
    result = completion.complete_selection(**draft)
    assert not validate(result, draft)[1]
    assert 'scene_changes' in result['selections'][0]['fallback_relaxations']
    flags = result['selections'][1]['fallback_relaxations']
    assert {'source_gap', 'overlap', 'visual_duplicate', 'scene_changes'} <= set(flags)
    assert len(visual.check_visual_duplicates(result['selections'], draft['sources'], draft['cache'])[0]) == 2


def test_draft_never_bypasses_media_bounds_duration_voice_or_gap(draft):
    result = completion.complete_selection(**draft)
    for update in ({'out_sec': 1801.}, {'out_sec': 3.}, {'voice_id': 'missing'},
                   {'identity_sample_consistent': True}, {'fallback_relaxations': ['invented']}):
        bad = deepcopy(result)
        bad['selections'][0].update(update)
        assert validate(bad, draft)[1]
    bad = deepcopy(result)
    first, second = bad['selections'][:2]
    second.update(in_sec=first['in_sec'], out_sec=first['out_sec'],
                  shot_start_sec=first['in_sec'], shot_end_sec=first['out_sec'])
    assert validate(bad, draft)[1]


def test_cancel_is_never_converted_to_a_draft(draft, monkeypatch):
    monkeypatch.setattr(completion, 'check_cancelled', lambda: (_ for _ in ()).throw(JobCancelled('cancel')))
    with pytest.raises(JobCancelled):
        completion.complete_selection(**draft)


def test_short_source_renders_real_full_duration_hold_asset(draft, tmp_path, monkeypatch):
    import av
    from autoedit.ai.cache import identity
    path = tmp_path/'short.mp4'
    with av.open(str(path), 'w') as target:
        stream = target.add_stream('libx264', rate=25)
        stream.width = stream.height = 64
        stream.pix_fmt = 'yuv420p'
        for i in range(10):
            rgb = np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            for packet in stream.encode(frame):
                target.mux(packet)
        for packet in stream.encode():
            target.mux(packet)
    draft['sources'] = [{'id': 's', 'path': str(path), 'identity': identity(path), **visual.probe(path)}]
    draft['slots'] = draft['slots'][:1]
    result = completion.complete_selection(**draft)
    row = result['selections'][0]
    assert 'freeze_frame' in row['fallback_relaxations']
    derived = next(s for s in draft['sources'] if s['id'] == row['source_id'])
    assert visual.probe(derived['path'])['duration_sec'] >= 1.
    assert not validate(result, draft)[1]
