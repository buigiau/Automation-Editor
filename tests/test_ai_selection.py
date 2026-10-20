from copy import deepcopy
import math
import pytest
from autoedit.ai.selection import validate_selection


def fixture():
    slots = [{'id': 'a', 'required_duration_sec': 2}, {'id': 'b', 'required_duration_sec': 2}]
    sources = [{'id': 's1', 'duration_sec': 100}, {'id': 's2', 'duration_sec': 100}]
    voices = [{'voice_id': 'v'}]
    character = {'id': 'car', 'description': 'Talking car', 'reference_ids': ['reference-1'],
                 'appearances': [{'source_id': 's1', 'start_sec': 0, 'end_sec': 100},
                                 {'source_id': 's2', 'start_sec': 0, 'end_sec': 100}]}
    rows = [{'slot_id': sid, 'source_id': 's1', 'voice_id': 'v', 'character_id': 'car',
             'in_sec': start, 'out_sec': start+2, 'shot_start_sec': start, 'shot_end_sec': start+5,
             'shot_id': sid, 'duplicate_group': '', 'reference_fallback': False,
             'audio_uncertain': True, 'reason': 'Clear expressive mouth and compatible sound'}
            for sid, start in [('a', 0), ('b', 20)]]
    return {'characters': [character], 'selections': rows, 'shortage_reason': ''}, slots, sources, voices


def check(result, slots, sources, voices, refs=('reference-1',)):
    return validate_selection(result, slots, sources, voices, refs, gap=5)


def test_unused_nested_tail_is_filled_but_only_timeline_prefix_requires_evidence():
    result, slots, sources, voices = fixture()
    slots[0].update(required_duration_sec=5.68, visible_duration_sec=.56)
    result['selections'][1]['source_id'] = 's2'
    result['selections'][0].update(out_sec=5.68, shot_end_sec=.8,
        timeline_visible_prefix=True, identity_end_sec=.56)
    result['characters'][0]['appearances'][0]['end_sec'] = .56
    rows, errors, _ = check(result, slots, sources, voices)
    assert not errors and rows[0]['out_sec'] == 5.68
    assert rows[0]['identity_end_sec'] == .56
    result['selections'][0]['identity_end_sec'] = .5
    _, errors, _ = check(result, slots, sources, voices)
    assert any('every source frame' in e['error'] for e in errors)


def test_unused_tail_exception_cannot_be_invented_or_exceed_original_source():
    result, slots, sources, voices = fixture()
    slots[0].update(required_duration_sec=5.68)
    result['selections'][0].update(out_sec=5.68, shot_end_sec=.8,
        timeline_visible_prefix=True, identity_end_sec=.56)
    _, errors, _ = check(result, slots, sources, voices)
    assert any('not authorized' in e['error'] for e in errors)
    slots[0]['visible_duration_sec'] = .56
    sources[0]['duration_sec'] = 5.
    result['characters'][0]['appearances'][0]['end_sec'] = .56
    _, errors, _ = check(result, slots, sources, voices)
    assert any('bounds' in e['error'] for e in errors)


def test_hidden_nested_padding_does_not_consume_the_gap_between_displayed_cuts():
    result, slots, sources, voices = fixture()
    slots[0].update(required_duration_sec=9., visible_duration_sec=.56)
    result['selections'][0].update(out_sec=9., shot_end_sec=.8,
        timeline_visible_prefix=True, identity_end_sec=.56)
    result['selections'][1].update(in_sec=6., out_sec=8., shot_start_sec=6., shot_end_sec=10.)
    rows, errors, _ = check(result, slots, sources, voices)
    assert len(rows) == 2 and not errors  # Displayed cuts have a 5.44s gap.
    result['selections'][1].update(in_sec=5., out_sec=7., shot_start_sec=5.)
    _, errors, _ = check(result, slots, sources, voices)
    assert any('gap' in e['error'] for e in errors)  # 4.44s is still rejected.


def test_tail_fallback_requires_opt_in_and_only_prefix_character_evidence():
    result, slots, sources, voices = fixture()
    slots[0]['required_duration_sec'] = 9.28
    result['selections'][1]['source_id'] = 's2'
    result['selections'][0].update(out_sec=9.28, shot_end_sec=7.8, transition_tail_fallback_sec=2.)
    result['characters'][0]['appearances'][0]['end_sec'] = 7.28
    rows, errors, _ = check(result, slots, sources, voices)
    assert len(rows) == 1 and 'Unauthorized' in errors[0]['error']
    rows, errors, _ = validate_selection(result, slots, sources, voices, ('reference-1',), 5,
        transition_tail_fallback_sec=2)
    assert not errors and len(rows) == 2
    result['characters'][0]['appearances'][0]['end_sec'] = 7.
    rows, errors, _ = validate_selection(result, slots, sources, voices, ('reference-1',), 5,
        transition_tail_fallback_sec=2)
    assert any(e['slot_id'] == 'a' and 'appearance evidence' in e['error'] for e in errors)


def test_tail_fallback_never_extends_past_available_source():
    result, slots, sources, voices = fixture()
    sources[0]['duration_sec'] = 8.
    result['characters'][0]['appearances'][0]['end_sec'] = 7.28
    slots[0]['required_duration_sec'] = 9.28
    result['selections'][0].update(out_sec=9.28, shot_end_sec=7.8, transition_tail_fallback_sec=2.)
    _, errors, _ = validate_selection(result, slots, sources, voices, ('reference-1',), 5,
        transition_tail_fallback_sec=2)
    assert any(e['slot_id'] == 'a' and 'bounds' in e['error'] for e in errors)


def test_unrestricted_scene_changes_require_the_authorized_final_slot():
    result, slots, sources, voices = fixture()
    slots[0]['required_duration_sec'] = 9.28
    result['selections'][1]['source_id'] = 's2'
    result['selections'][0].update(out_sec=9.28, shot_end_sec=3.,
        final_scene_transition_fallback=True, identity_end_sec=1.)
    result['characters'][0]['appearances'][0]['end_sec'] = 1.
    rows, errors, _ = check(result, slots, sources, voices)
    assert len(rows) == 1 and 'final timeline block' in errors[0]['error']
    slots[0]['allow_scene_changes'] = True
    rows, errors, _ = check(result, slots, sources, voices)
    assert not errors and len(rows) == 2


def test_final_block_still_reserves_its_full_source_interval_and_gap():
    result, slots, sources, voices = fixture()
    slots[0].update(required_duration_sec=9.28, allow_scene_changes=True)
    result['selections'][0].update(out_sec=9.28, shot_end_sec=3.,
        final_scene_transition_fallback=True, identity_end_sec=1.)
    result['selections'][1].update(in_sec=5., out_sec=7., shot_start_sec=5., shot_end_sec=8.)
    rows, errors, _ = check(result, slots, sources, voices)
    assert len(rows) == 1 and any('gap' in e['error'] for e in errors)


@pytest.mark.parametrize('description', ['Talking tree', 'Talking car', 'Human', 'Creature', 'Personified cup'])
def test_nonhuman_characters_are_not_filtered(description):
    result, *args = fixture()
    result['characters'][0]['description'] = description
    rows, errors, ranking = check(result, *args)
    assert len(rows) == 2 and not errors
    assert ranking[0]['estimated_exposure_sec'] == 200


@pytest.mark.parametrize('change,reason', [
    ({'shot_id': 'a'}, 'Continuous shot'),
    ({'shot_start_sec': 4}, 'Continuous shot'),
    ({'in_sec': 4, 'out_sec': 6, 'shot_start_sec': 4, 'shot_end_sec': 10}, 'Continuous shot'),
    ({'in_sec': 6, 'out_sec': 8, 'shot_start_sec': 5, 'shot_end_sec': 10}, 'gap'),
    ({'out_sec': 23}, 'duration'),
    ({'in_sec': -1}, 'bounds'),
    ({'in_sec': math.nan}, 'bounds'),
    ({'out_sec': math.inf}, 'bounds'),
    ({'voice_id': 'missing'}, 'missing'),
    ({'source_id': 'missing'}, 'missing'),
    ({'character_id': 'missing'}, 'missing'),
    ({'reference_fallback': True}, 'fallback'),
    ({'reason': ''}, 'reason'),
    ({'audio_uncertain': 'false'}, 'flags'),
])
def test_rejects_invalid_allocations(change, reason):
    result, *args = fixture()
    result['selections'][1].update(change)
    rows, errors, _ = check(result, *args)
    assert len(rows) == 1
    assert any(reason in e['error'] for e in errors)


def test_duplicate_group_blocks_across_sources():
    result, *args = fixture()
    result['selections'][1]['source_id'] = 's2'
    for row in result['selections']:
        row['duplicate_group'] = 'same-footage'
    rows, errors, _ = check(result, *args)
    assert len(rows) == 1 and 'Repeated footage' in errors[0]['error']


def test_shortage_fallback_permits_different_segments_in_same_continuous_shot():
    result,slots,sources,voices=fixture()
    for row in result['selections']:
        row.update(shot_id='long-shot',shot_start_sec=0,shot_end_sec=60)
    strict,errors,_=check(result,slots,sources,voices)
    assert len(strict)==1 and errors
    rows,errors,_=validate_selection(result,slots,sources,voices,('reference-1',),5,allow_same_shot=True)
    assert not errors and len(rows)==2
    assert [r['same_shot_fallback'] for r in rows]==[False,True]


def test_same_shot_fallback_does_not_allow_overlap_or_reduce_gap():
    result,slots,sources,voices=fixture()
    for row in result['selections']:
        row.update(shot_id='long-shot',shot_start_sec=0,shot_end_sec=60)
    result['selections'][1].update(in_sec=1,out_sec=3)
    rows,errors,_=validate_selection(result,slots,sources,voices,('reference-1',),5,allow_same_shot=True)
    assert len(rows)==1 and 'Overlap/source gap' in errors[0]['error']


def test_analyzed_window_blocks_fabricated_appearance_and_cuts_beyond_supplied_video():
    result,slots,sources,voices=fixture()
    for source in sources:
        source.update(analyzed_start_sec=0,analyzed_end_sec=15)
    rows,errors,_=check(result,slots,sources,voices)
    assert not rows and errors
    for span in result['characters'][0]['appearances']:
        span['end_sec']=15
    rows,errors,_=check(result,slots,sources,voices)
    assert len(rows)==1 and any(e['slot_id']=='b' and 'bounds' in e['error'] for e in errors)


def test_shortage_is_never_filled_by_repeating():
    result, *args = fixture()
    result['selections'].pop()
    rows, errors, _ = check(result, *args)
    assert len(rows) == 1 and errors == [{'slot_id': 'b', 'error': 'No valid distinct shot selected'}]


def test_multiple_references_and_fallback():
    result, *args = fixture()
    result['characters'][0]['reference_ids'] += ['reference-2']
    other = deepcopy(result['characters'][0])
    other.update(id='tree', reference_ids=[])
    result['characters'].append(other)
    result['selections'][1].update(character_id='tree', reference_fallback=True)
    rows, errors, _ = check(result, *args, refs=('reference-1', 'reference-2'))
    assert not errors and rows[1]['reference_fallback']


def test_removed_references_restore_automatic_mode_and_union_exposure():
    result, *args = fixture()
    result['characters'][0]['reference_ids'] = []
    result['characters'][0]['appearances'].append({'source_id': 's1', 'start_sec': 0, 'end_sec': 50})
    other = deepcopy(result['characters'][0])
    other.update(id='tree', appearances=[{'source_id': 's1', 'start_sec': 1, 'end_sec': 2}] * 50)
    result['characters'].append(other)
    rows, errors, ranking = check(result, *args, refs=())
    assert not errors and len(rows) == 2
    assert [(r['id'], r['estimated_exposure_sec']) for r in ranking] == [('car', 200), ('tree', 1)]


@pytest.mark.parametrize('result', [None, [], {}, {'characters': None, 'selections': None}, {'selections': [None]}])
def test_malformed_results_become_repair_errors(result):
    _, *args = fixture()
    rows, errors, _ = check(result, *args)
    assert not rows and errors
