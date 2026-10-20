from copy import deepcopy
from pathlib import Path
import time

import pytest

from autoedit.ai import discovery
from autoedit.ai.selection import validate_selection
from test_ai_local import media


@pytest.mark.parametrize('settings', [{'local_analysis_mode': 'unknown'},
    {'local_target_seconds': 0}, {'local_target_seconds': True}, {'local_target_seconds': float('inf')},
    {'local_max_seconds': -1}, {'local_max_seconds': True}, {'local_max_seconds': float('nan')}])
def test_invalid_discovery_settings_are_rejected_before_media_is_read(settings):
    with pytest.raises(ValueError):
        discovery.discovery_settings(settings)


@pytest.fixture
def search(monkeypatch, tmp_path):
    import autoedit.ai.local as local
    import autoedit.ai.visual as visual
    monkeypatch.setattr(local, 'transition_safe', lambda *a: True)
    monkeypatch.setattr(visual, 'fingerprints', lambda *a: [])
    monkeypatch.setattr(visual, 'similar', lambda *a: False)
    contract = {'backend_identity': {'model': 'test', 'digest': 'weights'},
        'local_analysis': discovery.discovery_settings({}),
        'video_sampling': {'video_fps': 3, 'shot_fallback': True},
        'sources': [{'id': 's', 'duration_sec': 1000., 'analyzed_start_sec': 0., 'analyzed_end_sec': 1000.}],
        'references': [], 'source_gap_sec': 5.,
        'slots': [{'id': 'a', 'required_duration_sec': 1.}, {'id': 'b', 'required_duration_sec': 1.}],
        'voices': [{'voice_id': 'v', 'name': 'hi', 'duration_sec': 1.}]}
    class Client:
        def check_resources(self):
            pass
    selector = discovery.ProgressiveSelector(Client(), tmp_path, lambda _: None)
    return contract, selector


def windows(count=100, shared_shot=False):
    for i in range(count):
        yield {'source_id': 's', 'shot': 0 if shared_shot else i,
            'shot_start_sec': 0. if shared_shot else i*10.,
            'shot_end_sec': 1000. if shared_shot else i*10.+4.,
            'start_sec': i*10.+.1, 'end_sec': i*10.+3.9, 'image': 'unused.jpg'}


def label_all(batch, *args):
    return {'cast': [{'id': 'gray-cat', 'reference_ids': []}], 'labels': [[0, 2] for _ in batch]}


def observation(value):
    name, mouth, reference = value.split('|')
    result = {'present': name != 'none', 'identity': name}
    if int(reference) >= 0:
        result['reference'] = int(reference)
    return result


def test_search_stops_when_slots_are_valid_without_reading_the_whole_source(search, monkeypatch):
    contract, selector = search
    decoded, requests = [], []
    def stream(*args, **kwargs):
        for window in windows():
            decoded.append(window)
            yield window
    def classify(batch, *args):
        requests.append(batch)
        return label_all(batch)
    monkeypatch.setattr(discovery, 'distributed_candidates', stream)
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    accepted, issues, _ = validate_selection(result, contract['slots'], contract['sources'], contract['voices'], gap=5)
    assert not issues and len(accepted) == 2
    assert len(decoded) == 2 < 100 and len(requests) == 1
    assert result['analysis_scope']['searched_until_sec']['s'] == 14
    assert all(not r['local_character_verified'] and r['audio_uncertain'] for r in accepted)


def test_shortage_expands_search_instead_of_assuming_a_three_minute_limit(search, monkeypatch):
    contract, selector = search
    requests = []
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows())
    def classify(batch, *args):
        requests.append(batch)
        result = label_all(batch)
        if len(requests) == 1:
            result['labels'] = [[-1, 0] for _ in batch]
            result['labels'][0] = [0, 2]
        return result
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(result['selections']) == 2 and len(requests) == 2
    assert result['analysis_scope']['searched_until_sec']['s'] == 24


def test_unrelated_characters_do_not_end_reference_search_prematurely(search, monkeypatch):
    contract, selector = search
    contract['references'] = [{'id': 'ref-1'}]
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows())
    calls = []
    def classify(batch, *args):
        calls.append(batch)
        result = label_all(batch)
        if len(calls) == 2:
            result['cast'][0]['reference_ids'] = ['ref-1']
        return result
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(calls) == 2 and len(result['selections']) == 2
    assert not any(r['reference_fallback'] for r in result['selections'])


def test_reference_search_disables_face_matching_and_uses_eager_neighbour_search(search, monkeypatch):
    from autoedit.ai import people, appearance
    contract, selector = search
    contract['references'] = [{'id': 'ref-1', 'path': 'reference.png'}]
    selector.client.single_image = True
    class People:
        stats = {}
        human_mode = True
        def __init__(self, *, enable_identity):
            assert enable_identity is False
        def preview(self, path):
            return {'subject_count': 2, 'priority': 0, 'reference_appearance_score': .8}
        def classify(self, *a, **kw):
            pytest.fail('Reference appearance search must never call facial identity matching')
    class Appearance:
        human_references = True
        def __init__(self, *a):
            pass
    monkeypatch.setattr(people, 'FastPeople', People)
    monkeypatch.setattr(appearance, 'ReferenceAppearance', Appearance)
    def stream(*args, **kwargs):
        assert kwargs['eager'] is True and kwargs['quick'] is True
        assert callable(kwargs['focus_seconds'])
        return windows(count=2)
    monkeypatch.setattr(discovery, 'distributed_candidates', stream)
    def classify(batch, *args):
        result = label_all(batch)
        result['cast'][0].update(id='reference-look-ref-1', reference_ids=['ref-1'])
        result['details'] = [{'subject_count': 2, 'sample_frames': 2, 'identity_consistent': False,
            'reference_match_scope': 'costume_appearance', 'appearance_sample_frames': 2} for _ in batch]
        return result
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(result['selections']) == 2
    assert result['analysis_scope']['reference_match_scope'] == 'costume_appearance'
    assert not result['analysis_scope']['solo_first']
    assert all(r['reference_match_scope'] == 'costume_appearance' for r in result['selections'])


def test_reference_search_secures_long_continuous_fills_before_short_ones(search, monkeypatch):
    contract, selector = search
    contract['references'] = [{'id': 'ref-1'}]
    contract['slots'] = [{'id': 'short', 'required_duration_sec': .56},
        {'id': 'long', 'required_duration_sec': 5.68},
        {'id': 'medium', 'required_duration_sec': 3.12},
        {'id': 'final', 'required_duration_sec': 9.28, 'allow_scene_changes': True}]
    stages, checked = [], []
    def stream(*args, **kwargs):
        assert args[2] == 5.68  # Final block's full 9.28s does not need continuous evidence.
        stages.append(kwargs['needed_duration']())
        for i, length in enumerate((1.8, 6.2, 3.6, 1.4)):
            start = i*20.
            if length < kwargs['needed_duration']()+.16:
                continue
            yield {'source_id': 's', 'shot': i, 'shot_start_sec': start,
                'shot_end_sec': start+length+.16, 'start_sec': start+.08,
                'end_sec': start+length+.08, 'image': 'unused.jpg'}
    def classify(batch, *args):
        checked.extend(w['end_sec']-w['start_sec']-.16 for w in batch)
        result = label_all(batch)
        result['cast'][0]['reference_ids'] = ['ref-1']
        return result
    monkeypatch.setattr(discovery, 'distributed_candidates', stream)
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(result['selections']) == 4
    assert checked[0] == pytest.approx(5.68)
    assert stages[:3] == [5.68, 3.12, 1.]
    assert next(r for r in result['selections'] if r['slot_id'] == 'final')['final_scene_transition_fallback']


def test_long_reference_search_cannot_starve_short_partial_results(search, monkeypatch):
    contract, selector = search
    contract['references'] = [{'id': 'ref-1'}]
    contract['slots'] = [{'id': 'long', 'required_duration_sec': 5.68},
                         {'id': 'short', 'required_duration_sec': .56}]
    now = [0.]
    monkeypatch.setattr(discovery.time, 'monotonic', lambda: now[0])
    stages = []
    def stream(*args, **kwargs):
        now[0] = 80.
        stages.append(kwargs['needed_duration']())
        if stages[-1] > 3.:
            return
        yield next(windows(count=1))
    def classify(batch, *args):
        result = label_all(batch)
        result['cast'][0]['reference_ids'] = ['ref-1']
        return result
    monkeypatch.setattr(discovery, 'distributed_candidates', stream)
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert stages[0] == .56
    assert [r['slot_id'] for r in result['selections']] == ['short']


def test_appearance_shortage_never_replaces_reference_with_unrelated_people(search, monkeypatch):
    contract, selector = search
    contract['references'] = [{'id': 'ref-1'}]
    selector.client.reference_appearance = object()
    selector.group_phase = True
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows(count=1))
    monkeypatch.setattr(discovery, 'classify', label_all)
    result = selector.select(contract)
    assert not result['selections']


def test_same_shot_fallback_only_after_strict_allocation_is_short(search, monkeypatch):
    contract, selector = search
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows(shared_shot=True))
    monkeypatch.setattr(discovery, 'classify', label_all)
    result = selector.select(contract)
    assert result['_autoedit_dedup_policy'] == 'distinct_segments'
    accepted, errors, _ = validate_selection(result, contract['slots'], contract['sources'], contract['voices'], gap=5, allow_same_shot=True)
    assert not errors and len(accepted) == 2
    assert accepted[1]['same_shot_fallback']


def test_group_scenes_cannot_fill_slots_before_later_solo_candidates(search, monkeypatch):
    contract, selector = search
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows(count=6))
    calls = []
    def classify(batch, *args):
        calls.append(batch)
        result = label_all(batch)
        result['details'] = [{'subject_count': 3 if len(calls) == 1 else 1, 'sample_frames': 2}
                             for _ in batch]
        return result
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(calls) == 2 and len(result['selections']) == 2
    assert all(r['subject_count'] == 1 and not r['group_scene_fallback'] for r in result['selections'])
    assert not result['analysis_scope']['group_fallback_enabled']


def test_group_fallback_is_used_only_after_solo_search_is_exhausted(search, monkeypatch):
    contract, selector = search
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows(count=2))
    def classify(batch, *args):
        result = label_all(batch)
        result['details'] = [{'subject_count': 2, 'sample_frames': 2} for _ in batch]
        return result
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(result['selections']) == 2
    assert all(r['subject_count'] == 2 and r['group_scene_fallback'] for r in result['selections'])
    assert result['analysis_scope']['group_fallback_enabled']


def test_fast_search_retries_native_frames_before_using_group_fallback(search, monkeypatch):
    from autoedit.ai import people
    contract, selector = search
    selector.client.single_image = True
    class People:
        human_mode = True
        stats = {}
        def preview(self, path):
            return {'subject_count': 1, 'priority': 0}
        def classify(self, *args, **kwargs):
            return {'id': 'person-1', 'reference_ids': [], 'subject_count': 1,
                    'mouth': 1, 'sample_frames': 3, 'identity_sample_frames': 2,
                    'identity_consistent': True, 'method': 'cpu-people-2'}
    monkeypatch.setattr(people, 'FastPeople', People)
    phases = []
    def stream(*args, **kwargs):
        phases.append(kwargs['quick'])
        return windows(count=0 if kwargs['quick'] else 2)
    monkeypatch.setattr(discovery, 'distributed_candidates', stream)
    result = selector.select(contract)
    assert phases == [True, False] and len(result['selections']) == 2
    assert result['analysis_scope']['native_discovery_retry']
    assert not result['analysis_scope']['group_fallback_enabled']


@pytest.mark.parametrize('subject_count', [0, 1])
def test_nonhuman_solo_candidates_survive_a_human_detection_in_mixed_footage(search, monkeypatch, subject_count):
    from autoedit.ai import people
    contract, selector = search
    selector.client.single_image = True
    class People:
        human_mode = True
        stats = {}
        def preview(self, path):
            return {'priority': 2, 'subject_count': 1, 'human_count': 0}
        def classify(self, *a, **k):
            pytest.fail('Animal-only candidates must use the Qwen fallback')
    monkeypatch.setattr(people, 'FastPeople', People)
    def stream(*a, **k):
        for window in windows(count=2):
            yield {**window, 'subject_preview': {'priority': 2, 'subject_count': subject_count, 'human_count': 0}}
    monkeypatch.setattr(discovery, 'distributed_candidates', stream)
    requests = []
    def classify(batch, *a):
        requests.append(batch)
        result = label_all(batch)
        result['details'] = [{'subject_count': 1, 'sample_frames': 2} for _ in batch]
        return result
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(result['selections']) == 2 and len(requests) == 2
    assert not result['analysis_scope']['group_fallback_enabled']


def test_budget_expiry_retains_a_partial_review_without_inventing_fills(search):
    contract, selector = search
    selector.streams = []
    selector.deadline = time.monotonic()-1
    result = selector.select(contract)
    assert not result['selections'] and result['shortage_reason']
    assert result['analysis_scope']['mode'] == 'until_filled'


def test_expired_repair_retains_valid_partial_rows_and_removes_rejections(search):
    contract, selector = search
    selector.deadline = time.monotonic()-1
    selector.best = {'characters': [], 'selections': [{'slot_id': 'a'}, {'slot_id': 'b'}], 'shortage_reason': ''}
    result = selector.result(contract, None, [{'slot_id': 'b', 'error': 'repeated frames'}], True)
    assert [r['slot_id'] for r in result['selections']] == ['a']


def test_pipeline_deadline_limits_model_time_as_well_as_discovery(search, monkeypatch):
    contract, selector = search
    selector.client.job_deadline = time.monotonic()-1
    # Creation is cheap; iteration would decode and must never occur here.
    def stream(*a, **k):
        pytest.fail('Expired job decoded video')
        yield
    monkeypatch.setattr(discovery, 'distributed_candidates', stream)
    assert not selector.select(contract)['selections']


def test_candidate_generator_is_lazy_and_shorter_shots_remain_usable(media, tmp_path):
    gen = discovery.candidate_stream(media, 3, 4., tmp_path, time.monotonic()+10,
                                     lambda _: None, minimum=1.)
    first = next(gen)
    assert first['shot_start_sec'] == 0 and first['shot_end_sec'] == pytest.approx(3)
    assert first['end_sec']-first['start_sec'] >= 1
    assert Path(first['image']).is_file()
    gen.close()


def test_long_continuous_take_yields_before_reaching_the_end(media, tmp_path, monkeypatch):
    import cv2
    # Treat the generated two-scene fixture as a continuous take to exercise
    # discovery without waiting for a shot-boundary event or the source end.
    monkeypatch.setattr(cv2, 'compareHist', lambda *a: 0.)
    gen = discovery.candidate_stream(media, 3, 1., tmp_path, time.monotonic()+10, lambda _: None)
    first = next(gen)
    assert first['shot'] == 0 and not first['shot_bounds_complete']
    assert first['shot_end_sec'] < media['duration_sec']
    assert first['end_sec'] <= first['shot_end_sec']
    gen.close()


def test_finished_shot_retains_middle_candidates_after_rolling_windows(media, tmp_path, monkeypatch):
    import cv2
    from PIL import Image
    monkeypatch.setattr(cv2, 'compareHist', lambda *a: 0.)
    found = list(discovery.candidate_stream(media, 3, 1., tmp_path,
        time.monotonic()+10, lambda _: None))
    assert any(not w['shot_bounds_complete'] and w['shot_end_sec'] < 6. for w in found)
    complete = found[-1]
    assert complete['shot_bounds_complete']
    assert complete['start_sec'] == pytest.approx(.08)
    assert complete['end_sec'] == pytest.approx(5.92, abs=.001)
    assert complete['end_sec']-complete['start_sec'] > 5.8
    assert found[0]['image'] != complete['image']
    with Image.open(found[0]['image']) as first, Image.open(complete['image']) as last:
        assert first.getpixel((0, 0))[0] > first.getpixel((0, 0))[2]
        assert last.getpixel((0, 0))[2] > last.getpixel((0, 0))[0]


def test_candidate_generator_resumes_across_cuts_and_finishes_last_shot(media, tmp_path):
    candidates = list(discovery.candidate_stream(media, 3, 1., tmp_path,
        time.monotonic()+10, lambda _: None))
    assert len(candidates) >= 2
    assert len({candidate['shot'] for candidate in candidates}) >= 2
    assert candidates[-1]['shot_bounds_complete']
    assert all(Path(candidate['image']).is_file() for candidate in candidates)


def test_region_order_covers_late_source_early_and_eventually_every_region():
    order = discovery.region_order(150)
    assert order[:3] == [0, 128, 64]
    assert sorted(order) == list(range(150))


def test_distributed_search_keeps_unused_windows_for_exhaustive_fallback(monkeypatch, tmp_path):
    source = {'id': 's', 'duration_sec': 36.}
    def stream(region, *a, **k):
        for i in range(2):
            start = region['analyzed_start_sec']+i*4
            yield {'source_id': 's', 'shot': i, 'shot_start_sec': start,
                'shot_end_sec': start+3, 'start_sec': start+.1, 'end_sec': start+2.9}
    monkeypatch.setattr(discovery, 'candidate_stream', stream)
    found = list(discovery.distributed_candidates(source, 3, 1, tmp_path,
        time.monotonic()+10, lambda _: None))
    assert len(found) == 6
    assert [w['search_region'][0] for w in found[:3]] == [0, 24, 12]
    assert len({w['start_sec'] for w in found}) == 6
    assert found[1]['shot'] == found[2]['shot'] == 'unknown'


def test_reference_candidates_are_eager_and_search_near_recent_matches(monkeypatch, tmp_path):
    source = {'id': 's', 'duration_sec': 120.}
    visited = []
    def stream(region, *a, **k):
        visited.append(region['analyzed_start_sec'])
        for i in range(2):
            start = region['analyzed_start_sec']+i*4
            yield {'source_id': 's', 'shot': i, 'shot_start_sec': start,
                'shot_end_sec': start+3, 'start_sec': start+.1, 'end_sec': start+2.9}
    monkeypatch.setattr(discovery, 'candidate_stream', stream)
    focus = []
    candidates = discovery.distributed_candidates(source, 3, 1, tmp_path,
        time.monotonic()+10, lambda _: None, eager=True, focus_seconds=lambda: focus)
    first = next(candidates)['start_sec']
    focus.append(2.)
    assert {first, next(candidates)['start_sec']} == {.1, 4.1}
    assert visited == [0.]
    assert next(candidates)['search_region'][0] == 12.
    assert visited == [0., 12.]


def test_region_seek_does_not_decode_earlier_frames_as_candidates(media, tmp_path):
    region = {**media, 'analyzed_start_sec': 3., 'analyzed_end_sec': 5.5}
    found = list(discovery.candidate_stream(region, 3, 1, tmp_path,
        time.monotonic()+10, lambda _: None))
    assert found and all(w['start_sec'] >= 3 for w in found)
    assert all(not w['shot_bounds_complete'] for w in found)


def test_reference_focus_cannot_starve_distant_regions(monkeypatch, tmp_path):
    visited = []
    def stream(region, *a, **k):
        start = region['analyzed_start_sec']
        visited.append(start)
        yield {'source_id': 's', 'shot': 0, 'shot_start_sec': start,
               'shot_end_sec': start+3, 'start_sec': start+.1, 'end_sec': start+2.9}
    monkeypatch.setattr(discovery, 'candidate_stream', stream)
    source = {'id': 's', 'duration_sec': 1440.}
    gen = discovery.distributed_candidates(source, 3, 1, tmp_path,
        time.monotonic()+10, lambda _: None, eager=True, focus_seconds=lambda: [100.])
    first = [next(gen) for _ in range(8)]
    assert any(w['start_sec'] > 700 for w in first)
    found = first + list(gen)
    assert sorted(visited) == list(range(0, 1440, 12))
    assert len(found) == len(visited) == 120


def test_reference_search_without_matches_keeps_distributed_order(search):
    _, selector = search
    assert selector.reference_focus('s') == []


def test_fourteen_fills_use_small_batches_and_preserve_valid_previous_cuts(search, monkeypatch):
    contract, selector = search
    contract['slots'] = [{'id': f'slot-{i}', 'required_duration_sec': 1.} for i in range(14)]
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows())
    calls = []
    def classify(batch, *args):
        calls.append(len(batch))
        return label_all(batch)
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(result['selections']) == 14 and calls == [6, 6, 2]
    assert result['analysis_scope']['candidate_count'] == 14


def test_unusable_transitions_are_rejected_before_model_inference(search, monkeypatch):
    import autoedit.ai.local as local
    contract, selector = search
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows())
    monkeypatch.setattr(local, 'transition_safe', lambda source, row, cache: row['in_sec'] >= 10)
    batches = []
    def classify(batch, *args):
        batches.append(batch)
        return label_all(batch)
    monkeypatch.setattr(discovery, 'classify', classify)
    assert len(selector.select(contract)['selections']) == 2
    assert min(w['start_sec'] for w in batches[0]) >= 10


def test_safe_alternative_inside_a_candidate_is_tried_before_discarding_footage(search, monkeypatch):
    from autoedit.ai import local
    contract, selector = search
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **kw: windows(count=2))
    monkeypatch.setattr(local, 'transition_safe', lambda source, row, cache:
                        row['in_sec'] < .3 or row['in_sec'] >= 10.)
    calls = []
    def classify(batch, *args):
        calls.append(batch)
        return label_all(batch)
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(result['selections']) == 2 and len(calls) == 1
    assert result['selections'][0]['in_sec'] == pytest.approx(.18)


def test_hidden_padding_does_not_block_a_later_displayed_cut(search, monkeypatch):
    contract, selector = search
    contract['slots'] = [{'id': 'nested', 'required_duration_sec': 3.12, 'visible_duration_sec': .68},
                         {'id': 'plain', 'required_duration_sec': .68}]
    def stream(*a, **kw):
        for start in (0.,6.):
            yield {'source_id':'s','shot':str(start),'shot_start_sec':start,'shot_end_sec':start+4.,
                   'start_sec':start+.1,'end_sec':start+3.9,'image':'unused.jpg'}
    monkeypatch.setattr(discovery,'distributed_candidates',stream)
    monkeypatch.setattr(discovery,'classify',label_all)
    result=selector.select(contract)
    rows, errors, _=validate_selection(result,contract['slots'],contract['sources'],contract['voices'],gap=5.)
    assert not errors and len(rows)==2
    first,second=rows
    assert second['in_sec'] >= first['identity_end_sec']+5.-1e-6
    assert second['in_sec'] < first['out_sec']+5.


def test_time_target_is_not_a_hard_stop_when_valid_footage_remains(search, monkeypatch):
    contract, selector = search
    selector.client.target_deadline = time.monotonic()-1
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows())
    monkeypatch.setattr(discovery, 'classify', label_all)
    result = selector.select(contract)
    assert len(result['selections']) == 2 and result['analysis_scope']['target_exceeded']
    assert result['analysis_scope']['stop_reason'] == 'enough_slots'
    assert selector.target_logged


def test_explicit_maximum_still_imposes_a_hard_deadline(search, monkeypatch):
    contract, selector = search
    contract['local_analysis']['max_seconds'] = .01
    selector.client.job_deadline = time.monotonic()-1
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: windows())
    result = selector.select(contract)
    assert not result['selections'] and result['analysis_scope']['stop_reason'] == 'time_budget'


def test_last_verified_candidate_is_finalized_without_extending_model_deadline(search, monkeypatch):
    contract, selector = search
    contract['slots'] = [{'id':str(i),'required_duration_sec':1.} for i in range(7)]
    now=[0.]
    monkeypatch.setattr(discovery.time,'monotonic',lambda:now[0])
    selector.client.job_deadline=1.
    selector.client.finalization_deadline=10.
    monkeypatch.setattr(discovery,'distributed_candidates',lambda *a,**kw:windows(count=7))
    calls=[]
    def classify(batch, contract, client, cache, registry, deadline, log):
        assert deadline == 1.
        calls.append(len(batch))
        if len(calls)==2:
            now[0]=1.5
        return label_all(batch)
    monkeypatch.setattr(discovery,'classify',classify)
    result=selector.select(contract)
    assert len(result['selections'])==7 and calls==[6,1]
    assert result['analysis_scope']['stop_reason']=='enough_slots'


def test_completed_region_discovery_resumes_without_decoding_again(media, tmp_path, monkeypatch):
    found = list(discovery.distributed_candidates(media, 3, 1., tmp_path,
        time.monotonic()+10, lambda _: None))
    assert found
    monkeypatch.setattr(discovery, 'candidate_stream', lambda *a, **k: pytest.fail('Completed regions decoded again'))
    assert list(discovery.distributed_candidates(media, 3, 1., tmp_path,
        time.monotonic()+10, lambda _: None)) == found


def test_long_shot_crossing_region_edge_is_available_at_full_duration(monkeypatch, tmp_path):
    source = {'id': 's', 'duration_sec': 90.}
    # A 10-second take crosses the first region edge (32.84s). Both halves
    # were previously too short for the 9.28-second fill.
    def stream(region, *a, **k):
        lo, hi = max(28., region['analyzed_start_sec']), min(38., region['analyzed_end_sec'])
        if hi > lo:
            yield {'source_id': 's', 'shot': 0, 'shot_start_sec': lo,
                'shot_end_sec': hi, 'start_sec': lo+.08, 'end_sec': hi-.08}
    monkeypatch.setattr(discovery, 'candidate_stream', stream)
    found = list(discovery.distributed_candidates(source, 3, 9.28, tmp_path,
        time.monotonic()+10, lambda _: None, minimum=.56))
    assert any(w['end_sec']-w['start_sec'] >= 9.28+.16 for w in found)


def test_short_candidates_are_not_screened_when_only_long_fills_remain(monkeypatch, tmp_path):
    source = {'id': 's', 'duration_sec': 60.}
    required = [1.]
    screened = []
    def stream(region, *a, **k):
        start = region['analyzed_start_sec']
        for length in (2., 7.):
            yield {'source_id': 's', 'shot': 0, 'shot_start_sec': start,
                'shot_end_sec': start+length, 'start_sec': start+.08,
                'end_sec': start+length-.08, 'image': str((start, length))}
    def screen(image):
        screened.append(image)
        return {'priority': 0}
    monkeypatch.setattr(discovery, 'candidate_stream', stream)
    candidates = discovery.distributed_candidates(source, 3, 6., tmp_path,
        time.monotonic()+10, lambda _: None, minimum=1., screen=screen,
        needed_duration=lambda: required[0])
    next(candidates)
    required[0] = 6.
    remaining = list(candidates)
    assert len(screened) == 4  # Two initially, then only the long cut in each region.
    assert all(w['end_sec']-w['start_sec'] >= 6.+.16 for w in remaining)


def test_deferred_groups_are_tried_at_target_without_exhausting_source(search, monkeypatch):
    contract, selector = search
    selector.client.target_deadline = time.monotonic()-1
    decoded = []
    def stream(*a, **k):
        for window in windows():
            decoded.append(window)
            yield {**window, 'subject_preview': {'subject_count': 2, 'priority': 1}}
    def classify(batch, *a):
        result = label_all(batch)
        result['details'] = [{'subject_count': 2, 'sample_frames': 3} for _ in batch]
        return result
    monkeypatch.setattr(discovery, 'distributed_candidates', stream)
    monkeypatch.setattr(discovery, 'classify', classify)
    result = selector.select(contract)
    assert len(result['selections']) == 2 and len(decoded) == 2
    assert result['analysis_scope']['group_fallback_enabled']
    assert all(r['group_scene_fallback'] for r in result['selections'])


def test_coarse_candidate_still_has_to_pass_native_transition_audit(media, tmp_path, monkeypatch):
    import cv2
    from autoedit.ai.local import transition_safe
    # Simulate a cut missed by coarse discovery. Its full-resolution audit
    # must still reject a proposed four-second fill spanning the real cut.
    with monkeypatch.context() as patch:
        patch.setattr(cv2, 'compareHist', lambda *a: 0.)
        gen = discovery.candidate_stream(media, 3, 4., tmp_path,
            time.monotonic()+10, lambda _: None, minimum=1., quick=True)
        window = next(gen)
        gen.close()
    assert window['start_sec'] < 3 < window['end_sec']
    assert not transition_safe(media, {'in_sec': window['start_sec'],
        'out_sec': window['end_sec']}, tmp_path)


def test_tail_fallback_fills_full_duration_across_a_real_cut(media, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoedit.ai import people
    from test_ai_local import contract_for
    class People:
        human_mode = True
        stats = {}
        def preview(self, path):
            return {'subject_count': 1, 'priority': 0}
        def classify(self, *args, **kwargs):
            return {'id': 'person-1', 'reference_ids': [], 'subject_count': 1,
                'mouth': 1, 'sample_frames': 3, 'identity_sample_frames': 2,
                'identity_consistent': True, 'method': 'cpu-people-3'}
    monkeypatch.setattr(people, 'FastPeople', People)
    contract = contract_for(media)
    contract['slots'] = [{'id': 'a', 'required_duration_sec': 4.4}]
    contract['video_sampling'].update(shot_fallback=True, transition_tail_fallback_sec=2)
    selector = discovery.ProgressiveSelector(SimpleNamespace(single_image=True), tmp_path, lambda _: None)
    result = selector.select(contract)
    rows, errors, _ = validate_selection(result, contract['slots'], contract['sources'], contract['voices'],
        gap=.5, transition_tail_fallback_sec=2)
    assert not errors and len(rows) == 1
    row = rows[0]
    assert row['out_sec']-row['in_sec'] == pytest.approx(4.4)
    assert row['in_sec'] < 3 < row['out_sec']
    assert row['transition_tail_fallback_sec'] == 2
    assert row['out_sec']-2 < 3 and row['shot_end_sec'] == pytest.approx(3)
    assert result['characters'][0]['appearances'][0]['end_sec'] == pytest.approx(row['out_sec']-2)


def test_region_generator_releases_deferred_groups_when_solo_target_passes(monkeypatch, tmp_path):
    clock = [0.]
    monkeypatch.setattr(discovery.time, 'monotonic', lambda: clock[0])
    decoded = []
    def stream(region, *a, **k):
        start = region['analyzed_start_sec']
        decoded.append(start)
        for length in (2., 10.):
            yield {'source_id': 's', 'shot': 0, 'shot_start_sec': start,
                'shot_end_sec': start+length, 'start_sec': start+.08,
                'end_sec': start+length-.08, 'image': str(length)}
    monkeypatch.setattr(discovery, 'candidate_stream', stream)
    gen = discovery.distributed_candidates({'id': 's', 'duration_sec': 90.}, 3, 9.28,
        tmp_path, 100., lambda _: None, minimum=1., solo_deadline=5.,
        screen=lambda path: {'priority': 0 if path == '2.0' else 1})
    assert next(gen)['end_sec'] < 3
    clock[0] = 6.
    assert next(gen)['end_sec'] > 9
    assert decoded == [0.]
    gen.close()


def test_final_block_can_change_scene_more_than_two_seconds_before_its_end(media, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoedit.ai import people
    from test_ai_local import contract_for
    class People:
        human_mode = True
        stats = {}
        def preview(self, path):
            return {'subject_count': 1, 'priority': 0}
        def classify(self, *a, **k):
            return {'id': 'person-1', 'reference_ids': [], 'subject_count': 1,
                'mouth': 1, 'sample_frames': 3, 'identity_sample_frames': 2,
                'identity_consistent': True, 'method': 'cpu-people-3'}
    monkeypatch.setattr(people, 'FastPeople', People)
    contract = contract_for(media)
    contract['slots'] = [{'id': 'final', 'required_duration_sec': 4.4, 'allow_scene_changes': True}]
    contract['video_sampling']['shot_fallback'] = True
    result = discovery.ProgressiveSelector(SimpleNamespace(single_image=True), tmp_path, lambda _: None).select(contract)
    rows, errors, _ = validate_selection(result, contract['slots'], contract['sources'], contract['voices'], gap=.5)
    assert not errors and len(rows) == 1
    row = rows[0]
    assert row['out_sec']-row['in_sec'] == pytest.approx(4.4)
    assert row['in_sec'] < 3 < row['out_sec']-2
    assert row['identity_end_sec'] < 3 and row['final_scene_transition_fallback']
    assert row['subject_count_scope'] == 'opening_samples'


def test_unseen_nested_tail_does_not_require_an_unbroken_full_length_shot(media, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoedit.ai import people
    from test_ai_local import contract_for
    class People:
        human_mode = True
        stats = {}
        def preview(self, path):
            return {'subject_count': 1, 'priority': 0}
        def classify(self, *a, **k):
            return {'id': 'person-1', 'reference_ids': [], 'subject_count': 1,
                    'mouth': 1, 'sample_frames': 3, 'identity_sample_frames': 2,
                    'identity_consistent': True, 'method': 'cpu-people-3'}
    monkeypatch.setattr(people, 'FastPeople', People)
    contract = contract_for(media)
    contract['slots'] = [{'id': 'nested', 'required_duration_sec': 4.4, 'visible_duration_sec': .56}]
    contract['video_sampling']['shot_fallback'] = True
    result = discovery.ProgressiveSelector(SimpleNamespace(single_image=True), tmp_path, lambda _: None).select(contract)
    rows, errors, _ = validate_selection(result, contract['slots'], contract['sources'], contract['voices'], gap=.5)
    assert not errors and len(rows) == 1
    row = rows[0]
    assert row['out_sec']-row['in_sec'] == pytest.approx(4.4)
    assert row['identity_end_sec']-row['in_sec'] == pytest.approx(.56)
    assert row['in_sec'] < 3 < row['out_sec']
    assert row['timeline_visible_prefix'] and not row.get('final_scene_transition_fallback')
    assert row['subject_count_scope'] == 'visible_template_prefix'
