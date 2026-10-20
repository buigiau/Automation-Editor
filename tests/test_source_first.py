from collections import Counter
from copy import deepcopy
import time

import numpy as np
import pytest
import soundfile as sf

from autoedit.audio.matching import OpeningSoundIndex, assign_rows, choose_neutral, search_budget
from autoedit.audio.search import SourceMatchSession, SpeechWorker, SourceAudioError
from autoedit.audio.scene_match import choose_audio
from autoedit.ai.local import choose_voice
from test_audio_taxonomy import recording


def voice(stem, duration=.3):
    return dict(recording(stem, duration), voice_id=stem)


def speech(word='I', start=2., end=2.3, prob=.95):
    return {'words': [{'word': word, 'start': start, 'end': end, 'prob': prob}], 'events': []}


def test_source_word_wins_even_when_wrong_voice_is_unused_and_fits_better():
    pool = [voice('ai'), voice('aii'), voice('cười', 2.), voice('da', 2.)]
    item, evidence = OpeningSoundIndex(pool).select({'in_sec': 2., 'out_sec': 4.}, speech(), Counter(ai=100, aii=100))
    assert item['voice_id'] in {'ai', 'aii'}
    assert evidence['cut_onset_matched'] and evidence['source_word'] == 'I'
    assert not evidence['phoneme_sync_verified']


@pytest.mark.parametrize('hint', ['neutral', 'speech', 'laugh', 'shout', 'surprise'])
def test_every_visual_only_fallback_excludes_events_and_non_neutral_files(hint):
    pool = [voice('cười', 2.), voice('ho khụ khụ', 2.), voice('a', 2.), voice('da', .2)]
    assert choose_voice(pool, hint, 2., [])['voice_id'] == 'da'
    assert choose_neutral(pool, 2.)['voice_id'] == 'da'


def test_event_needs_source_and_visual_proof_and_always_requires_review():
    index = OpeningSoundIndex([voice('cười'), voice('da')])
    data = {'words': [], 'events': [{'action': 'LAUGH', 'start': 2., 'end': 2.3, 'prob': .75}]}
    cut = {'in_sec': 2., 'out_sec': 3.}
    item, evidence = index.select(cut, data)
    assert item['voice_id'] in {'da', 'ai'} and not evidence['cut_onset_matched']
    item, evidence = index.select(dict(cut, visual_action='LAUGH'), data)
    assert item['voice_id'] == 'cười'
    assert evidence['needs_review'] and evidence['confidence_kind'] == 'heuristic-not-calibrated'
    assert index.select(dict(cut, visual_action='LAUGH'), {'words': [], 'events': []})[0]['voice_id'] == 'da'


@pytest.mark.parametrize('first', [
    {'word': 'fine', 'start': 2., 'end': 2.2, 'prob': .99},
    {'word': 'I', 'start': 1.8, 'end': 2.2, 'prob': .99},
    {'word': 'I', 'start': 2., 'end': 2.2, 'prob': .2},
])
def test_later_word_cannot_replace_an_unsupported_opening(first):
    index = OpeningSoundIndex([voice('ai'), voice('da')])
    item, evidence = index.select({'in_sec': 2., 'out_sec': 3.},
        {'words': [first, {'word': 'I', 'start': 2.3, 'end': 2.5, 'prob': .99}], 'events': []})
    assert item['voice_id'] in {'da', 'ai'} and not evidence['cut_onset_matched']


def test_manual_change_loses_match_without_moving_cut_or_claiming_sync():
    pool = [voice('ai'), voice('da'), voice('cười')]
    sources = [{'id': 's', 'source_speech': speech()}]
    rows = [{'slot_id': 'one', 'source_id': 's', 'in_sec': 2., 'out_sec': 3., 'voice_id': 'ai'}]
    assign_rows(rows, sources, pool)
    assert rows[0]['audio_match']['cut_onset_matched']
    rows[0]['in_sec'], rows[0]['out_sec'], rows[0]['voice_id'] = 2.4, 3.4, 'cười'
    assign_rows(rows, sources, pool, manual=True)
    assert rows[0]['in_sec'] == 2.4 and rows[0]['voice_id'] == 'cười'
    assert rows[0]['audio_match']['manual_override']
    assert not rows[0]['audio_match']['cut_onset_matched'] and 'source_sound' not in rows[0]


def test_word_finished_before_cut_does_not_block_the_real_opening():
    data = speech()
    data['words'].insert(0, {'word': 'fine', 'start': 1.8, 'end': 1.99, 'prob': .99})
    _, evidence = OpeningSoundIndex([voice('ai'), voice('da')]).select({'in_sec': 2., 'out_sec': 3.}, data)
    assert evidence['cut_onset_matched'] and evidence['source_word'] == 'I'


def test_manual_supported_pronunciation_variant_is_preserved():
    rows = [{'source_id': 's', 'in_sec': 2., 'out_sec': 3., 'voice_id': 'mai'}]
    assign_rows(rows, [{'id': 's', 'source_speech': speech('my')}], [voice('my'), voice('mai')], manual=True)
    assert rows[0]['voice_id'] == 'mai'
    assert rows[0]['audio_match']['method'] == 'source-audio-pronunciation'


def test_standard_and_backend_matcher_agree_and_no_wav_is_chosen_by_mouth():
    pool = [voice('ai'), voice('da'), voice('cười')]
    cut = {'in_sec': 2., 'out_sec': 3.}
    item, evidence = choose_audio(pool, cut, {'source_speech': speech()}, Counter())
    expected, _ = OpeningSoundIndex(pool).select(cut, speech())
    assert item['path'] == expected['path']
    assert evidence['cut_onset_matched']
    item, evidence = choose_audio(pool, cut, {'source_speech': {'words': [], 'events': []},
        'samples': [{'time_sec': 2.1, 'clear_face': True, 'category': 'HAHA', 'lip_aperture': .5}]}, Counter())
    assert item['voice_id'] in {'da', 'ai'} and not evidence['cut_onset_matched']


def test_incremental_regions_cache_across_outputs_and_voice_changes(tmp_path):
    source = {'id': 's', 'path': 'unused', 'identity': {'sha256': 'a'}, 'duration_sec': 61., 'analyzed_end_sec': 61.}
    calls = []
    def scan(**args):
        calls.append(args['ranges'])
        # A boundary word in both overlapping responses has one core owner.
        return dict(speech(start=29.9, end=30.3), has_audio=True, status='transcribed')
    cfg = {'audio': {}, 'video': {}}
    s = SourceMatchSession(cfg, [deepcopy(source)], [voice('ai'), voice('da')], tmp_path, scan=scan)
    assert len(list(s.regions())) == 3
    assert len(calls) == 3 and len(s.sources[0]['source_speech']['words']) == 1
    changed = SourceMatchSession(cfg, [deepcopy(source)], [voice('da')], tmp_path, scan=lambda **k: pytest.fail('ASR cache missed'))
    list(changed.regions())
    assert changed.stats['cache_hits'] == 3 and changed.stats['matched_onsets'] == 0


def test_expired_budget_does_not_start_asr_or_cache_silence(tmp_path):
    source = {'id': 's', 'path': 'unused', 'identity': {}, 'duration_sec': 60.}
    s = SourceMatchSession({'audio': {'source_match_search_budget_sec': 1}}, [source], [voice('da')],
        tmp_path, started=time.monotonic()-2, scan=lambda **k: pytest.fail('ASR started after deadline'))
    assert list(s.regions()) == [] and s.stats['stop_reason'] == 'time_budget'
    assert not list(tmp_path.rglob('*.json'))


def test_asr_failure_is_not_a_silent_source(tmp_path):
    def broken(**args):
        raise SourceAudioError('decoder unavailable')
    s = SourceMatchSession({}, [{'id': 's', 'path': 'unused', 'identity': {}, 'duration_sec': 5.}],
                           [voice('da')], tmp_path, scan=broken)
    with pytest.raises(SourceAudioError, match='decoder unavailable'):
        list(s.regions())
    assert not list(tmp_path.rglob('*.json'))


def test_real_worker_json_ranges_and_silent_audio_do_not_load_whisper(tmp_path):
    path = tmp_path/'silent.wav'
    sf.write(path, np.zeros(16000), 16000)
    worker = SpeechWorker(tmp_path/'cache')
    try:
        result = worker.request({'path': str(path), 'ranges': [[0., 1.]], 'cache_dir': str(tmp_path/'cache')},
                                time.monotonic()+20)
        assert result['status'] == 'no-speech' and result['words'] == []
    finally:
        worker.close()
    assert worker.process is None


def test_deadline_terminates_owned_asr_process(tmp_path, monkeypatch):
    import subprocess
    import sys
    from autoedit.audio import search
    original = subprocess.Popen
    children = []
    def hanging_worker(command, **kwargs):
        child = original([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(search.subprocess, 'Popen', hanging_worker)
    worker = SpeechWorker(tmp_path)
    began = time.monotonic()
    with pytest.raises(TimeoutError, match='budget'):
        worker.request({}, began+.2)
    assert time.monotonic()-began < 3
    assert children[0].poll() is not None and worker.process is None


def test_no_audio_metadata_skips_worker_and_remaining_regions(tmp_path):
    s = SourceMatchSession({}, [{'id': 's', 'path': 'unused', 'identity': {},
                                'duration_sec': 1800., 'has_audio': False}], [voice('da')], tmp_path,
                           scan=lambda **k: pytest.fail('ASR ran for a source without audio'))
    assert list(s.regions()) == []
    assert s.sources[0]['source_speech']['status'] == 'no-audio'
    assert s.stats['stop_reason'] == 'source_exhausted'


def test_boundary_word_is_offered_as_a_cut_once_not_lost_between_regions(tmp_path, monkeypatch):
    from autoedit.video import reader
    monkeypatch.setattr(reader, 'sampled_frames', lambda *a, **k: iter(()))
    source = {'id': 's', 'path': 'unused', 'identity': {}, 'duration_sec': 61.}
    s = SourceMatchSession({}, [source], [voice('ai')], tmp_path,
        scan=lambda **k: dict(speech(start=29.9, end=30.3), has_audio=True))
    windows = list(s.windows({'slots': [{'id': 'a', 'required_duration_sec': 1.}], 'video_sampling': {}}))
    assert len(windows) == 1 and windows[0]['source_sound']['start'] == 29.9


def test_source_windows_feed_local_selection_and_stop_before_visual_only_search(tmp_path, monkeypatch):
    import av
    from autoedit.ai.cache import identity
    from autoedit.ai import discovery
    from autoedit.ai.local import LocalSelector, local_settings
    path = tmp_path/'source.mkv'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=24)
        stream.width, stream.height, stream.pix_fmt = 160, 90, 'bgr0'
        for _ in range(48):
            frame = av.VideoFrame.from_ndarray(np.full((90, 160, 3), 100, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    source = {'id': 's', 'path': str(path), 'identity': identity(path), 'duration_sec': 2.,
              'analyzed_start_sec': 0., 'analyzed_end_sec': 2.}
    pool = [voice('ai'), voice('da'), voice('cười')]
    session = SourceMatchSession({}, [source], pool, tmp_path/'cache',
        scan=lambda **k: dict(speech(start=.4, end=.65), has_audio=True))
    contract = {'sources': [source], 'voices': pool, 'references': [], 'source_gap_sec': 0,
        'backend_identity': {'model': 'test'}, 'local_analysis': local_settings({}),
        'slots': [{'id': 'a', 'required_duration_sec': .5}], 'video_sampling': {'video_fps': 1, 'shot_fallback': True}}
    class Client:
        single_image = False
        def check_resources(self):
            pass
    monkeypatch.setattr(discovery, 'distributed_candidates', lambda *a, **k: pytest.fail('Visual search ran first'))
    monkeypatch.setattr(discovery, 'classify', lambda batch, *a: {
        'cast': [{'id': 'car', 'reference_ids': []}], 'labels': [[0, 2] for _ in batch]})
    selector = LocalSelector(Client(), tmp_path/'cache', lambda _: None)
    selector.source_session = session
    result = selector.select(contract)
    row = result['selections'][0]
    assert row['in_sec'] == .4 and row['out_sec'] == .9
    assert row['voice_id'] == 'ai' and row['audio_match']['source_word'] == 'I'
    assert result['analysis_scope']['stop_reason'] == 'enough_slots'
    assert session.stats['regions'] == 1


def test_cached_reassignment_does_not_let_long_slot_steal_only_short_slot_onset(tmp_path, monkeypatch):
    from autoedit.ai import local, visual
    monkeypatch.setattr(local, 'transition_safe', lambda *a, **k: True)
    monkeypatch.setattr(visual, 'fingerprints', lambda *a: [])
    monkeypatch.setattr(visual, 'similar', lambda *a: False)
    data = speech(start=1., end=1.3)
    data['words'] += speech(start=2.5, end=2.8)['words']
    pool = [voice('ai')]
    index = OpeningSoundIndex(pool)
    first, second = index.anchors(data)
    source = {'id': 's', 'path': 'unused', 'source_speech': data}
    def run(anchor, slot, duration):
        return {'character_id': 'car', 'source_id': 's', 'shot': 0, 'shot_start_sec': 0., 'shot_end_sec': 8.,
            'start_sec': anchor['start']-.08, 'end_sec': anchor['start']+duration+.08,
            'mouth': 2, 'voice_hint': 'speech', 'source_sound': anchor, 'sound_slot_id': slot}
    result = local.allocate({'sources': [source], 'voices': pool, 'references': [], 'source_gap_sec': 0,
        'slots': [{'id': 'long', 'required_duration_sec': 2.}, {'id': 'short', 'required_duration_sec': 1.}]},
        [{'id': 'car', 'reference_ids': [], 'appearances': [{'source_id': 's', 'start_sec': 0., 'end_sec': 8.}]}],
        [run(first, 'long', 2.), run(second, 'long', 2.), run(first, 'short', 1.)],
        tmp_path, allow_same_shot=True)
    assert len(result['selections']) == 2
    assert {r['slot_id']: r['in_sec'] for r in result['selections']} == {'long': 2.5, 'short': 1.}


def test_progressive_search_revisits_conflicting_onsets_without_rescanning_audio(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from autoedit.ai import local, discovery, visual
    monkeypatch.setattr(local, 'transition_safe', lambda *a, **k: True)
    monkeypatch.setattr(visual, 'fingerprints', lambda *a: [])
    monkeypatch.setattr(visual, 'similar', lambda *a: False)
    monkeypatch.setattr(discovery, 'classify', lambda batch, *a: {
        'cast': [{'id': 'car', 'reference_ids': []}], 'labels': [[0, 2] for _ in batch]})
    data = speech(start=1., end=1.3)
    data['words'] += speech(start=2.5, end=2.8)['words']
    pool = [voice('ai')]
    a, b = OpeningSoundIndex(pool).anchors(data)
    windows = [{'source_id': 's', 'shot': 0, 'shot_start_sec': 0., 'shot_end_sec': 8.,
                'start_sec': anchor['start']-.08, 'end_sec': anchor['start']+duration+.08,
                'source_sound': anchor, 'sound_slot_id': slot}
               for anchor, slot, duration in [(a, 'long', 2.), (a, 'short', 1.), (b, 'long', 2.)]]
    calls = []
    def stream(contract):
        calls.append('audio')
        yield from windows
    contract = {'sources': [{'id': 's', 'path': 'unused', 'source_speech': data, 'duration_sec': 8.,
                             'analyzed_end_sec': 8.}], 'voices': pool, 'references': [], 'source_gap_sec': 0,
        'backend_identity': {}, 'local_analysis': local.local_settings({}),
        'slots': [{'id': 'long', 'required_duration_sec': 2.}, {'id': 'short', 'required_duration_sec': 1.}],
        'video_sampling': {'video_fps': 1, 'shot_fallback': True}}
    selector = discovery.ProgressiveSelector(SimpleNamespace(single_image=False, check_resources=lambda: None),
                                            tmp_path, lambda _: None)
    selector.source_session = SimpleNamespace(full=False, deadline=time.monotonic()+5,
                                             windows=stream, release_for_vision=lambda: None)
    result = selector.select(contract, allow_same_shot=True)
    assert len(result['selections']) == 2 and calls == ['audio']
    assert {r['slot_id']: r['in_sec'] for r in result['selections']} == {'long': 2.5, 'short': 1.}


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan'), True, 'bad'])
def test_invalid_budget_is_rejected(value):
    with pytest.raises(ValueError, match='source_match_search_budget_sec'):
        search_budget({'source_match_search_budget_sec': value})


def test_voice_analysis_cache_is_shared_by_standard_and_ai_backends(tmp_path, monkeypatch):
    from autoedit.audio import analyzer
    from autoedit.ai.cache import voice_catalog
    sf.write(tmp_path/'hi.wav', .2*np.sin(np.arange(3200)*.2), 16000)
    cache = tmp_path/'cache'
    first = analyzer.analyze_audio_dir(tmp_path, cache_dir=cache)
    monkeypatch.setattr(analyzer, 'analyze_file', lambda *a: pytest.fail('Voice analysis repeated across backends'))
    second = voice_catalog({'directory': str(tmp_path)}, cache)
    assert first == second and first[0]['voice_id']


def test_budget_reserves_completion_time_inside_same_limit(tmp_path):
    started = time.monotonic()
    s = SourceMatchSession({'audio': {'source_match_search_budget_sec': 300}}, [], [], tmp_path, started=started)
    assert s.final_deadline == started+300
    assert 269.9 <= s.deadline-started <= 270.1


def test_draft_completion_respects_deadline_without_publishing_footage(tmp_path):
    from autoedit.ai.completion import complete_selection
    from autoedit.runtime import SelectionTimeout
    with pytest.raises(SelectionTimeout):
        complete_selection({'characters': [], 'selections': []}, [],
            [{'id': 'a', 'required_duration_sec': 1}],
            [{'id': 's', 'path': 'unused', 'duration_sec': 5.}], [voice('da')], [], 0,
            tmp_path, lambda _: None, deadline=time.monotonic()-1)
