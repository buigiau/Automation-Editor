from copy import deepcopy
from pathlib import Path
import time

from PIL import Image
import pytest

from autoedit.ai import single
from autoedit.ai.ollama import LocalAIError
from test_ai_local import media


@pytest.fixture
def frames(tmp_path):
    paths = [tmp_path/f'{i}.jpg' for i in range(2)]
    for path, color in zip(paths, ('red', 'blue')):
        Image.new('RGB', (64, 64), color).save(path)
    return paths


def cat(color='gray', **kwargs):
    return {'present': True, 'identity_clear': True, 'subject_count': 1,
            'species': 'cat', 'body_color': color, 'mouth': 2, **kwargs}


class Client:
    timeout = 100
    max_output_tokens = 1800
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
    def chat(self, *args):
        self.calls += 1
        return next(self.responses)


def classify_pair(monkeypatch, tmp_path, frames, responses, refs=()):
    monkeypatch.setattr(single, 'views', lambda *a: frames)
    client = Client(responses)
    contract = {'sources': [{'id': 's'}], 'references': list(refs), 'backend_identity': {'digest': 'real-model'}}
    windows = [{'source_id': 's', 'start_sec': 1., 'end_sec': 3.}]
    result = single.classify(windows, contract, client, tmp_path, {}, time.monotonic()+10, lambda _: None)
    return result, client, contract, windows


def test_both_cut_samples_must_confirm_the_same_physical_character(monkeypatch, tmp_path, frames):
    result, client, contract, windows = classify_pair(monkeypatch, tmp_path, frames,
        [cat('dark grey'), cat('gray')])
    assert result['cast'] == [{'id': 'gray-cat', 'reference_ids': []}]
    assert result['labels'] == [[0, 2]] and client.calls == 2
    assert client.timeout == 100 and client.max_output_tokens == 1800
    again = single.classify(windows, contract, client, tmp_path, {}, time.monotonic()+10, lambda _: None)
    assert again == result and client.calls == 2


@pytest.mark.parametrize('last', [cat('orange'), cat(identity_clear=False),
    cat(present=False, species='none', body_color='none'), cat(body_color='unknown')])
def test_changed_occluded_absent_or_uncertain_actor_is_rejected(monkeypatch, tmp_path, frames, last):
    result, *_ = classify_pair(monkeypatch, tmp_path, frames, [cat(), last])
    assert not result['cast'] and result['labels'] == [[-1, 0]]


def test_reference_match_requires_agreement_in_both_samples(monkeypatch, tmp_path, frames):
    refs = [{'id': 'r0', 'path': str(frames[0])}, {'id': 'r1', 'path': str(frames[1])}]
    result, *_ = classify_pair(monkeypatch, tmp_path, frames,
        [cat(reference=0), cat(reference=1)], refs)
    assert result['cast'][0]['reference_ids'] == []


def test_reference_look_can_change_dominant_colour_between_angles(monkeypatch, tmp_path, frames):
    refs = [{'id': 'r0', 'path': str(frames[0])}]
    result, *_ = classify_pair(monkeypatch, tmp_path, frames,
        [cat('orange', species='person', reference=0), cat('purple', species='person', reference=0)], refs)
    assert result['cast'] == [{'id': 'reference-look-r0', 'reference_ids': ['r0']}]


def test_unmatched_people_cannot_inherit_a_reference_from_a_shared_colour_label(monkeypatch, tmp_path):
    pictures = [tmp_path/f'{i}.png' for i in range(4)]
    for path, color in zip(pictures, ('red', 'blue', 'green', 'yellow')):
        Image.new('RGB', (64, 64), color).save(path)
    monkeypatch.setattr(single, 'views', lambda w, *a: pictures[:2] if w['start_sec'] == 0 else pictures[2:])
    refs = [{'id': 'r0', 'path': str(pictures[0])}]
    contract = {'sources': [{'id': 's'}], 'references': refs, 'backend_identity': {}}
    client = Client([cat('white', species='person', reference=r) for r in (0, 0, -1, -1)])
    windows = [{'source_id': 's', 'start_sec': start, 'end_sec': start+1} for start in (0, 10)]
    result = single.classify(windows, contract, client, tmp_path/'cache', {}, time.monotonic()+10, lambda _: None)
    assert [c['reference_ids'] for c in result['cast']] == [['r0'], []]
    assert result['cast'][0]['id'] != result['cast'][1]['id']


def test_appearance_observation_uses_only_masked_inputs_and_outfit_instructions(tmp_path, frames):
    class Appearance:
        references = [{'path': 'masked-reference.jpg'}]
        def masked(self, path):
            assert path == frames[0]
            return 'masked-target.jpg'
    client = Client([cat('orange', species='person', reference=0)])
    client.reference_appearance = Appearance()
    def chat(prompt, images, schema):
        assert images == ['masked-reference.jpg', 'masked-target.jpg']
        assert 'NEVER clothes' not in prompt
        assert 'never compare facial identity' in prompt
        assert 'shared colour alone is insufficient' in prompt
        return next(client.responses)
    client.chat = chat
    result = single.observe(frames[0], {'references': [{'id': 'r', 'path': 'original.png'}],
        'backend_identity': {}}, client, tmp_path/'cache', time.monotonic()+10)
    assert result['reference_ids'] == ['r']


def test_appearance_samples_do_not_claim_facial_identity(monkeypatch, tmp_path, frames):
    monkeypatch.setattr(single, 'views', lambda *a: frames)
    client = Client([{'reference_colors': ['orange'], 'samples': [{'reference': 0, 'readable': True, 'subject_count': 1, 'outfit_color': 'orange'}]*2}])
    class Appearance:
        references = [{'path': str(frames[0])}]
        def masked(self, path):
            return path
    client.reference_appearance = Appearance()
    contract = {'sources': [{'id': 's'}], 'references': [{'id': 'r', 'path': str(frames[0])}], 'backend_identity': {}}
    result = single.classify([{'source_id': 's', 'start_sec': 0, 'end_sec': 1}], contract,
        client, tmp_path/'cache', {}, time.monotonic()+10, lambda _: None)
    assert result['details'][0]['identity_consistent'] is False
    assert result['details'][0]['identity_sample_frames'] == 0
    assert result['details'][0]['reference_match_scope'] == 'costume_appearance'
    assert result['details'][0]['appearance_sample_frames'] == 2


def test_paired_appearance_request_is_cached_and_uses_both_masked_frames(tmp_path, frames):
    class Appearance:
        references = [{'path': 'masked-reference.jpg'}]
        def masked(self, path):
            return 'masked-'+path.name
    client = Client([{'reference_colors': ['orange'], 'samples': [{'reference': 0, 'readable': True, 'subject_count': 2, 'outfit_color': 'orange'}]*2}])
    client.reference_appearance = Appearance()
    def chat(prompt, inputs, schema):
        client.calls += 1
        assert inputs == ['masked-reference.jpg', 'masked-0.jpg', 'masked-1.jpg']
        assert 'LAST TWO' in prompt and 'Never compare facial identity' in prompt
        assert schema['properties']['samples']['maxItems'] == 2
        assert client.timeout <= 10 and client.max_output_tokens == 180
        return next(client.responses)
    client.chat = chat
    contract = {'references': [{'id': 'r', 'path': 'original.jpg'}], 'backend_identity': {}}
    result = single.observe_appearances(frames, contract, client, tmp_path/'cache', time.monotonic()+10)
    assert all(r['reference_ids'] == ['r'] and r['subject_count'] == 2 for r in result)
    assert single.observe_appearances(frames, contract, client, tmp_path/'cache', time.monotonic()+10) == result
    assert client.calls == 1 and client.timeout == 100 and client.max_output_tokens == 1800


def test_wardrobe_mismatch_in_either_sample_vetoes_a_model_request(tmp_path, frames):
    class Appearance:
        def compatible_references(self, path):
            return {'r'} if path == frames[0] else set()
    client = Client([])
    client.reference_appearance = Appearance()
    result = single.observe_appearances(frames, {'references': [{'id': 'r'}], 'backend_identity': {}},
        client, tmp_path/'cache', time.monotonic()+10)
    assert client.calls == 0 and all(not r['present'] for r in result)


def test_model_cannot_match_a_black_floral_shirt_to_an_orange_jacket(tmp_path, frames):
    class Appearance:
        references = [{'path': str(frames[0])}]
        def masked(self, path):
            return path
        def supported_outfit_colors(self, rid):
            return {'orange','purple'}
    client = Client([{'reference_colors': ['orange'], 'samples': [
        {'reference': 0, 'readable': True, 'subject_count': 1, 'outfit_color': 'black'}]*2}])
    client.reference_appearance = Appearance()
    result = single.observe_appearances(frames, {'references': [{'id': 'r'}], 'backend_identity': {}},
        client, tmp_path/'cache', time.monotonic()+10)
    assert all(not r['present'] and not r['reference_ids'] for r in result)
    assert all(r['raw']['reference_outfit_color'] == 'orange' for r in result)


def test_reference_secondary_costume_colour_can_be_read_from_another_angle(tmp_path, frames):
    class Appearance:
        references = [{'path': str(frames[0])}]
        def masked(self, path):
            return path
        def supported_outfit_colors(self, rid):
            return {'orange', 'purple'}
    client = Client([{'reference_colors': ['orange'], 'samples': [
        {'reference': 0, 'readable': True, 'subject_count': 1, 'outfit_color': color} for color in ('orange','purple')]}])
    client.reference_appearance = Appearance()
    result = single.observe_appearances(frames, {'references': [{'id': 'r'}], 'backend_identity': {}},
        client, tmp_path/'cache', time.monotonic()+10)
    assert all(r['present'] and r['reference_ids'] == ['r'] for r in result)


@pytest.mark.parametrize('response', [{'samples': []}, {'samples': [{}]*2},
    {'samples': [{'reference': 0, 'readable': 'yes', 'subject_count': 1}]*2},
    {'samples': [{'reference': True, 'readable': True, 'subject_count': 1}]*2}])
def test_invalid_paired_appearance_outputs_are_not_cached(tmp_path, frames, response):
    class Appearance:
        references = [{'path': str(frames[0])}]
        def masked(self, path):
            return path
    client = Client([response])
    client.reference_appearance = Appearance()
    with pytest.raises(LocalAIError, match='Invalid paired appearance'):
        single.observe_appearances(frames, {'references': [{'id': 'r'}], 'backend_identity': {}},
            client, tmp_path/'cache', time.monotonic()+10)
    assert not list((tmp_path/'cache/local/appearance-pairs').glob('*.json'))


@pytest.mark.parametrize('update', [{'present': 1}, {'identity_clear': 'yes'},
    {'mouth': True}, {'mouth': 4}, {'species': '42'}, {'body_color': None},
    {'subject_count': True}, {'subject_count': -1}])
def test_invalid_observations_are_never_cached(tmp_path, frames, update):
    with pytest.raises(LocalAIError, match='Invalid single-image'):
        single.observe(frames[0], {'references': [], 'backend_identity': {}},
            Client([cat(**update)]), tmp_path, time.monotonic()+10)
    assert not list((tmp_path/'local/single-observations').glob('*.json'))


def test_real_cut_views_are_inside_the_shot_and_cached(media, tmp_path, monkeypatch):
    pictures = single.views({'start_sec': .08, 'end_sec': 2.92}, media, tmp_path)
    assert len(pictures) == 2 and all(p.is_file() for p in pictures)
    import av
    monkeypatch.setattr(av, 'open', lambda *a: pytest.fail('Cached character views decoded video again'))
    assert single.views({'start_sec': .08, 'end_sec': 2.92}, media, tmp_path) == pictures
