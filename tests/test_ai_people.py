import numpy as np
import pytest

from autoedit.ai import people


@pytest.fixture
def observer(monkeypatch):
    screen = object.__new__(people.FastPeople)
    screen.stats = {'verified_cuts': 0, 'rejected_cuts': 0}
    screen.anchors = []
    class Embedding:
        def extract(self, *args):
            return np.array([1., 0.], dtype=np.float32)
    screen.embedding = Embedding()
    monkeypatch.setattr(people, 'cut_frames', lambda *a: iter([(t, np.zeros((96, 128, 3), np.uint8)) for t in (.2, .5, .8)]))
    def frame(count=1):
        return {'count': count, 'clear': [{'face': [10, 10, 40, 40], 'lip_aperture': .2}]}
    screen.inspect = lambda rgb: frame()
    return screen, frame


def classify(screen, tmp_path, **kwargs):
    return screen.classify({'start_sec': 0., 'end_sec': 1.}, {'path': 'source'}, tmp_path, **kwargs)


def test_reference_screen_cannot_invoke_facial_identity_even_accidentally(observer, tmp_path):
    screen, _ = observer
    screen.enable_identity = False
    with pytest.raises(RuntimeError, match='disabled for reference appearance'):
        classify(screen, tmp_path)


def test_fast_solo_requires_the_same_face_in_all_three_samples(observer, tmp_path):
    screen, _ = observer
    first = classify(screen, tmp_path)
    second = classify(screen, tmp_path)
    assert first['subject_count'] == 1 and first['sample_frames'] == 3
    assert first['id'] == second['id'] and len(screen.anchors) == 1
    assert first['method'] == people.VERSION


def test_extra_person_in_only_the_middle_frame_defers_the_entire_cut(observer, tmp_path):
    screen, frame = observer
    frames = iter([frame(), frame(2), frame()])
    screen.inspect = lambda rgb: next(frames)
    assert classify(screen, tmp_path) == {'deferred_group': True}
    assert not screen.anchors


def test_changing_identity_is_rejected_even_when_each_sample_has_one_person(observer, tmp_path):
    screen, _ = observer
    vectors = iter([np.array([1., 0.]), np.array([0., 1.]), np.array([1., 0.])])
    screen.embedding.extract = lambda *a: next(vectors)
    assert classify(screen, tmp_path) is None and not screen.anchors


def test_group_fallback_retains_the_measured_count(observer, tmp_path):
    screen, frame = observer
    screen.inspect = lambda rgb: frame(3)
    result = classify(screen, tmp_path, allow_group=True)
    assert result['subject_count'] == 3 and result['sample_frames'] == 3


def test_occluded_solo_faces_do_not_invent_a_confirmed_identity(observer, tmp_path):
    screen, _ = observer
    screen.inspect = lambda rgb: {'count': 1, 'clear': []}
    result = classify(screen, tmp_path)
    assert result['id'].startswith('person-unresolved-')
    assert not result['identity_consistent'] and result['identity_sample_frames'] == 0
    assert result['subject_count'] == 1 and result['sample_frames'] == 3
    assert classify(screen, tmp_path, allow_group=True) is None


def test_one_profile_sample_still_requires_two_agreeing_face_embeddings(observer, tmp_path):
    screen, frame = observer
    observations = iter([frame(), {'count': 1, 'clear': []}, frame()])
    screen.inspect = lambda rgb: next(observations)
    result = classify(screen, tmp_path)
    assert result['identity_consistent'] and result['identity_sample_frames'] == 2
    assert result['subject_count'] == 1 and result['sample_frames'] == 3


def test_leg_only_person_detection_cannot_fill_a_solo_slot(observer, tmp_path):
    screen, _ = observer
    screen.inspect = lambda rgb: {'count': 1, 'people': 1, 'clear': [], 'faces': [],
                                  'body_boxes': [[0, 0, 128, 65]]}
    assert classify(screen, tmp_path) is None
    assert 'cropped' in screen.last_rejection


def test_a_masked_person_with_head_inside_the_image_remains_valid_solo_footage(observer, tmp_path):
    screen, _ = observer
    screen.inspect = lambda rgb: {'count': 1, 'people': 1, 'clear': [], 'faces': [],
                                  'body_boxes': [[40, 12, 35, 78]]}
    result = classify(screen, tmp_path)
    assert result['subject_count'] == 1 and not result['identity_consistent']


def test_tail_fallback_identity_samples_only_the_stable_prefix(observer, tmp_path, monkeypatch):
    screen, _ = observer
    requested = []
    def frames(path, times, width):
        requested.extend(times)
        return iter((t, np.zeros((96, 128, 3), np.uint8)) for t in times)
    monkeypatch.setattr(people, 'cut_frames', frames)
    result = screen.classify({'start_sec': 0., 'end_sec': 9.44, 'transition_tail_fallback_sec': 2},
        {'path': 'source'}, tmp_path)
    assert result and len(requested) == 3
    assert max(requested) < 7.36
