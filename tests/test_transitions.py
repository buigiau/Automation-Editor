import numpy as np
import pytest

from autoedit.video.transitions import TransitionDetector, transition_metrics
from autoedit.match.matcher import match_slots_to_video
from test_shot_tiers import sample


def test_flash_return_detected_between_sparse_face_samples():
    detector = TransitionDetector()
    for t, value in [(0, 0), (.04, 255), (.08, 0), (.12, 0)]:
        detector.observe(t, np.full((90, 160, 3), value, dtype=np.uint8))
    assert detector.events == [.04, .08]


@pytest.mark.parametrize('events,allowed', [([1], False), ([2.9], False),
    ([4.2], True), ([3.8, 4.2], False), ([0], False), ([5], False), ([], True)])
def test_opening_tail_and_boundary_rules(events, allowed):
    assert transition_metrics(events, 0, 5)['transition_safe'] is allowed


def test_short_clip_never_allows_internal_transition():
    assert not transition_metrics([1.5], 0, 2)['transition_safe']


def test_similar_palette_cut_is_detected():
    detector = TransitionDetector()
    detector.observe(0, np.full((90, 160, 3), 80, dtype=np.uint8))
    detector.observe(.04, np.full((90, 160, 3), 160, dtype=np.uint8))
    assert detector.events == [.04]


def test_gradual_dissolve_is_detected_without_large_adjacent_changes():
    detector = TransitionDetector()
    for i in range(16):
        detector.observe(i / 30, np.full((90, 160, 3), 50 + i * 6, dtype=np.uint8))
    assert detector.events
    assert not transition_metrics(sorted(set(detector.events)), 0, 3)['transition_safe']


def test_hard_cut_does_not_generate_transitions_in_the_following_shot():
    detector = TransitionDetector()
    for i in range(48):
        detector.observe(i/24,np.full((90,160,3),30 if i < 24 else 180,np.uint8))
    assert detector.events == [1.]
    assert transition_metrics(detector.events,1.2,1.8)['transition_safe']


def test_textured_camera_pan_is_not_a_transition():
    import cv2
    texture = np.random.default_rng(41).integers(20,230,(90,160,3),dtype=np.uint8)
    texture = cv2.GaussianBlur(texture,(5,5),0)
    detector = TransitionDetector()
    for i in range(16):
        frame = cv2.warpAffine(texture,np.float32([[1,0,i],[0,1,0]]),(160,90),
                               borderMode=cv2.BORDER_REFLECT)
        detector.observe(i/30,frame)
    assert not detector.events


def test_matcher_prefers_uncut_body_shot_over_closeup_with_late_cut():
    samples = [sample(i / 10, 1 if i < 60 else 3) for i in range(120)]
    result = match_slots_to_video([{'id': '1', 'required_duration_sec': 5}], [],
        {'duration_sec': 12, 'sample_fps': 10, 'samples': samples,
         'transition_times_sec': [4.2, 6]})
    assert result[0]['video']['in_sec'] > 6
    assert result[0]['video']['selection_metrics']['transition_count'] == 0


def test_matcher_does_not_relax_opening_when_source_is_all_rapid_cuts():
    with pytest.raises(ValueError, match='stable opening'):
        match_slots_to_video([{'id': '1', 'required_duration_sec': 4}], [],
            {'duration_sec': 12, 'samples': [sample(i / 2, 1) for i in range(24)],
             'transition_times_sec': list(range(1, 12))})


def test_old_saved_analysis_cannot_silently_disable_transition_filter():
    with pytest.raises(ValueError, match='Restart AutoEdit'):
        match_slots_to_video([{'id': '5', 'required_duration_sec': 1}], [],
            {'duration_sec': 10, 'cache_identity': {'version': 6},
             'samples': [sample(i / 2, 1) for i in range(20)]})


@pytest.mark.parametrize('start,end', [(215.840625, 216.840625),
    (248.70679166666667, 249.70679166666667), (232.52395833333333, 233.56395833333332),
    (405.697, 406.697)])
def test_reported_kpop_unstable_cuts_including_intro_are_rejected(start, end):
    from pathlib import Path
    import av
    path = Path(r'C:\Users\admin\Downloads\YTSave_YouTube_Media_pV4Iak8HLs8_K-POP-DEMON-HUNTERS-All-Movie-Clips-2025_001_1080p.mp4')
    if not path.is_file():
        pytest.skip('Local user regression video unavailable')
    detector = TransitionDetector()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0
        container.seek(int((start - .8 + origin) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            time = float(frame.time) - origin
            if time < start - .8:
                continue
            if time > end + .8:
                break
            detector.observe(time, frame.reformat(width=160, height=90, format='bgr24').to_ndarray())
    assert not transition_metrics(sorted(set(detector.events)), start, end)['transition_safe']
