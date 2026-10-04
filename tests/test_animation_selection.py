from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytest

from autoedit.video.animation import cartoon_faces
from autoedit.video.faces import annotate_speaking, lip_motion_evidence
from autoedit.match.matcher import match_slots_to_video, InsufficientFootageError
from autoedit.audio.scene_match import choose_audio
from test_shot_tiers import sample


def drawing(mouth_height=8, offset=0):
    frame = np.full((240, 320, 3), 245, np.uint8)
    cv2.circle(frame, (160+offset, 120), 65, (0, 140, 245), -1)
    for x in (140+offset, 180+offset):
        cv2.ellipse(frame, (x, 105), (10, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (x, 106), 5, (0, 0, 0), -1)
    cv2.ellipse(frame, (160+offset, 139), (14, mouth_height), 0, 0, 360, (15, 15, 15), -1)
    return frame


def test_cartoon_geometry_requires_paired_pupils_shared_face_and_mouth():
    found = cartoon_faces(drawing())
    assert len(found) == 1 and found[0]['face_detected']
    for missing in ('eyes', 'mouth', 'pupils'):
        frame = drawing()
        if missing == 'eyes':
            cv2.rectangle(frame, (122, 87), (197, 121), (0, 140, 245), -1)
        elif missing == 'mouth':
            cv2.rectangle(frame, (142, 123), (178, 157), (0, 140, 245), -1)
        else:
            for x in (140, 180):
                cv2.ellipse(frame, (x,105), (10,12), 0, 0, 360, (255,255,255), -1)
        assert not cartoon_faces(frame), missing


def test_cartoon_speaking_uses_mouth_changes_not_camera_pan_or_open_mouth():
    from autoedit.video.analyzer import _visual_quality
    for moving_lips in (False, True):
        samples = []
        for i in range(12):
            frame = drawing(3 if moving_lips and i % 2 else 8, i)
            det = cartoon_faces(frame)[0]
            det.update(_visual_quality(cv2,frame,det)[0], time_sec=i/6)
            samples.append(det)
        annotate_speaking(samples,'animation')
        assert all(s['clear_face'] for s in samples)
        assert bool(sum(s['speaking'] for s in samples)) == moving_lips


def test_voiced_scene_rejects_static_person_even_when_soundtrack_matches():
    samples = [sample(i/2,2) for i in range(20)]
    video = {'duration_sec':10,'sample_fps':2,'samples':samples,
             'sound_anchors':[{'start':2.0,'end':2.2,'word':'I','confidence_tier':1}]}
    with pytest.raises(InsufficientFootageError,match='lip movement'):
        match_slots_to_video([{'id':'scene','required_duration_sec':1}],[],video,require_lip_motion=True)
    # Explicitly unvoiced fills can still use the independently confirmed face.
    assert match_slots_to_video([{'id':'intro','required_duration_sec':1}],[],video)


@pytest.mark.parametrize('pixel_noise', [False, True])
def test_speaking_context_cannot_qualify_static_or_pixel_noise_inside_cut(pixel_noise):
    samples = []
    for i in range(24):
        det = cartoon_faces(drawing())[0]
        det.update(time_sec=i/6, face_visible=1, clear_face=1, speaking=1,
                   closeup_score=1, mouth_clarity=1, activity_score=0,
                   lip_aperture=.2 + (1/det['mouth_box'][2] if pixel_noise and i%2 else 0))
        samples.append(det)
    assert not lip_motion_evidence(samples)
    with pytest.raises(InsufficientFootageError, match='lip movement'):
        match_slots_to_video([{'id':'scene','required_duration_sec':1}],[],
                            {'duration_sec':4,'sample_fps':6,'samples':samples},require_lip_motion=True)


def test_round_and_wide_mouth_changes_are_speech_even_with_constant_aspect_ratio():
    samples = [dict(time_sec=i/6, backend='animation-eyes-mouth', mouth_box=[0,0,20,10],
                    eye_distance=40, lip_aperture=.5, lip_width_ratio=.25 if i%2 else .6)
               for i in range(12)]
    assert lip_motion_evidence(samples)
    for s in samples:
        s['lip_width_ratio'] = .6 + (1/40 if s['time_sec']*6 % 2 else 0)
    assert not lip_motion_evidence(samples)


def test_unverified_spectral_laugh_cannot_override_measured_speech():
    samples = []
    for i in range(48):
        det = cartoon_faces(drawing(3 if i%2 else 8))[0]
        det.update(time_sec=i/6, face_visible=1, clear_face=1, speaking=1,
                   closeup_score=1, mouth_clarity=1, activity_score=.2)
        samples.append(det)
    video = {'duration_sec':8,'sample_fps':6,'samples':samples,
             'sound_anchors':[{'start':2,'end':2.5,'action':'LAUGH','prob':.75,'confidence_tier':2}]}
    cut = match_slots_to_video([{'id':'scene','required_duration_sec':1}],[],video,
                              require_lip_motion=True)[0]['video']
    assert not cut.get('source_sound')


def test_allocation_search_preserves_gaps_when_best_middle_cut_blocks_both_sides():
    samples = []
    for i in range(54):
        t = i/6
        det = cartoon_faces(drawing(3 if i%2 else 8))[0]
        visible = any(a <= t <= b for a,b in [(1,2+1/3),(3,4+1/3),(5,6+1/3)])
        det.update(time_sec=t, face_visible=int(visible), face_detected=visible,
                   clear_face=int(visible), speaking=int(visible),
                   closeup_score=1, mouth_clarity=1, activity_score=.2)
        samples.append(det)
    video = {'duration_sec':9,'sample_fps':6,'samples':samples,
             'sound_anchors':[{'start':3.,'end':3.5,'word':'I','confidence_tier':1}]}
    matches = match_slots_to_video([{'id':str(i),'required_duration_sec':1} for i in range(2)],
                                  [],video,2,require_lip_motion=True)
    cuts = sorted((m['video'] for m in matches),key=lambda c:c['in_sec'])
    assert cuts[0]['in_sec'] < 3 and cuts[1]['in_sec'] >= 5
    assert cuts[1]['in_sec']-cuts[0]['out_sec'] >= 2
    assert all(c['selection_metrics']['lip_motion_within_cut'] for c in cuts)


def test_static_main_character_cannot_outrank_talking_supporting_character():
    from test_characters import sample as cast_sample
    samples = [{**cast_sample(i/6,'main' if i<36 else 'supporting'),
                'lip_aperture':.05 if i%2 else .25,'lip_width_ratio':.6} for i in range(72)]
    for s in samples[:36]:
        s.update(speaking=0,clear_face=0)
    video={'duration_sec':12,'sample_fps':6,'samples':samples,
           'character_analysis':{'min_main_fraction':.85}}
    cut=match_slots_to_video([{'id':'scene','required_duration_sec':1}],[],video,require_lip_motion=True)[0]['video']
    # One bracketing sample may precede the talking interval.
    assert cut['in_sec'] >= 6-1/6-1e-8
    assert cut['selection_metrics']['speaking'] >= .35
    assert cut['selection_metrics']['lip_motion_within_cut']


def test_strict_dense_analysis_does_not_invent_a_neutral_voice():
    cut={'in_sec':0,'out_sec':1,'selection_metrics':{'tier':3,'require_lip_motion':True}}
    with pytest.raises(ValueError,match='unrelated neutral WAV'):
        choose_audio([],cut,{'samples':[]},Counter())


def test_strict_dense_analysis_rejects_static_lips_even_with_supported_word():
    cut={'in_sec':0,'out_sec':1,'selection_metrics':{'tier':1,'require_lip_motion':True}}
    samples=[dict(time_sec=i/20, clear_face=1, lip_aperture=.2, lip_width_ratio=.5)
             for i in range(20)]
    with pytest.raises(ValueError,match='no visible lip movement'):
        choose_audio([{'path':'hi.wav','name':'hi.wav','category':'HI','duration_sec':1}],cut,
                     {'samples':samples,'source_speech':{'words':[{'start':0,'end':.5,'word':'hi','prob':1}]}},
                     Counter())


def test_character_selection_preserves_frame_motion(tmp_path,monkeypatch):
    import autoedit.video.characters as chars
    from test_characters import face, track
    model=tmp_path/'model.onnx'; model.write_bytes(b'fake')
    samples=[{**face(),'time_sec':i/2,'faces':[{**face(),'activity_score':0}],
              'frame_width':640,'frame_height':360,'activity_score':.7} for i in range(4)]
    info={'source_kind':'animation','duration_sec':2,'sample_fps':2,'samples':samples}
    monkeypatch.setattr(chars,'EmbeddingModel',lambda *a: object())
    monkeypatch.setattr(chars,'_extract_tracks',lambda *a:[track('t',[1,0],0,2,0,list(range(4)))])
    result=chars.analyze_characters(info,tmp_path/'cache',{'ccip_model':str(model)})
    assert all(s['activity_score']==.7 for s in result['samples'])


def test_crop_extraction_uses_real_shots_to_keep_mouth_track(monkeypatch):
    import autoedit.video.characters as chars
    from test_characters import face
    samples = [{'time_sec':i/2, 'faces':[face()], 'frame_width':640, 'frame_height':360}
               for i in range(6)]
    info = {'path':'unused.mp4','samples':samples,'sample_fps':2,'duration_sec':3,
            'transition_times_sec':[.5,1,1.5], 'shot_times_sec':[1.5]}
    class Model:
        def extract(self, *args): return np.array([1.,0.])
    monkeypatch.setattr(chars, '_selected_source_frames',
                        lambda path,times: ((t,None) for t in times))
    tracks = chars._extract_tracks(info,chars.settings(),Model(),None)
    assert len(tracks) == 2
    assert samples[0]['faces'][0]['track_id'] == samples[2]['faces'][0]['track_id']
    assert samples[2]['faces'][0]['track_id'] != samples[3]['faces'][0]['track_id']


@pytest.mark.parametrize('seconds,expected',[ (25,False),(65,True),(150,True),(200,False),(300,False)])
def test_local_bumble_regression_does_not_replace_face_evidence_with_scenery(seconds,expected):
    path=Path(r'C:\Users\admin\Downloads\The Bumble Nums Search for Special Ingredients to Make Some Tasty Treats!_1080p.mp4')
    if not path.is_file(): pytest.skip('Local regression video unavailable')
    cap=cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
        ok,frame=cap.read(); assert ok
        assert bool(cartoon_faces(cv2.resize(frame,(640,360)))) == expected
    finally:
        cap.release()
