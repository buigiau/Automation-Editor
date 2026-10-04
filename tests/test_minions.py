from copy import deepcopy
import cv2
import numpy as np
import pytest

from autoedit.video.animation import minion_faces, MINION_EVIDENCE
from autoedit.video.characters import analyze_characters, settings, build_tracks, tag_ensemble_tracks, rank_characters
from autoedit.video.faces import MultiRegionFaceLandmarker
from autoedit.match.matcher import match_slots_to_video, InsufficientFootageError
from test_characters import face, sample


def drawing(eyes=1, missing=None, mouth_height=10):
    image=np.full((240,320,3),35,np.uint8)
    colour=(0,220,250) if missing != 'yellow' else (220,90,30)
    cv2.rectangle(image,(95,75),(225,205),colour,-1)
    cv2.ellipse(image,(160,75),(65,45),0,180,360,colour,-1)
    if missing != 'eyes':
        for x in ((160,) if eyes == 1 else (137,183)):
            radius=27 if eyes == 1 else 21
            cv2.circle(image,(x,85),radius,(160,160,160),-1)
            cv2.circle(image,(x,85),radius-5,(240,240,240),-1)
            if missing != 'pupils':
                cv2.circle(image,(x-3,87),7,(15,15,15),2 if missing == 'pupil_rings' else -1)
    if missing != 'mouth':
        cv2.ellipse(image,(160,145),(25,mouth_height),0,0,360,(15,15,15),-1)
    cv2.rectangle(image,(95,185),(225,205),(170,65,20),-1)
    return image


@pytest.mark.parametrize('eyes',[1,2])
def test_minions_geometry_supports_single_and_double_goggles(eyes):
    found=minion_faces(drawing(eyes))
    assert len(found) == 1
    assert found[0]['eye_count'] == eyes
    assert found[0]['character_group'] == 'yellow_minions'
    mx,my,mw,mh=found[0]['mouth_box']
    assert mx <= 160 <= mx+mw and my <= 145 <= my+mh


@pytest.mark.parametrize('missing',['eyes','pupils','pupil_rings','yellow'])
def test_colour_or_blank_goggles_cannot_qualify_a_minion(missing):
    assert not minion_faces(drawing(missing=missing))


def test_separate_bright_objects_inside_yellow_circle_are_not_one_eye():
    image=drawing()
    for y in (77,98):
        cv2.line(image,(139,y),(181,y),(0,220,250),4)
    assert not minion_faces(image)


def test_minions_profile_does_not_load_human_face_models(monkeypatch):
    import autoedit.video.faces as faces
    monkeypatch.setattr(faces,'FaceLandmarker',lambda:pytest.fail('human model should not load'))
    detector=MultiRegionFaceLandmarker(main_group='yellow_minions')
    try:
        assert detector.detect(drawing(),0)['faces'][0]['character_group'] == 'yellow_minions'
    finally:
        detector.close()


def test_automatic_ensemble_ranks_by_exposure_and_keeps_individual_tracks(tmp_path, monkeypatch):
    import autoedit.video.characters as chars
    minion={**face([50,50,40,60]),'face_evidence':MINION_EVIDENCE,
            'character_group':'yellow_minions','backend':'animation-eyes-mouth'}
    other=face([130,20,180,180])
    info={'path':'unused.mp4','source_kind':'animation','sample_fps':6,'duration_sec':4,
          'samples':[{'time_sec':i/6,'faces':([deepcopy(other)] if i < 12 else []) + [deepcopy(minion),
                        {**deepcopy(minion),'face':[330,50,40,60]}],
                      'frame_width':640,'frame_height':360,'activity_score':.4} for i in range(24)]}
    model=tmp_path/'ccip.onnx'; model.write_bytes(b'fake')
    monkeypatch.setattr(chars,'EmbeddingModel',lambda *a:None)
    def extract(video,config,*args):
        tracks=build_tracks(video['samples'],[],4,6,config)
        tag_ensemble_tracks(tracks,video['samples'])
        for track in tracks:
            if not track.get('character_group'):
                track['vectors']=[[1,0],[1,0]]
        return tracks
    monkeypatch.setattr(chars,'_extract_tracks',extract)
    result=analyze_characters(info,tmp_path,{'ccip_model':str(model)})
    assert all(s['character_id'] == 'character-001' for s in result['samples'])
    assert all(s['face'][2] == 40 and s['activity_score'] == .4 for s in result['samples'])
    assert len(result['character_analysis']['clusters'][0]['track_ids']) == 2
    assert result['character_analysis']['clusters'][0]['duration_sec'] == pytest.approx(4)
    assert result['character_analysis']['clusters'][0]['character_group'] == 'yellow_minions'
    assert not result['character_analysis'].get('strict_main_group')
    assert 'character_analysis' not in info


def test_old_group_metadata_cannot_disable_ranked_fallback():
    samples=[sample(i/2,'main' if i < 4 else 'supporting',1 if i < 4 else 2) for i in range(24)]
    video={'duration_sec':12,'sample_fps':2,'samples':samples,
           'character_analysis':{'strict_main_group':True,'min_main_fraction':.85}}
    cuts=match_slots_to_video([{'id':'cut','required_duration_sec':3}],[],video)
    assert cuts[0]['video']['character_selection']['decision'] == 'supporting'


def test_verified_group_cuts_retain_requested_lengths_and_gap():
    samples=[sample(i/2,'main') for i in range(40)]
    for s in samples:
        s['character_id']='group-yellow-minions'
    video={'duration_sec':20,'sample_fps':2,'samples':samples,
           'character_analysis':{'strict_main_group':True,'min_main_fraction':.85}}
    cuts=match_slots_to_video([{'id':str(i),'required_duration_sec':2} for i in range(3)],[],video,min_gap_sec=3)
    ranges=sorted((c['video']['in_sec'],c['video']['out_sec']) for c in cuts)
    assert all(b-a == pytest.approx(2) for a,b in ranges)
    assert all(b[0]-a[1] >= 3 for a,b in zip(ranges,ranges[1:]))


def test_legacy_group_preset_uses_automatic_cast_ranking():
    assert settings({'main_group':'yellow_minions'})['main_group'] == 'auto'
    assert settings()['main_group'] == 'auto'
    assert not settings({'main_group':'yellow_minions','enabled':False})['enabled']


def test_detected_minions_are_not_forced_first_when_another_actor_has_more_exposure():
    from test_characters import track
    minions={**track('minion',[1,0],0,3,0),'character_group':'yellow_minions'}
    human=track('human',[0,1],10,9,1)
    clusters,_,_=rank_characters([minions,human],settings(),.2)
    assert clusters[0]['track_ids'] == ['human']
    assert clusters[1]['character_group'] == 'yellow_minions'


def test_automatic_animation_detects_minions_without_a_group_picker(monkeypatch):
    import autoedit.video.faces as faces
    class NoHuman:
        def detect_all(self,*args):
            return []
        def close(self):
            pass
    monkeypatch.setattr(faces,'FaceLandmarker',NoHuman)
    detector=MultiRegionFaceLandmarker()
    try:
        assert any(f.get('character_group') == 'yellow_minions' for f in detector.detect(drawing(),0)['faces'])
    finally:
        detector.close()


def test_ensemble_label_preserves_measured_landmark_mouth_and_does_not_duplicate_face():
    from autoedit.video.faces import merge_animation_faces
    measured=face([100,50,80,120])
    minion={**measured,'character_group':'yellow_minions','face_evidence':MINION_EVIDENCE,
            'mouth_box':[100,100,50,10],'lip_aperture':.9}
    result=merge_animation_faces([measured],[minion])
    assert len(result) == 1
    assert result[0]['character_group'] == 'yellow_minions'
    assert result[0]['mouth_box'] == measured['mouth_box']
    assert 'character_group' not in measured


def test_tracking_hints_still_require_current_pupil_and_mouth_evidence():
    found=minion_faces(drawing())
    hints=found[0]['goggles']
    mouthless=minion_faces(drawing(missing='mouth'),hints)
    assert mouthless and mouthless[0]['mouth_box'] == [0,0,0,0]
    assert not minion_faces(drawing(missing='pupils'),hints)
    assert not minion_faces(np.zeros((240,320,3),np.uint8),hints)


def test_unreadable_minion_mouth_confirms_face_but_cannot_fill_voiced_cut():
    from autoedit.video.faces import annotate_speaking
    detector=MultiRegionFaceLandmarker(main_group='yellow_minions')
    samples=[]
    try:
        for i in range(12):
            det=detector.detect(drawing(missing='mouth'),i/6)
            det.update(time_sec=i/6,face_visible=float(det['face_detected']))
            samples.append(det)
    finally:
        detector.close()
    annotate_speaking(samples,'animation')
    assert all(s['face_detected'] and not s['clear_face'] and not s['speaking'] for s in samples)
    with pytest.raises(InsufficientFootageError,match='lip movement'):
        match_slots_to_video([{'id':'voice','required_duration_sec':1}],[],
                            {'duration_sec':2,'sample_fps':6,'samples':samples},require_lip_motion=True)


def test_flow_moves_goggle_hints_with_character_translation():
    from autoedit.video.animation import tracked_goggles
    previous=drawing()
    current=cv2.warpAffine(previous,np.float32([[1,0,7],[0,1,4]]),(320,240))
    found=minion_faces(previous)
    hints=tracked_goggles(cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY),
                         cv2.cvtColor(current,cv2.COLOR_BGR2GRAY),found)
    assert hints
    old=found[0]['goggles'][0]
    assert hints[0][:2] == pytest.approx([old[0]+7,old[1]+4],abs=1.)
    assert minion_faces(current,hints)


def test_minion_tracking_does_not_turn_camera_pan_into_speech():
    from autoedit.video.faces import annotate_speaking
    detector=MultiRegionFaceLandmarker(main_group='yellow_minions')
    samples=[]
    try:
        for i in range(12):
            image=cv2.warpAffine(drawing(),np.float32([[1,0,i],[0,1,0]]),(320,240))
            det=detector.detect(image,i/6)
            det['time_sec']=i/6
            samples.append(det)
    finally:
        detector.close()
    annotate_speaking(samples,'animation')
    assert sum(s['clear_face'] for s in samples) >= 10
    assert not any(s['speaking'] for s in samples)


def test_visible_closed_lip_stroke_contributes_to_real_open_close_motion():
    from autoedit.video.faces import annotate_speaking
    detector=MultiRegionFaceLandmarker(main_group='yellow_minions')
    samples=[]
    try:
        for i in range(12):
            det=detector.detect(drawing(mouth_height=0 if i%2 else 10),i/6)
            det['time_sec']=i/6
            samples.append(det)
    finally:
        detector.close()
    assert all(s.get('mouth_box',[0,0,0,0])[2] > 0 for s in samples)
    annotate_speaking(samples,'animation')
    assert all(s['clear_face'] for s in samples)
    assert all(s['speaking'] for s in samples)
