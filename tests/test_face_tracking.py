import cv2
import numpy as np

from autoedit.video.tracking import FaceFeatureTracker
from autoedit.match.matcher import _has_face


def textured_face():
    image = np.zeros((160,240,3),np.uint8)
    image[30:130,60:160] = np.random.default_rng(71).integers(30,220,(100,100,3),dtype=np.uint8)
    return image


def detection():
    return {'backend':'mediapipe-landmarks','face_detected':True,'face':[60,30,100,100],
            'mouth_box':[85,90,40,20],'lip_aperture':.8,'speaking':1,'clear_face':1}


def test_verified_pan_bridges_missing_detection_without_inventing_lips():
    frame = textured_face()
    tracker = FaceFeatureTracker()
    tracker.update(frame,0,[detection()])
    moved = cv2.warpAffine(frame,np.float32([[1,0,4],[0,1,2]]),(240,160))
    found = tracker.update(moved,1/6,[])
    assert len(found) == 1 and _has_face(found[0])
    assert np.allclose(found[0]['face'][:2],[64,32],atol=1)
    assert not found[0]['speaking'] and not found[0]['clear_face']
    assert 'lip_aperture' not in found[0]


def test_tracking_discards_subject_at_cut_and_cannot_bootstrap_scenery():
    frame = textured_face()
    tracker = FaceFeatureTracker()
    assert tracker.update(frame,0,[]) == []
    tracker.update(frame,1/6,[detection()])
    assert tracker.update(frame,2/6,[],cut=True) == []
    assert tracker.update(frame,3/6,[]) == []


def test_tracking_cannot_survive_a_flash_or_exceed_anchor_age():
    frame = textured_face()
    tracker = FaceFeatureTracker(max_age_sec=.4)
    tracker.update(frame,0,[detection()])
    assert tracker.update(np.full_like(frame,255),1/6,[]) == []
    tracker.update(frame,1,[detection()])
    assert tracker.update(frame,1+1/6,[])
    assert tracker.update(frame,1+2/6,[])
    assert tracker.update(frame,1.5,[]) == []
