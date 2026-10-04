"""Bridge brief landmark misses with independently verified image features.

Tracks establish visibility only. Lip geometry is never copied from an older
frame, and failed tracks, cuts and stale detections discard the subject.
"""
import numpy as np


class FaceFeatureTracker:
    def __init__(self, max_age_sec=1.0):
        self.previous = None
        self.faces = []
        self.max_age_sec = max_age_sec
        self.last_time = None

    def update(self, frame, time, detected, cut=False):
        import cv2
        from autoedit.video.characters import iou
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = [dict(face, feature_anchor_sec=time) for face in detected]
        if (self.previous is not None and not cut and self.last_time is not None
                and 0 < time-self.last_time <= .3):
            for face in self.faces:
                if time-face['feature_anchor_sec'] > self.max_age_sec:
                    continue
                tracked = track_face(self.previous, gray, face)
                if tracked and not any(iou(tracked['face'], other['face']) >= .25 for other in result):
                    result.append(tracked)
        self.previous, self.faces, self.last_time = gray, result, time
        return result


def track_face(previous, current, face):
    import cv2
    x,y,w,h = face['face']
    height,width = current.shape
    if min(w,h) < 20:
        return None
    mask = np.zeros_like(previous)
    mask[max(0,y):min(height,y+h),max(0,x):min(width,x+w)] = 255
    points = cv2.goodFeaturesToTrack(previous,80,.02,3,mask=mask)
    if points is None or len(points) < 12:
        return None
    moved,status,_ = cv2.calcOpticalFlowPyrLK(previous,current,points,None)
    if moved is None:
        return None
    back,back_status,_ = cv2.calcOpticalFlowPyrLK(current,previous,moved,None)
    if back is None:
        return None
    valid = ((status[:,0] != 0) & (back_status[:,0] != 0)
             & (np.linalg.norm(back[:,0]-points[:,0],axis=1) < .75))
    if valid.sum() < 12 or valid.mean() < .7:
        return None
    transform,inliers = cv2.estimateAffinePartial2D(points[valid],moved[valid],
        method=cv2.RANSAC,ransacReprojThreshold=2.)
    if transform is None or inliers.mean() < .75:
        return None
    original = points[valid,0][inliers[:,0] != 0]
    destination = moved[valid,0][inliers[:,0] != 0]
    if len(original) < 12 or np.ptp(original[:,0])*np.ptp(original[:,1]) < w*h*.25:
        return None
    scale = float(np.linalg.norm(transform[:,0]))
    if not .8 <= scale <= 1.25:
        return None
    differences = [abs(float(cv2.getRectSubPix(previous,(3,3),tuple(a)).mean())
                       - float(cv2.getRectSubPix(current,(3,3),tuple(b)).mean()))
                   for a,b in zip(original,destination)]
    if np.median(differences) > 15 or np.quantile(differences,.8) > 35:
        return None
    corners = np.array([[x,y,1],[x+w,y,1],[x,y+h,1],[x+w,y+h,1]],dtype=float) @ transform.T
    left,top = np.floor(corners.min(axis=0)).astype(int)
    right,bottom = np.ceil(corners.max(axis=0)).astype(int)
    if left < 0 or top < 0 or right > width or bottom > height:
        return None
    return {'backend':'feature-tracked-face','face_detected':True,
        'face_evidence':'reversible-face-features','feature_anchor_sec':face['feature_anchor_sec'],
        'verified_feature_count':len(original),'face':[int(left),int(top),int(right-left),int(bottom-top)],
        'frontal_score':0.,'mouth_box':[0,0,0,0],'category':'NEUTRAL','confidence':.7,
        'clear_face':0.,'speaking':0.,'lip_geometry_measured':False}
