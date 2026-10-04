"""Verified goggles/pupils and mouth geometry for supported animation styles.

This detector abstains on unsupported styles. It never treats a centre crop,
scene motion or a character embedding as evidence of a face.
"""
from itertools import combinations
import math

import numpy as np

from autoedit.video.mouth import classify_mouth


MINION_EVIDENCE = "goggle-pupil-yellow-mouth"
MINION_FACE_EVIDENCE = "goggle-pupil-yellow-face"


def minion_faces(frame, goggle_hints=()):
    """Conservative one/two-goggle geometry for yellow 3D Minions.

    Colour alone cannot qualify a face: a goggle, contrasting pupil and
    yellow cheeks are required. Unreadable mouths never count as speech.
    """
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    yellow = (hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 50)
    connected = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    _, skin_labels = cv2.connectedComponents(connected)

    def skin_parts(x,y,pad):
        x,y,pad = round(x),round(y),max(2,round(pad))
        parts=skin_labels[min(height,max(0,y-pad)):min(height,max(0,y+pad+1)),
                          min(width,max(0,x-pad)):min(width,max(0,x+pad+1))]
        values, counts=np.unique(parts,return_counts=True)
        return {int(v) for v,n in zip(values,counts) if v and n >= max(3,parts.size*.15)}
    circle_scale = min(1.,320/width)
    circle_gray = cv2.resize(gray,None,fx=circle_scale,fy=circle_scale,interpolation=cv2.INTER_AREA)
    circles = cv2.HoughCircles(cv2.GaussianBlur(circle_gray, (5, 5) if circle_scale == 1 else (3, 3), 0),
                               cv2.HOUGH_GRADIENT,1.1,12 if circle_scale == 1 else 8,
                               param1=100,param2=35 if circle_scale == 1 else 25,
                               minRadius=5 if circle_scale == 1 else 4,
                               maxRadius=round(min(width*.15,height*.3)*circle_scale))
    eyes, possible_eyes = [], []
    # Hough orders hypotheses by votes. Bound work on busy crowd shots instead
    # of evaluating hundreds of weak circles from clothing/background texture.
    hypotheses = [tuple(value*circle_scale for value in eye) for eye in goggle_hints]
    hypotheses.extend(() if circles is None else circles[0][:64])
    for cx, cy, radius in hypotheses:
        cx,cy,radius = float(cx)/circle_scale,float(cy)/circle_scale,float(radius)/circle_scale
        x0, x1 = max(0, int(cx-radius*1.7)), min(width, int(cx+radius*1.7)+1)
        y0, y1 = max(0, int(cy-radius*1.7)), min(height, int(cy+radius*1.7)+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        distance = np.hypot(xx-cx, yy-cy) / radius
        surround = (distance > 1.05) & (distance < 1.7)
        core = distance < .8
        if not core.any() or not surround.any() or yellow[y0:y1, x0:x1][surround].mean() < .18:
            continue
        values = gray[y0:y1, x0:x1][core]
        # A blank lens/window has no pupil. Require bright sclera and a
        # substantial dark interior, independent of the goggle rim.
        low, high = np.percentile(values, [5,85])
        if high - low < 35:
            continue
        bright = (hsv[y0:y1,x0:x1,1][core] < 125) & (values > 70)
        ring = (distance > .85) & (distance < 1.1)
        if not ring.any() or (hsv[y0:y1,x0:x1,1][ring] < 110).mean() < .5:
            continue
        if (bright.mean() < .3 or yellow[y0:y1,x0:x1][core].mean() > .6
                or (values < high*.65).mean() < .04):
            continue
        # The sclera must be one surface. Separate goggles, striped clothing
        # and several small faces inside a large circle cannot form one eye.
        sclera = ((hsv[y0:y1,x0:x1,1] < 125) &
                  (gray[y0:y1,x0:x1] > 70) & core).astype(np.uint8)
        _, _, sclera_parts, _ = cv2.connectedComponentsWithStats(sclera)
        largest_sclera = max(sclera_parts[1:,cv2.CC_STAT_AREA], default=0)
        if largest_sclera < .65 * sclera.sum():
            continue
        pupil_mask = ((gray[y0:y1,x0:x1] < min(110,high*.65))
                      & (distance < .75)).astype(np.uint8)*255
        pupils,_ = cv2.findContours(pupil_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        pupil_found = False
        for pupil in pupils:
            px,py,pw,ph = cv2.boundingRect(pupil)
            area = cv2.contourArea(pupil)
            if (max(3,radius*radius*.015) <= area <= radius*radius*.65
                    and .35 <= pw/ph <= 2.5 and max(pw,ph) < radius*1.1
                    and (pupil_mask[py:py+ph,px:px+pw] != 0).sum() >= .8 * area
                    and area/max(1,cv2.contourArea(cv2.convexHull(pupil))) > .65
                    and math.hypot(x0+px+pw/2-cx,y0+py+ph/2-cy) < radius*.8):
                pupil_found = True
                break
        # Check the same radial directions through sclera, metal and yellow
        # cheeks. A large circle spanning several small Minions may contain
        # all three colours without any actual goggle surface between them.
        angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
        def polar(factor):
            px=np.clip(np.round(cx+radius*factor*np.cos(angles)).astype(int),0,width-1)
            py=np.clip(np.round(cy+radius*factor*np.sin(angles)).astype(int),0,height-1)
            return hsv[py,px],gray[py,px],yellow[py,px]
        lens,lens_gray,_=polar(.65)
        rim,_,_=polar(.95)
        _,_,cheek=polar(1.3)
        surface = (lens[:,1] < 125) & (lens_gray > 70) & (rim[:,1] < 110) & cheek
        pupil_found = pupil_found and surface.mean() >= .20
        eye = (cx, cy, radius)
        if not pupil_found:
            possible_eyes.append(eye)
            continue
        if not any(math.hypot(cx-x,cy-y) < max(radius,r)*.8 for x,y,r in eyes):
            eyes.append(eye)
    verified_eyes = set(eyes)
    for cx,cy,radius in possible_eyes:
        if not any(math.hypot(cx-x,cy-y) < max(radius,r)*.8 for x,y,r in eyes):
            eyes.append((cx,cy,radius))
    # Pair only neighbouring goggles, then evaluate lone goggles too. Mouth
    # and cheek checks below prevent arbitrary yellow objects becoming faces.
    proposals = []
    for left, right in combinations(sorted(eyes), 2):
        dx, dy = right[0]-left[0], right[1]-left[1]
        if (.5*(left[2]+right[2]) <= dx <= 1.15*(left[2]+right[2])
                and (left in verified_eyes or right in verified_eyes)
                and abs(dy) < dx*.55 and min(left[2],right[2])/max(left[2],right[2]) > .5
                and not any(left[0] < eye[0] < right[0] and abs(eye[1]-(left[1]+right[1])/2) < max(left[2],right[2]) for eye in eyes)):
            proposals.append((left, right))
    proposals.extend((eye,) for eye in eyes if eye in verified_eyes)
    faces, used_eyes = [], set()
    for group in proposals:
        if any(eye in used_eyes for eye in group):
            continue
        cx = sum(e[0] for e in group)/len(group)
        cy = sum(e[1] for e in group)/len(group)
        radius = max(e[2] for e in group)
        scale = max(2*radius, group[-1][0]-group[0][0])
        mx0, mx1 = max(0, round(cx-scale*.8)), min(width, round(cx+scale*.8))
        my0 = max(0, round(max(e[1]+e[2] for e in group) + radius*.1))
        my1 = min(height, round(cy+radius*3.5))
        skin = yellow[my0:my1,mx0:mx1]
        if not skin.size or skin.mean() < .2:
            continue
        patch = hsv[my0:my1,mx0:mx1,2]
        cutoff = min(85, float(np.median(patch[skin]))*.45)
        dark = (patch < cutoff).astype(np.uint8)*255
        mouths,_ = cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        choices = []
        for contour in mouths:
            x,y,w,h = cv2.boundingRect(contour)
            if not (max(4,scale*.08) <= w <= scale*1.4 and 1 <= h <= scale*.65
                    and abs(mx0+x+w/2-cx) <= scale*.35 and y > 0
                    and my0+y+h/2 >= cy+radius*1.5
                    and x > 0 and x+w < patch.shape[1]-1 and y+h < patch.shape[0]-1):
                continue
            # Yellow around the mouth distinguishes it from overalls, gloves
            # and the gaps between several characters.
            pad = max(2,round(radius*.15))
            region = skin[max(0,y-pad):min(skin.shape[0],y+h+pad),max(0,x-pad):min(skin.shape[1],x+w+pad)]
            if region.mean() < .22:
                continue
            choices.append((x,y,w,h))
        if not choices:
            # A closed/occluded mouth must not erase an independently confirmed
            # Minion face. Keep it visible, with no lip or speaking evidence.
            parts = [skin_parts(ex,ey+er*1.5,er*.25) for ex,ey,er in group]
            x0 = max(0,round(min(e[0]-e[2] for e in group)-radius*.35))
            x1 = min(width,round(max(e[0]+e[2] for e in group)+radius*.35))
            y0 = max(0,round(min(e[1]-e[2] for e in group)-radius*.55))
            y1 = min(height,round(cy+radius*2.6))
            if (parts and set.intersection(*parts) and yellow[y0:y1,x0:x1].mean() >= .25
                    and not any(_overlap([x0,y0,x1-x0,y1-y0],f['face']) > .45 for f in faces)):
                faces.append(dict(backend="animation-eyes-mouth",face_detected=True,
                    face=[x0,y0,x1-x0,y1-y0],mouth_box=[0,0,0,0],mouth_clarity=0.,
                    frontal_score=.85,eye_distance=scale,eye_count=len(group),
                    goggles=[list(eye) for eye in group],category="NEUTRAL",confidence=.7,
                    face_evidence=MINION_FACE_EVIDENCE,character_group="yellow_minions"))
                used_eyes.update(group)
            continue
        x,y,mw,mh = max(choices,key=lambda b:b[2]*b[3])
        mx,my = mx0+x,my0+y
        # The goggles and both mouth corners must belong to the same yellow
        # cheek surface. This rejects pairing eyes across a Minions crowd.
        parts = [skin_parts(ex,ey+er*1.5,er*.25) for ex,ey,er in group]
        parts.extend(skin_parts(px,my+mh/2,radius*.2) for px in (mx-radius*.15,mx+mw+radius*.15))
        if not parts or not set.intersection(*parts):
            continue
        x0 = max(0,round(min(e[0]-e[2] for e in group)-radius*.35))
        x1 = min(width,round(max(e[0]+e[2] for e in group)+radius*.35))
        y0 = max(0,round(min(e[1]-e[2] for e in group)-radius*.55))
        y1 = min(height,round(my+mh+radius*.3))
        if yellow[y0:y1,x0:x1].mean() < .25:
            continue
        aperture = max(0.,(mh-1)/mw)
        openness, lip_width = min(1.,aperture*2), min(1.,mw/scale)
        roundness = max(0.,1-abs(aperture-.6))
        category, confidence = classify_mouth(openness,lip_width,roundness,0.)
        candidate = dict(backend="animation-eyes-mouth",face_detected=True,
                         face=[x0,y0,x1-x0,y1-y0],mouth_box=[mx,my,mw,mh],
                         frontal_score=.85,lip_aperture=aperture,lip_width_ratio=mw/scale,
                         eye_distance=scale,eye_count=len(group),openness=openness,width=lip_width,
                         goggles=[list(eye) for eye in group],
                         roundness=roundness,smile=0.,category=category,confidence=min(.85,confidence),
                         face_evidence=MINION_EVIDENCE,character_group="yellow_minions")
        if not any(_overlap(candidate['face'],f['face']) > .45 for f in faces):
            faces.append(candidate)
            used_eyes.update(group)
    return faces


def tracked_goggles(previous_gray, gray, faces):
    """Carry measured goggle proposals through an independently checked flow.

    These are hints only: the current frame must still pass colour, goggle
    and pupil checks. Mouth evidence is measured separately in that frame.
    """
    import cv2
    if previous_gray is None or previous_gray.shape != gray.shape:
        return []
    hints=[]
    height,width=gray.shape
    for face in faces:
        if not face.get('goggles'):
            continue
        x,y,w,h=face['face']
        mask=np.zeros_like(gray)
        mask[max(0,y):min(height,y+h),max(0,x):min(width,x+w)]=255
        points=cv2.goodFeaturesToTrack(previous_gray,60,.02,4,mask=mask)
        if points is None or len(points) < 6:
            continue
        moved,status,_=cv2.calcOpticalFlowPyrLK(previous_gray,gray,points,None)
        if moved is None or status is None:
            continue
        back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,previous_gray,moved,None)
        if back is None or back_status is None:
            continue
        valid=(status[:,0] != 0) & (back_status[:,0] != 0) & (np.linalg.norm(back[:,0]-points[:,0],axis=1) < 1.5)
        if valid.sum() < 6 or valid.mean() < .6:
            continue
        transform,inliers=cv2.estimateAffinePartial2D(points[valid],moved[valid],method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.)
        if transform is None or inliers.mean() < .7:
            continue
        scale=float(np.linalg.norm(transform[:,0]))
        if not .75 < scale < 1.33:
            continue
        for gx,gy,radius in face['goggles']:
            center=transform @ np.asarray([gx,gy,1.])
            if 0 <= center[0] < width and 0 <= center[1] < height:
                hints.append([float(center[0]),float(center[1]),radius*scale])
    return hints


def cartoon_faces(frame):
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bright = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 190)).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(bright, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    eyes = []
    for index, contour in enumerate(contours):
        if hierarchy[0][index][3] >= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if not (4 <= w <= width * .12 and 3 <= h <= height * .15 and .4 <= w / h <= 3):
            continue
        dark = gray[y:y+h, x:x+w] < 65
        if not .08 * w * h <= dark.sum() <= .8 * w * h:
            continue
        yy, xx = np.nonzero(dark)
        eyes.append((x, y, w, h, x + float(xx.mean()), y + float(yy.mean())))
    faces = []
    for one, two in combinations(eyes, 2):
        left, right = sorted((one, two), key=lambda eye: eye[4])
        dx, dy = right[4] - left[4], right[5] - left[5]
        size = max(left[2], right[2])
        if not (size * .8 <= dx <= size * 3.5 and abs(dy) <= max(3, dx * .4)
                and min(left[3], right[3]) / max(left[3], right[3]) >= .5):
            continue
        cx, cy = (left[4] + right[4]) / 2, (left[5] + right[5]) / 2
        # Both eyes must be embedded in one coloured face, not separate letters,
        # windows or unrelated high-contrast objects.
        x0, x1 = max(0, round(cx - dx * 1.5)), min(width, round(cx + dx * 1.5))
        y0, y1 = max(0, round(cy - dx)), min(height, round(cy + dx * 2))
        region = hsv[y0:y1, x0:x1]
        if not region.size:
            continue
        probes = [hsv[min(height-1, max(0, round(cy + offset*dx))), round(cx)]
                  for offset in (-.5, .25)]
        skin = max(probes, key=lambda pixel: int(pixel[1]))
        if skin[1] < 55 or skin[2] < 70:
            continue
        hue_delta = np.abs(region[:, :, 0].astype(float) - float(skin[0]))
        skin_mask = ((np.minimum(hue_delta, 180-hue_delta) < 12) &
                     (region[:, :, 1] > 40) & (region[:, :, 2] > 65)).astype(np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # This path is for a flat face surface. A crowd of 3D characters or
        # patterned clothing can share eye-like bright blobs and some colour,
        # but cannot provide one predominantly coloured face region.
        if skin_mask.mean() < .5:
            continue
        _, labels, stats, _ = cv2.connectedComponentsWithStats(skin_mask)
        point_labels = []
        for eye in (left, right):
            ex, ey = round(eye[4])-x0, min(y1-y0-1, round(eye[5]+eye[3])-y0)
            point_labels.append(labels[max(0, ey), min(x1-x0-1, max(0, ex))])
        if not point_labels[0] or point_labels[0] != point_labels[1]:
            continue
        body = stats[point_labels[0]]
        if body[cv2.CC_STAT_AREA] < dx * dx * 1.2:
            continue
        # Find a real dark mouth below the eyes, aligned to their midpoint.
        mx0, mx1 = max(0, round(cx-dx*.8)), min(width, round(cx+dx*.8))
        my0, my1 = max(0, round(cy+max(left[3],right[3])*.6)), min(height, round(cy+dx*1.5))
        dark = (gray[my0:my1, mx0:mx1] < 100).astype(np.uint8)*255
        if not dark.size:
            continue
        mouths, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mouths = [cv2.boundingRect(c) for c in mouths]
        mouths = [(x,y,w,h) for x,y,w,h in mouths if w >= 3 and h >= 1
                  and w <= dx*1.5 and h <= dx and abs(mx0+x+w/2-cx) <= dx*.45
                  and my0+y+h < y1-1]
        if not mouths:
            continue
        mx, my, mw, mh = max(mouths, key=lambda box: box[2]*box[3])
        mx, my = mx+mx0, my+my0
        aperture = max(0., (mh-1)/max(mw,1))
        openness, lip_width = min(1., aperture*2), min(1., mw/dx)
        roundness = max(0., 1-abs(aperture-.6))
        category, confidence = classify_mouth(openness, lip_width, roundness, 0.)
        face = [x0, y0, x1-x0, y1-y0]
        candidate = {"backend":"animation-eyes-mouth", "face_detected":True,
                     "face":face, "mouth_box":[mx,my,mw,mh], "frontal_score":1.,
                     "lip_aperture":aperture, "lip_width_ratio":mw/dx,
                     "eye_distance":dx, "eye_angle":math.atan2(dy,dx),
                     "openness":openness, "width":lip_width, "roundness":roundness,
                     "smile":0., "category":category, "confidence":min(.85,confidence),
                     "face_evidence":"paired-pupils-shared-colour-mouth"}
        if not any(_overlap(face, other['face']) > .45 for other in faces):
            faces.append(candidate)
    return faces


def _overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    area = max(0, min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return area/max(1,aw*ah+bw*bh-area)
