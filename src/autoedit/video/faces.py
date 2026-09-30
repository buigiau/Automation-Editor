"""Face landmarks and temporal lip motion. No scene-motion fallback."""

from pathlib import Path
import math

from autoedit.video.mouth import classify_mouth

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"


class FaceLandmarker:
    def __init__(self, model_path=None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker as Task, FaceLandmarkerOptions, RunningMode

        self.mp = mp
        path = Path(model_path or MODEL_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"Face landmark model missing: {path}. Run scripts/setup_face_model.py first.")
        self.task = Task.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), running_mode=RunningMode.VIDEO,
            num_faces=8, min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6))

    def close(self):
        self.task.close()

    def detect(self, frame, time_sec):
        candidates = self.detect_all(frame, time_sec)
        if not candidates:
            return {"backend": "mediapipe-landmarks", "face_detected": False,
                    "category": "NEUTRAL", "confidence": 0.0, "faces": []}
        result = dict(max(candidates, key=lambda d: d["face"][2] * d["face"][3]))
        result.update(faces=candidates, face_candidate_count=len(candidates))
        return result

    def detect_all(self, frame, time_sec):
        import numpy as np
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        result = self.task.detect_for_video(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb),
                                           int(round(time_sec * 1000)))
        h, w = frame.shape[:2]
        return [self._describe(points, w, h) for points in result.face_landmarks]

    @staticmethod
    def _describe(points, w, h):

        def distance(a, b):
            return math.hypot((points[a].x - points[b].x) * w, (points[a].y - points[b].y) * h)

        mouth_width = max(distance(61, 291), 1)
        aperture = distance(13, 14) / mouth_width
        width_ratio = mouth_width / max(distance(33, 263), 1)
        xs, ys = [p.x for p in points], [p.y for p in points]
        face = [int(min(xs) * w), int(min(ys) * h), int((max(xs) - min(xs)) * w), int((max(ys) - min(ys)) * h)]
        lips = [points[i] for i in (0, 13, 14, 17, 61, 291)]
        mx, my = int(min(p.x for p in lips) * w), int(min(p.y for p in lips) * h)
        mw = max(4, int((max(p.x for p in lips) - min(p.x for p in lips)) * w))
        mh = max(4, int((max(p.y for p in lips) - min(p.y for p in lips)) * h))
        center = (points[61].x + points[291].x) / 2
        frontal = max(0.0, 1 - abs(points[1].x - center) * w / mouth_width * 2)
        openness = min(1.0, aperture * 2)
        width = min(1.0, width_ratio)
        roundness = max(0.0, 1 - abs(aperture - 0.6))
        category, confidence = classify_mouth(openness, width, roundness, 0.0)
        return {"backend": "mediapipe-landmarks", "face_detected": True, "face": face,
                "face_keypoints": [[points[i].x * w, points[i].y * h]
                                   for i in (33, 133, 362, 263, 1, 61, 291)],
                "mouth_box": [mx, my, mw, mh], "frontal_score": frontal,
                "lip_aperture": aperture, "lip_width_ratio": width_ratio,
                "openness": openness, "width": width, "roundness": roundness, "smile": 0.0,
                "category": category, "confidence": confidence}


def _same_face(a, b):
    ax, ay, aw, ah = a["face"]
    bx, by, bw, bh = b["face"]
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return overlap / max(1, aw * ah + bw * bh - overlap) >= 0.45


class MultiRegionFaceLandmarker:
    """Track faces in overlapping views so small/grouped faces are not missed."""

    def __init__(self):
        self.detectors = []
        self.previous = None
        try:
            for _ in range(4):
                self.detectors.append(FaceLandmarker())
        except Exception:
            self.close()
            raise

    def close(self):
        for detector in self.detectors:
            detector.close()
        self.detectors = []

    def detect(self, frame, time_sec):
        candidates = self.detect_all(frame, time_sec)
        if not candidates:
            self.previous = None
            return {"backend": "mediapipe-landmarks", "face_detected": False,
                    "category": "NEUTRAL", "confidence": 0.0, "face_candidate_count": 0, "faces": []}
        def score(det):
            return (det.get("closeup_score", 0) + det.get("mouth_clarity", 0)
                    + (.15 if self.previous and _same_face(self.previous, det) else 0))
        result = dict(max(candidates, key=score))
        result.update(faces=candidates, face_candidate_count=len(candidates))
        self.previous = result
        return result

    def detect_all(self, frame, time_sec):
        import cv2
        from autoedit.video.analyzer import _visual_quality

        h, w = frame.shape[:2]
        regions = [(0, 0, w, h),
                   (0, int(.15 * h), int(.6 * w), int(.85 * h)),
                   (int(.2 * w), int(.15 * h), int(.8 * w), int(.85 * h)),
                   (int(.4 * w), int(.15 * h), w, int(.85 * h))]
        candidates = []
        for index, (detector, (x, y, right, bottom)) in enumerate(zip(self.detectors, regions)):
            if hasattr(detector, "detect_all"):
                found = detector.detect_all(frame[y:bottom, x:right], time_sec)
            else:
                det = detector.detect(frame[y:bottom, x:right], time_sec)
                found = [det] if det.get("face_detected") else []
            # Landmarks are measured in each crop; all subsequent scoring and
            # continuity checks must use coordinates in the original frame.
            for det in found:
                for key in ("face", "mouth_box"):
                    bx, by, bw, bh = det[key]
                    det[key] = [bx + x, by + y, bw, bh]
                if det.get("face_keypoints"):
                    det["face_keypoints"] = [[px + x, py + y] for px, py in det["face_keypoints"]]
                quality, _ = _visual_quality(cv2, frame, det)
                det.update(quality, face_detection_region=index)
                candidates.append(det)
        # Overlapping search regions must not count the same face several times.
        unique = []
        for det in sorted(candidates, key=lambda d: d.get("mouth_clarity", 0), reverse=True):
            if not any(_same_face(det, other) for other in unique):
                unique.append(det)
        return unique


def annotate_speaking(samples, source_kind="live_action"):
    """Require clear landmarks and repeated lip-shape changes in a local window.

    This is visual evidence of speaking, not verification of spoken words.
    Normalized lip geometry avoids counting camera/head motion as speech.
    """
    for sample in samples:
        sample["clear_face"] = float(
            sample.get("backend") == "mediapipe-landmarks" and sample.get("face_detected")
            and sample.get("closeup_score", 0) >= 0.25
            and sample.get("mouth_clarity", 0) >= 0.25
            and sample.get("frontal_score", 0) >= (0.35 if source_kind == "animation" else 0.55)
            and sample.get("mouth_box", [0, 0, 0, 0])[2] >= 18)
        sample["speaking"] = 0.0
    from bisect import bisect_left, bisect_right
    times = [sample["time_sec"] for sample in samples]
    for sample in samples:
        if not sample["clear_face"]:
            continue
        left = bisect_left(times, sample["time_sec"] - 0.8)
        right = bisect_right(times, sample["time_sec"] + 0.8)
        window = [s for s in samples[left:right]
                  if abs(s["time_sec"] - sample["time_sec"]) <= 0.8 and s["clear_face"]
                  and s.get("shot_id") == sample.get("shot_id")
                  and s.get("track_id") == sample.get("track_id")
                  and _same_face(sample, s)]
        if len(window) < 3:
            continue
        # A cut or missing-face interval must not become apparent lip motion.
        if any(b["time_sec"] - a["time_sec"] > 0.6 for a, b in zip(window, window[1:])):
            continue
        apertures = [s.get("lip_aperture", 0) for s in window]
        widths = [s.get("lip_width_ratio", 0) for s in window]
        changes = sum(abs(b - a) >= 0.015 for a, b in zip(apertures, apertures[1:]))
        moving = changes >= 2 and max(apertures) - min(apertures) >= 0.04
        reshaping = max(widths) - min(widths) >= 0.04 and max(apertures) - min(apertures) >= 0.025
        sample["speaking"] = float(moving or reshaping)
