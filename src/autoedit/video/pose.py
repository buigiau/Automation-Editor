"""Pose landmarks for the full-body shot fallback."""
from pathlib import Path
import math

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "pose_landmarker_lite.task"


def pose_metrics(points, width, height):
    missing = {"person_detected": False, "pose_confirmed": False,
               "person_box": None, "shot_scale": None}
    if len(points) < 33:
        return missing
    indices = {i for i, p in enumerate(points)
               if p.visibility >= .6 and p.presence >= .6
               and math.isfinite(p.x) and math.isfinite(p.y)
               and 0 <= p.x <= 1 and 0 <= p.y <= 1}
    # Four arbitrary landmarks are not sufficient evidence of a person.
    # Require a torso and limbs in the image, not just a tracked pose box.
    if (not {11, 12, 23, 24}.issubset(indices) or len(indices) < 8
            or len(indices & {13, 14, 15, 16, 25, 26, 27, 28}) < 2):
        return missing
    visible = [points[i] for i in indices]
    left = max(0.0, min(1.0, min(p.x for p in visible)))
    right = max(0.0, min(1.0, max(p.x for p in visible)))
    top = max(0.0, min(1.0, min(p.y for p in visible)))
    bottom = max(0.0, min(1.0, max(p.y for p in visible)))
    ratio = bottom - top
    box_width = right - left
    if (min(box_width, ratio) < .06 or max(box_width, ratio) < .22
            or box_width * ratio < .018):
        return missing
    shoulders = [(points[i].x, points[i].y) for i in (11, 12)]
    hips = [(points[i].x, points[i].y) for i in (23, 24)]
    shoulder_center = tuple(sum(p[j] for p in shoulders) / 2 for j in (0, 1))
    hip_center = tuple(sum(p[j] for p in hips) / 2 for j in (0, 1))
    if (math.dist(*shoulders) < .025 or math.dist(*hips) < .015
            or math.dist(shoulder_center, hip_center) < .06):
        return missing
    # Actual leg landmarks distinguish full body from upper body; a tiny
    # horizontal detection on a console must not become "distant full body".
    scale = "full_body" if len(indices & {25, 26, 27, 28}) >= 3 else "upper_body"
    return {"person_detected": True, "pose_confirmed": True,
            "person_box": [int(left * width), int(top * height),
                           int((right-left) * width), int(ratio * height)],
            "shot_scale": scale}


class PoseLandmarker:
    def __init__(self, model_path=None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker as Task, PoseLandmarkerOptions, RunningMode
        path = Path(model_path or MODEL_PATH)
        if not path.is_file():
            raise FileNotFoundError(f"Pose model missing: {path}. Run scripts/setup_face_model.py first.")
        self.mp = mp
        self.task = Task.create_from_options(PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)), running_mode=RunningMode.IMAGE,
            num_poses=3, min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6))

    def close(self):
        self.task.close()

    def detect(self, frame, time_sec):
        import numpy as np
        # Re-detect independently: video pose tracking was persisting body
        # landmarks onto screens/vehicles across edits in the source footage.
        result = self.task.detect(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(frame[:, :, ::-1])))
        h, w = frame.shape[:2]
        candidates = [pose_metrics(points, w, h) for points in result.pose_landmarks]
        confirmed = [m for m in candidates if m["pose_confirmed"]]
        selected = max(confirmed, key=lambda m: m["person_box"][2] * m["person_box"][3]) if confirmed else pose_metrics([], w, h)
        return {**selected, "pose_candidate_count": len(candidates)}
