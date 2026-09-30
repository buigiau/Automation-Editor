"""Local ONNX character features. Model files are explicit, fingerprinted inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[3] / "models" / "characters"
CCIP_MODEL = "ccip-caformer-24-randaug-pruned"


def model_path(kind, config):
    name = "ccip_model" if kind == "animation" else "arcface_model"
    default = MODEL_ROOT / ("ccip.onnx" if kind == "animation" else "arcface.onnx")
    path = Path(config.get(name) or default).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Character embedding model missing: {path}. "
            "Run scripts/setup_character_models.py --ccip for animation, or "
            "--arcface-file PATH for licensed ArcFace weights (--arcface installs public research weights). "
            "video.characters.enabled=false disables character filtering explicitly.")
    return path


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EmbeddingModel:
    def __init__(self, kind, config):
        import onnxruntime as ort
        self.kind = kind
        self.path = model_path(kind, config)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        expected = 384 if kind == "animation" else 112
        shape = self.input.shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError("Character model must accept NCHW RGB images, with three channels.")
        self.size = shape[2] if isinstance(shape[2], int) else expected
        if isinstance(shape[3], int) and shape[3] != self.size:
            raise ValueError("Character model input must be square.")

    def extract(self, frame, face, frame_width, frame_height, other_faces=()):
        import cv2
        height, width = frame.shape[:2]
        sx, sy = width / frame_width, height / frame_height
        x, y, w, h = face["face"]
        x, y, w, h = x * sx, y * sy, w * sx, h * sy
        if min(w, h) < 32:
            return None
        if self.kind == "live_action":
            points = np.asarray(face.get("face_keypoints", []), dtype=np.float32)
            if points.shape != (7, 2) or face.get("frontal_score", 0) < .35:
                return None
            points *= [sx, sy]
            eyes = sorted([points[:2].mean(axis=0), points[2:4].mean(axis=0)], key=lambda p: p[0])
            mouth = sorted([points[5], points[6]], key=lambda p: p[0])
            source = np.asarray([*eyes, points[4], *mouth], dtype=np.float32)
            target = np.asarray([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                                 [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32) * (self.size / 112)
            transform, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
            if transform is None:
                return None
            image = cv2.warpAffine(frame, transform, (self.size, self.size))
        else:
            # Include hair/head cues, while rejecting crops containing another face.
            left, top = max(0, int(x - .35 * w)), max(0, int(y - .55 * h))
            right, bottom = min(width, int(x + 1.35 * w)), min(height, int(y + 1.2 * h))
            for other in other_faces:
                if other is face:
                    continue
                ox, oy, ow, oh = other["face"]
                if left < (ox + ow / 2) * sx < right and top < (oy + oh / 2) * sy < bottom:
                    return None
            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                return None
            image = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        rgb = image[:, :, ::-1].astype(np.float32)
        if self.kind == "animation":
            rgb = (rgb / 255 - np.asarray([.48145466, .4578275, .40821073], np.float32)) / np.asarray(
                [.26862954, .26130258, .27577711], np.float32)
        else:
            rgb = (rgb - 127.5) / 127.5
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        vector = np.asarray(self.session.run(None, {self.input.name: tensor})[0], np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm <= 1e-8:
            raise ValueError("Character embedding model returned an invalid vector.")
        return vector / norm
