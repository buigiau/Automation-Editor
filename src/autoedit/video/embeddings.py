"""Local ONNX character features. Model files are explicit, fingerprinted inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from autoedit.acceleration import device_settings, is_accelerator_error, prepare_nvidia_runtime

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
        self.requested_device, self.device_index = device_settings(
            config.get("device", "auto"), config.get("device_index", 0))
        self.ort = ort
        self.fallback_reason = None
        self.session = self._create_session()
        self.input = self.session.get_inputs()[0]
        expected = 384 if kind == "animation" else 112
        shape = self.input.shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError("Character model must accept NCHW RGB images, with three channels.")
        self.size = shape[2] if isinstance(shape[2], int) else expected
        if isinstance(shape[3], int) and shape[3] != self.size:
            raise ValueError("Character model input must be square.")

    def _session(self, providers):
        options = self.ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        return self.ort.InferenceSession(str(self.path), options, providers=providers)

    def _create_session(self):
        if self.requested_device != "cpu":
            prepare_nvidia_runtime()
            if "CUDAExecutionProvider" in self.ort.get_available_providers():
                try:
                    if hasattr(self.ort, "preload_dlls"):
                        self.ort.preload_dlls()
                    session = self._session([
                        ("CUDAExecutionProvider", {"device_id": self.device_index, "use_tf32": 0,
                                                   "cudnn_conv_use_max_workspace": 0}),
                        "CPUExecutionProvider"])
                    if "CUDAExecutionProvider" not in session.get_providers():
                        raise RuntimeError("CUDAExecutionProvider could not initialize")
                    # Own the fallback so execution metadata cannot falsely report CUDA.
                    session.disable_fallback()
                    self.device = "cuda"
                    return session
                except Exception as exc:
                    if self.requested_device == "cuda" or not is_accelerator_error(exc):
                        raise
                    self.fallback_reason = str(exc)
            elif self.requested_device == "cuda":
                raise RuntimeError("CUDAExecutionProvider unavailable; install the GPU requirements")
            else:
                self.fallback_reason = "CUDAExecutionProvider unavailable"
        self.device = "cpu"
        return self._session(["CPUExecutionProvider"])

    @property
    def execution(self):
        return {"device": self.device, "providers": self.session.get_providers(),
                "device_index": self.device_index if self.device == "cuda" else None,
                "fallback_reason": self.fallback_reason}

    def _run(self, tensor):
        try:
            return self.session.run(None, {self.input.name: tensor})
        except Exception as exc:
            if self.device != "cuda" or self.requested_device != "auto" or not is_accelerator_error(exc):
                raise
            self.session = None
            self.fallback_reason = str(exc)
            self.device = "cpu"
            self.session = self._session(["CPUExecutionProvider"])
            return self.session.run(None, {self.input.name: tensor})

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
        vector = np.asarray(self._run(tensor)[0], np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or norm <= 1e-8:
            raise ValueError("Character embedding model returned an invalid vector.")
        return vector / norm
