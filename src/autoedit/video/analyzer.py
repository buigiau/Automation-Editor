"""Sample a source video and emit timed mouth/expression segments."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

from autoedit.video.mouth import classify_mouth, roi_metrics
from autoedit.video.pool import bounded_segments


def _try_cv2():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for video analysis") from exc
    return cv2, np


def _haar_face(cv2):
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.exists():
        return None
    cascade = cv2.CascadeClassifier(str(path))
    return cascade if not cascade.empty() else None


def _detect_opencv(cv2, np, frame) -> dict[str, Any] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = getattr(_detect_opencv, "_cascade", None)
    if cascade is None:
        cascade = _haar_face(cv2)
        _detect_opencv._cascade = cascade
    faces = []
    if cascade is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        # Center-weighted fallback for cartoon / failed Haar
        h, w = gray.shape
        x, y, fw, fh = int(w * 0.25), int(h * 0.15), int(w * 0.5), int(h * 0.7)
        mouth = gray[y + int(fh * 0.55) : y + fh, x : x + fw]
        metrics = roi_metrics(mouth)
        cat, conf = classify_mouth(**{k: metrics[k] for k in ("openness", "width", "roundness", "smile")})
        return {
            "backend": "opencv-center",
            "face_detected": False,
            "face": [x, y, fw, fh],
            "category": cat,
            "confidence": conf * 0.6,
            **metrics,
        }
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    mouth = gray[y + int(fh * 0.55) : y + fh, x : x + fw]
    metrics = roi_metrics(mouth)
    cat, conf = classify_mouth(**{k: metrics[k] for k in ("openness", "width", "roundness", "smile")})
    return {
        "backend": "opencv-haar",
        "face_detected": True,
        "face": [int(x), int(y), int(fw), int(fh)],
        "category": cat,
        "confidence": conf,
        **metrics,
    }


def _visual_quality(cv2, frame, det, previous_gray=None, previous_det=None):
    """Observable quality hints, not semantic judgements of an interesting scene."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 36))
    motion = 0.0 if previous_gray is None else float(cv2.absdiff(small, previous_gray).mean()) / 255
    face_detected = bool(det.get("face_detected"))
    closeup = clarity = 0.0
    if face_detected:
        h, w = gray.shape
        x, y, fw, fh = det["face"]
        closeup = min(1.0, max(0, fw * fh) / max(1, w * h) / 0.22)
        mx, my, mw, mh = det.get("mouth_box") or [x, y + int(fh * 0.55), fw, int(fh * 0.45)]
        # The visible outline is the sharp evidence for a smooth cartoon mouth.
        # A closed one-pixel lip has no interior texture; include its actual edge.
        pad = max(1, round(min(mw,mh)*.15)) if det.get('backend') == 'animation-eyes-mouth' and mw > 0 and mh > 0 else 0
        crop = gray[max(0, my-pad):min(h, my + mh+pad), max(0, mx-pad):min(w, mx + mw+pad)]
        if crop.size:
            sharpness = min(1.0, float(cv2.Laplacian(crop, cv2.CV_64F).var()) / 150)
            visible = float(mx >= 0 and my >= 0 and mx + mw <= w and my + mh <= h)
            reliability = 1.0 if det.get("backend") in {"mediapipe-landmarks", "animation-eyes-mouth"} else 0.55
            clarity = sharpness * visible * reliability * float(det.get("frontal_score", 1))
    mouth_activity = 0.0
    if face_detected and previous_det and previous_det.get("face_detected") and motion < 0.3:
        mouth_activity = min(1.0, abs(det.get("openness", 0) - previous_det.get("openness", 0)) * 5)
    return {"closeup_score": closeup, "mouth_clarity": clarity,
            "face_visible": float(face_detected), "mouth_activity": mouth_activity,
            "activity_score": min(1.0, motion / 0.12)}, small


def _merge_segments(
    samples: list[dict[str, Any]],
    min_sec: float,
    gap: float,
    sample_interval: float = 0.2,
) -> list[dict[str, Any]]:
    if not samples:
        return []
    close_gap = max(gap, sample_interval * 1.51)

    def finish(cur: dict[str, Any]) -> dict[str, Any] | None:
        cur["confidence"] = cur["confidence"] / cur["n"]
        if cur["end_sec"] <= cur["start_sec"]:
            cur["end_sec"] = cur["start_sec"] + sample_interval
        else:
            cur["end_sec"] = cur["end_sec"] + sample_interval
        cur["duration_sec"] = cur["end_sec"] - cur["start_sec"]
        if cur["duration_sec"] >= min_sec:
            return cur
        return None

    segments: list[dict[str, Any]] = []
    cur = {
        "category": samples[0]["category"],
        "start_sec": samples[0]["time_sec"],
        "end_sec": samples[0]["time_sec"],
        "confidence": samples[0]["confidence"],
        "n": 1,
        "backend": samples[0].get("backend"),
    }
    for sample in samples[1:]:
        same = sample["category"] == cur["category"]
        close = sample["time_sec"] - cur["end_sec"] <= close_gap + 1e-6
        if same and close:
            cur["end_sec"] = sample["time_sec"]
            cur["confidence"] += sample["confidence"]
            cur["n"] += 1
        else:
            done = finish(cur)
            if done:
                segments.append(done)
            cur = {
                "category": sample["category"],
                "start_sec": sample["time_sec"],
                "end_sec": sample["time_sec"],
                "confidence": sample["confidence"],
                "n": 1,
                "backend": sample.get("backend"),
            }
    done = finish(cur)
    if done:
        segments.append(done)
    for i, seg in enumerate(segments):
        seg["id"] = f"vseg-{i:04d}"
        seg["confidence"] = round(float(seg["confidence"]), 3)
        seg["start_sec"] = round(float(seg["start_sec"]), 3)
        seg["end_sec"] = round(float(seg["end_sec"]), 3)
        seg["duration_sec"] = round(float(seg["duration_sec"]), 3)
        seg.pop("n", None)
    return segments


ANALYSIS_VERSION = 15


def analyze_video(
    path: str | Path,
    sample_fps: float = 6.0,
    max_seconds: float = 0.0,
    min_segment_sec: float = 0.25,
    merge_gap_sec: float = 0.20,
    backend: str = "auto",
    progress: Callable[[str], None] | None = None,
    cache_dir: str | Path | None = None,
    source_kind: str = "live_action",
    main_group: str = "auto",
) -> dict[str, Any]:
    import gzip
    import hashlib
    import json
    import math
    import time
    from autoedit.video.faces import FaceLandmarker, MultiRegionFaceLandmarker, MODEL_PATH, annotate_speaking
    from autoedit.video.reader import sampled_frames
    from autoedit.video.pose import PoseLandmarker, MODEL_PATH as POSE_MODEL_PATH

    if not math.isfinite(sample_fps) or not 2 <= sample_fps <= 10:
        raise ValueError("Video sampling must be between 2 and 10 fps for temporal lip detection.")
    if backend not in {"auto", "mediapipe", "opencv"}:
        raise ValueError("Unknown video backend")
    if source_kind not in {"live_action", "animation"}:
        raise ValueError("Unknown video source kind")
    if main_group not in {"auto", "yellow_minions"}:
        raise ValueError("Unknown main character group")
    if main_group == "yellow_minions" and (source_kind != "animation" or backend == "opencv"):
        raise ValueError("Yellow Minions require animation with the auto/mediapipe backend")
    src = Path(path).resolve()
    stat = src.stat()
    identity = {"path": str(src), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sample_fps": sample_fps, "max_seconds": max_seconds,
                "min_segment_sec": min_segment_sec, "merge_gap_sec": merge_gap_sec,
                "backend": backend, "source_kind": source_kind, "version": ANALYSIS_VERSION,
                "pose_model_mtime": POSE_MODEL_PATH.stat().st_mtime_ns if POSE_MODEL_PATH.is_file() else None,
                "model_mtime": MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else None}
    if main_group != "auto":
        identity.update(main_group=main_group, main_group_version=4)
    if source_kind == 'animation':
        identity['animation_detector_version'] = 7
    cache = None
    if cache_dir:
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cache = Path(cache_dir) / (key + ".json.gz")
        if cache.is_file():
            try:
                info = json.loads(gzip.decompress(cache.read_bytes()))
                if info.get("cache_identity") == identity:
                    analyzed_end = min(float(info["duration_sec"]), float(info["analyzed_seconds"]))
                    info["segments"] = bounded_segments(info.get("segments", []), analyzed_end, min_segment_sec)
                    if progress:
                        progress("Video analysis cache hit: reusing face/lip samples; no video rescan.")
                    return info
            except (OSError, ValueError, EOFError):
                pass

    cv2, np = _try_cv2()
    detector = pose = None
    if backend != "opencv":
        # Fail explicitly: missing/broken landmarks must never fall back to a scene crop.
        try:
            detector = ((MultiRegionFaceLandmarker() if main_group == "auto" else
                         MultiRegionFaceLandmarker(main_group=main_group))
                        if source_kind == "animation" else MultiRegionFaceLandmarker(include_animation=False))
            if main_group == "auto":
                pose = PoseLandmarker()
        except ImportError as exc:
            if detector:
                detector.close()
            raise RuntimeError("MediaPipe is required for face/lip detection. Install requirements.txt.") from exc
        except Exception:
            if detector:
                detector.close()
            raise
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        if pose:
            pose.close()
        if detector:
            detector.close()
        raise FileNotFoundError(f"Cannot open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps
    cap.release()
    limit = duration if max_seconds <= 0 else min(duration, max_seconds)
    if progress:
        detector_name = ("minion-goggles-mouth" if main_group == "yellow_minions" else
                         "mediapipe-landmarks" if detector else "opencv-diagnostic")
        progress(f"Video scan: {duration / 60:.1f} min, {sample_fps:g} samples/s, sequential decoder, backend={detector_name}")
    started = last_report = time.perf_counter()
    samples = []
    backends = Counter()
    previous_gray = previous_det = None
    from autoedit.video.transitions import TransitionDetector
    transitions = TransitionDetector()
    from autoedit.video.tracking import FaceFeatureTracker
    tracker = FaceFeatureTracker()
    previous_sample_time = -1.
    def track_between_samples(t, decoded):
        if not detector or not tracker.faces:
            return
        width = 640
        if decoded.width > width:
            decoded = decoded.reformat(width=width, height=max(2,round(decoded.height*width/decoded.width)),format='bgr24')
        tracker.update(decoded.to_ndarray(format='bgr24'), t, [],
                       cut=any(tracker.last_time < event <= t for event in transitions.events))
    frames = sampled_frames(src, sample_fps, limit, transition_detector=transitions,
                            between_samples=track_between_samples)
    try:
        for t, frame in frames:
            det = detector.detect(frame, t) if detector else _detect_opencv(cv2, np, frame)
            if detector:
                faces = tracker.update(frame,t,det.get('faces',[]),
                    cut=any(previous_sample_time < event <= t for event in transitions.events))
                if faces:
                    subject = max(faces,key=lambda face:face['face'][2]*face['face'][3])
                    det = {**subject,'faces':faces,'face_candidate_count':len(faces)}
            previous_sample_time = t
            quality, previous_gray = _visual_quality(cv2, frame, det, previous_gray, previous_det)
            det.update(quality)
            det["frame_width"] = int(frame.shape[1])
            det["frame_height"] = int(frame.shape[0])
            for face in det.get("faces", []):
                face.update(_visual_quality(cv2, frame, face)[0])
            det.update(pose.detect(frame, t) if pose else {"person_detected": False, "pose_confirmed": False, "person_box": None, "shot_scale": None})
            det["person_visible"] = float(det["person_detected"])
            det["time_sec"] = t
            samples.append(det)
            previous_det = det
            backends[det["backend"]] += 1
            now = time.perf_counter()
            if progress and now - last_report >= 3:
                elapsed = now - started
                eta = elapsed * max(0, limit - t) / max(t, 0.1)
                faces = sum(bool(s.get("face_detected")) for s in samples)
                progress(f"Video {100 * t / max(limit, 1):.1f}% | source {t:.0f}/{limit:.0f}s | elapsed {elapsed:.0f}s | ETA {eta:.0f}s | samples={len(samples)} faces={faces}")
                last_report = now
    finally:
        frames.close()
        if pose:
            pose.close()
        if detector:
            detector.close()
    annotate_speaking(samples, source_kind=source_kind)
    elapsed = time.perf_counter() - started
    actual_end = min(limit, samples[-1]["time_sec"] + 1 / sample_fps) if samples else 0
    if actual_end + 1 / sample_fps < limit:
        raise RuntimeError(f"Video decode ended early at {actual_end:.1f}/{limit:.1f}s; incomplete analysis was not cached.")
    if progress:
        progress(f"Video done in {elapsed:.1f}s: clear-face samples={sum(s['clear_face'] for s in samples):.0f}, speaking samples={sum(s['speaking'] for s in samples):.0f}")
    info = {"path": str(src), "duration_sec": duration, "fps": fps, "sample_fps": sample_fps,
            "analyzed_seconds": actual_end, "backend_counts": dict(backends), "sample_count": len(samples),
            "segments": bounded_segments(
                _merge_segments(samples, min_segment_sec, merge_gap_sec, sample_interval=1 / sample_fps),
                actual_end, min_segment_sec),
            "samples": samples, "source_kind": source_kind, "quality_method": "multiregion-verified-face-tracks-v9",
            "analysis_elapsed_sec": round(elapsed, 3), "cache_identity": identity,
            "transition_times_sec": sorted(set(transitions.events)), "transition_method": "motion-verified-shot-local-v4",
            "shot_times_sec": sorted(set(transitions.shot_events)), "shot_method": "adjacent-frame-histogram-v1"}
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_bytes(gzip.compress(json.dumps(info).encode()))
        temporary.replace(cache)
    return info
