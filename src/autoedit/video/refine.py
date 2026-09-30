"""Dense lip samples only for already selected cuts; no long-source rescan."""
from pathlib import Path
import hashlib
import json
from bisect import bisect_right


def refine_selected_cuts(path, video_slots, cache_dir, progress=None, sample_fps=20, source_kind="live_action",
                         character_samples=None):
    import av
    from autoedit.video.faces import FaceLandmarker, MultiRegionFaceLandmarker, MODEL_PATH, annotate_speaking
    from autoedit.video.analyzer import _visual_quality, _try_cv2
    source = Path(path).resolve()
    ranges = sorted({(s["video"]["in_sec"], s["video"]["out_sec"]) for s in video_slots})
    references = sorted([s for s in (character_samples or [])
                         if any(start - .5 <= s["time_sec"] <= end + .5 for start, end in ranges)],
                        key=lambda s: s["time_sec"])
    reference_times = [s["time_sec"] for s in references]
    reference_identity = [{k: s.get(k) for k in ("time_sec", "face", "frame_width", "frame_height",
                                                "character_id", "track_id", "shot_id")}
                          for s in references]
    identity = [str(source), source.stat().st_size, source.stat().st_mtime_ns, ranges, sample_fps,
                source_kind, MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else None, reference_identity, 3]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    dest = Path(cache_dir) / ("lips-" + key + ".json")
    if dest.is_file():
        return json.loads(dest.read_text(encoding="utf-8"))
    if progress:
        progress(f"Refining lips at {sample_fps} samples/s for {len(ranges)} selected cuts only")
    cv2, _ = _try_cv2()
    detector = MultiRegionFaceLandmarker() if source_kind == "animation" else FaceLandmarker()
    samples = []
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            stream.codec_context.thread_count = 4
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0
            for start, end in ranges:
                container.seek(int((start + origin) / float(stream.time_base)), stream=stream, backward=True)
                next_time = start
                for frame in container.decode(stream):
                    if frame.time is None:
                        continue
                    t = float(frame.time) - origin
                    if t >= end:
                        break
                    if t + 1e-6 < next_time:
                        continue
                    frame = frame.reformat(width=960, height=round(frame.height * 960/frame.width), format="bgr24")
                    image = frame.to_ndarray(format="bgr24")
                    det = detector.detect(image, t)
                    if character_samples is not None:
                        det = select_reference_face(det.get("faces", []), image.shape[1], image.shape[0],
                                                    t, references, reference_times)
                    quality, _ = _visual_quality(cv2, image, det)
                    det.update(quality)
                    det["time_sec"] = t
                    samples.append(det)
                    next_time += 1/sample_fps
    finally:
        detector.close()
    annotate_speaking(samples, source_kind=source_kind)
    info = {"samples": samples, "sample_fps": sample_fps, "method": "selected-lips-20fps",
            "ranges": ranges, "path": str(source)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(info), encoding="utf-8")
    return info


def select_reference_face(faces, width, height, time, references, reference_times):
    """Keep dense lip analysis on the subject chosen in the source scan."""
    from autoedit.video.characters import iou
    missing = {"backend": "mediapipe-landmarks", "face_detected": False,
               "category": "NEUTRAL", "confidence": 0.0}
    index = bisect_right(reference_times, time) - 1
    choices = [i for i in (index, index + 1) if 0 <= i < len(references)]
    if not choices or not faces:
        return missing
    reference = references[min(choices, key=lambda i: abs(reference_times[i] - time))]
    if abs(reference["time_sec"] - time) > .5 or not reference.get("face"):
        return missing
    x, y, w, h = reference["face"]
    sx, sy = width / reference["frame_width"], height / reference["frame_height"]
    expected = [x * sx, y * sy, w * sx, h * sy]
    ordered = sorted(faces, key=lambda face: iou(face["face"], expected), reverse=True)
    selected = ordered[0]
    if iou(selected["face"], expected) < .25:
        return missing
    if len(ordered) > 1 and iou(selected["face"], expected) - iou(ordered[1]["face"], expected) < .1:
        return missing
    return {**selected, **{k: reference.get(k) for k in ("character_id", "track_id", "shot_id")}}
