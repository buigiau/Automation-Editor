"""Copy the video stream for Premiere fills without exposing timeline audio.

Whisper still reads the original source. The separate fill asset contains no
audio stream, so a host's interpretation of audioTrackIndex=-1 cannot overwrite
the configured beat. Encoded video packets are copied without re-encoding.
"""
import hashlib
import json
from pathlib import Path
import time


def prepare_video_only_source(path, output_dir, progress=None):
    import av

    src = Path(path).resolve()
    stat = src.stat()
    identity = {"path": str(src), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "version": 1}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(output_dir)
    dest, metadata = root / (key + "-video-only.mov"), root / (key + ".json")
    if dest.is_file() and metadata.is_file():
        try:
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            if saved["identity"] == identity and saved["size"] == dest.stat().st_size:
                with av.open(str(dest)) as check:
                    if len(check.streams.video) == 1 and not check.streams.audio:
                        if progress:
                            progress("Reusing video-only fill source; original soundtrack remains available to Whisper.")
                        return str(dest.resolve())
        except (ValueError, OSError, KeyError):
            pass
    root.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(".tmp")
    if progress:
        progress("Preparing video-only fill source to preserve configured timeline audio (stream copy).")
    try:
        with av.open(str(src)) as source:
            if not source.streams.video:
                raise ValueError("Source has no video stream")
            stream = source.streams.video[0]
            origin = stream.start_time or 0
            time_base = stream.time_base
            source_duration = float(stream.duration * time_base) if stream.duration else None
            with av.open(str(temp), "w", format="mov", options={"avoid_negative_ts": "disabled"}) as target:
                video = target.add_stream_from_template(stream)
                video.metadata.update(stream.metadata)
                last_report = time.monotonic()
                for packet in source.demux(stream):
                    if packet.size == 0:
                        continue
                    # Match the zero-based source times used by the video scanner.
                    if packet.pts is not None:
                        packet.pts -= origin
                    if packet.dts is not None:
                        packet.dts -= origin
                    position = float(packet.pts * time_base) if packet.pts is not None else 0
                    packet.stream = video
                    target.mux(packet)
                    if progress and time.monotonic() - last_report >= 3:
                        progress(f"Video-only copy: {position:.1f}s copied")
                        last_report = time.monotonic()
        with av.open(str(temp)) as check:
            if len(check.streams.video) != 1 or check.streams.audio:
                raise ValueError("Fill source must contain video only")
            stream = check.streams.video[0]
            copied_start = float((stream.start_time or 0) * stream.time_base)
            copied_duration = float(stream.duration * stream.time_base) if stream.duration else None
            tolerance = 1 / float(stream.average_rate or 25) + .001
            if abs(copied_start) > tolerance or (source_duration is not None and
                    (copied_duration is None or abs(copied_duration-source_duration) > tolerance)):
                raise ValueError("Video-only copy changed source timing; no plan was generated")
        temp.replace(dest)
        metadata.write_text(json.dumps({"identity": identity, "size": dest.stat().st_size}), encoding="utf-8")
    finally:
        temp.unlink(missing_ok=True)
    return str(dest.resolve())
