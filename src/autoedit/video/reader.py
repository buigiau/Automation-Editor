"""Decode once, select by timestamps, resize only selected frames."""

def sampled_frames(path, sample_fps, max_seconds=0, width=640, transition_detector=None,
                   between_samples=None, start_sec=0):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        fps = float(stream.average_rate or 25)
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0
        next_time = start_sec
        if start_sec > 0:
            container.seek(int((start_sec + origin) / float(stream.time_base)), stream=stream, backward=True)
        for index, frame in enumerate(container.decode(stream)):
            if start_sec > 0 and frame.time is None:
                raise RuntimeError('Cannot resume video without frame timestamps')
            time = (float(frame.time) - origin) if frame.time is not None else index / fps
            if time + 1e-6 < start_sec:
                continue
            if max_seconds > 0 and time >= max_seconds:
                break
            if transition_detector is not None:
                thumbnail = frame.reformat(width=160, height=90, format="bgr24")
                transition_detector.observe(time, thumbnail.to_ndarray(format="bgr24"))
            if time + 1e-6 < next_time:
                if between_samples is not None:
                    between_samples(time, frame)
                continue
            if frame.width > width:
                frame = frame.reformat(width=width, height=max(2, round(frame.height * width / frame.width)), format="bgr24")
            yield time, frame.to_ndarray(format="bgr24")
            # Preserve actual VFR timestamps; never duplicate frames to meet a requested rate.
            next_time = (int((time + 1e-6) * sample_fps) + 1) / sample_fps
