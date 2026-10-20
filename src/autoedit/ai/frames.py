"""Decode ordered cut samples with one seek, including native-frame timestamps."""
from autoedit.runtime import check_cancelled


def cut_frames(path, times, width):
    import av
    times = list(times)
    if not times or times != sorted(times):
        raise ValueError('Cut sample times must be ordered.')
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        origin = float((stream.start_time or 0)*stream.time_base)
        container.seek(int((times[0]+origin)/float(stream.time_base)), stream=stream, backward=True)
        index = 0
        for frame in container.decode(stream):
            check_cancelled()
            if frame.time is None or frame.time-origin < times[index]-.001:
                continue
            size = min(width, frame.width)
            rgb = frame.reformat(width=size, height=max(2, round(frame.height*size/frame.width)),
                                 format='rgb24').to_ndarray()
            while index < len(times) and frame.time-origin >= times[index]-.001:
                yield frame.time-origin, rgb
                index += 1
            if index == len(times):
                return
    raise ValueError('Cannot decode all requested cut frames.')
