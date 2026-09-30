from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from autoedit.video.silent_source import prepare_video_only_source


def source_with_audio(path, origin=0):
    with av.open(str(path), "w", options={"avoid_negative_ts": "disabled"}) as out:
        video = out.add_stream("libx264", rate=10)
        video.width = video.height = 64
        video.pix_fmt = "yuv420p"
        video.codec_context.max_b_frames = 2
        audio = out.add_stream("aac", rate=16000)
        audio.layout = "mono"
        for i in range(40):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i*5, np.uint8), format="rgb24")
            frame.pts = i+int(origin*10)
            frame.time_base = Fraction(1, 10)
            out.mux(video.encode(frame))
        out.mux(video.encode())
        frame = av.AudioFrame.from_ndarray(np.full((1, 80000), .2, np.float32), format="fltp", layout="mono")
        frame.sample_rate = 16000
        frame.time_base = Fraction(1, 16000)
        frame.pts = 0
        out.mux(audio.encode(frame))
        out.mux(audio.encode())
    return path


@pytest.mark.parametrize("origin", [0, 1])
def test_fill_source_has_no_audio_and_preserves_every_frame_and_timing(tmp_path, origin):
    source = source_with_audio(tmp_path / "source.mp4", origin)
    before = source.read_bytes()
    copied = Path(prepare_video_only_source(source, tmp_path / "fills"))
    with av.open(str(source)) as src, av.open(str(copied)) as dst:
        assert src.streams.audio
        assert len(dst.streams.video) == 1 and len(dst.streams.audio) == 0
        originals = list(src.decode(video=0))
        copies = list(dst.decode(video=0))
        assert len(originals) == len(copies) == 40
        for a, b in zip(originals, copies):
            assert float(b.time) == pytest.approx(float(a.time)-origin, abs=1e-6)
            assert np.array_equal(a.to_ndarray(), b.to_ndarray())
    assert source.read_bytes() == before
    copied_time = copied.stat().st_mtime_ns
    assert prepare_video_only_source(source, tmp_path / "fills") == str(copied)
    assert copied.stat().st_mtime_ns == copied_time


def test_modified_source_gets_a_new_fill_asset(tmp_path):
    import os
    source = source_with_audio(tmp_path / "source.mp4")
    first = prepare_video_only_source(source, tmp_path / "fills")
    os.utime(source, ns=(source.stat().st_atime_ns, source.stat().st_mtime_ns+1000000000))
    assert prepare_video_only_source(source, tmp_path / "fills") != first
