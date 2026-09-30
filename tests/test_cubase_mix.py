import math
import wave
from array import array
from pathlib import Path

from autoedit.cubase.adapter import apply_cubase_plan
from autoedit.schema import empty_plan


def _tone(path: Path, seconds: float = 0.2, hz: float = 440.0, rate: int = 44100) -> None:
    n = int(seconds * rate)
    samples = array("h")
    for i in range(n):
        v = int(16000 * math.sin(2 * math.pi * hz * i / rate))
        samples.append(v)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())


def test_mixdown_places_clip_at_slot_time(tmp_path):
    wav = tmp_path / "a.wav"
    _tone(wav)
    plan = empty_plan()
    plan["slots"] = [
        {
            "id": "slot-01",
            "premiere": {"nested_sequence": "1", "duration_sec": 0.2},
            "cubase": {"track_name": "Sampler Track 01", "track_index": 1, "start_sec": 1.0},
            "audio": {"path": str(wav), "duration_sec": 0.2, "category": "A"},
            "video": {"in_sec": 2.0, "out_sec": 2.2, "category": "A"},
        }
    ]
    plan["mixdown"]["sample_rate"] = 48000
    result = apply_cubase_plan(plan, tmp_path)
    mix = Path(result["mixdown"]["path"])
    assert mix.exists()
    with wave.open(str(mix), "rb") as wf:
        assert wf.getnchannels() == 2
        assert wf.getframerate() == 48000
        n = wf.getnframes()
        raw = wf.readframes(n)
    # 16-bit stereo
    import numpy as np

    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    # Silence before 1.0s, energy after
    pre = pcm[: int(0.5 * 48000)]
    post = pcm[int(1.0 * 48000) : int(1.15 * 48000)]
    assert int(np.abs(pre).max()) < 50
    assert int(np.abs(post).max()) > 100
