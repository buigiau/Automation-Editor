from types import SimpleNamespace as NS

import av
import numpy as np
import pytest
import soundfile as sf

from autoedit.audio import source_speech as speech


def test_decode_preserves_source_offset_and_resamples_stereo(tmp_path):
    path = tmp_path / "source.wav"
    rate = 48000
    pcm = np.zeros((rate*5, 2), dtype=np.float32)
    pcm[int(2.75*rate):int(3.25*rate)] = .5
    sf.write(path, pcm, rate)
    with av.open(str(path)) as container:
        result = speech._decode_window(container, container.streams.audio[0], 1.5, 1, 2)
    assert len(result) == 16000
    assert np.abs(result[:3900]).max() < .001
    # FFmpeg's stereo-to-mono matrix uses equal-power channel coefficients.
    assert result[4200:11800].mean() == pytest.approx(.5 * np.sqrt(2), abs=.001)
    assert np.abs(result[12100:]).max() < .001


def test_word_timestamps_and_cache_invalidate_by_range_settings_and_source(tmp_path, monkeypatch):
    path = tmp_path / "source.wav"
    sf.write(path, .4*np.sin(np.arange(160000)*.2), 16000)
    calls = []

    class FakeModel:
        def transcribe(self, samples, **kwargs):
            calls.append(kwargs)
            assert len(samples) == 3*16000
            good = NS(text="What?", no_speech_prob=.01, words=[
                NS(word="What?", start=1.25, end=1.5, probability=.97)])
            hallucinated = NS(text="Thanks for watching!", no_speech_prob=.01, words=good.words)
            silence = NS(text="What?", no_speech_prob=.95, words=good.words)
            return iter([good, hallucinated, silence]), NS()

    monkeypatch.setattr(speech, "_model", lambda name: FakeModel())
    first = speech.transcribe_source(path, [(5, 6)], tmp_path / "cache")
    assert first["has_audio"]
    assert first["words"] == [{"word": "What?", "start": 5.25, "end": 5.5, "prob": .97}]
    assert calls[0]["word_timestamps"] and calls[0]["vad_filter"]
    assert speech.transcribe_source(path, [(5, 6)], tmp_path / "cache") == first
    assert len(calls) == 1
    speech.transcribe_source(path, [(6, 7)], tmp_path / "cache")
    speech.transcribe_source(path, [(6, 7)], tmp_path / "cache", language=None)
    speech.transcribe_source(path, [(6, 7)], tmp_path / "cache", language=None, model="medium")
    import os
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns+1000000000))
    speech.transcribe_source(path, [(6, 7)], tmp_path / "cache", language=None, model="medium")
    assert len(calls) == 5


def test_silent_track_and_video_without_audio_do_not_load_whisper(tmp_path, monkeypatch):
    from test_talking_faces import make_video

    def unexpected(*args):
        raise AssertionError("No speech audio: should not load Whisper")

    monkeypatch.setattr(speech, "_model", unexpected)
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(32000), 16000)
    silent = speech.transcribe_source(path, [(0, 1)], tmp_path / "cache")
    assert silent["status"] == "no-speech" and silent["has_audio"]
    video = speech.transcribe_source(make_video(tmp_path), [(0, 1)], tmp_path / "cache")
    assert video["status"] == "no-audio" and not video["has_audio"]
    assert silent["words"] == video["words"] == []


def test_broken_asr_is_not_cached_as_a_silent_source(tmp_path, monkeypatch):
    path = tmp_path / "source.wav"
    sf.write(path, np.full(32000, .2), 16000)

    def broken(name):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(speech, "_model", broken)
    with pytest.raises(RuntimeError, match="model unavailable"):
        speech.transcribe_source(path, [(0, 1)], tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.json"))
