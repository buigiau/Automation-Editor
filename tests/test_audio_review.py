from html.parser import HTMLParser
import numpy as np
import soundfile as sf
from autoedit.audio.review import write_audio_review


def test_review_has_source_audio_and_voice_for_each_track(tmp_path):
    source = tmp_path / "source.wav"
    sample = tmp_path / "sample.wav"
    sf.write(source, np.full(32000, .3), 16000)
    sf.write(sample, np.full(8000, .5), 16000)
    slot = {"premiere": {"nested_sequence_uid": "uid1", "nested_sequence": "1", "timeline_start_sec": 0,
                        "timeline_end_sec": 1}, "audio": {"path": str(sample)},
            "video": {"in_sec": .5, "out_sec": 1.5},
            "audio_match": {"source_word": "I<script>", "needs_review": True},
            "cubase": {"sample_path": str(sample), "track_name": "Sampler Track 01"}}
    plan = {"project": {"source_video": str(source)}, "cubase_slots": [slot, slot]}
    path = write_audio_review(plan, tmp_path, {"words": [{"word": "I<script>", "start": .6, "end": .8}]})
    html = path.read_text(encoding="utf-8")
    assert "I&lt;script&gt;" in html and "<script>" not in html
    assert len(list((tmp_path / "review_source_audio").glob("*.wav"))) == 1
    audio_links = []
    class Parser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "audio":
                audio_links.append(dict(attrs)["src"])
    Parser().feed(html)
    assert len(audio_links) == 4
    assert all((tmp_path / link).is_file() for link in audio_links)
    pcm, rate = sf.read(tmp_path / audio_links[0])
    assert len(pcm) == rate and abs(pcm.mean()-.3) < .001


def test_review_uses_each_video_soundtrack_and_its_own_transcript(tmp_path):
    sources = [tmp_path / "first.wav", tmp_path / "second.wav"]
    sample = tmp_path / "sample.wav"
    sf.write(sample, np.full(8000, .5), 16000)
    for path, amplitude in zip(sources, (.2, .7)):
        sf.write(path, np.full(32000, amplitude), 16000)
    slots = [{"premiere": {"nested_sequence_uid": f"uid{i}", "nested_sequence": str(i),
                           "timeline_start_sec": i, "timeline_end_sec": i+1},
              "audio": {"path": str(sample)},
              "video": {"source_video": str(source), "in_sec": .5, "out_sec": 1.5},
              "audio_match": {"needs_review": True},
              "cubase": {"sample_path": str(sample), "track_name": f"Sampler Track {i}"}}
             for i, source in enumerate(sources)]
    speech = {"sources": {str(source): {"words": [{"word": word, "start": .6, "end": .8}]}
                          for source, word in zip(sources, ("first-word", "second-word"))}}
    plan = {"project": {"source_video": str(sources[0])}, "cubase_slots": slots}
    review = write_audio_review(plan, tmp_path, speech).read_text(encoding="utf-8")
    assert "first-word" in review and "second-word" in review
    for index, amplitude in enumerate((.2, .7), 1):
        pcm, rate = sf.read(tmp_path / "review_source_audio" / f"scene_{index:02d}.wav")
        assert len(pcm) == rate
        assert abs(pcm.mean()-amplitude) < .001
