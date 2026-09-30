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
