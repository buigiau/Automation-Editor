from collections import Counter
from pathlib import Path
import unicodedata

import numpy as np
import pytest
import soundfile as sf

from autoedit.audio.analyzer import analyze_file
from autoedit.audio.scene_match import choose_audio
from autoedit.audio.taxonomy import GROUPS, TAXONOMY, filename_label, lookup
from autoedit.audio.phonetics import filename_phonetics


def recording(stem, length=.4):
    return {"path": stem + ".wav", "category": filename_label(stem), "duration_sec": length,
            "phonetics": filename_phonetics(stem)}


def test_inventory_is_complete_unique_and_accent_sensitive():
    assert len(TAXONOMY) == 127 == sum(map(len, GROUPS.values()))
    folder = Path(r"D:\Editor\Voice")
    if folder.exists():
        assert {p.stem for p in folder.iterdir() if p.suffix.lower() == ".wav"} == set(TAXONOMY)
    assert lookup("oi")[0] == "EMOTION"
    assert lookup("ôi")[0] == "NEUTRAL"
    assert lookup(unicodedata.normalize("NFD", "CƯỜI ĐÀN ÔNG"))[1].label == "LAUGH_MAN"
    for stem in ("ai", "aii", "hi", "bye", "gru", "rô", "ó", "ôi", "ê", "ựa"):
        assert lookup(stem)[0] == "NEUTRAL"


@pytest.mark.parametrize("stem,label", [("what", "WHAT"), ("why", "WHY"), ("woa", "WOA"),
    ("yeah", "YEAH"), ("hi", "HI"), ("no", "NO"), ("fight", "FIGHT"), ("love", "LOVE"),
    ("cười trẻ con", "LAUGH_KID"), ("cười đàn ông", "LAUGH_MAN"),
    ("còi oto", "HORN"), ("ting", "CHIME"), ("tinggg", "BELL"), ("beepp", "BEEP")])
def test_word_labels_survive_waveform_analysis(tmp_path, stem, label):
    path = tmp_path / (stem + ".wav")
    sf.write(path, .5*np.sin(np.arange(6400)*.2), 16000)
    info = analyze_file(path)
    assert info["category"] == label
    assert info["segments"][0]["category"] == label
    assert info["group"] == lookup(stem)[0]


@pytest.mark.parametrize("kind", ["animation", "live_action"])
@pytest.mark.parametrize("tier", [1, 4])
def test_source_word_beats_visemes_and_neutral_policy(kind, tier):
    info = {"source_kind": kind, "samples": [{"time_sec": 10.3, "clear_face": 1,
             "category": "A", "lip_aperture": .2}], "source_speech": {"model": "small", "words": [
             {"word": "What?", "start": 10.2, "end": 10.6, "prob": .95}]}}
    chosen, evidence = choose_audio([recording("a"), recording("hi"), recording("what")],
        {"in_sec": 10, "out_sec": 11, "selection_metrics": {"tier": tier}}, info, Counter())
    assert chosen["path"] == "what.wav"
    assert chosen["placement_offset_sec"] == pytest.approx(.2)
    assert evidence["source_word_start_sec"] == 10.2
    assert evidence["method"] == "source-audio-whisper-exact-word"


@pytest.mark.parametrize("word,start,end,prob", [("What", 10.2, 10.6, .3),
    ("What", 9.9, 10.3, .99), ("What", 10.8, 11.1, .99), ("whatever", 10.2, 10.6, .99)])
def test_no_partial_weak_or_prefix_word_matches(word, start, end, prob):
    info = {"source_speech": {"words": [{"word": word, "start": start, "end": end, "prob": prob}]}}
    chosen, evidence = choose_audio([recording("what"), recording("hi")],
        {"in_sec": 10, "out_sec": 11}, info, Counter())
    assert chosen["path"] == "hi.wav"
    assert evidence["method"] == "neutral-no-readable-viseme"


def test_unreadable_fallback_cycles_distinct_neutral_files_only():
    pool = [recording(s) for s in ("what", "a", "beepp", "hi", "love", "bye")]
    pool += [dict(pool[-1], segment_id="duplicate")]
    used, paths = Counter(), []
    for _ in range(6):
        chosen, evidence = choose_audio(pool, {"in_sec": 0, "out_sec": 1}, {}, used)
        paths.append(chosen["path"])
        used[chosen["path"]] += 1
        assert evidence["method"] == "neutral-no-readable-viseme"
    assert paths == ["hi.wav", "love.wav", "bye.wav"] * 2


def test_readable_face_keeps_viseme_matching_when_no_source_word():
    chosen, evidence = choose_audio([recording("a"), recording("u")], {"in_sec": 0, "out_sec": .4},
        {"samples": [{"time_sec": .1, "clear_face": 1, "speaking": 1,
                      "category": "A", "lip_aperture": .2}]}, Counter())
    assert chosen["path"] == "a.wav"
    assert evidence["method"] == "filename-visemes-and-dense-lip-rhythm"


def test_vowel_categories_and_expression_actions_keep_their_visual_meaning():
    from autoedit.audio.scene_match import _shape
    for category in ("A", "E", "O", "U", "LAUGH", "COUGH"):
        assert _shape(category) == category


@pytest.mark.parametrize("word", ["I", "I!", "eye"])
def test_english_i_matches_vietnamese_ai_even_when_reused(word):
    pool = [recording("rô"), recording("ai"), recording("aii")]
    info = {"source_speech": {"words": [{"word": word, "start": 10.2, "end": 10.6, "prob": .95}]}}
    chosen, evidence = choose_audio(pool, {"in_sec": 10, "out_sec": 11}, info,
                                   Counter({"ai.wav": 50, "aii.wav": 50}), recent=pool[1:])
    assert chosen["category"] == "AI"
    assert evidence["method"] == "source-audio-whisper-exact-word"


def test_visual_fallback_avoids_recent_word_variants_and_marks_review():
    pool = [recording(s) for s in ("ai", "aii", "hi", "bye")]
    info = {"samples": [{"time_sec": .1, "clear_face": 1, "speaking": 1,
                         "category": "A", "lip_aperture": .2}]}
    chosen, evidence = choose_audio(pool, {"in_sec": 0, "out_sec": .4}, info,
                                   Counter(), recent=pool[:1])
    assert chosen["category"] != "AI"
    assert evidence["needs_review"]


def test_contraction_is_not_falsely_treated_as_a_complete_i():
    info = {"source_speech": {"words": [{"word": "I'm", "start": .1, "end": .3, "prob": .99}]}}
    _, evidence = choose_audio([recording("ai")], {"in_sec": 0, "out_sec": 1}, info, Counter())
    assert evidence["method"] != "source-audio-whisper-exact-word"
