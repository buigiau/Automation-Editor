"""Independent word, phrase and cut-boundary cases using the full inventory."""
from collections import Counter
import unicodedata

import pytest

from autoedit.audio.pronunciation import VoiceIndex, build_sound_anchors
from autoedit.audio.scene_match import choose_audio
from autoedit.audio.taxonomy import TAXONOMY
from autoedit.match.matcher import match_slots_to_video
from test_audio_taxonomy import recording
from test_shot_tiers import sample


@pytest.fixture(scope="module")
def library():
    return [recording(stem) for stem in TAXONOMY]


@pytest.mark.parametrize("text,stems", [
    ("buy", {"bye"}), ("by", {"bye"}), ("high", {"hi"}), ("weight", {"wait"}),
    ("too", {"two"}), ("two", {"two"}), ("ewe", {"iu", "you"}), ("yew", {"iu", "you"}),
    ("pea", {"pii"}), ("pee", {"pii"}), ("gee", {"gi", "giii"}), ("damn", {"dam", "damm"}),
    ("em", {"em"}), ("roe", {"rô"}), ("my", {"my", "mai", "mái"}),
    ("ohhhhh!", {"oh", "ohh", "ohhh"}), ("ahhh", {"a", "aa"}), ("ewwww", {"eww"}),
    ("heyyyy", {"hey to", "heyy", "heyyy"}), ("yesss", {"yes", "yess"}),
    ("whaaat?", {"what", "whatt"}), ("why?", {"why"}), ("whooo", {"who", "whu", "huu"}),
    ("noooo", {"no", "no2"}), ("wow!", {"woww", "wowww"}),
    ("yeah!", {"yeah", "yeahh", "yeah trầm"}), ("yay", {"yay", "yaayy"}),
    ("3", {"three"}), ("5", {"five"}), ("10", {"ten"}), ("okay", {"ok"}),
    ("“Hi!”", {"hi"}),
    ("fine", set()), ("four", set()), ("eight", set()), ("fighting", set()),
    ("fighter", set()), ("whatever", set()), ("where", set()), ("buying", set()),
    ("height", set()), ("weighted", set()), ("noisy", set()), ("I'm", set()),
    ("I’m", set()), ("I've", set()), ("cough", set()), ("laugh", set()), ("sigh", set()),
    ("horn", set()), ("beep", set()),
    # Without an acoustic pronunciation, reduced/ambiguous readings abstain.
    ("to", set()), ("a", set()), ("won", set()), ("woah", set()), ("whoa", set()),
])
def test_full_inventory_matches_other_words_without_prefix_or_ambiguous_guesses(library, text, stems):
    result = VoiceIndex(library).match({"word": text})
    assert {m["path"] for m in result} == {s+".wav" for s in stems}


@pytest.mark.parametrize("sentence,index,expected", [
    (["You", "should", "wait"], 0, {"you", "iu"}),
    (["You", "should", "wait"], 1, set()),
    (["You", "should", "wait"], 2, {"wait"}),
    (["My", "eye", "hurts"], 0, {"my", "mai", "mái"}),
    (["My", "eye", "hurts"], 1, {"ai", "aii", "ái"}),
    (["My", "eye", "hurts"], 2, set()),
    (["Fine", "fight", "fighting"], 0, set()),
    (["Fine", "fight", "fighting"], 1, {"fight"}),
    (["Fine", "fight", "fighting"], 2, set()),
    (["Too", "high", "here"], 0, {"two"}),
    (["Too", "high", "here"], 1, {"hi"}),
])
def test_different_cut_starts_use_only_the_opening_word(library, sentence, index, expected):
    words = [{"word": text, "start": 1+i*.5, "end": 1.4+i*.5, "prob": .96}
             for i, text in enumerate(sentence)]
    chosen, evidence = choose_audio(library, {"in_sec": words[index]["start"], "out_sec": 4},
                                   {"source_speech": {"words": words}}, Counter())
    if expected:
        assert chosen["path"] in {s+".wav" for s in expected}
        assert evidence["source_word"] == sentence[index]
        assert chosen["placement_offset_sec"] == 0
    else:
        assert not evidence.get("cut_onset_matched")


@pytest.mark.parametrize("word,paths", [("buy", {"bye.wav"}), ("weight", {"wait.wav"}),
    ("pea", {"pii.wav"}), ("gee", {"gi.wav", "giii.wav"}), ("ohhhh", {"oh.wav", "ohh.wav", "ohhh.wav"})])
@pytest.mark.parametrize("onset", [0.0, .073, 4.217])
def test_other_words_supply_exact_cut_onsets_even_between_video_samples(library, word, paths, onset):
    speech = {"words": [{"word": word, "start": onset, "end": onset+.3, "prob": .98}]}
    video = {"duration_sec": 10, "sample_fps": 2, "samples": [sample(i/2, 1) for i in range(20)],
             "sound_anchors": build_sound_anchors(library, speech)}
    match = match_slots_to_video([{"id": "scene", "required_duration_sec": 1.5}], [], video)[0]
    assert match["video"]["in_sec"] == onset
    assert match["video"]["out_sec"] == pytest.approx(onset+1.5)
    chosen, evidence = choose_audio(library, match["video"], {}, Counter())
    assert chosen["path"] in paths
    assert evidence["source_word"] == word and chosen["placement_offset_sec"] == 0


@pytest.mark.parametrize("action,expected", [("LAUGH", {"cười", "cười trẻ con", "cười đàn ông", "ha", "hah", "heha", "heheha"}),
                                           ("SIGH", {"thở dài", "thở dài 2"}),
                                           ("GROAN", {"gru", "grunt", "uuugh"})])
def test_event_classes_have_other_correct_recordings(library, action, expected):
    matches = VoiceIndex(library).match({"action": action})
    assert {m["path"] for m in matches} == {s+".wav" for s in expected}


def test_unicode_filename_forms_keep_the_same_pronunciation_match():
    item = recording("mái")
    item["stem"] = unicodedata.normalize("NFD", "MÁI")
    assert VoiceIndex([item]).match({"word": "MY!"})[0]["path"] == "mái.wav"


def test_confident_onset_beats_medium_confidence_then_medium_beats_visual_fallback(library):
    speech = {"words": [{"word": "wait", "start": 1.13, "end": 1.5, "prob": .65},
                        {"word": "my", "start": 7.13, "end": 7.5, "prob": .95}]}
    video = {"duration_sec": 12, "sample_fps": 2, "samples": [sample(i/2, 1) for i in range(24)],
             "sound_anchors": build_sound_anchors(library, speech)}
    slot = [{"id": "scene", "required_duration_sec": 1.5}]
    cut = match_slots_to_video(slot, [], video)[0]["video"]
    assert cut["in_sec"] == 7.13
    video["sound_anchors"] = video["sound_anchors"][:1]
    cut = match_slots_to_video(slot, [], video)[0]["video"]
    assert cut["in_sec"] == 1.13
    chosen, evidence = choose_audio(library, cut, video, Counter())
    assert chosen["path"] == "wait.wav" and evidence["needs_review"]


def test_ambiguous_opening_is_not_replaced_with_a_later_matched_word(library):
    speech = {"source_speech": {"words": [{"word": "to", "start": 0, "end": .2, "prob": .99},
                                           {"word": "wait", "start": .2, "end": .5, "prob": .99}]}}
    _, evidence = choose_audio(library, {"in_sec": 0, "out_sec": 1}, speech, Counter())
    assert not evidence.get("cut_onset_matched")
