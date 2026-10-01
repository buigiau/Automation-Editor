from collections import Counter

import pytest

from autoedit.audio.pronunciation import VoiceIndex, build_sound_anchors, item_pronunciations
from autoedit.audio.taxonomy import TAXONOMY
from autoedit.audio.scene_match import choose_audio, build_sampler_slots
from autoedit.match.matcher import match_slots_to_video
from test_audio_taxonomy import recording
from test_shot_tiers import sample


def word(text, start=2.13, end=2.5, prob=.95):
    return {"word": text, "start": start, "end": end, "prob": prob}


def test_every_curated_speech_filename_has_a_pronunciation():
    for stem, (_, sound) in TAXONOMY.items():
        if sound.action == "SPEECH":
            assert item_pronunciations({"stem": stem}), stem


@pytest.mark.parametrize("text,stems,expected", [
    ("know", ["no", "no2", "go"], {"no.wav", "no2.wav"}),
    ("I", ["a", "ai", "aii"], {"ai.wav", "aii.wav"}),
    ("my", ["mai", "mái", "my", "bye"], {"mai.wav", "mái.wav", "my.wav"}),
    ("row", ["rô", "no"], {"rô.wav"}),
    ("ah", ["a", "u"], {"a.wav"}),
    ("a", ["a", "ai"], set()),
    ("fine", ["fight", "five"], set()),
    ("I'm", ["ai"], set()),
])
def test_matching_uses_complete_pronunciation_and_vietnamese_labels(text, stems, expected):
    index = VoiceIndex([recording(stem) for stem in stems])
    assert {m["path"] for m in index.match(word(text))} == expected


def test_only_the_first_sound_is_used_even_when_later_words_match_better():
    pool = [recording(s) for s in ("ai", "no", "hi")]
    words = [word("I", 2, 2.2), word("don't", 2.2, 2.5), word("know", 2.5, 2.9)]
    speech = {"source_speech": {"words": words}}
    chosen, evidence = choose_audio(pool, {"in_sec": 2, "out_sec": 3}, speech, Counter())
    assert chosen["path"] == "ai.wav"
    assert evidence["source_word"] == "I"
    chosen, evidence = choose_audio(pool, {"in_sec": 2.5, "out_sec": 3}, speech, Counter())
    assert chosen["path"] == "no.wav"
    assert evidence["source_word"] == "know"
    assert chosen["placement_offset_sec"] == 0


@pytest.mark.parametrize("first", [word("don't", 2, 2.2), word("I", 2, 2.2, .2), word("I", 1.9, 2.2)])
def test_unsupported_weak_or_midword_opening_blocks_later_supported_word(first):
    pool = [recording("ai"), recording("no"), recording("hi")]
    _, evidence = choose_audio(pool, {"in_sec": 2, "out_sec": 3},
                              {"source_speech": {"words": [first, word("know", 2.21, 2.6)]}}, Counter())
    assert not evidence.get("cut_onset_matched")


def video_with(speech, pool, duration=12):
    return {"duration_sec": duration, "sample_fps": 2,
            "samples": [sample(i/2, 1) for i in range(duration*2)],
            "source_speech": speech, "sound_anchors": build_sound_anchors(pool, speech)}


def test_library_matched_onset_beats_better_visual_cut_and_survives_sampler_selection():
    pool = [recording("no"), recording("no2"), recording("hi")]
    speech = {"words": [word("know", 2.13, 2.5), word("hi", 2.5, 2.9, .7)]}
    video = video_with(speech, pool)
    # A readable, valid tier 2 near the match vs tier 1 elsewhere.
    for s in video["samples"]:
        if 1.5 <= s["time_sec"] <= 4.5:
            s.update(sample(s["time_sec"], 2))
    match = match_slots_to_video([{"id": "scene", "required_duration_sec": 2}], [], video)[0]
    cut = match["video"]
    assert cut["in_sec"] == 2.13
    assert cut["out_sec"] == pytest.approx(4.13)
    assert match["status"] == "sound_matched"
    # Even replacement ASR/denser lips cannot change a selected onset to hi.
    chosen, evidence = choose_audio(pool, cut, {}, Counter({"no.wav": 10}))
    assert chosen["path"] == "no2.wav"
    assert evidence["source_word"] == "know"
    assert evidence["method"] == "source-audio-pronunciation"
    assert chosen["placement_offset_sec"] == 0


def test_insufficient_sound_matches_use_visual_fallback_and_preserve_lengths_and_gap():
    pool = [recording("ai")]
    video = video_with({"words": [word("I", 2.13, 2.5)]}, pool)
    matches = match_slots_to_video([{"id": str(i), "required_duration_sec": 2} for i in range(3)],
                                  [], video, min_gap_sec=1)
    assert sum(m["status"] == "sound_matched" for m in matches) == 1
    cuts = sorted((m["video"]["in_sec"], m["video"]["out_sec"]) for m in matches)
    assert all(b-a == pytest.approx(2) for a, b in cuts)
    assert all(right[0]-left[1] >= 1-1e-8 for left, right in zip(cuts, cuts[1:]))


def test_weak_long_or_transition_unsafe_sound_is_not_shifted_to_fit():
    pool = [recording("no")]
    speech = {"words": [word("know", 2.13, 2.5, .3), word("know", 10.5, 11.5)]}
    video = video_with(speech, pool)
    matches = match_slots_to_video([{"id": "scene", "required_duration_sec": 3}], [], video)
    assert matches[0]["status"] != "sound_matched"  # 10.5+3 is past source end.
    video = video_with({"words": [word("know")]}, pool)
    video["transition_times_sec"] = [2.1]
    cut = match_slots_to_video([{"id": "scene", "required_duration_sec": 2}], [], video)[0]
    assert not cut["video"].get("source_sound")
    assert cut["video"]["selection_metrics"]["transition_safe"]


def test_cough_event_matches_cough_recording_and_never_the_spoken_word_cough():
    pool = [recording("ho khụ khụ"), recording("hi")]
    event = {"action": "COUGH", "start": 2.13, "end": 2.7, "prob": .75}
    video = video_with({"words": [], "events": [event]}, pool)
    cut = match_slots_to_video([{"id": "scene", "required_duration_sec": 2}], [], video)[0]["video"]
    chosen, evidence = choose_audio(pool, cut, video, Counter())
    assert chosen["path"] == "ho khụ khụ.wav"
    assert evidence["method"] == "source-audio-event" and evidence["needs_review"]
    assert not VoiceIndex(pool).match(word("cough"))


def test_no_anchors_for_unknown_words_invalid_timestamps_or_duplicate_segments():
    pool = [recording("no"), dict(recording("no"), segment_id="expanded")]
    anchors = build_sound_anchors(pool, {"words": [word("know"), word("unknown"),
                                                  word("know", float("nan"), 2.5)]})
    assert len(anchors) == 1 and len(anchors[0]["matches"]) == 1


def test_missing_anchored_sample_is_an_error_instead_of_matching_a_later_word():
    cut = {"in_sec": 2.13, "out_sec": 3.13,
           "source_sound": build_sound_anchors([recording("no")], {"words": [word("know")]})[0]}
    with pytest.raises(ValueError, match="missing from the Voice"):
        choose_audio([recording("hi")], cut, {}, Counter())


def test_repeated_scene_keeps_the_same_opening_sample_and_existing_midi():
    from test_sampler_scenes import fixture
    scenes, slots, inventory = fixture()
    pool = [recording("ai"), recording("aii"), recording("no"), recording("no2")]
    speech = {"words": [word("I", 0, .2), word("know", 10, 10.3)]}
    for slot, anchor in zip(slots, build_sound_anchors(pool, speech)):
        slot["video"]["source_sound"] = anchor
    chosen = build_sampler_slots(scenes, slots, pool, {}, inventory)
    assert chosen[0]["audio"]["path"] in {"ai.wav", "aii.wav"}
    assert chosen[1]["audio"]["path"] in {"no.wav", "no2.wav"}
    assert chosen[0]["audio"] == chosen[2]["audio"]
    assert all(s["audio"]["placement_offset_sec"] == 0 and s["cubase"]["preserve_existing_midi"] for s in chosen)
