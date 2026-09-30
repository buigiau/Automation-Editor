from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autoedit.audio.scene_match import build_sampler_slots
from autoedit.cubase.inspect import inspect_sampler_tracks
from autoedit.cubase.sampler import render_sampler_bundle
from autoedit.cubase.sampler import verify_sampler_inputs
import json
from autoedit.premiere.prproj import scene_run_slots, inspect_prproj


def fragment(uid, start, offset=0):
    return {"nested_sequence_uid": uid, "nested_sequence": uid, "start_sec": start,
            "end_sec": start + 0.5, "source_in_sec": offset, "source_out_sec": offset + 0.5}


def fixture():
    scenes = scene_run_slots([{"index": 1, "items": [fragment("A", 0), fragment("A", .5, .5),
                                                    fragment("B", 1), fragment("A", 1.5)]}])
    slots = [{"premiere": {"nested_sequence_uid": uid, "duration_sec": 1},
              "video": {"in_sec": i*10, "out_sec": i*10+1, "category": uid}}
             for i, uid in enumerate(("A", "B"))]
    slots[1]["video"]["category"] = "O"
    tracks = {"path": "template.cpr", "tracks": [{"index": i, "name": f"Sampler Track {i:02d}"}
                                                  for i in range(1, 4)]}
    return scenes, slots, tracks


def test_reprises_map_to_new_tracks_with_same_source_sample():
    scenes, slots, inventory = fixture()
    # No readable lips: rotate the neutral pool; a reprise retains its sample.
    audio = [{"path": "hi.wav", "category": "HI", "duration_sec": 1},
             {"path": "love.wav", "category": "LOVE", "duration_sec": 1}]
    result = build_sampler_slots(scenes, slots, audio, {}, inventory)
    assert [s["audio"]["path"] for s in result] == ["hi.wav", "love.wav", "hi.wav"]
    assert [s["cubase"]["track_index"] for s in result] == [1, 2, 3]
    assert [len(s["premiere"]["instances"]) for s in result] == [2, 1, 1]
    assert result[0]["premiere"]["instances"][1]["source_in_sec"] == .5


def test_track_count_mismatch_never_wraps_or_truncates():
    scenes, slots, inventory = fixture()
    inventory["tracks"].pop()
    with pytest.raises(ValueError, match="no wraparound"):
        build_sampler_slots(scenes, slots, [], {}, inventory)


def test_gap_and_uid_change_create_new_scene_even_with_same_display_name():
    a, b, c = fragment("uid1", 0), fragment("uid2", .5), fragment("uid2", 2)
    for item in (a, b, c):
        item["nested_sequence"] = "1"
    assert len(scene_run_slots([{"index": 1, "items": [c, b, a]}])) == 3


def test_sampler_wav_is_local_and_preview_repeats_source_offsets(tmp_path):
    source = tmp_path / "hi.wav"
    sf.write(source, np.r_[np.full(4000, .2), np.full(4000, .6)], 8000)
    scenes, slots, inventory = fixture()
    for scene in scenes:
        scene["first_start_sec"] += 2
        scene["last_end_sec"] += 2
        for instance in scene["instances"]:
            instance["start_sec"] += 2
            instance["end_sec"] += 2
    candidates = [{"path": str(source), "category": "A", "duration_sec": 1}]
    plan = {"project": {"cubase_project": "test.cpr"}, "mixdown": {"sample_rate": 8000},
            "cubase_slots": build_sampler_slots(scenes, slots, candidates, {}, inventory)}
    render_sampler_bundle(plan, tmp_path / "out")
    sample, rate = sf.read(plan["cubase_slots"][0]["cubase"]["sample_path"])
    assert len(sample) == rate  # No two seconds of leading timeline silence.
    assert sample[0, 0] == pytest.approx(.2, abs=.001)
    preview, _ = sf.read(plan["audio_preview"]["path"])
    assert abs(preview[:2*rate]).max() == 0
    assert preview[int(2.6*rate), 0] == pytest.approx(.6, abs=.001)
    assert preview[int(3.6*rate), 0] == pytest.approx(.2, abs=.001)  # reprise resets
    assert not Path(plan["mixdown"]["path"]).exists()
    assert plan["mixdown"]["status"] == "awaiting_cubase_export"


@pytest.mark.parametrize("premiere,cubase,count,last", [
    (r"D:\Editor\Soda Pop\soda pop\Soda pop_1_1.prproj", r"D:\Editor\Soda Pop\soda pop\soda pop.cpr", 16, ["3", "2"]),
    (r"D:\Editor\Golden\Adobe Premiere Pro Auto-Save\golden_1--f80fc78b-3e33-59a8-be37-bfe00ee618bc-2026-09-23_22-05-49.prproj", r"D:\Editor\Golden\golden.cpr", 14, ["13", "14"]),
])
def test_real_paired_templates(premiere, cubase, count, last):
    if not Path(premiere).is_file() or not Path(cubase).is_file():
        pytest.skip("Local reference templates unavailable")
    before = Path(cubase).read_bytes()
    scenes = inspect_prproj(premiere)["scene_slots"]
    inventory = inspect_sampler_tracks(cubase)
    assert len(scenes) == len(inventory["tracks"]) == count
    assert [s["nested_sequence"] for s in scenes[-2:]] == last
    assert Path(cubase).read_bytes() == before


def test_late_source_word_does_not_make_short_midi_notes_silent(tmp_path):
    path = tmp_path / "you.wav"
    sf.write(path, np.r_[np.zeros(160), np.full(4000, .5), np.zeros(160)], 8000)
    scenes, slots, inventory = fixture()
    plan = {"project": {"cubase_project": "test.cpr"}, "mixdown": {"sample_rate": 8000},
            "cubase_import": {"application_status": "stale_verified"},
            "cubase_slots": build_sampler_slots(scenes, slots,
                [{"path": str(path), "category": "YOU", "duration_sec": .54}], {}, inventory)}
    for slot in plan["cubase_slots"]:
        slot["audio"]["placement_offset_sec"] = .44
    render_sampler_bundle(plan, tmp_path / "out")
    sample, rate = sf.read(plan["cubase_slots"][0]["cubase"]["sample_path"])
    assert np.abs(sample[:int(.1*rate)]).mean() > .4  # A short MIDI note is audible.
    assert len(sample)/rate == pytest.approx(.5)
    preview, _ = sf.read(plan["audio_preview"]["path"])
    assert np.abs(preview[:int(.44*rate)]).max() == 0
    assert preview[int(.48*rate), 0] > .4
    manifest = json.loads((tmp_path / "out/cubase_import.json").read_text())
    assert manifest["tracks"][0]["placement_offset_sec"] == .44  # Not the last repeat's offset.
    assert "cubase_import" not in plan
    assert len(verify_sampler_inputs(plan["cubase_slots"])) == 3


@pytest.mark.parametrize("silence", [True, False])
def test_import_preflight_rejects_silent_or_late_samples(tmp_path, silence):
    path = tmp_path / "bad.wav"
    pcm = np.zeros(8000)
    if not silence:
        pcm[7520:] = .5  # Reported track 2/16 regression: .94 seconds of silence.
    sf.write(path, pcm, 8000)
    with pytest.raises(ValueError, match="Silent|silence"):
        verify_sampler_inputs([{"cubase": {"track_name": "Sampler Track 02", "sample_path": str(path)}}])
