"""Explicit sampler subsets preserve track names, numbering and legacy defaults."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from autoedit.audio.scene_match import build_sampler_slots
from autoedit.cubase.inspect import inspect_sampler_tracks, parse_sampler_tracks, select_sampler_tracks
from autoedit.cubase.sampler import render_sampler_bundle
from autoedit.premiere.prproj import inspect_prproj
from autoedit.validate import validate_inputs
from test_sampler_scenes import fixture


def cpr(tmp_path, names):
    path = tmp_path / "paired.cpr"
    path.write_bytes(b"RIFF\x00\x00\x00\x00NUND\x00" + b"\x00".join(n.encode("ascii") for n in names) + b"\x00")
    return path


def test_inspector_preserves_case_padding_and_numeric_order(tmp_path):
    path = cpr(tmp_path, ["SAMPLER TRACK 13", "Sampler Track 02", "sampler track 1", "SAMPLER TRACK 13"])
    before = path.read_bytes()
    assert inspect_sampler_tracks(path)["tracks"] == [
        {"index": 1, "name": "sampler track 1"}, {"index": 2, "name": "Sampler Track 02"},
        {"index": 13, "name": "SAMPLER TRACK 13"}]
    assert path.read_bytes() == before


def test_conflicting_track_names_are_still_rejected(tmp_path):
    with pytest.raises(ValueError, match="Ambiguous"):
        inspect_sampler_tracks(cpr(tmp_path, ["Sampler Track 01", "SAMPLER TRACK 1"]))


@pytest.mark.parametrize("selection", [None, "", "  ", []])
def test_blank_selection_keeps_original_inventory_and_mapping(selection):
    scenes, slots, inventory = fixture()
    selected = select_sampler_tracks(inventory, selection, len(scenes))
    assert selected is inventory
    pool = [{"path": "hi.wav", "category": "HI", "duration_sec": 1}]
    assert build_sampler_slots(scenes, slots, pool, {}, selected) == build_sampler_slots(scenes, slots, pool, {}, inventory)


@pytest.mark.parametrize("selection,expected", [("2-13", list(range(2, 14))), ("1, 3-5", [1, 3, 4, 5]),
                                                ([2, 3, 4], [2, 3, 4])])
def test_range_and_list_selection(selection, expected):
    assert parse_sampler_tracks(selection) == expected


@pytest.mark.parametrize("selection", ["0-2", "3-2", "2,2", "3,2", "2-4,4", "2-", "2,,3",
                                       "all", "1.5", [True, 2], [2, "3"], [-1, 2], 2, "1-1000000000"])
def test_invalid_selections_are_rejected(selection):
    with pytest.raises(ValueError):
        parse_sampler_tracks(selection)


def test_selected_sampler_numbers_keep_reprise_and_timing_bindings():
    scenes, slots, _ = fixture()
    inventory = {"path": "paired.cpr", "tracks": [{"index": i, "name": f"SAMPLER TRACK {i}"}
                                                   for i in range(1, 5)]}
    original = json.loads(json.dumps(inventory))
    selected = select_sampler_tracks(inventory, "2-4", len(scenes))
    result = build_sampler_slots(scenes, slots, [{"path": "hi.wav", "category": "HI", "duration_sec": 1}], {}, selected)
    assert inventory == original
    assert [s["cubase"]["track_index"] for s in result] == [2, 3, 4]
    assert [s["cubase"]["track_name"] for s in result] == ["SAMPLER TRACK 2", "SAMPLER TRACK 3", "SAMPLER TRACK 4"]
    assert [s["premiere"]["nested_sequence_uid"] for s in result] == ["A", "B", "A"]
    assert result[0]["premiere"]["instances"] == scenes[0]["instances"]
    assert result[2]["audio"] == result[0]["audio"]
    # No implicit subset/offset is allowed in the legacy path.
    with pytest.raises(ValueError, match="no wraparound"):
        build_sampler_slots(scenes, slots, [], {}, {**selected, "selected_track_indices": None})


def test_missing_or_wrong_count_selection_stops_before_mapping():
    inventory = {"tracks": [{"index": i, "name": f"Sampler Track {i}"} for i in range(1, 4)]}
    with pytest.raises(ValueError, match="exactly one"):
        select_sampler_tracks(inventory, "2-3", 3)
    with pytest.raises(ValueError, match="missing"):
        select_sampler_tracks(inventory, "2-4", 3)


def test_validation_checks_selected_tracks_and_scene_count(tmp_path):
    from test_recursive_nests import multilevel
    project = multilevel(tmp_path)  # Three scene runs.
    cubase = cpr(tmp_path, [f"SAMPLER TRACK {i}" for i in range(1, 5)])
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    voice = tmp_path / "voice"
    voice.mkdir()
    (voice / "hi.wav").write_bytes(b"RIFF")
    kwargs = dict(premiere_project=project, cubase_project=cubase, source_video=video, audio_directory=voice)
    assert validate_inputs(**kwargs, sampler_tracks="2-4") == []
    assert any("exactly one" in e for e in validate_inputs(**kwargs, sampler_tracks="2-3"))
    assert any("missing" in e for e in validate_inputs(**kwargs, sampler_tracks="3-5"))


def test_cli_accepts_sampler_range_without_changing_default():
    from autoedit.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(["run"]).sampler_tracks is None
    assert parser.parse_args(["run", "--sampler-tracks", "2-13"]).sampler_tracks == "2-13"


def test_real_zoomally_pair_maps_4_to_15_into_2_to_13_and_renders_bundle(tmp_path):
    directory = Path(r"D:\Editor\Zoomally")
    projects = list(directory.glob("ZOONOMALY THEME SONG 2 file *_1.prproj"))
    cubase = directory / "ZONOMALY CBS.cpr"
    if len(projects) != 1 or not cubase.is_file():
        pytest.skip("Local Zoomally pair unavailable")
    project = projects[0]
    before = [hashlib.sha256(p.read_bytes()).digest() for p in (project, cubase)]
    info = inspect_prproj(project)
    inventory = inspect_sampler_tracks(cubase)
    assert [t["index"] for t in inventory["tracks"]] == list(range(1, 14))
    selected = select_sampler_tracks(inventory, "2-13", len(info["scene_slots"]))
    video_slots = [{"premiere": {"nested_sequence_uid": s["nested_sequence_uid"],
                                  "duration_sec": s["required_duration_sec"]},
                    "video": {"in_sec": i * 10, "out_sec": i * 10 + s["required_duration_sec"]}}
                   for i, s in enumerate(info["slots"])]
    voice = tmp_path / "hi.wav"
    sf.write(voice, np.full(2000, .3), 8000)
    slots = build_sampler_slots(info["scene_slots"], video_slots,
                                [{"path": str(voice), "category": "HI", "duration_sec": .25}], {}, selected)
    assert [(s["premiere"]["nested_sequence"], s["cubase"]["track_name"]) for s in slots] == [
        (str(i + 2), f"SAMPLER TRACK {i}") for i in range(2, 14)]
    plan = {"project": {"cubase_project": str(cubase)}, "cubase_slots": slots,
            "mixdown": {"sample_rate": 8000}}
    render_sampler_bundle(plan, tmp_path / "out")
    manifest = json.loads((tmp_path / "out/cubase_import.json").read_text(encoding="utf-8"))
    assert [s["track_index"] for s in manifest["tracks"]] == list(range(2, 14))
    assert [Path(s["sample"]).name for s in manifest["tracks"]] == [f"Sampler_Track_{i:02d}.wav" for i in range(2, 14)]
    assert not any("SAMPLER TRACK 1," in text for text in manifest["instructions"])
    assert [hashlib.sha256(p.read_bytes()).digest() for p in (project, cubase)] == before
