"""Native (D) companion tracks opt in without changing legacy run mapping."""
import copy
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import soundfile as sf

from autoedit.audio.scene_match import build_sampler_slots
from autoedit.cubase.inspect import inspect_sampler_tracks, select_sampler_tracks
from autoedit.cubase.sampler import render_sampler_bundle
from autoedit.premiere.prproj import inspect_prproj, scene_run_slots
from autoedit.validate import validate_inputs
from test_sampler_scenes import fragment


def native_cpr(tmp_path, names):
    path = tmp_path / "duplicate.cpr"
    nodes = []
    for name in names:
        encoded = name.encode("ascii")
        nodes.append(b"\x80\x00\x00\xbf" + struct.pack(">II", 1, len(encoded) + 4)
                     + encoded + b"\x00\xef\xbb\xbf")
    path.write_bytes(b"RIFF\x00\x00\x00\x00NUND" + b"".join(nodes))
    return path


def duplicate_fixture(tmp_path, names=None):
    path = native_cpr(tmp_path, names or ["SAMPLER TRACK 1", "SAMPLER TRACK 2",
                                        "SAMPLER TRACK 3", "SAMPLER TRACK 3 (D)", "SAMPLER TRACK 4"])
    # The (D) number intentionally differs from the original scene's track.
    scenes = scene_run_slots([{"index": 1, "items": [fragment("A", 0), fragment("B", .5),
                                                    fragment("A", 1), fragment("C", 1.5)]}])
    parents = [{"premiere": {"nested_sequence_uid": uid, "duration_sec": 1},
                "video": {"in_sec": i * 10, "out_sec": i * 10 + 1}}
               for i, uid in enumerate(("A", "B", "C"))]
    return scenes, parents, inspect_sampler_tracks(path)


def test_only_native_duplicate_nodes_enable_the_extra_case(tmp_path):
    scenes, _, inventory = duplicate_fixture(tmp_path)
    assert [t["index"] for t in inventory["tracks"]] == [1, 2, 3, 4]
    assert [t["name"] for t in inventory["ordered_tracks"]] == [
        "SAMPLER TRACK 1", "SAMPLER TRACK 2", "SAMPLER TRACK 3",
        "SAMPLER TRACK 3 (D)", "SAMPLER TRACK 4"]
    path = native_cpr(tmp_path, ["SAMPLER TRACK 1", "SAMPLER TRACK 2", "SAMPLER TRACK 3", "SAMPLER TRACK 4"])
    path.write_bytes(path.read_bytes() + b"SAMPLER TRACK 3 (D)\0")
    ordinary = inspect_sampler_tracks(path)
    assert "ordered_tracks" not in ordinary
    with pytest.raises(ValueError, match="exactly one"):
        select_sampler_tracks(ordinary, "2-4", len(scenes), scenes=scenes)


def test_unsupported_duplicate_node_envelope_keeps_legacy_count_validation(tmp_path):
    scenes, _, inventory = duplicate_fixture(tmp_path)
    path = Path(inventory["path"])
    raw = path.read_bytes()
    label = b"SAMPLER TRACK 3 (D)"
    raw = raw.replace(struct.pack(">I", len(label) + 4) + label,
                      struct.pack(">I", len(label) + 5) + label)
    path.write_bytes(raw)
    inventory = inspect_sampler_tracks(path)
    assert "ordered_tracks" not in inventory
    with pytest.raises(ValueError, match="exactly one"):
        select_sampler_tracks(inventory, "2-4", len(scenes), scenes=scenes)


def test_duplicate_mapping_uses_uid_and_keeps_original_inventory(tmp_path):
    scenes, parents, inventory = duplicate_fixture(tmp_path)
    before = copy.deepcopy(inventory)
    # Display names may collide; only UID identity determines a reprise.
    for scene in scenes:
        scene["nested_sequence"] = "same name"
    selected = select_sampler_tracks(inventory, "2-4", len(scenes), scenes=scenes)
    assert inventory == before
    assert selected["selected_track_indices"] == [2, 3, 4]
    assert selected["mapping_mode"] == "duplicate_scene_tracks"
    slots = build_sampler_slots(scenes, parents, [{"path": "hi.wav", "category": "HI", "duration_sec": 1}], {}, selected)
    assert [s["cubase"]["track_name"] for s in slots] == [
        "SAMPLER TRACK 2", "SAMPLER TRACK 3", "SAMPLER TRACK 3 (D)", "SAMPLER TRACK 4"]
    assert [s["cubase"]["track_index"] for s in slots] == [2, 3, 2, 4]
    assert slots[2]["cubase"]["duplicate_of"] == "SAMPLER TRACK 2"
    assert slots[2]["audio"] == slots[0]["audio"]
    assert [s["premiere"]["instances"] for s in slots] == [s["instances"] for s in scenes]


def test_blank_and_one_track_per_run_keep_legacy_behavior_with_duplicate_nodes(tmp_path):
    scenes, parents, inventory = duplicate_fixture(tmp_path)
    assert select_sampler_tracks(inventory, "", len(scenes), scenes=scenes) is inventory
    selected = select_sampler_tracks(inventory, "1-4", len(scenes), scenes=scenes)
    assert "mapping_mode" not in selected
    slots = build_sampler_slots(scenes, parents, [{"path": "hi.wav", "category": "HI", "duration_sec": 1}], {}, selected)
    assert [s["cubase"]["track_index"] for s in slots] == [1, 2, 3, 4]
    assert all("duplicate_of" not in s["cubase"] for s in slots)


def test_ai_blank_selection_infers_unique_native_uid_mapping(tmp_path):
    scenes, _, inventory = duplicate_fixture(tmp_path)
    inferred = select_sampler_tracks(inventory, "", len(scenes), scenes=scenes, infer_duplicates=True)
    explicit = select_sampler_tracks(inventory, "2-4", len(scenes), scenes=scenes)
    assert inferred == explicit
    # Unsupported reprise order must never fall back to numbered 1-4.
    scenes[2]["nested_sequence_uid"] = "new-uid"
    with pytest.raises(ValueError, match="Cannot infer"):
        select_sampler_tracks(inventory, "", len(scenes), scenes=scenes, infer_duplicates=True)


def test_sampler_cache_changes_when_track_assignments_change(tmp_path):
    scenes, parents, inventory = duplicate_fixture(tmp_path)
    selected = select_sampler_tracks(inventory, "2-4", len(scenes), scenes=scenes)
    paths = [tmp_path / "hi.wav", tmp_path / "bye.wav"]
    for i, path in enumerate(paths):
        sf.write(path, np.full(2000, .2 + .3*i), 8000)
    voices = [{"path": str(p), "category": "HI", "duration_sec": .25} for p in paths]
    slots = build_sampler_slots(scenes, parents, voices[:1], {}, selected)
    slots[1]["audio"] = copy.deepcopy(voices[1])
    plan = {"project": {"cubase_project": inventory["path"]}, "cubase_slots": slots,
            "mixdown": {"sample_rate": 8000}}
    render_sampler_bundle(plan, tmp_path / "out")
    old_paths = [s["cubase"]["sample_path"] for s in slots]
    slots[0]["cubase"]["track_index"], slots[1]["cubase"]["track_index"] = 3, 2
    slots[0]["cubase"]["track_name"], slots[1]["cubase"]["track_name"] = "SAMPLER TRACK 3", "SAMPLER TRACK 2"
    slots[2]["cubase"].update(track_index=3, duplicate_of="SAMPLER TRACK 3")
    render_sampler_bundle(plan, tmp_path / "out")
    assert Path(slots[0]["cubase"]["sample_path"]).parent != Path(old_paths[0]).parent
    for slot, old_path in zip(slots, old_paths):
        np.testing.assert_array_equal(sf.read(slot["cubase"]["sample_path"])[0], sf.read(old_path)[0])


@pytest.mark.parametrize("names", [
    ["SAMPLER TRACK 1", "SAMPLER TRACK 2", "SAMPLER TRACK 3", "SAMPLER TRACK 4"],
    ["SAMPLER TRACK 1", "SAMPLER TRACK 2", "SAMPLER TRACK 3 (D)", "SAMPLER TRACK 3", "SAMPLER TRACK 4"],
    ["SAMPLER TRACK 1", "SAMPLER TRACK 2", "SAMPLER TRACK 3", "SAMPLER TRACK 4", "SAMPLER TRACK 3 (D)"],
    ["SAMPLER TRACK 1", "SAMPLER TRACK 2", "SAMPLER TRACK 3", "SAMPLER TRACK 3 (D)", "SAMPLER TRACK 3 (D)", "SAMPLER TRACK 4"],
    ["SAMPLER TRACK 1", "SAMPLER TRACK 2", "SAMPLER TRACK 3", "SAMPLER TRACK 3 (D)"],
])
def test_missing_extra_or_misordered_tracks_still_fail_before_mapping(tmp_path, names):
    scenes, _, inventory = duplicate_fixture(tmp_path, names)
    with pytest.raises(ValueError, match="exactly one"):
        select_sampler_tracks(inventory, "2-4", len(scenes), scenes=scenes)


def test_mismatched_uid_layout_cannot_use_duplicate_count_as_a_bypass(tmp_path):
    scenes, _, inventory = duplicate_fixture(tmp_path)
    scenes[2]["nested_sequence_uid"] = "C"
    scenes[3]["nested_sequence_uid"] = "A"
    with pytest.raises(ValueError, match="exactly one"):
        select_sampler_tracks(inventory, "2-4", len(scenes), scenes=scenes)


def test_duplicate_binding_is_checked_again_before_audio_selection(tmp_path):
    scenes, parents, inventory = duplicate_fixture(tmp_path)
    selected = select_sampler_tracks(inventory, "2-4", len(scenes), scenes=scenes)
    selected["tracks"][2]["index"] = 3
    with pytest.raises(ValueError, match="no wraparound"):
        build_sampler_slots(scenes, parents, [], {}, selected)


def test_bundle_imports_companion_and_reuses_wav_without_losing_preview_run(tmp_path):
    scenes, parents, inventory = duplicate_fixture(tmp_path)
    selected = select_sampler_tracks(inventory, "2-4", len(scenes), scenes=scenes)
    voice = tmp_path / "hi.wav"
    sf.write(voice, np.full(4000, .3), 8000)
    slots = build_sampler_slots(scenes, parents, [{"path": str(voice), "category": "HI", "duration_sec": .5}], {}, selected)
    plan = {"project": {"cubase_project": inventory["path"]}, "cubase_slots": slots,
            "mixdown": {"sample_rate": 8000}}
    render_sampler_bundle(plan, tmp_path / "out")
    assert slots[2]["cubase"]["sample_path"] == slots[0]["cubase"]["sample_path"]
    manifest = json.loads((tmp_path / "out/cubase_import.json").read_text())
    assert [s["track_name"] for s in manifest["tracks"]] == [s["cubase"]["track_name"] for s in slots]
    assert "SAMPLER TRACK 3 (D)" in manifest["instructions"][0]
    assert "SAMPLER TRACK 1," not in manifest["instructions"][0]
    preview, rate = sf.read(plan["audio_preview"]["path"])
    assert np.abs(preview[rate:int(1.5 * rate)]).mean() > .25
    assert len(list((tmp_path / "out/sampler_samples").glob("*/*.wav"))) == 3


def test_real_damtucosita_validation_mapping_and_bundle_leave_projects_unchanged(tmp_path):
    directory = Path(r"D:\Editor\damtucosita")
    project, cubase = directory / "DAMTUCOSITA flip_3.prproj", directory / "DAME TU COSITA new.cpr"
    if not project.is_file() or not cubase.is_file():
        pytest.skip("Local Dame Tu Cosita pair unavailable")
    before = [hashlib.sha256(p.read_bytes()).digest() for p in (project, cubase)]
    info, inventory = inspect_prproj(project), inspect_sampler_tracks(cubase)
    scenes = info["scene_slots"]
    assert len(scenes) == 15 and len(info["slots"]) == 14
    selected = select_sampler_tracks(inventory, "2-15", len(scenes), scenes=scenes)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    voice = tmp_path / "voice"
    voice.mkdir()
    wav = voice / "hi.wav"
    sf.write(wav, np.full(2000, .3), 8000)
    assert validate_inputs(premiere_project=project, cubase_project=cubase, source_video=video,
                           audio_directory=voice, sampler_tracks="2-15") == []
    parents = [{"premiere": {"nested_sequence_uid": s["nested_sequence_uid"],
                              "duration_sec": s["required_duration_sec"]},
                "video": {"in_sec": i * 10, "out_sec": i * 10 + s["required_duration_sec"]}}
               for i, s in enumerate(info["slots"])]
    slots = build_sampler_slots(scenes, parents, [{"path": str(wav), "category": "HI", "duration_sec": .25}], {}, selected)
    assert [(s["premiere"]["nested_sequence"], s["cubase"]["track_name"]) for s in slots] == [
        *[(str(i), f"SAMPLER TRACK {i + 1}") for i in range(1, 14)],
        ("3", "SAMPLER TRACK 2 (D)"), ("14", "SAMPLER TRACK 15")]
    assert slots[13]["audio"] == slots[2]["audio"]
    assert slots[13]["cubase"]["duplicate_of"] == "SAMPLER TRACK 4"
    plan = {"project": {"cubase_project": str(cubase)}, "cubase_slots": slots,
            "mixdown": {"sample_rate": 8000}}
    render_sampler_bundle(plan, tmp_path / "out")
    assert slots[13]["cubase"]["sample_path"] == slots[2]["cubase"]["sample_path"]
    assert slots[-1]["cubase"]["track_index"] == 15
    assert [hashlib.sha256(p.read_bytes()).digest() for p in (project, cubase)] == before


def test_pipeline_passes_uid_runs_through_validation_and_duplicate_mapping(tmp_path, monkeypatch):
    import autoedit.pipeline as pipeline
    from test_recursive_nests import project

    premiere = project(tmp_path, [
        ("root", "PJ 5 - demo", [[], [("A", 0, .5, 0, .5), ("B", .5, 1, 0, .5),
                                      ("A", 1, 1.5, 0, .5), ("C", 1.5, 2, 0, .5)]]),
        ("A", "A", [[]]), ("B", "B", [[]]), ("C", "C", [[]]),
    ])
    _, _, inventory = duplicate_fixture(tmp_path)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video")
    voice = tmp_path / "voice"
    voice.mkdir()
    wav = voice / "hi.wav"
    sf.write(wav, np.full(4000, .3), 8000)
    monkeypatch.setattr(pipeline, "analyze_audio_dir", lambda *a, **kw: [
        {"id": "hi", "path": str(wav), "name": "hi.wav", "duration_sec": .5, "category": "HI"}])
    monkeypatch.setattr(pipeline, "expand_match_units", lambda items: [])
    monkeypatch.setattr(pipeline, "analyze_video", lambda *a, **kw: {
        "path": str(video), "duration_sec": 100,
        "samples": [{"time_sec": i / 2, "face_visible": 1} for i in range(200)]})
    monkeypatch.setattr(pipeline, "transcribe_source", lambda *a, **kw: {"has_audio": False, "words": []})
    monkeypatch.setattr(pipeline, "prepare_video_only_source", lambda *a, **kw: "video-only.mov")
    monkeypatch.setattr(pipeline, "refine_selected_cuts", lambda *a, **kw: {"samples": []})
    monkeypatch.setattr(pipeline, "write_audio_review", lambda *a, **kw: None)
    config = {"job": {"output_dir": str(tmp_path / "out")},
              "premiere": {"project": str(premiere), "source_media": str(video)},
              "audio": {"directory": str(voice)},
              "cubase": {"project": inventory["path"], "sampler_tracks": "2-4", "sample_rate": 8000},
              "video": {"search_fps": 0, "require_lip_motion": False, "require_clear_face": False,
                        "characters": {"enabled": False}}}
    plan = pipeline.run_pipeline(config)["plan"]
    slots = plan["cubase_slots"]
    assert [s["cubase"]["track_name"] for s in slots] == [
        "SAMPLER TRACK 2", "SAMPLER TRACK 3", "SAMPLER TRACK 3 (D)", "SAMPLER TRACK 4"]
    assert slots[2]["audio"] == slots[0]["audio"]
    assert slots[2]["cubase"]["sample_path"] == slots[0]["cubase"]["sample_path"]
    assert len(plan["slots"]) == 3
    assert "3 numbered sampler tracks + 1 (D) companions" in (tmp_path / "out/run.log").read_text()
