"""Multilevel templates keep wrapper edits and fill shared footage leaves."""
import gzip
import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from autoedit.plan.generator import build_edit_plan
from autoedit.premiere.adapter import apply_payload
from autoedit.premiere.prproj import inspect_prproj, seconds_to_ticks
from autoedit.validate import validate_inputs


def project(tmp_path, sequences):
    root = ET.Element("Project")
    oid = 0

    def obj(tag):
        nonlocal oid
        oid += 1
        return ET.SubElement(root, tag, ObjectID=str(oid))

    def ref(parent, tag, target):
        ET.SubElement(parent, tag, ObjectRef=target.get("ObjectID"))

    for uid, name, tracks in sequences:
        seq = ET.SubElement(root, "Sequence", ObjectUID=uid)
        ET.SubElement(seq, "Name").text = name
        groups = ET.SubElement(seq, "TrackGroups")
        group = obj("VideoTrackGroup")
        ref(ET.SubElement(groups, "Pair"), "Second", group)
        for index, rows in enumerate(tracks):
            track = obj("VideoClipTrack")
            ET.SubElement(track, "Index").text = str(index)
            ref(group, "Track", track)
            items = ET.SubElement(track, "TrackItems")
            for child, start, end, inn, out in rows:
                item = obj("VideoClipTrackItem")
                timing = ET.SubElement(item, "TrackItem")
                for tag, value in (("Start", start), ("End", end)):
                    ET.SubElement(timing, tag).text = str(seconds_to_ticks(value))
                ref(items, "Item", item)
                subclip, clip, source = obj("SubClip"), obj("VideoClip"), obj("VideoSequenceSource")
                ref(item, "SubClip", subclip)
                ref(subclip, "Clip", clip)
                inner = ET.SubElement(clip, "Clip")
                ref(inner, "Source", source)
                for tag, value in (("InPoint", inn), ("OutPoint", out)):
                    ET.SubElement(inner, tag).text = str(seconds_to_ticks(value))
                ET.SubElement(ET.SubElement(source, "SequenceSource"), "Sequence", ObjectURef=child)
    path = tmp_path / "nested.prproj"
    path.write_bytes(gzip.compress(ET.tostring(root)))
    return path


def multilevel(tmp_path):
    return project(tmp_path, [
        ("root", "BEAT MAU", [[], [("wrap", 10, 12, 1, 3), ("b", 12, 13, 0, 1),
                                     ("wrap", 20, 22, 0, 2)], [], [("a", 12, 12.5, .5, 1)]]),
        ("wrap", "Wrapper", [[], [("inner", 0, 4, 0, 4)]]),
        ("inner", "Inner wrapper", [[], [("a", 0, 2, 3, 5), ("a", 2, 4, 0, 2)]]),
        ("a", "same name", [[]]), ("b", "same name", [[]]),
        ("unused", "same name", [[]]),
    ])


def test_shared_leaves_offsets_reprises_and_overlay_tracks(tmp_path):
    path = multilevel(tmp_path)
    before = path.read_bytes()
    info = inspect_prproj(path)
    assert info["template_sequence"] == "BEAT MAU"
    assert info["template_selection"] == "unique_root"
    assert [s["nested_sequence_uid"] for s in info["slots"]] == ["a", "b"]
    a, b = info["slots"]
    assert a["required_duration_sec"] == 5  # Full local refs, including the trimmed portion.
    assert a["occurrences"] == 4
    assert [(i["start_sec"], i["end_sec"], i["source_in_sec"], i["source_out_sec"])
            for i in a["instances"]] == [(10, 11, 4, 5), (11, 12, 0, 1),
                                          (12, 12.5, .5, 1), (20, 22, 3, 5)]
    assert [i["template_video_track_index"] for i in a["instances"]] == [1, 1, 3, 1]
    scenes = info["scene_slots"]
    assert [s["nested_sequence_uid"] for s in scenes] == ["a", "b", "a"]
    assert [len(s["instances"]) for s in scenes] == [2, 1, 1]
    assert {s["sequence_uid"] for s in info["nested_structure"]} == {"root", "wrap", "inner"}
    assert b["required_duration_sec"] == 1
    assert inspect_prproj(path, video_track_index=0)["slots"] == []
    assert inspect_prproj(path, template_sequence="Typo")["template_sequence"] is None
    assert path.read_bytes() == before


def test_ambiguous_roots_require_configuration(tmp_path):
    path = project(tmp_path, [("one", "One", [[], [("a", 0, 1, 0, 1)]]),
                              ("two", "Two", [[], [("a", 0, 1, 0, 1)]]), ("a", "1", [[]])])
    with pytest.raises(ValueError, match="Multiple template"):
        inspect_prproj(path)
    assert inspect_prproj(path, template_sequence="Two")["template_sequence"] == "Two"


def test_cycle_is_rejected_before_planning(tmp_path):
    path = project(tmp_path, [("root", "Main", [[], [("wrap", 0, 1, 0, 1)]]),
                              ("wrap", "Wrapper", [[], [("root", 0, 1, 0, 1)]])])
    with pytest.raises(ValueError, match="Cyclic"):
        inspect_prproj(path, template_sequence="Main")


def test_retimed_wrapper_requires_timing_map(tmp_path):
    path = project(tmp_path, [("root", "Main", [[], [("wrap", 0, 1, 0, 2)]]),
                              ("wrap", "Wrapper", [[], [("a", 0, 2, 0, 2)]]), ("a", "1", [[]])])
    with pytest.raises(ValueError, match="Retimed nested"):
        inspect_prproj(path, template_sequence="Main")


def test_validation_accepts_detected_template(tmp_path):
    path = multilevel(tmp_path)
    cpr, video = tmp_path / "paired.cpr", tmp_path / "source.mp4"
    cpr.write_bytes(b"cpr")
    video.write_bytes(b"video")
    audio = tmp_path / "voice"
    audio.mkdir()
    (audio / "hi.wav").write_bytes(b"RIFF")
    assert validate_inputs(premiere_project=path, cubase_project=cpr,
                           source_video=video, audio_directory=audio) == []


@pytest.mark.parametrize("sampler_selection", [None, "2-4"])
def test_pipeline_keeps_detected_name_structure_and_audio_scene_bindings(tmp_path, monkeypatch, sampler_selection):
    import autoedit.pipeline as pipeline

    path = multilevel(tmp_path)
    monkeypatch.setattr(pipeline, "require_inputs", lambda **kw: None)
    monkeypatch.setattr(pipeline, "inspect_sampler_tracks", lambda path: {
        "path": path, "tracks": [{"index": i, "name": f"Sampler Track {i:02d}"}
                                  for i in range(1, 5 if sampler_selection else 4)]})
    monkeypatch.setattr(pipeline, "analyze_audio_dir", lambda *a, **kw: [
        {"id": "hi", "path": "hi.wav", "name": "hi.wav", "duration_sec": 1, "category": "HI"}])
    monkeypatch.setattr(pipeline, "expand_match_units", lambda items: [])
    monkeypatch.setattr(pipeline, "analyze_video", lambda *a, **kw: {
        "path": "source.mp4", "duration_sec": 100,
        "samples": [{"time_sec": i / 2, "face_visible": 1} for i in range(200)]})
    monkeypatch.setattr(pipeline, "transcribe_source", lambda *a, **kw: {"has_audio": False, "words": []})
    monkeypatch.setattr(pipeline, "prepare_video_only_source", lambda *a, **kw: "video-only.mov")
    monkeypatch.setattr(pipeline, "refine_selected_cuts", lambda *a, **kw: {"samples": []})
    monkeypatch.setattr(pipeline, "write_audio_review", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "apply_cubase_plan", lambda *a, **kw: {})
    config = {"job": {"output_dir": str(tmp_path / "output")},
              "premiere": {"project": str(path), "source_media": "source.mp4"},
              "cubase": {"project": "paired.cpr", "sampler_tracks": sampler_selection},
              "video": {"require_lip_motion": False, "characters": {"enabled": False}}}
    plan = pipeline.run_pipeline(config)["plan"]
    assert plan["project"]["template_sequence"] == "BEAT MAU"
    assert "template_sequence" not in config["premiere"]  # Runtime detection does not change future jobs.
    assert len(plan["slots"]) == 2
    assert [s["premiere"]["nested_sequence_uid"] for s in plan["cubase_slots"]] == ["a", "b", "a"]
    assert [s["cubase"]["track_index"] for s in plan["cubase_slots"]] == list(range(2, 5) if sampler_selection else range(1, 4))
    assert plan["cubase_slots"][0]["premiere"]["instances"][0]["source_in_sec"] == 4
    assert plan["cubase_slots"][2]["premiere"]["instances"][0]["start_sec"] == 20
    assert apply_payload(plan)["nested_structure"] == plan["nested_structure"]


ZOOMALLY = Path(r"D:\Editor\Zoomally")


def test_real_zoomally_project_targets_only_4_through_15():
    paths = list(ZOOMALLY.glob("ZOONOMALY THEME SONG 2 file *_1.prproj"))
    if len(paths) != 1:
        pytest.skip("Local Zoomally project unavailable")
    path = paths[0]
    before = hashlib.sha256(path.read_bytes()).digest()
    info = inspect_prproj(path)
    assert info["template_sequence"] == "BEAT MAU"
    assert [s["nested_sequence"] for s in info["slots"]] == [str(i) for i in range(4, 16)]
    assert [s["nested_sequence"] for s in info["scene_slots"]] == [str(i) for i in range(4, 16)]
    for name in ("4", "8", "12"):
        slot = next(s for s in info["slots"] if s["nested_sequence"] == name)
        assert slot["occurrences"] == 28
        assert {i["template_video_track_index"] for i in slot["instances"]} == {1, 3}
    assert info["slots"][-1]["required_duration_sec"] == pytest.approx(2.36)
    last_instance = info["scene_slots"][-1]["instances"][-1]
    assert last_instance["end_sec"] == pytest.approx(103.36)
    assert last_instance["source_in_sec"] == pytest.approx(.08)
    assert last_instance["source_out_sec"] == pytest.approx(1.08)  # Wrapper trim propagated.
    matches = [{"video": {"in_sec": i * 10, "out_sec": i * 10 + s["required_duration_sec"]}}
               for i, s in enumerate(info["slots"])]
    plan = build_edit_plan(config={"premiere": {"template_sequence": info["template_sequence"]}},
                           slots=info["slots"], matches=matches, video_info={"duration_sec": 1000})
    plan["nested_structure"] = info["nested_structure"]
    payload = apply_payload(plan)
    assert len(payload["fill_nested_sequences"]) == 12
    assert payload["nested_structure"] == info["nested_structure"]
    assert all(a["video_track_index"] == a["overwrite_at_sec"] == 0
               for a in payload["fill_nested_sequences"])
    assert all(not a["nested_sequence"].startswith("Nested") for a in payload["fill_nested_sequences"])
    assert hashlib.sha256(path.read_bytes()).digest() == before
