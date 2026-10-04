import json
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import pytest

from autoedit.cubase import host


@pytest.mark.parametrize("expected_indices,selected_indices,name_pattern", [
    ((1, 2), (1,), "Sampler Track {i:02d}"),
    (tuple(range(2, 14)), tuple(range(1, 13)), "SAMPLER TRACK {i}"),
])
def test_import_reads_manual_selection_before_any_project_or_selection_action(tmp_path, monkeypatch, expected_indices, selected_indices, name_pattern):
    plan = {"project": {"cubase_project": "locked.cpr"}, "cubase_slots": [
        {"cubase": {"track_name": name_pattern.format(i=i)}} for i in expected_indices]}
    path = tmp_path / "edit-plan.json"
    path.write_text(json.dumps(plan))
    tracks = ''.join(f'<obj class="MSamplerTrackEvent"><obj name="Node"><string name="Name" '
                     f'value="{name_pattern.format(i=i)}"/></obj></obj>' for i in selected_indices)
    root = ET.fromstring(f'<root><list name="track">{tracks}</list></root>')
    calls = []
    class Window:
        def is_visible(self): return True
        def window_text(self): return "Cubase Pro Project - locked"
        def type_keys(self, *args, **kwargs): raise AssertionError("Selection shortcuts must not be sent")
        def menu_select(self, *args): raise AssertionError("Must stop before Save As or import")
    class App:
        def connect(self, **kwargs): return self
        def windows(self): return [Window()]
    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(Application=lambda **kw: App()))
    monkeypatch.setattr(host, "verify_sampler_inputs", lambda slots: [])
    def export(app, window, destination):
        calls.append(destination.name)
        return ET.ElementTree(root)
    monkeypatch.setattr(host, "_export", export)
    with pytest.raises(RuntimeError, match="manually"):
        host.import_samples(path, log=lambda _: None)
    assert calls == ["before.xml"]
    assert json.loads(path.read_text()) == plan


@pytest.mark.parametrize("track_indices,name_pattern", [((1, 2), "Sampler Track {i:02d}"),
                                                        (tuple(range(2, 14)), "SAMPLER TRACK {i}")])
def test_import_updates_open_project_without_saving_and_cleans_success_artifacts(tmp_path, monkeypatch, track_indices, name_pattern):
    track_xml = ''.join(
        f'<obj class="MSamplerTrackEvent"><obj name="Node"><string name="Name" value="{name_pattern.format(i=i)}"/></obj></obj>'
        for i in track_indices
    )
    root = ET.fromstring(f'<root><list name="track">{track_xml}</list></root>')
    project = tmp_path / "paired.cpr"
    plan = {
        "project": {"cubase_project": str(project)},
        "cubase_slots": [{"cubase": {"track_name": name_pattern.format(i=i), "sample_path": f"sample{i}.wav"}}
                         for i in track_indices],
    }
    plan_path = tmp_path / "edit-plan.json"
    plan_path.write_text(json.dumps(plan))
    commands = []
    exports = []

    class Window:
        def is_visible(self): return True
        def window_text(self): return "Cubase Pro Project - paired *"
        def set_focus(self): pass
        def menu_select(self, command): commands.append(command)

    class Dialog:
        def wait(self, *_args, **_kwargs): pass
        def wait_not(self, *_args, **_kwargs): pass
        def set_focus(self): pass
        def rectangle(self): return SimpleNamespace(width=lambda: 1145, height=lambda: 717)
        def click_input(self, **_kwargs): pass
        def child_window(self, **_kwargs): return SimpleNamespace(click_input=lambda: None)
        def capture_as_image(self):
            def save(path): Path(path).write_bytes(b"dialog screenshot")
            return SimpleNamespace(save=save)

    class App:
        def connect(self, **_kwargs): return self
        def windows(self): return [Window()]
        def window(self, **_kwargs): return Dialog()

    monkeypatch.setitem(sys.modules, "pywinauto", SimpleNamespace(Application=lambda **_kw: App()))
    monkeypatch.setattr(host, "verify_sampler_inputs", lambda _slots: [{"ok": True}])
    monkeypatch.setattr(host, "_file_dialog", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(host, "_set_checkbox", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(host, "prepare_archive", lambda _before, _slots, dest: (Path(dest).write_text("archive"), dest)[1])
    monkeypatch.setattr(host, "verify_import", lambda *_args: {"verified_tracks": 2, "midi_preserved": True})

    def export(_app, _window, destination):
        exports.append(Path(destination).name)
        Path(destination).write_text("export")
        return ET.ElementTree(root)

    monkeypatch.setattr(host, "_export", export)
    result = host.import_samples(plan_path, log=lambda _message: None)

    assert commands.count("File->Import->Track Archive...") == 1
    assert all("Save As..." not in command for command in commands)
    assert exports == ["before.xml", "after.xml"]
    assert result["project_path"] == str(project.resolve())
    assert result["project_saved"] is False
    assert "project_copy" not in result
    assert project.exists() is False
    session = next((tmp_path / "cubase_imports").iterdir())
    assert sorted(p.name for p in session.iterdir()) == ["verification.json"]
    assert json.loads((session / "verification.json").read_text())["project_saved"] is False
    saved_plan = json.loads(plan_path.read_text())
    assert saved_plan["cubase_import"]["project_saved"] is False


def test_cli_always_uses_manual_selection_including_legacy_flag(monkeypatch):
    from autoedit.cli import build_parser
    calls = []
    monkeypatch.setattr(host, "import_samples", lambda path, log: calls.append(path))
    for options in ([], ["--manual-tracks"]):
        args = build_parser().parse_args(["cubase-import", "test.json", *options])
        assert args.func(args) == 0
    assert calls == ["test.json", "test.json"]
