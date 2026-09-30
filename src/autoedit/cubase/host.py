"""Cubase 13 desktop import with read-back verification in the open project.

Select the template's sampler tracks manually, then invoke import. Native
file dialogs and menu commands handle I/O. The custom Import Options dialog
uses a checked, version-specific layout; unknown layouts stop before import.
"""
from pathlib import Path
from datetime import datetime
import json
import time

from autoedit.cubase.archive import prepare_archive, sampler_tracks, verify_import, ET
from autoedit.cubase.sampler import verify_sampler_inputs


def _file_dialog(app, title, filename, save=False):
    dialog = app.window(title=title)
    dialog.wait("visible", timeout=20)
    dialog.set_focus()
    dialog.child_window(class_name="Edit", control_id=1001 if save else 1148).set_edit_text(str(Path(filename).resolve()))
    dialog.type_keys("%s" if save else "%o")
    dialog.wait_not("visible", timeout=20)


def _export(app, window, path):
    window.set_focus()
    window.menu_select("File->Export->Selected Tracks...")
    dialog = app.window(title="Export Selected Tracks")
    dialog.wait("visible", timeout=10)
    dialog.set_focus()
    time.sleep(.3)
    rect = dialog.rectangle()
    if (rect.width(), rect.height()) != (570, 228):
        dialog.type_keys("{ESC}")
        raise RuntimeError("Unsupported Export Tracks dialog layout; no import was performed")
    dialog.click_input(coords=(47, 100))  # Reference Media Files
    dialog.click_input(coords=(425, 200))
    _file_dialog(app, "Save As", path, save=True)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if Path(path).is_file():
            try:
                return ET.parse(path)
            except ET.ParseError:
                pass
        time.sleep(.2)
    raise RuntimeError("Cubase did not export selected tracks")


def _checked(dialog, y):
    """Recognize the inner tick, excluding the checkbox border (100% DPI)."""
    pixels = dialog.capture_as_image().convert("RGB")
    values = [min(pixels.getpixel((x, yy))) for x in range(568, 577) for yy in range(y-4, y+5)]
    return sum(v > 150 for v in values) >= 5


def _set_checkbox(dialog, y, checked):
    if _checked(dialog, y) != checked:
        dialog.click_input(coords=(572, y))
        time.sleep(.2)
    if _checked(dialog, y) != checked:
        raise RuntimeError("Could not verify Cubase import checkbox; import stopped")


def import_samples(plan_path, log=print):
    from pywinauto import Application
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("superseded_by"):
        raise ValueError("This plan was superseded. Use " + plan["superseded_by"])
    slots = plan.get("cubase_slots") or []
    if not slots:
        raise ValueError("Run the tool first to create sampler assignments")
    sample_checks = verify_sampler_inputs(slots)
    # This profile is tested against Cubase 13, not guessed on other versions.
    app = Application(backend="win32").connect(path="Cubase13.exe", timeout=5)
    windows = [w for w in app.windows() if w.is_visible() and w.window_text().startswith("Cubase Pro Project - ")]
    if len(windows) != 1:
        raise RuntimeError("Open only the paired project window in Cubase 13")
    window = windows[0]
    expected_title = Path(plan["project"]["cubase_project"]).stem
    if window.window_text().removeprefix("Cubase Pro Project - ").rstrip(" *") != expected_title:
        raise RuntimeError("The open Cubase project does not match this plan. Open the paired project first.")
    session = plan_path.parent / "cubase_imports" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session.mkdir(parents=True)
    before, after = session / "before.xml", session / "after.xml"
    log(f"Reading your Cubase selection; manually select all {len(slots)} numbered Sampler Tracks before import")
    root = _export(app, window, before).getroot()
    expected = {s["cubase"]["track_name"] for s in slots}
    if set(sampler_tracks(root)) != expected:
        raise RuntimeError(f"Cubase selection did not contain exactly the expected {len(slots)} Sampler Tracks. "
                           "Select the numbered Sampler Tracks manually, then retry Import Cubase. No samples were changed.")
    archive = prepare_archive(before, slots, session / "samples.xml")
    log("Loading samples into the open Cubase project; the project will remain unsaved")
    window.menu_select("File->Import->Track Archive...")
    _file_dialog(app, "Locate Track File", archive)
    dialog = app.window(title="Import Options")
    dialog.wait("visible", timeout=20)
    dialog.set_focus()
    time.sleep(.4)
    rect = dialog.rectangle()
    if (rect.width(), rect.height()) != (1145, 717):
        dialog.type_keys("{ESC}")
        raise RuntimeError("Cubase Import Options layout is unsupported. Use Windows 100% display scaling; no import was performed.")
    dialog.click_input(coords=(115, 95))  # Select All source tracks
    dialog.click_input(coords=(455, 95))  # Select Matching existing tracks
    time.sleep(.3)
    # Import no events, parts, automation or timeline range; only settings.
    for y, state in ((100, False), (202, False), (274, True), (304, False), (612, False)):
        _set_checkbox(dialog, y, state)
    dialog.capture_as_image().save(str(session / "import-options.png"))
    log("Loading samples into matching tracks; MIDI and automation import are disabled")
    dialog.child_window(title="OK", class_name="Button").click_input()
    dialog.wait_not("visible", timeout=30)
    _export(app, window, after)
    verification = verify_import(before, after, slots)
    result = {**verification, "application_status": "samples_loaded_and_verified",
              "sample_files_checked": sample_checks, "rendered_audio_verified": False,
              "project_path": str(Path(plan["project"]["cubase_project"]).resolve()),
              "project_saved": False}
    (session / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plan["cubase_import"] = result
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = plan_path.parent / "cubase_import.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(result)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_files = (before, archive, after, session / "import-options.png")
    for temporary in temporary_files:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            log(f"Could not remove temporary import file {temporary}: {exc}")
    log(f"Verified {len(slots)} samples; MIDI, sampler parameters and effects preserved. "
        f"Project remains unsaved: {result['project_path']}")
    return result
