from pathlib import Path

from autoedit.premiere.prproj import inspect_prproj, ticks_to_seconds, seconds_to_ticks

EXAMPLE = Path(
    r"D:\Editor\Soda Pop\soda pop\Adobe Premiere Pro Auto-Save"
    r"\Soda pop_1--54ea6537-c9cb-4b62-45a0-a808f7e2364b-2026-09-21_20-49-47.prproj"
)
TEMPLATE = Path(r"D:\Editor\Soda Pop\soda pop\Soda pop.prproj")


def test_ticks():
    assert abs(ticks_to_seconds(seconds_to_ticks(1.0)) - 1.0) < 1e-6


def test_example_project_slots():
    if not EXAMPLE.exists():
        return
    info = inspect_prproj(EXAMPLE)
    names = {s["name"] for s in info["sequences"]}
    assert "PJ 5 - demo" in names
    assert "Sequence 04" in names
    slot_names = [s["nested_sequence"] for s in info["slots"]]
    assert "1" in slot_names
    assert len(info["slots"]) >= 10
    assert len(info.get("instance_slots") or []) >= 50


def test_template_project_slots():
    if not TEMPLATE.exists():
        return
    info = inspect_prproj(TEMPLATE)
    assert info["template_sequence"] == "PJ 5 - demo"
    assert info["slots"]
