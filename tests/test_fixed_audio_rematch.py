import copy
import pytest

from autoedit.audio.rematch import rematch_fixed_slots, verify_fixed_layout
from autoedit.audio.scene_match import build_sampler_slots
from autoedit.premiere.adapter import apply_payload
from test_sampler_scenes import fixture


def saved_plan():
    scenes, slots, inventory = fixture()
    candidates = [{"path": "hi.wav", "category": "HI", "duration_sec": .4}]
    return {"project": {"premiere_project": "locked.prproj", "cubase_project": "locked.cpr"},
            "slots": slots, "intro_fill": {"in_sec": 50, "out_sec": 51},
            "template_audio_tracks": [{"start_sec": 0}],
            "cubase_slots": build_sampler_slots(scenes, slots, candidates, {}, inventory)}


def test_rematch_changes_only_samples_and_preserves_all_scene_track_bindings():
    before = saved_plan()
    before["cubase_slots"][0]["cubase"]["custom_routing"] = "Keep this output"
    snapshot = copy.deepcopy(before)
    info = {"source_speech": {"words": [{"word": "I", "start": .1, "end": .3, "prob": .95}]}}
    result = rematch_fixed_slots(before, [{"path": "ai.wav", "category": "AI", "duration_sec": .3}], info)
    assert before == snapshot
    assert result["operation_scope"] == "sample_assignment_only"
    assert result["cubase_slots"][0]["audio"]["path"] == "ai.wav"
    assert result["cubase_slots"][0]["audio_match"]["source_word"] == "I"
    assert verify_fixed_layout(before, result)["verified_tracks"] == 3
    with pytest.raises(ValueError, match="cannot modify Premiere"):
        apply_payload(result)


@pytest.mark.parametrize("field", ["video", "premiere", "cubase"])
def test_layout_check_detects_any_changed_cut_timing_or_track_settings(field):
    before = saved_plan()
    after = copy.deepcopy(before)
    after["cubase_slots"][0][field]["unexpected_change"] = 1
    with pytest.raises(ValueError, match="changed"):
        verify_fixed_layout(before, after)


def test_rematch_rejects_superseded_shifted_cuts():
    plan = saved_plan()
    plan["source_word_cut_alignment"] = [{"scene": "1", "after_sec": 1}]
    with pytest.raises(ValueError, match="original saved run"):
        rematch_fixed_slots(plan, [], {})
