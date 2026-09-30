from autoedit.schema import empty_plan, validate_plan, save_plan, load_plan


def test_roundtrip(tmp_path):
    plan = empty_plan()
    plan["slots"] = [
        {
            "id": "slot-01",
            "premiere": {"nested_sequence": "1", "duration_sec": 0.5},
        }
    ]
    path = tmp_path / "edit-plan.json"
    save_plan(plan, path)
    loaded = load_plan(path)
    assert loaded["version"] == 1
    assert validate_plan(loaded) == []
