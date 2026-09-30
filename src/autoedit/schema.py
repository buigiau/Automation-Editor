"""edit-plan.json schema helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAN_VERSION = 1
CATEGORIES = ("A", "E", "O", "U", "LAUGH", "HAHA", "CRY", "CLOSED", "NEUTRAL")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_plan(**kwargs: Any) -> dict[str, Any]:
    plan = {
        "version": PLAN_VERSION,
        "created_at": utc_now(),
        "project": {
            "premiere_project": "",
            "source_sequence": "",
            "template_sequence": "PJ 5 - demo",
            "source_video": "",
            "source_media": "",
            "cubase_project": "",
        },
        "categories": list(CATEGORIES),
        "audio": [],
        "video_segments": [],
        "slots": [],
        "mixdown": {
            "path": "",
            "duration_sec": 0.0,
            "sample_rate": 48000,
            "channels": 2,
        },
        "notes": [],
    }
    plan.update(kwargs)
    return plan


def load_plan(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(data.get("version", 0)) != PLAN_VERSION:
        raise ValueError(f"Unsupported edit-plan version: {data.get('version')}")
    return data


def save_plan(plan: dict[str, Any], path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dest


def validate_plan(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("version") != PLAN_VERSION:
        errors.append("version must be 1")
    if "slots" not in plan or not isinstance(plan["slots"], list):
        errors.append("slots[] is required")
        return errors
    for i, slot in enumerate(plan["slots"]):
        prefix = f"slots[{i}]"
        if "id" not in slot:
            errors.append(f"{prefix}.id missing")
        premiere = slot.get("premiere") or {}
        if "duration_sec" not in premiere and "nested_sequence" not in premiere:
            errors.append(f"{prefix}.premiere needs duration_sec or nested_sequence")
    return errors
