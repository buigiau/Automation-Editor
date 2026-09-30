"""Filename phonetics and voice catalog derived from the curated taxonomy."""
import csv
from pathlib import Path
from autoedit.audio.taxonomy import lookup


def filename_phonetics(stem):
    row = lookup(stem)
    sound = row[1] if row else None
    return {"label": stem, "action": sound.action if sound else "UNCLEAR",
            "visemes": list(sound.visemes) if sound else [],
            "source": "user-filename" if sound else "unknown",
            "needs_review": sound.review if sound else True}


def write_catalog(items, destination):
    fields = ["name", "group", "category", "action", "visemes", "active_start_sec", "active_end_sec", "duration_sec", "needs_review"]
    with Path(destination).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in items:
            phon = item.get("phonetics") or {}
            writer.writerow({"name": item.get("name") or Path(item["path"]).name,
                "group": item.get("group"), "category": item.get("category"), "action": phon.get("action"),
                "visemes": " ".join(phon.get("visemes", [])), "needs_review": phon.get("needs_review", True),
                **{key: item.get(key) for key in ("active_start_sec", "active_end_sec", "duration_sec")}})
