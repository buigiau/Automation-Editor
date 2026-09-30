"""Neutral fallback pool from the shared Voice taxonomy."""
from autoedit.audio.taxonomy import item_group


def select_audio_for_source(items, source_kind):
    if source_kind not in {"live_action", "animation"}:
        raise ValueError("Source type must be live_action or animation.")
    if source_kind != "animation":
        return items
    selected = []
    for item in items:
        if item_group(item) == "NEUTRAL":
            selected.append({**item, "audio_style": "NEUTRAL"})
    if not selected:
        raise ValueError("Animation fallback requires neutral recordings in the Voice taxonomy.")
    return selected
