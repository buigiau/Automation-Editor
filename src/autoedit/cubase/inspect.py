"""Read-only inventory of named sampler tracks in supported CPR templates.

This extracts embedded track names, not MIDI, tempo, effects or plugin state.
It must never be used as a binary CPR writer.
"""
from pathlib import Path
import re


def inspect_sampler_tracks(path):
    src = Path(path)
    raw = src.read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"NUND":
        raise ValueError(f"Not a supported Cubase CPR template: {src}")
    names = {}
    for match in re.finditer(rb"(?<![A-Za-z])Sampler Track ([0-9]+)\x00", raw, re.IGNORECASE):
        index = int(match[1])
        name = match[0][:-1].decode("ascii")
        if index in names and names[index] != name:
            raise ValueError(f"Ambiguous sampler names for track {index}")
        names[index] = name
    if not names:
        raise ValueError("No numbered Sampler Track names found in the Cubase template")
    return {"path": str(src.resolve()), "method": "embedded-name-inventory",
            "tracks": [{"index": i, "name": names[i]} for i in sorted(names)]}


def parse_sampler_tracks(selection):
    """Blank selects the legacy inventory; explicit ranges use numeric order."""
    if selection is None or selection == "" or selection == []:
        return None
    if isinstance(selection, str):
        if not selection.strip():
            return None
        indices = []
        for part in selection.split(","):
            match = re.fullmatch(r"\s*([0-9]+)\s*(?:-\s*([0-9]+)\s*)?", part)
            if not match:
                raise ValueError("Use sampler numbers or ranges, for example 2-13 or 1,3-5")
            start, end = int(match[1]), int(match[2] or match[1])
            if start < 1 or end < start or end - start > 10000:
                raise ValueError("Sampler ranges must contain positive, increasing track numbers")
            indices.extend(range(start, end + 1))
    elif isinstance(selection, list):
        indices = list(selection)
    else:
        raise ValueError("Sampler tracks must be a range string or a list of numbers")
    if (not indices or any(type(i) is not int or i < 1 for i in indices)
            or indices != sorted(set(indices))):
        raise ValueError("Sampler track numbers must be positive, unique and in increasing order")
    return indices


def select_sampler_tracks(inventory, selection, scene_count):
    """Opt in to a subset without renumbering or changing the CPR inventory."""
    indices = parse_sampler_tracks(selection)
    if indices is None:
        return inventory
    if len(indices) != scene_count:
        raise ValueError(f"Selected {len(indices)} sampler tracks for {scene_count} Premiere scene runs; "
                         "select exactly one track per scene")
    available = {track["index"]: track for track in inventory["tracks"]}
    missing = [i for i in indices if i not in available]
    if missing:
        raise ValueError(f"Selected sampler tracks are missing from the Cubase project: {missing}")
    return {**inventory, "available_tracks": inventory["tracks"],
            "tracks": [available[i] for i in indices], "selected_track_indices": indices}
