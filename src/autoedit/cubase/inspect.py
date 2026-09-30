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
    for match in re.finditer(rb"(?<![A-Za-z])Sampler Track ([0-9]+)\x00", raw):
        index = int(match[1])
        name = match[0][:-1].decode("ascii")
        if index in names and names[index] != name:
            raise ValueError(f"Ambiguous sampler names for track {index}")
        names[index] = name
    if not names:
        raise ValueError("No numbered Sampler Track names found in the Cubase template")
    return {"path": str(src.resolve()), "method": "embedded-name-inventory",
            "tracks": [{"index": i, "name": names[i]} for i in sorted(names)]}
