"""Read-only inventory of named sampler tracks in supported CPR templates.

This extracts embedded track names, not MIDI, tempo, effects or plugin state.
It must never be used as a binary CPR writer.
"""
from pathlib import Path
import re
import struct


def _ordered_sampler_nodes(raw):
    """Read track-node names, excluding instrument/automation string copies.

    This narrow CPR node envelope is also used by the (D) template. Unknown
    envelopes do not enable duplicate mapping; numbered-name inventory still
    follows the legacy reader below.
    """
    tracks = []
    pattern = rb"\x80\x00\x00\xbf.{4}(.{4})([^\x00]{1,256})\x00\xef\xbb\xbf"
    for node in re.finditer(pattern, raw, re.DOTALL):
        match = re.fullmatch(rb"Sampler Track ([0-9]+)( \(D\))?", node[2], re.IGNORECASE)
        if not match or struct.unpack(">I", node[1])[0] != len(node[2]) + 4:
            continue
        track = {"index": int(match[1]), "name": node[2].decode("ascii")}
        if match[2]:
            track["is_duplicate"] = True
        tracks.append(track)
    return tracks


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
    inventory = {"path": str(src.resolve()), "method": "embedded-name-inventory",
                 "tracks": [{"index": i, "name": names[i]} for i in sorted(names)]}
    ordered = _ordered_sampler_nodes(raw)
    if any(t.get("is_duplicate") for t in ordered):
        inventory["ordered_tracks"] = ordered
    return inventory


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


def duplicate_scene_tracks(inventory, indices, scenes):
    """Enable (D) companions only when their order exactly fits UID reprises.

    The number printed in a (D) name is a label, not a reliable source binding.
    Premiere UID identity chooses its sample; CPR track order chooses which
    companion plays that run. Every selected primary and companion must fit.
    """
    ordered = inventory.get("ordered_tracks") or []
    if (not indices or not scenes or len(indices) >= len(scenes)
            or len(indices) != len({s["nested_sequence_uid"] for s in scenes})
            or indices != list(range(indices[0], indices[-1] + 1))
            or not any(t.get("is_duplicate") for t in ordered)):
        return None
    starts = [i for i, t in enumerate(ordered)
              if not t.get("is_duplicate") and t["index"] == indices[0]]
    if len(starts) != 1:
        return None
    candidates = []
    for track in ordered[starts[0]:]:
        if not track.get("is_duplicate") and track["index"] not in indices:
            break
        candidates.append(track)
    if len(candidates) != len(scenes) or len({t["name"] for t in candidates}) != len(candidates):
        return None
    available = {t["index"]: t for t in inventory.get("available_tracks", inventory["tracks"])}
    first, primary_indices, mapped = {}, [], []
    for scene, track in zip(scenes, candidates):
        uid = scene["nested_sequence_uid"]
        if uid in first:
            if not track.get("is_duplicate"):
                return None
            original = first[uid]
            mapped.append({**track, "index": original["index"], "duplicate_of": original["name"]})
        else:
            if track.get("is_duplicate") or available.get(track["index"]) != track:
                return None
            first[uid] = track
            primary_indices.append(track["index"])
            mapped.append(dict(track))
    return mapped if primary_indices == indices else None


def select_sampler_tracks(inventory, selection, scene_count, *, scenes=None, infer_duplicates=False):
    """Opt in to a subset without renumbering or changing the CPR inventory."""
    indices = parse_sampler_tracks(selection)
    if indices is None:
        if infer_duplicates and inventory.get("ordered_tracks"):
            # AI defaults must honor native companion tracks instead of silently
            # assigning a reprise to the next numbered track. Only a unique,
            # fully validated native-order/UID mapping can be inferred.
            count = len({s["nested_sequence_uid"] for s in scenes or []})
            candidates = []
            for track in inventory["tracks"]:
                trial = list(range(track["index"], track["index"] + count))
                mapped = duplicate_scene_tracks(inventory, trial, scenes)
                if mapped is not None and len(mapped) == scene_count:
                    candidates.append(trial)
            if len(candidates) != 1:
                raise ValueError("Cannot infer a unique sampler mapping for native (D) tracks; "
                                 "set the numbered sampler range explicitly")
            return select_sampler_tracks(inventory, candidates[0], scene_count, scenes=scenes)
        return inventory
    if len(indices) != scene_count:
        mapped = None
        if scenes is not None and len(scenes) == scene_count:
            mapped = duplicate_scene_tracks(inventory, indices, scenes)
        if mapped is not None:
            return {**inventory, "available_tracks": inventory["tracks"],
                    "tracks": mapped, "selected_track_indices": indices,
                    "mapping_mode": "duplicate_scene_tracks"}
        raise ValueError(f"Selected {len(indices)} sampler tracks for {scene_count} Premiere scene runs; "
                         "select exactly one track per scene")
    available = {track["index"]: track for track in inventory["tracks"]}
    missing = [i for i in indices if i not in available]
    if missing:
        raise ValueError(f"Selected sampler tracks are missing from the Cubase project: {missing}")
    return {**inventory, "available_tracks": inventory["tracks"],
            "tracks": [available[i] for i in indices], "selected_track_indices": indices}
