"""Offline inspector for gzipped Premiere .prproj XML."""

from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

TICKS_PER_SECOND = 254016000000.0


def ticks_to_seconds(ticks: str | int | float | None) -> float | None:
    if ticks is None or ticks == "":
        return None
    try:
        return int(str(ticks).strip()) / TICKS_PER_SECOND
    except (TypeError, ValueError):
        return None


def seconds_to_ticks(seconds: float) -> int:
    return int(round(float(seconds) * TICKS_PER_SECOND))


class _Index:
    def __init__(self, root: ET.Element):
        self.root = root
        self.by_oid: dict[str, ET.Element] = {}
        self.by_uid: dict[str, ET.Element] = {}
        for el in root.iter():
            if "ObjectID" in el.attrib:
                self.by_oid[el.attrib["ObjectID"]] = el
            if "ObjectUID" in el.attrib:
                self.by_uid[el.attrib["ObjectUID"]] = el

    def resolve(self, el: ET.Element | None) -> ET.Element | None:
        if el is None:
            return None
        if "ObjectRef" in el.attrib:
            return self.by_oid.get(el.attrib["ObjectRef"])
        if "ObjectURef" in el.attrib:
            return self.by_uid.get(el.attrib["ObjectURef"])
        return el


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        return ""
    return el.text.strip()


def _open_root(path: Path) -> ET.Element:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        xml = gzip.decompress(raw)
    else:
        xml = raw
    return ET.fromstring(xml)


def _sequence_objects(idx: _Index) -> list[ET.Element]:
    seqs = []
    for el in idx.root.iter("Sequence"):
        if el.attrib.get("ObjectUID"):
            seqs.append(el)
    return seqs


def _track_items(idx: _Index, track: ET.Element) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    clipitems = None
    for el in track.iter("TrackItems"):
        clipitems = el
        break
    if clipitems is None:
        return items
    for ref in list(clipitems):
        item = idx.resolve(ref)
        if item is None:
            continue
        start_el = end_el = None
        for el in item.iter("Start"):
            start_el = el
            break
        for el in item.iter("End"):
            end_el = el
            break
        start = ticks_to_seconds(_text(start_el) or "0")
        end = ticks_to_seconds(_text(end_el))
        master = None
        seq_name = None
        media_path = None
        seq_uid = None
        source_in = 0.0
        source_out = None
        for el in item.iter("SubClip"):
            sc = idx.resolve(el)
            if sc is None:
                continue
            clip = idx.resolve(sc.find("Clip"))
            if clip is not None:
                source_in = ticks_to_seconds(clip.findtext(".//InPoint") or "0")
                source_out = ticks_to_seconds(clip.findtext(".//OutPoint"))
                source = idx.resolve(clip.find(".//Source"))
                if source is not None:
                    seq = idx.resolve(source.find(".//Sequence"))
                    if seq is not None:
                        seq_uid = seq.get("ObjectUID")
                        seq_name = seq.findtext("Name")
            for ch in sc.iter():
                obj = idx.resolve(ch) if ("ObjectRef" in ch.attrib or "ObjectURef" in ch.attrib) else ch
                if obj is None:
                    continue
                if obj.tag == "Sequence":
                    seq_name = _text(obj.find("Name")) or seq_name
                if obj.tag in ("MasterClip", "Clip"):
                    master = _text(obj.find("Name")) or master
                if obj.tag in ("ActualMediaFilePath", "FilePath") and obj.text:
                    media_path = obj.text.strip()
        items.append(
            {
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start if start is not None and end is not None else None,
                "source_in_sec": source_in,
                "source_out_sec": source_out,
                "nested_sequence_uid": seq_uid,
                "master": master,
                "nested_sequence": seq_name,
                "media_path": media_path,
            }
        )
    return items


def _video_tracks(idx: _Index, sequence: ET.Element) -> list[dict[str, Any]]:
    tracks_out: list[dict[str, Any]] = []
    tgs = sequence.find("TrackGroups")
    if tgs is None:
        return tracks_out
    for pair in list(tgs):
        second = pair.find("Second")
        tg = idx.resolve(second) if second is not None else None
        if tg is None or tg.tag != "VideoTrackGroup":
            continue
        for tr_ref in tg.iter("Track"):
            tr = idx.resolve(tr_ref)
            if tr is None or tr.tag != "VideoClipTrack":
                continue
            index = None
            for el in tr.iter("Index"):
                try:
                    index = int(el.text or "0")
                    break
                except ValueError:
                    pass
            tracks_out.append(
                {
                    "index": index if index is not None else len(tracks_out),
                    "uid": tr.attrib.get("ObjectUID"),
                    "items": _track_items(idx, tr),
                }
            )
    return tracks_out


def _audio_tracks(idx: _Index, sequence: ET.Element) -> list[dict[str, Any]]:
    tracks_out: list[dict[str, Any]] = []
    tgs = sequence.find("TrackGroups")
    if tgs is None:
        return tracks_out
    for pair in list(tgs):
        second = pair.find("Second")
        tg = idx.resolve(second) if second is not None else None
        if tg is None or tg.tag != "AudioTrackGroup":
            continue
        for tr_ref in tg.iter("Track"):
            tr = idx.resolve(tr_ref)
            if tr is None or "Audio" not in tr.tag:
                continue
            index = None
            for el in tr.iter("Index"):
                try:
                    index = int(el.text or "0")
                    break
                except ValueError:
                    pass
            tracks_out.append(
                {
                    "index": index if index is not None else len(tracks_out),
                    "uid": tr.attrib.get("ObjectUID"),
                    "items": _track_items(idx, tr),
                }
            )
    return tracks_out


def _media_paths(idx: _Index) -> list[str]:
    paths: list[str] = []
    for tag in ("ActualMediaFilePath", "FilePath"):
        for el in idx.root.iter(tag):
            t = (el.text or "").strip()
            if t and not t.isdigit() and t not in paths:
                paths.append(t)
    return paths


def nested_sequence_slots(
    template_tracks: list[dict[str, Any]],
    video_track_index: int = 1,
) -> list[dict[str, Any]]:
    """Group nested-sequence instances on a template video track into slots.

    Slot order is first timeline appearance. This matches PJ 5 - demo V2
    where nested sequences named 1..14 are the slots.
    """
    track = None
    for t in template_tracks:
        if t.get("index") == video_track_index:
            track = t
            break
    if track is None:
        return []

    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in sorted(track["items"], key=lambda x: x.get("start_sec") or 0):
        name = item.get("nested_sequence")
        key = item.get("nested_sequence_uid")
        if not name or not key:
            continue
        # Skip adjustment layers / beat audio masquerading as video
        if name.lower().startswith("adjustment"):
            continue
        if key not in groups:
            groups[key] = {
                "id": f"slot-{name}",
                "nested_sequence": name,
                "nested_sequence_uid": key,
                "required_duration_sec": 0.0,
                "first_start_sec": item.get("start_sec"),
                "last_end_sec": item.get("end_sec"),
                "max_instance_duration_sec": item.get("duration_sec") or 0.0,
                "occurrences": 0,
                "instances": [],
            }
            order.append(key)
        g = groups[key]
        g["required_duration_sec"] = max(
            g["required_duration_sec"],
            item.get("source_out_sec") or ((item.get("source_in_sec") or 0) + (item.get("duration_sec") or 0)),
        )
        g["occurrences"] += 1
        dur = item.get("duration_sec") or 0.0
        if dur > (g["max_instance_duration_sec"] or 0):
            g["max_instance_duration_sec"] = dur
        if item.get("end_sec") is not None:
            g["last_end_sec"] = max(g["last_end_sec"] or 0, item["end_sec"])
        g["instances"].append(
            {
                "start_sec": item.get("start_sec"),
                "end_sec": item.get("end_sec"),
                "duration_sec": item.get("duration_sec"),
                "source_in_sec": item.get("source_in_sec"),
                "source_out_sec": item.get("source_out_sec"),
            }
        )
    slots = [groups[n] for n in order]
    for i, slot in enumerate(slots):
        slot["index"] = i
        slot["id"] = f"slot-{i+1:02d}"
        slot["mode"] = "nested_sequence"
    return slots


def timeline_instance_slots(
    template_tracks: list[dict[str, Any]],
    video_track_index: int = 1,
) -> list[dict[str, Any]]:
    """One slot per clip instance on the template video track, timeline order."""
    track = None
    for t in template_tracks:
        if t.get("index") == video_track_index:
            track = t
            break
    if track is None and template_tracks:
        track = template_tracks[min(video_track_index, len(template_tracks) - 1)]
    if track is None:
        return []

    slots: list[dict[str, Any]] = []
    for item in track["items"]:
        name = item.get("nested_sequence") or item.get("master")
        if not name:
            continue
        if str(name).lower().startswith("adjustment"):
            continue
        i = len(slots)
        slots.append(
            {
                "index": i,
                "id": f"slot-{i+1:03d}",
                "nested_sequence": name,
                "first_start_sec": item.get("start_sec"),
                "last_end_sec": item.get("end_sec"),
                "max_instance_duration_sec": item.get("duration_sec") or 0.0,
                "occurrences": 1,
                "instances": [
                    {
                        "start_sec": item.get("start_sec"),
                        "end_sec": item.get("end_sec"),
                        "duration_sec": item.get("duration_sec"),
                    }
                ],
                "mode": "timeline_clip",
            }
        )
    return slots


def inspect_prproj(
    path: str | Path,
    source_sequence: str = "",
    template_sequence: str = "PJ 5 - demo",
    video_track_index: int = 1,
) -> dict[str, Any]:
    p = Path(path)
    root = _open_root(p)
    idx = _Index(root)
    sequences: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for seq in _sequence_objects(idx):
        name = _text(seq.find("Name"))
        sid = _text(seq.find("ID"))
        info = {
            "name": name,
            "id": sid,
            "uid": seq.attrib.get("ObjectUID"),
            "video_tracks": _video_tracks(idx, seq),
            "audio_tracks": _audio_tracks(idx, seq),
        }
        sequences.append(info)
        by_name[name] = info

    template = by_name.get(template_sequence)
    source = by_name.get(source_sequence) if source_sequence else None
    nested = nested_sequence_slots(template["video_tracks"], video_track_index) if template else []
    by_uid = {s["uid"]: s for s in sequences}
    for slot in nested:
        child = by_uid[slot["nested_sequence_uid"]]
        # Keep adjustment layers on their existing tracks. The footage lives on V1.
        footage = next((t for t in child["video_tracks"] if t["index"] == 0), None)
        content_end = max((i.get("end_sec") or 0 for i in (footage or {}).get("items", [])), default=0)
        slot["required_duration_sec"] = max(slot["required_duration_sec"], content_end)
        slot["fill_video_track_index"] = 0
    instances = timeline_instance_slots(template["video_tracks"], video_track_index) if template else []

    return {
        "path": str(p.resolve()),
        "sequences": [
            {
                "name": s["name"],
                "id": s["id"],
                "uid": s["uid"],
                "video_track_count": len(s["video_tracks"]),
                "video_item_counts": [len(t["items"]) for t in s["video_tracks"]],
            }
            for s in sequences
        ],
        "source_sequence": source_sequence if source else None,
        "template_sequence": template_sequence if template else None,
        "media_paths": _media_paths(idx),
        "slots": nested,
        "instance_slots": instances,
        "scene_slots": scene_run_slots(template["video_tracks"], video_track_index) if template else [],
        "intro_slot": intro_slot(template["video_tracks"], nested, video_track_index) if template else None,
        "template_video_tracks": template["video_tracks"] if template else [],
        "template_audio_tracks": template["audio_tracks"] if template else [],
        "source_video_tracks": source["video_tracks"] if source else [],
    }


def intro_slot(tracks, nested, video_track_index=1):
    """The empty lead-in belongs to video only, never to the numbered scenes.

    Do not guess how to replace existing intro graphics/effects. An occupied
    lead-in is reported for review instead of silently overwriting it.
    """
    if not nested:
        return None
    end = min(float(s["first_start_sec"]) for s in nested)
    if end <= 1e-6:
        return None
    occupied = [(t["index"], item) for t in tracks for item in t["items"]
                if item["start_sec"] < end - 1e-6 and item["end_sec"] > 0]
    if occupied:
        return {"status": "occupied", "start_sec": 0, "end_sec": end,
                "reason": "Existing intro content is preserved; automatic gap fill is not applicable."}
    return {"id": "intro", "mode": "intro_gap", "status": "empty",
            "start_sec": 0.0, "end_sec": end, "required_duration_sec": end,
            "video_track_index": video_track_index, "audio_track_index": -1}


def scene_run_slots(template_tracks, video_track_index=1):
    """One audio scene per contiguous run of one nested UID, including later reprises."""
    track = next((t for t in template_tracks if t.get("index") == video_track_index), {})
    items = sorted((x for x in track.get("items", []) if x.get("nested_sequence_uid")),
                   key=lambda x: x["start_sec"])
    runs = []
    for item in items:
        start, end = item["start_sec"], item["end_sec"]
        if end <= start:
            raise ValueError("Invalid Premiere scene timing")
        if runs and start < runs[-1]["last_end_sec"] - 1e-6:
            raise ValueError("Overlapping Premiere scene fragments cannot map to ordered sampler tracks")
        if (not runs or runs[-1]["nested_sequence_uid"] != item["nested_sequence_uid"]
                or abs(start - runs[-1]["last_end_sec"]) > 1e-6):
            runs.append({"id": f"scene-{len(runs)+1:02d}", "index": len(runs),
                         "nested_sequence": item["nested_sequence"],
                         "nested_sequence_uid": item["nested_sequence_uid"],
                         "first_start_sec": start, "last_end_sec": end, "instances": [],
                         "mode": "scene_run"})
        runs[-1]["last_end_sec"] = end
        runs[-1]["instances"].append(dict(item))
    return runs
