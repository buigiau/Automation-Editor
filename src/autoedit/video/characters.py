"""Shot-local face tracks, conservative identity clusters and automatic cast ranking."""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np

from autoedit.video.embeddings import EmbeddingModel, fingerprint, model_path

CHARACTER_VERSION = 1
DEFAULTS = {
    "enabled": True, "coverage_target": .75, "min_main_fraction": .85,
    "arcface_model": "", "ccip_model": "", "arcface_cosine_distance": .35,
    "ccip_cosine_distance": .20, "max_track_gap_sec": .5,
    "embeddings_per_track": 8, "min_cluster_seconds": 3.0, "min_cluster_shots": 2,
}


def settings(config=None):
    result = {**DEFAULTS, **(config or {})}
    if not isinstance(result["enabled"], bool):
        raise ValueError("video.characters.enabled must be true or false")
    for key in ("coverage_target", "min_main_fraction"):
        value = float(result[key])
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"video.characters.{key} must be in (0, 1]")
        result[key] = value
    for key in ("arcface_cosine_distance", "ccip_cosine_distance"):
        value = float(result[key])
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"video.characters.{key} must be in (0, 1)")
        result[key] = value
    for key in ("max_track_gap_sec", "min_cluster_seconds"):
        value = float(result[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"video.characters.{key} must be positive")
        result[key] = value
    for key in ("embeddings_per_track", "min_cluster_shots"):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or value < 1:
            raise ValueError(f"video.characters.{key} must be a positive integer")
        result[key] = int(value)
    return result


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return overlap / max(1, aw * ah + bw * bh - overlap)


def prominent_face(faces, width, height):
    def score(face):
        x, y, w, h = face["face"]
        distance = math.hypot((x + w / 2) / width - .5, (y + h / 2) / height - .5)
        return max(0, w * h) * (1 - .4 * min(1, distance / .7072))
    return max(faces, key=score) if faces else None


def build_tracks(samples, transitions, duration, sample_fps, config, shot_times=None):
    """Associate one face per track per timestamp; every transition resets tracks."""
    tracks, active = [], []
    interval = 1 / sample_fps
    previous_shot = None
    shot_times = transitions if shot_times is None else shot_times
    for index, sample in enumerate(samples):
        time = sample["time_sec"]
        boundary_index = bisect_right(transitions, time)
        shot = bisect_right(shot_times, time)
        sample["shot_id"] = shot
        if boundary_index != previous_shot:
            active = []
        previous_shot = boundary_index
        active = [tid for tid in active if time - tracks[tid]["last_time"] <= config["max_track_gap_sec"]]
        faces = sample.get("faces", [])
        candidates = []
        for fi, face in enumerate(faces):
            for tid in active:
                overlap = iou(face["face"], tracks[tid]["last_box"])
                if overlap >= .25:
                    candidates.append((overlap, tid, fi))
        assigned_tracks, assigned_faces = set(), {}
        for _, tid, fi in sorted(candidates, reverse=True):
            if tid not in assigned_tracks and fi not in assigned_faces:
                assigned_tracks.add(tid)
                assigned_faces[fi] = tid
        following = samples[index + 1]["time_sec"] if index + 1 < len(samples) else duration
        boundary = transitions[boundary_index] if boundary_index < len(transitions) else duration
        end = min(following, time + interval, duration, boundary)
        for fi, face in enumerate(faces):
            tid = assigned_faces.get(fi)
            if tid is None:
                tid = len(tracks)
                tracks.append({"id": f"track-{tid:05d}", "shot_id": shot, "observations": [],
                               "intervals": [], "vectors": []})
                active.append(tid)
            track = tracks[tid]
            track.update(last_time=time, last_box=face["face"])
            track["observations"].append([index, fi])
            if end > time:
                track["intervals"].append([time, end])
            face["track_id"] = track["id"]
    return tracks


def merged_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1] + 1e-8:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def exposure(intervals):
    return sum(end - start for start, end in merged_intervals(intervals))


def _selected_source_frames(path, times):
    """Second sequential decode; materialize only requested full-resolution frames."""
    import av
    position = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0
        fps = float(stream.average_rate or 25)
        for index, frame in enumerate(container.decode(stream)):
            time = float(frame.time) - origin if frame.time is not None else index / fps
            if position >= len(times):
                break
            if time < times[position] - 1e-5:
                continue
            if abs(time - times[position]) > 1e-4:
                raise RuntimeError(f"Could not recover character crop frame at {times[position]:.3f}s")
            yield times[position], frame.to_ndarray(format="bgr24")
            position += 1
    if position != len(times):
        raise RuntimeError("Video decode ended before all character crops were extracted")


def _extract_tracks(info, config, model, progress):
    samples = info["samples"]
    transitions = sorted(set(info.get("transition_times_sec") or []))
    duration = float(info.get("analyzed_seconds") or info["duration_sec"])
    tracks = build_tracks(samples, transitions, duration, float(info["sample_fps"]), config,
                          info.get("shot_times_sec"))
    requests = defaultdict(list)
    for track in tracks:
        observations = track["observations"]
        # Spread representatives through the track, picking the clearest in each bucket.
        for bucket in np.array_split(np.arange(len(observations)), min(config["embeddings_per_track"], len(observations))):
            choices = [observations[int(i)] for i in bucket]
            si, fi = max(choices, key=lambda p: samples[p[0]]["faces"][p[1]].get("mouth_clarity", 0)
                         + samples[p[0]]["faces"][p[1]].get("frontal_score", 0))
            requests[samples[si]["time_sec"]].append((track, si, fi))
    if progress:
        progress(f"Character crops: {len(tracks)} shot-local tracks; {len(requests)} full-resolution frames")
    frames = _selected_source_frames(info["path"], sorted(requests))
    try:
        count = 0
        for time, frame in frames:
            for track, si, fi in requests[time]:
                sample = samples[si]
                vector = model.extract(frame, sample["faces"][fi], sample["frame_width"],
                                       sample["frame_height"], sample["faces"])
                if vector is not None:
                    track["vectors"].append(vector.tolist())
            count += 1
            if progress and count % 100 == 0:
                progress(f"Character embeddings: {count}/{len(requests)} crop frames")
    finally:
        frames.close()
    for track in tracks:
        track.pop("last_box", None)
        track.pop("last_time", None)
    return tracks


def cluster_tracks(tracks, threshold):
    """Complete-link agglomeration; co-visible separate tracks cannot merge."""
    valid, vectors = [], []
    for track in tracks:
        track["identity_status"] = "unknown"
        features = np.asarray(track["vectors"], dtype=np.float32)
        if len(features) < 2:
            continue
        similarities = features @ features.T
        medoid = int(np.argmax(similarities.mean(axis=1)))
        consistent = features[1 - similarities[medoid] <= threshold]
        if len(consistent) < 2 or len(consistent) / len(features) < .75:
            track["identity_status"] = "noise"
            continue
        centroid = consistent.mean(axis=0)
        centroid /= np.linalg.norm(centroid)
        track["identity_status"] = "clustered"
        track["identity_confidence"] = float(np.clip(1 - np.max(1 - consistent @ centroid) / threshold, 0, 1))
        if len(consistent) != len(features):
            track["identity_confidence"] = min(.4, track["identity_confidence"])
        valid.append(track)
        vectors.append(centroid)
    if not valid:
        return []
    if len(valid) > 5000:
        raise ValueError("More than 5000 identity tracks; split the source or limit video.max_seconds")
    vectors = np.asarray(vectors, dtype=np.float32)
    distances = np.clip(1 - vectors @ vectors.T, 0, 2)
    observations = [set(si for si, _ in track["observations"]) for track in valid]
    for i in range(len(valid)):
        distances[i, i] = np.inf
        for j in range(i):
            if observations[i] & observations[j]:
                distances[i, j] = distances[j, i] = np.inf
    groups = {i: [i] for i in range(len(valid))}
    generations = [0] * len(valid)
    queue = [(float(distances[i, j]), i, j, 0, 0) for i in range(len(valid)) for j in range(i)
             if distances[i, j] <= threshold]
    heapq.heapify(queue)
    while queue:
        distance, i, j, gi, gj = heapq.heappop(queue)
        if i not in groups or j not in groups or generations[i] != gi or generations[j] != gj:
            continue
        groups[i].extend(groups.pop(j))
        generations[i] += 1
        distances[i] = np.maximum(distances[i], distances[j])
        distances[:, i] = distances[i]
        for k in groups:
            if k != i and distances[i, k] <= threshold:
                heapq.heappush(queue, (float(distances[i, k]), i, k, generations[i], generations[k]))
    return [[valid[i] for i in indices] for indices in groups.values()]


def rank_characters(tracks, config, threshold):
    groups = cluster_tracks(tracks, threshold)
    clusters = []
    for group in groups:
        intervals = merged_intervals([part for track in group for part in track["intervals"]])
        shots = sorted({track["shot_id"] for track in group})
        clusters.append({"track_ids": [track["id"] for track in group], "shot_ids": shots,
                         "shot_count": len(shots), "duration_sec": exposure(intervals), "intervals": intervals})
    clusters.sort(key=lambda c: (-c["duration_sec"], -c["shot_count"], c["track_ids"][0]))
    total = sum(cluster["duration_sec"] for cluster in clusters)
    covered = 0.0
    for index, cluster in enumerate(clusters):
        cluster.update(id=f"character-{index + 1:03d}", rank=index + 1, is_main=False)
        eligible = cluster["shot_count"] >= config["min_cluster_shots"] or cluster["duration_sec"] >= config["min_cluster_seconds"]
        if eligible and covered < total * config["coverage_target"]:
            cluster["is_main"] = True
            covered += cluster["duration_sec"]
    by_track = {tid: cluster for cluster in clusters for tid in cluster["track_ids"]}
    return clusters, by_track, (covered / total if total else 0)


def analyze_characters(video_info, cache_dir, config=None, progress=None):
    config = settings(config)
    if not config["enabled"]:
        return video_info
    if "sample_fps" not in video_info or any("faces" not in s for s in video_info.get("samples", [])):
        raise ValueError("Character filtering needs current multi-face analysis. Regenerate video analysis.")
    kind = video_info.get("source_kind", "live_action")
    path = model_path(kind, config)
    identity = {"source": video_info.get("cache_identity"), "model_sha256": fingerprint(path),
                "source_kind": kind, "version": CHARACTER_VERSION,
                "max_track_gap_sec": config["max_track_gap_sec"], "embeddings_per_track": config["embeddings_per_track"]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f"characters-{key}.json.gz"
    info = deepcopy(video_info)
    tracks = None
    if cache.is_file():
        try:
            data = json.loads(gzip.decompress(cache.read_bytes()))
            if data.get("identity") == identity:
                tracks = data["tracks"]
                if progress:
                    progress("Character embedding cache hit: no crop decode or model inference")
        except (OSError, ValueError, EOFError, KeyError):
            pass
    if tracks is None:
        tracks = _extract_tracks(info, config, EmbeddingModel(kind, config), progress)
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_bytes(gzip.compress(json.dumps({"identity": identity, "tracks": tracks}).encode()))
        temporary.replace(cache)
    threshold = config["ccip_cosine_distance" if kind == "animation" else "arcface_cosine_distance"]
    clusters, by_track, achieved = rank_characters(tracks, config, threshold)
    for track in tracks:
        cluster = by_track.get(track["id"])
        for si, fi in track["observations"]:
            sample = info["samples"][si]
            sample["shot_id"] = track["shot_id"]
            face = sample["faces"][fi]
            face.update(track_id=track["id"], character_id=cluster["id"] if cluster else None,
                        character_rank=cluster["rank"] if cluster else None,
                        identity_confidence=track.get("identity_confidence", 0),
                        character_status=(("main" if cluster["is_main"] else "supporting")
                                          if track.get("identity_confidence", 0) >= .5 else "unknown")
                                          if cluster else track["identity_status"])
    for sample in info["samples"]:
        selected = prominent_face(sample["faces"], sample["frame_width"], sample["frame_height"])
        if selected:
            sample.update(selected)
            sample["face_visible"] = 1.0
        else:
            sample.update(character_status="unknown", character_id=None, character_rank=None, identity_confidence=0)
    from autoedit.video.faces import annotate_speaking
    annotate_speaking(info["samples"], kind)
    info["character_analysis"] = {"version": CHARACTER_VERSION, "model_sha256": identity["model_sha256"],
        "model_path": str(path), "embedding_model": "ccip" if kind == "animation" else "arcface",
        "clustering": "complete-link-cosine-with-cooccurrence-constraints", "cosine_distance": threshold,
        "coverage_target": config["coverage_target"], "coverage_achieved": achieved,
        "min_main_fraction": config["min_main_fraction"], "clusters": clusters,
        "tracks": [{k: v for k, v in track.items() if k != "vectors"} for track in tracks]}
    if progress:
        progress(f"Characters: {len(clusters)} clusters, {sum(c['is_main'] for c in clusters)} main, "
                 f"coverage={achieved:.1%}; unknown tracks={sum(t['identity_status'] == 'unknown' for t in tracks)}")
    return info


def cut_character_metrics(samples, start, end, sample_interval, min_main_fraction):
    """Integrate prominent-subject identity over the cut; unknown is never noise."""
    durations = defaultdict(float)
    identities = defaultdict(float)
    confidence = 0.0
    for index, sample in enumerate(samples):
        following = samples[index + 1]["time_sec"] if index + 1 < len(samples) else end
        weight = max(0, min(end, following, sample["time_sec"] + sample_interval) - max(start, sample["time_sec"]))
        state = sample.get("character_status", "unknown")
        durations[state] += weight
        if sample.get("character_id"):
            identities[sample["character_id"]] += weight
            confidence += weight * sample.get("identity_confidence", 0)
    length = end - start
    main = durations["main"] / length
    supporting = durations["supporting"] / length
    unknown = durations["unknown"] / length
    noise = durations["noise"] / length
    if noise >= .5 and main + supporting < .1:
        state = "reject"
    elif main >= min_main_fraction and noise <= .05 and confidence / length >= .5:
        state = "main"
    elif supporting >= min_main_fraction and main + unknown < 1 - min_main_fraction:
        state = "supporting"
    else:
        state = "uncertain"
    ranks = [s["character_rank"] for s in samples if s.get("character_rank") is not None
             and start <= s["time_sec"] < end]
    return {"decision": state, "main_fraction": main, "supporting_fraction": supporting,
            "unknown_fraction": unknown, "noise_fraction": noise,
            "character_ids": sorted(identities, key=lambda cid: (-identities[cid], cid)),
            "best_rank": min(ranks) if ranks else None, "identity_confidence": confidence / length,
            "needs_review": state != "main", "reason": {"main": "main-character",
                "uncertain": "uncertain-character-identity", "supporting": "supporting-character-fallback",
                "reject": "noise-only"}[state]}
