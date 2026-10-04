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
from autoedit.acceleration import device_settings

CHARACTER_VERSION = 8
DEFAULTS = {
    "enabled": True, "coverage_target": .75, "min_main_fraction": .85,
    "arcface_model": "", "ccip_model": "", "arcface_cosine_distance": .35,
    "ccip_cosine_distance": .20, "max_track_gap_sec": .5,
    "embeddings_per_track": 8, "min_cluster_seconds": 3.0, "min_cluster_shots": 2,
    "device": "auto", "device_index": 0,
    "main_group": "auto",
    "embedding_model": "auto",
}


def settings(config=None):
    result = {**DEFAULTS, **(config or {})}
    device_settings(result["device"], result["device_index"])
    if not isinstance(result["enabled"], bool):
        raise ValueError("video.characters.enabled must be true or false")
    if result["main_group"] not in {"auto", "yellow_minions"}:
        raise ValueError("video.characters.main_group must be auto or yellow_minions")
    # Compatibility with the earlier GUI preset. Cast selection is automatic;
    # a detected ensemble participates in exposure ranking alongside everyone.
    result['main_group'] = 'auto'
    if result['embedding_model'] not in {'auto','arcface','ccip'}:
        raise ValueError('video.characters.embedding_model must be auto, arcface or ccip')
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
        return max(0, w * h) * (1 - .4 * min(1, distance / .7072)) * (1 + 4 * face.get("speaking", 0))
    return max(faces, key=score) if faces else None


def build_tracks(samples, transitions, duration, sample_fps, config, shot_times=None):
    """Associate one face per track per timestamp; every transition resets tracks."""
    from autoedit.video.objects import subject_family
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
        nested_links = set()
        for fi, face in enumerate(faces):
            for tid in active:
                if face.get('subject_kind','face') != tracks[tid].get('subject_kind','face'):
                    continue
                family = subject_family(face)
                previous_family = tracks[tid].get('subject_family')
                if family and previous_family and family != previous_family:
                    continue
                overlap = iou(face.get('track_box',face['face']), tracks[tid]["last_box"])
                if overlap < .25 and family and family == previous_family:
                    # The detector may alternate between a head and its body.
                    # Only associate a nested region when it encloses one
                    # compatible actor in both the old and new observations.
                    box = face.get('track_box', face['face'])
                    old = tracks[tid]['last_box']
                    x,y,w,h = box
                    ox,oy,ow,oh = old
                    intersection = max(0,min(x+w,ox+ow)-max(x,ox))*max(0,min(y+h,oy+oh)-max(y,oy))
                    outer = box if w*h >= ow*oh else old
                    bx,by,bw,bh = outer
                    def inside(b):
                        return bx <= b[0]+b[2]/2 <= bx+bw and by <= b[1]+b[3]/2 <= by+bh
                    current_count = sum(subject_family(f) == family and inside(f.get('track_box',f['face'])) for f in faces)
                    previous_count = sum(tracks[t].get('subject_family') == family and inside(tracks[t]['last_box']) for t in active)
                    if intersection/max(1,min(w*h,ow*oh)) >= .85 and current_count == previous_count == 1:
                        overlap = .25
                        nested_links.add((tid,fi))
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
                tracks.append({"id": f"track-{tid:05d}", "shot_id": shot,
                               'subject_kind':face.get('subject_kind','face'), "observations": [],
                               "intervals": [], "vectors": []})
                active.append(tid)
            track = tracks[tid]
            if (tid,fi) in nested_links:
                face.update(track_association='unambiguous-nested-head-body',
                            track_previous_box=list(track['last_box']))
                track.setdefault('nested_associations',[]).append({
                    'observation':[index,fi],'previous_box':list(track['last_box'])})
            if subject_family(face):
                track['subject_family'] = subject_family(face)
            track.update(last_time=time, last_box=face.get('track_box',face['face']))
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
    tracks = build_tracks(samples, info.get("shot_times_sec", transitions), duration, float(info["sample_fps"]), config,
                          info.get("shot_times_sec"))
    tag_ensemble_tracks(tracks, samples)
    requests = defaultdict(list)
    for track in tracks:
        if track.get('character_group'):
            continue
        observations = track["observations"]
        # Spread representatives through the track, picking the clearest in each bucket.
        for bucket in np.array_split(np.arange(len(observations)), min(config["embeddings_per_track"], len(observations))):
            choices = [observations[int(i)] for i in bucket]
            si, fi = max(choices, key=lambda p: samples[p[0]]["faces"][p[1]].get("mouth_clarity", 0)
                         + samples[p[0]]["faces"][p[1]].get("frontal_score", 0)
                         + samples[p[0]]["faces"][p[1]].get("subject_clarity", 0))
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
            conflicting_type = (valid[i].get('subject_family') and valid[j].get('subject_family')
                                and valid[i]['subject_family'] != valid[j]['subject_family'])
            if observations[i] & observations[j] or conflicting_type:
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


def tag_ensemble_tracks(tracks, samples):
    """Recognized ensemble members keep independent co-visible face tracks."""
    evidence = {'goggle-pupil-yellow-mouth', 'goggle-pupil-yellow-face'}
    for track in tracks:
        members = [samples[si]['faces'][fi] for si, fi in track['observations']]
        confirmed = sum(face.get('character_group') == 'yellow_minions' and
                        face.get('group_evidence', face.get('face_evidence')) in evidence for face in members)
        if members and confirmed / len(members) >= .85:
            track.update(character_group='yellow_minions', identity_status='clustered', identity_confidence=.85)


APPEARANCE_GROUP_EVIDENCE = {'repeated-disjoint-lookalikes-ccip', 'typed-whole-character-lookalikes-ccip'}


def appearance_ensembles(groups, samples, max_distance=.08, typed_distance=.12):
    """Aggregate verified lookalikes while retaining separate physical tracks.

    Require strongly agreeing appearance features and repeated, disjoint faces
    in the same frames. Concurrent copies count once toward screen time.
    """
    if not groups or not samples:
        return groups
    # Fixed appearance anchors come from the most observed prototype, not a
    # moving centroid that can gradually drift through unrelated creatures.
    groups = sorted(groups,key=lambda group:-exposure([interval for t in group for interval in t['intervals']]))
    centroids = []
    observations = []
    for group in groups:
        vector = np.asarray([v for t in group for v in t['vectors']],dtype=np.float32).mean(axis=0)
        vector /= max(float(np.linalg.norm(vector)),1e-8)
        centroids.append(vector)
        observations.append({si:fi for track in group for si,fi in track['observations']})
    families = [[i] for i in range(len(groups))]
    def repeated_distinct(i,j):
        types_i = {t['subject_family'] for t in groups[i] if t.get('subject_family')}
        types_j = {t['subject_family'] for t in groups[j] if t.get('subject_family')}
        if types_i and types_j and types_i != types_j:
            return False
        common = observations[i].keys() & observations[j].keys()
        return sum(iou(samples[si]['faces'][observations[i][si]]['face'],
                       samples[si]['faces'][observations[j][si]]['face']) < .05
                   for si in common) >= 3
    def typed_views(i,j):
        members=groups[i]+groups[j]
        # Whole heads and bodies have different crop distributions. Independent
        # creature type evidence plus closely agreeing appearance prototypes
        # can establish a visual ensemble across shots. Never use this weaker
        # whole-body comparison to combine human identities or unknown labels.
        return (all(t.get('subject_kind')=='object' and t.get('subject_family')=='creature' for t in members)
                and len({t['shot_id'] for t in members}) >= 2)
    for i in range(len(groups)):
        for j in range(i):
            distance = 1-float(centroids[i] @ centroids[j])
            typed = typed_views(i,j)
            limit = typed_distance if typed else max_distance
            if distance > limit or not (typed or repeated_distinct(i,j)):
                continue
            left = next(f for f in families if i in f)
            right = next(f for f in families if j in f)
            if left is right:
                continue
            combined = left+right
            if all(typed_views(a,b) for a in combined for b in combined if a != b):
                # Opposite head/body views need not resemble each other as
                # closely as each resembles the stable canonical prototype.
                anchor = min(combined)
                compatible = all(1-float(centroids[anchor] @ centroids[a]) <= typed_distance for a in combined)
            else:
                compatible = all(1-float(centroids[a] @ centroids[b]) <= max_distance for a in left for b in right)
            if not compatible:
                continue
            left.extend(right)
            families.remove(right)
    result=[]
    for number,family in enumerate(families):
        members=[track for index in family for track in groups[index]]
        if len(family) > 1:
            for track in members:
                track.update(character_group=f'appearance-ensemble-{number+1:03d}',
                             group_evidence=('typed-whole-character-lookalikes-ccip'
                                 if all(typed_views(a,b) for a in family for b in family if a != b)
                                 else 'repeated-disjoint-lookalikes-ccip'))
        result.append(members)
    return result


def rank_characters(tracks, config, threshold, appearance_samples=None):
    for track in tracks:
        if track.get('group_evidence') in APPEARANCE_GROUP_EVIDENCE:
            track.pop('character_group',None)
            track.pop('group_evidence',None)
    groups = cluster_tracks([track for track in tracks if not track.get('character_group')], threshold)
    if appearance_samples is not None:
        groups = appearance_ensembles(groups,appearance_samples)
    ensembles = defaultdict(list)
    for track in tracks:
        if track.get('character_group') and track.get('group_evidence') not in APPEARANCE_GROUP_EVIDENCE:
            ensembles[track['character_group']].append(track)
            track.update(identity_status='clustered', identity_confidence=.85)
    groups.extend(ensembles.values())
    clusters = []
    for group in groups:
        intervals = merged_intervals([part for track in group for part in track["intervals"]])
        shots = sorted({track["shot_id"] for track in group})
        clusters.append({"track_ids": [track["id"] for track in group], "shot_ids": shots,
                         "shot_count": len(shots), "duration_sec": exposure(intervals), "intervals": intervals,
                         **({'character_group':group[0]['character_group']} if group[0].get('character_group') else {})})
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


def _embedding_tracks(info, cache_dir, config, kind, progress):
    path = model_path(kind, config)
    identity = {"source": info.get("cache_identity"), "model_sha256": fingerprint(path),
                "source_kind": kind, "version": CHARACTER_VERSION,
                "device": config["device"], "device_index": config["device_index"],
                "max_track_gap_sec": config["max_track_gap_sec"], "embeddings_per_track": config["embeddings_per_track"]}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f"characters-{key}.json.gz"
    tracks = None
    execution = None
    if cache.is_file():
        try:
            data = json.loads(gzip.decompress(cache.read_bytes()))
            if data.get("identity") == identity:
                tracks = data["tracks"]
                execution = data.get("execution")
                if progress:
                    progress("Character embedding cache hit: no crop decode or model inference")
        except (OSError, ValueError, EOFError, KeyError):
            pass
    if tracks is None:
        model = EmbeddingModel(kind, config)
        if progress:
            progress(f"Character runtime: {getattr(model, 'execution', None)}")
        tracks = _extract_tracks(info, config, model, progress)
        execution = getattr(model, "execution", None)
        if progress and execution and execution.get("fallback_reason"):
            progress(f"Character runtime: CPU fallback: {execution['fallback_reason']}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_bytes(gzip.compress(json.dumps({"identity": identity, "tracks": tracks,
                                                       "execution": execution}).encode()))
        temporary.replace(cache)
    return tracks,execution,identity,path


def analyze_characters(video_info, cache_dir, config=None, progress=None):
    config = settings(config)
    if not config["enabled"]:
        return video_info
    if "sample_fps" not in video_info or any("faces" not in s for s in video_info.get("samples", [])):
        raise ValueError("Character filtering needs current multi-face analysis. Regenerate video analysis.")
    source_kind = video_info.get('source_kind','live_action')
    kind = {'arcface':'live_action','ccip':'animation'}.get(config['embedding_model'],source_kind)
    object_regions=any(f.get('subject_kind')=='object' for s in video_info['samples'] for f in s['faces'])
    if config['embedding_model']=='auto' and object_regions:
        kind='animation'
        if progress:
            progress('Whole-character regions: using CCIP appearance features independently of face landmarks')
    info = deepcopy(video_info)
    tracks,execution,identity,path = _embedding_tracks(info,cache_dir,config,kind,progress)
    threshold = config["ccip_cosine_distance" if kind == "animation" else "arcface_cosine_distance"]
    clusters, by_track, achieved = rank_characters(tracks, config, threshold,
        appearance_samples=info['samples'] if kind == 'animation' else None)
    auto_decision = None
    # Human face features can fragment stylised/CGI actors across expressions.
    # Choose the appearance model only when its measured track consistency
    # improves. Explicit model paths/policies retain the user's chosen model.
    uncertain = sum(t['identity_status'] in {'unknown','noise'} for t in tracks)
    if (config['embedding_model'] == 'auto' and kind == 'live_action'
            and not config['arcface_model'] and len(tracks) >= 8 and uncertain/len(tracks) >= .45):
        from autoedit.video.embeddings import MODEL_ROOT
        appearance_path = Path(config.get('ccip_model') or MODEL_ROOT/'ccip.onnx')
        if appearance_path.is_file():
            trial = deepcopy(video_info)
            alternate = _embedding_tracks(trial,cache_dir,config,'animation',progress)
            trial_tracks,trial_execution,trial_identity,trial_path = alternate
            trial_threshold = config['ccip_cosine_distance']
            trial_clusters,trial_by_track,trial_achieved = rank_characters(trial_tracks,config,trial_threshold,
                appearance_samples=trial['samples'])
            trial_uncertain = sum(t['identity_status'] in {'unknown','noise'} for t in trial_tracks)
            if trial_tracks and trial_uncertain/len(trial_tracks) <= uncertain/len(tracks)-.1:
                auto_decision = {'reason':'appearance-model-improves-track-consistency',
                    'arcface_uncertain_fraction':uncertain/len(tracks),
                    'ccip_uncertain_fraction':trial_uncertain/len(trial_tracks)}
                info,tracks,execution,identity,path = trial,trial_tracks,trial_execution,trial_identity,trial_path
                kind,threshold = 'animation',trial_threshold
                clusters,by_track,achieved = trial_clusters,trial_by_track,trial_achieved
                if progress:
                    progress('Character auto model: CCIP appearance features improve identity continuity '
                             f'({uncertain}/{len(trial_tracks)} -> {trial_uncertain}/{len(trial_tracks)} uncertain tracks).')
    for track in tracks:
        cluster = by_track.get(track["id"])
        nested_associations = {tuple(entry['observation']):entry['previous_box']
                               for entry in track.get('nested_associations',[])}
        for si, fi in track["observations"]:
            sample = info["samples"][si]
            sample["shot_id"] = track["shot_id"]
            face = sample["faces"][fi]
            if (si,fi) in nested_associations:
                face.update(track_association='unambiguous-nested-head-body',
                            track_previous_box=nested_associations[(si,fi)])
            face.update(track_id=track["id"], character_id=cluster["id"] if cluster else None,
                        character_rank=cluster["rank"] if cluster else None,
                        identity_confidence=track.get("identity_confidence", 0),
                        character_status=(("main" if cluster["is_main"] else "supporting")
                                          if track.get("identity_confidence", 0) >= .5 else "unknown")
                                          if cluster else track["identity_status"])
    _annotate_subjects(info, source_kind)
    info["character_analysis"] = {"version": CHARACTER_VERSION, "selection_policy":"exposure-ranked-cast",
        "model_sha256": identity["model_sha256"],
        "execution": execution,
        "model_path": str(path), "embedding_model": "ccip" if kind == "animation" else "arcface",
        'requested_embedding_model':config['embedding_model'],'auto_decision':auto_decision,
        "clustering": "complete-link-cosine-with-cooccurrence-constraints", "cosine_distance": threshold,
        "coverage_target": config["coverage_target"], "coverage_achieved": achieved,
        "min_main_fraction": config["min_main_fraction"], "clusters": clusters,
        "tracks": [{k: v for k, v in track.items() if k != "vectors"} for track in tracks]}
    if progress:
        progress(f"Characters: {len(clusters)} clusters, {sum(c['is_main'] for c in clusters)} main, "
                 f"coverage={achieved:.1%}; unknown tracks={sum(t['identity_status'] == 'unknown' for t in tracks)}")
        for cluster in clusters[:5]:
            progress(f"Cast rank {cluster['rank']}: {cluster.get('character_group',cluster['id'])}; "
                     f"observed {cluster['duration_sec']:.2f}s")
    return info


def _annotate_subjects(info, kind):
    # Measure all tracked mouths before choosing the subject.
    from autoedit.video.faces import annotate_speaking
    observations, originals = [], []
    for sample in info["samples"]:
        for face in sample["faces"]:
            observations.append({**face, "time_sec": sample["time_sec"], "shot_id": sample.get("shot_id")})
            originals.append(face)
    annotate_speaking(observations, source_kind=kind)
    for face, measured in zip(originals, observations):
        face.update(clear_face=measured["clear_face"], speaking=measured["speaking"])
    for sample in info["samples"]:
        faces = sample["faces"]
        # Exposure rank precedes speech/size. The matcher also evaluates every
        # separate face track when strict speech needs another visible actor.
        def priority(face):
            return ({'main':0,'supporting':1,'unknown':2,'noise':3}.get(face.get('character_status'),2),
                    face.get('character_rank') or 1000000, not face.get('speaking'))
        best = min(map(priority, faces), default=None)
        candidates = [face for face in faces if priority(face) == best]
        selected = prominent_face(candidates, sample["frame_width"], sample["frame_height"])
        if selected:
            # Preserve motion measured on the complete frame.
            sample.update({k:v for k,v in selected.items() if k not in {"activity_score", "mouth_activity"}})
            sample["face_visible"] = float(bool(selected.get('face_detected')))
        else:
            sample.update(character_status="unknown", character_id=None, character_rank=None, identity_confidence=0)
            sample.update(face_detected=False, face_visible=0., mouth_clarity=0., closeup_score=0.)
    annotate_speaking(info["samples"], kind)


def cut_character_metrics(samples, start, end, sample_interval, min_main_fraction):
    """Integrate prominent-subject identity over the cut; unknown is never noise."""
    durations = defaultdict(float)
    identities = defaultdict(float)
    rank_durations = defaultdict(float)
    confidence = 0.0
    for index, sample in enumerate(samples):
        following = samples[index + 1]["time_sec"] if index + 1 < len(samples) else end
        weight = max(0, min(end, following, sample["time_sec"] + sample_interval) - max(start, sample["time_sec"]))
        state = sample.get("character_status", "unknown")
        durations[state] += weight
        if sample.get("character_id"):
            identities[sample["character_id"]] += weight
            confidence += weight * sample.get("identity_confidence", 0)
            if sample.get('character_rank') is not None:
                rank_durations[sample['character_rank']] += weight
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
            "rank_fractions": {rank:seconds/length for rank,seconds in rank_durations.items()},
            "needs_review": state != "main", "reason": {"main": "main-character",
                "uncertain": "uncertain-character-identity", "supporting": "supporting-character-fallback",
                "reject": "noise-only"}[state]}
