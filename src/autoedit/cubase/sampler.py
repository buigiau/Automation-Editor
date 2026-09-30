"""Prepare local sampler inputs and an explicitly labelled Premiere timing preview."""
import csv
import json
import hashlib
import math
from pathlib import Path

import numpy as np
import soundfile as sf

RENDER_VERSION = 2  # Cubase samples have no visual-onset silence.


def inspect_sample(path):
    """Measure the actual WAV; a valid path alone does not verify playback."""
    pcm, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError(f"Empty or invalid sampler WAV: {path}")
    energy = np.max(np.abs(pcm), axis=1)
    audible = np.flatnonzero(energy > max(1e-4, float(energy.max()) * .001))
    if not len(audible):
        raise ValueError(f"Silent sampler WAV: {path}")
    return {"duration_sec": len(pcm)/rate, "peak": float(energy.max()),
            "rms": float(np.sqrt(np.mean(pcm.astype(np.float64)**2))),
            "audible_start_sec": int(audible[0])/rate,
            "audible_end_sec": (int(audible[-1])+1)/rate}


def verify_sampler_inputs(slots):
    results = []
    for slot in slots:
        cubase = slot["cubase"]
        stats = inspect_sample(cubase["sample_path"])
        if stats["audible_start_sec"] > .02:
            raise ValueError(f"{cubase['track_name']} starts with {stats['audible_start_sec']:.3f}s silence. "
                             "Regenerate the sampler bundle before importing into Cubase.")
        results.append({"track_name": cubase["track_name"], **stats})
    return results


def render_sampler_bundle(plan, output_dir):
    from autoedit.cubase.adapter import _resample, _write_wav_stereo

    out = Path(output_dir)
    plan.pop("cubase_import", None)  # A rebuilt bundle has not been loaded yet.
    slots = plan["cubase_slots"]
    identity = [{"audio": s["audio"], "duration": s["premiere"]["duration_sec"],
                 "mtime": Path(s["audio"]["path"]).stat().st_mtime_ns} for s in slots]
    digest = hashlib.sha256(json.dumps([RENDER_VERSION, identity, plan.get("mixdown", {}).get("sample_rate") or 48000], sort_keys=True).encode()).hexdigest()[:16]
    samples_dir = out / "sampler_samples" / digest
    samples_dir.mkdir(parents=True, exist_ok=True)
    rate = int(plan.get("mixdown", {}).get("sample_rate") or 48000)
    total = max(s["premiere"]["timeline_end_sec"] for s in slots) + 0.25
    mix = np.zeros((int(np.ceil(total * rate)), 2), np.float32)
    rows = []
    for slot in slots:
        audio, prem, cubase = slot["audio"], slot["premiere"], slot["cubase"]
        decoded, src_rate = sf.read(audio["path"], dtype="float32", always_2d=True)
        lo = max(0, int(round(float(audio.get("segment_start_sec") or 0) * src_rate)))
        hi = int(round(float(audio.get("segment_end_sec") or decoded.shape[0] / src_rate) * src_rate))
        decoded = decoded[lo:min(hi, len(decoded))].mean(axis=1)
        if not len(decoded):
            raise ValueError(f"Empty selected sample for {cubase['track_name']}")
        if not np.isfinite(decoded).all():
            raise ValueError(f"Invalid selected sample for {cubase['track_name']}")
        mono = _resample(decoded, src_rate, rate)
        n = int(round(prem["duration_sec"] * rate))
        sample = np.zeros(n, np.float32)
        placement = float(audio.get("placement_offset_sec") or 0)
        if not math.isfinite(placement):
            raise ValueError(f"Invalid sample onset for {cubase['track_name']}")
        onset = int(round(placement * rate))
        if onset < 0 or onset >= n:
            raise ValueError(f"Invalid sample onset for {cubase['track_name']}")
        sample[onset:onset+min(n-onset, len(mono))] = mono[:n-onset]
        # The instrument retriggers at each MIDI note. Visual placement belongs
        # only in the timing preview, never in the instrument's sample file.
        audible = np.flatnonzero(np.abs(mono) > max(1e-4, float(np.max(np.abs(mono))) * .001))
        if not len(audible):
            raise ValueError(f"Silent selected sample for {cubase['track_name']}")
        sampler = mono[audible[0]:audible[-1]+1]
        path = samples_dir / f"Sampler_Track_{cubase['track_index']:02d}.wav"
        if not path.exists():  # Cubase may hold the previous batch open.
            _write_wav_stereo(path, np.column_stack((sampler, sampler)), rate)
        stats = inspect_sample(path)
        if stats["audible_start_sec"] > .02:
            raise ValueError(f"Invalid cached sampler onset for {cubase['track_name']}")
        cubase["sample_path"] = str(path.resolve())
        cubase["sample_validation"] = stats
        cubase["sample_render_version"] = RENDER_VERSION
        cubase["timing_mode"] = "existing-midi-triggers"
        cubase["source_onset_applied_to_sampler"] = False
        if placement > .02:
            slot["audio_match"]["needs_review"] = True
            slot["audio_match"]["timing_review"] = "Source word offset is preview-only; check MIDI trigger against the mouth."
        # This preview uses Premiere's actual resets and source offsets. It does
        # not simulate MIDI notes, sampler envelopes, pitch or Cubase effects.
        for instance in prem["instances"]:
            start, end = instance["start_sec"], instance["end_sec"]
            offset = float(instance.get("source_in_sec") or 0)
            source_end = instance.get("source_out_sec")
            source_end = float(source_end) if source_end is not None else offset + end-start
            if offset < 0 or source_end > prem["duration_sec"] + 1e-6 or source_end <= offset:
                raise ValueError(f"Invalid repeat source offsets in {slot['id']}")
            if abs(source_end - offset - (end-start)) > 0.002:
                raise ValueError(f"Retimed scene {slot['id']} requires a Cubase-rendered preview")
            a, b = int(round(start * rate)), int(round(end * rate))
            source_a = int(round(offset * rate))
            chunk = sample[source_a:source_a+b-a]
            mix[a:a+len(chunk)] += chunk[:, None]
        rows.append({"scene_id": slot["id"], "scene": prem["nested_sequence"],
                     "nested_sequence_uid": prem["nested_sequence_uid"],
                     "track_name": cubase["track_name"], "track_index": cubase["track_index"],
                     "timeline_start_sec": prem["timeline_start_sec"], "timeline_end_sec": prem["timeline_end_sec"],
                     "sample": str(path.resolve()), "sample_duration_sec": stats["duration_sec"],
                     "sample_validation": stats, "preview_duration_sec": n/rate,
                     "source": audio["path"], "segment_start_sec": lo/src_rate, "segment_end_sec": hi/src_rate,
                     "placement_offset_sec": placement,
                     "category": audio["category"], "audio_match": slot["audio_match"],
                     "repeat_count": len(prem["instances"]), "instances": prem["instances"]})
    peak = float(np.max(np.abs(mix)))
    if peak > 1:
        mix *= 0.95 / peak
    preview = out / "premiere_audio_preview.wav"
    _write_wav_stereo(preview, mix, rate)
    # The host panel's final import path is reserved for a real Cubase export.
    plan["mixdown"] = {"path": str((out / "cubase_mixdown.wav").resolve()), "duration_sec": total - 0.25,
                       "sample_rate": rate, "channels": 2, "status": "awaiting_cubase_export"}
    plan["audio_preview"] = {"path": str(preview.resolve()), "duration_sec": total,
                             "render_kind": "premiere-repeat-preview", "includes_cubase_midi_fx": False}
    instructions = [
        "Open the paired Cubase template in Cubase 13. Manually select all numbered Sampler Tracks before clicking Import Cubase.",
        "Click Import Cubase in AutoEdit, or run: autoedit cubase-import edit-plan.json. Windows display scaling must be 100% for this tested Cubase 13 import profile.",
        "The importer loads samples into the open Cubase project and verifies unchanged MIDI, sampler parameters and effects without saving; review and save manually if desired.",
        "Sampler WAVs start at the voice onset. Source-word offsets apply only to the Premiere preview; Cubase uses existing MIDI triggers.",
        "Audition against Premiere. Visual viseme/rhythm matching is an estimate; audio_match.needs_review flags uncertain choices.",
        "Export the Cubase mix to cubase_mixdown.wav in this output directory; use Import mixdown WAV in Premiere.",
        f"Export from timeline time 0 to at least {total - 0.25:.3f} seconds; Premiere imports that scene range.",
        "premiere_audio_preview.wav only follows Premiere repeat offsets; it does not contain Cubase MIDI pitch, envelopes or effects.",
    ]
    if plan.get("operation_scope") == "sample_assignment_only":
        instructions = instructions[:4] + [
            "Sample assignment only: keep Premiere unchanged; do not Apply or import a mixdown into Premiere for this operation.",
            "Cubase MIDI, track order, timing, routing, automation, sampler settings and effects must remain as configured.",
            "Open audio_review.html to compare source audio and selected voices; audition the loaded samples in Cubase.",
        ]
    manifest = {"mode": "sampler_per_scene_v1", "cubase_project": plan["project"]["cubase_project"],
                "operation_scope": plan.get("operation_scope"),
                "application_status": "samples_prepared_not_loaded_into_cubase",
                "tracks": rows, "mixdown": plan["mixdown"], "audio_preview": plan["audio_preview"],
                "instructions": instructions}
    manifest_path = out / "cubase_import.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for row in rows:
        row.update(source_word=row["audio_match"].get("source_word", ""),
                   match_method=row["audio_match"]["method"],
                   needs_review=row["audio_match"].get("needs_review", True))
    fields = ["track_name", "scene", "timeline_start_sec", "timeline_end_sec", "sample", "source", "category",
              "source_word", "match_method", "needs_review", "placement_offset_sec"]
    with (out / "sampler_mapping.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out / "SAMPLER_IMPORT.txt").write_text("\n".join(instructions), encoding="utf-8")
    return {"manifest": str(manifest_path.resolve()), "tracks": rows, "mixdown": plan["mixdown"]}
