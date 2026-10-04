"""Rematch samples into a new folder without changing existing cuts or tracks."""
import argparse
import json
from pathlib import Path

from autoedit.audio.analyzer import analyze_file
from autoedit.audio.phonetics import write_catalog
from autoedit.audio.review import write_audio_review
from autoedit.audio.rematch import rematch_fixed_slots, verify_fixed_layout
from autoedit.cubase.sampler import render_sampler_bundle, verify_sampler_inputs
from autoedit.schema import save_plan, validate_plan


def rebuild(source_dir, output_dir):
    src, out = Path(source_dir).resolve(), Path(output_dir).resolve()
    if src == out or (out / "edit-plan.json").exists():
        raise ValueError("Choose a new output folder; saved runs are not overwritten.")
    def read(name):
        return json.loads((src / name).read_text(encoding="utf-8"))
    plan, speech, video = read("edit-plan.json"), read("source_transcript.json"), read("video_analysis.json")
    sources = plan["project"].get("source_videos") or [{"path": plan["project"]["source_video"]}]
    for entry in sources:
        source = Path(entry["path"])
        source_speech = speech.get("sources", {}).get(entry["path"], speech)
        identity = source_speech["cache_identity"]
        stat = source.stat()
        if (Path(identity["path"]).resolve() != source.resolve() or identity["size"] != stat.st_size
                or identity["mtime_ns"] != stat.st_mtime_ns):
            raise ValueError("Source video changed; run the full pipeline again.")
    out.mkdir(parents=True, exist_ok=True)
    # Refresh actual WAV analysis, so edits to the voice library are measured.
    audio = [analyze_file(path) for path in dict.fromkeys(item["path"] for item in read("audio_analysis.json"))]
    lips = read("lip_analysis.json")
    lips.update(source_kind=plan.get("video_analysis_meta", {}).get("source_kind", "live_action"), source_speech=speech)
    inspect, inventory = read("premiere_inspect.json"), read("cubase_inspect.json")
    original = plan
    previous = original["cubase_slots"]
    plan = rematch_fixed_slots(original, audio, lips)
    result = render_sampler_bundle(plan, out)
    plan["cubase_manifest"] = result["manifest"]
    validation = verify_sampler_inputs(plan["cubase_slots"])
    preservation = verify_fixed_layout(original, plan)
    write_audio_review(plan, out, speech)
    errors = validate_plan(plan)
    if errors:
        raise ValueError("Invalid rebuilt plan: " + "; ".join(errors))
    plan["notes"].append("Audio rematched from saved run; Cubase playback has not been verified for this new bundle.")
    save_plan(plan, out / "edit-plan.json")
    write_catalog(audio, out / "voice_catalog.csv")
    for name, value in (("source_transcript.json", speech), ("video_analysis.json", video),
                        ("lip_analysis.json", lips), ("audio_analysis.json", audio),
                        ("premiere_inspect.json", inspect), ("cubase_inspect.json", inventory)):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"source_run": str(src), "cut_changes": [], "sample_checks": validation,
              "layout_preservation": preservation,
              "rendered_audio_verified": False, "tracks": []}
    for old, new in zip(previous, plan["cubase_slots"]):
        row = {"track": new["cubase"]["track_name"], "before": Path(old["audio"]["path"]).name,
               "after": Path(new["audio"]["path"]).name, "source_word": new["audio_match"].get("source_word"),
               "method": new["audio_match"]["method"], "needs_review": new["audio_match"]["needs_review"]}
        report["tracks"].append(row)
        print(row)
    (out / "repair_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote " + str(out / "edit-plan.json"))
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir")
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()
    rebuild(args.source_dir, args.output_dir)
