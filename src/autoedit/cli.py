"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoedit.config import apply_inputs, load_config
from autoedit.premiere.adapter import write_apply_payload
from autoedit.schema import load_plan, save_plan
from autoedit.validate import require_inputs


def _print(msg: str) -> None:
    print(msg, flush=True)


def cmd_inspect(args: argparse.Namespace) -> int:
    from autoedit.premiere.prproj import inspect_prproj

    data = inspect_prproj(
        args.prproj,
        source_sequence=args.source,
        template_sequence=args.template,
        video_track_index=args.video_track,
    )
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _print(f"Wrote {args.out}")
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    _print(f"slots: {len(data.get('slots') or [])}")
    for slot in data.get("slots") or []:
        _print(
            f"  {slot['id']} nested={slot.get('nested_sequence')!r} "
            f"first={slot.get('first_start_sec')} occ={slot.get('occurrences')} "
            f"max_dur={slot.get('max_instance_duration_sec')}"
        )
    return 0


def cmd_analyze_audio(args: argparse.Namespace) -> int:
    from autoedit.audio.analyzer import analyze_audio_dir

    items = analyze_audio_dir(args.directory, pattern=args.glob, max_files=args.max_files)
    text = json.dumps(items, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        _print(f"Wrote {args.out} ({len(items)} files)")
    else:
        print(text)
    counts: dict[str, int] = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    _print("categories: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


def cmd_analyze_video(args: argparse.Namespace) -> int:
    from autoedit.video.analyzer import analyze_video

    info = analyze_video(
        args.video,
        sample_fps=args.sample_fps,
        max_seconds=args.max_seconds,
        backend=args.backend,
        source_kind=args.source_kind,
        progress=_print,
    )
    text = json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        _print(f"Wrote {args.out}")
    else:
        print(text)
    _print(f"segments={len(info.get('segments') or [])} backends={info.get('backend_counts')}")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from autoedit.gui import launch

    return launch(args.config)


def cmd_run(args: argparse.Namespace) -> int:
    from autoedit.pipeline import run_pipeline

    cfg = load_config(args.config)
    if args.source_kind:
        cfg.setdefault("video", {})["source_kind"] = args.source_kind
    if args.source_gap is not None:
        cfg.setdefault("video", {})["source_gap_sec"] = args.source_gap
    if args.main_group:
        cfg.setdefault("video", {}).setdefault("characters", {})["main_group"] = args.main_group
        if args.main_group != "auto":
            cfg["video"]["characters"]["enabled"] = True
    if args.sampler_tracks is not None:
        cfg.setdefault("cubase", {})["sampler_tracks"] = args.sampler_tracks
    apply_inputs(
        cfg,
        premiere=args.premiere,
        cubase=args.cubase,
        video=args.video,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
    )
    prem = cfg.get("premiere") or {}
    cubase = cfg.get("cubase") or {}
    audio = cfg.get("audio") or {}
    require_inputs(
        premiere_project=prem.get("project") or "",
        cubase_project=cubase.get("project") or "",
        source_video=prem.get("source_media") or "",
        audio_directory=audio.get("directory") or "",
        template_sequence=prem.get("template_sequence") or "PJ 5 - demo",
        video_track_index=int(prem.get("video_track_index", 1)),
        sampler_tracks=cubase.get("sampler_tracks"),
    )
    result = run_pipeline(cfg, log=_print)
    plan = result["plan"]
    out_dir = Path(cfg.get("job", {}).get("output_dir") or "./output")
    write_apply_payload(plan, out_dir / "premiere_apply.json")
    _print(f"edit-plan: {result['plan_path']}")
    _print(f"mixdown ({(plan.get('mixdown') or {}).get('status', 'ready')}): {(plan.get('mixdown') or {}).get('path')}")
    if plan.get("audio_mode") == "sampler_per_scene_v1":
        _print(f"sampler samples: {len(plan['cubase_slots'])}; see {out_dir / 'sampler_mapping.csv'}")
    unmatched = [s for s in plan.get("slots") or [] if s.get("match_status") in ("unmatched", "empty")]
    _print(f"slots={len(plan.get('slots') or [])} unmatched={len(unmatched)}")
    return 0


def cmd_mix(args: argparse.Namespace) -> int:
    from autoedit.cubase.adapter import apply_cubase_plan

    plan = load_plan(args.plan)
    out = Path(args.output_dir or Path(args.plan).parent)
    result = apply_cubase_plan(plan, out)
    save_plan(plan, args.plan)
    write_apply_payload(plan, out / "premiere_apply.json")
    _print(json.dumps(result["mixdown"], indent=2))
    return 0


def cmd_cubase_import(args: argparse.Namespace) -> int:
    from autoedit.cubase.host import import_samples
    import_samples(args.plan, log=_print)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoedit", description="Premiere + Cubase mouth/expression automation")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("inspect-premiere", help="Read slots from a .prproj without opening Premiere")
    s.add_argument("prproj")
    s.add_argument("--source", default="", help="Optional source sequence name (unused by the default workflow)")
    s.add_argument("--template", default="PJ 5 - demo")
    s.add_argument("--video-track", type=int, default=1)
    s.add_argument("--out")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("analyze-audio", help="Classify a folder of short voice files")
    s.add_argument("directory")
    s.add_argument("--glob", default="*.wav")
    s.add_argument("--max-files", type=int, default=0)
    s.add_argument("--out")
    s.set_defaults(func=cmd_analyze_audio)

    s = sub.add_parser("analyze-video", help="Detect mouth/expression segments in a source video")
    s.add_argument("video")
    s.add_argument("--sample-fps", type=float, default=6)
    s.add_argument("--max-seconds", type=float, default=30)
    s.add_argument("--backend", default="auto", choices=["auto", "mediapipe", "opencv"])
    s.add_argument("--source-kind", default="live_action", choices=["live_action", "animation"])
    s.add_argument("--out")
    s.set_defaults(func=cmd_analyze_video)

    s = sub.add_parser("gui", help="Pick Premiere / Cubase / video / audio folder and run")
    s.add_argument("-c", "--config")
    s.set_defaults(func=cmd_gui)

    s = sub.add_parser("run", help="Full pipeline: inspect, analyze, match, plan, mixdown")
    s.add_argument("-c", "--config")
    s.add_argument("--premiere", help="Premiere .prproj")
    s.add_argument("--cubase", help="Cubase .cpr")
    s.add_argument("--video", action="append", help="Source video file; repeat for multiple videos")
    s.add_argument("--audio-dir", help="Folder of short voice WAV files")
    s.add_argument("--source-kind", choices=["live_action", "animation"], help="Animation uses neutral voices when no source word matches")
    s.add_argument("--source-gap", type=float, help="Minimum seconds between selected source ranges (default 5)")
    s.add_argument("--main-group", choices=["auto", "yellow_minions"], help=argparse.SUPPRESS)
    s.add_argument("--sampler-tracks", help="Sampler numbers to fill, e.g. 2-13 or 1,3-5; omitted keeps configured mapping")
    s.add_argument("-o", "--output-dir")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("cubase-mix", help="Rebuild stems + mixdown from an existing edit-plan.json")
    s.add_argument("plan")
    s.add_argument("-o", "--output-dir")
    s.set_defaults(func=cmd_mix)
    s = sub.add_parser("cubase-import", help="Load samples into manually selected Cubase 13 sampler tracks, preserving MIDI and FX")
    s.add_argument("plan")
    s.add_argument("--manual-tracks", action="store_true", help=argparse.SUPPRESS)  # Legacy flag; selection is always manual.
    s.set_defaults(func=cmd_cubase_import)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows pipes may default to a code page without Vietnamese filenames.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list:
        argv_list = ["gui"]
    parser = build_parser()
    args = parser.parse_args(argv_list)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
