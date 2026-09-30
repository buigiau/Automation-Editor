"""Review cached analysis against real project slots without editing Premiere."""

import argparse
import json
import time
from pathlib import Path

from autoedit.match.matcher import match_slots_to_video
from autoedit.premiere.prproj import inspect_prproj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis")
    parser.add_argument("project")
    parser.add_argument("--gap", type=float, default=10)
    parser.add_argument("--output", default="output/diagnostics")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    video = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    project = inspect_prproj(args.project, template_sequence="PJ 5 - demo", video_track_index=1)
    started = time.perf_counter()
    matches = match_slots_to_video(project["slots"], [], video, min_gap_sec=args.gap)
    elapsed = time.perf_counter() - started
    rows = [{"slot": slot["nested_sequence"], "required_duration_sec": slot["required_duration_sec"],
             **match["video"]} for slot, match in zip(project["slots"], matches)]
    report = {"source": video["path"], "project": args.project, "gap_sec": args.gap,
              "scan_elapsed_sec": video["analysis_elapsed_sec"], "selection_elapsed_sec": elapsed,
              "sample_count": video["sample_count"],
              "clear_face_samples": sum(s["clear_face"] for s in video["samples"]),
              "speaking_samples": sum(s["speaking"] for s in video["samples"]), "cuts": rows}
    (out / "selection-review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected {len(rows)} full-duration cuts in {elapsed:.2f}s; gap={args.gap:g}s")
    for row in rows:
        print(f"clip {row['slot']}: {row['in_sec']:.3f}-{row['out_sec']:.3f}s {row['selection_metrics']}")

    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video["path"])
    strips = []
    try:
        # Three actual frames per cut let a reviewer compare visible lip shapes.
        for row in rows:
            tiles = []
            for fraction in (0, 0.5, 0.95):
                timestamp = row["in_sec"] + row["used_duration_sec"] * fraction
                cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode review frame at {timestamp}s")
                tile = cv2.resize(frame, (384, 216))
                tile = cv2.copyMakeBorder(tile, 26, 0, 0, 0, cv2.BORDER_CONSTANT)
                label = f"Clip {row['slot']} | {int(timestamp // 60):02d}:{timestamp % 60:05.2f}"
                cv2.putText(tile, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                tiles.append(tile)
            strips.append(np.hstack(tiles))
    finally:
        cap.release()
    for start in range(0, len(strips), 7):
        dest = out / f"selected-faces-{start // 7 + 1}.jpg"
        cv2.imwrite(str(dest), np.vstack(strips[start:start + 7]))
        print(dest)


if __name__ == "__main__":
    main()
