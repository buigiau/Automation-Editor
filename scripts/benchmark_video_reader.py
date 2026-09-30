"""Compare old seek-per-sample decoding to sequential decoding on the same source."""
import argparse
import time

import cv2
from autoedit.video.reader import sampled_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--sample-fps", type=float, default=2)
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.video)
    count = 0
    started = time.perf_counter()
    for i in range(int(args.seconds * args.sample_fps)):
        cap.set(cv2.CAP_PROP_POS_MSEC, i / args.sample_fps * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, round(frame.shape[0] * 640 / frame.shape[1])))
        count += 1
    old = time.perf_counter() - started
    cap.release()
    started = time.perf_counter()
    new_count = sum(1 for _ in sampled_frames(args.video, args.sample_fps, args.seconds))
    new = time.perf_counter() - started
    print(f"seek-per-sample: {old:.3f}s ({count} frames)")
    print(f"sequential: {new:.3f}s ({new_count} frames), ratio={old / max(new, 0.001):.2f}x")
