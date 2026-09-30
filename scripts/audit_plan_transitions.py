"""Decode actual saved cut ranges, audit every frame, and render review evidence."""
import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np

from autoedit.video.transitions import TransitionDetector, transition_metrics


def read_cut(container, stream, start, end):
    origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0
    container.seek(int(max(0, start + origin) / stream.time_base), stream=stream, backward=True)
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        time = float(frame.time) - origin
        if time < start:
            continue
        if time >= end:
            break
        yield time, frame.reformat(width=640, height=360, format='bgr24').to_ndarray()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    parser.add_argument('--output', required=True)
    parser.add_argument('--slots', nargs='*')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    slots = [(s['premiere']['nested_sequence'], s['video']) for s in plan['slots']]
    if plan.get('intro_fill'):
        slots.append(('intro', plan['intro_fill']))
    slots = [(name, cut) for name, cut in slots if not args.slots or name in args.slots]
    rows = []
    with av.open(plan['project']['source_video']) as container, av.open(str(out / 'cuts.mp4'), 'w') as dest:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 4
        video = dest.add_stream('libx264', rate=stream.average_rate)
        video.width, video.height, video.pix_fmt = 640, 360, 'yuv420p'
        video.options = {'crf': '18', 'preset': 'fast'}
        index = 0
        for name, cut in slots:
            start, end = cut['in_sec'], cut['out_sec']
            detector = TransitionDetector()
            frames = []
            for time, image in read_cut(container, stream, max(0, start - .8), end + .8):
                detector.observe(time, cv2.resize(image, (160, 90)))
                if start <= time < end:
                    frames.append((time, image.copy()))
            events = sorted(set(detector.events))
            metrics = transition_metrics(events, start, end)
            rows.append({'slot': name, 'in_sec': start, 'out_sec': end, **metrics,
                         'frame_count': len(frames), 'events_sec': [t for t in events if start - .12 <= t <= end + .12]})
            tiles = []
            for time, image in frames:
                cv2.putText(image, f'Clip {name} | {time:.3f}s | +{time-start:.3f}s', (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                tiles.append(cv2.resize(image, (256, 144)))
                frame = av.VideoFrame.from_ndarray(image, format='bgr24')
                frame.pts, frame.time_base = index, 1 / stream.average_rate
                index += 1
                for packet in video.encode(frame):
                    dest.mux(packet)
            if not tiles:
                raise RuntimeError(f'No decoded frames in clip {name}')
            while len(tiles) % 6:
                tiles.append(np.zeros_like(tiles[0]))
            grid = np.vstack([np.hstack(tiles[i:i+6]) for i in range(0, len(tiles), 6)])
            cv2.imwrite(str(out / f'clip-{name}.jpg'), grid)
            print(name, start, end, metrics, flush=True)
        for packet in video.encode():
            dest.mux(packet)
    (out / 'audit.json').write_text(json.dumps({'plan': str(Path(args.plan).resolve()), 'cuts': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
