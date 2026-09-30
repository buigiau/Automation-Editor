"""Select and render a silent cut preview from a real analyzed source."""
import argparse
import json
from pathlib import Path

import av

from autoedit.match.matcher import match_slots_to_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis')
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--seconds', type=float, default=3)
    args = parser.parse_args()
    path = Path(args.analysis)
    info = json.loads(path.read_text())
    slots = [{'id': str(i + 1), 'required_duration_sec': args.seconds} for i in range(args.count)]
    matches = match_slots_to_video(slots, [], info, min_gap_sec=5)
    report = {'source': info['path'], 'analyzed_seconds': info['analyzed_seconds'],
              'detected_transitions': len(info['transition_times_sec']), 'cuts': matches}
    (path.parent / 'selection.json').write_text(json.dumps(report, indent=2))
    preview = path.parent / 'preview.mp4'
    with av.open(info['path']) as source, av.open(str(preview), 'w') as dest:
        stream = source.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = 4
        out = dest.add_stream('libx264', rate=stream.average_rate)
        out.width, out.height, out.pix_fmt = 960, 540, 'yuv420p'
        out.options = {'crf': '20', 'preset': 'fast'}
        origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else 0
        index = 0
        for match in matches:
            cut = match['video']
            start, end = cut['in_sec'], cut['out_sec']
            print(f'{start:.3f}-{end:.3f}: {cut["selection_metrics"]}', flush=True)
            source.seek(int((start + origin) / stream.time_base), stream=stream, backward=True)
            for frame in source.decode(stream):
                if frame.time is None:
                    continue
                t = float(frame.time) - origin
                if t < start:
                    continue
                if t >= end:
                    break
                frame = frame.reformat(width=960, height=540, format='yuv420p')
                frame.pts = index
                frame.time_base = 1 / stream.average_rate
                index += 1
                for packet in out.encode(frame):
                    dest.mux(packet)
        for packet in out.encode():
            dest.mux(packet)
    print(preview)


if __name__ == '__main__':
    main()
