"""Read-only comparison of the saved Zoomally run and a manual Premiere edit."""
import argparse
import json
from pathlib import Path
import time
import sys

from autoedit.match.matcher import match_slots_to_video, _face_track_candidates
from autoedit.premiere.prproj import (
    _Index, _open_root, _sequence_objects, _video_tracks, inspect_prproj,
)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manual', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    video = json.loads((args.run/'video_analysis.json').read_text(encoding='utf-8'))
    info = json.loads((args.run/'premiere_inspect.json').read_text(encoding='utf-8'))
    manual = inspect_prproj(args.manual)
    idx = _Index(_open_root(args.manual))
    sequences = [{ 'name': s.findtext('Name'), 'uid': s.get('ObjectUID'),
                   'tracks': _video_tracks(idx, s)} for s in _sequence_objects(idx)]
    report = {'manual_project': str(args.manual), 'manual_media': manual['media_paths'],
              'manual_sequences': sequences, 'slot_durations': [s['required_duration_sec'] for s in info['slots']],
              'sources': [], 'experiments': []}
    for span in video.get('source_spans', []):
        samples = [s for s in video['samples'] if span['offset_sec'] <= s['time_sec'] < span['end_sec']]
        counts = {k: sum(bool(s.get(k)) for s in samples) for k in
                  ('face_detected', 'person_detected', 'clear_face', 'speaking')}
        max_run, current, previous = 0, 0, None
        for s in samples:
            if s.get('face_detected'):
                current = current + 1 if previous is not None and s['time_sec']-previous < .22 else 1
                max_run = max(max_run, current)
                previous = s['time_sec']
            else:
                previous, current = None, 0
        report['sources'].append({**span, 'samples': len(samples), **counts,
                                  'max_contiguous_face_samples': max_run})
    for strict in (True, False):
        candidates = _face_track_candidates(info['slots'], video, strict, 1)
        print('TRACK CANDIDATES', strict, {round(k, 3): len(v) for k, v in candidates.items()}, flush=True)
        for gap in (2, 0):
            started = time.perf_counter()
            experiment = {'require_lip_motion': strict, 'source_gap_sec': gap}
            try:
                matches = match_slots_to_video(info['slots'], [], video, min_gap_sec=gap,
                                              require_lip_motion=strict)
                experiment.update(success=True, cuts=[m['video'] for m in matches])
            except ValueError as exc:
                experiment.update(success=False, error=str(exc))
            experiment['elapsed_sec'] = round(time.perf_counter()-started, 3)
            report['experiments'].append(experiment)
            print('EXPERIMENT', strict, gap, experiment['success'], experiment.get('error', ''),
                  experiment['elapsed_sec'], flush=True)
    # This is a diagnostic ablation only. Never change production transition
    # settings or present these cuts as safe to import.
    without_transitions = {**video, 'transition_times_sec': []}
    ablation = {'require_lip_motion': True, 'source_gap_sec': 2, 'transition_filter': False}
    try:
        matches = match_slots_to_video(info['slots'], [], without_transitions,
                                      min_gap_sec=2, require_lip_motion=True)
        ablation.update(success=True, cuts=[m['video'] for m in matches])
    except ValueError as exc:
        ablation.update(success=False, error=str(exc))
    report['transition_ablation'] = ablation
    print('TRANSITION ABLATION', ablation['success'], flush=True)
    (args.out/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('MANUAL SOURCES', manual['media_paths'], flush=True)
    import cv2
    import numpy as np
    from autoedit.video.transitions import transition_metrics
    selected = [s for s in sequences if s['name'] in [str(n) for n in range(4,16)]]
    masters = {item['master'] for seq in selected for track in seq['tracks']
               if track['index'] == 0 for item in track['items']}
    source = next(span for span in video['source_spans'] if Path(span['path']).name in masters)
    cap = cv2.VideoCapture(source['path'])
    manual_cuts = []
    panels = []
    for seq in sequences:
        if seq['name'] in [str(n) for n in range(4, 16)]:
            item = next(t for t in seq['tracks'] if t['index'] == 0)['items'][0]
            slot = next(s for s in info['slots'] if s['nested_sequence'] == seq['name'])
            start = item['source_in_sec']
            # Compare the opening used by the failed template, not the longer
            # unused source handles stored in the manual nested sequences.
            end = start + slot['required_duration_sec']
            cached = [s for s in video['samples']
                      if start <= s['time_sec']-source['offset_sec'] < end]
            cut = {'sequence': seq['name'], 'start_sec': start, 'end_sec': end,
                   'stored_handle_end_sec': item['source_out_sec'], 'samples': len(cached),
                   **{k:sum(bool(s.get(k)) for s in cached) for k in ('face_detected','clear_face','speaking')},
                   **transition_metrics(video['transition_times_sec'], start+source['offset_sec'], end+source['offset_sec'])}
            manual_cuts.append(cut)
            print('MANUAL OPENING', cut, flush=True)
            row = []
            for t in (start, (start+end)/2, max(start,end-.05)):
                cap.set(cv2.CAP_PROP_POS_MSEC,t*1000)
                ok, frame = cap.read()
                if not ok:
                    frame = np.zeros((180,320,3),np.uint8)
                image = np.zeros((220,320,3),np.uint8)
                image[40:] = cv2.resize(frame,(320,180))
                nearest = min(video['samples'],key=lambda s:abs(s['time_sec']-source['offset_sec']-t))
                cv2.putText(image, f"Manual {seq['name']} | source {t:.2f}s", (5,15), cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
                flags = 'face=%d clear=%d speech=%d' % tuple(bool(nearest.get(k)) for k in ('face_detected','clear_face','speaking'))
                cv2.putText(image, flags, (5,33), cv2.FONT_HERSHEY_SIMPLEX,.4,(100,210,255),1)
                row.append(image)
            panels.append(np.hstack(row))
    cap.release()
    report['manual_openings_at_failed_template_lengths'] = manual_cuts
    for page in range(2):
        cv2.imwrite(str(args.out/f'manual-openings-{page+1}.jpg'),np.vstack(panels[page*6:(page+1)*6]))
    (args.out/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('REPORT', str(args.out/'comparison.json'))


if __name__ == '__main__':
    main()
