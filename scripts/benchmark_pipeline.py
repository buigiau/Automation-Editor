"""Compare full/early-stop runs, including startup and final artifacts, on real inputs."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time
import uuid

from autoedit.config import load_config
from autoedit.worker import run_fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='YAML with real Premiere/Cubase, source and Voice paths')
    parser.add_argument('--output-dir', default='output/pipeline-benchmark')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--modes', nargs='+', choices=['until_filled', 'full'], default=['until_filled', 'full'])
    parser.add_argument('--repeats', type=int, default=2, help='First run has a fresh cache; following runs reuse it')
    parser.add_argument('--object-compute-type', choices=['float32', 'float16'], default='float32')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    root = Path(args.output_dir)/f'{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}'
    root.mkdir(parents=True)
    rows = []
    config = load_config(args.config)
    for mode in args.modes:
        cfg = deepcopy(config)
        cfg['job']['output_dir'] = str((root/mode).resolve())
        cfg['video'].update(analysis_mode=mode, speech_device=args.device)
        cfg['video']['characters']['device'] = args.device
        cfg['video']['object_detection'].update(device=args.device, compute_type=args.object_compute_type)
        for repeat in range(args.repeats):
            started = time.perf_counter()
            row = {'mode': mode, 'cache': 'cold' if repeat == 0 else 'warm', 'repeat': repeat+1}
            print(f'Benchmark {mode}: {row["cache"]} cache', flush=True)
            try:
                result = run_fresh(cfg, log=print)
                meta = result['plan']['video_analysis_meta']
                row.update(status='completed', analysis=meta, slots=len(result['plan']['slots']),
                           artifacts=cfg['job']['output_dir'])
            except Exception as exc:
                row.update(status='failed', error=str(exc))
            row['elapsed_sec'] = round(time.perf_counter()-started, 3)
            row['under_10_minutes'] = row['status'] == 'completed' and row['elapsed_sec'] < 600
            performance = Path(cfg['job']['output_dir'])/'performance.json'
            if performance.is_file():
                row['performance'] = json.loads(performance.read_text(encoding='utf-8'))
            rows.append(row)
            (root/'report.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Report: {(root/"report.json").resolve()}', flush=True)
    return int(any(row['status'] == 'failed' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
