"""Compare candidate/legacy/full runs, including queue wait, on real inputs."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time
import uuid
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from autoedit.config import load_config
from autoedit.worker import run_fresh


def main():
    # Restored download names can contain Unicode variation selectors.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, action='append', help='Real project YAML; repeat to test two queued projects')
    parser.add_argument('--output-dir', default='output/pipeline-benchmark')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--modes', nargs='+', choices=['until_filled', 'legacy', 'full'], default=['until_filled', 'legacy'])
    parser.add_argument('--concurrent', action='store_true', help='Launch selected projects together to verify job admission')
    parser.add_argument('--repeats', type=int, default=2, help='First run has a fresh cache; following runs reuse it')
    parser.add_argument('--object-compute-type', choices=['auto', 'float32', 'float16'], default='auto')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    root = Path(args.output_dir)/f'{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}'
    root.mkdir(parents=True)
    print(f'Benchmark directory: {root.resolve()}; create cancel.request here to stop gracefully',flush=True)
    rows = []
    configs = [load_config(path) for path in args.config]
    def benchmark(index, config, mode, repeat):
        cfg = deepcopy(config)
        cfg['job']['output_dir'] = str((root/f'project-{index+1}'/mode).resolve())
        cfg['video'].update(analysis_mode='until_filled' if mode=='legacy' else mode, speech_device=args.device)
        if mode=='legacy':
            cfg['video']['search_fps']=0
        elif mode=='until_filled' and not cfg['video'].get('search_fps'):
            cfg['video']['search_fps']=1
        cfg['video']['characters']['device'] = args.device
        cfg['video']['object_detection'].update(device=args.device, compute_type=args.object_compute_type)
        started = time.perf_counter()
        row = {'project':args.config[index], 'mode':mode, 'cache':'cold' if repeat==0 else 'warm', 'repeat':repeat+1}
        print(f'Benchmark project {index+1}, {mode}: {row["cache"]} cache', flush=True)
        cancel_event, finished = threading.Event(), threading.Event()
        def cancellation_watch():
            while not finished.wait(.2):
                if (root/'cancel.request').is_file():
                    cancel_event.set()
                    return
        watcher = threading.Thread(target=cancellation_watch,daemon=True)
        watcher.start()
        try:
            result = run_fresh(cfg, log=lambda msg:print(f'[{index+1}/{mode}] {msg}',flush=True),cancel_event=cancel_event)
            meta = result['plan']['video_analysis_meta']
            row.update(status='completed', analysis=meta, slots=len(result['plan']['slots']),
                       artifacts=cfg['job']['output_dir'])
        except Exception as exc:
            row.update(status='cancelled' if cancel_event.is_set() else 'failed', error=str(exc))
        finally:
            finished.set()
            watcher.join(timeout=2)
        row['elapsed_sec'] = round(time.perf_counter()-started, 3)
        row['under_10_minutes'] = row['status']=='completed' and row['elapsed_sec']<600
        performance=Path(cfg['job']['output_dir'])/'performance.json'
        if performance.is_file():
            row['performance']=json.loads(performance.read_text(encoding='utf-8'))
            row['processing_under_10_minutes']=(row['status']=='completed' and row['performance']['total_sec']<600)
        return row
    for mode in args.modes:
        for repeat in range(args.repeats):
            if args.concurrent:
                with ThreadPoolExecutor(max_workers=len(configs)) as pool:
                    futures=[pool.submit(benchmark,index,config,mode,repeat) for index,config in enumerate(configs)]
                    rows.extend(future.result() for future in futures)
            else:
                rows.extend(benchmark(index,config,mode,repeat) for index,config in enumerate(configs))
            (root/'report.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Report: {(root/"report.json").resolve()}', flush=True)
    return int(any(row['status'] != 'completed' for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
