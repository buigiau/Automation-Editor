"""Compare Gemini sampling densities on the same bounded source evidence."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

from autoedit.ai.cache import write,read
from autoedit.config import load_config
from autoedit.presets import load_settings,apply_preset
from autoedit.worker import run_fresh


def main():
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',default='golden',choices=['golden','soda_pop','cosita'])
    parser.add_argument('--video',required=True)
    parser.add_argument('--root',required=True)
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--fps',nargs='+',type=float,default=[1,3])
    parser.add_argument('--source-seconds',type=float,default=180)
    parser.add_argument('--request-timeout',type=float,default=600)
    args=parser.parse_args()
    root=Path(args.root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    config=apply_preset(load_config('config.yaml'),load_settings(),args.preset)
    config['premiere']['source_media']=[args.video]
    config['video']['characters'].update(reference_images=[],reference_regions={})
    config['ai'].update(max_source_seconds=args.source_seconds,shot_fallback=True,
        request_timeout_sec=args.request_timeout,cache_dir=str(Path(args.cache_dir).resolve()))
    report={'started_at':datetime.now().isoformat(),'model':config['ai']['model'],
        'source':args.video,'preset':args.preset,'evidence_seconds':args.source_seconds,
        'trials':[],'scope':'one fresh AI selection per FPS, shared preuploaded video and Voice cache; no manual edits'}
    cancel,done=threading.Event(),threading.Event()
    def watch():
        while not done.wait(.2):
            if (root/'cancel.request').exists():
                cancel.set()
                return
    watcher=threading.Thread(target=watch,daemon=True)
    watcher.start()
    try:
        for fps in args.fps:
            if cancel.is_set():break
            cfg=deepcopy(config)
            out=root/f'{fps:g}fps'
            cfg['ai']['video_fps']=fps
            cfg['job']['output_dir']=str(out)
            row={'fps':fps,'output':str(out),'evidence_seconds':args.source_seconds}
            start=time.perf_counter()
            print(f'=== {fps:g} FPS / {args.source_seconds:g} seconds ===',flush=True)
            try:
                result=run_fresh(cfg,log=lambda message:print(message,flush=True),cancel_event=cancel)
                plan=result['plan']
                row.update(status='completed',slots=len(plan['slots']),scene_runs=len(plan['cubase_slots']),
                    plan_path=result['plan_path'],same_shot_fallbacks=sum(bool(s['video'].get('same_shot_fallback')) for s in plan['slots']),
                    audio_uncertain=sum(bool(s['video'].get('audio_uncertain')) for s in plan['slots']))
            except Exception as exc:
                row.update(status='cancelled' if cancel.is_set() else 'failed',error=str(exc))
            row.update(wall_sec=round(time.perf_counter()-start,3),performance=read(out/'performance.json'))
            validation=read(out/'ai-validation.json')
            if validation:
                row['validation']={'accepted':len(validation.get('accepted',[])),
                    'errors':validation.get('errors',[]),'repair_count':validation.get('repair_count'),
                    'cache_hit':validation.get('cache_hit')}
            report['trials'].append(row)
            write(root/'report.json',report)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    finally:
        done.set()
        watcher.join(timeout=2)
    return int(any(row['status']!='completed' for row in report['trials']))


if __name__=='__main__':
    raise SystemExit(main())
