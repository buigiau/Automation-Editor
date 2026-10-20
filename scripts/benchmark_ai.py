"""Measure real AI autofill, then reuse its cache in another output folder."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

from autoedit.config import load_config
from autoedit.presets import apply_preset, load_settings
from autoedit.worker import run_fresh


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', required=True, choices=['soda_pop', 'golden', 'cosita'])
    parser.add_argument('--video', required=True, action='append')
    parser.add_argument('--root', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model')
    parser.add_argument('--backend', choices=['gemini', 'local'])
    parser.add_argument('--local-model')
    parser.add_argument('--local-device', choices=['cpu', 'gpu'])
    parser.add_argument('--local-analysis-mode', choices=['until_filled', 'full'])
    parser.add_argument('--local-target-seconds', type=float)
    parser.add_argument('--local-max-seconds', type=float, help='Optional hard limit; 0 searches until filled')
    parser.add_argument('--video-fps', type=float, default=1)
    parser.add_argument('--max-source-seconds', type=float, default=0)
    parser.add_argument('--reference', action='append', default=[], help='Reference appearance image; repeat for multiple looks')
    parser.add_argument('--transition-tail-fallback-sec', type=float)
    parser.add_argument('--cache-dir', help='Share upload/Voice cache across independent FPS trials')
    parser.add_argument('--cold', action='store_true', help='Require an empty cache and benchmark only a fresh first output')
    parser.add_argument('--request-timeout', type=float, default=300,
                        help='Maximum wait for one generation request, 1-1800 seconds')
    parser.add_argument('--phases', nargs='+', choices=['first', 'cached'], default=['first', 'cached'])
    args = parser.parse_args()
    root = Path(args.root).resolve()
    cache_path = Path(args.cache_dir).resolve() if args.cache_dir else root/'cache'
    if args.cold:
        if args.phases != ['first']:
            parser.error('--cold requires --phases first; cached rebuilds are not fresh benchmarks')
        if cache_path.exists() and any(cache_path.iterdir()):
            parser.error('--cold requires a new empty cache directory; existing caches are never deleted')
    root.mkdir(parents=True, exist_ok=True)
    previous = root/'report.json'
    history = json.loads(previous.read_text(encoding='utf-8')) if previous.is_file() else {}
    cfg = apply_preset(load_config(args.config), load_settings(), args.preset)
    cfg['premiere']['source_media'] = args.video
    cfg['video']['characters']['reference_images'] = args.reference
    cfg['video']['characters']['reference_regions'] = {}
    cfg['ai'].update(cache_dir=str(cache_path),
                     video_fps=args.video_fps,max_source_seconds=args.max_source_seconds)
    if args.model:
        cfg['ai']['model'] = args.model
    if args.backend:
        cfg['ai']['backend'] = args.backend
    if args.local_model:
        cfg['ai']['local_model'] = args.local_model
    if args.local_device:
        cfg['ai']['local_device'] = args.local_device
    if args.local_analysis_mode:
        cfg['ai']['local_analysis_mode'] = args.local_analysis_mode
    if args.local_target_seconds is not None:
        cfg['ai']['local_target_seconds'] = args.local_target_seconds
    if args.local_max_seconds is not None:
        cfg['ai']['local_max_seconds'] = args.local_max_seconds
    if args.transition_tail_fallback_sec is not None:
        cfg['ai']['transition_tail_fallback_sec'] = args.transition_tail_fallback_sec
    cfg['ai']['request_timeout_sec'] = args.request_timeout
    report = {'started_at': datetime.now().isoformat(), 'preset': args.preset,
              'backend': cfg['ai'].get('backend', 'gemini'),
              'model': cfg['ai'].get('local_model') if cfg['ai'].get('backend') == 'local' else cfg['ai']['model'],
              'sources': args.video, 'runs': [],
              'references': args.reference,
              'cold_start': args.cold, 'cache_dir': str(cache_path),
              'local_device': cfg['ai'].get('local_device', 'cpu'),
              'local_analysis_mode': cfg['ai'].get('local_analysis_mode', 'until_filled'),
              'local_target_seconds': cfg['ai'].get('local_target_seconds', 300),
              'local_max_seconds': cfg['ai'].get('local_max_seconds', 0),
              'request_timeout_sec': args.request_timeout,
              'video_fps': args.video_fps, 'max_source_seconds': args.max_source_seconds,
              'transition_tail_fallback_sec': cfg['ai'].get('transition_tail_fallback_sec', 0),
              'final_scene_transition_fallback': cfg['ai'].get('final_scene_transition_fallback', False),
              'previous_attempts': history.get('previous_attempts', []) + history.get('runs', []),
              'manual_review_sec': None, 'host_import_sec': None,
              'scope': 'artifact generation; host import and subjective playback not measured'}
    from autoedit.ai import appearance, discovery, single
    report['code_versions'] = {'discovery': discovery.VERSION, 'single': single.VERSION,
                               'reference_appearance': appearance.VERSION}
    print('Benchmark: '+str(root), flush=True)
    cancel, done = threading.Event(), threading.Event()
    def watch():
        while not done.wait(.2):
            if (root/'cancel.request').exists():
                cancel.set()
                return
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        for phase in args.phases:
            run = deepcopy(cfg)
            out = root/phase
            run['job']['output_dir'] = str(out)
            row = {'phase': phase, 'output': str(out)}
            started = time.perf_counter()
            print('Starting '+phase, flush=True)
            try:
                result = run_fresh(run, log=lambda message: print(message, flush=True), cancel_event=cancel)
                plan = result['plan']
                row.update(status='completed', slots=len(plan['slots']), scene_runs=len(plan['cubase_slots']),
                    analysis_scope=plan.get('ai_selection', {}).get('analysis_scope'),
                    plan_path=result['plan_path'],
                    reference_fallbacks=sum(bool(s['video'].get('reference_fallback')) for s in plan['slots']),
                    solo_scenes=sum(s['video'].get('subject_count') == 1 for s in plan['slots']),
                    group_scene_fallbacks=sum(bool(s['video'].get('group_scene_fallback')) for s in plan['slots']),
                    unresolved_identities=sum(s['video'].get('identity_sample_consistent') is False for s in plan['slots']),
                    uncertain_audio=sum(bool(s['video'].get('audio_uncertain')) for s in plan['slots']))
            except Exception as exc:
                row.update(status='cancelled' if cancel.is_set() else 'failed', error=str(exc))
                print(row['error'], flush=True)
            row['wall_sec'] = round(time.perf_counter()-started, 3)
            for name in ('performance', 'ai-validation'):
                path = out/(name+'.json')
                if path.is_file():
                    row[name] = json.loads(path.read_text(encoding='utf-8'))
            report['runs'].append(row)
            (root/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k,v in row.items() if k not in ('performance', 'ai-validation')}, ensure_ascii=False), flush=True)
            if row['status'] != 'completed':
                return 1
    finally:
        done.set()
        watcher.join(timeout=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
