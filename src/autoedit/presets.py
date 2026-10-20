"""Persistent project pairs and AI settings; run references are never persisted."""
from copy import deepcopy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / 'autoedit.settings.json'
PRESETS = {'soda_pop': 'Soda Pop', 'golden': 'Golden', 'cosita': 'Cosita'}


def defaults():
    return {'presets': {key: {'premiere_project': '', 'cubase_project': '',
            'template_sequence': 'PJ 5 - demo', 'video_track_index': 1,
            'sampler_tracks': ''} for key in PRESETS},
            'voice_directory': r'D:\Editor\Voice', 'api_key': '',
            'backend': 'gemini', 'model': 'gemini-3.8-flash', 'last_preset': 'soda_pop',
            'local_model': 'qwen3-vl:2b-instruct', 'local_url': 'http://127.0.0.1:11435', 'local_batch_frames': 12,
            'local_device': 'cpu', 'local_threads': 2, 'local_gpu_layers': 8,
            'local_analysis_mode': 'until_filled', 'local_target_seconds': 300, 'local_max_seconds': 0,
            'video_fps': 1, 'max_source_seconds': 0, 'shot_fallback': True, 'transition_tail_fallback_sec': 0,
            'final_scene_transition_fallback': False, 'shortage_policy': 'complete',
            'last_output': str(ROOT / 'output'), 'source_gap_sec': 5.0,
            'cache_dir': str(ROOT / 'output' / '.ai-cache')}


def load_settings(path=None):
    result = defaults()
    path = Path(path or SETTINGS_PATH)
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        for key in result:
            if key != 'presets' and key in saved:
                result[key] = saved[key]
        for key in PRESETS:
            result['presets'][key].update(saved.get('presets', {}).get(key, {}))
    return result


def save_settings(settings, path=None):
    # Allowlist prevents images, source videos and transient worker state leaking here.
    data = {key: deepcopy(settings[key]) for key in defaults() if key in settings}
    path = Path(path or SETTINGS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def apply_preset(config, settings, preset):
    if preset not in PRESETS:
        raise ValueError(f'Unknown preset: {preset}')
    pair = settings['presets'][preset]
    cfg = deepcopy(config)
    cfg.setdefault('premiere', {}).update(project=pair['premiere_project'],
        template_sequence=pair['template_sequence'], video_track_index=int(pair['video_track_index']))
    cfg.setdefault('cubase', {}).update(project=pair['cubase_project'], sampler_tracks=pair['sampler_tracks'])
    cfg.setdefault('audio', {})['directory'] = settings['voice_directory']
    cfg.setdefault('ai', {}).update(enabled=True, preset=preset, backend=settings.get('backend', 'gemini'), model=settings['model'],
        local_model=settings.get('local_model', 'qwen3-vl:2b-instruct'), local_url=settings.get('local_url', 'http://127.0.0.1:11435'),
        local_batch_frames=settings.get('local_batch_frames', 12),
        local_device=settings.get('local_device', 'cpu'), local_threads=settings.get('local_threads', 2),
        local_gpu_layers=settings.get('local_gpu_layers', 8),
        local_analysis_mode=settings.get('local_analysis_mode', 'until_filled'),
        local_target_seconds=settings.get('local_target_seconds', 300),
        local_max_seconds=settings.get('local_max_seconds', 0),
        api_key=settings.get('api_key') or os.environ.get('GEMINI_API_KEY', ''), cache_dir=settings['cache_dir'],
        video_fps=settings.get('video_fps',1), max_source_seconds=settings.get('max_source_seconds',0),
        transition_tail_fallback_sec=settings.get('transition_tail_fallback_sec', 0),
        final_scene_transition_fallback=settings.get('final_scene_transition_fallback', False),
        shortage_policy=settings.get('shortage_policy', 'complete'),
        shot_fallback=settings.get('shot_fallback',True))
    cfg.setdefault('video', {})['source_gap_sec'] = float(settings['source_gap_sec'])
    return cfg
