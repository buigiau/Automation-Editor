"""Offline artifact smoke tests against optional local paired templates (no paid API)."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import soundfile as sf
from autoedit.ai import pipeline as ai
from autoedit.ai.cache import identity, read
from autoedit.config import load_config
from autoedit.premiere.prproj import inspect_prproj

PAIRS = [
    ('soda_pop', r'D:\Editor\Soda Pop\soda pop\Soda pop_1_1.prproj', r'D:\Editor\Soda Pop\soda pop\soda pop.cpr', 16),
    ('golden', r'D:\Editor\Golden\golden_1.prproj', r'D:\Editor\Golden\golden.cpr', 14),
    ('cosita', r'D:\Editor\damtucosita\DAMTUCOSITA flip_3.prproj', r'D:\Editor\damtucosita\DAME TU COSITA new.cpr', 15),
]


@pytest.mark.parametrize('preset,premiere,cubase,scene_count', PAIRS)
def test_actual_pair_exports_one_fill_per_uid_and_voice_per_scene(tmp_path, monkeypatch, preset, premiere, cubase, scene_count):
    if not Path(premiere).is_file() or not Path(cubase).is_file():
        pytest.skip('Local template unavailable')
    before = [identity(premiere), identity(cubase)]
    inspect = inspect_prproj(premiere)
    slots = list(inspect['slots'])
    if (inspect.get('intro_slot') or {}).get('status') == 'empty':
        slots.append(inspect['intro_slot'])
    positions, end = {}, 1.
    for slot in slots:
        duration = slot['required_duration_sec']
        positions[slot['id']] = (end, end+duration)
        end += duration+1
    path = tmp_path/'source.mp4'
    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 255, (96, 128, 3), dtype=np.uint8) for _ in slots]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (128, 96))
    assert writer.isOpened()
    for i in range(int((end+1)*25)):
        index = next((j for j, (a,b) in enumerate(positions.values()) if a-.1 <= i/25 <= b+.1), 0)
        writer.write(frames[index])
    writer.release()
    voice = tmp_path/'voice'
    voice.mkdir()
    sf.write(voice/'HI.wav', np.sin(np.arange(16000)*.2)*.3, 16000)
    cfg = load_config(None)
    cfg['premiere'].update(project=premiere, source_media=str(path))
    cfg['cubase']['project'] = cubase
    cfg['audio']['directory'] = str(voice)
    cfg['video']['source_gap_sec'] = .5
    cfg['ai'].update(enabled=True, preset=preset, api_key='fixture', cache_dir=str(tmp_path/'cache'))
    cfg['job']['output_dir'] = str(tmp_path/'out')
    if preset == 'cosita':
        _, _, checked, _ = ai.preflight(cfg)
        actual = {s['id']: s for s in checked}
        assert actual['slot-03']['required_duration_sec'] == pytest.approx(5.68)
        assert actual['slot-03']['visible_duration_sec'] == pytest.approx(.56)
        assert actual['slot-13']['required_duration_sec'] == pytest.approx(3.12)
        assert actual['slot-13']['visible_duration_sec'] == pytest.approx(.68)
    class Client:
        def __init__(self, *args):
            pass
        def upload(self, *args):
            return {'text': 'offline synthetic video'}
        def generate(self, parts, schema):
            contract = json.loads(parts[0]['text'].split('REQUIREMENTS:\n')[1])
            return {'shortage_reason': '', 'characters': [{'id': 'tree', 'description': 'talking tree',
                'reference_ids': [], 'appearances': [{'source_id': 'source-1', 'start_sec': 0, 'end_sec': end}]}],
                'selections': [{'slot_id': slot['id'], 'source_id': 'source-1', 'in_sec': positions[slot['id']][0],
                    'out_sec': positions[slot['id']][1], 'shot_id': slot['id'], 'shot_start_sec': positions[slot['id']][0],
                    'shot_end_sec': positions[slot['id']][1]+.1, 'duplicate_group': '',
                    'voice_id': contract['voices'][0]['voice_id'], 'character_id': 'tree',
                    'reference_fallback': False, 'audio_uncertain': True, 'reason': 'Synthetic offline integration test'}
                    for slot in slots]}
    monkeypatch.setattr(ai, 'GeminiClient', Client)
    result = ai.run_ai_pipeline(cfg)
    plan = result['plan']
    assert len(plan['slots']) == len(inspect['slots'])
    assert len(plan['cubase_slots']) == scene_count
    if preset == 'cosita':
        sampler = plan['cubase_slots']
        assert [s['cubase']['track_name'] for s in sampler] == [
            *[f'SAMPLER TRACK {i}' for i in range(2, 15)], 'SAMPLER TRACK 2 (D)', 'SAMPLER TRACK 15']
        assert sampler[13]['cubase']['duplicate_of'] == 'SAMPLER TRACK 4'
        assert sampler[13]['cubase']['sample_path'] == sampler[2]['cubase']['sample_path']
    assert [identity(premiere), identity(cubase)] == before
    assert plan['template_audio_tracks'] == inspect['template_audio_tracks']
    assert all(Path(s['cubase']['sample_path']).is_file() for s in plan['cubase_slots'])
    assert read(tmp_path/'out/ai-validation.json')['complete']
    assert all(Path(s['video']['video_only_source']).is_file() for s in plan['slots'])
