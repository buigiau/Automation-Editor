from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from autoedit.ai import pipeline as ai
from autoedit.ai.cache import read, write
from autoedit.config import load_config
from autoedit.presets import defaults, save_settings, load_settings, apply_preset


@pytest.mark.parametrize('reused', [False, True])
def test_only_last_timeline_scene_gets_transition_permission(monkeypatch, reused):
    cfg = load_config(None)
    cfg['ai']['final_scene_transition_fallback'] = True
    slots = [{'id': 'last-by-time', 'nested_sequence_uid': 'z', 'required_duration_sec': 9.28,
        'first_start_sec': 0. if reused else 20.},
        {'id': 'last-in-array', 'nested_sequence_uid': 'a', 'required_duration_sec': 1., 'first_start_sec': 0.}]
    scenes = [{'nested_sequence_uid': uid, 'first_start_sec': start, 'last_end_sec': start+1.,
        'instances': [{'start_sec': start, 'end_sec': start+1., 'source_in_sec': 0., 'source_out_sec': 1.}]}
        for uid, start in [('a', 0.), ('z', 20.)]]
    monkeypatch.setattr(ai, 'require_inputs', lambda **k: None)
    monkeypatch.setattr(ai, 'inspect_prproj', lambda *a, **k: {'slots': slots, 'scene_slots': scenes})
    monkeypatch.setattr(ai, 'inspect_sampler_tracks', lambda *a: {})
    monkeypatch.setattr(ai, 'select_sampler_tracks', lambda *a, **k: {'tracks': [{'index': 1}, {'index': 2}]})
    _, _, checked, _ = ai.preflight(cfg)
    assert [s['id'] for s in checked if s.get('allow_scene_changes')] == ([] if reused else ['last-by-time'])


@pytest.fixture
def job(tmp_path, monkeypatch):
    cfg = load_config(None)
    for name in ('p.prproj', 'c.cpr', 'source.mp4'):
        (tmp_path / name).write_bytes(b'fixture')
    voice_dir = tmp_path / 'voices'
    voice_dir.mkdir()
    sf.write(voice_dir / 'HI.wav', np.sin(np.arange(8000)*.2)*.3, 16000)
    cfg['premiere'].update(project=str(tmp_path/'p.prproj'), source_media=str(tmp_path/'source.mp4'))
    cfg['cubase']['project'] = str(tmp_path/'c.cpr')
    cfg['audio']['directory'] = str(voice_dir)
    cfg['job']['output_dir'] = str(tmp_path/'out')
    cfg['ai'].update(enabled=True, api_key='test-secret', cache_dir=str(tmp_path/'cache'))
    slots = [{'id': 'fill-a', 'nested_sequence': 'A', 'nested_sequence_uid': 'uid-a',
        'required_duration_sec': 1, 'first_start_sec': 0, 'last_end_sec': 3, 'occurrences': 2,
        'instances': [{'start_sec': 0, 'end_sec': 1, 'source_in_sec': 0, 'source_out_sec': 1},
                      {'start_sec': 2, 'end_sec': 3, 'source_in_sec': 0, 'source_out_sec': 1}]}]
    scenes = [{'id': f'scene-{i}', 'index': i, 'nested_sequence': 'A', 'nested_sequence_uid': 'uid-a',
               'first_start_sec': start, 'last_end_sec': start+1, 'instances': [slots[0]['instances'][i]]}
              for i, start in enumerate((0, 2))]
    inspect = {'slots': slots, 'scene_slots': scenes, 'template_sequence': 'test', 'template_audio_tracks': [{'id': 'beat'}]}
    inventory = {'path': str(tmp_path/'c.cpr'), 'tracks': [{'index': i, 'name': f'Sampler Track {i:02d}'} for i in (1, 2)]}
    monkeypatch.setattr(ai, 'preflight', lambda config: (inspect, inventory, slots, 5))
    monkeypatch.setattr(ai, 'probe', lambda path: {'duration_sec': 30, 'has_audio': True})
    monkeypatch.setattr(ai, 'check_visual_duplicates', lambda rows, *args: (rows, []))
    import autoedit.video.silent_source as silent
    monkeypatch.setattr(silent, 'prepare_video_only_source', lambda path, *args, **kwargs: path)
    # Media decoding/ASR is covered separately; transport fixtures contain dummy
    # files and supply deterministic source word evidence to the shared service.
    from autoedit.audio.search import SourceMatchSession
    from autoedit.video import reader
    original_init = SourceMatchSession.__init__
    def session_init(self, *args, **kwargs):
        kwargs['scan'] = lambda **k: {'has_audio': True, 'status': 'transcribed', 'events': [],
            'words': [{'word': 'Hi', 'start': t, 'end': t+.2, 'prob': .99} for t in (1., 12.)]}
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(SourceMatchSession, '__init__', session_init)
    monkeypatch.setattr(reader, 'sampled_frames', lambda *a, **k: iter(()))
    calls = []
    class Client:
        def __init__(self, *args):
            pass
        def upload(self, path, stamp):
            calls.append(('upload', path))
            return {'fileData': {'fileUri': 'fake', 'mimeType': 'video/mp4'}}
        def generate(self, parts, schema):
            calls.append(('generate', parts))
            contract = json.loads(parts[0]['text'].split('REQUIREMENTS:\n')[1])
            return {'characters': [{'id': 'car', 'description': 'car', 'reference_ids': [],
                    'appearances': [{'source_id': 'source-1', 'start_sec': 0, 'end_sec': 30}]}],
                'selections': [{'slot_id': 'fill-a', 'source_id': 'source-1', 'in_sec': 1., 'out_sec': 2.,
                    'shot_id': 'shot-1', 'shot_start_sec': 0, 'shot_end_sec': 10, 'duplicate_group': '',
                    'voice_id': contract['voices'][0]['voice_id'], 'character_id': 'car',
                    'reference_fallback': False, 'audio_uncertain': True, 'reason': 'HI onset'}], 'shortage_reason': ''}
    monkeypatch.setattr(ai, 'GeminiClient', Client)
    return cfg, calls, Client


def test_export_repeats_templates_and_preserves_audio(job):
    cfg, calls, _ = job
    result = ai.run_ai_pipeline(cfg)
    plan = result['plan']
    assert len(plan['slots']) == 1 and len(plan['cubase_slots']) == 2
    assert plan['template_audio_tracks'] == [{'id': 'beat'}]
    assert not plan['project']['import_mixdown']
    assert plan['mixdown']['status'] == 'awaiting_cubase_export'
    assert all(s['cubase']['preserve_existing_midi'] for s in plan['cubase_slots'])
    assert len({s['audio']['voice_id'] for s in plan['cubase_slots']}) == 1
    payload = read(Path(cfg['job']['output_dir'])/'premiere_apply.json')
    assert len(payload['fill_nested_sequences']) == 1
    assert payload['fill_nested_sequences'][0]['audio_track_index'] == -1
    selection = plan['slots'][0]['video']['character_selection']
    assert selection['decision'] == 'uncertain' and selection['needs_review'] is True
    assert selection['character_ids'] == ['car'] and selection['reason'] == 'HI onset'
    assert payload['fill_nested_sequences'][0]['character_selection'] == selection
    assert all(s['video']['character_selection'] == selection for s in plan['cubase_slots'])
    for path in Path(cfg['job']['output_dir']).glob('*.json'):
        assert 'test-secret' not in path.read_text(encoding='utf-8')
    assert len([c for c in calls if c[0]=='generate']) == 1


def test_visible_prefix_exports_full_nested_fill_and_review_keeps_exact_evidence(job, monkeypatch):
    cfg, calls, client = job
    _, _, slots, _ = ai.preflight(cfg)
    slots[0].update(required_duration_sec=5.68, visible_duration_sec=1.)
    original = client.generate
    def generate(self, *args):
        result = original(self, *args)
        result['selections'][0].update(out_sec=6.68, shot_end_sec=2.2,
            timeline_visible_prefix=True, identity_end_sec=2.)
        return result
    monkeypatch.setattr(client, 'generate', generate)
    result = ai.run_ai_pipeline(cfg)
    row = result['plan']['slots'][0]['video']
    assert row['out_sec']-row['in_sec'] == pytest.approx(5.68)
    assert row['identity_end_sec']-row['in_sec'] == 1.
    out = Path(cfg['job']['output_dir'])
    payload = read(out/'premiere_apply.json')['fill_nested_sequences'][0]
    assert payload['slot_duration_sec'] == 5.68
    snapshot = read(out/'ai-review.json')
    edits = out/'edits.json'
    write(edits, {'revision': snapshot['revision'], 'edits': [{'slot_id': 'fill-a', 'in_sec': 1.04}]})
    cfg['_ai_review'] = str(edits)
    updated = ai.run_ai_pipeline(cfg)['plan']['slots'][0]['video']
    assert updated['identity_end_sec'] == pytest.approx(2.04)
    assert updated['out_sec'] == pytest.approx(6.72)
    assert len([c for c in calls if c[0] == 'generate']) == 1


def test_changing_output_reuses_ai_voice_and_upload_work(job, monkeypatch):
    cfg, calls, _ = job
    ai.run_ai_pipeline(cfg)
    cfg['job']['output_dir'] += '-second'
    import autoedit.audio.analyzer as audio
    monkeypatch.setattr(audio, 'analyze_file', lambda *a: pytest.fail('Voice cache missed'))
    ai.run_ai_pipeline(cfg)
    assert len([c for c in calls if c[0]=='generate']) == 1
    assert len([c for c in calls if c[0]=='upload']) == 1


def test_model_invalidates_result_cache(job):
    cfg, calls, _ = job
    ai.run_ai_pipeline(cfg)
    cfg['ai']['model'] = 'another-model'
    ai.run_ai_pipeline(cfg)
    assert len([c for c in calls if c[0]=='generate']) == 2


def test_local_backend_does_not_upload_or_use_gemini_and_records_provenance(job, monkeypatch):
    cfg, calls, fake = job
    original = fake.generate
    class LocalClient:
        def __init__(self, *a, **k):
            pass
        def identity(self):
            return {'model': 'qwen3-vl:4b-instruct', 'digest': 'weights-v1'}
        def unload(self):
            calls.append(('unload',))
    class Selector:
        def __init__(self, *a):
            pass
        def select(self, contract, *a, **k):
            return original(None, [{'text': 'REQUIREMENTS:\n'+json.dumps(contract)}], {})
    monkeypatch.setattr(ai, 'OllamaClient', LocalClient)
    monkeypatch.setattr(ai, 'NativeVisionClient', LocalClient)
    monkeypatch.setattr(ai, 'LocalSelector', Selector)
    monkeypatch.setattr(ai, 'GeminiClient', lambda *a: pytest.fail('Local backend used Gemini'))
    cfg['ai'].update(backend='local', api_key='')
    plan = ai.run_ai_pipeline(cfg)['plan']
    assert plan['ai_selection']['backend'] == 'local'
    assert plan['slots'][0]['video']['character_selection']['method'] == 'qwen-local-visual'
    assert plan['slots'][0]['video']['character_selection']['decision'] == 'uncertain'
    assert plan['slots'][0]['video']['character_selection']['needs_review'] is True
    assert plan['cubase_slots'][0]['audio_match']['method'] == 'neutral-source-unmatched'
    assert not any(c[0] == 'upload' for c in calls)
    assert any(c[0] == 'unload' for c in calls)
    cfg['job']['output_dir'] += '-cached'
    ai.run_ai_pipeline(cfg)
    assert len([c for c in calls if c[0] == 'generate']) == 1


def test_progressive_backend_exports_complete_bundle_and_honest_search_scope(job, monkeypatch):
    from autoedit.ai import discovery, local, visual
    cfg, calls, _ = job
    class Client:
        def __init__(self, *a, **k):
            pass
        def identity(self):
            return {'model': 'test', 'digest': 'weights'}
        def check_resources(self):
            pass
        def unload(self):
            calls.append(('unload',))
    monkeypatch.setattr(ai, 'OllamaClient', Client)
    monkeypatch.setattr(ai, 'NativeVisionClient', Client)
    monkeypatch.setattr(ai, 'GeminiClient', lambda *a: pytest.fail('Progressive backend used Gemini'))
    consumed = []
    def candidates(self, contract):
        source = self.sources[0]
        self.scan(source, 0, 30)
        anchor = source['sound_anchors'][0]
        for index in range(100):
            consumed.append(index)
            yield {'source_id': 'source-1', 'shot': index, 'shot_start_sec': index,
                'shot_end_sec': index+3., 'start_sec': anchor['start']-.08, 'end_sec': anchor['start']+1.08,
                'source_sound': anchor, 'sound_slot_id': 'fill-a',
                'image': 'not-used.jpg'}
    from autoedit.audio.search import SourceMatchSession
    monkeypatch.setattr(SourceMatchSession, 'windows', candidates)
    monkeypatch.setattr(discovery, 'classify', lambda batch, *a: {
        'cast': [{'id': 'car', 'reference_ids': []}], 'labels': [[0, 2] for _ in batch],
        'details': [{'subject_count': 1, 'sample_frames': 3, 'identity_sample_frames': 0,
                     'identity_consistent': False, 'method': 'cpu-people-2'} for _ in batch]})
    monkeypatch.setattr(local, 'transition_safe', lambda *a: True)
    monkeypatch.setattr(visual, 'fingerprints', lambda *a: [])
    monkeypatch.setattr(visual, 'similar', lambda *a: False)
    cfg['ai'].update(backend='local', local_analysis_mode='until_filled', api_key='')
    plan = ai.run_ai_pipeline(cfg)['plan']
    assert len(plan['slots']) == 1 and len(plan['cubase_slots']) == 2
    assert plan['ai_selection']['analysis_scope']['stop_reason'] == 'enough_slots'
    assert plan['video_analysis_meta']['character_ranking_scope'] == 'classified_candidate_windows'
    assert not plan['slots'][0]['video']['local_character_verified']
    assert plan['slots'][0]['video']['character_selection']['method'] == 'cpu-people-2'
    assert plan['cubase_slots'][0]['audio_match']['method'] == 'source-audio-whisper-exact-word'
    assert not plan['slots'][0]['video']['identity_sample_consistent']
    assert plan['slots'][0]['video']['character_sample_frames'] == 0
    assert len(consumed) == 1 < 100


def test_fps_is_sent_to_gemini_and_invalidates_selection_cache(job):
    cfg,calls,_=job
    ai.run_ai_pipeline(cfg)
    first_parts=next(c[1] for c in calls if c[0]=='generate')
    assert next(p['videoMetadata'] for p in first_parts if 'videoMetadata' in p)['fps']==1
    cfg['ai']['video_fps']=3
    plan=ai.run_ai_pipeline(cfg)['plan']
    requests=[c[1] for c in calls if c[0]=='generate']
    assert len(requests)==2
    assert next(p['videoMetadata'] for p in requests[1] if 'videoMetadata' in p)['fps']==3
    assert plan['ai_selection']['video_sampling']['video_fps']==3


@pytest.mark.parametrize('enabled',[True,False])
def test_same_shot_shortage_fallback_survives_cache_and_review(job,monkeypatch,enabled):
    cfg,calls,client=job
    cfg['ai']['shortage_policy'] = 'strict'
    cfg['ai']['shot_fallback']=enabled
    inspect,inventory,slots,gap=ai.preflight(cfg)
    other=deepcopy(slots[0])
    other.update(id='fill-b',nested_sequence='B',nested_sequence_uid='uid-b',
                 first_start_sec=2,last_end_sec=3,occurrences=1,instances=[slots[0]['instances'][1]])
    slots.append(other)
    inspect['scene_slots'][1].update(nested_sequence='B',nested_sequence_uid='uid-b')
    original=client.generate
    count=[]
    def generate(self,parts,schema):
        result=original(self,parts,schema)
        count.append(1)
        result['selections'][0].update(shot_start_sec=0,shot_end_sec=30,shot_id='same-shot')
        if len(count)>1:
            row=deepcopy(result['selections'][0])
            row.update(slot_id='fill-b',in_sec=12,out_sec=13)
            result['selections'].append(row)
        else:
            result['shortage_reason']='Only one continuous shot with enough distinct content'
        return result
    monkeypatch.setattr(client,'generate',generate)
    if not enabled:
        with pytest.raises(ai.SelectionIncomplete):
            ai.run_ai_pipeline(cfg)
        assert len(count)==2
        return
    plan=ai.run_ai_pipeline(cfg)['plan']
    assert plan['slots'][1]['video']['same_shot_fallback']
    assert plan['ai_selection']['deduplication_policy']=='distinct_segments'
    assert 'SHORTAGE FALLBACK AUTHORIZED' in [c[1] for c in calls if c[0]=='generate'][1][-1]['text']
    cfg['job']['output_dir']+='-cached'
    again=ai.run_ai_pipeline(cfg)['plan']
    assert again['slots'][1]['video']['same_shot_fallback'] and len(count)==2
    out=Path(cfg['job']['output_dir'])
    snapshot=read(out/'ai-review.json')
    edits=out/'edits.json'
    write(edits,{'revision':snapshot['revision'],'edits':[]})
    cfg['_ai_review']=str(edits)
    reviewed=ai.run_ai_pipeline(cfg)['plan']
    assert reviewed['slots'][1]['video']['same_shot_fallback'] and len(count)==2


def test_single_repair_and_partial_survives_without_stale_plan(job, monkeypatch):
    cfg, _, client = job
    cfg['ai']['shortage_policy'] = 'strict'
    ai.run_ai_pipeline(cfg)
    cfg['ai']['model'] = 'new-model'
    count = []
    def generate(self, parts, schema):
        count.append(1)
        return {'characters': [], 'selections': [], 'shortage_reason': 'Not enough shots',
                'analysis_scope': {'stop_reason': 'time_budget', 'candidate_count': 11, 'hard_max_seconds': 570}}
    monkeypatch.setattr(client, 'generate', generate)
    with pytest.raises(ai.SelectionIncomplete, match='Search stopped: time_budget; 11 candidates checked') as stopped:
        ai.run_ai_pipeline(cfg)
    assert 'Hard local time limit: 570s' in str(stopped.value)
    assert 'audio.source_match_search_budget_sec' in str(stopped.value)
    assert 'remain cached' in str(stopped.value)
    out = Path(cfg['job']['output_dir'])
    assert len(count) == 2
    assert (out/'ai_review.html').is_file()
    assert not (out/'edit-plan.json').exists()
    assert not (out/'premiere_apply.json').exists()
    assert read(out/'ai-validation.json')['repair_count'] == 1
    assert read(out/'ai-validation.json')['analysis_scope']['stop_reason'] == 'time_budget'


@pytest.mark.parametrize('stop_reason', ['time_budget', 'source_exhausted'])
def test_default_shortage_exports_full_draft_preserves_good_fill_and_allows_review(job, monkeypatch, stop_reason):
    from autoedit.ai import local, visual
    cfg, calls, client = job
    inspect, inventory, slots, _ = ai.preflight(cfg)
    other = deepcopy(slots[0])
    other.update(id='fill-b', nested_sequence='B', nested_sequence_uid='uid-b',
                 first_start_sec=2, last_end_sec=3, occurrences=1, instances=[slots[0]['instances'][1]])
    slots.append(other)
    inspect['scene_slots'][1].update(nested_sequence='B', nested_sequence_uid='uid-b')
    original = client.generate
    def short(self, *args):
        result = original(self, *args)
        result['analysis_scope'] = {'stop_reason': stop_reason, 'candidate_count': 64}
        return result
    monkeypatch.setattr(client, 'generate', short)
    monkeypatch.setattr(visual, 'fingerprints', lambda *a: [{'std': 20}]*8)
    monkeypatch.setattr(visual, 'similar', lambda *a: False)
    monkeypatch.setattr(local, 'transition_safe', lambda *a: True)
    plan = ai.run_ai_pipeline(cfg)['plan']
    assert len(plan['slots']) == 2 and len(plan['cubase_slots']) == 2
    assert plan['slots'][0]['video']['character_id'] == 'car'
    assert plan['slots'][0]['video']['in_sec'] == 1.
    assert plan['slots'][1]['video']['completion_fallback']
    assert plan['slots'][1]['video']['character_selection']['method'] == 'draft-completion-2-neutral-speech'
    assert plan['ai_selection']['completion']['validated_count'] == 1
    assert plan['ai_selection']['analysis_scope']['stop_reason'] == stop_reason
    out = Path(cfg['job']['output_dir'])
    assert all((out/name).is_file() for name in ('edit-plan.json', 'premiere_apply.json', 'cubase_import.json'))
    report = read(out/'ai-validation.json')
    assert report['complete'] and not report['selection_complete']
    assert len(report['selection_errors']) == 1 and len(report['accepted']) == 2
    snap = read(out/'ai-review.json')
    edits = out/'edits.json'
    write(edits, {'revision': snap['revision'], 'edits': [{'slot_id': 'fill-b', 'in_sec': 15.}]})
    cfg['_ai_review'] = str(edits)
    reviewed = ai.run_ai_pipeline(cfg)['plan']
    assert reviewed['slots'][1]['video']['in_sec'] == 15.
    assert reviewed['ai_selection']['completion']['fallback_count'] == 1
    assert len([c for c in calls if c[0] == 'generate']) == 2


def test_local_inference_failure_exports_draft_without_cloud(job, monkeypatch):
    from autoedit.ai import local, visual
    cfg, calls, _ = job
    cfg['ai'].update(backend='local')
    class Client:
        def __init__(self, *a, **k): pass
        def identity(self): return {'model': 'test'}
        def unload(self): calls.append(('unload',))
    class Selector:
        def __init__(self, *a, **k): pass
        def select(self, *a, **k): raise ai.LocalAIError('inference timed out')
    monkeypatch.setattr(ai, 'NativeVisionClient', Client)
    monkeypatch.setattr(ai, 'LocalSelector', Selector)
    monkeypatch.setattr(ai, 'GeminiClient', lambda *a: pytest.fail('Cloud fallback'))
    monkeypatch.setattr(visual, 'fingerprints', lambda *a: [{'std': 20}]*8)
    monkeypatch.setattr(visual, 'similar', lambda *a: False)
    monkeypatch.setattr(local, 'transition_safe', lambda *a: True)
    plan = ai.run_ai_pipeline(cfg)['plan']
    assert plan['ai_selection']['completion']['fallback_count'] == 1
    assert plan['ai_selection']['analysis_scope']['stop_reason'] == 'analysis_unavailable'
    assert not any(c[0] == 'upload' for c in calls)


def test_repair_succeeds_on_second_response(job, monkeypatch):
    cfg, calls, client = job
    original = client.generate
    count = []
    def generate(self, parts, schema):
        count.append(1)
        if len(count) == 1:
            return {'characters': [], 'selections': []}
        return original(self, parts, schema)
    monkeypatch.setattr(client, 'generate', generate)
    result = ai.run_ai_pipeline(cfg)
    assert result['plan']['slots'] and len(count) == 2


def test_corrections_rebuild_without_ai_and_reject_outside_shot(job):
    cfg, calls, _ = job
    ai.run_ai_pipeline(cfg)
    out = Path(cfg['job']['output_dir'])
    snap = read(out/'ai-review.json')
    snap['review_provenance'] = {'manual_visual_recovery': True}
    write(out/'ai-review.json', snap)
    edits = out/'edits.json'
    write(edits, {'revision': snap['revision'], 'edits': [{'slot_id': 'fill-a', 'in_sec': 3.}]})
    cfg['_ai_review'] = str(edits)
    result = ai.run_ai_pipeline(cfg)
    assert result['plan']['slots'][0]['video']['in_sec'] == 3.
    assert result['plan']['ai_selection']['review_provenance']['manual_visual_recovery']
    assert len([c for c in calls if c[0]=='generate']) == 1
    snap = read(out/'ai-review.json')
    write(edits, {'revision': snap['revision'], 'edits': [{'slot_id': 'fill-a', 'in_sec': 25.}]})
    with pytest.raises(ValueError, match='rejected'):
        ai.run_ai_pipeline(cfg)
    assert read(out/'edit-plan.json')['slots'][0]['video']['in_sec'] == 3.


def test_api_failure_never_runs_legacy_pipeline(job, monkeypatch):
    from itertools import count
    from types import SimpleNamespace
    cfg, _, client = job
    def generate(*args):
        raise RuntimeError('API unavailable')
    monkeypatch.setattr(client, 'generate', generate)
    monkeypatch.setattr(ai, 'time', SimpleNamespace(perf_counter=count().__next__))
    import autoedit.pipeline as pipeline
    monkeypatch.setattr(pipeline, 'analyze_video', lambda *a, **k: pytest.fail('Legacy scan'))
    with pytest.raises(RuntimeError, match='API unavailable'):
        pipeline._run_pipeline(cfg)
    assert cfg['_timings']['ai'] == 1


def test_preflight_failure_never_calls_ai(job, monkeypatch):
    cfg, calls, _ = job
    def invalid(config):
        raise ValueError('Sampler mismatch')
    monkeypatch.setattr(ai, 'preflight', invalid)
    with pytest.raises(ValueError, match='Sampler mismatch'):
        ai.run_ai_pipeline(cfg)
    assert calls == []


def test_multiple_sources_are_uploaded_and_source_local_times_preserved(job, monkeypatch):
    cfg, calls, client = job
    first = Path(cfg['premiere']['source_media'])
    second = first.with_name('second.mp4')
    second.write_bytes(b'other source')
    cfg['premiere']['source_media'] = [str(first), str(second)]
    original = client.generate
    def generate(self, *args):
        result = original(self, *args)
        result['selections'][0]['source_id'] = 'source-2'
        result['characters'][0]['appearances'][0]['source_id'] = 'source-2'
        return result
    monkeypatch.setattr(client, 'generate', generate)
    plan = ai.run_ai_pipeline(cfg)['plan']
    assert len([c for c in calls if c[0]=='upload']) == 2
    assert plan['slots'][0]['source_video'] == str(second)
    assert plan['slots'][0]['video']['in_sec'] == 1.
    assert len(plan['project']['source_videos']) == 2


def test_adding_and_removing_reference_invalidates_or_restores_cache(job, monkeypatch):
    from PIL import Image
    cfg, calls, client = job
    ai.run_ai_pipeline(cfg)
    image = Path(cfg['job']['output_dir'])/'reference.png'
    Image.new('RGB', (16, 16), 'green').save(image)
    cfg['video']['characters']['reference_images'] = [str(image)]
    original = client.generate
    def generate(self, *args):
        result = original(self, *args)
        result['characters'][0]['reference_ids'] = ['reference-1']
        return result
    monkeypatch.setattr(client, 'generate', generate)
    ai.run_ai_pipeline(cfg)
    cfg['video']['characters']['reference_images'] = []
    ai.run_ai_pipeline(cfg)
    assert len([c for c in calls if c[0]=='generate']) == 2


def test_cancelled_job_never_calls_ai(job):
    from autoedit.runtime import cancellation, JobCancelled
    cfg, calls, _ = job
    cancel = Path(cfg['ai']['cache_dir']).parent/'cancel'
    cancel.touch()
    with pytest.raises(JobCancelled), cancellation(cancel):
        ai.run_ai_pipeline(cfg)
    assert not calls


def test_loaded_configs_do_not_share_reference_lists_or_defaults():
    first = load_config(None)
    first['video']['characters']['reference_images'] = ['one-run.png']
    first['premiere']['source_media'] = ['one-run.mp4']
    second = load_config(None)
    assert second['video']['characters'].get('reference_images', []) == []
    assert second['premiere']['source_media'] == ''


@pytest.mark.parametrize('preset', ['soda_pop', 'golden', 'cosita'])
def test_preset_persistence_never_saves_run_references(tmp_path, preset):
    settings = defaults()
    settings['presets'][preset].update(premiere_project='template.prproj', cubase_project='template.cpr', sampler_tracks='2-5')
    settings['reference_images'] = ['private.png']
    path = tmp_path/'settings.json'
    save_settings(settings, path)
    saved = load_settings(path)
    assert 'reference_images' not in json.loads(path.read_text())
    cfg = apply_preset(load_config(None), saved, preset)
    assert cfg['premiere']['project'] == 'template.prproj'
    assert cfg['cubase']['sampler_tracks'] == '2-5'
    assert cfg['ai']['preset'] == preset and cfg['ai']['enabled']
