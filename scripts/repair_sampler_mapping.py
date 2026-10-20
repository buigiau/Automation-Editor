"""Rebind an existing AI bundle using validated native tracks, without AI analysis."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil

from autoedit.ai.cache import identity, read, write
from autoedit.ai.review import write_review
from autoedit.cubase.inspect import inspect_sampler_tracks, select_sampler_tracks
from autoedit.cubase.sampler import render_sampler_bundle, verify_sampler_inputs
from autoedit.schema import save_plan


def repair(plan_path, destination=None):
    plan_path = Path(plan_path).resolve()
    original = read(plan_path)
    snapshot = read(plan_path.parent / 'ai-review.json')
    if not original or not snapshot or original.get('superseded_by'):
        raise ValueError('A current AI plan and its ai-review.json are required')
    for stamp in snapshot['template_identity']:
        if identity(stamp['path']) != stamp:
            raise ValueError('Template changed since analysis; repair stopped')
    scenes = snapshot['inspect']['scene_slots']
    slots = original['cubase_slots']
    if len(slots) != len(scenes):
        raise ValueError('Plan scene count differs from the saved analysis')
    for slot, scene in zip(slots, scenes):
        if (slot['premiere']['nested_sequence_uid'] != scene['nested_sequence_uid']
                or slot['premiere']['instances'] != scene['instances']):
            raise ValueError('Plan scene order/instances differ from the saved analysis')
    inventory = select_sampler_tracks(inspect_sampler_tracks(original['project']['cubase_project']),
        snapshot['config']['cubase'].get('sampler_tracks'), len(scenes), scenes=scenes, infer_duplicates=True)
    if inventory.get('mapping_mode') != 'duplicate_scene_tracks':
        raise ValueError('No validated companion-track mapping found; repair stopped')
    plan = deepcopy(original)
    for slot, track in zip(plan['cubase_slots'], inventory['tracks']):
        slot['cubase'].update(track_index=track['index'], track_name=track['name'])
        slot['cubase'].pop('duplicate_of', None)
        if track.get('is_duplicate'):
            slot['cubase']['duplicate_of'] = track['duplicate_of']
    first = {}
    for slot in plan['cubase_slots']:
        uid = slot['premiere']['nested_sequence_uid']
        if uid in first and (slot['audio'] != first[uid]['audio']
                or slot['premiere']['duration_sec'] != first[uid]['premiere']['duration_sec']):
            raise ValueError('Reprise has a different Voice or duration; repair stopped')
        first[uid] = slot
    verify_sampler_inputs(slots)
    out = Path(destination).resolve() if destination else plan_path.parent
    out.mkdir(parents=True, exist_ok=True)
    if out == plan_path.parent:
        backup = out / 'sampler_mapping_backups' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup.mkdir(parents=True)
        for name in ('edit-plan.json', 'cubase_import.json', 'sampler_mapping.csv',
                     'SAMPLER_IMPORT.txt', 'premiere_audio_preview.wav', 'ai-review.json', 'ai_review.html'):
            source = out / name
            if source.is_file():
                shutil.copy2(source, backup / name)
    result = render_sampler_bundle(plan, out)
    verify_sampler_inputs(plan['cubase_slots'])
    plan['cubase_manifest'] = result['manifest']
    snapshot['inventory'] = inventory
    snapshot['config']['cubase']['sampler_tracks'] = inventory['selected_track_indices']
    write(out / 'ai-review.json', snapshot)
    write_review(snapshot, out, plan)
    plan['ai_review'] = str(out / 'ai_review.html')
    save_plan(plan, out / 'edit-plan.json')
    assert plan['slots'] == original['slots']
    assert plan['project'] == original['project']
    assert [s['audio'] for s in plan['cubase_slots']] == [s['audio'] for s in slots]
    print(json.dumps({'plan': str(out / 'edit-plan.json'), 'tracks': [
        {'scene': s['premiere']['nested_sequence'], 'track': s['cubase']['track_name'],
         'duplicate_of': s['cubase'].get('duplicate_of')} for s in plan['cubase_slots']]}, indent=2))
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan')
    parser.add_argument('--output', help='Prepare a separate bundle for review')
    args = parser.parse_args()
    repair(args.plan, args.output)
