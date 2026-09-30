from autoedit.premiere.prproj import inspect_prproj
from itertools import groupby
from pathlib import Path
import re

projects = [
    (r'D:\Editor\Soda Pop\soda pop\Soda pop_1_1.prproj', r'D:\Editor\Soda Pop\soda pop\soda pop.cpr'),
    (r'D:\Editor\Golden\Adobe Premiere Pro Auto-Save\golden_1--f80fc78b-3e33-59a8-be37-bfe00ee618bc-2026-09-23_22-05-49.prproj', r'D:\Editor\Golden\golden.cpr'),
]
for premiere, cubase in projects:
    data = inspect_prproj(premiere)
    items = next(t for t in data['template_video_tracks'] if t['index'] == 1)['items']
    items = sorted([x for x in items if x.get('nested_sequence_uid')], key=lambda x: x['start_sec'])
    groups = [list(g) for _, g in groupby(items, key=lambda x: x['nested_sequence_uid'])]
    print(premiere)
    print('SCENE RUNS', [(g[0]['nested_sequence'], len(g), g[0]['start_sec'], g[-1]['end_sec']) for g in groups])
    raw = Path(cubase).read_bytes()
    print('SAMPLER NAMES', sorted(set(x.decode() for x in re.findall(rb'Sampler Track \d+\x00', raw))))
    for match in list(re.finditer(rb'PSamplerTrackAudioClip|\.wav\x00|\.WAV\x00', raw))[:8]:
        print('CPR SAMPLE RECORD', match.start(), repr(raw[max(0, match.start()-90):match.end()+60]))
