from collections import Counter

import numpy as np
import pytest

from autoedit.match.matcher import InsufficientFootageError, match_slots_to_video
from autoedit.video.composition import annotate_compositions, frame_signature, group_compositions, promotional_overlay
from autoedit.video.pool import combine_video_infos


def footage(groups, ranks=None):
    samples = []
    for j, group in enumerate(groups):
        for k in range(12):
            sample = {'time_sec':j*6+k/2, 'face_visible':1, 'clear_face':1,
                      'readable_face':1, 'speaking':1, 'mouth_clarity':1,
                      'closeup_score':.6, 'composition_id':group}
            if ranks:
                rank = ranks[j]
                sample.update(character_id=f'actor-{rank}', character_rank=rank,
                              character_confidence=1., character_decision='main' if rank == 1 else 'supporting')
            samples.append(sample)
    info = {'duration_sec':len(groups)*6, 'sample_fps':2, 'samples':samples,
            'require_clear_face':True, 'max_similar_cuts':2}
    if ranks:
        info['character_analysis'] = {'min_main_fraction':.85}
    return info


def slots(count):
    return [{'id':str(i), 'required_duration_sec':1.5} for i in range(count)]


def counts(matches):
    return Counter(g for m in matches for g in m['video']['selection_metrics']['composition_ids'])


def test_distant_recurring_compositions_are_limited_without_losing_lengths_or_gap():
    info = footage(['table', 'closeup', 'table', 'solo', 'table', 'family', 'table'])
    matches = match_slots_to_video(slots(6), [], info, min_gap_sec=2)
    assert counts(matches)['table'] <= 2
    assert max(counts(matches).values()) <= 2
    ranges = sorted((m['video']['in_sec'], m['video']['out_sec']) for m in matches)
    assert all(b-a == pytest.approx(1.5) for a,b in ranges)
    assert all(b[0]-a[1] >= 2-1e-8 for a,b in zip(ranges, ranges[1:]))


def test_diversity_limit_cannot_be_evaded_by_allocation_search():
    info = footage(['table']*6)
    with pytest.raises(InsufficientFootageError, match='compositions are limited to 2'):
        match_slots_to_video(slots(3), [], info, min_gap_sec=2)
    info['max_similar_cuts'] = 0
    assert len(match_slots_to_video(slots(3), [], info, min_gap_sec=2)) == 3


def test_primary_actor_still_precedes_secondary_when_composition_is_available():
    info = footage(['table', 'solo', 'closeup', 'family'], [1, 2, 1, 2])
    matches = match_slots_to_video(slots(3), [], info, min_gap_sec=2)
    assert all(m['video']['character_selection']['best_rank'] == 1 for m in matches)
    assert max(counts(matches).values()) <= 2


def test_supporting_actor_can_fill_after_primary_composition_capacity_is_used():
    info = footage(['table', 'table', 'solo', 'family'], [1, 1, 2, 2])
    matches = match_slots_to_video(slots(4), [], info, min_gap_sec=2)
    assert counts(matches)['table'] == 2
    assert any(m['video']['character_selection']['best_rank'] == 2 for m in matches)


def test_composition_groups_do_not_chain_through_a_gradual_zoom():
    features = np.array([[0]*20, [10]*20, [20]*20], dtype=np.uint8)
    labels = group_compositions(features, distance=.06)
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_spatial_signature_distinguishes_closeup_from_same_background_wide_view():
    wide = np.full((360,640,3), [160,220,230], np.uint8)
    wide[110:180,130:200] = [190,50,90]
    wide[80:155,300:380] = [70,160,80]
    wide[110:180,450:510] = [40,70,130]
    close = wide.copy()
    close[40:290,120:450] = [190,50,90]
    changed_subtitles = wide.copy()
    changed_subtitles[310:] = [255,255,255]
    labels = group_compositions([frame_signature(wide),frame_signature(changed_subtitles),frame_signature(close)])
    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_measurements_are_cached_by_source_and_actual_sample_times(tmp_path, monkeypatch):
    path = tmp_path/'source.mp4'
    path.write_bytes(b'source')
    calls = []
    def frames(source, times):
        calls.append(times)
        for time in times:
            yield time, np.full((64,96,3), int(time*20), np.uint8)
    monkeypatch.setattr('autoedit.video.characters._selected_source_frames', frames)
    info = {'path':str(path), 'fps':30, 'samples':[{'time_sec':t} for t in [0., .5, 1.]]}
    annotate_compositions(info, tmp_path/'cache')
    annotate_compositions(info, tmp_path/'cache')
    assert len(calls) == 1
    assert all(s['composition_id'] for s in info['samples'])
    info['samples'].append({'time_sec':1.5})
    annotate_compositions(info, tmp_path/'cache')
    assert len(calls) == 2
    assert calls[-1] == [1.5]
    cached_labels = [s['composition_id'] for s in info['samples']]
    annotate_compositions(info, tmp_path/'fresh-cache')
    assert [s['composition_id'] for s in info['samples']] == cached_labels
    # Same path with changed source content must never reuse old measurements.
    path.write_bytes(b'changed source')
    annotate_compositions(info, tmp_path/'cache')
    assert calls[-1] == [0., .5, 1., 1.5]


def test_composition_ids_from_independent_sources_do_not_collide():
    info = footage(['view-0'])
    info['path'] = 'one.mp4'
    other = {**info, 'path':'two.mp4'}
    pool = combine_video_infos([info,other], 2)
    assert len({s['composition_id'] for s in pool['samples']}) == 2


def test_central_promotional_widget_is_excluded_but_subtitles_and_corner_logos_remain():
    sample = {'frame_width':640,'frame_height':360,'prop_regions':[
        {'label':'a watermark','score':.4,'box':[243,162,158,33]},
        {'label':'a sheet','score':.5,'box':[228,151,183,58],'frame_fraction':.046}]}
    assert promotional_overlay(sample)
    sample['prop_regions'][0]['box'] = [130,298,380,42]
    assert not promotional_overlay(sample)
    sample['prop_regions'][0]['box'] = [16,13,62,65]
    assert not promotional_overlay(sample)
    info = footage(['promotion', 'solo'])
    for s in info['samples']:
        s['promotional_overlay'] = float(s['time_sec'] < 6)
    result = match_slots_to_video(slots(1), [], info)
    assert result[0]['video']['in_sec'] >= 6
