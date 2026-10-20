import math
import pytest
from autoedit.ai.sampling import sampling_settings, annotate_sources, video_part


@pytest.mark.parametrize('key,value', [('video_fps',0),('video_fps',25),('video_fps',True),
    ('video_fps',math.nan),('video_fps',math.inf),('max_source_seconds',-1),
    ('max_source_seconds',None),('shot_fallback','true'), ('transition_tail_fallback_sec', -1),
    ('transition_tail_fallback_sec', 2.01), ('transition_tail_fallback_sec', True)])
def test_invalid_video_evidence_config_fails_early(key,value):
    with pytest.raises(ValueError):
        sampling_settings({key:value})


def test_sampling_metadata_keeps_original_duration_and_caps_evidence():
    cfg=sampling_settings({'video_fps':3,'max_source_seconds':180})
    sources=annotate_sources([{'id':'a','duration_sec':600},{'id':'b','duration_sec':90}],cfg)
    assert sources[0]['duration_sec']==600
    assert [s['analyzed_end_sec'] for s in sources]==[180,90]
    part=video_part({'fileData':{'fileUri':'test','mimeType':'video/mp4'}},sources[0],cfg)
    assert part['videoMetadata']=={'fps':3.,'startOffset':'0s','endOffset':'180s'}


def test_zero_limit_preserves_full_source():
    assert annotate_sources([{'duration_sec':37}],sampling_settings({}))[0]['analyzed_end_sec']==37
