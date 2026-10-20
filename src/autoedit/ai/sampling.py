"""Validate the video evidence window and its Gemini sampling density."""
import math


def sampling_settings(config):
    values = {'video_fps': config.get('video_fps', 1),
              'max_source_seconds': config.get('max_source_seconds', 0),
              'transition_tail_fallback_sec': config.get('transition_tail_fallback_sec', 0)}
    for key, value in values.items():
        if isinstance(value, bool):
            raise ValueError(f'ai.{key} must be a finite number')
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'ai.{key} must be a finite number') from None
        if not math.isfinite(values[key]):
            raise ValueError(f'ai.{key} must be a finite number')
    if not 0 < values['video_fps'] <= 24:
        raise ValueError('ai.video_fps must be greater than 0 and at most 24')
    if values['max_source_seconds'] < 0:
        raise ValueError('ai.max_source_seconds must be non-negative (0 = entire source)')
    if not 0 <= values['transition_tail_fallback_sec'] <= 2:
        raise ValueError('ai.transition_tail_fallback_sec must be between 0 and 2 seconds')
    fallback = config.get('shot_fallback', True)
    if not isinstance(fallback, bool):
        raise ValueError('ai.shot_fallback must be true or false')
    final_fallback = config.get('final_scene_transition_fallback', False)
    if not isinstance(final_fallback, bool):
        raise ValueError('ai.final_scene_transition_fallback must be true or false')
    return {**values, 'shot_fallback': fallback, 'final_scene_transition_fallback': final_fallback}


def annotate_sources(sources, sampling):
    limit = sampling['max_source_seconds']
    return [{**source, 'analyzed_start_sec': 0.,
             'analyzed_end_sec': min(source['duration_sec'], limit) if limit else source['duration_sec']}
            for source in sources]


def video_part(upload, source, sampling):
    return {**upload, 'videoMetadata': {
        'fps': sampling['video_fps'],
        'startOffset': f"{source['analyzed_start_sec']:g}s",
        'endOffset': f"{source['analyzed_end_sec']:g}s"}}
