"""Adapters for existing pipeline fixtures that replace source decoding/ASR.

These tests use logical, nonexistent video paths. Keep their injected ASR in
process; real worker/deadline/cache behavior is exercised in test_source_first.
"""
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def injected_standard_audio(monkeypatch, request):
    from autoedit import pipeline
    from autoedit.audio import source_speech
    from autoedit.audio.search import SourceMatchSession
    from autoedit.ai import cache
    original_init, original_identity = SourceMatchSession.__init__, cache.identity

    def identity(path):
        if pipeline.transcribe_source is not source_speech.transcribe_source and not Path(path).exists():
            return {'path': str(path), 'fixture': request.node.nodeid}
        return original_identity(path)

    def initialize(self, *args, **kwargs):
        if pipeline.transcribe_source is not source_speech.transcribe_source and 'scan' not in kwargs:
            def scan(**arguments):
                arguments.pop('cache_dir', None)
                result = pipeline.transcribe_source(arguments.pop('path'), arguments.pop('ranges'), **arguments)
                return {**result, 'events': result.get('events', []),
                        'status': result.get('status', 'transcribed' if result.get('has_audio') else 'no-audio')}
            kwargs['scan'] = scan
            args = list(args)
            args[3] = request.getfixturevalue('tmp_path')/'shared-audio-cache'
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(cache, 'identity', identity)
    monkeypatch.setattr(SourceMatchSession, '__init__', initialize)
    original_presence = pipeline._source_has_audio
    monkeypatch.setattr(pipeline, '_source_has_audio', lambda path:
        False if pipeline.transcribe_source is not source_speech.transcribe_source and not Path(path).exists()
        else original_presence(path))
