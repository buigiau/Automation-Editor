from pathlib import Path
import shutil
import cv2
import numpy as np
from autoedit.ai.visual import check_visual_duplicates, probe, fingerprints
from autoedit.ai.cache import identity


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12, (128, 96))
    assert writer.isOpened()
    rng = np.random.default_rng(12)
    backgrounds = [rng.integers(0, 256, (96, 128, 3), dtype=np.uint8) for _ in range(2)]
    for i in range(96):
        phase = i % 48
        frame = backgrounds[phase//24].copy()
        cv2.rectangle(frame, ((phase%24)*3, 20), ((phase%24)*3+20, 50), (255, 0, 0), -1)
        writer.write(frame)
    writer.release()


def test_nested_tail_not_played_by_parent_is_excluded_from_visual_comparison(tmp_path):
    path = tmp_path/'source.mp4'
    make_video(path)
    source = {'id': 's', 'path': str(path), 'identity': identity(path), **probe(path)}
    actual = fingerprints(source, {'in_sec': .1, 'out_sec': 5.78, 'timeline_visible_prefix': True,
                                   'identity_end_sec': .66}, tmp_path/'cache')
    expected = fingerprints(source, {'in_sec': .1, 'out_sec': .66}, tmp_path/'cache')
    assert actual == expected and len(actual) == 8


def test_real_frames_reject_duplicate_at_other_timestamp_and_file(tmp_path):
    path = tmp_path/'a.mp4'
    make_video(path)
    copy = tmp_path/'copy.mp4'
    shutil.copyfile(path, copy)
    sources = [{'id': key, 'path': str(p), 'identity': identity(p), **probe(p)} for key, p in [('a', path), ('b', copy)]]
    rows = [{'slot_id': '1', 'source_id': 'a', 'in_sec': .1, 'out_sec': 1.5},
            {'slot_id': '2', 'source_id': 'a', 'in_sec': 4.1, 'out_sec': 5.5},
            {'slot_id': '3', 'source_id': 'b', 'in_sec': .1, 'out_sec': 1.5},
            {'slot_id': '4', 'source_id': 'b', 'in_sec': 2.1, 'out_sec': 3.5}]
    accepted, errors = check_visual_duplicates(rows, sources, tmp_path/'cache')
    assert [r['slot_id'] for r in accepted] == ['1', '4']
    assert [e['slot_id'] for e in errors] == ['2', '3']


def test_fingerprints_are_reused_without_decoding(tmp_path, monkeypatch):
    from autoedit.ai.visual import fingerprints
    import av
    path = tmp_path/'a.mp4'
    make_video(path)
    source = {'path': str(path), 'identity': identity(path)}
    row = {'in_sec': .1, 'out_sec': 1.5}
    first = fingerprints(source, row, tmp_path/'cache')
    def forbidden(*args):
        raise AssertionError('Re-decoded cached frames')
    monkeypatch.setattr(av, 'open', forbidden)
    assert fingerprints(source, row, tmp_path/'cache') == first


def test_eight_duplicate_samples_use_one_seek(tmp_path, monkeypatch):
    from autoedit.ai.visual import fingerprints
    import av
    path = tmp_path/'one-seek.mp4'
    make_video(path)
    original_open = av.open
    seeks = []
    class Container:
        def __init__(self, wrapped): self.wrapped = wrapped
        def __enter__(self): return self
        def __exit__(self, *args): return self.wrapped.__exit__(*args)
        def __getattr__(self, name): return getattr(self.wrapped, name)
        def seek(self, *args, **kwargs):
            seeks.append(args)
            return self.wrapped.seek(*args, **kwargs)
    monkeypatch.setattr(av, 'open', lambda *a, **k: Container(original_open(*a, **k)))
    result = fingerprints({'path': str(path), 'identity': identity(path)},
                          {'in_sec': .1, 'out_sec': 1.5}, tmp_path/'cache')
    assert len(result) == 8 and len(seeks) == 1
