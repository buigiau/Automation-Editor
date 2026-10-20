from pathlib import Path

import pytest

from autoedit.job_queue import JobQueue


def config(output):
    return {'job': {'output_dir': str(output)}, 'premiere': {'source_media': ['first.mp4']},
            'ai': {'backend': 'local'}, 'video': {'characters': {'reference_images': ['a.png']}}}


def test_submissions_snapshot_configuration_and_reserve_separate_outputs(tmp_path):
    queue = JobQueue()
    cfg = config(tmp_path)
    first = queue.add('Same job', 'Soda Pop', cfg)
    cfg['premiere']['source_media'][0] = 'second.mp4'
    cfg['video']['characters']['reference_images'].append('b.png')
    cfg['ai']['backend'] = 'gemini'
    second = queue.add('Same job', 'Golden', cfg)
    assert first.config['premiere']['source_media'] == ['first.mp4']
    assert first.config['video']['characters']['reference_images'] == ['a.png']
    assert first.config['ai']['backend'] == 'local'
    assert second.config['ai']['backend'] == 'gemini'
    assert first.output_dir != second.output_dir
    assert first.output_dir.is_dir() and second.output_dir.is_dir()
    assert first.output_dir.parent == tmp_path
    assert cfg['job']['output_dir'] == str(tmp_path)


def test_fifo_pause_failure_and_cancelled_pending_job(tmp_path):
    queue = JobQueue()
    first, second, third = [queue.add(str(i), 'Soda Pop', config(tmp_path)) for i in range(3)]
    queue.paused = True
    assert queue.start_next() is None
    queue.paused = False
    assert queue.start_next() is first
    assert queue.start_next() is None
    queue.cancel(second)
    queue.finish(first, error=RuntimeError('Failed media'))
    assert first.state == 'error'
    assert first.error == 'Failed media'
    assert second.state == 'cancelled'
    assert queue.start_next() is third
    queue.finish(third, result={'plan_path': 'third.json'})
    assert third.state == 'done'
    assert third.result['plan_path'] == 'third.json'
    assert queue.start_next() is None


def test_active_cancellation_waits_for_worker_completion(tmp_path):
    queue = JobQueue()
    first = queue.add('one', 'Soda Pop', config(tmp_path))
    second = queue.add('two', 'Soda Pop', config(tmp_path))
    queue.start_next()
    queue.cancel(first)
    assert first.state == 'cancelling'
    assert first.cancel_event.is_set()
    assert not second.cancel_event.is_set()
    assert queue.start_next() is None
    queue.finish(first, error=RuntimeError('Stopped'))
    assert first.state == 'cancelled'
    assert queue.start_next() is second


def test_close_cancels_pending_jobs_and_never_starts_another(tmp_path):
    queue = JobQueue()
    first = queue.add('one', 'Soda Pop', config(tmp_path))
    second = queue.add('two', 'Soda Pop', config(tmp_path))
    queue.start_next()
    with pytest.raises(ValueError):
        queue.finish(second)
    queue.cancel_all()
    assert first.state == 'cancelling'
    assert second.state == 'cancelled'
    queue.finish(first)
    assert first.state == 'cancelled'
    assert queue.start_next() is None


def test_output_directory_failure_does_not_add_job(tmp_path):
    queue = JobQueue()
    not_a_folder = tmp_path / 'file'
    not_a_folder.touch()
    with pytest.raises(OSError):
        queue.add('one', 'Soda Pop', config(not_a_folder))
    assert queue.jobs == []


def test_job_names_cannot_escape_output_directory(tmp_path):
    job = JobQueue().add('../../escape / ..', 'Soda Pop', config(tmp_path))
    assert job.output_dir.parent == tmp_path
    assert Path(job.config['job']['output_dir']).is_absolute()
