import json
import os
from pathlib import Path

import pytest

from autoedit import worker


@pytest.mark.parametrize('device,visible', [('cpu', '-1'), ('cuda', '0')])
def test_cpu_worker_hides_cuda_and_bounds_compute_threads(tmp_path, monkeypatch, device, visible):
    package = tmp_path/'autoedit'
    package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'worker.py').write_text(Path(worker.__file__).read_text())
    (package/'pipeline.py').write_text(
        'import json, os, cv2\nfrom pathlib import Path\n'
        'def run_pipeline(config, log):\n'
        '    plan = {"cuda": os.environ.get("CUDA_VISIBLE_DEVICES"), '
        '"opencv_threads": cv2.getNumThreads(), "omp_threads": os.environ["OMP_NUM_THREADS"]}\n'
        '    (Path(config["job"]["output_dir"])/"edit-plan.json").write_text(json.dumps(plan))\n')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    result = worker.run_fresh({'job': {'output_dir': str(tmp_path)}, 'video': {
        'speech_device': device, 'characters': {'device': 'cpu'},
        'object_detection': {'enabled': False}}})
    assert result['plan'] == {'cuda': visible, 'opencv_threads': 2, 'omp_threads': '2'}


def test_already_cancelled_job_does_not_launch_or_create_outputs(tmp_path,monkeypatch):
    import threading
    from autoedit.runtime import JobCancelled
    event=threading.Event()
    event.set()
    monkeypatch.setattr(worker.subprocess,'Popen',lambda *a,**kw:pytest.fail('Cancelled job was launched'))
    with pytest.raises(JobCancelled):
        worker.run_fresh({'job':{'output_dir':str(tmp_path/'unused')}},cancel_event=event)
    assert not (tmp_path/'unused').exists()


def test_log_callback_failure_stops_child_and_cleans_cancellation_marker(tmp_path,monkeypatch):
    package=tmp_path/'autoedit'
    package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'worker.py').write_text(Path(worker.__file__).read_text())
    (package/'pipeline.py').write_text('import time\ndef run_pipeline(config,log):\n    log("ready")\n    time.sleep(30)\n')
    monkeypatch.setenv('PYTHONPATH',str(tmp_path))
    children=[]
    popen=worker.subprocess.Popen
    def spawn(*a,**kw):
        child=popen(*a,**kw)
        children.append(child)
        return child
    monkeypatch.setattr(worker.subprocess,'Popen',spawn)
    def failed_log(message):
        raise ValueError('log window closed')
    with pytest.raises(ValueError,match='log window closed'):
        worker.run_fresh({'job':{'output_dir':str(tmp_path)}},log=failed_log)
    assert children[0].poll() is not None
    assert not list(tmp_path.glob('.cancel-*'))


def test_gui_jobs_load_updated_code_and_do_not_return_stale_plan(tmp_path, monkeypatch):
    package = tmp_path / 'autoedit'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'worker.py').write_text(Path(worker.__file__).read_text())
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    source = package / 'pipeline.py'
    config = {'job': {'output_dir': str(tmp_path)}}
    messages = []
    for revision in ('old', 'new version'):
        source.write_text(
            'import json, os\nfrom pathlib import Path\n'
            'def run_pipeline(config, log):\n'
            f'    log({revision!r})\n'
            '    path = Path(config["job"]["output_dir"]) / "edit-plan.json"\n'
            f'    path.write_text(json.dumps({{"revision": {revision!r}, "pid": os.getpid()}}))\n')
        result = worker.run_fresh(config, log=messages.append)
        assert result['plan']['revision'] == revision
        assert result['plan']['pid'] != os.getpid()
    assert messages == ['old', 'new version']
    source.write_text('def run_pipeline(config, log):\n    raise ValueError("selection failed")\n')
    with pytest.raises(RuntimeError, match='selection failed'):
        worker.run_fresh(config)
