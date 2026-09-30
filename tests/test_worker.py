import json
import os
from pathlib import Path

import pytest

from autoedit import worker


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
