import tkinter as tk
import threading
import time
import gc

import pytest
from autoedit.ai.gui import Application
from autoedit.config import load_config
from autoedit.presets import defaults


@pytest.fixture(scope='module')
def tk_host():
    # A desktop application owns one Tcl interpreter; isolate windows, not Tk runtimes.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()
    gc.collect()


def test_preset_ui_starts_with_session_only_references(monkeypatch, tk_host):
    import autoedit.ai.gui as gui
    monkeypatch.setattr(gui, 'load_settings', defaults)
    cfg = load_config(None)
    cfg['video']['characters']['reference_images'] = ['stale-reference.png']
    root = tk.Toplevel(tk_host)
    root.withdraw()
    try:
        app = Application(root, cfg)
        root.update_idletasks()
        assert app.references == []
        assert app.preset.get() == 'Soda Pop'
        assert app.run_button['text'] == '+ Thêm job'
        assert not app.active
    finally:
        root.destroy()
        del app
        gc.collect()


@pytest.fixture
def studio(monkeypatch, tmp_path, tk_host):
    import autoedit.ai.gui as gui
    settings = defaults()
    settings.update(backend='local', last_output=str(tmp_path))
    monkeypatch.setattr(gui, 'load_settings', lambda: settings)
    monkeypatch.setattr(gui, 'save_settings', lambda settings: None)
    gc.collect()
    root = tk.Toplevel(tk_host)
    root.withdraw()
    app = Application(root, load_config(None))
    video = tmp_path / 'source.mp4'
    video.touch()
    app.videos.insert('end', str(video))
    yield app
    app.job_queue.cancel_all()
    app.cancel.set()
    # Consume worker events before destroying Tcl, without starting queued work.
    deadline = time.monotonic() + 3
    while app.active and time.monotonic() < deadline:
        root.update()
        time.sleep(.01)
    if root.winfo_exists():
        root.destroy()
    del app
    gc.collect()


def pump(app, predicate):
    deadline = time.monotonic() + 4
    while not predicate() and time.monotonic() < deadline:
        app.root.update()
        time.sleep(.01)
    assert predicate()


@pytest.mark.parametrize('backend', ['local', 'gemini'])
def test_add_jobs_while_running_uses_snapshots_and_runs_sequentially(studio, monkeypatch, backend):
    app = studio
    app.settings['backend'] = backend
    first_entered, release = threading.Event(), threading.Event()
    calls = []
    def worker(config, log, cancel_event):
        calls.append(config)
        log('worker ' + str(len(calls)))
        if len(calls) == 1:
            first_entered.set()
            assert release.wait(3)
        return {'plan_path': config['job']['output_dir'] + '/edit-plan.json'}
    monkeypatch.setattr('autoedit.worker.run_fresh', worker)
    app.run()
    try:
        pump(app, first_entered.is_set)
        assert 'disabled' not in app.run_button.state()
        assert 'disabled' not in app.settings_button.state()
        app.preset.set('Golden')
        app.settings['local_model'] = 'next-model'
        app.run()
        assert len(app.job_queue.jobs) == 2
        first, second = app.job_queue.jobs
        assert second.state == 'pending'
        assert len(calls) == 1
        assert first.config['ai']['local_model'] != 'next-model'
        assert second.config['ai']['local_model'] == 'next-model'
        assert first.output_dir != second.output_dir
        assert second.preset == 'Golden'
    finally:
        release.set()
    pump(app, lambda: all(job.state == 'done' for job in app.job_queue.jobs))
    assert [cfg['ai']['preset'] for cfg in calls] == ['soda_pop', 'golden']
    assert app.counts['done'].get() == '2'
    assert 'worker 1' in first.logs and 'worker 2' not in first.logs
    assert 'worker 2' in second.logs and 'worker 1' not in second.logs
    assert app.result_path.get() == str(second.output_dir)


def test_error_advances_queue_but_pause_and_pending_cancel_are_honored(studio, monkeypatch):
    app = studio
    app.toggle_pause()
    for _ in range(3):
        app.run()
    first, second, third = app.job_queue.jobs
    app.job_list.selection_set(second.id)
    app.cancel_selected()
    calls = []
    def worker(config, **kwargs):
        calls.append(config['job']['output_dir'])
        if len(calls) == 1:
            raise RuntimeError('Cannot decode')
        return {'plan_path': 'ok.json'}
    monkeypatch.setattr('autoedit.worker.run_fresh', worker)
    assert not app.active
    assert all(job.started_at is None for job in (first, second, third))
    app.toggle_pause()
    pump(app, lambda: third.state == 'done')
    assert first.state == 'error'
    assert first.error == 'Cannot decode'
    assert second.state == 'cancelled'
    assert calls == [str(first.output_dir), str(third.output_dir)]


def test_cancelling_worker_keeps_next_job_waiting_until_worker_exits(studio, monkeypatch):
    app = studio
    exiting, release = threading.Event(), threading.Event()
    calls = []
    def worker(config, log, cancel_event):
        calls.append(config)
        if len(calls) == 1:
            assert cancel_event.wait(3)
            exiting.set()
            assert release.wait(3)
        return {}
    monkeypatch.setattr('autoedit.worker.run_fresh', worker)
    app.run()
    app.run()
    first, second = app.job_queue.jobs
    app.cancel_active()
    try:
        pump(app, exiting.is_set)
        assert first.state == 'cancelling'
        assert second.state == 'pending'
        assert len(calls) == 1
    finally:
        release.set()
    pump(app, lambda: second.state == 'done')
    assert first.state == 'cancelled'
    assert not second.cancel_event.is_set()


def test_pause_does_not_interrupt_active_job(studio, monkeypatch):
    app = studio
    release = threading.Event()
    def worker(config, cancel_event, **kwargs):
        assert release.wait(3)
        assert not cancel_event.is_set()
        return {}
    monkeypatch.setattr('autoedit.worker.run_fresh', worker)
    app.run()
    app.run()
    first, second = app.job_queue.jobs
    app.toggle_pause()
    release.set()
    pump(app, lambda: not app.active)
    assert first.state == 'done'
    assert second.state == 'pending'
    app.toggle_pause()
    pump(app, lambda: second.state == 'done')


def test_result_actions_follow_selected_job_not_new_form_output(studio, monkeypatch, tmp_path):
    app = studio
    monkeypatch.setattr('autoedit.worker.run_fresh', lambda *args, **kwargs: {})
    app.run()
    pump(app, lambda: not app.active)
    job = app.job_queue.jobs[0]
    review = job.output_dir / 'ai_review.html'
    review.write_text('review', encoding='utf-8')
    app.output.set(str(tmp_path / 'next-output'))
    opened = []
    monkeypatch.setattr('autoedit.ai.gui.webbrowser.open', opened.append)
    app.select_job()
    app.open_review()
    assert opened == [review.resolve().as_uri()]
    assert app.result_path.get() == str(job.output_dir)


def test_reject_empty_source_submission_without_creating_job(studio, monkeypatch):
    app = studio
    app.videos.delete(0, 'end')
    errors = []
    monkeypatch.setattr('autoedit.ai.gui.messagebox.showerror', lambda *args, **kwargs: errors.append(args))
    app.run()
    assert errors
    assert app.job_queue.jobs == []


def test_reset_form_does_not_change_waiting_job(studio, tmp_path):
    app = studio
    app.toggle_pause()
    reference = tmp_path / 'reference.png'
    reference.touch()
    app.references.append(str(reference))
    app.run()
    job = app.job_queue.jobs[0]
    app.reset_form()
    assert app.videos.size() == 0
    assert app.references == []
    assert job.config['premiere']['source_media'] == [str(tmp_path / 'source.mp4')]
    assert job.config['video']['characters']['reference_images'] == [str(reference)]
    assert job.state == 'pending'


def test_close_stops_current_worker_without_starting_waiting_job(studio, monkeypatch):
    app = studio
    calls = []
    def worker(config, cancel_event, **kwargs):
        calls.append(config)
        assert cancel_event.wait(3)
        return {}
    monkeypatch.setattr('autoedit.worker.run_fresh', worker)
    app.run()
    app.run()
    first, second = app.job_queue.jobs
    app.close()
    pump(app, lambda: not app.root.winfo_exists())
    assert len(calls) == 1
    assert first.state == second.state == 'cancelled'
    assert second.started_at is None


@pytest.mark.parametrize('action', ['corrections', 'cubase'])
def test_result_changes_use_selected_job_and_keep_its_log(studio, monkeypatch, tmp_path, action):
    app = studio
    monkeypatch.setattr('autoedit.worker.run_fresh', lambda *args, **kwargs: {})
    app.run()
    pump(app, lambda: not app.active)
    job = app.job_queue.jobs[0]
    (job.output_dir / 'edit-plan.json').write_text('{}')
    app.output.set(str(tmp_path / 'different-output'))
    calls = []
    if action == 'corrections':
        corrections = job.output_dir / 'autoedit-corrections.json'
        corrections.write_text('{}')
        monkeypatch.setattr('autoedit.ai.gui.filedialog.askopenfilename', lambda **kwargs: str(corrections))
        def worker(config, log, **kwargs):
            calls.append(config['job']['output_dir'])
            log('applied corrections')
            return {}
        monkeypatch.setattr('autoedit.worker.run_fresh', worker)
        app.apply_edits()
        expected = str(job.output_dir)
        message = 'applied corrections'
    else:
        def importer(path, log):
            calls.append(str(path))
            log('imported samples')
        monkeypatch.setattr('autoedit.cubase.host.import_samples', importer)
        app.import_cubase()
        assert 'disabled' in app.cancel_button.state()
        expected = str(job.output_dir / 'edit-plan.json')
        message = 'imported samples'
    pump(app, lambda: not app.active)
    assert calls == [expected]
    assert app.selected_job() is job
    assert message in job.logs
    assert app.showing_log
