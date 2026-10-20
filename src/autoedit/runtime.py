"""Cross-process job admission, cooperative cancellation and memory telemetry."""
from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
import errno
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
_cancel_file = ContextVar('autoedit_cancel_file', default=None)
_selection_deadline = ContextVar('autoedit_selection_deadline', default=float('inf'))


class SelectionTimeout(TimeoutError):
    pass


class JobCancelled(RuntimeError):
    pass


def check_cancelled():
    path = _cancel_file.get()
    if path and Path(path).exists():
        raise JobCancelled('Job cancelled; verified checkpoints are saved for the next run.')
    if time.monotonic() >= _selection_deadline.get():
        raise SelectionTimeout('Source audio/scene search budget exhausted; export uses completed evidence.')


@contextmanager
def selection_deadline(deadline):
    token = _selection_deadline.set(min(deadline, _selection_deadline.get()))
    try:
        yield
    finally:
        _selection_deadline.reset(token)


def cancellation_file():
    return _cancel_file.get()


@contextmanager
def cancellation(path=None):
    token = _cancel_file.set(path)
    try:
        check_cancelled()
        yield
    finally:
        _cancel_file.reset(token)


@contextmanager
def job_lock(log=None, path=None):
    """An OS lock is released even if a worker crashes; never delete the lock file."""
    path = Path(path or ROOT / 'output/.runtime/analysis.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with path.open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        waiting = False
        while True:
            check_cancelled()
            handle.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if not waiting and log:
                    log('Queued: another AutoEdit job is running. This job will start automatically; Cancel removes it from the queue.')
                waiting = True
                time.sleep(.2)
        try:
            if waiting and log:
                log('Queue cleared; starting this job.')
            yield time.perf_counter() - started
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def memory_snapshot():
    if os.name != 'nt':
        return {}
    class Status(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ('total_phys', 'avail_phys', 'total_page',
                'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]
    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {}
    return {'ram_total_bytes': status.total_phys, 'ram_available_bytes': status.avail_phys,
            'commit_available_bytes': status.avail_page}


class MemoryMonitor:
    """Sample system pressure, including isolated detector processes, every second."""
    def __enter__(self):
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.minimum = {}
        self.sample()
        def monitor():
            while not self.stopped.wait(1):
                self.sample()
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        data = memory_snapshot()
        with self.lock:
            for key, value in data.items():
                self.minimum[key] = min(self.minimum.get(key, value), value)

    @property
    def report(self):
        with self.lock:
            data = dict(self.minimum)
        if 'ram_total_bytes' in data:
            data['system_ram_peak_used_bytes'] = data['ram_total_bytes'] - data['ram_available_bytes']
        return data

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join(timeout=2)


def release_inference(runtime):
    """Do not keep ONNX/Whisper allocations while loading the next GPU model."""
    import gc
    runtime.clear()
    from autoedit.audio.source_speech import _MODELS
    _MODELS.clear()
    gc.collect()
