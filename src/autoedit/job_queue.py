"""Session job queue. Only the UI thread changes job state; workers emit events.

Each submission owns a configuration snapshot, cancellation token and output
directory. Preparing the next submission cannot change an active job.
"""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import threading
import time
from uuid import uuid4


@dataclass
class Job:
    id: str
    name: str
    preset: str
    config: dict
    state: str = 'pending'
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str = ''
    result: dict | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    logs: deque = field(default_factory=lambda: deque(maxlen=1500), repr=False)

    @property
    def output_dir(self):
        return Path(self.config['job']['output_dir'])

    def elapsed(self):
        if self.started_at is None:
            return '—'
        seconds = max(0, int((self.finished_at or time.time()) - self.started_at))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f'{hours:d}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


class JobQueue:
    def __init__(self):
        self.jobs: list[Job] = []
        self.current: Job | None = None
        self.paused = False

    def add(self, name, preset, config):
        snapshot = deepcopy(config)
        job_id = uuid4().hex
        slug = re.sub(r'[^\w-]+', '-', name, flags=re.UNICODE).strip('-')[:48] or 'job'
        folder = Path(snapshot['job']['output_dir']).expanduser().resolve() / (
            datetime.now().strftime('%Y%m%d-%H%M%S') + f'-{slug}-{job_id[:8]}')
        # Reserve atomically, even when several submissions have the same name.
        folder.mkdir(parents=True, exist_ok=False)
        snapshot['job']['output_dir'] = str(folder)
        job = Job(job_id, name, preset, snapshot)
        job.logs.append('Đã xếp hàng. Cấu hình được giữ riêng cho job này.')
        self.jobs.append(job)
        return job

    def start_next(self):
        if self.paused or self.current is not None:
            return None
        job = next((job for job in self.jobs if job.state == 'pending'), None)
        if job is not None:
            job.state = 'running'
            job.started_at = time.time()
            self.current = job
        return job

    def cancel(self, job):
        if job.state == 'pending':
            job.cancel_event.set()
            job.state = 'cancelled'
            job.finished_at = time.time()
            job.logs.append('Đã hủy trước khi chạy.')
        elif job.state == 'running':
            job.state = 'cancelling'
            job.cancel_event.set()
            job.logs.append('Đang dừng worker; job tiếp theo sẽ đợi worker thoát.')

    def finish(self, job, *, result=None, error=None):
        if job is not self.current:
            raise ValueError('Only the active job can finish')
        job.finished_at = time.time()
        job.result = result
        job.error = str(error) if error is not None else ''
        if job.cancel_event.is_set():
            job.state = 'cancelled'
        else:
            job.state = 'error' if error is not None else 'done'
        self.current = None

    def cancel_all(self):
        self.paused = True
        for job in self.jobs:
            self.cancel(job)
