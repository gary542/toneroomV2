"""A tiny in-process job queue.

One worker thread, FIFO, so the ML models (which are not safe to run
concurrently on one device) never overlap. This is deliberately boring:
no Redis, no Celery, no extra service to run on a small VPS. If V2
ever needs more than one worker or more than one host, replace this
module with a real queue — the API and the pipeline do not depend on it.
"""

import queue
import threading

from .jobs import Job, new_job
from .pipeline.run import run_job

_queue: queue.Queue[str] = queue.Queue()
_started = False
_lock = threading.Lock()


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            job = Job.load(job_id)
            if job is not None:
                run_job(job)
        finally:
            _queue.task_done()


def start_worker() -> None:
    global _started
    with _lock:
        if _started:
            return
        _started = True
    thread = threading.Thread(target=_worker_loop, daemon=True, name="toneroom-worker")
    thread.start()


def enqueue(job: Job) -> Job:
    start_worker()
    _queue.put(job.id)
    return job


def create_and_enqueue(mode: str, finish: str, script: str, language: str,
                       consent: bool) -> Job:
    job = new_job(mode, finish, script, language, consent)
    return enqueue(job)
