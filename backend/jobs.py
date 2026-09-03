"""Background job registry for streaming (inference/training) endpoints.

A job runs the blocking demo function in a worker thread. The thread binds its
log queue via log_bus so the demo's console output streams out, and pushes its own
structured progress/stats/done events onto the same queue. SSE endpoints drain it.

Single-process, single-worker uvicorn is assumed: only one training job at a time
(guarded by the caller with a 409), while lightweight jobs may overlap."""

import queue
import threading
import uuid

from backend import log_bus


class Job:
    def __init__(self, kind):
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.queue = queue.Queue()
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.thread = None


_jobs = {}
_lock = threading.Lock()

# Sentinel put on the queue when a job finishes, so SSE drains know to stop.
END = ("__end__", None)


def create(kind):
    job = Job(kind)
    with _lock:
        _jobs[job.id] = job
    return job


def get(job_id):
    with _lock:
        return _jobs.get(job_id)


def has_active(kind):
    with _lock:
        return any(j.kind == kind and not j.done.is_set() for j in _jobs.values())


def emit(job, event, payload):
    """Push a structured event onto the job queue (drained by SSE)."""
    job.queue.put((event, payload))


def start(job, target):
    """Run `target(job)` in a worker thread with the job's log queue active."""

    def _run():
        log_bus.set_active_queue(job.queue)
        try:
            target(job)
        except Exception as exc:  # surface to the stream instead of dying silently
            job.error = repr(exc)
            job.queue.put(("error", {"message": repr(exc)}))
        finally:
            job.done.set()
            job.queue.put(END)
            log_bus.clear_active_queue()

    job.thread = threading.Thread(target=_run, name=f"job-{job.kind}-{job.id}", daemon=True)
    job.thread.start()
    return job
