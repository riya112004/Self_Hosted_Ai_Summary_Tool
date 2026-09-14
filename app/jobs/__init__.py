"""
Background job subsystem.

Large requests return a job_id immediately (HTTP 202); the work runs in the
background and the client polls GET /api/v1/jobs/{id} until status ==
completed / failed.

Two backends, switchable via JOBS_BACKEND:
  - threads (default, zero deps)  - in-process daemon thread + in-memory store
  - celery (production grade)     - Redis broker -> Celery worker -> MongoDB results

When JOBS_BACKEND=celery but the broker is unreachable, submit() falls back
to the thread backend so the server keeps working during local dev.
"""
import logging
import os
import threading
import uuid
from datetime import datetime, timezone

SMALL_MAX_ROWS = 1000

logger = logging.getLogger("ai-summarizer")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self) -> dict:
        with self._lock:
            job = {
                "id": uuid.uuid4().hex[:12],
                "status": "pending",
                "created_at": _now(),
                "summary": None,
                "error": None,
            }
            self._jobs[job["id"]] = job
            return job

    def mark_running(self, job_id: str) -> None:
        self._jobs[job_id]["status"] = "running"

    def complete(self, job_id: str, summary: dict) -> None:
        self._jobs[job_id]["summary"] = summary
        self._jobs[job_id]["status"] = "completed"

    def fail(self, job_id: str, error: str) -> None:
        self._jobs[job_id]["error"] = error
        self._jobs[job_id]["status"] = "failed"

    def get(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def stats(self) -> dict:
        counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for job in self._jobs.values():
            counts[job["status"]] = counts.get(job["status"], 0) + 1
        return counts


_store = JobStore()


def submit(target, *args, **kwargs) -> str:
    """Queue target(*args, **kwargs) as a background job. Returns job_id."""
    if os.getenv("JOBS_BACKEND", "threads") == "celery":
        job_id = _celery_launch(target, args, kwargs)
        if job_id is not None:
            return job_id
    return _thread_launch(target, list(args), dict(kwargs))


def get(job_id: str) -> dict | None:
    if os.getenv("JOBS_BACKEND", "threads") == "celery":
        job = _celery_status(job_id)
        if job is not None:
            return job
    return _store.get(job_id)


def stats() -> dict:
    return _store.stats()


def _thread_launch(target, args: list, kwargs: dict) -> str:
    job = _store.create()
    _run_thread(job["id"], target, args, kwargs)
    return job["id"]


def _run_thread(job_id: str, target, args: list, kwargs: dict) -> None:
    _store.mark_running(job_id)

    def worker() -> None:
        try:
            _store.complete(job_id, target(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - surfaced via job payload
            _store.fail(job_id, str(exc))

    threading.Thread(target=worker, daemon=True).start()


def _celery_launch(target, args: tuple, kwargs: dict) -> str | None:
    """Dispatch to a celery task. Returns job_id (task id) or None on failure."""
    try:
        from ..worker import celery_app, TASKS

        name = getattr(target, "__name__", None)
        task = TASKS.get(name)
        if task is None:
            logger.warning(
                "No celery task registered for %r; using thread backend", name
            )
            return None
        result = task.delay(*args, **kwargs)
        return result.id
    except Exception as exc:  # broker unreachable etc.
        logger.warning("Celery submit failed (%s); using thread backend", exc)
        return None


_STATE_TO_STATUS = {
    "PENDING": "pending",
    "RECEIVED": "pending",
    "STARTED": "running",
    "RETRY": "pending",
    "SUCCESS": "completed",
    "FAILURE": "failed",
}


def _celery_status(job_id: str) -> dict | None:
    """Read task status + result from the worker via AsyncResult."""
    try:
        from ..worker import celery_app

        res = celery_app.AsyncResult(job_id)
        return {
            "id": res.id,
            "status": _STATE_TO_STATUS.get(res.state, "pending"),
            "summary": res.result if res.successful() else None,
            "error": str(res.result) if res.failed() else None,
            "created_at": None,
        }
    except Exception as exc:
        logger.warning("Could not read celery job %s (%s)", job_id, exc)
        return None


__all__ = ["SMALL_MAX_ROWS", "submit", "get", "stats"]