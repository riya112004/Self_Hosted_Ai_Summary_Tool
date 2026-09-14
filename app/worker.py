"""
Celery + Redis background workers.

    FastAPI -> Redis Queue -> Celery Worker -> Profiler -> LLM -> Validator

Why: a 100,000-row summary must not hold an HTTP request open for minutes.
Large POSTs return a job_id immediately; the worker does the heavy lifting.

Run the worker (Windows-safe pool):
    celery -A app.worker.celery_app worker --loglevel=info --pool=solo
"""
import os

from dotenv import load_dotenv

load_dotenv()

from celery import Celery  # noqa: E402

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("CELERY_MONGO_DB", "ai_summarizer")

celery_app = Celery(
    "ai-summarizer",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=["app.worker"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_hijack_root_logger=False,
    mongodb_backend_settings={
        "database": MONGO_DB,
        "taskmeta_collection": "celery_taskmeta",
    },
)


def _run_summary(records, run_llm: bool = True) -> dict:
    from .timing import finish, mark, start

    t_total = start()
    if not run_llm:
        from .profile import build_data_profile
        from .summary.deterministic import build_profile_summary

        t0 = start()
        profile = build_data_profile(records)
        mark("worker: build data profile", t0)

        t0 = start()
        out = build_profile_summary(profile, records=records).model_dump()
        mark("worker: deterministic summary", t0)
        finish("worker._run_summary", t_total)
        return out

    from .llm import LLMService
    from .profile import build_data_profile, build_representative_sample

    t0 = start()
    profile = build_data_profile(records)
    mark("worker: build data profile", t0)

    t0 = start()
    sample = build_representative_sample(records)
    summary = LLMService().generate_summary_from_profile(
        profile, auto_fallback=True, sample=sample
    )
    mark("worker: LLM summary", t0)
    finish("worker._run_summary", t_total)
    return summary.model_dump()


@celery_app.task(name="summarize_records")
def task_summarize_records(records, run_llm: bool = True) -> dict:
    """Profiler -> LLM -> validator -> summary dict (any row count)."""
    return _run_summary(records, run_llm)


# Maps plain callable names (from the API layer) to registered celery tasks.
TASKS = {
    "_run_summary": task_summarize_records,
}

__all__ = ["celery_app", "task_summarize_records"]