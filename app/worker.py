"""
Celery + Redis background workers.

    FastAPI -> Redis Queue -> Celery Worker -> Profiler -> LLM -> Validator
                                       |
                                       v
                                  MongoDB (results)

Why: a 100,000-row summary must not hold an HTTP request open for minutes.
Large POSTs return a job_id immediately; the worker does the heavy lifting.

Pipeline invariants still hold:
  - the LLM NEVER executes database queries (worker uses the same read-only
    app.database.query gate).
  - results are stored in MongoDB (result backend), not PostgreSQL.

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


def _run_summary(records, summary_type: str, run_llm: bool = True) -> dict:
    from .timing import finish, mark, start

    t_total = start()
    if not run_llm:
        from .profile import build_data_profile
        from .summary.deterministic import build_profile_summary

        t0 = start()
        profile = build_data_profile(records)
        mark("worker: build data profile", t0)

        t0 = start()
        out = build_profile_summary(profile, summary_type, records=records).model_dump()
        mark("worker: deterministic summary", t0)
        finish("worker._run_summary", t_total)
        return out

    from .llm import LLMService
    from .profile import build_data_profile

    t0 = start()
    profile = build_data_profile(records)
    mark("worker: build data profile", t0)

    t0 = start()
    summary = LLMService().generate_summary_from_profile(
        profile, summary_type, auto_fallback=True
    )
    mark(f"worker: LLM summary (mode={summary_type})", t0)
    finish("worker._run_summary", t_total)
    return summary.model_dump()


def _run_db_summary(connection, statement, summary_type: str, run_llm: bool = True) -> dict:
    from .database import query as run_db_query

    result = run_db_query(connection, statement)
    return _run_summary(result["records"], summary_type, run_llm)


@celery_app.task(name="summarize_records")
def task_summarize_records(records, summary_type: str, run_llm: bool = True) -> dict:
    """Profiler -> LLM -> validator -> summary dict (any row count)."""
    return _run_summary(records, summary_type, run_llm)


@celery_app.task(name="summarize_database")
def task_summarize_database(
    connection, statement, summary_type: str, run_llm: bool = True
) -> dict:
    """Read-only query -> profiler -> LLM -> validator -> summary dict."""
    return _run_db_summary(connection, statement, summary_type, run_llm)


# Maps plain callable names (from the API layer) to registered celery tasks.
TASKS = {
    "_run_summary": task_summarize_records,
    "_run_db_summary": task_summarize_database,
}

__all__ = ["celery_app", "task_summarize_records", "task_summarize_database"]