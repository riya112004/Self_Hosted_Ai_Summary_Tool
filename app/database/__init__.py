"""
Database connector entry point.

    user -> config -> read-only connection -> query -> rows -> UDR

The user supplies the query; the LLM never receives a database handle.
The connection is always opened read-only: only guard-verified read
operations can run.

Example:
    store = get_store({"uri": "mongodb://localhost:27017", "db": "sales", "collection": "orders"})
    with store:
        result = store.run({"query_type": "find", "query": {"status": "active"}, "limit": 100})
"""
import os

from dotenv import load_dotenv

from .base import ReadOnlyStore, QueryRejected, guard_query

load_dotenv()

_ALLOWED_URI_SCHEMES = ("mongodb://", "mongodb+srv://")


def _default_connection() -> dict:
    return {
        "uri": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "db": os.getenv("MONGO_DB", "test"),
        "collection": os.getenv("MONGO_COLLECTION", "records"),
    }


def get_store(connection: dict | None = None, max_rows: int = 5000) -> ReadOnlyStore:
    """Build a connected, read-only store.

    connection may override env defaults: {"uri", "db", "collection"}.
    """
    from .mongo_store import MongoStore

    conn = {**_default_connection(), **(connection or {})}

    uri = conn["uri"] or ""
    if not uri.startswith(_ALLOWED_URI_SCHEMES):
        raise QueryRejected(
            f"Only mongodb/mongodb+srv URIs are allowed, got scheme of {uri!r}"
        )

    store = MongoStore(conn, max_rows=max_rows)
    store.connect()
    return store


def query(connection: dict | None, statement: dict) -> dict:
    """One-shot read-only query. Returns {"source_type", "count", "records"}."""
    with get_store(connection) as store:
        result = store.run(statement)
    return {
        "source_type": "mongodb",
        "count": result["count"],
        "records": result["records"],
    }


__all__ = ["ReadOnlyStore", "QueryRejected", "guard_query", "get_store", "query"]