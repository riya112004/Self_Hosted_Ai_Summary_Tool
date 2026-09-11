"""
Read-only database access layer.

Security model (per plan):
  - The USER supplies the query string; the LLM never does.
  - Only a fixed set of read query types may ever execute.
  - Any write operation (insert/update/delete/...) is rejected before it
    reaches the server.
  - Aggregate pipelines containing write stages ($out/$merge) or server-side
    JS ($where/$function/$accumulator) are rejected.
  - Row caps are enforced so no runaway query floods memory.
"""
from abc import ABC, abstractmethod

READ_ONLY_QUERY_TYPES = frozenset({"find", "aggregate", "count", "distinct"})

WRITE_QUERY_TYPES = frozenset(
    {
        "insert_one",
        "insert_many",
        "update_one",
        "update_many",
        "replace_one",
        "delete_one",
        "delete_many",
        "find_one_and_update",
        "find_one_and_replace",
        "find_one_and_delete",
        "bulk_write",
        "create_index",
        "drop",
        "rename",
        "create",
        "map_reduce",
    }
)

AGGREGATE_WRITE_STAGES = frozenset({"$out", "$merge"})
AGGREGATE_JS_STAGES = frozenset({"$where", "$function", "$accumulator"})

DEFAULT_MAX_ROWS = 5000


class QueryRejected(Exception):
    """Raised when a statement violates the read-only policy."""


def _contains_forbidden_key(value) -> bool:
    """Recursively scan pipeline for write/JS stages nested anywhere."""
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                continue
            if key in AGGREGATE_WRITE_STAGES or key in AGGREGATE_JS_STAGES:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, list):
        for item in value:
            if _contains_forbidden_key(item):
                return True
    return False


def guard_query(statement: dict, max_rows: int = DEFAULT_MAX_ROWS) -> int:
    """
    Validate a query statement against the read-only policy.

    Returns the effective row cap for the statement.

    Raises QueryRejected on any violation.
    """
    query_type = statement.get("query_type")

    if query_type not in READ_ONLY_QUERY_TYPES:
        raise QueryRejected(
            f"Query type {query_type!r} is not allowed. "
            f"Read-only types: {sorted(READ_ONLY_QUERY_TYPES)}"
        )

    requested = statement.get("limit")
    requested = requested if requested is not None else max_rows
    limit = min(int(requested), max_rows)

    query = statement.get("query")
    if query_type == "aggregate":
        if not isinstance(query, list):
            raise QueryRejected("aggregate query must be a list of pipeline stages")
        if _contains_forbidden_key(query):
            raise QueryRejected(
                "aggregate pipeline contains write ($out/$merge) or server-side "
                "JS ($where/$function/$accumulator) stages, which are not allowed"
            )

    return max(limit, 1)


class ReadOnlyStore(ABC):
    """
    Base class for all database stores.

    Stores only ever expose read operations through run(); write operations
    are rejected by guard_query() before touching the database.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abstractmethod
    def connect(self) -> None:
        """Open the (read-only) connection."""

    @abstractmethod
    def run(self, statement: dict) -> dict:
        """Execute a read-only statement. Returns {"records": [...], "count": N}."""

    @abstractmethod
    def close(self) -> None:
        """Release the connection."""

    def __enter__(self) -> "ReadOnlyStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()