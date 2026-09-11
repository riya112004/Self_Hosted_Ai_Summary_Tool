import random


def _freeze_value(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(val)) for key, val in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    return value


def _record_key(record: dict):
    return tuple(sorted((key, _freeze_value(value)) for key, value in record.items()))


def sample_first(records: list[dict], n: int = 5) -> list[dict]:
    return records[:n]


def sample_last(records: list[dict], n: int = 5) -> list[dict]:
    return records[-n:]


def sample_random(records: list[dict], n: int = 5, seed: int = 42) -> list[dict]:
    if len(records) <= n:
        return records[:]
    return random.Random(seed).sample(records, n)


def sample_top(records: list[dict], n: int = 5, metric_col: str | None = None) -> list[dict]:
    if metric_col is None:
        metric_col = _first_numeric_col(records)
    return sorted(
        records, key=lambda r: r.get(metric_col) or float("-inf"), reverse=True
    )[:n]


def sample_bottom(records: list[dict], n: int = 5, metric_col: str | None = None) -> list[dict]:
    if metric_col is None:
        metric_col = _first_numeric_col(records)
    return sorted(records, key=lambda r: r.get(metric_col) or float("inf"))[:n]


def sample_recent(records: list[dict], n: int = 5, date_col: str | None = None) -> list[dict]:
    if date_col is None:
        date_col = _first_date_col(records)
    return sorted(records, key=lambda r: str(r.get(date_col) or ""), reverse=True)[:n]


def sample_anomalies(records: list[dict], n: int = 5, metric_col: str | None = None) -> list[dict]:
    if metric_col is None:
        metric_col = _first_numeric_col(records)
    values = [r.get(metric_col) for r in records if isinstance(r.get(metric_col), (int, float))]
    if not values:
        return records[:n]

    ordered = sorted(values)
    q1 = ordered[len(ordered) // 4]
    q3 = ordered[(3 * len(ordered)) // 4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = [
        r for r in records
        if isinstance(r.get(metric_col), (int, float))
        and (r[metric_col] < lower or r[metric_col] > upper)
    ]
    return outliers[:n]


def select_sample(
    records: list[dict],
    n: int = 5,
    strategy: str = "mixed",
    metric_col: str | None = None,
    date_col: str | None = None,
    seed: int = 42,
) -> list[dict]:
    """Select a small representative sample using the requested strategy."""
    if not records:
        return []

    if strategy == "first":
        return sample_first(records, n)
    if strategy == "last":
        return sample_last(records, n)
    if strategy == "random":
        return sample_random(records, n, seed)
    if strategy == "top":
        return sample_top(records, n, metric_col)
    if strategy == "bottom":
        return sample_bottom(records, n, metric_col)
    if strategy == "recent":
        return sample_recent(records, n, date_col)
    if strategy == "anomalies":
        return sample_anomalies(records, n, metric_col)

    if strategy == "mixed":
        chosen: list[dict] = []
        chosen.append(records[0])
        chosen.extend(sample_random(records, 1, seed))
        if metric_col is not None or _has_numeric(records):
            chosen.extend(sample_top(records, 1, metric_col))
            chosen.extend(sample_bottom(records, 1, metric_col))
        if date_col is not None or _has_date(records):
            chosen.extend(sample_recent(records, 1, date_col))
        chosen.extend(sample_anomalies(records, n, metric_col))

        deduped: list[dict] = []
        seen: set = set()
        for row in chosen:
            key = _record_key(row)
            if key not in seen:
                seen.add(key)
                deduped.append(row)
        return deduped[:n]

    raise ValueError(f"Unknown strategy: {strategy}")


def _first_numeric_col(records: list[dict]) -> str | None:
    from app.profiling.schema_detector import detect_schema

    schema = detect_schema(records)
    for c in schema:
        if c["dtype"] == "numeric":
            return c["name"]
    return None


def _first_date_col(records: list[dict]) -> str | None:
    from app.profiling.schema_detector import detect_schema

    schema = detect_schema(records)
    for c in schema:
        if c["dtype"] == "date":
            return c["name"]
    return None


def _has_numeric(records: list[dict]) -> bool:
    return any(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for r in records for v in r.values()
    )


def _has_date(records: list[dict]) -> bool:
    return _first_date_col(records) is not None


def build_context(
    records: list[dict],
    n: int = 5,
    sample_type: str = "mixed",
    metric_col: str | None = None,
    date_col: str | None = None,
    seed: int = 42,
) -> dict:
    """
    Build the compact LLM context: schema + pre-aggregated statistics
    + a small representative sample — instead of sending raw thousands of rows.
    """
    from app.profiling.schema_detector import detect_schema
    from app.profiling.profiler import profile_dataset

    if not records:
        return {
            "total_rows": 0,
            "schema": [],
            "statistics": profile_dataset([]),
            "sample": {"strategy": sample_type, "rows": []},
        }

    schema = detect_schema(records)
    if metric_col is None:
        metric_col = _first_numeric_col(records)
    if date_col is None:
        date_col = _first_date_col(records)

    sample = select_sample(
        records,
        n=n,
        strategy=sample_type,
        metric_col=metric_col,
        date_col=date_col,
        seed=seed,
    )

    return {
        "total_rows": len(records),
        "schema": schema,
        "statistics": profile_dataset(records),
        "sample": {
            "strategy": sample_type,
            "metric_col": metric_col,
            "date_col": date_col,
            "row_count": len(sample),
            "rows": sample,
        },
    }