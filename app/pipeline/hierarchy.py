import polars as pl


def choose_group_col(records: list[dict], max_groups: int = 10) -> str | None:
    """
    Pick the first low-cardinality categorical column suitable for grouping
    (works with Polars `group_by`). Returns None if nothing fits.
    """
    from app.profiling.schema_detector import detect_schema

    schema = detect_schema(records)
    for col in schema:
        if col["dtype"] != "categorical":
            continue
        unique = {r.get(col["name"]) for r in records}
        unique.discard(None)
        if 1 < len(unique) <= max_groups:
            return col["name"]
    return None


def aggregate_groups(records: list[dict], group_col: str) -> list[dict]:
    """
    Single-pass Polars aggregation — the "SQL/Polars aggregation" step.
    One row per group: row_count + min/max/mean/sum for every numeric column.
    """
    df = pl.DataFrame(records).with_columns(
        pl.col(group_col).cast(pl.Utf8, strict=False)
    )

    from app.profiling.schema_detector import detect_schema

    schema = detect_schema(records)
    numeric_cols = [
        c["name"]
        for c in schema
        if c["dtype"] == "numeric" and c["name"] in df.columns
    ]

    exprs = [pl.len().alias("count")]
    for col in numeric_cols:
        exprs.extend(
            [
                pl.col(col).min().alias(f"{col}_min"),
                pl.col(col).max().alias(f"{col}_max"),
                pl.col(col).mean().alias(f"{col}_mean"),
                pl.col(col).sum().alias(f"{col}_sum"),
            ]
        )

    grouped = df.group_by(group_col).agg(exprs).sort(group_col)
    return grouped.to_dicts()


def group_samples(records: list[dict], group_col: str) -> dict[str, list[dict]]:
    """Split records into per-group member lists (used for stratified sampling)."""
    members: dict[str, list[dict]] = {}
    for r in records:
        key = str(r.get(group_col))
        if key is None:
            continue
        members.setdefault(key, []).append(r)
    return members


def build_group_payload(
    group_col: str,
    group_name: str,
    agg_row: dict,
    sample_rows: list[dict],
    n: int = 3,
) -> dict:
    """
    Compact per-group payload sent to the LLM (map step):
    {"group", "row_count", "aggregation", "sample"}.
    """
    from app.sampling.representative import select_sample

    sample = select_sample(sample_rows, n=n, strategy="mixed")
    aggregation = {
        k: v for k, v in agg_row.items() if k not in (group_col, "count")
    }
    return {
        "group": group_name,
        "row_count": agg_row["count"],
        "aggregation": aggregation,
        "sample": sample,
    }