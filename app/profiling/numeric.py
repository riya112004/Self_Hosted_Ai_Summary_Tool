import polars as pl


def profile_numeric(series: pl.Series) -> dict:
    """Deterministic numeric profiling using Polars — no LLM."""
    return {
        "count": int(series.count()),
        "sum": float(series.sum()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
        "null_count": int(series.null_count()),
    }