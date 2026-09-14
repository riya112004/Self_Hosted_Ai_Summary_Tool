import polars as pl


def profile_categorical(series: pl.Series) -> dict:
    """Deterministic categorical profiling — value counts + percentages."""
    total = series.len()
    if total == 0:
        return {}

    vc = series.value_counts().sort("count", descending=True)
    result = {}
    for row in vc.iter_rows(named=True):
        value = str(row[series.name])
        count = row["count"]
        result[value] = {
            "count": count,
            "percentage": round(count / total * 100, 1),
        }
    # Keep value counts at the top level for existing consumers while adding
    # compact metadata needed by the LLM profile.
    result["unique_count"] = int(series.n_unique())
    result["null_count"] = int(series.null_count())
    return result