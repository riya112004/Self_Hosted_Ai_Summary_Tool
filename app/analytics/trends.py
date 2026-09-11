import polars as pl


def pct_change(current, previous):
    """Percent change between two values. Returns None for zero/None base."""
    if previous is None or current is None or previous == 0:
        return None
    return round((current - previous) / abs(previous) * 100, 2)


def _monthly_series(df: pl.DataFrame, date_col: str, metric_col: str, agg: str) -> pl.DataFrame:
    agg_expr = getattr(pl.col(metric_col), agg)()
    cleaned = df.with_columns(pl.col(date_col).str.to_date(strict=False).alias("_d"))
    non_null = cleaned.filter(pl.col("_d").is_not_null())
    return (
        non_null.with_columns(pl.col("_d").dt.strftime("%Y-%m").alias("_period"))
        .group_by("_period")
        .agg(agg_expr.alias("value"))
        .sort("_period")
    )


def growth_mom(df: pl.DataFrame, date_col: str, metric_col: str, agg: str = "sum") -> dict:
    """Month-over-month % changes. Returns {prev_month→next_month: pct_change}."""
    monthly = _monthly_series(df, date_col, metric_col, agg)
    periods = monthly["_period"].to_list()
    values = monthly["value"].to_list()

    changes = {}
    for i in range(1, len(values)):
        changes[f"{periods[i-1]}→{periods[i]}"] = pct_change(values[i], values[i-1])
    return changes


def growth_yoy(df: pl.DataFrame, date_col: str, metric_col: str, agg: str = "sum") -> dict:
    """Year-over-year % changes. Returns {prev_year→next_year: pct_change}."""
    agg_expr = getattr(pl.col(metric_col), agg)()
    cleaned = df.with_columns(pl.col(date_col).str.to_date(strict=False).alias("_d"))
    non_null = cleaned.filter(pl.col("_d").is_not_null())
    yearly = (
        non_null.with_columns(pl.col("_d").dt.strftime("%Y").alias("_period"))
        .group_by("_period")
        .agg(agg_expr.alias("value"))
        .sort("_period")
    )

    periods = yearly["_period"].to_list()
    values = yearly["value"].to_list()

    changes = {}
    for i in range(1, len(values)):
        changes[f"{periods[i-1]}→{periods[i]}"] = pct_change(values[i], values[i-1])
    return changes


def growth_overall(df: pl.DataFrame, date_col: str, metric_col: str, agg: str = "sum") -> float | None:
    """Overall % growth from first period to last period."""
    monthly = _monthly_series(df, date_col, metric_col, agg)
    values = monthly["value"].to_list()
    if len(values) < 2:
        return None
    return pct_change(values[-1], values[0])