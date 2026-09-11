import polars as pl


def _aggregate(
    df: pl.DataFrame,
    group_col: str,
    metric_col: str,
    agg: str = "sum",
    n: int = 5,
    descending: bool = True,
) -> dict:
    expr = getattr(pl.col(metric_col), agg)()
    result = (
        df.group_by(group_col)
        .agg(expr.alias("value"))
        .sort("value", descending=descending)
        .head(n)
    )
    return {str(row[group_col]): row["value"] for row in result.iter_rows(named=True)}


def top_categories(
    df: pl.DataFrame,
    group_col: str,
    metric_col: str,
    agg: str = "sum",
    n: int = 5,
) -> dict:
    """Top n categories by aggregated metric — highest values."""
    return _aggregate(df, group_col, metric_col, agg, n, descending=True)


def bottom_categories(
    df: pl.DataFrame,
    group_col: str,
    metric_col: str,
    agg: str = "sum",
    n: int = 5,
) -> dict:
    """Bottom n categories by aggregated metric — lowest values."""
    return _aggregate(df, group_col, metric_col, agg, n, descending=False)