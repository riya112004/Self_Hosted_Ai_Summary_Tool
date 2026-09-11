import polars as pl


def correlation_matrix(df: pl.DataFrame, cols: list | None = None) -> dict:
    """Pairwise Pearson correlation for numeric columns. Returns {colA__colB: corr}."""
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    if cols is not None:
        numeric_cols = [c for c in cols if c in df.columns and df[c].dtype.is_numeric()]

    result = {}
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1:]:
            corr = df.select(pl.corr(pl.col(a), pl.col(b))).item()
            result[f"{a}__{b}"] = round(corr, 4) if corr is not None else None
    return result