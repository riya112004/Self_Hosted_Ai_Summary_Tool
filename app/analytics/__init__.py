from concurrent.futures import ThreadPoolExecutor

import polars as pl

from .trends import growth_mom, growth_yoy, growth_overall
from .aggregations import top_categories, bottom_categories
from .anomalies import detect_outliers, detect_peaks, detect_drops
from .correlations import correlation_matrix


def compute_analytics(records: list[dict], date_col: str | None = None) -> dict:
    """
    Deterministic analytics on records using Polars.
    Only computes sections for which the data has the required columns.
    """
    if not records:
        return {"trends": {}, "aggregations": {}, "anomalies": {}, "correlations": {}}

    df = pl.DataFrame(records)

    from app.profiling.schema_detector import detect_schema

    schema = detect_schema(records)
    numeric_cols = [c["name"] for c in schema if c["dtype"] == "numeric"]
    categorical_cols = [c["name"] for c in schema if c["dtype"] == "categorical"]
    date_cols = [c["name"] for c in schema if c["dtype"] == "date"]
    if date_col is None and date_cols:
        date_col = date_cols[0]

    result = {"trends": {}, "aggregations": {}, "anomalies": {}, "correlations": {}}

    # Trends — requires a date column + at least one numeric column
    if date_col and date_col in df.columns and numeric_cols:
        def _compute_trend(col: str):
            mom = growth_mom(df, date_col, col)
            yoy = growth_yoy(df, date_col, col)
            overall = growth_overall(df, date_col, col)
            payload = {}
            if mom:
                payload[f"{col}_growth_mom"] = list(mom.values())[-1]
            if yoy:
                payload[f"{col}_growth_yoy"] = list(yoy.values())[-1]
            if overall is not None:
                payload[f"{col}_growth_overall"] = overall
            return payload

        with ThreadPoolExecutor(max_workers=min(8, len(numeric_cols) or 1)) as executor:
            trends = {}
            for payload in executor.map(_compute_trend, numeric_cols):
                trends.update(payload)
        result["trends"] = trends

    # Aggregations — requires a categorical col + numeric col
    if categorical_cols and numeric_cols:
        agg = {}
        with ThreadPoolExecutor(max_workers=min(8, len(categorical_cols) or 1)) as executor:
            for cat_col, top_bottom in executor.map(
                lambda col: (col, {
                    "top": top_categories(df, col, numeric_cols[0]),
                    "bottom": bottom_categories(df, col, numeric_cols[0]),
                }),
                categorical_cols,
            ):
                agg[cat_col] = top_bottom
        result["aggregations"] = agg

    # Anomalies — requires numeric columns
    if numeric_cols:
        anomalies = {}
        with ThreadPoolExecutor(max_workers=min(8, len(numeric_cols) or 1)) as executor:
            for col, outlier_stats in executor.map(
                lambda col: (col, detect_outliers(df[col])),
                numeric_cols,
            ):
                anomalies[col] = outlier_stats
        result["anomalies"] = anomalies

    # Correlations — requires 2+ numeric columns
    if len(numeric_cols) >= 2:
        result["correlations"] = correlation_matrix(df, numeric_cols)

    return result