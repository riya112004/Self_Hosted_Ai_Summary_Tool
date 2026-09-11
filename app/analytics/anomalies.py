import polars as pl


def detect_outliers(series: pl.Series, method: str = "iqr") -> dict:
    """Detect outliers using IQR method. Returns count, values, and bounds."""
    s = series.drop_nulls()
    if s.len() == 0:
        return {"count": 0, "outliers": [], "lower_bound": None, "upper_bound": None}

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = s.filter((s < lower) | (s > upper)).to_list()
    return {
        "count": len(outliers),
        "outliers": outliers,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def detect_peaks(df: pl.DataFrame, col: str, drop_threshold: float | None = None) -> list:
    """Detect local peaks: value higher than both neighbors."""
    values = df[col].to_list()
    peaks = []
    for i in range(1, len(values) - 1):
        prev, cur, nxt = values[i - 1], values[i], values[i + 1]
        if prev is None or cur is None or nxt is None:
            continue
        if cur > prev and cur > nxt:
            peaks.append(
                {
                    "index": i,
                    "value": cur,
                    "previous": prev,
                    "next": nxt,
                    "pct_above_prev": round((cur - prev) / abs(prev) * 100, 2) if prev != 0 else None,
                }
            )
    return peaks


def detect_drops(df: pl.DataFrame, col: str, threshold: float = -10.0) -> list:
    """Detect significant drops between consecutive values below threshold %."""
    values = df[col].to_list()
    drops = []
    for i in range(1, len(values)):
        prev, cur = values[i - 1], values[i]
        if prev is None or cur is None or prev == 0:
            continue
        change = (cur - prev) / abs(prev) * 100
        if change <= threshold:
            drops.append(
                {
                    "index": i,
                    "from": prev,
                    "to": cur,
                    "pct_change": round(change, 2),
                }
            )
    return drops