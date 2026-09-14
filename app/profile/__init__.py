"""Universal CSV / data profiling pipeline.

Every number in the profile is computed by Python (Polars) from the FULL
dataset - so the middle rows' patterns are captured by the statistics, not
by any sample.  The LLM never receives the raw dataset: it only gets this
verified compact profile (plus a tiny first/random/last representative
excerpt for row-shape context) and is told to explain it.  This prevents
invented facts ("33 molecular entries", "atomic mass units") from leaking
into summaries.

    CSV / Excel -> build_data_profile(records) -> verified profile
                                + build_representative_sample(records) -> LLM
"""

import math

import polars as pl

from .domain import detect_domain


def _round(value, nd: int = 4):
    if value is None:
        return None
    try:
        f = float(value)
        if not math.isfinite(f):
            return None
        return round(f, nd)
    except (TypeError, ValueError):
        return value


def _norm_records(records) -> list[dict]:
    from app.structure import flatten_record

    if records is None:
        return []
    if not isinstance(records, list):
        return [{"value": records}]
    out = []
    for r in records:
        if isinstance(r, dict):
            out.append(flatten_record(r))
        else:
            out.append({"value": r})
    return out


_NUMERIC_TYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def _is_numeric(dt) -> bool:
    return any(isinstance(dt, t) for t in _NUMERIC_TYPES)


def _cv(stats: dict):
    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        return float("inf") if std else 0.0
    return (std / abs(mean)) if mean else float("inf")


def build_data_profile(
    records,
    top_variance: int = 10,
    corr_cap: int = 40,
    top_missing: int = 5,
    top_cat_cols: int = 8,
    top_cat_values: int = 5,
    top_outliers: int = 8,
    list_preview: int = 15,
) -> dict:
    """Deterministic, LLM-safe profile of a list of dict records."""
    records = _norm_records(records)

    empty = {
        "dataset": {"rows": 0, "columns": 0},
        "data_types": {"numeric": 0, "categorical": 0, "text": 0, "datetime": 0, "boolean": 0},
        "quality": {},
        "column_categories": {},
        "semantics": {},
        "statistics": {},
        "correlations": {},
        "categorical_breakdown": [],
        "class_imbalance": [],
        "anomalies": {},
        "domain": detect_domain([]),
        "patterns": {},
    }
    if not records:
        return empty

    try:
        df = pl.DataFrame(records)
    except Exception:
        # fall back to a single-key frame so nothing crashes
        df = pl.DataFrame({"value": [r for r in records]})

    rows = df.height
    cols = df.width
    schema = dict(df.schema)
    columns = list(schema.keys())

    # ---- 3. data type categories -----------------------------------------
    numeric, categorical, text_cols, datetime_cols, boolean_cols = [], [], [], [], []
    for name, dt in schema.items():
        if _is_numeric(dt):
            numeric.append(name)
        elif isinstance(dt, (pl.Datetime, pl.Date, pl.Time)):
            datetime_cols.append(name)
        elif isinstance(dt, pl.Boolean):
            boolean_cols.append(name)
        elif isinstance(dt, (pl.String, pl.Categorical)):
            try:
                length = df[name].cast(pl.String).str.len_chars().mean() or 0
                uniq = df[name].n_unique()
            except Exception:
                length, uniq = 0, 0
            # free text / id-like columns are excluded from the profile's
            # value breakdown - raw values must not reach the LLM
            if length > 60 or (rows and uniq > rows * 0.5):
                text_cols.append(name)
            else:
                categorical.append(name)
        else:
            categorical.append(name)

    def _preview(items: list[str]) -> list:
        return items[:list_preview] + (
            [f"... {len(items) - list_preview} more"] if len(items) > list_preview else []
        )

    def _build_semantics(cols: list[str]) -> dict:
        from .semantics import build_semantics

        return build_semantics(cols)

    # ---- 4. data quality ---------------------------------------------------
    null_counts = df.null_count().row(0)
    missing_total = int(sum(null_counts))
    missing_by_col = [
        {"column": c, "missing": int(n)}
        for c, n in zip(columns, null_counts)
        if n and n > 0
    ]
    missing_by_col.sort(key=lambda x: -x["missing"])
    missing_by_col = missing_by_col[:top_missing]

    unique_per_col = df.select(pl.all().n_unique()).row(0)
    constant_columns = [c for c, u in zip(columns, unique_per_col) if u <= 1]
    duplicate_rows = int(df.height - df.unique().height)

    quality = {
        "missing_values_total": missing_total,
        "missing_by_column_top": missing_by_col,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unique_value_counts": {
            c: int(u) for c, u in zip(columns, unique_per_col) if u > 1
        } if len(columns) <= 40 else {},
    }

    # ---- 5/6. numeric statistics -------------------------------------------
    stats_per_col = {}
    for col in numeric:
        ser = df[col]
        try:
            stats_per_col[col] = {
                "min": _round(ser.min()),
                "max": _round(ser.max()),
                "mean": _round(ser.mean()),
                "median": _round(ser.median()),
                "std": _round(ser.std()),
            }
        except Exception:
            continue

    ranked = sorted(stats_per_col.items(), key=lambda kv: -_cv(kv[1]))
    important = [
        {"column": c, **s} for c, s in ranked[:top_variance]
    ]

    global_min = global_max = None
    for c, s in stats_per_col.items():
        if s["min"] is not None and (global_min is None or s["min"] < global_min[0]):
            global_min = (s["min"], c)
        if s["max"] is not None and (global_max is None or s["max"] > global_max[0]):
            global_max = (s["max"], c)

    # ---- correlations (capped to most-relevant numeric columns, no numpy) ---
    corr_cols = [c for c, _ in ranked[:corr_cap]]
    pos_pairs, neg_pairs = [], []
    if len(corr_cols) >= 2:
        means = {c: df[c].mean() for c in corr_cols}
        stds = {c: df[c].std(ddof=0) for c in corr_cols}
        for i in range(len(corr_cols)):
            for j in range(i + 1, len(corr_cols)):
                a, b = corr_cols[i], corr_cols[j]
                ma, mb = means[a], means[b]
                if ma is None or mb is None:
                    continue
                try:
                    ab = df.select(pl.col(a).mul(pl.col(b)).mean()).item()
                except Exception:
                    continue
                sa, sb = stds[a], stds[b]
                if not sa or not sb or not ab:
                    continue
                cov = float(ab - ma * mb)
                v = cov / (sa * sb)
                if not math.isfinite(v):
                    continue
                pair = (a, b, round(v, 3))
                pos_pairs.append(pair) if v >= 0 else neg_pairs.append(pair)
    pos_pairs.sort(key=lambda p: -p[2])
    neg_pairs.sort(key=lambda p: p[2])

    # ---- 7. anomalies (IQR; phrase as potential, never "wrong") -------------
    total_outliers = 0
    outlier_rows = []
    for col in numeric:
        s = df[col].drop_nulls()
        if s.len() == 0:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        if q1 is None or q3 is None:
            continue
        iqr = q3 - q1
        if iqr is None or float(iqr) == 0:
            continue
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        n_out = int((s < lo).sum() + (s > hi).sum())
        total_outliers += n_out
        if n_out:
            outlier_rows.append(
                {"column": col, "outliers": n_out, "share": round(n_out / s.len(), 3)}
            )
    outlier_rows.sort(key=lambda x: -x["share"])

    def _safe_key(value):
        if isinstance(value, (list, tuple, set)):
            return tuple(_safe_key(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((str(k), _safe_key(v)) for k, v in value.items()))
        return value

    # ---- categorical breakdown + class imbalance ---------------------------
    cat_report = []
    imbalance = []
    for col in sorted(
        categorical, key=lambda c: len(df[c].unique()) if df[c].null_count() < df[c].len() else 0, reverse=True
    )[:top_cat_cols]:
        ser = df[col].drop_nulls()
        if ser.len() == 0:
            continue
        vc = ser.value_counts()
        count_col = vc.columns[1]
        total = int(vc[count_col].sum())
        top = vc.head(top_cat_values)
        items = [
            {"value": str(v), "count": int(c)}
            for v, c in zip(top[col].to_list(), top[count_col].to_list())
        ]

        value_counts = []
        for value, count in zip(vc[col].to_list(), vc[count_col].to_list()):
            try:
                key = _safe_key(value)
            except TypeError:
                key = str(value)
            value_counts.append((key, int(count)))

        sinks = {key: count for key, count in value_counts}
        dominant_share = (max(sinks.values()) / total) if total else 0
        unique_n = len(sinks)
        cat_report.append(
            {"column": col, "unique": unique_n, "top_values": items}
        )
        if unique_n > 2 and dominant_share >= 0.8:
            imbalance.append(
                {
                    "column": col,
                    "dominant_class_share": round(dominant_share, 3),
                    "suggestion": "potential class imbalance detected",
                }
            )

    # ---- patterns ------------------------------------------------------------
    patterns = {}
    detected_bands = sorted(
        {tok for col in columns for tok in ("alpha", "beta", "gamma", "delta", "theta")
         if tok in col.lower()}
    )
    if detected_bands:
        patterns["frequency_bands"] = detected_bands
    if total_outliers:
        patterns["potential_outliers"] = (
            f"{total_outliers} potential statistical outliers across numeric columns (IQR)"
        )

    return {
        "dataset": {"rows": rows, "columns": cols},
        "data_types": {
            "numeric": len(numeric),
            "categorical": len(categorical),
            "text": len(text_cols),
            "datetime": len(datetime_cols),
            "boolean": len(boolean_cols),
        },
        "quality": quality,
        "column_categories": {
            "numeric": _preview(numeric),
            "categorical": _preview(categorical),
            "text": _preview(text_cols),
            "datetime": _preview(datetime_cols),
            "boolean": _preview(boolean_cols),
        },
        "semantics": _build_semantics(columns),
        "statistics": {
            "important_numeric_columns": important,
            "global_min": {"value": global_min[0], "column": global_min[1]} if global_min else None,
            "global_max": {"value": global_max[0], "column": global_max[1]} if global_max else None,
        },
        "correlations": {
            "top_positive": [{"features": [a, b], "value": v} for a, b, v in pos_pairs[:5]],
            "top_negative": [{"features": [a, b], "value": v} for a, b, v in neg_pairs[:5]],
        },
        "categorical_breakdown": cat_report,
        "class_imbalance": imbalance,
        "anomalies": {
            "total_potential_outliers": total_outliers,
            "top_columns": outlier_rows[:top_outliers],
        },
        "domain": detect_domain(columns),
        "patterns": patterns,
    }


# ---- representative sample for LLM context ----------------------------------
# The full-data profile captures every column's statistics including the middle
# rows.  A small labelled sample (first / random / last) gives the LLM a feel
# for the actual row shapes without flooding the prompt with raw data.

_SAMPLE_STR_MAX = 60
_SAMPLE_PER_BUCKET = 5
_SAMPLE_SEED = 42


def _compact_sample_row(row: dict) -> dict:
    compact = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > _SAMPLE_STR_MAX:
            compact[key] = value[:_SAMPLE_STR_MAX] + "…"
        else:
            compact[key] = value
    return compact


def build_representative_sample(
    records,
    per_bucket: int = _SAMPLE_PER_BUCKET,
    seed: int = _SAMPLE_SEED,
) -> dict:
    """Small first / random / last excerpt of the dataset.

    Context for the LLM only.  Middle-data patterns are captured by the
    full-data statistics (build_data_profile), not by these rows.  The
    "random" bucket is seeded so the sample is reproducible.
    """
    rows = _norm_records(records)
    if not rows:
        return {"total_rows": 0, "per_bucket": per_bucket, "buckets": {}}

    from app.sampling.representative import (
        sample_first,
        sample_last,
        sample_random,
    )

    buckets = {
        "first": [_compact_sample_row(r) for r in sample_first(rows, per_bucket)],
        "random": [_compact_sample_row(r) for r in sample_random(rows, per_bucket, seed)],
        "last": [_compact_sample_row(r) for r in sample_last(rows, per_bucket)],
    }
    return {
        "total_rows": len(rows),
        "per_bucket": per_bucket,
        "buckets": buckets,
    }