"""Deterministic, LLM-free summary builder.

Builds a complete SummaryOutput straight from the computed analytics,
profilng and quality checks. Runs in microseconds on any dataset and is
model-independent - it is the universal fallback when no inference server
is configured, and the instant path when ``run_llm=false``.
"""

from concurrent.futures import ThreadPoolExecutor

from app.profiling.schema_detector import detect_schema
from app.profiling.profiler import profile_dataset
from app.quality import check_quality
from app.analytics import compute_analytics
from app.schemas.summary import SummaryOutput


def _clean(value, ndigits: int = 4):
    """Round floats for readable output without fabricating precision."""
    if isinstance(value, float):
        return round(value, ndigits)
    return value


def _value_spotlights(records: list[dict] | None) -> list[str]:
    """Short categorical labels worth mentioning - computed locally in Python,
    shown to the user, never sent to any LLM. Free prose and id-like long
    tokens are skipped."""
    if not records:
        return []
    from collections import Counter, defaultdict

    values: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        if not isinstance(r, dict):
            continue
        for key, v in r.items():
            if isinstance(v, str) and 0 < len(v) <= 40 and not v.startswith(("http", "@")):
                values[key][v.strip()] += 1

    spotlights = []
    for key, counter in values.items():
        uniq = len(counter)
        if uniq > 15:
            continue  # id-like / high-cardinality -> no signal
        total = sum(counter.values())
        top = counter.most_common(5)
        shown = ", ".join(f"'{v}' x{n}" if n > 1 else f"'{v}'" for v, n in top)
        spotlights.append(f"{key} ({uniq} distinct): {shown}")
    return spotlights[:5]


def _anomaly_lines(analytics: dict) -> list[str]:
    lines = []
    for col, res in (analytics.get("anomalies") or {}).items():
        count = res.get("count", 0) if isinstance(res, dict) else 0
        if count:
            bounds = (
                f", bounds [{_clean(res.get('lower_bound'))}, "
                f"{_clean(res.get('upper_bound'))}]"
                if isinstance(res, dict) and res.get("lower_bound") is not None
                else ""
            )
            lines.append(f"{col}: {count} outlier(s){bounds}")
    return lines


def _executive_summary(records, quality, analytics, schema, profile) -> str:
    numeric = len([c for c in schema if c["dtype"] == "numeric"])
    categorical = len([c for c in schema if c["dtype"] == "categorical"])
    date_cols = len([c for c in schema if c["dtype"] == "date"])
    outlier_total = sum(
        res.get("count", 0) if isinstance(res, dict) else 0
        for res in (analytics.get("anomalies") or {}).values()
    )

    parts = [
        f"This dataset contains {len(records)} records across {len(schema)} fields "
        f"({numeric} numeric, {categorical} categorical, {date_cols} date)."
    ]
    quality_score = quality.get("quality_score")
    if quality_score is not None:
        verdict = "high" if quality_score >= 80 else ("moderate" if quality_score >= 50 else "low")
        parts.append(f"Data quality is {verdict} (score {quality_score}).")
    if outlier_total:
        parts.append(f"Anomaly screening flagged {outlier_total} outlier value(s).")
    if (analytics.get("trends") or {}):
        first = list(analytics["trends"].items())[0]
        parts.append(f"Latest trend signal: {first[0]} = {_clean(first[1])}.")
    return " ".join(parts)


def _key_findings(analytics: dict, profile: dict, quality: dict) -> list[str]:
    findings = []

    for cat_col, agg in (analytics.get("aggregations") or {}).items():
        top = (agg.get("top") or {}) if isinstance(agg, dict) else {}
        bottom = (agg.get("bottom") or {}) if isinstance(agg, dict) else {}
        if top:
            key, value = next(iter(top.items()))
            findings.append(f"Top {cat_col}: {key} (metric {_clean(value)}).")
        if bottom:
            key, value = next(iter(bottom.items()))
            findings.append(f"Lowest {cat_col}: {key} (metric {_clean(value)}).")

    for col, res in (analytics.get("anomalies") or {}).items():
        count = res.get("count", 0) if isinstance(res, dict) else 0
        if count:
            findings.append(f"{col} has {count} potential outlier value(s).")

    for pair, corr in (analytics.get("correlations") or {}).items():
        if isinstance(corr, (int, float)) and abs(corr) >= 0.7:
            strength = "positive" if corr > 0 else "negative"
            findings.append(f"Correlation {pair} is strongly {strength} ({_clean(corr)}).")

    for col, stats in (profile.get("numeric") or {}).items():
        if stats.get("null_count"):
            findings.append(f"{col} has {stats['null_count']} missing value(s).")

    if not findings:
        findings.append(
            "No outliers, strong correlations or missing values detected in this sample."
        )
    return findings


def _recommendations(analytics: dict, quality: dict) -> list[str]:
    recs = []

    code = quality.get("code")
    if code:
        recs.append(f"Data quality check raised: {code}.")

    for col, res in (analytics.get("anomalies") or {}).items():
        count = res.get("count", 0) if isinstance(res, dict) else 0
        if count:
            recs.append(f"Investigate the {count} outlier value(s) flagged in {col}.")

    for pair, corr in (analytics.get("correlations") or {}).items():
        if isinstance(corr, (int, float)) and abs(corr) >= 0.7:
            recs.append(f"Review the strong correlation between {pair.replace('-', ' and ')}.")

    if not recs:
        recs.append("Data looks healthy - keep periodic anomaly screening enabled.")
    return recs


def build_profile_summary(
    profile: dict, records: list[dict] | None = None
) -> SummaryOutput:
    """Deterministic SummaryOutput built from the verified data profile.

    Explains what the raw data actually is in plain language: semantic
    dataset kind, per-column meaning, value categories and notable values -
    no LLM involved. Optionally takes the (normalised) records to add
    readable value spotlights - these stay in Python and are shown to the
    user; they never reach the LLM.
    """

    def _num(v, nd=2):
        return _clean(v, nd) if isinstance(v, (int, float)) else v

    def _type_count(v):
        # streamed profiles store column-name lists here; in-memory stores counts
        return len(v) if isinstance(v, (list, tuple, set)) else (v or 0)

    dataset = profile.get("dataset") or {}
    rows = dataset.get("rows", 0)
    cols = dataset.get("columns", 0)
    dt = profile.get("data_types") or {}
    _stats = profile.get("statistics") or {}
    quality = profile.get("quality") or {}
    domain = profile.get("domain") or {}

    semantics = profile.get("semantics") or {}
    kind = semantics.get("kind") or "tabular"
    summary_text = semantics.get("summary") or (
        f"This dataset has {rows} record(s) across {cols} column(s)."
    )
    categories = semantics.get("categories") or []
    roles = semantics.get("column_roles") or {}
    col_roles = [
        f"{col.replace('_', ' ')} - {roles[col].replace('_', ' ')}"
        for col in (profile.get("columns") or [])[:20] if col in roles
    ]
    findings = list(categories)
    if col_roles:
        findings += ["Field meanings: " + "; ".join(col_roles) + "."]

    exec_parts = [summary_text]
    value_spotlights = _value_spotlights(records)
    if value_spotlights:
        exec_parts.append("Notable values: " + "; ".join(value_spotlights) + ".")
    if domain.get("label") and domain.get("label") != "Unknown" and domain.get("confidence", 0) >= 0.6:
        exec_parts.append(
            f"The column names suggest the domain '{domain['label']}'."
        )
    missing = quality.get("missing_values_total", 0)
    if missing:
        exec_parts.append(f"There are {missing} missing value(s) to be aware of.")

    type_parts = []
    for label, key in (
        ("numeric", "numeric"), ("categorical", "categorical"), ("text", "text"),
        ("date", "datetime"), ("boolean", "boolean"),
    ):
        n = _type_count(dt.get(key, 0))
        if n:
            type_parts.append(f"{n} {label}")
    if type_parts:
        exec_parts.append("Columns break down into " + ", ".join(type_parts) + ".")

    calculated = {
        "row_count": rows,
        "column_count": cols,
        "dataset_kind": kind,
        "numeric_columns": _type_count(dt.get("numeric", 0)),
        "categorical_columns": _type_count(dt.get("categorical", 0)),
        "missing_values_total": missing,
    }
    if domain.get("label") and domain.get("label") != "Unknown":
        calculated["domain"] = f"{domain['label']} ({_num(domain.get('confidence'), 3)})"

    return SummaryOutput(
        title=f"general summary - {rows} records, {cols} columns",
        executive_summary=" ".join(exec_parts),
        key_findings=findings,
        calculated_metrics={k: _clean(v) for k, v in calculated.items()},
        anomalies_detected=[],
        recommendations=[],
    )


def build_fast_summary(
    records: list[dict]
) -> SummaryOutput:
    """Deterministic SummaryOutput - no LLM involved, never raises."""
    schema = detect_schema(records)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_profile = executor.submit(profile_dataset, records)
        future_quality = executor.submit(check_quality, records)
        future_analytics = executor.submit(compute_analytics, records)
        profile = future_profile.result()
        quality = future_quality.result()
        analytics = future_analytics.result()

    title = f"general summary - {len(records)} records, {len(schema)} fields"
    calculated = {
        "row_count": len(records),
        "column_count": len(schema),
        "quality_score": quality.get("quality_score"),
        **{
            f"{col}_outliers": (
                res.get("count", 0) if isinstance(res, dict) else 0
            )
            for col, res in (analytics.get("anomalies") or {}).items()
        },
        **{
            f"{col}_trend": value
            for col, value in (analytics.get("trends") or {}).items()
        },
    }

    return SummaryOutput(
        title=title,
        executive_summary=_executive_summary(
            records, quality, analytics, schema, profile
        ),
        key_findings=_key_findings(analytics, profile, quality),
        calculated_metrics={k: _clean(v) for k, v in calculated.items()},
        anomalies_detected=_anomaly_lines(analytics),
        recommendations=_recommendations(analytics, quality),
    )