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
    profile: dict, mode: str = "system", records: list[dict] | None = None
) -> SummaryOutput:
    """Deterministic SummaryOutput built from the verified data profile.

    Explains the dataset on its own terms: column meaning, value ranges,
    dominant categories, correlations, outliers - no LLM involved. Optionally
    takes the (normalised) records to add readable value spotlights - these
    stay in Python and are shown to the user; they never reach the LLM.
    """

    def _num(v, nd=2):
        return _clean(v, nd) if isinstance(v, (int, float)) else v

    dataset = profile.get("dataset") or {}
    rows = dataset.get("rows", 0)
    cols = dataset.get("columns", 0)
    dt = profile.get("data_types") or {}
    stats = profile.get("statistics") or {}
    important = stats.get("important_numeric_columns") or []
    corr = profile.get("correlations") or {}
    anomalies = profile.get("anomalies") or {}
    quality = profile.get("quality") or {}
    domain = profile.get("domain") or {}
    cat_report = profile.get("categorical_breakdown") or []
    imbalance = profile.get("class_imbalance") or []
    total_out = anomalies.get("total_potential_outliers", 0)

    # ---- narrative -------------------------------------------------------
    parts = [f"This dataset has {rows} record(s) across {cols} column(s)."]
    type_parts = []
    for label, key in (
        ("numeric", "numeric"), ("categorical", "categorical"), ("text", "text"),
        ("date", "datetime"), ("boolean", "boolean"),
    ):
        n = dt.get(key, 0)
        if n:
            type_parts.append(f"{n} {label}")
    if type_parts:
        parts.append("Columns break down into " + ", ".join(type_parts) + ".")

    if important[:3]:
        chunks = []
        for c in important[:3]:
            name, lo, hi = c["column"], c.get("min"), c.get("max")
            extra = f", average {_num(c.get('mean'))}"
            if c.get("median") is not None and c.get("mean") != c.get("median"):
                extra += f", median {_num(c.get('median'))}"
            chunks.append(f"{name} ranges {_num(lo)} to {_num(hi)}{extra}")
        parts.append("Key numeric metrics: " + "; ".join(chunks) + ".")

    if cat_report:
        pieces = []
        for c in cat_report[:3]:
            topv = (c.get("top_values") or [{}])[0]
            pieces.append(
                f"{c['column']} has {c.get('unique')} unique values, most common "
                f"'{topv.get('value', '')}'"
            )
        parts.append("Categories: " + "; ".join(pieces) + ".")

    value_spotlights = _value_spotlights(records)
    if value_spotlights:
        parts.append("Notable values: " + "; ".join(value_spotlights) + ".")

    if total_out:
        parts.append(f"IQR screening flagged {total_out} potential outlier value(s).")
    missing = quality.get("missing_values_total", 0)
    if missing:
        parts.append(f"There are {missing} missing value(s) to be aware of.")
    dup = quality.get("duplicate_rows", 0)
    if dup:
        parts.append(f"{dup} duplicate row(s) were detected.")
    if domain.get("label") and domain.get("label") != "Unknown" and domain.get("confidence", 0) >= 0.6:
        parts.append(
            f"Column names suggest the domain '{domain['label']}' "
            f"(confidence {_num(domain.get('confidence'), 3)})."
        )

    # ---- key findings -----------------------------------------------------
    findings = []
    for c in important[:5]:
        findings.append(
            f"{c['column']} ranges {_num(c.get('min'))} to {_num(c.get('max'))} "
            f"(average {_num(c.get('mean'))}, median {_num(c.get('median'))})."
        )
    for c in cat_report[:4]:
        topv = (c.get("top_values") or [{}])[0]
        findings.append(
            f"{c['column']} is dominated by '{topv.get('value', '-')}' "
            f"({c.get('unique')} unique values)."
        )
    for it in (corr.get("top_positive") or [])[:2]:
        fs = it.get("features") or []
        findings.append(f"{fs[0]} and {fs[1]} correlate ({_num(it.get('value'), 3)}).")
    for it in (corr.get("top_negative") or [])[:2]:
        fs = it.get("features") or []
        findings.append(f"{fs[0]} and {fs[1]} move in opposite directions ({_num(it.get('value'), 3)}).")
    for c in (anomalies.get("top_columns") or [])[:4]:
        findings.append(
            f"{c['column']} has {c['outliers']} potential outlier(s) "
            f"({round((c.get('share') or 0) * 100)}% of rows)."
        )
    for it in imbalance[:3]:
        findings.append(
            f"{it['column']} is imbalanced - {round((it.get('dominant_class_share') or 0) * 100)}% "
            f"of rows fall in one class."
        )
    for spotlight in value_spotlights:
        findings.append(spotlight)
    if not findings:
        findings.append("No strong patterns detected - the columns are relatively uniform.")

    # ---- recommendations --------------------------------------------------
    recs = []
    for c in (anomalies.get("top_columns") or [])[:4]:
        recs.append(f"Review the {c['outliers']} flagged value(s) in {c['column']}.")
    if missing:
        recs.append("Inspect the missing values before drawing conclusions.")
    dup = quality.get("duplicate_rows", 0)
    if dup:
        recs.append("Consider de-duplicating the data before deeper analysis.")
    for c in (quality.get("constant_columns") or [])[:3]:
        recs.append(f"'{c}' is constant and adds no signal - safe to exclude.")
    for it in (corr.get("top_positive") or [])[:1]:
        fs = it.get("features") or []
        recs.append(f"{fs[0]} and {fs[1]} carry almost the same information - one may be redundant.")
    if not recs:
        recs.append("Enable 'run_llm' for a narrative, interpretation-driven explanation.")

    # ---- structured fields ------------------------------------------------
    calculated = {
        "row_count": rows,
        "column_count": cols,
        "numeric_columns": dt.get("numeric", 0),
        "categorical_columns": dt.get("categorical", 0),
        "missing_values_total": missing,
        "duplicate_rows": dup,
        "constant_column_count": len(quality.get("constant_columns") or []),
        "potential_outliers": total_out,
        **{
            f"{c['column']}_outliers": c["outliers"]
            for c in (anomalies.get("top_columns") or [])
        },
    }
    if domain.get("label") and domain.get("label") != "Unknown":
        calculated["domain"] = f"{domain['label']} ({_num(domain.get('confidence'), 3)})"

    anomaly_lines = [
        f"{c['column']}: {c['outliers']} outlier(s) "
        f"({round((c.get('share') or 0) * 100)}% of rows)"
        for c in (anomalies.get("top_columns") or [])[:8]
    ]

    return SummaryOutput(
        title=f"{mode} summary - {rows} records, {cols} columns",
        executive_summary=" ".join(parts),
        key_findings=findings,
        calculated_metrics={k: _clean(v) for k, v in calculated.items()},
        anomalies_detected=anomaly_lines,
        recommendations=recs,
    )


def build_fast_summary(
    records: list[dict], mode: str = "system"
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

    title = f"{mode} summary - {len(records)} records, {len(schema)} fields"
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