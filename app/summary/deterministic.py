"""Deterministic, LLM-free summary builder.

Builds a complete SummaryOutput straight from the computed
profilng and quality checks. Runs in microseconds on any dataset and is
model-independent - it is the universal fallback when no inference server
is configured, and the instant path when ``run_llm=false``.
"""

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
