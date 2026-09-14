"""Structured-report summaries for nested JSON.

A nested JSON is NOT a flat dataset and must not be force-fed the data
pipeline as "1 row, 7 columns". Universal rule still holds here:

    Python computes every fact
        - top-level scalar fields
        - each array section: count + kind (table vs list of items)
        - object arrays are profiled as their own mini-table (statuses,
          roles, metrics) - never raw prose
    LLM (optional) explains ONLY those computed facts
    deterministic summary explains the report on its own terms when the
    LLM is off
"""

from collections import Counter

from app.schemas.summary import SummaryOutput
from app.timing import mark, start


def is_nested_report(obj) -> bool:
    """True when a JSON object's values are lists/objects (depth > 1)."""
    return isinstance(obj, dict) and any(
        isinstance(v, (list, dict)) for v in obj.values()
    )


def _scalar(value):
    """Keep scalars as-is; never send nested structures to the sub-profiler."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_MAX_FIELDS = 12
_GENERIC_SEMANTIC = {
    "other detail",
    "temperature / physical reading",
    "status / category info",
}


def _is_marker(value) -> bool:
    """Marker strings from structure.py ('[grid 10x10]', '[nested list ...]')
    are structural state, not summarizable content - keep them out of prose."""
    return isinstance(value, str) and value.startswith("[") and value.endswith("]")


def _category_aggregates(profile: dict, cap_cols: int = 5, cap_vals: int = 5) -> list:
    """Value-frequency aggregates from a verified profile - Python computed
    counts, never raw rows.  Used to summarize large object arrays instead of
    dumping example items."""
    out = []
    for item in (profile.get("categorical_breakdown") or [])[:cap_cols]:
        col = item.get("column")
        if not col:
            continue
        top = (item.get("top_values") or [])[:cap_vals]
        out.append(
            {
                "field": col,
                "unique": item.get("unique", 0),
                "top": [
                    {"value": t.get("value"), "count": int(t.get("count", 0))}
                    for t in top
                ],
            }
        )
    return out


def _numeric_aggregates(profile: dict, cap: int = 6) -> list:
    """Min / max / mean / median per numeric column - keeps the LLM's
    quantitative claims grounded in computed facts."""
    out = []
    for item in (
        (profile.get("statistics") or {}).get("important_numeric_columns") or []
    )[:cap]:
        col = item.get("column")
        if not col:
            continue
        out.append(
            {
                "field": col,
                "min": item.get("min"),
                "max": item.get("max"),
                "mean": item.get("mean"),
                "median": item.get("median"),
            }
        )
    return out


def digest_report(obj: dict, include_spotlights: bool = True) -> dict:
    """Compute verified facts from a nested JSON report. Pure Python.

    `include_spotlights` adds readable value previews to the digest. These stay
    in Python and are shown to the user; they never reach the LLM, so the LLM
    digest is always computed with include_spotlights=False.

    Large object arrays are NOT sent to the LLM as items.  They are aggregated
    by Python into categorical value counts, numeric ranges and - only for the
    user-facing digest - a tiny representative preview.
    """
    title_fields = {}
    sections = []
    for key, value in obj.items():
        if isinstance(value, dict):
            scalars = {}
            for k, v in value.items():
                if isinstance(v, (list, dict)):
                    if isinstance(v, list) and v and all(
                        isinstance(x, (str, int, float, bool)) for x in v
                    ):
                        scalars[k] = ", ".join(str(x) for x in v[:20])
                    continue
                scalars[k] = _scalar(v)
            section = {
                "name": key,
                "kind": "object",
                "count": 1,
                "fields": list(value.keys()),
            }
            sections.append(section)
            for k, v in scalars.items():
                title_fields[f"{key}.{k}"] = v
        elif isinstance(value, list):
            dict_count = sum(1 for x in value if isinstance(x, dict))
            if value and dict_count >= len(value) * 0.5:
                from app.profile import build_data_profile
                from app.structure import flatten_record, scan_tokens
                from .deterministic import build_profile_summary

                dict_items = [x for x in value if isinstance(x, dict)]
                raw_records = [flatten_record(x) for x in dict_items]
                records = [
                    {k: v for k, v in r.items() if not _is_marker(v)}
                    for r in raw_records
                ]
                t0 = start()
                sub_profile = build_data_profile(records)
                sub = build_profile_summary(
                    sub_profile,
                    records=records if include_spotlights else None,
                )
                mark(f"report: profile section '{key}'", t0)
                n_rows = len(dict_items)
                section = {
                    "name": key,
                    "kind": "table",
                    "record_count": n_rows,
                    "column_count": sub.calculated_metrics.get("column_count", 0),
                    "summary": sub.executive_summary,
                    "findings": sub.key_findings[:3],
                    "metrics": sub.calculated_metrics,
                }
                section["categorical_values"] = _category_aggregates(sub_profile)
                section["numeric_stats"] = _numeric_aggregates(sub_profile)
                tokens = Counter()
                for item in dict_items:
                    scan_tokens(item, tokens)
                if tokens:
                    total = sum(tokens.values())
                    section["action_tokens"] = [
                        {"token": t, "count": n}
                        for t, n in tokens.most_common(12)
                    ]
                    section["action_tokens_total"] = total
                    section["action_token_count"] = len(tokens)
                sections.append(section)
            else:
                from app.structure import scan_tokens

                section = {
                    "name": key,
                    "kind": "list",
                    "count": len(value),
                }
                all_scalars = bool(value) and all(
                    isinstance(x, (str, int, float, bool)) for x in value
                )
                counts = Counter()
                if all_scalars:
                    for x in value:
                        counts[str(x)] += 1
                else:
                    for x in value:
                        scan_tokens(x, counts)
                if counts:
                    section["value_counts"] = [
                        {"value": v, "count": n}
                        for v, n in counts.most_common(_MAX_FIELDS)
                    ]
                sections.append(section)
        else:
            title_fields[key] = _scalar(value)
    return {"title_fields": title_fields, "sections": sections}


def _deterministic_summary(facts: dict) -> SummaryOutput:
    tf = facts["title_fields"]
    sections = facts["sections"]
    table_names = [s["name"] for s in sections if s["kind"] == "table"]
    list_names = [s["name"] for s in sections if s["kind"] == "list"]

    parts = []

    # Narrative lead: honour a human-readable description when the report has
    # one (generic field, no domain assumption). Falls back to the scalar facts.
    desc = (
        tf.get("description")
        or tf.get("info.description")
        or tf.get("summary")
        or tf.get("info.summary")
    )
    if desc and not _is_marker(desc):
        parts.append(str(desc).strip())
    elif tf:
        parts.append(
            "This structured report covers a single entity with these "
            + "; ".join(f"{k}: {v}" for k, v in list(tf.items())[:8])
            + "."
        )

    # Group nested-object scalars (e.g. configuration.episodeSteps) into one
    # readable "scenario settings" line instead of a flat key dump.
    config_like = [
        (k.split(".", 1)[1], v)
        for k, v in tf.items()
        if "." in k and not _is_marker(v) and not k.startswith("info.")
    ]
    if config_like:
        parts.append(
            "Scenario settings: "
            + ", ".join(f"{k} {v}" for k, v in config_like[:14])
            + "."
        )

    section_desc = []
    for s in sections:
        if s["kind"] == "table":
            label = f"{s['record_count']} records"
        elif s["kind"] == "list":
            label = f"{s['count']} items"
        else:
            label = "nested object"
        section_desc.append(f"{s['name']} ({label})")
    if section_desc:
        parts.append(
            "Its array sections are " + ", ".join(section_desc) + "."
        )
        if table_names:
            parts.append(
                "The object arrays are summarized as their own tables below: "
                + ", ".join(table_names)
                + "."
            )
            for s in sections:
                if s["kind"] == "table":
                    parts.append(f"{s['name']}: {s['summary'].strip()}")

    findings = []
    for s in sections:
        if s["kind"] == "table":
            tokens = s.get("action_tokens") or []
            if tokens:
                total = s.get("action_tokens_total", 0)
                shown = ", ".join(f"{t['token']} ({t['count']})" for t in tokens[:6])
                findings.append(
                    f"{s['name']}: {s['record_count']} records, {len(tokens)} distinct "
                    f"action tokens observed {total} time(s): {shown}."
                )
            findings.append(f"{s['name']}: {s['summary'].strip()}")
            findings.extend(
                f for f in s["findings"] if f.lower() not in _GENERIC_SEMANTIC
            )
            for agg in s.get("categorical_values") or []:
                top = [
                    t["value"] for t in (agg.get("top") or [])
                    if len(str(t["value"])) <= 60 and "[" not in str(t["value"])
                ]
                if not top:
                    continue
                shown = ", ".join(
                    f"{t['value']} ({t['count']})"
                    for t in (agg.get("top") or [])
                    if t["value"] in top
                )
                findings.append(
                    f"{s['name']}.{agg['field']}: {agg['unique']} distinct - top: {shown}."
                )
        elif s["kind"] == "list":
            counts = s.get("value_counts") or []
            if counts:
                shown = "; ".join(f"{c['value']} ({c['count']})" for c in counts[:8])
                findings.append(f"{s['name']} ({s['count']} items): {shown}.")
        else:
            section_tf = {k: v for k, v in tf.items() if k.startswith(s["name"] + ".")}
            if section_tf:
                shown = "; ".join(
                    f"{k[len(s['name']) + 1:]}: {v}" for k, v in section_tf.items()
                )
                findings.append(f"{s['name']}: {shown}")
            else:
                findings.append(
                    f"{s['name']} contains fields: {', '.join(s['fields'])}"
                )

    metrics = {
        "top_level_fields": len(tf),
        "array_sections": len(sections),
    }
    for s in sections:
        metrics[f"count.{s['name']}"] = s.get("record_count") or s.get("count", 0)
        if s["kind"] == "table":
            metrics[f"columns.{s['name']}"] = s.get("column_count", 0)

    recs = []
    issue_section = next((s for s in sections if s["name"].lower() in ("issues", "risks", "problems") and s["kind"] == "list"), None)
    if issue_section and issue_section["count"]:
        first = ", ".join(f"{i!r}" for i in (issue_section.get("items") or issue_section.get("value_counts") or [])[:2])
        recs.append(f"Resolve the {issue_section['count']} open issue(s) - including {first}.")
    next_section = next((s for s in sections if s["name"].lower() in ("next_steps", "next steps", "plan", "todo")), None)
    if next_section:
        recs.append(f"Proceed with the outlined {next_section.get('record_count') or next_section.get('count', 0)} next step(s).")
    if not recs:
        recs.append("No explicit issues or next steps are listed in this report.")

    title_source = tf.get("info.title") or tf.get("title") or tf.get("report_title") or tf.get("name") or tf.get("company")
    if title_source:
        title = f"{title_source} Report"
    else:
        title = f"Structured report - {len(tf)} fields"
    executive_summary = " ".join(parts)

    return SummaryOutput(
        title=title,
        executive_summary=executive_summary,
        key_findings=findings,
        calculated_metrics=metrics,
        anomalies_detected=[],
        recommendations=recs,
    )


def build_report_summary(
    obj: dict,
    prefer_llm: bool = False,
    llm_service=None,
) -> SummaryOutput:
    # LLM digest is always computed without value spotlights: names and
    # raw items stay in Python (shown to the user), never in the prompt.
    facts = digest_report(obj, include_spotlights=not prefer_llm)

    if prefer_llm and llm_service is not None:
        try:
            t0 = start()
            out = llm_service.generate_summary_from_profile(
                {"report_digest": facts}, auto_fallback=True
            )
            mark(f"report: LLM narrative", t0)
            return out
        except Exception:
            import logging

            logging.getLogger("structured_report").exception(
                "LLM failed for structured report; using deterministic summary"
            )

    return _deterministic_summary(facts)