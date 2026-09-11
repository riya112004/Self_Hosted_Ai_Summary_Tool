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


_MAX_ITEM_CHARS = 160
_MAX_ITEMS = 8


def digest_report(obj: dict, mode: str = "system", include_spotlights: bool = True) -> dict:
    """Compute verified facts from a nested JSON report. Pure Python.

    `include_spotlights` adds readable value previews (employee names,
    statuses) to the digest. These stay in Python and are shown to the user;
    they never reach the LLM, so the LLM digest is always computed with
    include_spotlights=False.
    """
    title_fields = {}
    sections = []
    for key, value in obj.items():
        if isinstance(value, dict):
            sections.append(
                {
                    "name": key,
                    "kind": "object",
                    "count": 1,
                    "fields": list(value.keys()),
                }
            )
        elif isinstance(value, list):
            if value and all(isinstance(x, dict) for x in value):
                from app.profile import build_data_profile
                from .deterministic import build_profile_summary

                records = [{k: _scalar(v) for k, v in x.items()} for x in value]
                t0 = start()
                sub = build_profile_summary(
                    build_data_profile(records),
                    mode,
                    records=records if include_spotlights else None,
                )
                mark(f"report: profile section '{key}'", t0)
                sections.append(
                    {
                        "name": key,
                        "kind": "table",
                        "record_count": len(value),
                        "summary": sub.executive_summary,
                        "findings": sub.key_findings[:3],
                        "metrics": sub.calculated_metrics,
                    }
                )
            else:
                items = [str(x) for x in value]
                sections.append(
                    {
                        "name": key,
                        "kind": "list",
                        "count": len(items),
                        "items": [
                            i[:_MAX_ITEM_CHARS] for i in items[:_MAX_ITEMS]
                        ],
                    }
                )
        else:
            title_fields[key] = _scalar(value)
    return {"title_fields": title_fields, "sections": sections}


def _deterministic_summary(facts: dict, mode: str) -> SummaryOutput:
    tf = facts["title_fields"]
    sections = facts["sections"]
    table_names = [s["name"] for s in sections if s["kind"] == "table"]
    list_names = [s["name"] for s in sections if s["kind"] == "list"]

    parts = []
    top = ", ".join(f"{k}: {v}" for k, v in tf.items())
    if top:
        parts.append(f"This structured report has {len(tf)} top-level fields: {top}.")

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
            findings.append(f"{s['name']}: {s['summary'].strip()}")
            findings.extend(s["findings"])
        elif s["kind"] == "list":
            preview = "; ".join(s["items"])[:_MAX_ITEM_CHARS * 3]
            findings.append(f"{s['name']} ({s['count']} items): {preview}")
        else:
            findings.append(f"{s['name']} contains fields: {', '.join(s['fields'])}")

    metrics = {
        "top_level_fields": len(tf),
        "array_sections": len(sections),
    }
    for s in sections:
        metrics[f"count.{s['name']}"] = s.get("record_count") or s.get("count", 0)

    recs = []
    issue_section = next((s for s in sections if s["name"].lower() in ("issues", "risks", "problems") and s["kind"] == "list"), None)
    if issue_section and issue_section["count"]:
        first = ", ".join(f"{i!r}" for i in issue_section["items"][:2])
        recs.append(f"Resolve the {issue_section['count']} open issue(s) - including {first}.")
    next_section = next((s for s in sections if s["name"].lower() in ("next_steps", "next steps", "plan", "todo")), None)
    if next_section:
        recs.append(f"Proceed with the outlined {next_section.get('count') or next_section.get('record_count', 0)} next step(s).")
    if not recs:
        recs.append("No explicit issues or next steps are listed in this report.")

    title_source = tf.get("title") or tf.get("report_title") or tf.get("name") or tf.get("company")
    title = f"Report: {title_source}" if title_source else f"Structured report - {len(tf)} fields"
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
    mode: str = "system",
    prefer_llm: bool = False,
    llm_service=None,
) -> SummaryOutput:
    # LLM digest is always computed without value spotlights: names and
    # raw items stay in Python (shown to the user), never in the prompt.
    facts = digest_report(obj, mode, include_spotlights=not prefer_llm)

    if prefer_llm and llm_service is not None:
        try:
            t0 = start()
            out = llm_service.generate_summary_from_profile(
                {"report_digest": facts}, mode, auto_fallback=True
            )
            mark(f"report: LLM narrative (mode={mode})", t0)
            return out
        except Exception:
            import logging

            logging.getLogger("structured_report").exception(
                "LLM failed for structured report; using deterministic summary"
            )

    return _deterministic_summary(facts, mode)