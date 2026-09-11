import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

SECTION_HEADERS = [
    "DATASET SCHEMA",
    "DATA QUALITY",
    "CALCULATED STATISTICS",
    "TRENDS",
    "ANOMALIES",
    "REPRESENTATIVE RECORDS",
]


def _load_template(template: str) -> tuple[str, str]:
    """Return (system_instructions, output_requirements) from the txt template."""
    text = (PROMPTS_DIR / f"{template}.txt").read_text(encoding="utf-8")
    if "[OUTPUT]" in text:
        system, output = text.split("[OUTPUT]", 1)
        return system.strip(), output.strip()
    return text.strip(), ""


def _fmt_schema(schema) -> str:
    return "\n".join(f"- {c['name']} ({c['dtype']})" for c in schema)


def _fmt_quality(quality) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in quality.items())


def _fmt_statistics(statistics) -> str:
    parts = []

    numeric = statistics.get("numeric", {})
    for col, stats in numeric.items():
        pairs = ", ".join(f"{k}={v}" for k, v in stats.items())
        parts.append(f"- {col}: {pairs}")

    categorical = statistics.get("categorical", {})
    for col, stats in categorical.items():
        pairs = ", ".join(f"{k}={v}" for k, v in stats.items())
        parts.append(f"- {col}: {pairs}")

    dates = statistics.get("dates", {})
    for col, stats in dates.items():
        pairs = ", ".join(f"{k}={v}" for k, v in stats.items())
        parts.append(f"- {col}: {pairs}")

    duplicates = statistics.get("duplicates", {})
    if duplicates:
        pairs = ", ".join(f"{k}={v}" for k, v in duplicates.items())
        parts.append(f"- duplicates: {pairs}")

    return "\n".join(parts)


def _fmt_trends(analytics) -> str:
    parts = []
    for col, trend in analytics.get("trends", {}).items():
        parts.append(f"- {col}: {trend}")

    for cat_col, agg in analytics.get("aggregations", {}).items():
        parts.append(f"- {cat_col} (top): {agg.get('top')}")
        parts.append(f"- {cat_col} (bottom): {agg.get('bottom')}")

    correlations = analytics.get("correlations", {})
    for pair, corr in correlations.items():
        parts.append(f"- correlation {pair}: {corr}")

    return "\n".join(parts)


def _fmt_anomalies(analytics) -> str:
    if not analytics:
        return "- none detected"
    parts = []
    for col, result in analytics.get("anomalies", {}).items():
        parts.append(f"- {col}: {result}")

    if not analytics.get("anomalies"):
        parts.append("- none detected")
    return "\n".join(parts)


def _fmt_samples(rows) -> str:
    return "\n".join(
        f"{i}. {json.dumps(row, default=str, ensure_ascii=False)}"
        for i, row in enumerate(rows, start=1)
    )


def _json_output_instruction() -> str:
    """STRICT JSON requirement that mirrors the Pydantic validation schema."""
    from app.schemas.summary import SummaryOutput

    schema = SummaryOutput.model_json_schema()
    props = json.dumps(schema["properties"], indent=2)
    return (
        "Return your response as STRICT JSON only - no markdown, no code fences, "
        "no extra prose before or after the JSON object.\n"
        f"Expected structure:\n{props}\n\n"
        '"calculated_metrics" may contain any metric names and values; '
        "every other field must be present and match its type exactly.\n"
        "The structure above lists field NAMES and TYPES. Your JSON must contain the "
        "actual VALUES directly for each key. Never wrap a value in an object "
        "containing 'type' or '$ref' - for example 'title' must be a plain string, "
        "'key_findings' a flat list of strings, never a schema description."
    )


def _extract_context(data) -> dict:
    """Normalize any accepted input into a context dict + extra sections."""
    if isinstance(data, list):
        from app.sampling.representative import build_context
        from app.analytics import compute_analytics
        from app.quality import check_quality

        context = build_context(data)
        quality = check_quality(data)
        analytics = compute_analytics(data)
        return {
            "context": context,
            "sections": {"quality": quality, "analytics": analytics},
        }

    if isinstance(data, dict) and "context" in data:
        payload = dict(data)
        context = payload.pop("context")
        return {
            "context": context,
            "sections": {
                "analytics": payload.get("analytics"),
                "quality": payload.get("quality"),
            },
        }

    # already a context dict (build_context output)
    return {
        "context": data,
        "sections": {
            "analytics": data.get("analytics"),
            "quality": data.get("quality"),
        },
    }


def _apply_mode(system: str, mode: str) -> str:
    """Append the mode's focus areas to the SYSTEM INSTRUCTIONS block."""
    from app.modes import get_mode

    config = get_mode(mode)
    focus = config["focus"]
    if not focus:
        return system
    focus_block = "\n".join(f"- {item}" for item in focus)
    return f"{system}\n\nFocus on:\n{focus_block}"


def build_prompt(data, mode: str = "system", template: str | None = None) -> str:
    """
    Assemble the full LLM prompt with sections:
      SYSTEM INSTRUCTIONS, DATASET SCHEMA, DATA QUALITY, CALCULATED STATISTICS,
      TRENDS, ANOMALIES, REPRESENTATIVE RECORDS, OUTPUT REQUIREMENTS.

    `data` may be:
      - a list of raw records           (everything is computed deterministically)
      - a build_context() output dict
      - a pipeline payload dict with a "context" key

    `mode` selects a summary configuration (system/executive/financial/technical).
    A raw `template` name is accepted for backwards compatibility.
    """
    if template is not None:
        mode = template

    from app.modes import get_mode

    config = get_mode(mode)
    system, output = _load_template(config["template"])
    system = _apply_mode(system, mode)

    extracted = _extract_context(data)
    context = extracted["context"]
    sections = extracted["sections"]

    parts = [system]

    schema = context.get("schema")
    if schema:
        parts.append("DATASET SCHEMA\n" + _fmt_schema(schema))

    quality = sections.get("quality") or context.get("quality")
    if quality:
        parts.append("DATA QUALITY\n" + _fmt_quality(quality))

    statistics = context.get("statistics")
    if statistics:
        parts.append("CALCULATED STATISTICS\n" + _fmt_statistics(statistics))

    analytics = sections.get("analytics")
    if analytics:
        if analytics.get("trends") or analytics.get("aggregations") or analytics.get("correlations"):
            parts.append("TRENDS\n" + _fmt_trends(analytics))
        parts.append("ANOMALIES\n" + _fmt_anomalies(analytics))

    sample = context.get("sample") or {}
    sample_rows = sample.get("rows", []) if isinstance(sample, dict) else []
    if isinstance(sample, list):
        sample_rows = sample
    if sample_rows:
        parts.append("REPRESENTATIVE RECORDS\n" + _fmt_samples(sample_rows))

    if output:
        parts.append("OUTPUT REQUIREMENTS\n" + output)

    parts.append("JSON OUTPUT (STRICT)\n" + _json_output_instruction())

    return "\n\n".join(parts)