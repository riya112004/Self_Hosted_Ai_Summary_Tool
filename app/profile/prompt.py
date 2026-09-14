"""Prompt contract for profile-based summarization.

The LLM receives the verified data profile (full-data statistics) plus a
small representative sample (first / random / last rows) for context.  The
sample lets the model see actual row shapes; every quantitative claim must
still be grounded in the verified statistics.
"""

import json

from app.llm.universal import build_system_section

SUMMARY_SCHEMA_TEXT = """{
  "title": string,
  "executive_summary": string,
  "key_findings": [string, ...],
  "calculated_metrics": { "short_name": number_or_string, ... },
  "anomalies_detected": [string, ...] (empty if none detected),
  "recommendations": [string, ...]
}"""


def _render_sample(sample: dict) -> str:
    total = sample.get("total_rows", 0)
    per = sample.get("per_bucket", 0)
    buckets = sample.get("buckets") or {}
    lines = [f"Excerpt from {total} total rows ({per} rows per bucket):"]
    for label in ("first", "random", "last"):
        rows = buckets.get(label) or []
        if not rows:
            continue
        tag = label.upper() if label != "random" else "RANDOM (seeded)"
        entries = "\n".join(
            f"- {i}. {json.dumps(r, default=str, ensure_ascii=False)}"
            for i, r in enumerate(rows, 1)
        )
        lines.append(f"{tag}\n{entries}")
    return "\n\n".join(lines)


def build_profile_prompt(
    profile: dict, sample: dict | None = None
) -> str:
    blocks = [
        build_system_section(kind="data")
        + "\n\nVERIFIED DATA PROFILE (computed by Python from ALL rows — "
        "do not recalculate):\n"
        + json.dumps(profile, ensure_ascii=False, indent=2),
    ]

    if sample:
        blocks.append(
            "REPRESENTATIVE SAMPLE (illustrative context only — middle-data "
            "patterns are already captured by the statistics above):\n"
            + _render_sample(sample)
        )

    blocks.append(
        "JSON OUTPUT (STRICT) - target schema:\n"
        + SUMMARY_SCHEMA_TEXT
        + "\n\nKeep executive_summary between 200 and 400 words - plain, human "
        "language about what the dataset represents. "
        "Preserve the exact numbers from the verified profile."
    )
    return "\n\n".join(blocks)