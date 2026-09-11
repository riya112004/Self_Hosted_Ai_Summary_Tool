"""Prompt contract for profile-based summarization.

The LLM receives ONLY the verified data profile, never raw records. The
system rules themselves are the shared universal block - nothing here is
hard-coded per input type, only the way the verified facts are presented.
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


def build_profile_prompt(profile: dict, mode: str = "system") -> str:
    return (
        build_system_section(mode, kind="data")
        + "\n\nVERIFIED DATA PROFILE (computed by Python, do not recalculate):\n"
        + json.dumps(profile, ensure_ascii=False, indent=2)
        + "\n\nJSON OUTPUT (STRICT) - target schema:\n"
        + SUMMARY_SCHEMA_TEXT
    )