import json
import re

from pydantic import BaseModel


class SummaryOutput(BaseModel):
    """Pydantic validation layer for structured LLM output.

    ``fact_guard`` is populated by the FactGuard layer (app.llm.guard) after
    the LLM output is parsed - it is never expected from the model itself.
    """

    title: str
    executive_summary: str
    key_findings: list[str]
    calculated_metrics: dict
    anomalies_detected: list[str]
    recommendations: list[str]
    fact_guard: dict | None = None


class StructuredOutputError(ValueError):
    """Raised when the LLM output cannot be parsed or validated."""

    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.raw = raw


def extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM text (tolerates code fences + surrounding prose)."""
    if not text or not text.strip():
        raise StructuredOutputError("LLM returned empty output", text or "")

    stripped = re.sub(r"```(?:json)?", "", text).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise StructuredOutputError("No JSON object found in LLM output", text)

    candidate = stripped[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            f"Invalid JSON in LLM output: {exc}", text
        ) from exc


def parse_summary(text: str, strict: bool = True) -> SummaryOutput:
    """
    Validate raw LLM text against SummaryOutput.

    strict=True  -> raise StructuredOutputError on any failure.
    strict=False -> best-effort: fill missing fields, keep raw text as fallback.
    """
    try:
        data = extract_json(text)
    except StructuredOutputError as exc:
        if strict:
            raise
        return _fallback_summary(text)

    try:
        return SummaryOutput.model_validate(data)
    except Exception as exc:
        if strict:
            raise StructuredOutputError(
                f"LLM output failed schema validation: {exc}", text
            ) from exc
        return _fallback_summary(text, data)


def _unwrap(value):
    """
    Fix the classic small-model failure: the model echoes the JSON schema
    back instead of values, e.g. "title": {"title": "Sales Analysis",
    "type": "string"}. Extract the real value when a dict looks like an
    echoed schema entry.
    """
    if isinstance(value, dict) and any(k in value for k in ("type", "$ref")):
        for key in ("value", "content", "title", "label", "default"):
            if key in value and not isinstance(value[key], (dict, list)):
                return value[key]
    return value


def _fallback_summary(text: str, data: dict | None = None) -> SummaryOutput:
    """Best-effort SummaryOutput when the LLM output isn't schema-compliant."""
    if data is None:
        data = {}
    title = _unwrap(data.get("title", "Data Summary"))
    exec_summary = _unwrap(data.get("executive_summary", text[:500]) or text[:500])
    key_findings = [_unwrap(f) for f in data.get("key_findings", [])]
    anomalies = [_unwrap(a) for a in data.get("anomalies_detected", [])]
    recommendations = [_unwrap(r) for r in data.get("recommendations", [])]
    return SummaryOutput(
        title=str(title),
        executive_summary=str(exec_summary),
        key_findings=[str(f) for f in key_findings],
        calculated_metrics=dict(_unwrap(data.get("calculated_metrics", {})) or {}),
        anomalies_detected=[str(a) for a in anomalies],
        recommendations=[str(r) for r in recommendations],
    )