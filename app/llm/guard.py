"""FactGuard - numeric verification layer for LLM summaries.

Every number that reaches the user must trace back to a fact Python computed
from the data (the verified digest). The LLM is free to write prose, but each
number it publishes is a CLAIM. FactGuard:

  1. builds the ALLOWED-FACT set by flattening every numeric leaf of the
     verified profile digest and the representative sample - the only
     numbers the model could ever have seen;
  2. extracts every number from the SummaryOutput;
  3. keeps a sentence / list item / metric only when ALL of its numbers match
     an allowed fact (within rounding tolerance). Any container carrying an
     unverifiable number is dropped;
  4. when too many guards fail, the deterministic digest summary becomes the
     anchor and the LLM output is discarded ("LLM as garnish only").

Non-numeric prose is never evaluated: it cannot be proven correct, but it also
cannot be proven wrong against the digest.
"""

from __future__ import annotations

import re

from app.schemas.summary import SummaryOutput

# ---- number extraction -----------------------------------------------------
# 1,234,567.89 / -0.5 / 60% / 500K / 2M / 3B / 42 million / 1.2 billion
_NUM_RE = re.compile(
    r"(?<![\w.])(?P<num>-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)\s*"
    r"(?P<suffix>(?:%|[KMBk]|\b(?:million|billion|thousand)\b))?",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SUFFIX_MULT = {"K": 1e3, "M": 1e6, "B": 1e9,
                "THOUSAND": 1e3, "MILLION": 1e6, "BILLION": 1e9}


def _fact_variants(f: float) -> set[float]:
    """All the ways a verified fact could legitimately re-appear in prose.

    Handles rounding (median 492.034 -> "492"), thousand separators,
    K/M/B forms (500000 -> "500K") and percent forms of shares
    (0.84 -> "84%").
    """
    var: set[float] = set()
    if f != f:  # NaN
        return var
    f = float(f)
    var.add(f)
    for d in range(5):  # rounding tolerance: 0..4 decimal places
        var.add(round(f, d))
    if f.is_integer():
        i = int(f)
        var.add(float(i))
        var.add(i / 1e3)
        var.add(i / 1e6)
        var.add(i / 1e9)
    if 0.0 < abs(f) < 1.0:  # a share may be written as a percent
        p = f * 100.0
        var.add(round(p, 4))
        var.add(round(p, 2))
        var.add(round(p, 0))
    return var


def _numeric_leaves(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numeric_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numeric_leaves(v)
    elif isinstance(obj, (int, float)):
        yield float(obj)


def _allowed_facts(profile: dict | None, sample: dict | None) -> set[float]:
    """Every numeric variant the model could legitimately reproduce."""
    facts: set[float] = set()
    for obj in (profile, sample):
        if not obj:
            continue
        for leaf in _numeric_leaves(obj):
            facts |= _fact_variants(leaf)
    return facts


def _number_claims(text: str) -> list[tuple[float, str]]:
    """Extract (numeric value, suffix) claims from a text fragment."""
    claims = []
    for m in _NUM_RE.finditer(text):
        raw = m.group("num")
        suffix = (m.group("suffix") or "").upper().rstrip("%")
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix in _SUFFIX_MULT:
            v *= _SUFFIX_MULT[suffix]
        claims.append((v, suffix))
    return claims


def _verified(claim_value: float, suffix: str, allowed: set[float]) -> bool:
    if suffix in _SUFFIX_MULT:
        # "1.23M" is readable rounding, not a fresh fact - allow within 2%.
        return any(abs(claim_value - f) <= 0.02 * abs(f) for f in allowed)
    # plain / percent claims must match an allowed variant almost exactly.
    tol = 1e-3
    return any(abs(claim_value - f) <= tol for f in allowed)


class _Tally:
    __slots__ = ("found", "verified")

    def __init__(self):
        self.found = 0
        self.verified = 0


def __verify(text: str, allowed: set[float], tally: _Tally) -> bool:
    """Verify every number in `text`. Returns True when all match."""
    ok = True
    count = 0
    for value, suffix in _number_claims(text):
        count += 1
        tally.found += 1
        if _verified(value, suffix, allowed):
            tally.verified += 1
        else:
            ok = False
    if count:
        return ok
    return True


def _guard_sentences(text: str, allowed: set[float], tally: _Tally) -> tuple[str, list[str]]:
    """Drop every sentence that carries an unverifiable number."""
    kept, dropped = [], []
    for sentence in _SENTENCE_SPLIT.split(text):
        if __verify(sentence, allowed, tally):
            kept.append(sentence)
        else:
            dropped.append(sentence)
    return " ".join(kept), dropped


def _guard_list(items, allowed: set[float], tally: _Tally, label: str, dropped) -> list:
    kept = []
    for item in items:
        if not isinstance(item, str):
            kept.append(item)
            continue
        if __verify(item, allowed, tally):
            kept.append(item)
        else:
            dropped.append(f"{label}: {item[:80]}")
    return kept


def _fallback_anchor(profile: dict | None) -> SummaryOutput:
    if profile and "report_digest" in profile:
        from app.summary.structured_report import _deterministic_summary

        return _deterministic_summary(profile["report_digest"])
    from app.summary.deterministic import build_profile_summary

    return build_profile_summary(profile or {})


def apply_fact_guard(
    summary: SummaryOutput,
    profile: dict | None,
    sample: dict | None = None,
    min_verified_ratio: float = 0.5,
) -> SummaryOutput:
    """Verify every number in the LLM summary against the verified digest.

    Returns the guarded summary (dropping containers that carry unverifiable
    numeric claims), or - when too many claims fail - the deterministic
    digest summary as anchor. Always carries a ``fact_guard`` report.
    """
    allowed = _allowed_facts(profile, sample)
    if not allowed:
        return summary.model_copy(
            update={
                "fact_guard": {
                    "status": "skipped",
                    "note": "Digest had no numeric facts to verify against.",
                }
            }
        )

    tally = _Tally()
    dropped: list[str] = []

    title = summary.title
    if __verify(title, allowed, tally):
        guarded_title = title
    else:
        dropped.append(f"title: {title[:80]}")
        guarded_title = "General summary"

    exec_summary, dropped_sentences = _guard_sentences(
        summary.executive_summary, allowed, tally
    )
    dropped.extend(f"executive_summary: {s[:80]}" for s in dropped_sentences)

    key_findings = _guard_list(summary.key_findings, allowed, tally, "key_findings", dropped)
    anomalies = _guard_list(summary.anomalies_detected, allowed, tally, "anomalies_detected", dropped)
    recommendations = _guard_list(summary.recommendations, allowed, tally, "recommendations", dropped)

    metrics = {}
    for key, value in (summary.calculated_metrics or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if __verify(str(value), allowed, tally):
                metrics[key] = value
            else:
                dropped.append(f"calculated_metrics.{key}: {value}")
        else:
            metrics[key] = value

    if tally.found == 0:
        return summary.model_copy(
            update={
                "fact_guard": {
                    "status": "pass",
                    "note": "No numeric claims in the summary; nothing to verify.",
                }
            }
        )

    verified_ratio = tally.verified / tally.found
    if (
        verified_ratio < min_verified_ratio
        or (dropped and not exec_summary.strip())
    ):
        # Too many guards failed -> deterministic digest becomes the anchor.
        report = {
            "status": "fallback",
            "claims_found": tally.found,
            "claims_verified": tally.verified,
            "claims_dropped": tally.found - tally.verified,
            "dropped": dropped[:8],
            "note": "Too many numeric claims could not be verified against the digest; "
            "using the deterministic digest summary as the anchor.",
        }
        return _fallback_anchor(profile).model_copy(update={"fact_guard": report})

    report = {
        "status": "guarded",
        "claims_found": tally.found,
        "claims_verified": tally.verified,
        "claims_dropped": tally.found - tally.verified,
        "dropped": dropped[:8],
        "note": "Numeric claims verified against the digest; unverifiable claims dropped.",
    }
    return summary.model_copy(
        update={
            "title": guarded_title,
            "executive_summary": exec_summary,
            "key_findings": key_findings,
            "anomalies_detected": anomalies,
            "recommendations": recommendations,
            "calculated_metrics": metrics,
            "fact_guard": report,
        }
    )