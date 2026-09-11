"""Evidence-based domain detection from column names.

Heuristics only - never the LLM. If the signals are weak the system says
"Unknown" instead of guessing, because an invented domain label is worse
than admitting uncertainty.
"""

import re

DOMAIN_SIGNALS = {
    "EEG / clinical": [
        "eeg", "coh", "coherence", "alpha", "beta", "gamma", "delta",
        "theta", "spectral", "power", "channel", "electrode", "disorder",
        "signal", "brainwave",
    ],
    "molecular / chemistry": [
        "atom", "molecule", "dipole", "polariz", "binding", "energy",
        "ccsd", "charge", "geometry", "hartree", "weight",
    ],
    "medical / healthcare": [
        "patient", "diagnosis", "icd", "bmi", "glucose", "heart",
        "pressure", "pulse", "medication", "symptom", "blood", "cholesterol",
    ],
    "finance / accounting": [
        "revenue", "cost", "profit", "margin", "amount", "price",
        "transaction", "invoice", "income", "expense",
    ],
    "sales / marketing": [
        "customer", "campaign", "lead", "conversion", "channel", "funnel",
        "audience", "pipeline", "ops",
    ],
    "web / product analytics": [
        "session", "click", "pageview", "bounce", "impression", "visitor",
        "utm", "traffic", "active", "retention",
    ],
    "telecom / network": [
        "imsi", "msisdn", "cell", "throughput", "latency", "signal",
        "handover", "bandwidth", "packet", "subscriber",
    ],
    "iot / devices": [
        "device", "sensor", "temperature", "vibration", "voltage", "current",
        "equipment", "fault", "reading", "humidity", "payload",
    ],
    "human resources": [
        "employee", "salary", "department", "tenure", "attrition", "hire",
        "manager", "onboarding", "absenteeism",
    ],
    "education": [
        "exam", "score", "grade", "student", "enrollment", "curriculum",
        "semester", "gpa", "attendance",
    ],
}

_WORD_LIKE = re.compile(r"[a-z0-9]+")


def detect_domain(columns: list[str], max_evidence: int = 8) -> dict:
    """Return {"label", "confidence", "evidence"} for a list of column names."""
    if not columns:
        return {"label": "Unknown", "confidence": 0.0, "evidence": []}

    scores = {dom: 0 for dom in DOMAIN_SIGNALS}
    evidence: dict[str, list[str]] = {dom: [] for dom in DOMAIN_SIGNALS}

    for col in columns:
        # normalize separators to spaces so token prefixes/suffixes match
        words = {w for w in _WORD_LIKE.findall(col.lower())}
        if not words:
            continue
        for dom, toks in DOMAIN_SIGNALS.items():
            for tok in toks:
                if tok in words:
                    scores[dom] += 1
                    evidence[dom].append(f"{col} (→ {tok})")

    best = max(scores, key=scores.get)
    score = scores[best]

    if score < 2:
        return {
            "label": "Unknown",
            "confidence": round(min(0.3, score / 4), 2),
            "evidence": evidence[best][:max_evidence],
        }

    confidence = min(0.97, 0.5 + 0.08 * score)
    return {
        "label": best,
        "confidence": round(confidence, 2),
        "evidence": evidence[best][:max_evidence],
    }