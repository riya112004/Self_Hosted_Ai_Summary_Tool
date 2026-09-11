import re

from pydantic import BaseModel

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<![\d])(?:\+?\d{1,3}[-\s.]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
API_KEY_RE = re.compile(
    r"\b(?:sk-|pk_live_|ghp_|glpat-|xox[baprs]-|AIza)[A-Za-z0-9_-]{10,}\b"
)

RISK_LEVELS = {
    "email": "medium",
    "phone": "medium",
    "ssn": "high",
    "password": "high",
    "token": "high",
    "api_key": "high",
}

COLUMN_HINTS = {
    "email": {"email"},
    "phone": {"phone", "mobile", "contact", "tel", "cell"},
    "ssn": {"ssn", "social"},
    "password": {"password", "passwd", "pass", "pwd"},
    "token": {"token", "auth", "session", "refresh", "jwt"},
    "api_key": {"api_key", "apikey", "secret", "private_key", "client_secret"},
}


class PIIFinding(BaseModel):
    column: str
    pii_type: str
    risk: str
    detected_by: str  # "column_name" | "value"
    count: int


def risk_of(pii_type: str) -> str:
    return RISK_LEVELS.get(pii_type, "low")


def detect_value_type(value: str) -> str | None:
    """Detect the PII type of a single string value (or None)."""
    if not value or not isinstance(value, str):
        return None
    if EMAIL_RE.search(value):
        return "email"
    if SSN_RE.search(value):
        return "ssn"
    if JWT_RE.search(value):
        return "token"
    if API_KEY_RE.search(value):
        return "api_key"
    if PHONE_RE.search(value):
        return "phone"
    return None


def column_hint_type(column: str) -> str | None:
    """Match a column name against known PII hints."""
    name = column.lower()
    for pii_type, keys in COLUMN_HINTS.items():
        if any(key in name for key in keys):
            return pii_type
    return None


def detect_pii(records: list[dict]) -> list[PIIFinding]:
    """
    Scan records for PII.
    Two detection layers:
      1. column_name  - the column name itself reveals the PII ('email', 'api_key', ...)
      2. value        - a string cell matches a PII pattern (email, SSN, phone, token, api key)
    """
    if not records:
        return []

    columns = {col for record in records for col in record.keys()}
    findings: list[PIIFinding] = []

    for column in sorted(columns):
        values = [record.get(column) for record in records if isinstance(record, dict)]

        hint = column_hint_type(column)
        if hint is not None:
            findings.append(
                PIIFinding(
                    column=column,
                    pii_type=hint,
                    risk=risk_of(hint),
                    detected_by="column_name",
                    count=len(values),
                )
            )

        # value-based detection (any column)
        matched: dict[str, int] = {}
        for v in values:
            t = detect_value_type(v) if isinstance(v, str) else None
            if t is not None and t != hint:
                matched[t] = matched.get(t, 0) + 1
        for pii_type, count in matched.items():
            findings.append(
                PIIFinding(
                    column=column,
                    pii_type=pii_type,
                    risk=risk_of(pii_type),
                    detected_by="value",
                    count=count,
                )
            )

    return findings