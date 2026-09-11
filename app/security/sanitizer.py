import re

from .pii import detect_pii, detect_value_type


def mask_email(value: str) -> str:
    if "@" not in value:
        return "[REDACTED]"
    local, _, domain = value.partition("@")
    if not local:
        return "[REDACTED]@" + domain
    return local[:2] + "***@" + domain


def mask_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        return "***-***-" + digits[-4:]
    return "[REDACTED]"


def mask_ssn(value: str) -> str:
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        return "***-**-" + digits[-4:]
    return "[REDACTED]"


def mask_secret(value: str) -> str:
    if len(value) > 4:
        return value[:2] + "***" + value[-2:]
    return "[REDACTED]"


def mask_value(value, pii_type: str) -> str:
    """Mask a single value according to its PII type."""
    if not isinstance(value, str):
        return "[REDACTED]"
    if pii_type == "email":
        return mask_email(value)
    if pii_type == "phone":
        return mask_phone(value)
    if pii_type == "ssn":
        return mask_ssn(value)
    if pii_type == "password":
        return "[REDACTED]"
    return mask_secret(value)  # token, api_key


def sanitize(
    records: list[dict],
    policy: str = "mask",
    remove_columns: list[str] | None = None,
    columns: list[str] | None = None,
) -> dict:
    """
    Sanitize records before they reach profiler/LLM.

    policy="mask" -> PII values are masked in place.
    policy="drop" -> PII columns are removed entirely.

    remove_columns -> always remove these columns, PII or not.
    columns        -> restrict PII handling to only these columns
                      (all other columns pass through untouched).

    Returns {"sanitized": records, "removed_columns": [...], "findings": [...]}.
    """
    if not records:
        return {"sanitized": [], "removed_columns": [], "findings": []}

    remove_cols = set(remove_columns or [])
    findings = detect_pii(records)

    if columns is not None:
        allowed = set(columns)
        findings = [f for f in findings if f.column in allowed]

    column_hint_types = {}
    for f in findings:
        if f.detected_by == "column_name":
            column_hint_types[f.column] = f.pii_type

    # columns detected only by value pattern
    value_type_columns = {}
    for f in findings:
        if f.detected_by == "value":
            types = value_type_columns.setdefault(f.column, [])
            if f.pii_type not in types:
                types.append(f.pii_type)

    sanitized = []
    for record in records:
        out = {}
        for col, value in record.items():
            if col in remove_cols:
                continue

            if col in column_hint_types:
                if policy == "drop":
                    continue
                out[col] = mask_value(value, column_hint_types[col])
                continue

            if col in value_type_columns and isinstance(value, str):
                t = detect_value_type(value)
                if t in value_type_columns[col]:
                    out[col] = mask_value(value, t)
                    continue

            out[col] = value
        sanitized.append(out)

    return {
        "sanitized": sanitized,
        "removed_columns": list(remove_cols),
        "findings": [f.model_dump() for f in findings],
    }