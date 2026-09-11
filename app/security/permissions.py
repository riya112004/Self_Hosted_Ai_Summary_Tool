from .pii import detect_pii
from .sanitizer import sanitize

ROLE_PERMISSIONS = {
    "admin": {"read_sanitized", "read_raw"},
    "analyst": {"read_sanitized"},
    "viewer": {"read_sanitized"},
}

# These types are stripped even from privileged (raw) access.
HIGH_RISK_TYPES = {"password", "token", "api_key"}


class PermissionDenied(PermissionError):
    pass


def can(role: str, action: str) -> bool:
    return action in ROLE_PERMISSIONS.get(role, set())


def require(role: str, action: str) -> None:
    if not can(role, action):
        raise PermissionDenied(
            f"Role {role!r} is not allowed to perform {action!r}."
        )


def access(records: list[dict], role: str) -> dict:
    """
    Role-aware data access:
      - everyone gets read_sanitized
      - analyst/viewer -> all PII masked
      - admin (read_raw) -> raw records, but high-risk PII columns
        (passwords, tokens, api keys) are always dropped
    """
    require(role, "read_sanitized")

    findings = detect_pii(records)
    high_risk_columns = [
        f.column
        for f in findings
        if f.detected_by == "column_name" and f.risk == "high"
    ]

    if can(role, "read_raw"):
        result = sanitize(
            records,
            policy="drop",
            columns=high_risk_columns,
            remove_columns=high_risk_columns,
        )
        return {
            "role": role,
            "mode": "raw",
            "records": result["sanitized"],
            "removed_columns": result["removed_columns"],
            "findings": result["findings"],
        }

    result = sanitize(records, policy="mask")
    return {
        "role": role,
        "mode": "sanitized",
        "records": result["sanitized"],
        "removed_columns": result["removed_columns"],
        "findings": result["findings"],
    }