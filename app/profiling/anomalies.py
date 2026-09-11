from typing import Any, Dict, List


def _freeze_value(value: Any):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(val)) for key, val in value.items()))
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    return value


def _record_key(record: Dict[str, Any]):
    return tuple(sorted((key, _freeze_value(value)) for key, value in record.items()))


def detect_duplicates(records: List[Dict[str, Any]]) -> dict:
    """Detect duplicate records — deterministic, no LLM."""
    total = len(records)
    unique = len({_record_key(record) for record in records})
    return {
        "total_records": total,
        "unique_records": unique,
        "duplicate_records": total - unique,
    }