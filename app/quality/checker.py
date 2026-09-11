import re
from typing import Any, Dict, List

_DATE_PATTERN = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_DD_MM_YYYY = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}")


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


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _is_invalid_number(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return False
    if not isinstance(value, str) or value.strip() == "":
        return False
    stripped = value.strip()
    if _DATE_PATTERN.match(stripped) or _DD_MM_YYYY.match(stripped):
        return False
    has_digits = any(c.isdigit() for c in stripped)
    has_alpha = any(c.isalpha() for c in stripped)
    return has_digits and has_alpha


def _is_invalid_date(value: Any) -> bool:
    if not isinstance(value, str) or value.strip() == "":
        return False
    stripped = value.strip()
    has_separator = "-" in stripped or "/" in stripped
    if has_separator:
        return not bool(_DATE_PATTERN.match(stripped) or _DD_MM_YYYY.match(stripped))
    return False


def _count_duplicates(records: List[Dict[str, Any]]) -> int:
    seen: set = set()
    duplicates = 0
    for record in records:
        key = _record_key(record)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def check_quality(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run data quality checks on records.
    Returns quality summary with score 0-100.
    """
    total_rows = len(records)

    if total_rows == 0:
        return {
            "rows": 0,
            "duplicate_rows": 0,
            "null_values": 0,
            "empty_fields": 0,
            "invalid_numbers": 0,
            "invalid_dates": 0,
            "quality_score": 100.0,
        }

    columns = list(records[0].keys())
    total_cells = total_rows * len(columns)

    duplicate_rows = _count_duplicates(records)

    null_values = 0
    empty_fields = 0
    invalid_numbers = 0
    invalid_dates = 0

    for record in records:
        for col, value in record.items():
            if value is None:
                null_values += 1
            elif isinstance(value, str) and value.strip() == "":
                empty_fields += 1
            elif _is_invalid_number(value):
                invalid_numbers += 1
            elif _is_invalid_date(value):
                invalid_dates += 1

    problem_cells = null_values + empty_fields + invalid_numbers + invalid_dates

    if total_cells > 0:
        quality_score = round((1 - problem_cells / total_cells) * 100, 2)
    else:
        quality_score = 100.0

    return {
        "rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "null_values": null_values,
        "empty_fields": empty_fields,
        "invalid_numbers": invalid_numbers,
        "invalid_dates": invalid_dates,
        "quality_score": quality_score,
    }