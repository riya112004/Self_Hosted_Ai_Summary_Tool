import re
from typing import Any, List

# Date patterns: YYYY-MM-DD, DD/MM/YYYY, YYYY-MM-DD HH:MM:SS, etc.
_YYYY_MM_DD = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}")
_DD_MM_YYYY = re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}")
_MONTH_NAME_PATTERN = re.compile(r"\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)")


def _is_id_column(name: str) -> bool:
    lower = name.lower()
    return lower == "id" or lower.endswith("_id")


def _is_date_value(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_YYYY_MM_DD.match(value) or _DD_MM_YYYY.match(value) or _MONTH_NAME_PATTERN.search(value))


def _is_numeric(values: List[Any]) -> bool:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return False
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null)


def _is_categorical(values: List[Any], threshold: float = 0.7) -> bool:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return False
    unique = len({str(v) for v in non_null})
    return unique / len(non_null) <= threshold


def detect_column_type(column_name: str, values: List[Any]) -> str:
    """
    Deterministically detect the semantic type of a column.
    No LLM involved — pure Python.
    """
    # 1. Name-based: ID detection
    if _is_id_column(column_name):
        return "id"

    non_null = [v for v in values if v is not None]

    # 2. Value-based: date detection
    if non_null and all(_is_date_value(v) for v in non_null[:20]):
        return "date"

    # 3. Value-based: numeric detection
    if _is_numeric(non_null):
        return "numeric"

    # 4. Value-based: categorical detection
    if _is_categorical(non_null):
        return "categorical"

    return "string"