from typing import Any, Dict, List

from .type_detector import detect_column_type


def detect_schema(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Given a list of dict records, detect the schema (column name + semantic type).
    Returns: [{"name": "customer_id", "dtype": "id"}, ...]
    """
    if not records:
        return []

    all_columns: set[str] = set()
    for record in records:
        all_columns.update(record.keys())

    schema = []
    for col in sorted(all_columns):
        values = [record.get(col) for record in records]
        col_type = detect_column_type(col, values)
        schema.append({"name": col, "dtype": col_type})

    return schema