from typing import Any


def flatten(nested: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dictionary using dotted notation."""
    items = {}
    for k, v in nested.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def convert_json(data: Any) -> dict:
    """
    Convert JSON input (single dict or list of dicts) to Universal Data Representation.
    Nested dicts are flattened using dotted notation.
    """
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("Input must be a JSON object or array")

    flattened_records = [flatten(record) for record in data if isinstance(record, dict)]

    return {
        "source_type": "json",
        "records": flattened_records,
    }