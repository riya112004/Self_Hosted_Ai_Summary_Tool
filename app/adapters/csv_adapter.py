import csv
import io
from typing import Any


def _coerce(value: str) -> Any:
    """Coerce numeric-looking CSV fields to int/float, keep the rest as text."""
    s = (value or "").strip()
    if s == "":
        return value
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return value


def convert_csv(content: str) -> dict:
    """
    Convert CSV string input to Universal Data Representation.

    First row is treated as the header; empty columns are skipped; blank
    lines are ignored; numeric fields are coerced to int/float so the
    profiler recognises them as numeric.

    Returns {"source_type": "csv", "records": [...]}.
    """
    if not isinstance(content, str):
        raise ValueError("CSV input must be a string")

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or not [h for h in reader.fieldnames if h and h.strip()]:
        raise ValueError("CSV must include a header row")

    records = []
    for row in reader:
        record = {}
        for header, value in row.items():
            key = (header or "").strip()
            if not key:
                continue
            record[key] = _coerce(value)
        if record:
            records.append(record)

    return {
        "source_type": "csv",
        "records": records,
    }