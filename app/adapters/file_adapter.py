"""Convert uploaded file content into records (universal data representation).

Supports CSV, TSV, semicolon-delimited text, JSON arrays, JSON lines and a
single JSON object. Delimiters are detected from the header line so the same
loader works for comma, tab and semicolon files without user configuration.
"""

import csv
import io
import json

from .csv_adapter import _coerce


def _detect_delimiter(content: str) -> str:
    first_line = content.splitlines()[0] if content.splitlines() else ""
    for delim in ("\t", ";", ","):
        if delim in first_line:
            return delim
    return ","


def _json_records(content: str) -> list[dict]:
    text = content.strip()
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON file root must be an array of objects or a single object")
        return [r for r in data if isinstance(r, dict)]

    if text.startswith("{") and "\n" not in text.strip():
        data = json.loads(text)
        if isinstance(data, dict):
            # a single record, or {key: [records...]} style wrapper -> collect dicts
            if all(isinstance(v, list) for v in data.values()):
                rows = [r for values in data.values() for r in values if isinstance(r, dict)]
                if rows:
                    return rows
            return [data]
        raise ValueError("JSON file root must be an array of objects or a single object")

    # JSON Lines / NDJSON: one object per line
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            records.append(obj)
        elif isinstance(obj, list):
            records.extend(r for r in obj if isinstance(r, dict))
    if not records:
        raise ValueError("JSON file contained no object records")
    return records


def _csv_records(content: str, delimiter: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
    if not reader.fieldnames or not [h for h in reader.fieldnames if h and h.strip()]:
        raise ValueError("CSV file must include a header row")

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
    return records


def load_records_from_text(filename: str, content: str) -> dict:
    """
    Parse a file's text content into {"source_type": ..., "records": [...]}.

    The source type is guessed from the file extension, but any kind of
    delimited text is treated as a table when no recognized extension matches.
    """
    if not content or not content.strip():
        raise ValueError("Uploaded file is empty")

    name = (filename or "").lower()
    if name.endswith(".json") or name.endswith(".jsonl"):
        return {"source_type": "json", "records": _json_records(content)}

    delimiter = _detect_delimiter(content)
    source_type = {"\t": "tsv", ";": "csv"}.get(delimiter, "csv")
    records = _csv_records(content, delimiter)
    if not records:
        raise ValueError("File contains no data rows")
    return {"source_type": source_type, "records": records}