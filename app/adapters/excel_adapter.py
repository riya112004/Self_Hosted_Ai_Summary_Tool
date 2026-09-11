from io import BytesIO
from typing import Any

from python_calamine import CalamineWorkbook


def _coerce(value: Any) -> Any:
    """Keep numbers native; integral floats become int; None becomes ''."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (bool, int, float)):
        return value
    return str(value)


def convert_excel(content: bytes, source_type: str = "excel") -> dict:
    """
    Convert binary Excel (.xlsx/.xls) input to Universal Data Representation.

    The first non-empty worksheet is used; the first row is treated as the
    header. Numeric cells stay native (int/float/bool) so the profiler
    recognises them as numeric without string coercion.

    Returns {"source_type": ..., "records": [...]}.
    """
    if not isinstance(content, bytes):
        raise ValueError("Excel input must be bytes")

    wb = CalamineWorkbook.from_filelike(BytesIO(content))
    sheet = wb.get_sheet_by_index(0)
    rows = [[_coerce(cell) for cell in row] for row in sheet.to_python()]

    header = rows[0] if rows else []
    if not header or not [h for h in header if str(h).strip()]:
        raise ValueError("Excel must include a header row")

    records = []
    for row in rows[1:]:
        record = {}
        for idx, name in enumerate(header):
            key = str(name or "").strip()
            if not key:
                continue
            value = row[idx] if idx < len(row) else ""
            record[key] = value
        if record:
            records.append(record)

    return {
        "source_type": source_type,
        "records": records,
    }