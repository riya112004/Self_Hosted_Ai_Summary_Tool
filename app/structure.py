"""Universal nested-path structure detection and token scanning.

Turns arbitrary nested JSON into flat, path-labelled records so the generic
profiler can measure them.  No domain-specific rules here - only shape:

    - scalar                  -> kept as-is
    - dict                    -> recursed as "parent.child"
    - small array of objects  -> unrolled as "parent[0].child"
    - big array of objects    -> marker string
    - small scalar list       -> joined comma-string (e.g. "Italian,Pizza")
    - large/deep list         -> compact marker (never raw JSON)
    - grid (list of lists)    -> "[grid NxM]" marker
"""

from collections import Counter

_MAX_UNROLL = 4
_MAX_SCALAR_JOIN = 12


def flatten_record(item: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in item.items():
        name = f"{prefix}.{key}" if prefix else key
        _flatten_value(value, name, out)
    return out


def _flatten_value(value, name: str, out: dict) -> None:
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            _flatten_value(sub_value, f"{name}.{sub_key}", out)
    elif isinstance(value, list):
        if not value:
            out[name] = None
        elif all(isinstance(x, dict) for x in value):
            if len(value) <= _MAX_UNROLL:
                for i, sub in enumerate(value):
                    for sub_key, sub_value in sub.items():
                        _flatten_value(sub_value, f"{name}[{i}].{sub_key}", out)
            else:
                out[name] = f"[nested array of {len(value)} objects]"
        elif all(isinstance(x, (str, int, float, bool)) for x in value):
            if len(value) <= _MAX_SCALAR_JOIN:
                out[name] = ", ".join(str(x) for x in value)
            else:
                out[name] = f"[list of {len(value)} items]"
        elif all(isinstance(x, list) for x in value):
            inner_len = len(value[0]) if value[0] else 0
            out[name] = f"[grid {len(value)}x{inner_len}]"
        else:
            out[name] = f"[nested list of {len(value)} items]"
    else:
        out[name] = value


def scan_tokens(value, counter: Counter, depth: int = 0) -> None:
    """Recursively extract string tokens from nested list structures.

    Counts first elements of list-of-lists (operation records like
    ['BUY_PRODUCT','WHEAT',13] → BUY_PRODUCT) and scalar strings in
    flat string lists (like ['PASS'] → PASS).

    Skips grids (list of lists where first elements are None/numeric).
    Caps recursion depth to avoid runaway on deeply nested structures.
    """
    if depth > 6:
        return
    if isinstance(value, dict):
        for v in value.values():
            scan_tokens(v, counter, depth + 1)
    elif isinstance(value, list):
        if not value:
            return
        if all(isinstance(x, list) for x in value):
            first_elems = [x[0] for x in value if x]
            if all(isinstance(f, str) and 0 < len(f) <= 40 for f in first_elems if f is not None):
                for f in first_elems:
                    if isinstance(f, str):
                        counter[f] += 1
            elif any(isinstance(x, dict) for x in value[0:1]):
                scan_tokens(value[0], counter, depth + 1)
        elif all(isinstance(x, str) for x in value):
            if all(0 < len(x) <= 40 for x in value):
                for x in value:
                    counter[x] += 1
        elif any(isinstance(x, (dict, list)) for x in value[:3]):
            for x in value[:3]:
                scan_tokens(x, counter, depth + 1)
