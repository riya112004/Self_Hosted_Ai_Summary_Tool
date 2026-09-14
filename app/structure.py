"""Universal nested-path structure detection.

Turns arbitrary nested JSON into flat, path-labelled records so the generic
profiler can measure them.  No domain-specific rules here - only shape:
    - scalar          -> kept as-is
    - dict            -> recursed as "parent.child"
    - small array of object -> unrolled as "parent[0].child"
    - big/deep array of object -> kept as "parent[]" (marked nested)
    - list of scalars -> kept as-is (List column)

Examples:
    restaurant  ->  restaurant.name, restaurant.cuisines,
                    restaurant.user_rating.aggregate_rating
"""

_MAX_UNROLL = 4


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
        if value and all(isinstance(x, dict) for x in value):
            if len(value) <= _MAX_UNROLL:
                for i, sub in enumerate(value):
                    for sub_key, sub_value in sub.items():
                        _flatten_value(sub_value, f"{name}[{i}].{sub_key}", out)
            else:
                out[name] = f"[nested array of {len(value)} objects]"
        else:
            out[name] = value
    else:
        out[name] = value