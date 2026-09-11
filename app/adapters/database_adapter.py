def convert_db_query(rows: list) -> dict:
    """
    Convert raw database query rows (List[dict]) into the Universal Data
    Representation used by the summary pipeline.

    Rows are emitted as-is; the profiler downstream normalizes types.
    Returns {"source_type", "count", "records"}.
    """
    return {
        "source_type": "mongodb",
        "count": len(rows),
        "records": list(rows),
    }