from concurrent.futures import ThreadPoolExecutor


def profile_dataset(records: list[dict]) -> dict:
    """
    Deterministic profiling using Polars.
    Only imports and runs modules needed for the detected column types.
    """
    from .anomalies import detect_duplicates

    if not records:
        return {
            "numeric": {},
            "categorical": {},
            "dates": {},
            "duplicates": detect_duplicates([]),
        }

    from .schema_detector import detect_schema

    schema = detect_schema(records)

    # Detect which types exist before importing anything
    has_numeric = any(c["dtype"] == "numeric" for c in schema)
    has_categorical = any(c["dtype"] == "categorical" for c in schema)
    has_dates = any(c["dtype"] == "date" for c in schema)

    numeric_stats = {}
    categorical_stats = {}
    date_stats = {}

    # Only create DataFrame if at least one profiler module is needed
    if has_numeric or has_categorical or has_dates:
        import polars as pl

        df = pl.DataFrame(records)

        def _profile_numeric_column(col_name: str):
            from .numeric import profile_numeric
            return col_name, profile_numeric(df[col_name])

        def _profile_categorical_column(col_name: str):
            from .categorical import profile_categorical
            return col_name, profile_categorical(df[col_name])

        def _profile_date_column(col_name: str):
            from .dates import profile_dates
            return col_name, profile_dates(df[col_name])

        if has_numeric:
            numeric_columns = [c["name"] for c in schema if c["dtype"] == "numeric"]
            with ThreadPoolExecutor(max_workers=min(8, len(numeric_columns) or 1)) as executor:
                for name, stats in executor.map(_profile_numeric_column, numeric_columns):
                    numeric_stats[name] = stats

        if has_categorical:
            categorical_columns = [c["name"] for c in schema if c["dtype"] == "categorical"]
            with ThreadPoolExecutor(max_workers=min(8, len(categorical_columns) or 1)) as executor:
                for name, stats in executor.map(_profile_categorical_column, categorical_columns):
                    categorical_stats[name] = stats

        if has_dates:
            date_columns = [c["name"] for c in schema if c["dtype"] == "date"]
            with ThreadPoolExecutor(max_workers=min(8, len(date_columns) or 1)) as executor:
                for name, stats in executor.map(_profile_date_column, date_columns):
                    date_stats[name] = stats

    return {
        "numeric": numeric_stats,
        "categorical": categorical_stats,
        "dates": date_stats,
        "duplicates": detect_duplicates(records),
    }