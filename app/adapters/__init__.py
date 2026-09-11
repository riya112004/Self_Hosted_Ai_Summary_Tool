def convert(data, source_type: str = "json") -> dict:
    """
    Route input to the correct adapter based on source_type.
    Only the matching adapter module is imported and executed.
    """
    if source_type == "json":
        from .json_adapter import convert_json

        return convert_json(data)
    elif source_type in ("csv", "xlsx", "xls"):
        if source_type == "csv":
            from .csv_adapter import convert_csv

            return convert_csv(data)
        from .excel_adapter import convert_excel

        return convert_excel(data, source_type=source_type)
    elif source_type == "database":
        from .database_adapter import convert_db_query

        return convert_db_query(data)
    elif source_type == "text":
        from .text_adapter import convert_text

        return convert_text(data)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")