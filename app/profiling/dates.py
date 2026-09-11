import polars as pl


def profile_dates(series: pl.Series) -> dict:
    """Deterministic date profiling — range, monthly, daily distribution."""
    if series.dtype == pl.Utf8:
        series = series.str.to_date(strict=False)

    if series.dtype == pl.Datetime:
        series = series.cast(pl.Date, strict=False)

    if series.null_count() == series.len():
        return {"error": "all values are null"}

    earliest = str(series.min())
    latest = str(series.max())

    valid = series.drop_nulls()

    month_df = (
        valid.dt.month()
        .alias("_month")
        .to_frame("_month")
        .group_by("_month")
        .len()
        .sort("_month")
    )
    records_per_month = {
        str(row["_month"]): row["len"] for row in month_df.iter_rows(named=True)
    }

    day_df = (
        valid.dt.weekday()
        .alias("_weekday")
        .to_frame("_weekday")
        .group_by("_weekday")
        .len()
        .sort("_weekday")
    )
    records_per_day = {
        str(row["_weekday"]): row["len"] for row in day_df.iter_rows(named=True)
    }

    return {
        "earliest": earliest,
        "latest": latest,
        "records_per_month": records_per_month,
        "records_per_day": records_per_day,
        "null_count": int(series.null_count()),
    }