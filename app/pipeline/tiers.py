def detect_tier(row_count: int) -> str:
    """
    Route datasets by size:
      <= 1,000        -> "small"   (100 / 1,000 rows)    Profile -> Sample -> LLM
      <= 100,000      -> "medium"  (10,000 / 100,000)    Profile -> Aggregate -> Sample -> LLM
      >  100,000      -> "large"   (100,000+ / 1M)       DB/Polars agg -> Grouped summaries -> LLM
    """
    if row_count <= 1_000:
        return "small"
    if row_count <= 100_000:
        return "medium"
    return "large"