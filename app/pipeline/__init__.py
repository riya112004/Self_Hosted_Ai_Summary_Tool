from .tiers import detect_tier
from . import hierarchy


def process(
    records: list[dict],
    summarize=None,
    n: int = 20,
    max_groups: int = 10,
) -> dict:
    """
    Route the dataset through the right strategy based on row count.

    `summarize` is an optional callable that turns one payload dict into a
    short text summary (e.g. lambda payload: llm_service.generate_summary(payload)).
    When omitted, the deterministic part runs and `summary` stays None — LLM
    wiring stays fully pluggable.

    Small  (<=1,000)   : profile -> sample -> LLM
    Medium (<=100,000) : profile -> aggregate (analytics) -> sample -> LLM
    Large  (>100,000)  : Polars aggregation -> grouped summaries (map) -> LLM synthesis (reduce)
    """
    row_count = len(records)
    tier = detect_tier(row_count)

    if not records:
        return {
            "tier": "empty",
            "row_count": 0,
            "context": None,
            "summary": None,
        }

    if tier == "small":
        from app.sampling.representative import build_context

        context = build_context(records, n=n)
        return {
            "tier": tier,
            "row_count": row_count,
            "context": context,
            "summary": summarize({"context": context}) if summarize else None,
        }

    if tier == "medium":
        from app.sampling.representative import build_context
        from app.analytics import compute_analytics

        context = build_context(records, n=n)
        analytics = compute_analytics(records)
        payload = {"context": context, "analytics": analytics}
        return {
            "tier": tier,
            "row_count": row_count,
            "context": context,
            "analytics": analytics,
            "summary": summarize(payload) if summarize else None,
        }

    # ---- large: hierarchical map-reduce ----
    group_col = hierarchy.choose_group_col(records, max_groups=max_groups)
    groups: list[dict] = []

    if group_col is not None:
        agg_rows = hierarchy.aggregate_groups(records, group_col)
        members = hierarchy.group_samples(records, group_col)

        for agg_row in agg_rows[:max_groups]:
            key = str(agg_row[group_col])
            payload = hierarchy.build_group_payload(
                group_col, key, agg_row, members.get(key, []), n=n
            )
            entry = {
                "group": payload["group"],
                "row_count": payload["row_count"],
                "aggregation": payload["aggregation"],
                "sample": payload["sample"],
            }
            if summarize:
                entry["summary"] = summarize(payload)
            groups.append(entry)

    # reduce step: synthesize one global summary from the grouped summaries
    from app.sampling.representative import build_context

    global_stats = build_context(records, n=2)
    reduce_payload = {
        "group_col": group_col,
        "group_count": len(groups),
        "group_summaries": groups,
        "global_statistics": global_stats,
    }
    return {
        "tier": tier,
        "row_count": row_count,
        "group_col": group_col,
        "groups": groups,
        "summary": summarize(reduce_payload) if summarize else None,
    }