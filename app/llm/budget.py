"""Digest budget controller.

The LLM context window is a hard limit. The digest (verified profile +
representative sample) is the largest flexible block in the prompt, so it must
fit the context minus the reserved output budget. This module:

  1. estimates the prompt in tokens (chars/4, a standard heuristic);
  2. when the budget is exceeded, deterministically compresses the digest
     through a fixed ladder - keeping the most informative parts (top-k
     statistics, dominant categories, strongest correlations) instead of
     blind-truncating text mid-structure;
  3. always ends with a digest small enough to leave room for a 200-400 word
     summary around the exact stats.

Compression is lossy but bounded, reproducible and never cuts inside a number
or a categorical value. The controller is pure: it never mutates the caller's
digest.
"""

from __future__ import annotations

import copy
import math
import os
import re

from app.profile.prompt import build_profile_prompt

# ---- token estimation ------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Heuristic token count (used by OpenAI/T5-style estimators: ~4 chars/token)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip() or str(default))
    except (TypeError, ValueError):
        return default


def input_budget_tokens() -> int:
    """Max prompt tokens: context minus output budget (env overridable).

    Env precedence: DIGEST_INPUT_TOKENS > (OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT)
    > default 2000.
    """
    override = os.getenv("DIGEST_INPUT_TOKENS")
    if override is not None and override.strip():
        try:
            return max(256, int(override.strip()))
        except ValueError:
            pass
    ctx = _int_env("OLLAMA_NUM_CTX", 2048)
    out = _int_env("OLLAMA_NUM_PREDICT", 500)
    return max(256, ctx - out)


# ---- deterministic compression ladder ---------------------------------------

_LADDER = [
    # level 0: no truncation
    {"important": None, "cat_cols": None, "cat_vals": None, "anom": None,
     "corr": None, "missing": None, "imbalance": None, "uniq": None,
     "preview": None, "sample_rows": None, "drop_uniq": False},
    # level 1
    {"important": 6, "cat_cols": 5, "cat_vals": 4, "anom": 5, "corr": 4,
     "missing": 4, "imbalance": 4, "uniq": 30, "preview": 10,
     "sample_rows": None, "drop_uniq": False},
    # level 2
    {"important": 4, "cat_cols": 3, "cat_vals": 3, "anom": 3, "corr": 2,
     "missing": 2, "imbalance": 2, "uniq": 15, "preview": 6,
     "sample_rows": None, "drop_uniq": False},
    # level 3
    {"important": 2, "cat_cols": 2, "cat_vals": 2, "anom": 2, "corr": 1,
     "missing": 1, "imbalance": 1, "uniq": 8, "preview": 3,
     "sample_rows": 2, "drop_uniq": False},
    # level 4
    {"important": 1, "cat_cols": 1, "cat_vals": 1, "anom": 1, "corr": 1,
     "missing": 1, "imbalance": 0, "uniq": 0, "preview": 1,
     "sample_rows": 1, "drop_uniq": True},
    # level 5: minimal structurally-valid digest (still has exact stats)
    {"important": 1, "cat_cols": 0, "cat_vals": 1, "anom": 0, "corr": 0,
     "missing": 0, "imbalance": 0, "uniq": 0, "preview": 0,
     "sample_rows": 0, "drop_uniq": True},
]


def _clip(items, limit):
    if limit is None:
        return items
    return items[:limit]


def _compress_digest(profile: dict, sample: dict | None, level: int):
    """Apply compression `level` to a deep copy of the digest. Pure."""
    p = copy.deepcopy(profile)
    s = copy.deepcopy(sample) if sample else None
    cfg = _LADDER[level]

    stats = p.get("statistics") or {}
    if "important_numeric_columns" in stats:
        stats["important_numeric_columns"] = _clip(
            stats["important_numeric_columns"], cfg["important"]
        )
    p["statistics"] = stats

    p["categorical_breakdown"] = _clip(
        p.get("categorical_breakdown") or [], cfg["cat_cols"]
    )
    for item in p["categorical_breakdown"]:
        item["top_values"] = _clip(item.get("top_values") or [], cfg["cat_vals"])

    p["class_imbalance"] = _clip(p.get("class_imbalance") or [], cfg["imbalance"])

    anomalies = p.get("anomalies") or {}
    if "top_columns" in anomalies:
        anomalies["top_columns"] = _clip(anomalies["top_columns"], cfg["anom"])
    p["anomalies"] = anomalies

    corr = p.get("correlations") or {}
    if "top_positive" in corr:
        corr["top_positive"] = _clip(corr["top_positive"], cfg["corr"])
    if "top_negative" in corr:
        corr["top_negative"] = _clip(corr["top_negative"], cfg["corr"])
    p["correlations"] = corr

    quality = p.get("quality") or {}
    if "missing_by_column_top" in quality:
        quality["missing_by_column_top"] = _clip(
            quality["missing_by_column_top"], cfg["missing"]
        )
    if cfg["drop_uniq"]:
        quality.pop("unique_value_counts", None)
    if cfg["uniq"] is not None:
        uvc = quality.get("unique_value_counts")
        if isinstance(uvc, dict):
            quality["unique_value_counts"] = dict(list(uvc.items())[: cfg["uniq"]])
    p["quality"] = quality

    if "column_categories" in p and cfg["preview"] is not None:
        for key, items in p["column_categories"].items():
            if isinstance(items, list):
                p["column_categories"][key] = _clip(items, max(cfg["preview"], 1))

    if s:
        buckets = s.get("buckets") or {}
        if cfg["sample_rows"] is not None and cfg["sample_rows"] > 0:
            for label, rows in buckets.items():
                buckets[label] = rows[: cfg["sample_rows"]]
        elif cfg["sample_rows"] == 0:
            buckets = {}
        s["buckets"] = buckets
        s["per_bucket"] = cfg["sample_rows"] if cfg["sample_rows"] not in (None, 0) else s.get("per_bucket", 0)

    return p, s


def build_budgeted_prompt(profile: dict, sample: dict | None = None):
    """Build the LLM prompt from a digest that fits the token budget.

    Returns ``(prompt, fitted_profile, fitted_sample)``. The fitted digest is
    the exact one the model sees - downstream verification (FactGuard) should
    use the same copies so that allowed numbers == numbers the model saw.
    """
    budget = input_budget_tokens()
    for level in range(len(_LADDER)):
        prof, smp = _compress_digest(profile, sample, level)
        prompt = build_profile_prompt(prof, sample=smp)
        if estimate_tokens(prompt) <= budget or level == len(_LADDER) - 1:
            return prompt, prof, smp
    prof, smp = _compress_digest(profile, sample, len(_LADDER) - 1)
    return build_profile_prompt(prof, sample=smp), prof, smp