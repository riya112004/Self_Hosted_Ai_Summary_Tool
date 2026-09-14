"""Semantic, deterministic understanding of what a dataset is about.

Pure Python - no LLM. Column names + the verified domain detector are the only
inputs, so nothing is ever "invented": if the signal is weak the output says
``general / tabular`` instead of guessing. This powers the general output
("mere data ko samjho aur batao isme kya hai") and enriches every profile with
a human-readable one-liner about its contents.
"""

from __future__ import annotations

import re

from .domain import detect_domain

# ---- column role detection --------------------------------------------------
# First matching regex wins. Order matters: specific patterns first.
_ROLE_RULES: list[tuple[str, str, str]] = [
    # regex, role, info-category phrase
    (r"(^|_)(first|last|middle|full)(_|name)?(_?name)?$|(^|_)name$", "person_name", "Employee / person detail"),
    (r"(employee|worker|staff|person|member)(_)?(name)?", "person_name", "Employee / person detail"),
    (r"(email|e[_ ]?mail)", "contact", "Contact / email details"),
    (r"(phone|mobile|contact[_ ]?no|tel(ephone)?)", "contact", "Contact / phone details"),
    (r"(address|city|state|region|country|location|district|zip|pin[_ ]?code|lat|lon|longitude|latitude)", "location", "Geographic / location info"),
    (r"(salary|wage|compensation|income|pay|benefit|gross|net[_ ]?(salary)?)", "money", "Salary / compensation"),
    (r"(revenue|price|cost|amount|fee|budget|freight|charge|value|premium|msrp)", "money", "Monetary figures"),
    (r"(joining|hire|start|end|birth|date|created|updated|timestamp|_at$|_date$|_time$)", "date", "Dates / time info"),
    (r"(dept|department|org(anization)?|team|division|branch|unit|group|company|firm)", "org", "Department / organisation"),
    (r"(^|_)(id|key|code|sku|no|num|number|ref)(_|$)", "id", "Identifier fields"),
    (r"(status|type|category|segment|class|role|gender|grade|tier|priority|level|stage|state)", "category", "Status / category info"),
    (r"(score|rating|rank|grade|percentile)", "score", "Scores / ratings"),
    (r"(qty|quantity|count|number_of|total|sum|size|num_)(_|$)|_count$|count$", "count", "Counts / quantities"),
    (r"(pct|percent|rate|ratio|share|\bper)", "percent", "Percentages / rates"),
    (r"(is_|has_|flag|active|enabled|disabled|eligible|approved|boolean)", "flag", "Yes/no flags"),
    (r"(title|description|note|comment|summary|text|message|content|detail|reason|name)", "text", "Descriptions / labels"),
]

_ROLE_MISS = "unknown"
_ROLE_MISS_LABEL = "Other detail"


def _match_role(column: str) -> str | None:
    lower = column.lower()
    for pattern, role, _phrase in _ROLE_RULES:
        if re.search(pattern, lower):
            return role
    return None


def detect_column_roles(columns: list[str]) -> dict[str, str]:
    """Map each column name to a semantic role label."""
    roles: dict[str, str] = {}
    for col in columns:
        role = _match_role(col)
        roles[col] = role or _ROLE_MISS
    return roles


def role_category(role: str) -> str:
    for _pattern, _role, phrase in _ROLE_RULES:
        if _role == role:
            return phrase
    return _ROLE_MISS_LABEL


_KIND_BY_DOMAIN = {
    "finance / accounting": "financial / accounting",
    "sales / marketing": "sales / marketing",
    "web / product analytics": "web / product analytics",
    "telecom / network": "telecom / network",
    "iot / devices": "IoT / device telemetry",
    "human resources": "employee / workforce",
    "medical / healthcare": "medical / healthcare",
    "education": "education",
    "EEG / clinical": "clinical / EEG",
    "molecular / chemistry": "molecular / chemistry",
}


def _infer_kind(columns: list[str], roles: dict[str, str], domain: dict) -> str:
    if domain and domain.get("confidence", 0) >= 0.5 and domain.get("label") != "Unknown":
        return _KIND_BY_DOMAIN.get(domain["label"], domain["label"])

    role_set = set(roles.values())
    lower_join = " ".join(columns).lower()
    if "person_name" in role_set and ("dept" in lower_join or "org" in role_set):
        return "employee / workforce"
    if "money" in role_set and "org" in role_set:
        return "financial / accounting"
    if "location" in role_set and "person_name" in role_set:
        return "people / contact"
    return "general / tabular"


def build_semantics(columns: list[str]) -> dict:
    """Deterministic one-line understanding of the dataset.

    Returns ``{"kind", "summary", "categories", "column_roles"}``. Column
    roles are the semantic label per column; ``categories`` deduplicates the
    per-role info phrases; ``summary`` is a human sentence grounded entirely
    in the column names.
    """
    roles = detect_column_roles(columns)
    domain = detect_domain(columns)
    kind = _infer_kind(columns, roles, domain)

    col_words = [c.replace("_", " ") for c in columns[:8]]
    article = "an" if kind.split("/")[0].strip().lower().startswith(("a", "e", "i", "o", "u")) else "a"
    summary = (
        f"This is {article} {kind} dataset. It contains records covering "
        + (", ".join(col_words) if col_words else "various fields")
        + "."
    )

    categories: list[str] = []
    seen: set[str] = set()
    for col in columns:
        role = roles.get(col, _ROLE_MISS)
        phrase = role_category(role)
        if phrase not in seen:
            seen.add(phrase)
            categories.append(phrase)
    if not categories:
        categories.append("Data content")

    return {
        "kind": kind,
        "summary": summary,
        "categories": categories,
        "column_roles": roles,
        "domain": domain,
    }