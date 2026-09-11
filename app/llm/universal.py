"""Single universal prompt contract for every input type.

ONE canonical rule set + ONE focus table drive all summary-generation
prompts. Nothing else about "how to summarise" is written anywhere else
in the live LLM path:

    Python computes facts -> verified profile / structural metadata
    LLM explains ONLY those facts
    every claim grounded, nothing invented, STRICT JSON out

The only difference between input types is WHAT Python hands the LLM
(the verified data profile for tables, structural metadata + text for
documents). The prompt prose itself is shared by all of them; the mode
focus lines are data-driven config, never hand-written per input type.
"""

UNIVERSAL_RULES = """You are a summarization engine.

Generate a concise summary using ONLY the verified information provided
below. You never receive raw data - only facts computed by Python.

Rules:
1. Do not invent dataset meaning.
2. Do not infer units unless explicitly provided.
3. Do not call something anomalous unless it was detected.
4. Preserve important numerical metrics exactly.
5. If domain confidence is low, say "domain uncertain".
6. Separate facts from recommendations.
7. Never introduce information not present in the supplied facts.
8. Do not recalculate any statistic - trust the computed numbers.
9. Return STRICT JSON only (no markdown, no code fences).
10. Write values directly - never wrap them in {"title": ..., "type": ...}
    objects, even if the target schema looks nested."""

FOCUS = {
    "system": {
        "data": [
            "overall structure (rows / columns) and type mix",
            "data quality facts (missing, duplicates, constant columns)",
            "key numerical facts and notable statistics",
            "detected domain and its confidence",
            "actionable recommendations grounded in the profile",
        ]
    },
    "executive": {
        "data": [
            "top-level KPIs and scale",
            "domain and business-relevant facts",
            "major risks or quality issues",
            "clear recommendations",
        ],
        "document": [
            "Overview",
            "Key Findings",
            "Business Impact",
            "Important Recommendations",
        ],
    },
    "financial": {
        "data": [
            "anything that looks financial (revenue, cost, margins, amounts)",
            "financial anomalies if any were detected",
            "recommendations with financial impact",
        ]
    },
    "technical": {
        "data": [
            "schema, types, quality and scale facts",
            "correlations and patterns",
            "data engineering caveats (constant columns, imbalance)",
            "technical recommendations",
        ],
        "document": [
            "Architecture",
            "Components",
            "Technology Stack",
            "Data Flow",
            "Implementation",
            "Security",
            "Performance",
            "Scalability",
        ],
    },
    "research": {
        "document": [
            "Research Objective",
            "Problem",
            "Approach",
            "Key Findings",
            "Evidence",
            "Limitations",
            "Conclusion",
        ]
    },
    "educational": {
        "document": [
            "Main Concepts",
            "Important Terms",
            "Step-by-Step Explanation",
            "Key Takeaways",
        ]
    },
    "general": {
        "document": [
            "What this is about",
            "Why it matters",
            "How it works",
            "Key recommendations",
        ]
    },
}


def focus_for(mode: str, kind: str = "data") -> list[str]:
    """Focus lines for a mode within an input kind (data | document)."""
    focuses = FOCUS.get(mode, {})
    lines = focuses.get(kind) or focuses.get("data") or FOCUS["system"]["data"]
    return list(lines)


def document_modes() -> list[str]:
    """All modes that have a document-kind focus."""
    return [m for m, cfg in FOCUS.items() if "document" in cfg]


def build_system_section(mode: str, kind: str = "data") -> str:
    """Universal system block: the shared rules + the mode's focus lines."""
    lines = focus_for(mode, kind)
    focus_block = "\n".join(f"- {item}" for item in lines)
    return f"{UNIVERSAL_RULES}\n\nFocus on:\n{focus_block}"