"""Single universal prompt contract for every input type.

ONE canonical rule set + ONE focus set drive all summary-generation
prompts. Nothing else about "how to summarise" is written anywhere else
in the live LLM path:

    Python computes facts -> verified profile / structural metadata
    LLM explains ONLY those facts
    every claim grounded, nothing invented, STRICT JSON out

The only difference between input types is WHAT Python hands the LLM
(the verified data profile for tables, structural metadata + text for
documents). The prompt prose itself is shared by all of them.
"""

UNIVERSAL_RULES = """You are a data analyst.

Analyze the provided data profile.

First determine what the data represents based only on the field names,
representative values, structure and statistics. Then generate a concise
factual summary.

Rules:
1. Do not assign a domain/category without evidence.
2. Do not use generic categories such as financial, accounting, sales, HR,
   etc. unless supported by the data.
3. Use actual field names and representative values as evidence.
4. Mention the main entity represented by the dataset.
5. Mention important attributes and meaningful statistics.
6. Do not invent information - never make claims not present in the profile.
7. Do not simply list column types.
8. Avoid generic statements such as "other columns are categorical".
9. If the domain cannot be confidently identified, describe the dataset
   neutrally instead of guessing.
10. Preserve important numerical metrics exactly and never recalculate them.
11. Return STRICT JSON only (no markdown, no code fences).
12. Write values directly - never wrap them in {"title": ..., "type": ...}
    objects, even if the target schema looks nested.
13. A "REPRESENTATIVE SAMPLE" section, if present, shows a tiny first/random/
    last excerpt for row-shape context only - never generalize dataset-level
    claims from the sample; always anchor them in the verified profile."""

FOCUS = {
    "data": [
        "what the data represents (based on field names and values)",
        "the main entity of the dataset",
        "important attributes and meaningful statistics",
        "structure of the data (records, columns, types)",
    ],
    "document": [
        "What this is about",
        "What it contains",
        "How it is organised",
        "Key points and recommendations",
    ],
}


def focus_for(kind: str = "data") -> list[str]:
    """Shared focus lines for an input kind (data | document)."""
    return list(FOCUS.get(kind, FOCUS["data"]))


def build_system_section(kind: str = "data") -> str:
    """Universal system block: the shared rules + the focus lines."""
    lines = focus_for(kind)
    focus_block = "\n".join(f"- {item}" for item in lines)
    return f"{UNIVERSAL_RULES}\n\nFocus on:\n{focus_block}"