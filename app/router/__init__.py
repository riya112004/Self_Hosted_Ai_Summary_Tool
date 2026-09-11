"""Universal input classifier.

Decides at the entry point which pipeline an input should be routed to:

    INPUT
     ├── structured  -> JSON parser path
     ├── tabular     -> schema / profiler / analytics path (existing)
     └── document    -> text extraction / structure / chunking path

The classifier only looks at cheap, deterministic signals (shape, delimiters,
markers, extension). It never calls the LLM.
"""

import json


KINDS = ("structured", "tabular", "document")

_EXT_KIND = {
    "csv": "tabular",
    "tsv": "tabular",
    "tab": "tabular",
    "xlsx": "tabular",
    "xls": "tabular",
    "json": "structured",
    "jsonl": "structured",
    "txt": "document",
    "text": "document",
    "md": "document",
    "markdown": "document",
    "html": "document",
    "htm": "document",
    "rst": "document",
    "log": "document",
    "pdf": "document",
}

PDF_MAGIC = b"%PDF-"


def _classify_text(text: str) -> dict:
    lines = [ln for ln in text.splitlines()]
    first = (text.lstrip()[:1]) if text.strip() else ""

    # 1) JSON-looking
    if first == "[" or first == "{":
        try:
            json.loads(text)
            return {
                "kind": "structured",
                "subtype": "json",
                "confidence": 1.0,
                "reason": "content parses as JSON",
            }
        except Exception:
            pass  # fall through - maybe markdown/csv

    # 2) Markdown heading is a strong document signal
    if any(ln.lstrip().startswith("#") for ln in lines if ln.strip()):
        return {
            "kind": "document",
            "subtype": "markdown",
            "confidence": 0.95,
            "reason": "markdown headings detected",
        }

    # 3) Delimited table: header + rows sharing a separator. Skip prose lines
    #    that contain no records (e.g. paragraphs with a stray comma).
    nonblank = [ln for ln in lines if ln.strip()]
    if len(nonblank) >= 2:
        best = None
        for delim in ("\t", ";", "|", ","):
            counts = [ln.count(delim) for ln in nonblank]
            non_zero = [c for c in counts if c > 0]
            if len(non_zero) < len(nonblank) * 0.8:
                continue
            # header must not grossly mismatch the row shape
            if counts[0] == 0:
                continue
            consistent = sum(1 for c in counts if c == counts[0]) / len(counts)
            if consistent >= 0.7:
                best = (delim, consistent, counts[0])
                break
        if best:
            delim, consistent, cols = best
            subtype = {"\t": "tsv", "|": "csv", ";": "csv", ",": "csv"}[delim]
            return {
                "kind": "tabular",
                "subtype": subtype,
                "confidence": round(consistent, 2),
                "reason": f"{cols + 1}-column delimiter table ({delim!r})",
            }

    # 4) Everything else is prose -> document
    return {
        "kind": "document",
        "subtype": "plain-text",
        "confidence": 0.7,
        "reason": "free-form text / prose",
    }


def classify(content, source_type: str | None = None) -> dict:
    """
    Classify any accepted input (raw text, list, dict) into a pipeline kind.

    An explicit ``source_type`` hint (from the caller or the file extension)
    wins; otherwise the content is classified by shape.
    """
    if isinstance(content, (list, dict)):
        return {
            "kind": "structured",
            "subtype": "json",
            "confidence": 1.0,
            "reason": "already structured data",
        }

    if not isinstance(content, str):
        raise ValueError("Unsupported input type")

    if content.startswith("%PDF-"):
        return {
            "kind": "document",
            "subtype": "pdf",
            "confidence": 1.0,
            "reason": "PDF magic bytes detected",
        }

    if source_type:
        key = (source_type or "").strip().lower()
        key = key.split("/")[-1]  # tolerate mime types like text/markdown
        if key in _EXT_KIND:
            return {
                "kind": _EXT_KIND[key],
                "subtype": key,
                "confidence": 0.9,
                "reason": f"explicit source_type={key!r}",
            }

    text = content.lstrip("\ufeff")
    result = _classify_text(text)
    result["subtype"] = result.get("subtype") or ""
    return result