"""Document text pipeline: extract -> structure -> deterministic metrics.

Python owns every fact here. The LLM never counts words, sections or code
blocks; this module computes structural metadata deterministically and hands
the LLM a compact, section-anchored view of the text so its claims can carry
"source_section" evidence.
"""

import re
from dataclasses import dataclass, field

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# PDF exporters print page footers/headers on every page ("Page 1",
# "Page 1 of 20", "3/20", bare page digits). pypdf often extracts them
# before the real body text, so they would steal the title slot. Blank
# them before any structure detection; real content like "Page 1:
# Introduction" (has a colon + text) or "Climate Change Overview"
# never matches.
PAGE_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"page\s*\d+(?:\s*of\s*\d+)?"
    r"|\d+\s*/\s*\d+"
    r"|\d+\s+of\s+\d+"
    r"|\d{1,4}"
    r")\s*$",
    re.IGNORECASE,
)


@dataclass
class Section:
    heading: str
    level: int
    line: int
    text: str = ""


@dataclass
class DocumentStructure:
    title: str
    sections: list = field(default_factory=list)
    word_count: int = 0
    char_count: int = 0
    paragraph_count: int = 0
    table_count: int = 0
    code_block_count: int = 0
    heading_count: int = 0


def _strip_html(text: str) -> str:
    if "<" not in text or ">" not in text:
        return text
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return text


def extract_text(content: str, source_type: str | None = None) -> str:
    """Normalize any supported text source into clean UTF-8 text."""
    if not isinstance(content, str):
        raise ValueError("Document content must be text")

    text = content.lstrip("\ufeff")
    key = (source_type or "").lower()
    if key in ("html", "htm"):
        text = _strip_html(text)

    # normalize windows vs unix newlines; strip trailing whitespace per line
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def _is_plain_heading(line: str) -> bool:
    s = line.strip()
    if not (2 <= len(s) <= 90):
        return False
    if s.endswith(".") and len(s.split()) > 6:
        return False  # likely a sentence
    if re.match(r"^(\d{1,3}([\.\)])\s+|\*|-|\u2022\s)", s):
        return True
    # mostly-uppercase short line
    letters = [c for c in s if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) >= 0.8:
        return True
    return False


def detect_structure(text: str, source_type: str | None = None) -> DocumentStructure:
    """Split text into sections, counting structure. Heading text is preserved."""
    text = extract_text(text, source_type)
    lines = [
        "" if PAGE_MARKER_RE.match(line) else line for line in text.split("\n")
    ]
    text = "\n".join(lines)

    title = ""
    sections: list[Section] = []

    code_mode = False
    code_blocks = 0
    tables = 0
    prev_blank = True

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        if re.match(r"^```", line):
            code_mode = not code_mode
            if code_mode:
                code_blocks += 1
            i += 1
            continue

        if not code_mode and line.startswith("|") and "|" in line[1:]:
            tables += 1

        heading = None
        md = HEADING_RE.match(line)
        if md:
            heading = (md.group(2).strip(), len(md.group(1)))
        elif not code_mode and not prev_blank and _is_plain_heading(line):
            heading = (line, 3)

        if heading:
            if not title:
                # the first heading is the document title, not a content section
                title = heading[0]
            else:
                sections.append(Section(heading=heading[0], level=heading[1], line=i))
        elif not title and line and i < 3:
            title = line[:120]

        prev_blank = line == ""
        i += 1

    # assign body text to each section
    for idx, sec in enumerate(sections):
        end = sections[idx + 1].line if idx + 1 < len(sections) else len(lines)
        sec.text = "\n".join(lines[sec.line : end]).strip("\n")

    if not sections and text.strip():
        sections.append(Section(heading=title or "Document", level=1, line=0, text=text.strip()))

    paragraphs = 0
    for block in re.split(r"\n\s*\n", text):
        if block.strip():
            paragraphs += 1

    return DocumentStructure(
        title=title or "Untitled document",
        sections=sections,
        word_count=len(re.findall(r"\S+", text)),
        char_count=len(text),
        paragraph_count=paragraphs,
        table_count=tables,
        code_block_count=code_blocks,
        heading_count=len(sections),
    )


def doc_metrics(structure: DocumentStructure) -> dict:
    """Deterministic document-level metrics (Python, never the LLM)."""
    return {
        "word_count": structure.word_count,
        "char_count": structure.char_count,
        "section_count": len(structure.sections),
        "paragraph_count": structure.paragraph_count,
        "table_count": structure.table_count,
        "code_block_count": structure.code_block_count,
        "heading_count": structure.heading_count,
    }


def section_outline(structure: DocumentStructure, max_words: int = 120) -> str:
    """Compact section listing with brief per-section tags for the prompt."""
    parts = []
    for sec in structure.sections:
        preview = " ".join(sec.text.split())[:120]
        parts.append(f"- [{sec.level}] {sec.heading}: {preview}")
    return "\n".join(parts)


def section_anchored_text(structure: DocumentStructure, max_chars: int = 20000) -> str:
    """Text with headings preserved, trimmed to a hard bound for the prompt."""
    chunks = []
    used = 0
    for sec in structure.sections[:40]:
        block = f"{'#' * sec.level} {sec.heading}\n{sec.text}"
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)
    if used == 0:
        chunks.append(structure.sections[0].text[:max_chars])
    return "\n\n".join(chunks)