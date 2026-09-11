"""Pydantic validation layer for the document summarization pipeline."""

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    """Deterministic document metrics - computed by Python, never the LLM."""

    word_count: int = 0
    char_count: int = 0
    section_count: int = 0
    paragraph_count: int = 0
    table_count: int = 0
    code_block_count: int = 0
    heading_count: int = 0
    chunk_count: int = 0


class DocumentUnderstanding(BaseModel):
    topic: str = ""
    document_purpose: str = ""
    audience: str = ""
    domain: str = ""
    document_type: str = ""


class Evidence(BaseModel):
    """A claim paired with the span of the source document that supports it."""

    finding: str = ""
    evidence: str = ""
    source_section: str = ""


class DocumentSummary(BaseModel):
    title: str = ""
    document_understanding: DocumentUnderstanding = Field(
        default_factory=DocumentUnderstanding
    )
    summary: dict = Field(default_factory=dict)
    key_points: list[str] = Field(default_factory=list)
    important_numbers: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    source_metadata: SourceMetadata | None = None


DOCUMENT_SUMMARY_SCHEMA = """{
  "title": string,
  "document_understanding": {
    "topic": string,
    "document_purpose": string,
    "audience": string,
    "domain": string,
    "document_type": string
  },
  "summary": {
    "any mode-specific keys": "each key a string or a list of strings,
                              NEVER a nested object"
  },
  "key_points": ["bullet", ...],
  "important_numbers": ["e.g. '2.5x faster', '100k rows' - only numbers that
                         appear in the document text", ...],
  "technologies": ["only tools/tech explicitly mentioned", ...],
  "risks": ["concerns/limitations explicitly stated or clearly implied", ...],
  "recommendations": ["actions grounded in the text", ...],
  "evidence": [
    {
      "finding": "one concrete claim",
      "evidence": "short verbatim-ish quote or precise paraphrase from the text",
      "source_section": "exact heading of the section that supports this claim"
    }
  ]
}"""


def document_modes() -> list[str]:
    """Document-kind modes - single source lives in app.llm.universal.FOCUS."""
    from app.llm.universal import document_modes as _doc_modes

    return _doc_modes()