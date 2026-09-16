"""Document summarization orchestration.

Rule that keeps this trustworthy:

    Python owns facts   -> counts, sections, chunks, structure, metadata
    LLM owns meaning    -> topic, sections' role, narrative, technologies

Where possible every claim is paired with a "source_section" that must match
a real heading in the document - fabricated sections are dropped
deterministically after the LLM responds. Long documents (> threshold chars)
are summarized via map-reduce: each chunk separately, then a final synthesis.
"""

import logging
import os
import re

from app.documents import detect_structure, doc_metrics, section_outline, section_anchored_text
from app.documents.chunker import chunk_by_words, chunk_sections, needs_map_reduce
from app.llm.universal import build_system_section
from app.providers import get_provider
from app.schemas.document_summary import (
    DOCUMENT_SUMMARY_SCHEMA,
    DocumentSummary,
    DocumentUnderstanding,
    Evidence,
    SourceMetadata,
)
from app.schemas.summary import extract_json

# Chunk bands for map-reduce. Bigger chunks = fewer sequential LLM round
# trips. Values are read when a run starts so dotenv-loaded settings apply.


def _env_int(name: str, default: int) -> int:
    """Read an int env var defensively; empty/invalid values fall back."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning("ignoring non-integer %s=%r", name, raw)
        return default


def _chunk_settings() -> tuple[int, int, int]:
    return (
        _env_int("DOC_CHUNK_MIN_WORDS", 500),
        _env_int("DOC_CHUNK_TARGET_WORDS", 800),
        _env_int("DOC_CHUNK_MAX_WORDS", 1000),
    )


def build_document_prompt(structure, body: str | None = None) -> str:
    metadata_line = (
        f"- title: {structure.title}\n"
        f"- word_count: {structure.word_count}\n"
        f"- char_count: {structure.char_count}\n"
        f"- sections: {len(structure.sections)}\n"
        f"- headings: {structure.heading_count}\n"
        f"- paragraphs: {structure.paragraph_count}\n"
        f"- code blocks: {structure.code_block_count}\n"
        f"- tables: {structure.table_count}"
    )
    text_block = body if body else section_anchored_text(structure)

    return (
        build_system_section(kind="document")
        + "\n\nDETERMINISTIC METADATA\n"
        + metadata_line
        + "\n\nDOCUMENT TEXT\n"
        + text_block
        + "\n\nJSON OUTPUT (STRICT)\n"
        'Return STRICT JSON only - no markdown, no code fences, matching:\n'
        + DOCUMENT_SUMMARY_SCHEMA
        + '\n\n"evidence" items must cite a "source_section" that is an exact heading '
        'present in the document. Do not invent section names. "summary" keys must '
        "map to strings or arrays of strings - never nested objects. Be concise; "
        'aim for under 800 words total.'
    )


def _chunk_summary_prompt(chunk) -> str:
    label = ", ".join(chunk.sections) if chunk.sections else "this part"
    return (
        build_system_section(kind="document")
        + f"\n\nSummarize ONLY the {label} part of the document shown below. "
        "Write a plain-prose summary (~120-150 words) using only facts present "
        "in this chunk. Cover every main point; do not refer to other parts. "
        "Return plain prose only - NO JSON, no bullets, no markdown.\n\n"
        f"CHUNK ({chunk.word_count} words)\n{chunk.text[:8000]}"
    )


class DocumentSummarizer:
    def __init__(self, provider=None):
        self.provider = provider or get_provider()

    def summarize(
        self,
        content: str,
        source_type: str | None = None,
        run_llm: bool = True,
    ) -> DocumentSummary:
        from app.timing import mark, start

        t_total = start()
        t0 = start()
        structure = detect_structure(content, source_type)
        mark("document: detect_structure", t0)
        metadata = SourceMetadata(**doc_metrics(structure))

        if not run_llm:
            result = self._deterministic(structure, metadata)
        else:
            try:
                if needs_map_reduce(structure.char_count):
                    result = self._map_reduce(structure, metadata)
                else:
                    result = self._single_pass(structure, metadata)
            except Exception as exc:
                # Mirror the tabular path (_run_summary): an LLM failure must
                # degrade to a usable deterministic digest, never a 500.
                logging.getLogger(__name__).exception(
                    "LLM document summary failed; using deterministic digest"
                )
                result = self._deterministic(
                    structure, metadata, llm_error=str(exc)
                )

        mark(f"document: total (run_llm={run_llm})", t_total)
        return result

    # ---------- paths ----------

    def _deterministic(
        self,
        structure,
        metadata: SourceMetadata,
        llm_error: str | None = None,
    ) -> DocumentSummary:
        outline = [f"{'#' * s.level} {s.heading}" for s in structure.sections]
        digest = {
            "outline": outline or ["(no headings detected)"],
            "first_words": " ".join(section_anchored_text(structure, 3000).split())[:600],
        }
        if llm_error:
            digest["llm_error"] = llm_error
        return DocumentSummary(
            title=structure.title,
            summary=digest,
            source_metadata=metadata,
        )

    def _single_pass(self, structure, metadata: SourceMetadata) -> DocumentSummary:
        from app.timing import mark, start

        t0 = start()
        prompt = build_document_prompt(structure)
        mark("document: build_prompt", t0)

        # Long structured output: the full document JSON (title + understanding +
        # key_points + technologies + risks + evidence) far exceeds the shared
        # 500-token cap, and the strict-JSON instruction sits after the document
        # text so the default 2048-token context can truncate it away. Give the
        # document pass its own larger budget per call.
        t0 = start()
        text = self.provider.chat(
            prompt,
            json_mode=True,
            options={
                "num_ctx": _env_int("DOC_NUM_CTX", 8192),
                "num_predict": _env_int("DOC_NUM_PREDICT", 2000),

            },
        )
        mark("document: LLM single_pass", t0)

        t0 = start()
        result = self._apply_result(text, structure, metadata)
        mark("document: apply_result (guards)", t0)
        return result

    def _map_reduce(self, structure, metadata: SourceMetadata) -> DocumentSummary:
        from app.timing import mark, start

        min_words, target_words, max_words = _chunk_settings()
        chunks = chunk_by_words(
            structure.sections,
            min_words=min_words,
            target_words=target_words,
            max_words=max_words,
        )
        metadata.chunk_count = len(chunks)
        print(
            f"[step] document: map-reduce with {len(chunks)} chunk(s), "
            f"{min_words}-{max_words} words each"
        )

        parts = []
        chunk_failures = 0
        for i, chunk in enumerate(chunks, start=1):
            t0 = start()
            try:
                raw = self.provider.chat(_chunk_summary_prompt(chunk), json_mode=False)
                if raw.strip():
                    parts.append((chunk, raw.strip()))
                mark(f"document: LLM chunk {i}/{len(chunks)} ({chunk.word_count} words)", t0)
            except Exception as exc:
                chunk_failures += 1
                print(f"[step] document: chunk {i}/{len(chunks)} failed: {exc}")

        if not parts:
            raise RuntimeError(
                f"all {len(chunks)} document chunk summaries failed "
                f"({chunk_failures}/{len(chunks)} chunks errored)"
            )

        # assemble every chunk summary into one comprehensive overview - no
        # second synthesis pass that would re-compress and drop content
        blocks = []
        for i, (chunk, text) in enumerate(parts, start=1):
            label = ", ".join(chunk.sections) if chunk.sections else f"part {i}"
            blocks.append(f"[Part {i} - {label}]\n{text}")
        if chunk_failures:
            blocks.append(
                f"(Note: {chunk_failures}/{len(chunks)} chunks failed during "
                "map-reduce; the parts above cover the surviving chunks.)"
            )

        return DocumentSummary(
            title=structure.title,
            summary={"overview": "\n\n".join(blocks)},
            source_metadata=metadata,
        )

    # ---------- parsing + fact guard ----------

    def _apply_result(
        self, raw: str, structure, metadata: SourceMetadata
    ) -> DocumentSummary:
        data = None
        try:
            data = extract_json(raw)
            summary = DocumentSummary.model_validate(data)
        except Exception:
            summary = self._rescue_partial(data if isinstance(data, dict) else {}, raw, structure)

        summary.title = summary.title or structure.title
        summary.source_metadata = metadata
        summary.document_understanding.document_type = summary.document_understanding.document_type or "document"

        # deterministic guards: never cite invented sections, never let the
        # LLM fill metadata or fabricate flagged numbers
        headings = [h.casefold() for s in structure.sections for h in [s.heading]]
        summary.evidence = [
            e for e in summary.evidence
            if e.finding and e.source_section
            and _matches_heading(e.source_section, headings)
        ]
        summary.important_numbers = [
            n for n in summary.important_numbers if re.search(r"\d", n)
        ]
        summary.technologies = [
            t for t in summary.technologies
            if _mentioned(t, structure)
        ]
        return summary

    @staticmethod
    def _rescue_partial(data: dict, raw: str, structure) -> DocumentSummary:
        """When model_validate fails (common: summary returned as a flat
        string instead of a dict), salvage every field that *did* parse
        correctly and coalesce the summary field into a dict.
        When nothing parsed at all, degrade to a deterministic digest - never
        dump raw LLM prose into the summary."""
        if not data:
            outline = [s.heading for s in structure.sections]
            return DocumentSummary(
                title=structure.title,
                summary={
                    "outline": outline or ["(no headings detected)"],
                    "first_words": " ".join(
                        section_anchored_text(structure, 3000).split()
                    )[:600],
                    "llm_error": "LLM output was not parseable JSON; "
                    "showing deterministic digest.",
                },
            )

        summary_value = data.get("summary")
        if isinstance(summary_value, str):
            data["summary"] = {"overview": summary_value}
        elif isinstance(summary_value, list):
            data["summary"] = {"key_findings": [str(x) for x in summary_value]}
        elif not isinstance(summary_value, dict):
            data["summary"] = {}

        try:
            return DocumentSummary.model_validate(data)
        except Exception:
            s = DocumentSummary(
                title=data.get("title") or structure.title,
                document_understanding=DocumentUnderstanding(**{
                    k: v for k, v in (data.get("document_understanding") or {}).items()
                    if k in DocumentUnderstanding.model_fields
                }),
                key_points=data.get("key_points") or [],
                important_numbers=data.get("important_numbers") or [],
                technologies=data.get("technologies") or [],
                risks=data.get("risks") or [],
                recommendations=data.get("recommendations") or [],
                evidence=[Evidence(**e) for e in (data.get("evidence") or []) if isinstance(e, dict)],
            )
            coalesced = data.get("summary")
            s.summary = coalesced if isinstance(coalesced, dict) else {"raw": raw[:1500]}
            return s


def _matches_heading(section: str, headings: list[str]) -> bool:
    ref = section.casefold()
    return any(h in ref or ref in h or section.casefold().strip("#/- ") in h for h in headings)


def _mentioned(tech: str, structure) -> bool:
    ref = tech.strip().casefold()
    if not ref or len(ref) < 3:
        return False
    haystack = f"{structure.title}\n{section_outline(structure)}"
    return any(sec.text.casefold().count(ref) for sec in structure.sections) or ref in haystack.casefold()


def summarize_document(content, source_type=None, run_llm=True) -> DocumentSummary:
    """Core entry point used by the API layer."""
    return DocumentSummarizer().summarize(content, source_type, run_llm)