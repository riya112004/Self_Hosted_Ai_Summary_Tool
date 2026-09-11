"""Semantic chunking for long documents.

Python splits at section/paragraph boundaries so no meaning is cut mid-way.
Each chunk is summarized separately and the chunk summaries are then
synthesized into the final document narrative.
"""

from dataclasses import dataclass, field
import re


@dataclass
class Chunk:
    index: int
    text: str
    sections: list[str] = field(default_factory=list)
    word_count: int = 0


def chunk_sections(
    sections, max_chars: int = 6000, max_chunks: int = 12
) -> list[Chunk]:
    """
    Group adjacent sections into chunks that stay under ``max_chars``.
    A single oversized section is split on paragraph boundaries.
    """
    chunks: list[Chunk] = []
    current = Chunk(index=0, text="", sections=[])

    def flush():
        nonlocal current
        if current.text.strip():
            current.word_count = len(current.text.split())
            chunks.append(current)
        current = Chunk(index=len(chunks) + 1, text="", sections=[])

    for sec in sections:
        block = f"## {sec.heading}\n{sec.text}"
        if sec.text.strip():
            current.sections.append(sec.heading)

        if len(current.text) + len(block) <= max_chars:
            if len(current.sections) == 1:
                current.text += block
            else:
                current.text += "\n\n" + block
            continue

        flush()
        if len(block) <= max_chars:
            current.text = block
            current.sections = [sec.heading]
        else:
            # oversized single section -> split on paragraphs
            for para in re.split(r"\n\s*\n", sec.text):
                fragment = f"## {sec.heading}\n{para}"
                if len(current.text) + len(fragment) > max_chars and current.text.strip():
                    flush()
                current.text += ("\n\n" if current.text.strip() else "") + fragment
                if sec.heading not in current.sections:
                    current.sections.append(sec.heading)

    flush()
    return chunks[:max_chunks]


def needs_map_reduce(text_chars: int, threshold: int = 12000) -> bool:
    return text_chars > threshold


def chunk_by_words(
    sections,
    min_words: int = 500,
    target_words: int = 800,
    max_words: int = 1000,
) -> list[Chunk]:
    """
    Word-count based chunking for LLM map-reduce.

    Each chunk stays in the 500-1000 word band (configurable): paragraphs are
    accumulated across section boundaries until the next one would push the
    chunk over ``max_words`` while the chunk already has ``min_words``.
    Oversized single paragraphs are split into ``target_words`` fragments so
    no chunk asks the model to swallow a wall of text in one go.
    """
    chunks: list[Chunk] = []
    current = Chunk(index=0, text="", sections=[])

    def flush():
        nonlocal current
        if current.text.strip():
            current.word_count = len(current.text.split())
            chunks.append(current)
        current = Chunk(index=len(chunks) + 1, text="", sections=[])

    def add(fragment: str, heading: str):
        nonlocal current
        if (
            current.text.strip()
            and current.word_count >= min_words
            and current.word_count + len(fragment.split()) > max_words
        ):
            flush()
        if heading not in current.sections:
            current.sections.append(heading)
        if current.text.strip():
            current.text += "\n\n" + fragment
        else:
            current.text = fragment
        current.word_count = len(current.text.split())

    for sec in sections:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec.text) if p.strip()]
        if not paragraphs:
            continue
        for para in paragraphs:
            words = para.split()
            if len(words) <= max_words:
                add(f"## {sec.heading}\n{para}", sec.heading)
            else:
                # giant paragraph -> brake into target-sized fragments
                for i in range(0, len(words), target_words):
                    add(
                        f"## {sec.heading}\n{' '.join(words[i:i + target_words])}",
                        sec.heading,
                    )

    flush()
    return chunks