"""PDF text extraction.

Kept isolated so PDF support never slows down the rest of the pipeline -
pypdf is imported lazily and only when a PDF actually arrives.
"""

import logging

logger = logging.getLogger(__name__)


def extract_pdf_text(data: bytes) -> str:
    import io

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support requires the 'pypdf' package (pip install pypdf)"
        ) from exc

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    result = "\n\n".join(pages).strip()
    logger.info("extracted %d chars", len(result))
    return result