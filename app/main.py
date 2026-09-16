import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import jobs
from .documents.pdf import extract_pdf_text
from .llm import LLMService

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AI Summary Tool")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logger = logging.getLogger("ai-summarizer")

llm_service = LLMService()


def _normalize_records(records):
    if not isinstance(records, list):
        return [{"value": records}] if records is not None else []
    normalized = []
    for item in records:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append({"value": item})
    return normalized


class AutoSummaryRequest(BaseModel):
    data: str | list | dict | None = None
    data_b64: str | None = None
    filename: str | None = None
    source_type: str | None = None
    run_llm: bool = True


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


def _run_summary(records, prefer_llm: bool = True) -> dict:
    from .profile import build_data_profile, build_representative_sample
    from .summary.deterministic import build_profile_summary
    from .timing import finish, mark, start

    t_total = start()
    records = _normalize_records(records)
    mark("_run_summary: normalize records", t_total)

    t0 = start()
    profile = build_data_profile(records)
    mark("_run_summary: build data profile", t0)

    if prefer_llm:
        try:
            t0 = start()
            sample = build_representative_sample(records)
            out = llm_service.generate_summary_from_profile(
                profile, auto_fallback=True, sample=sample
            )
            mark("_run_summary: LLM summary", t0)
            finish("_run_summary", t_total)
            return out.model_dump()
        except Exception:
            logger.exception("LLM summary failed; using deterministic summary")

    t0 = start()
    out = build_profile_summary(profile, records=records)
    mark("_run_summary: deterministic summary", t0)
    finish("_run_summary", t_total)
    return out.model_dump()


def _run_streamed_csv_summary(csv_text: str, prefer_llm: bool = True) -> dict:
    from .adapters.csv_stream import incremental_profile_csv
    from .summary.deterministic import build_profile_summary
    from .timing import finish, mark, start

    t_total = start()
    t0 = start()
    result = incremental_profile_csv(csv_text)
    profile, sample = result["profile"], result["sample"]
    mark(f"csv_stream: profile ({profile.get('profile_method')})", t0)

    if prefer_llm:
        try:
            t0 = start()
            out = llm_service.generate_summary_from_profile(
                profile, auto_fallback=True, sample=sample
            )
            mark("csv_stream: LLM summary", t0)
            summary = out.model_dump()
        except Exception:
            logger.exception("LLM summary failed; using deterministic summary")
            summary = build_profile_summary(profile, records=None).model_dump()
    else:
        summary = build_profile_summary(profile, records=None).model_dump()
    finish("_run_streamed_csv_summary", t_total)

    return {
        "record_count": profile["dataset"]["rows"],
        "profile_method": profile.get("profile_method"),
        "summary": summary,
    }


def _run_document_auto(content: str, source_format, request, detection: dict) -> dict:
    from .llm.document_service import summarize_document as run_doc_summary
    from .timing import mark, start

    t0 = start()
    result = run_doc_summary(
        content,
        source_type=source_format,
        run_llm=request.run_llm,
    )
    mark(f"_run_document_auto: document summary (run_llm={request.run_llm})", t0)
    return {
        "input_type": "document",
        "document_type": result.document_understanding.document_type or "document",
        "detection": detection,
        "pipeline": "document",
        "run_llm": request.run_llm,
        "summary": result.model_dump(),
    }


def _coerce_records(kind: str, content):
    if kind == "tabular":
        return content
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return _normalize_records(content)
    if isinstance(content, str):
        import json as _json

        try:
            parsed = _json.loads(content)
            if isinstance(parsed, list):
                return _normalize_records(parsed)
            if isinstance(parsed, dict):
                return [parsed]
            return [{"value": parsed}]
        except Exception:
            return [{"value": content}]
    return _normalize_records(content)


@app.post("/api/v1/summarize/auto")
def summarize_auto(request: AutoSummaryRequest):
    import base64

    from .router import classify as run_classifier
    from .timing import finish, mark, start

    t_total = start()

    content = request.data

    if request.data_b64:
        t0 = start()
        raw = base64.b64decode(request.data_b64)
        mark("summarize/auto: base64 decode", t0)
        if raw.startswith(b"%PDF-"):
            t0 = start()
            try:
                text = extract_pdf_text(raw)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF text extraction failed: {exc}",
                ) from exc
            mark("summarize/auto: extract_pdf_text", t0)
            if not text.strip():
                logger.warning("PDF extracted to empty text; returning 400")
                raise HTTPException(
                    status_code=400,
                    detail="PDF contains no extractable text (scanned or image-only PDF?)",
                )
            detection = {
                "kind": "document",
                "subtype": "pdf",
                "confidence": 1.0,
                "reason": "PDF magic bytes detected",
            }
            return _run_document_auto(text, "pdf", request, detection)

        filename = (request.filename or "").lower()
        src_hint = (request.source_type or "").lower().split("/")[-1]
        if (
            raw.startswith((b"PK\x03\x04", b"\xd0\xcf\x11\xe0"))
            or filename.endswith((".xlsx", ".xls"))
            or src_hint in ("xlsx", "xls")
        ):
            from .adapters.excel_adapter import convert_excel

            subtype = "xlsx" if raw.startswith(b"PK\x03\x04") else "xls"
            t0 = start()
            try:
                records = convert_excel(raw, source_type=subtype)["records"]
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Excel parsing failed: {exc}",
                ) from exc
            mark(f"summarize/auto: excel convert ({subtype})", t0)
            detection = {
                "kind": "tabular",
                "subtype": subtype,
                "confidence": 1.0,
                "reason": f"{subtype} workbook detected",
            }
            summary = _run_summary(records, prefer_llm=request.run_llm)
            finish("summarize/auto", t_total)
            return {
                "input_type": "tabular",
                "detection": detection,
                "pipeline": "data",
                "record_count": len(records),
                "summary": summary,
            }

        t0 = start()
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw.decode("latin-1", errors="replace")
        mark("summarize/auto: decode bytes", t0)

    t0 = start()
    detection = run_classifier(content, request.source_type)
    mark("summarize/auto: classify input", t0)
    kind = detection["kind"]

    if kind in ("tabular", "structured"):
        if kind == "structured" and isinstance(content, str):
            import json as _json

            try:
                parsed_content = _json.loads(content)
            except Exception:
                parsed_content = None
        else:
            parsed_content = content

        from .summary.structured_report import build_report_summary, is_nested_report

        if isinstance(parsed_content, dict) and is_nested_report(parsed_content):
            t0 = start()
            summary = build_report_summary(
                parsed_content,
                prefer_llm=request.run_llm,
                llm_service=llm_service,
            ).model_dump()
            mark("summarize/auto: structured report digest", t0)
            finish("summarize/auto", t_total)
            return {
                "input_type": "structured",
                "detection": {
                    "kind": "structured",
                    "subtype": "nested-json",
                    "confidence": 1.0,
                    "reason": "nested JSON report detected (object with array/object values)",
                },
                "pipeline": "report",
                "run_llm": request.run_llm,
                "summary": summary,
            }

        t0 = start()
        records = _coerce_records(kind, content)
        if kind == "tabular":
            from .adapters.csv_stream import should_stream_csv

            if isinstance(records, str) and should_stream_csv(records):
                streamed = _run_streamed_csv_summary(records, prefer_llm=request.run_llm)
                finish("summarize/auto", t_total)
                return {
                    "input_type": kind,
                    "detection": detection,
                    "pipeline": "data",
                    "record_count": streamed["record_count"],
                    "profile_method": streamed["profile_method"],
                    "summary": streamed["summary"],
                }

            from .adapters.csv_adapter import convert_csv

            try:
                records = convert_csv(records)["records"]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        mark("summarize/auto: coerce -> records", t0)
        summary = _run_summary(records, prefer_llm=request.run_llm)
        finish("summarize/auto", t_total)
        return {
            "input_type": kind,
            "detection": detection,
            "pipeline": "data",
            "record_count": len(records),
            "summary": summary,
        }

    source_format = (
        None
        if request.source_type in ("auto", "document", None)
        else request.source_type
    )
    out = _run_document_auto(content, source_format, request, detection)
    finish("summarize/auto", t_total)
    return out


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "jobs": jobs.stats(),
    }


@app.get("/api/v1/models")
def models():
    try:
        provider = llm_service.provider
        return {
            "provider": getattr(provider, "name", "unknown"),
            "models": provider.list_models(),
        }
    except Exception as exc:
        logger.exception("Failed to list models")
        raise HTTPException(
            status_code=503,
            detail=f"Inference server unreachable. (error: {exc})",
        ) from exc
