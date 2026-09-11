import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import jobs
from .documents.pdf import extract_pdf_text
from .llm import LLMService
from .llm.prompt_builder import build_prompt
from .modes import list_modes
from .quality import check_quality
from .schemas.document_summary import document_modes
from .schemas.summary import StructuredOutputError

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="AI Summary Tool")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logger = logging.getLogger("ai-summarizer")

llm_service = LLMService()


def _normalize_records(records):
    """Coerce list-like payloads into record dictionaries expected by the analysis pipeline."""
    if not isinstance(records, list):
        return [{"value": records}] if records is not None else []

    normalized = []
    for item in records:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append({"value": item})
    return normalized


class DataRequest(BaseModel):
    data: list
    summary_type: str = "system"
    run_llm: bool = True


class ConvertRequest(BaseModel):
    data: list | dict
    source_type: str = "json"


class SampleRequest(BaseModel):
    data: list
    n: int = 5
    sample_type: str = "mixed"


class SanitizeRequest(BaseModel):
    data: list
    policy: str = "mask"
    remove_columns: list[str] = []


class AccessRequest(BaseModel):
    data: list
    role: str


class ConnectionConfig(BaseModel):
    uri: str | None = None
    db: str | None = None
    collection: str | None = None


class QueryStatement(BaseModel):
    query_type: str
    query: dict | list | None = None
    projection: dict | None = None
    sort: dict | None = None
    key: str | None = None
    limit: int = 100


class QueryRequest(BaseModel):
    connection: ConnectionConfig | None = None
    statement: QueryStatement


class CsvSummaryRequest(BaseModel):
    data: str
    summary_type: str = "system"
    run_llm: bool = True


class DatabaseSummaryRequest(BaseModel):
    connection: ConnectionConfig | None = None
    statement: QueryStatement
    summary_type: str = "system"
    run_llm: bool = True


class DocumentSummaryRequest(BaseModel):
    content: str
    source_type: str = "auto"
    summary_type: str = "general"
    run_llm: bool = True


class ClassifyRequest(BaseModel):
    data: str | list | dict
    source_type: str | None = None


class FullTestRequest(BaseModel):
    data: list | dict | str
    source_type: str = "json"
    summary_type: str = "system"
    run_llm: bool = True
    sanitize: bool = True
    connection: ConnectionConfig | None = None
    statement: QueryStatement | None = None


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/v1/modes")
def modes():
    return {
        "modes": list_modes()
    }


@app.post("/api/v1/convert")
def convert(request: ConvertRequest):
    from .adapters import convert as adapt_convert

    return adapt_convert(request.data, request.source_type)


@app.post("/api/v1/schema")
def schema(request: DataRequest):
    from .profiling.schema_detector import detect_schema

    return {
        "schema": detect_schema(request.data),
    }


@app.post("/api/v1/profile")
def profile(request: DataRequest):
    from .profiling.profiler import profile_dataset

    return profile_dataset(request.data)


@app.post("/api/v1/analytics")
def analytics(request: DataRequest):
    from .analytics import compute_analytics

    return compute_analytics(request.data)


@app.post("/api/v1/sample")
def sample(request: SampleRequest):
    from .sampling.representative import build_context

    return build_context(request.data, n=request.n, sample_type=request.sample_type)


@app.post("/api/v1/sanitize")
def sanitize(request: SanitizeRequest):
    from .security.sanitizer import sanitize as sanitize_records

    return sanitize_records(
        request.data,
        policy=request.policy,
        remove_columns=request.remove_columns,
    )


@app.post("/api/v1/access")
def access(request: AccessRequest):
    from .security.permissions import access as role_access
    from .security.permissions import PermissionDenied

    try:
        return role_access(request.data, request.role)
    except PermissionDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/api/v1/pipeline")
def pipeline(request: DataRequest):
    from .pipeline import process

    return process(request.data)


@app.post("/api/v1/prompt")
def prompt(request: DataRequest):
    if request.summary_type not in list_modes():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown summary_type: {request.summary_type}. Available: {list_modes()}",
        )
    return {
        "prompt": build_prompt(request.data, mode=request.summary_type),
    }


def _validate_mode(summary_type: str) -> None:
    if summary_type not in list_modes():
        raise HTTPException(
            status_code=400,
            detail=f"Unknown summary_type: {summary_type}. Available: {list_modes()}",
        )


def _run_summary(records, summary_type: str, prefer_llm: bool = True) -> dict:
    """
    Universal rule: Python computes the verified data profile, then the LLM
    summarises ONLY that profile. The LLM never sees the raw dataset, so it
    cannot invent facts ("33 molecular entries"). The profile-based
    deterministic summary explains the data on its own terms even when the
    LLM is skipped entirely or fails.
    """
    from .profile import build_data_profile
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
            out = llm_service.generate_summary_from_profile(
                profile, summary_type, auto_fallback=True
            )
            mark(f"_run_summary: LLM summary (mode={summary_type})", t0)
            finish("_run_summary", t_total)
            return out.model_dump()
        except Exception:
            logger.exception("LLM summary failed; using deterministic summary")

    t0 = start()
    out = build_profile_summary(profile, summary_type, records=records)
    mark("_run_summary: deterministic summary", t0)
    finish("_run_summary", t_total)
    return out.model_dump()


def _handle_summarize(records, summary_type: str, run_llm: bool = True):
    """Small requests -> sync response. Large requests -> 202 + job_id."""
    _validate_mode(summary_type)

    def do_summary():
        return {"summary": _run_summary(records, summary_type, prefer_llm=run_llm)}

    if not run_llm:
        return {"summary": _run_summary(records, summary_type, prefer_llm=False)}

    if len(records) <= jobs.SMALL_MAX_ROWS:
        try:
            return do_summary()
        except StructuredOutputError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"LLM output could not be parsed: {exc}",
            ) from exc
        except Exception as exc:
            logger.exception("Summarize failed: inference server error")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference server unreachable or failed. "
                    f"Is the LLM provider running? (error: {exc})"
                ),
            ) from exc

    job_id = jobs.submit(_run_summary, records, summary_type, run_llm)
    return JSONResponse(
        {"job_id": job_id, "status": "pending"},
        status_code=202,
    )


def _run_db_summary(connection, statement, summary_type: str, run_llm: bool = True) -> dict:
    from .database import query as run_db_query

    result = run_db_query(connection, statement)
    return _run_summary(result["records"], summary_type, prefer_llm=run_llm)


@app.post("/api/v1/summarize")
def summarize(request: DataRequest):
    return _handle_summarize(request.data, request.summary_type, request.run_llm)


@app.post("/api/v1/summarize/json")
def summarize_json(request: DataRequest):
    return _handle_summarize(request.data, request.summary_type, request.run_llm)


@app.post("/api/v1/summarize/csv")
def summarize_csv(request: CsvSummaryRequest):
    from .adapters.csv_adapter import convert_csv

    try:
        records = convert_csv(request.data)["records"]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _handle_summarize(records, request.summary_type, request.run_llm)


@app.post("/api/v1/summarize/database")
def summarize_database(request: DatabaseSummaryRequest):
    """Runs in the background: row count is unknown until the query."""
    _validate_mode(request.summary_type)
    connection = (
        request.connection.model_dump() if request.connection is not None else None
    )
    statement = request.statement.model_dump(exclude_none=True)
    job_id = jobs.submit(
        _run_db_summary, connection, statement, request.summary_type, request.run_llm
    )
    return JSONResponse(
        {"job_id": job_id, "status": "pending"},
        status_code=202,
    )


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


@app.post("/api/v1/quality")
def quality(request: DataRequest):
    return check_quality(request.data)


def _build_data_report(records, source_type: str, summary_type: str, run_llm: bool, sanitize: bool) -> dict:
    """Run ONLY the tabular/structured modules - never the document pipeline."""
    from .analytics import compute_analytics
    from .pipeline import process as run_pipeline
    from .profile import build_data_profile
    from .profiling.profiler import profile_dataset
    from .profiling.schema_detector import detect_schema
    from .sampling.representative import build_context
    from .security.permissions import access as role_access
    from .security.sanitizer import sanitize as sanitize_records

    report: dict = {"endpoint": "full", "errors": {}}

    def section(key, fn):
        try:
            report[key] = fn()
        except Exception as exc:
            logger.exception("full-test section %r failed", key)
            report["errors"][key] = str(exc)
        return report.get(key)

    if source_type == "csv":
        from .adapters.csv_adapter import convert_csv

        records = section("conversion", lambda: convert_csv(records)["records"]) or []
    elif isinstance(records, dict):
        records = [records]

    report["meta"] = {
        "source_type": source_type,
        "summary_type": summary_type,
        "record_count": len(records),
    }

    section("modes", list_modes)
    section("schema", lambda: detect_schema(records))
    section("quality", lambda: check_quality(records))
    section("profile", lambda: profile_dataset(records))
    section("analytics", lambda: compute_analytics(records))
    section("sample", lambda: build_context(records, n=5))

    def pipeline_section():
        result = run_pipeline(records)
        return {
            "tier": result.get("tier"),
            "row_count": result.get("row_count"),
            "summary_computed": result.get("summary") is not None,
        }

    section("pipeline", pipeline_section)
    section("provider", lambda: getattr(llm_service.provider, "name", "unknown"))
    from .llm.prompt_builder import build_prompt

    section("prompt", lambda: build_prompt(records, mode=summary_type))

    if sanitize:
        def sanitize_section():
            out = sanitize_records(records, policy="mask")
            roles = {}
            try:
                roles["analyst"] = {"mode": role_access(records, "analyst")["mode"]}
                admin = role_access(records, "admin")
                roles["admin"] = {
                    "mode": admin["mode"],
                    "dropped_columns": admin["removed_columns"],
                }
            except Exception as exc:
                roles = {"note": f"role demo unavailable: {exc}"}
            return {
                "findings": [
                    f.model_dump() if hasattr(f, "model_dump") else f
                    for f in out["findings"]
                ],
                "masked_first_row": out["sanitized"][0] if out["sanitized"] else None,
                "roles": roles,
            }

        section("sanitize", sanitize_section)

    if run_llm:
        def llm_summary_section():
            try:
                return llm_service.generate_summary_from_profile(
                    build_data_profile(records), summary_type, auto_fallback=True
                ).model_dump()
            except Exception as exc:
                report["errors"]["summary"] = (
                    f"LLM unavailable; used deterministic fallback: {exc}"
                )
                return _run_summary(records, summary_type, prefer_llm=False)

        section("summary", llm_summary_section)
    else:
        section(
            "summary",
            lambda: _run_summary(records, summary_type, prefer_llm=False),
        )

    return report


def _run_document_auto(content: str, source_format, request, detection: dict) -> dict:
    from .llm.document_service import summarize_document as run_doc_summary
    from .timing import mark, start

    mode = request.summary_type if request.summary_type in document_modes() else "general"
    t0 = start()
    result = run_doc_summary(
        content,
        source_type=source_format,
        mode=mode,
        run_llm=request.run_llm,
    )
    mark(f"_run_document_auto: document summary (mode={mode}, run_llm={request.run_llm})", t0)
    return {
        "input_type": "document",
        "document_type": result.document_understanding.document_type or "document",
        "detection": detection,
        "pipeline": "document",
        "mode": mode,
        "run_llm": request.run_llm,
        "summary": result.model_dump(),
    }


def _coerce_records(kind: str, content):
    if kind == "tabular":
        return content  # raw text; converted inside _build_data_report
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


class AutoSummaryRequest(BaseModel):
    data: str | list | dict | None = None
    data_b64: str | None = None
    filename: str | None = None
    source_type: str | None = None
    summary_type: str = "system"
    run_llm: bool = True
    sanitize: bool = True
    connection: ConnectionConfig | None = None
    statement: QueryStatement | None = None


@app.post("/api/v1/summarize/auto")
def summarize_auto(request: AutoSummaryRequest):
    """
    Universal entry point: detect FIRST, run ONLY the matching pipeline and
    return ONLY the summary of the input.

      * PDF magic / pdf / markdown / txt / html / prose  -> document summary
      * JSON list / single object                        -> data summary
      * CSV / TSV / delimited table                      -> data summary

    The response is always {"input_type", "detection", "pipeline", "summary"}
    plus a couple of small fields - no diagnostic sections leak through.

    `data_b64` accepts base64 file bytes (used by the UI for PDFs and by
    Postman users); `data` accepts raw text or already-parsed data.
    """
    import base64

    from .router import classify as run_classifier
    from .timing import finish, mark, start

    t_total = start()

    # ---- database branch (mirror /api/v1/summarize/database) ----
    if request.connection is not None and request.statement is not None:
        from .database import QueryRejected, query as run_db_query

        connection = request.connection.model_dump()
        statement = request.statement.model_dump(exclude_none=True)
        t0 = start()
        try:
            result = run_db_query(connection, statement)
        except QueryRejected as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Database query failed: connection error")
            raise HTTPException(
                status_code=503,
                detail=f"Could not reach the database. (error: {exc})",
            ) from exc
        mark("summarize/auto: database query", t0)
        records = result["records"]
        mode = request.summary_type if request.summary_type in list_modes() else "system"
        summary = _run_summary(records, mode, prefer_llm=request.run_llm)
        finish("summarize/auto", t_total)
        return {
            "input_type": "structured",
            "detection": {
                "kind": "structured",
                "subtype": "database",
                "confidence": 1.0,
                "reason": "database query results",
            },
            "pipeline": "data",
            "mode": mode,
            "record_count": len(records),
            "run_llm": request.run_llm,
            "summary": summary,
        }

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
            mode = request.summary_type if request.summary_type in list_modes() else "system"
            summary = _run_summary(records, mode, prefer_llm=request.run_llm)
            finish("summarize/auto", t_total)
            return {
                "input_type": "tabular",
                "detection": detection,
                "pipeline": "data",
                "mode": mode,
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
            mode = request.summary_type if request.summary_type in list_modes() else "system"
            summary = build_report_summary(
                parsed_content,
                mode=mode,
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
                "mode": mode,
                "run_llm": request.run_llm,
                "summary": summary,
            }

        t0 = start()
        records = _coerce_records(kind, content)
        if kind == "tabular":
            from .adapters.csv_adapter import convert_csv

            try:
                records = convert_csv(records)["records"]
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        mark("summarize/auto: coerce -> records", t0)
        mode = request.summary_type if request.summary_type in list_modes() else "system"
        summary = _run_summary(records, mode, prefer_llm=request.run_llm)
        finish("summarize/auto", t_total)
        return {
            "input_type": kind,
            "detection": detection,
            "pipeline": "data",
            "mode": mode,
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


@app.post("/api/v1/full")
def full_test(request: FullTestRequest):
    """
    One-shot diagnostic: runs every feature on the same data and returns a
    single report. Each section is isolated - a failing section lands in
    `errors` instead of killing the whole response.

    Set run_llm=false for a fast deterministic report; run_llm=true waits
    for the inference server (slow on CPU).
    """
    report = _build_data_report(
        request.data,
        source_type=request.source_type,
        summary_type=request.summary_type,
        run_llm=request.run_llm,
        sanitize=request.sanitize,
    )

    if request.statement is not None:
        def database_section():
            from .database import query as run_db_query

            connection = (
                request.connection.model_dump()
                if request.connection is not None
                else None
            )
            result = run_db_query(
                connection, request.statement.model_dump(exclude_none=True)
            )
            return {
                "source_type": result["source_type"],
                "count": result["count"],
                "records_preview": result["records"][:3],
            }

        try:
            report["database"] = database_section()
        except Exception as exc:
            logger.exception("database section failed")
            report["errors"]["database"] = str(exc)

    return report


@app.post("/api/v1/query")
def query(request: QueryRequest):
    from .database import QueryRejected, query as run_query

    connection = (
        request.connection.model_dump() if request.connection is not None else None
    )
    try:
        return run_query(connection, request.statement.model_dump(exclude_none=True))
    except QueryRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Database query failed: connection error")
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the database. (error: {exc})",
        ) from exc


MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@app.post("/api/v1/upload")
def upload_summarize(
    file: UploadFile = File(...),
    summary_type: str = Form("system"),
    run_llm: bool = Form(False),
):
    """
    Upload a CSV / TSV / JSON / JSONL / XLSX / XLS file and summarize it in
    one call.

    Delimiters (comma, tab, semicolon) are detected automatically. Numeric
    fields are coerced from text. Defaults to run_llm=false for an instant,
    fully deterministic report; set run_llm=true to add the LLM narrative.
    """
    from .adapters.file_adapter import load_records_from_text
    from .timing import finish, mark, start

    t_total = start()
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 20 MB limit")
    mark("upload: read file bytes", t_total)

    name = (file.filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        from .adapters.excel_adapter import convert_excel

        t0 = start()
        try:
            parsed = convert_excel(content, source_type="xlsx" if name.endswith(".xlsx") else "xls")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Excel parsing failed: {exc}") from exc
        mark("upload: excel convert", t0)
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = content.decode("latin-1")
            except UnicodeDecodeError as exc:
                raise HTTPException(status_code=400, detail="Unsupported file encoding") from exc

        t0 = start()
        try:
            parsed = load_records_from_text(file.filename or "", text)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        mark("upload: parse records from file", t0)

    _validate_mode(summary_type)
    summary = _run_summary(parsed["records"], summary_type, prefer_llm=run_llm)
    finish("upload", t_total)

    return {
        "source_type": parsed["source_type"],
        "file_name": file.filename,
        "record_count": len(parsed["records"]),
        "summary": summary,
    }


@app.post("/api/v1/classify")
def classify_input(request: ClassifyRequest):
    """Tell the caller which pipeline an input belongs to (never the LLM)."""
    from .router import classify as run_classifier

    result = run_classifier(request.data, request.source_type)
    result["known_modes"] = document_modes()
    return result


@app.post("/api/v1/summarize/document")
def summarize_document_endpoint(request: DocumentSummaryRequest):
    """
    Route documents here (auto-detected or classifier-confirmed): text
    extraction, structure detection, deterministic metrics, then the LLM
    writes the meaning-bearing summary. run_llm=false returns a fast
    deterministic document digest.
    """
    from .llm.document_service import summarize_document as run_doc_summary

    from app.schemas.document_summary import document_modes as _doc_modes

    mode = request.summary_type
    if mode not in _doc_modes():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown document summary_type: {mode}. "
                f"Available: {_doc_modes()}"
            ),
        )

    source_format = (
        None if request.source_type in ("auto", "document") else request.source_type
    )
    result = run_doc_summary(
        request.content,
        source_type=source_format,
        mode=mode,
        run_llm=request.run_llm,
    )

    return {
        "input_type": "document",
        "document_type": result.document_understanding.document_type or "document",
        "mode": mode,
        "summary": result.model_dump(),
    }