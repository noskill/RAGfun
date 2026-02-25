from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from app.chunking import chunk_by_strategy
from app.clients import LandingAIClient, RetrievalClient, StorageClient, VLMClient
from app.config import Settings, load_settings
from app.extraction import extract_text_non_vlm, normalize_to_pdf, pdf_to_page_pngs
from app.logging_setup import setup_json_logging
from app.models import ChunkMeta, Locator, ProcessRequest, ProcessResponse

logger = logging.getLogger("processor")

REQS = Counter("processor_requests_total", "Requests", ["endpoint", "status"])
LAT = Histogram("processor_latency_seconds", "Latency", ["stage"])

# Standard HTTP server metrics (shared names across services; Prometheus "job" label disambiguates)
HTTP_INFLIGHT = Gauge("http_server_inflight_requests", "In-flight HTTP requests", ["method", "route"])
HTTP_REQS = Counter("http_server_requests_total", "HTTP requests", ["method", "route", "status"])
HTTP_LAT = Histogram(
    "http_server_request_duration_seconds",
    "HTTP request duration (seconds)",
    ["method", "route", "status"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60),
)
HTTP_REQ_SIZE = Histogram(
    "http_server_request_size_bytes",
    "HTTP request size (bytes), from Content-Length when available",
    ["method", "route"],
    buckets=(0, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000),
)
HTTP_RESP_SIZE = Histogram(
    "http_server_response_size_bytes",
    "HTTP response size (bytes), from Content-Length when available",
    ["method", "route", "status"],
    buckets=(0, 200, 500, 1_000, 2_000, 5_000, 10_000, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000),
)

# Business-quality metrics
PROCESSOR_DEGRADED = Counter("processor_degraded_total", "Degradation events", ["kind"])
PROCESSOR_PARTIAL = Counter("processor_partial_total", "Partial processing results", ["endpoint"])
PROCESSOR_PATH = Counter("processor_path_total", "Processing path decisions", ["path"])
PROCESSOR_PAGES = Histogram("processor_pages", "Pages processed", buckets=(1, 2, 3, 5, 10, 20, 35, 50, 100))
PROCESSOR_CHUNKS = Histogram("processor_chunks", "Chunks produced", buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000))
PROCESSOR_EXTRACTED_CHARS = Histogram(
    "processor_extracted_chars",
    "Extracted characters",
    buckets=(0, 200, 1_000, 5_000, 20_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000),
)

# Pre-create common label series so Grafana panels show 0 instead of "No data" right after startup.
for _p in ("vlm", "landing_ai", "non_vlm", "skipped_duplicate"):
    PROCESSOR_PATH.labels(_p).inc(0)
PROCESSOR_PARTIAL.labels(endpoint="/v1/process").inc(0)

class AppState:
    settings: Settings | None = None
    config_error: str | None = None
    storage: StorageClient | None = None
    retrieval: RetrievalClient | None = None
    vlm: VLMClient | None = None
    landing: LandingAIClient | None = None


state = AppState()


def _landing_parse_to_pages(parse_json: dict, *, max_pages: int | None) -> list[str]:
    def _as_int(v: object) -> int | None:
        try:
            return int(v)  # type: ignore[arg-type]
        except Exception:
            return None

    # Preferred: splits (when split=page)
    page_map: dict[int, list[str]] = {}
    splits = parse_json.get("splits")
    if isinstance(splits, list) and splits:
        for sp in splits:
            if not isinstance(sp, dict):
                continue
            md = sp.get("markdown") or ""
            if not md:
                continue
            pages = sp.get("pages")
            if isinstance(pages, list) and pages:
                for p in pages:
                    pi = _as_int(p)
                    if pi is None or pi < 0:
                        continue
                    if max_pages is not None and pi >= max_pages:
                        continue
                    page_map.setdefault(pi, []).append(md)
        if page_map:
            max_idx = max(page_map)
            pages_text = ["" for _ in range(max_idx + 1)]
            for p, parts in page_map.items():
                pages_text[p] = "\n\n".join(parts)
            return pages_text

    # Fallback: chunks with grounding.page
    page_map = {}
    chunks = parse_json.get("chunks")
    if isinstance(chunks, list):
        for ch in chunks:
            if not isinstance(ch, dict):
                continue
            md = ch.get("markdown") or ""
            if not md:
                continue
            grounding = ch.get("grounding") if isinstance(ch.get("grounding"), dict) else {}
            pi = _as_int(grounding.get("page"))
            if pi is None or pi < 0:
                pi = 0
            if max_pages is not None and pi >= max_pages:
                continue
            page_map.setdefault(pi, []).append(md)
        if page_map:
            max_idx = max(page_map)
            pages_text = ["" for _ in range(max_idx + 1)]
            for p, parts in page_map.items():
                pages_text[p] = "\n\n".join(parts)
            return pages_text

    # Final fallback: single markdown blob
    md = parse_json.get("markdown") or ""
    if md:
        return [md]
    return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.settings = load_settings()
    except Exception as e:
        state.config_error = str(e)
        setup_json_logging("INFO")
        logger.error("config_error", extra={"extra": {"error": state.config_error}})
        yield
        return

    setup_json_logging(state.settings.log_level)
    state.storage = StorageClient(base_url=str(state.settings.storage_url), timeout_s=state.settings.storage_timeout_s)
    state.retrieval = RetrievalClient(base_url=str(state.settings.retrieval_url), timeout_s=state.settings.retrieval_timeout_s)
    if state.settings.vlm_provider == "landing_ai":
        state.landing = LandingAIClient(
            parse_url=str(state.settings.landing_parse_url),
            api_key=state.settings.landing_api_key.get_secret_value() if state.settings.landing_api_key else None,
            model=state.settings.landing_model,
            split=state.settings.landing_split,
            timeout_s=state.settings.landing_timeout_s,
        )
    else:
        state.vlm = VLMClient(
            base_url=str(state.settings.vlm_base_url),
            api_key=state.settings.vlm_api_key.get_secret_value() if state.settings.vlm_api_key else None,
            model=state.settings.vlm_model,
            timeout_s=state.settings.vlm_timeout_s,
        )
    yield


app = FastAPI(title="Doc Processor", version="0.1.0", lifespan=lifespan)

@app.middleware("http")
async def http_metrics(request: Request, call_next):
    # Avoid self-scrape noise
    path = request.url.path
    if path in ("/v1/metrics", "/metrics"):
        return await call_next(request)

    method = request.method
    route_obj = request.scope.get("route")
    route = getattr(route_obj, "path", None) or path

    def _cl(headers) -> int | None:
        try:
            v = headers.get("content-length")
            return int(v) if v is not None else None
        except Exception:
            return None

    req_size = _cl(request.headers)

    HTTP_INFLIGHT.labels(method=method, route=route).inc()
    start = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
        return response
    finally:
        dur_s = max(0.0, time.perf_counter() - start)
        status = response.status_code if response is not None else 500
        HTTP_REQS.labels(method=method, route=route, status=str(status)).inc()
        HTTP_LAT.labels(method=method, route=route, status=str(status)).observe(dur_s)
        if req_size is not None:
            HTTP_REQ_SIZE.labels(method=method, route=route).observe(req_size)
        resp_size = _cl(response.headers) if response is not None else None
        if resp_size is not None:
            HTTP_RESP_SIZE.labels(method=method, route=route, status=str(status)).observe(resp_size)
        HTTP_INFLIGHT.labels(method=method, route=route).dec()


@app.get("/v1/healthz")
async def healthz():
    return {"ok": True}


@app.get("/v1/readyz")
async def readyz(response: Response):
    if state.config_error:
        response.status_code = 503
        return {"ready": False, "config_error": state.config_error}
    ready = state.storage is not None and state.retrieval is not None and state.vlm is not None
    if not ready:
        response.status_code = 503
    return {"ready": ready}


@app.get("/v1/version")
async def version():
    if state.settings is None:
        return {"service": {"name": "doc-processor"}, "config_error": state.config_error}
    return {"service": {"name": state.settings.service_name}, "config": state.settings.safe_summary()}


@app.get("/v1/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    if state.config_error:
        REQS.labels(endpoint="/v1/process", status="503").inc()
        return ProcessResponse(ok=False, doc_id=req.document.doc_id, error="config_error", detail=state.config_error)
    assert state.settings is not None
    assert state.storage is not None
    assert state.retrieval is not None
    assert state.vlm is not None

    doc_id = req.document.doc_id

    degraded: list[str] = []
    partial = False

    # Fetch metadata + bytes from storage
    with LAT.labels("storage_meta").time():
        meta = await state.storage.get_metadata(doc_id=doc_id)
    if not meta:
        REQS.labels(endpoint="/v1/process", status="404").inc()
        return ProcessResponse(ok=False, doc_id=doc_id, error="not_found", detail="doc_id not found in storage")

    # Exact deduplication: keep note, but still index per doc_id (retrieval contract).
    try:
        dedup = ((meta.get("extra") or {}).get("dedup") or {}) if isinstance(meta.get("extra"), dict) else {}
        duplicate_of = dedup.get("duplicate_of") if isinstance(dedup, dict) else None
    except Exception:
        duplicate_of = None
    if duplicate_of:
        PROCESSOR_PATH.labels("duplicate_detected").inc()
        degraded.append("duplicate_detected")

    filename = meta.get("title") or meta.get("doc_id") or doc_id
    content_type = meta.get("content_type")

    with LAT.labels("storage_get").time():
        raw, header_ct = await state.storage.get_file(doc_id=doc_id)
    content_type = content_type or header_ct

    if not raw:
        REQS.labels(endpoint="/v1/process", status="400").inc()
        return ProcessResponse(ok=False, doc_id=doc_id, error="empty_file", detail="empty content from storage")

    # Decide path: PDF/DOC/DOCX -> VLM; XML -> parse; else -> utf-8 fallback.
    pdf_bytes, norm_ct = None, None
    try:
        with LAT.labels("normalize").time():
            pdf_bytes, norm_ct = normalize_to_pdf(raw, content_type, filename)
    except Exception as e:
        degraded.append("office_convert_failed")
        partial = True
        pdf_bytes = None
        logger.warning("normalize_to_pdf_failed", extra={"extra": {"doc_id": doc_id, "error": str(e)}})

    pages_text: list[str] = []
    pages = None

    if pdf_bytes is None:
        # Non-VLM fallback (XML/plain)
        with LAT.labels("extract_non_vlm").time():
            ed = extract_text_non_vlm(raw, content_type, filename)
        pages_text = ed.pages_text
        pages = len(pages_text)
        content_type = ed.content_type or content_type
        degraded.append("vlm_skipped")
        PROCESSOR_PATH.labels("non_vlm").inc()
    else:
        if state.settings.vlm_provider == "landing_ai":
            if state.landing is None:
                REQS.labels(endpoint="/v1/process", status="500").inc()
                return ProcessResponse(ok=False, doc_id=doc_id, error="landing_ai_unavailable", detail="LandingAI client not configured")

            with LAT.labels("landing_parse").time():
                parse_json = await state.landing.parse_pdf(pdf_bytes=pdf_bytes, filename=filename)
            pages_text = _landing_parse_to_pages(parse_json, max_pages=state.settings.max_pages)
            pages = len(pages_text)
            if pages == 0:
                REQS.labels(endpoint="/v1/process", status="400").inc()
                return ProcessResponse(ok=False, doc_id=doc_id, content_type=norm_ct, error="no_pages", detail="LandingAI returned no pages")

            if all(not (t or "").strip() for t in pages_text):
                REQS.labels(endpoint="/v1/process", status="400").inc()
                return ProcessResponse(
                    ok=False,
                    doc_id=doc_id,
                    content_type=norm_ct,
                    pages=pages,
                    error="empty_text",
                    detail="LandingAI returned empty text for all pages",
                    partial=partial,
                    degraded=degraded,
                )

            content_type = norm_ct
            PROCESSOR_PATH.labels("landing_ai").inc()
        else:
            # VLM path: render pages -> ask VLM per page.
            with LAT.labels("pdf_render").time():
                pngs = pdf_to_page_pngs(
                    pdf_bytes,
                    max_pages=state.settings.max_pages,
                    max_side_px=state.settings.max_image_side_px,
                )
            pages = len(pngs)
            if pages == 0:
                REQS.labels(endpoint="/v1/process", status="400").inc()
                return ProcessResponse(ok=False, doc_id=doc_id, content_type=norm_ct, error="no_pages", detail="no pages rendered")

            async def one(i: int, b: bytes) -> tuple[int, str]:
                assert state.vlm is not None
                t = await state.vlm.page_to_text(png_bytes=b)
                return (i, t)

            with LAT.labels("vlm").time():
                # bounded parallelism
                sem = asyncio.Semaphore(4)

                async def run_one(i: int, b: bytes):
                    async with sem:
                        return await one(i, b)

                results = await asyncio.gather(*(run_one(i, b) for i, b in enumerate(pngs)), return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    degraded.append("vlm_page_failed")
                    partial = True
                    continue
                i, t = r
                if t:
                    # 1-based page number in locator
                    while len(pages_text) < i + 1:
                        pages_text.append("")
                    pages_text[i] = t

            # Ensure list length
            if len(pages_text) < pages:
                pages_text.extend([""] * (pages - len(pages_text)))

            if all(not t.strip() for t in pages_text):
                REQS.labels(endpoint="/v1/process", status="400").inc()
                return ProcessResponse(
                    ok=False,
                    doc_id=doc_id,
                    content_type=norm_ct,
                    pages=pages,
                    error="empty_text",
                    detail="VLM returned empty text for all pages",
                    partial=partial,
                    degraded=degraded,
                )

            content_type = norm_ct
            PROCESSOR_PATH.labels("vlm").inc()

    # Build chunks (preserve page in locator when we have pages)
    chunks: list[ChunkMeta] = []
    global_idx = 0
    for pi, page_text in enumerate(pages_text):
        for part in chunk_by_strategy(
            page_text,
            strategy=state.settings.chunk_strategy,
            chunk_size=state.settings.chunk_size_chars,
            overlap=state.settings.chunk_overlap_chars,
        ):
            chunks.append(
                ChunkMeta(
                    chunk_id=f"{doc_id}:{global_idx}",
                    doc_id=doc_id,
                    chunk_index=global_idx,
                    text=part,
                    locator=Locator(page=pi + 1) if pages and pages > 1 else None,
                    title=req.document.title,
                    uri=req.document.uri,
                    lang=req.document.lang,
                    tags=req.document.tags,
                    acl=req.document.acl,
                    source=req.document.source,
                    tenant_id=req.document.tenant_id,
                    project_id=req.document.project_id,
                )
            )
            global_idx += 1

    if not chunks:
        extracted_chars = sum(len(t) for t in pages_text if t)
        REQS.labels(endpoint="/v1/process", status="200").inc()
        return ProcessResponse(
            ok=True,
            doc_id=doc_id,
            content_type=content_type,
            pages=pages,
            extracted_chars=extracted_chars,
            chunks=0,
            skipped=True,
            error="no_chunks",
            detail="no text extracted/chunked",
        )

    upsert_payload = {
        "mode": "chunks",
        "document": req.document.model_dump(exclude_none=True),
        "chunks": [c.model_dump(exclude_none=True) for c in chunks],
        "refresh": bool(req.refresh),
    }

    extracted_chars = sum(len(t) for t in pages_text if t)

    with LAT.labels("retrieval_upsert").time():
        retrieval_resp = await state.retrieval.index_upsert(payload=upsert_payload)

    retrieval_ok = True
    retrieval_partial = False
    if isinstance(retrieval_resp, dict):
        retrieval_ok = retrieval_resp.get("ok") is True
        retrieval_partial = bool(retrieval_resp.get("partial"))

    if not retrieval_ok or retrieval_partial:
        # Log retrieval failure details for debugging.
        try:
            logger.error(
                "retrieval_upsert_failed",
                extra={
                    "extra": {
                        "doc_id": doc_id,
                        "retrieval_ok": retrieval_ok,
                        "retrieval_partial": retrieval_partial,
                        "retrieval_error": retrieval_resp.get("error") if isinstance(retrieval_resp, dict) else None,
                        "retrieval_degraded": retrieval_resp.get("degraded") if isinstance(retrieval_resp, dict) else None,
                    }
                },
            )
        except Exception:
            logger.error("retrieval_upsert_failed", extra={"extra": {"doc_id": doc_id}})
        REQS.labels(endpoint="/v1/process", status="502").inc()
        return ProcessResponse(
            ok=False,
            doc_id=doc_id,
            content_type=content_type,
            pages=pages,
            extracted_chars=extracted_chars,
            chunks=len(chunks),
            retrieval=retrieval_resp,
            partial=bool(retrieval_partial),
            degraded=degraded,
            error="retrieval_upsert_failed",
            detail={"retrieval_ok": retrieval_ok, "retrieval_partial": retrieval_partial},
        )

    if partial:
        PROCESSOR_PARTIAL.labels(endpoint="/v1/process").inc()
    for k in degraded:
        PROCESSOR_DEGRADED.labels(str(k)).inc()
    if pages is not None:
        PROCESSOR_PAGES.observe(int(pages))
    PROCESSOR_CHUNKS.observe(len(chunks))
    PROCESSOR_EXTRACTED_CHARS.observe(int(extracted_chars))

    REQS.labels(endpoint="/v1/process", status="200").inc()
    return ProcessResponse(
        ok=True,
        doc_id=doc_id,
        content_type=content_type,
        pages=pages,
        extracted_chars=extracted_chars,
        chunks=len(chunks),
        retrieval=retrieval_resp,
        partial=partial,
        degraded=degraded,
    )
