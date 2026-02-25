from __future__ import annotations

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import Settings, load_settings
from app.database import DatabaseClient
from app.metrics import (
    ERRS,
    HTTP_INFLIGHT,
    HTTP_LAT,
    HTTP_REQS,
    HTTP_REQ_SIZE,
    HTTP_RESP_SIZE,
    LAT,
    REQS,
    STORAGE_BYTES,
    STORAGE_DEDUP,
    STORAGE_DOCS,
    STORAGE_UPLOAD_BYTES,
)
from app.models import DocumentMetadata, DocumentSearchRequest, DocumentSearchResponse, PatchExtraRequest, PatchExtraByIdRequest, StoreDocumentResponse
from app.observability import TraceContextFilter, setup_json_logging
from app.storage import StorageBackend, create_storage_backend

logger = logging.getLogger("storage")

async def _run_sync(fn, *args, **kwargs):
    return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))


class AppState:
    settings: Settings
    config_error: str | None = None
    db: DatabaseClient | None = None
    storage: StorageBackend | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # config + logging
    try:
        state.settings = load_settings()
    except Exception as e:
        state.config_error = str(e)
        setup_json_logging("INFO")
        logging.getLogger().addFilter(TraceContextFilter())
        logger.error("config_error", extra={"error": state.config_error})
        yield
        return

    setup_json_logging(state.settings.log_level)
    logging.getLogger().addFilter(TraceContextFilter())

    # Initialize database
    try:
        state.db = DatabaseClient(
            state.settings.db_url,
            min_conn=state.settings.db_pool_min,
            max_conn=state.settings.db_pool_max,
        )
        state.db.ensure_schema()
        logger.info("database_initialized")
    except Exception as e:
        logger.error("database_init_error", extra={"error": str(e)})
        state.db = None

    # Initialize storage backend
    try:
        state.storage = create_storage_backend(state.settings)
        logger.info("storage_backend_initialized", extra={"backend": state.settings.storage_backend})
    except Exception as e:
        logger.error("storage_init_error", extra={"error": str(e)})
        state.storage = None

    yield

    # Cleanup
    if state.db:
        state.db.close()


app = FastAPI(title="Document Storage", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors."""
    if request.url.path == "/v1/documents/by-id/extra":
        logger.error("by_id_extra_validation_error", extra={"errors": exc.errors(), "body": await request.body()})
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.middleware("http")
async def access_log(request: Request, call_next):
    # Prometheus HTTP metrics (avoid self-scrape noise)
    path = request.url.path
    if path not in ("/v1/metrics", "/metrics"):
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
        start_perf = time.perf_counter()
    else:
        method = request.method
        route = request.url.path
        req_size = None
        start_perf = None

    start = time.time()
    response: Response | None = None
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = (time.time() - start) * 1000.0
        status = response.status_code if response is not None else 500

        # Prometheus HTTP metrics
        if start_perf is not None:
            dur_s = max(0.0, time.perf_counter() - start_perf)
            HTTP_REQS.labels(method=method, route=route, status=str(status)).inc()
            HTTP_LAT.labels(method=method, route=route, status=str(status)).observe(dur_s)
            if req_size is not None:
                HTTP_REQ_SIZE.labels(method=method, route=route).observe(req_size)
            resp_size = None
            try:
                resp_size = int(response.headers.get("content-length")) if response is not None and response.headers.get("content-length") else None
            except Exception:
                resp_size = None
            if resp_size is not None:
                HTTP_RESP_SIZE.labels(method=method, route=route, status=str(status)).observe(resp_size)
            HTTP_INFLIGHT.labels(method=method, route=route).dec()

        # Don't log DELETE requests that return 404 - documents may already be deleted
        if request.method == "DELETE" and status == 404:
            logger.debug(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query,
                    "duration_ms": dur_ms,
                    "status": status,
                },
            )
        else:
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query,
                    "duration_ms": dur_ms,
                    "status": status,
                },
            )


@app.get("/v1/healthz")
async def healthz():
    return {"ok": True}


@app.get("/v1/readyz")
async def readyz(response: Response):
    if state.config_error:
        response.status_code = 503
        return {"ready": False, "config_error": state.config_error}

    db_ok = False
    storage_ok = False
    if state.db:
        db_ok = bool(await _run_sync(state.db.health))
    if state.storage:
        storage_ok = bool(await _run_sync(state.storage.health))
    ready = db_ok and storage_ok

    if not ready:
        response.status_code = 503
    return {"ready": ready, "db": db_ok, "storage": storage_ok}


@app.get("/v1/version")
async def version():
    if getattr(state, "settings", None) is None:
        return {"service": {"name": "document-storage"}, "config_error": state.config_error}
    return {"service": {"name": state.settings.service_name}, "config": state.settings.safe_summary()}


@app.get("/v1/metrics")
async def metrics():
    # Refresh "capacity" gauges from DB on scrape (cheap enough for small dev DB)
    if state.db is not None:
        try:
            stats = await _run_sync(state.db.get_usage_stats)
            STORAGE_DOCS.set(int(stats.get("docs") or 0))
            STORAGE_BYTES.set(int(stats.get("bytes") or 0))
        except Exception:
            # best-effort: never fail scrape
            pass
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/documents/store", response_model=StoreDocumentResponse)
async def store_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    title: str | None = Form(None),
    uri: str | None = Form(None),
    source: str | None = Form(None),
    lang: str | None = Form(None),
    tags: str | None = Form(None),  # comma-separated
    tenant_id: str | None = Form(None),
    project_id: str | None = Form(None),
    acl: str | None = Form(None),  # comma-separated
    metadata: str | None = Form(None),  # JSON string for extra fields
):
    """Store document file and metadata."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/store", status="503").inc()
        return StoreDocumentResponse(ok=False, doc_id=doc_id, error="config_error")

    if not state.db or not state.storage:
        REQS.labels(endpoint="/v1/documents/store", status="503").inc()
        return StoreDocumentResponse(ok=False, doc_id=doc_id, error="service_unavailable")

    try:
        with LAT.labels(stage="read_file").time():
            content = await file.read()

        # Validate file size
        max_size_bytes = state.settings.max_file_size_mb * 1024 * 1024
        if len(content) > max_size_bytes:
            REQS.labels(endpoint="/v1/documents/store", status="413").inc()
            ERRS.labels(stage="store", kind="file_too_large").inc()
            return StoreDocumentResponse(
                ok=False, doc_id=doc_id, error=f"File size exceeds {state.settings.max_file_size_mb}MB"
            )

        # Validate content type
        content_type = file.content_type or "application/octet-stream"
        if content_type not in state.settings.allowed_content_types_list:
            logger.warning("content_type_not_allowed", extra={"content_type": content_type, "doc_id": doc_id})

        # Parse metadata
        metadata_dict: dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("invalid_metadata_json", extra={"doc_id": doc_id})

        # Build full metadata
        full_metadata = {
            "title": title,
            "uri": uri,
            "source": source,
            "lang": lang,
            "tags": [t.strip() for t in tags.split(",")] if tags else [],
            "tenant_id": tenant_id,
            "project_id": project_id,
            "acl": [a.strip() for a in acl.split(",")] if acl else [],
            **metadata_dict,
        }

        # Store file
        # Storage backends expect a file-like object with .read()
        with LAT.labels(stage="store_file").time():
            storage_id, stored_size, content_hash = await _run_sync(
                state.storage.store, doc_id, io.BytesIO(content), content_type
            )

        # Deduplication (exact, bytes-level): storage_id is deterministic (sha256-based) for both local and s3 backends.
        duplicate_of: str | None = None
        try:
            duplicate_of = await _run_sync(state.db.find_any_doc_id_by_storage_id, storage_id=storage_id, exclude_doc_id=doc_id)
        except Exception as e:
            # best-effort: never fail uploads due to dedup bookkeeping
            logger.warning("dedup_lookup_failed", extra={"doc_id": doc_id, "storage_id": storage_id, "error": str(e)})
            duplicate_of = None

        # Store metadata
        with LAT.labels(stage="store_metadata").time():
            existing_extra: dict[str, Any] = {}
            if isinstance(metadata_dict, dict):
                v = metadata_dict.get("extra")
                if isinstance(v, dict):
                    existing_extra = v

            dedup_extra = {
                "is_duplicate": bool(duplicate_of),
                "duplicate_of": duplicate_of,
                "storage_id": storage_id,
                "method": "storage_id_sha256",
            }
            doc_meta = await _run_sync(
                state.db.store_metadata,
                doc_id=doc_id,
                storage_id=storage_id,
                metadata={**full_metadata, "extra": {**existing_extra, "dedup": dedup_extra}},
                content_type=content_type,
                size=stored_size,
            )

        REQS.labels(endpoint="/v1/documents/store", status="200").inc()
        # Dedup metrics (based on DB lookup result)
        outcome = "duplicate" if bool(duplicate_of) else "unique"
        STORAGE_DEDUP.labels(outcome=outcome).inc()
        STORAGE_UPLOAD_BYTES.labels(outcome=outcome).inc(len(content))
        return StoreDocumentResponse(
            ok=True,
            storage_id=storage_id,
            doc_id=doc_id,
            size=len(content),
            content_type=content_type,
            stored_at=doc_meta.stored_at or datetime.utcnow(),
            duplicate=bool(duplicate_of),
            duplicate_of=duplicate_of,
        )

    except Exception as e:
        logger.error("store_document_error", extra={"doc_id": doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/store", status="500").inc()
        ERRS.labels(stage="store", kind="exception").inc()
        return StoreDocumentResponse(ok=False, doc_id=doc_id, error=str(e))


# NOTE: doc_id can contain "/" (e.g. repo:path/to/file). FastAPI path params won't match encoded slashes reliably.
# Provide "safe" endpoints under /by-id/ using query params (must be defined BEFORE /{doc_id} routes).
@app.get("/v1/documents/by-id")
async def get_document_by_id(doc_id: str):
    """Retrieve document file by doc_id (supports slashy doc_ids via query param)."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/by-id", status="503").inc()
        raise HTTPException(status_code=503, detail="config_error")

    if not state.db or not state.storage:
        REQS.labels(endpoint="/v1/documents/by-id", status="503").inc()
        raise HTTPException(status_code=503, detail="service_unavailable")

    try:
        logger.info("get_document_by_id_request", extra={"doc_id": doc_id, "doc_id_repr": repr(doc_id), "endpoint": "by-id"})
        meta = await _run_sync(state.db.get_metadata, doc_id)
        if not meta:
            logger.warning("get_document_by_id_not_found", extra={"doc_id": doc_id, "endpoint": "by-id"})
            REQS.labels(endpoint="/v1/documents/by-id", status="404").inc()
            raise HTTPException(status_code=404, detail="Document not found")

        if not meta.storage_id:
            REQS.labels(endpoint="/v1/documents/by-id", status="404").inc()
            raise HTTPException(status_code=404, detail="Storage ID not found")

        content = await _run_sync(state.storage.retrieve, meta.storage_id)
        if not content:
            REQS.labels(endpoint="/v1/documents/by-id", status="404").inc()
            raise HTTPException(status_code=404, detail="File not found in storage")

        headers = {}
        if meta.content_type:
            headers["Content-Type"] = meta.content_type
        REQS.labels(endpoint="/v1/documents/by-id", status="200").inc()
        return Response(content=content, headers=headers, media_type=meta.content_type or "application/octet-stream")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_document_by_id_error", extra={"doc_id": doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/by-id", status="500").inc()
        ERRS.labels(stage="get", kind="exception").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents/{doc_id}")
async def get_document(doc_id: str):
    """Retrieve document file by doc_id."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/{doc_id}", status="503").inc()
        raise HTTPException(status_code=503, detail="config_error")

    if not state.db or not state.storage:
        REQS.labels(endpoint="/v1/documents/{doc_id}", status="503").inc()
        raise HTTPException(status_code=503, detail="service_unavailable")

    try:
        # Get metadata
        meta = await _run_sync(state.db.get_metadata, doc_id)
        if not meta:
            REQS.labels(endpoint="/v1/documents/{doc_id}", status="404").inc()
            raise HTTPException(status_code=404, detail="Document not found")

        # Get file
        if not meta.storage_id:
            REQS.labels(endpoint="/v1/documents/{doc_id}", status="404").inc()
            raise HTTPException(status_code=404, detail="Storage ID not found")

        content = await _run_sync(state.storage.retrieve, meta.storage_id)
        if not content:
            REQS.labels(endpoint="/v1/documents/{doc_id}", status="404").inc()
            raise HTTPException(status_code=404, detail="File not found in storage")

        # Return file with proper headers
        headers = {}
        if meta.content_type:
            headers["Content-Type"] = meta.content_type
        if meta.title:
            filename = meta.title
        else:
            filename = doc_id

        REQS.labels(endpoint="/v1/documents/{doc_id}", status="200").inc()
        return Response(
            content=content,
            headers=headers,
            media_type=meta.content_type or "application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_document_error", extra={"doc_id": doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/{doc_id}", status="500").inc()
        ERRS.labels(stage="get", kind="exception").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/documents/by-id")
async def delete_document_by_id(doc_id: str):
    """Delete document and its metadata (supports slashy doc_ids via query param)."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/by-id", status="503").inc()
        raise HTTPException(status_code=503, detail="config_error")

    if not state.db or not state.storage:
        REQS.labels(endpoint="/v1/documents/by-id", status="503").inc()
        raise HTTPException(status_code=503, detail="service_unavailable")

    try:
        meta = await _run_sync(state.db.get_metadata, doc_id)
        if not meta:
            # Don't log 404 for DELETE requests - documents may already be deleted
            logger.debug("delete_document_by_id_not_found", extra={"doc_id": doc_id})
            REQS.labels(endpoint="/v1/documents/by-id", status="404").inc()
            raise HTTPException(status_code=404, detail="Document not found")

        storage_id = meta.storage_id

        # Delete metadata first (so we can safely decide whether to delete the underlying blob).
        deleted = await _run_sync(state.db.delete_metadata, doc_id)

        file_deleted = False
        remaining_refs = 0
        if deleted and storage_id:
            try:
                remaining_refs = await _run_sync(state.db.count_docs_by_storage_id, storage_id=storage_id)
                if remaining_refs == 0:
                    file_deleted = bool(await _run_sync(state.storage.delete, storage_id))
            except Exception as e:
                logger.warning("delete_blob_refcount_failed", extra={"doc_id": doc_id, "storage_id": storage_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/by-id", status="200").inc()
        return {"ok": True, "deleted": deleted, "storage_id": storage_id, "file_deleted": file_deleted, "remaining_refs": remaining_refs}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_document_by_id_error", extra={"doc_id": doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/by-id", status="500").inc()
        ERRS.labels(stage="delete", kind="exception").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete document and its metadata."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/{doc_id}", status="503").inc()
        raise HTTPException(status_code=503, detail="config_error")

    if not state.db or not state.storage:
        REQS.labels(endpoint="/v1/documents/{doc_id}", status="503").inc()
        raise HTTPException(status_code=503, detail="service_unavailable")

    try:
        # Get metadata to find storage_id
        meta = await _run_sync(state.db.get_metadata, doc_id)
        if not meta:
            REQS.labels(endpoint="/v1/documents/{doc_id}", status="404").inc()
            raise HTTPException(status_code=404, detail="Document not found")

        storage_id = meta.storage_id

        # Delete metadata first (so we can safely decide whether to delete the underlying blob).
        deleted = await _run_sync(state.db.delete_metadata, doc_id)

        file_deleted = False
        remaining_refs = 0
        if deleted and storage_id:
            try:
                remaining_refs = await _run_sync(state.db.count_docs_by_storage_id, storage_id=storage_id)
                if remaining_refs == 0:
                    file_deleted = bool(await _run_sync(state.storage.delete, storage_id))
            except Exception as e:
                logger.warning("delete_blob_refcount_failed", extra={"doc_id": doc_id, "storage_id": storage_id, "error": str(e)})

        REQS.labels(endpoint="/v1/documents/{doc_id}", status="200").inc()
        return {"ok": True, "deleted": deleted, "storage_id": storage_id, "file_deleted": file_deleted, "remaining_refs": remaining_refs}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_document_error", extra={"doc_id": doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/{doc_id}", status="500").inc()
        ERRS.labels(stage="delete", kind="exception").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents/by-id/metadata", response_model=DocumentMetadata)
async def get_metadata_by_id(doc_id: str):
    """Get document metadata only (supports slashy doc_ids via query param)."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/by-id/metadata", status="503").inc()
        raise HTTPException(status_code=503, detail="config_error")

    if not state.db:
        REQS.labels(endpoint="/v1/documents/by-id/metadata", status="503").inc()
        raise HTTPException(status_code=503, detail="service_unavailable")

    try:
        logger.info("get_metadata_by_id_request", extra={"doc_id": doc_id, "doc_id_repr": repr(doc_id), "doc_id_len": len(doc_id)})
        # Try to find the document
        meta = await _run_sync(state.db.get_metadata, doc_id)
        if not meta:
            # Log what we searched for and try a few variations
            logger.warning("get_metadata_by_id_not_found", extra={"doc_id": doc_id, "doc_id_bytes": doc_id.encode('utf-8').hex()})
            # Try URL-decoded version if it contains %XX
            import urllib.parse
            decoded = urllib.parse.unquote(doc_id)
            if decoded != doc_id:
                logger.info("get_metadata_by_id_trying_decoded", extra={"decoded": decoded})
                meta = await _run_sync(state.db.get_metadata, decoded)
            if not meta:
                REQS.labels(endpoint="/v1/documents/by-id/metadata", status="404").inc()
                raise HTTPException(status_code=404, detail="Document not found")
        REQS.labels(endpoint="/v1/documents/by-id/metadata", status="200").inc()
        return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_metadata_by_id_error", extra={"doc_id": doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/by-id/metadata", status="500").inc()
        ERRS.labels(stage="get_metadata", kind="exception").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/documents/{doc_id}/metadata", response_model=DocumentMetadata)
async def get_metadata(doc_id: str):
    """Get document metadata only."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/{doc_id}/metadata", status="503").inc()
        raise HTTPException(status_code=503, detail="config_error")

    if not state.db:
        REQS.labels(endpoint="/v1/documents/{doc_id}/metadata", status="503").inc()
        raise HTTPException(status_code=503, detail="service_unavailable")

    try:
        meta = await _run_sync(state.db.get_metadata, doc_id)
        if not meta:
            REQS.labels(endpoint="/v1/documents/{doc_id}/metadata", status="404").inc()
            raise HTTPException(status_code=404, detail="Document not found")

        REQS.labels(endpoint="/v1/documents/{doc_id}/metadata", status="200").inc()
        return meta

    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_metadata_error", extra={"doc_id": doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/{doc_id}/metadata", status="500").inc()
        ERRS.labels(stage="get_metadata", kind="exception").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/documents/by-id/extra", response_model=DocumentMetadata)
async def patch_extra_by_id(req: PatchExtraByIdRequest):
    """Merge-patch `extra` JSON for a document (supports slashy doc_ids via body param)."""
    logger.info("patch_extra_by_id_called", extra={"req_doc_id": getattr(req, 'doc_id', 'MISSING'), "req_type": type(req).__name__})
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/by-id/extra", status="503").inc()
        raise HTTPException(status_code=503, detail="config_error")

    if not state.db:
        REQS.labels(endpoint="/v1/documents/by-id/extra", status="503").inc()
        raise HTTPException(status_code=503, detail="service_unavailable")

    try:
        logger.info("patch_extra_by_id_request", extra={"doc_id": req.doc_id, "doc_id_repr": repr(req.doc_id), "patch_keys": list(req.patch.keys())})
        with LAT.labels(stage="patch_extra").time():
            meta = await _run_sync(state.db.patch_extra, doc_id=req.doc_id, patch=req.patch)
        if not meta:
            logger.warning("patch_extra_by_id_not_found", extra={"doc_id": req.doc_id})
            REQS.labels(endpoint="/v1/documents/by-id/extra", status="404").inc()
            raise HTTPException(status_code=404, detail="Document not found")
        REQS.labels(endpoint="/v1/documents/by-id/extra", status="200").inc()
        return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error("patch_extra_by_id_error", extra={"doc_id": req.doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/by-id/extra", status="500").inc()
        ERRS.labels(stage="patch_extra", kind="exception").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/documents/{doc_id}/extra", response_model=DocumentMetadata)
async def patch_extra(doc_id: str, req: PatchExtraRequest):
    """Merge-patch `extra` JSON for a document."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/{doc_id}/extra", status="503").inc()
        raise HTTPException(status_code=503, detail="config_error")

    if not state.db:
        REQS.labels(endpoint="/v1/documents/{doc_id}/extra", status="503").inc()
        raise HTTPException(status_code=503, detail="service_unavailable")

    try:
        with LAT.labels(stage="patch_extra").time():
            meta = await _run_sync(state.db.patch_extra, doc_id=doc_id, patch=req.patch)
        if not meta:
            REQS.labels(endpoint="/v1/documents/{doc_id}/extra", status="404").inc()
            raise HTTPException(status_code=404, detail="Document not found")
        REQS.labels(endpoint="/v1/documents/{doc_id}/extra", status="200").inc()
        return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error("patch_extra_error", extra={"doc_id": doc_id, "error": str(e)})
        REQS.labels(endpoint="/v1/documents/{doc_id}/extra", status="500").inc()
        ERRS.labels(stage="patch_extra", kind="exception").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/documents/search", response_model=DocumentSearchResponse)
async def search_documents(req: DocumentSearchRequest):
    """Search documents by metadata filters."""
    if state.config_error:
        REQS.labels(endpoint="/v1/documents/search", status="503").inc()
        return DocumentSearchResponse(ok=False, error="config_error")

    if not state.db:
        REQS.labels(endpoint="/v1/documents/search", status="503").inc()
        return DocumentSearchResponse(ok=False, error="service_unavailable")

    try:
        with LAT.labels(stage="search").time():
            docs, total = await _run_sync(state.db.search_metadata, req)

        REQS.labels(endpoint="/v1/documents/search", status="200").inc()
        return DocumentSearchResponse(ok=True, documents=docs, total=total)

    except Exception as e:
        logger.error("search_documents_error", extra={"error": str(e)})
        REQS.labels(endpoint="/v1/documents/search", status="500").inc()
        ERRS.labels(stage="search", kind="exception").inc()
        return DocumentSearchResponse(ok=False, error=str(e))
        REQS.labels(endpoint="/v1/documents/stats", status="200").inc()
        return stats
    except Exception as e:
        logger.error("documents_stats_error", extra={"error": str(e)})
        REQS.labels(endpoint="/v1/documents/stats", status="500").inc()
        ERRS.labels(stage="stats", kind="exception").inc()
        return {"ok": False, "error": str(e)}


@app.get("/v1/collections")
async def list_collections(tenant_id: str | None = None, limit: int = 1000):
    """List distinct project_id values (aka "collections") with counts."""
    if state.config_error:
        REQS.labels(endpoint="/v1/collections", status="503").inc()
        return {"ok": False, "error": "config_error"}

    if not state.db:
        REQS.labels(endpoint="/v1/collections", status="503").inc()
        return {"ok": False, "error": "service_unavailable", "collections": []}

    try:
        with LAT.labels(stage="collections").time():
            cols = await _run_sync(state.db.list_collections, tenant_id=tenant_id, limit=limit)
        REQS.labels(endpoint="/v1/collections", status="200").inc()
        return {"ok": True, "collections": cols}
    except Exception as e:
        logger.error("list_collections_error", extra={"error": str(e)})
        REQS.labels(endpoint="/v1/collections", status="500").inc()
        ERRS.labels(stage="collections", kind="exception").inc()
        return {"ok": False, "error": str(e), "collections": []}
