from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import Settings, load_settings
from app.clients.embeddings import EmbeddingsClient
from app.clients.opensearch import OpenSearchClient
from app.clients.qdrant import QdrantFacade
from app.clients.rerank import LocalReranker, Reranker, RerankClient
from app.indexing_logic import delete as delete_logic
from app.indexing_logic import delete_batch as delete_batch_logic
from app.indexing_logic import upsert as upsert_logic
from app.metrics import HTTP_INFLIGHT, HTTP_LAT, HTTP_REQS, HTTP_REQ_SIZE, HTTP_RESP_SIZE, RAG_PARTIAL, RAG_RERANK, REQS
from app.models import (
    IndexDeleteBatchRequest,
    IndexDeleteRequest,
    IndexExistsRequest,
    IndexExistsResponse,
    IndexUpsertRequest,
    SearchRequest,
)
from app.observability import TraceContextFilter, setup_json_logging, setup_otel
from app.search_logic import search as search_logic

logger = logging.getLogger("rag")


class AppState:
    settings: Settings
    config_error: str | None = None
    deps: dict[str, dict] = {}
    os: OpenSearchClient | None = None
    qdrant: QdrantFacade | None = None
    embedder: EmbeddingsClient | None = None
    reranker: Reranker | None = None


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
        # keep app running for /healthz + /version diagnostics; /readyz will be false
        yield
        return

    setup_json_logging(state.settings.log_level)
    logging.getLogger().addFilter(TraceContextFilter())
    setup_otel(state.settings.otel_enabled, state.settings.otel_service_name, fastapi_app=app)

    # Pre-create common label series so Grafana panels show 0 instead of "No data" right after startup.
    RAG_PARTIAL.labels(endpoint="/v1/search", kind="partial").inc(0)
    RAG_PARTIAL.labels(endpoint="/v1/search", kind="partial_rerank").inc(0)
    for outcome in ("applied", "skipped", "failed", "unavailable"):
        RAG_RERANK.labels(outcome=outcome).inc(0)

    # Build clients and validate essential resources (best-effort; readiness reflects availability).
    state.embedder = EmbeddingsClient(
        provider=state.settings.embedding_provider,
        vector_size=state.settings.vector_size,
        url=str(state.settings.embedding_url) if state.settings.embedding_url else None,
        model=state.settings.embedding_model,
        api_key=state.settings.embedding_api_key.get_secret_value() if state.settings.embedding_api_key else None,
        dimensions=state.settings.embedding_dimensions,
        timeout_s=state.settings.embedding_timeout_s,
    )

    if state.settings.rerank_mode != "disabled":
        if state.settings.rerank_provider == "local":
            try:
                model_id = state.settings.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                state.reranker = LocalReranker(
                    model_id=model_id,
                    device=state.settings.rerank_device,
                    max_length=512,
                    batch_size=32,
                )
                logger.info("reranker_provider_local", extra={"model": model_id, "device": state.settings.rerank_device})
            except ImportError as e:
                logger.warning("reranker_local_unavailable", extra={"error": str(e)})
                state.reranker = None
        else:
            if state.settings.rerank_url is not None:
                state.reranker = RerankClient(
                    url=str(state.settings.rerank_url),
                    model=state.settings.rerank_model,
                    api_key=state.settings.rerank_api_key.get_secret_value() if state.settings.rerank_api_key else None,
                    timeout_s=state.settings.rerank_timeout_s,
                )
            else:
                state.reranker = None
    else:
        state.reranker = None

    state.deps = {}
    now = time.time()

    # OpenSearch
    try:
        state.os = OpenSearchClient(
            url=str(state.settings.os_url),
            username=state.settings.os_username,
            password=state.settings.os_password.get_secret_value() if state.settings.os_password else None,
            index_alias=state.settings.os_index_alias,
            index_prefix=state.settings.os_index_prefix,
        )
        ok = await asyncio.to_thread(state.os.ping)
        if ok:
            await asyncio.to_thread(state.os.ensure_index_and_alias)
        state.deps["opensearch"] = {"available": bool(ok), "checked_at": now, "error": None if ok else "ping_failed"}
    except Exception as e:
        state.os = None
        state.deps["opensearch"] = {"available": False, "checked_at": now, "error": str(e)}

    # Qdrant
    try:
        state.qdrant = QdrantFacade(
            url=str(state.settings.qdrant_url),
            api_key=state.settings.qdrant_api_key.get_secret_value() if state.settings.qdrant_api_key else None,
            collection=state.settings.qdrant_collection,
            vector_size=state.settings.vector_size,
            distance=state.settings.vector_distance,
        )
        ok = await asyncio.to_thread(state.qdrant.health)
        if ok:
            await asyncio.to_thread(state.qdrant.ensure_collection)
        state.deps["qdrant"] = {"available": bool(ok), "checked_at": now, "error": None if ok else "health_failed"}
    except Exception as e:
        state.qdrant = None
        state.deps["qdrant"] = {"available": False, "checked_at": now, "error": str(e)}

    # Rerank + embeddings are considered "available" if configured; actual call health is deferred.
    state.deps["rerank"] = {
        "available": state.settings.rerank_mode == "disabled" or state.reranker is not None,
        "checked_at": now,
        "error": None,
    }
    state.deps["embeddings"] = {
        "available": state.settings.embedding_provider == "mock" or state.settings.embedding_url is not None,
        "checked_at": now,
        "error": None,
    }

    try:
        yield
    finally:
        # Gracefully close reusable HTTP clients (keep-alive pools).
        try:
            if state.embedder is not None:
                await state.embedder.aclose()
        except Exception:
            pass


app = FastAPI(title="Hybrid Retrieval", version="0.1.0", lifespan=lifespan)

@app.middleware("http")
async def access_log(request: Request, call_next):
    # Prometheus HTTP metrics (avoid self-scrape noise)
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

        dur_ms = dur_s * 1000.0
        logger.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "query": request.url.query,
                "duration_ms": dur_ms,
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

    # At least one retrieval backend should be available for readiness.
    os_ok = bool(state.deps.get("opensearch", {}).get("available"))
    qd_ok = bool(state.deps.get("qdrant", {}).get("available"))
    os_required = bool(state.settings.require_opensearch)
    qd_required = bool(state.settings.require_qdrant)
    ready = os_ok or qd_ok
    if os_required and not os_ok:
        ready = False
    if qd_required and not qd_ok:
        ready = False
    if not ready:
        response.status_code = 503
    return {"ready": ready, "deps": state.deps}


@app.get("/v1/version")
async def version():
    if getattr(state, "settings", None) is None:
        return {"service": {"name": "hybrid-retrieval"}, "config_error": state.config_error}
    return {"service": {"name": state.settings.service_name}, "config": state.settings.safe_summary()}


@app.get("/v1/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# placeholders: will be implemented next
@app.post("/v1/index/upsert")
async def index_upsert(payload: IndexUpsertRequest):
    if state.config_error:
        REQS.labels(endpoint="/v1/index/upsert", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    assert state.embedder is not None

    os_required = bool(state.settings.require_opensearch)
    qd_required = bool(state.settings.require_qdrant)
    os_ok = bool(state.deps.get("opensearch", {}).get("available"))
    qd_ok = bool(state.deps.get("qdrant", {}).get("available"))
    if os_required and not os_ok:
        REQS.labels(endpoint="/v1/index/upsert", status="503").inc()
        return {"ok": False, "error": "opensearch_unavailable", "detail": "OpenSearch is required but unavailable"}
    if qd_required and not qd_ok:
        REQS.labels(endpoint="/v1/index/upsert", status="503").inc()
        return {"ok": False, "error": "qdrant_unavailable", "detail": "Qdrant is required but unavailable"}
    
    # Add timeout wrapper for large documents (15 minutes max)
    try:
        r = await asyncio.wait_for(
            upsert_logic(
                req=payload,
                os_client=state.os,
                qdrant=state.qdrant,
                embedder=state.embedder,
                chunk_strategy=state.settings.chunk_strategy,
                max_tokens=state.settings.chunk_max_tokens,
                overlap_tokens=state.settings.chunk_overlap_tokens,
                chunk_encoding=state.settings.chunk_encoding,
                embedding_batch_size=state.settings.embedding_batch_size,
                embedding_concurrency=state.settings.embedding_concurrency,
                embedding_contextual_headers_enabled=state.settings.embedding_contextual_headers_enabled,
                embedding_contextual_headers_max_chars=state.settings.embedding_contextual_headers_max_chars,
            ),
            timeout=900.0  # 15 minutes for very large documents
        )
        REQS.labels(endpoint="/v1/index/upsert", status="200" if r.ok else "207").inc()
        return r
    except asyncio.TimeoutError:
        doc_id = payload.document.doc_id if payload.document else "unknown"
        text_size = len(payload.text) if payload.text else 0
        logging.error(
            "index_upsert_timeout",
            extra={"extra": {"doc_id": doc_id, "text_size_chars": text_size, "text_size_mb": text_size / (1024 * 1024)}}
        )
        REQS.labels(endpoint="/v1/index/upsert", status="504").inc()
        return {
            "ok": False,
            "partial": True,
            "errors": [{"error": "timeout", "detail": f"Indexing timeout for doc_id={doc_id} (text_size={text_size} chars)"}]
        }


@app.post("/v1/index/delete")
async def index_delete(payload: IndexDeleteRequest):
    if state.config_error:
        REQS.labels(endpoint="/v1/index/delete", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    r = await delete_logic(req=payload, os_client=state.os, qdrant=state.qdrant)
    REQS.labels(endpoint="/v1/index/delete", status="200" if r.ok else "207").inc()
    return r


@app.post("/v1/index/delete-batch")
async def index_delete_batch(payload: IndexDeleteBatchRequest):
    """
    Delete chunks for multiple doc_ids in bulk. Much faster than N single deletes.
    """
    if state.config_error:
        REQS.labels(endpoint="/v1/index/delete-batch", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    r = await delete_batch_logic(req=payload, os_client=state.os, qdrant=state.qdrant)
    REQS.labels(endpoint="/v1/index/delete-batch", status="200" if r.ok else "207").inc()
    return r


@app.post("/v1/index/exists", response_model=IndexExistsResponse)
async def index_exists(payload: IndexExistsRequest):
    """
    Fast "is document indexed" check by doc_id list.
    Uses OpenSearch doc_id terms aggregation (does not depend on query string semantics).
    """
    if state.config_error:
        REQS.labels(endpoint="/v1/index/exists", status="503").inc()
        return IndexExistsResponse(ok=False, indexed_doc_ids=[], counts={})
    if not state.os:
        REQS.labels(endpoint="/v1/index/exists", status="503").inc()
        return IndexExistsResponse(ok=False, indexed_doc_ids=[], counts={})

    try:
        counts = state.os.doc_counts_by_doc_id(payload.doc_ids)
        indexed = [doc_id for doc_id, c in counts.items() if c > 0]
        REQS.labels(endpoint="/v1/index/exists", status="200").inc()
        return IndexExistsResponse(ok=True, indexed_doc_ids=indexed, counts=counts)
    except Exception:
        REQS.labels(endpoint="/v1/index/exists", status="500").inc()
        return IndexExistsResponse(ok=False, indexed_doc_ids=[], counts={})


@app.post("/v1/search")
async def search(payload: SearchRequest):
    if state.config_error:
        REQS.labels(endpoint="/v1/search", status="503").inc()
        return {"ok": False, "error": "config_error", "detail": state.config_error}
    assert state.embedder is not None
    r = await search_logic(
        req=payload,
        os_client=state.os,
        qdrant=state.qdrant,
        embedder=state.embedder,
        reranker=state.reranker,
        rerank_mode=state.settings.rerank_mode,
        rerank_timeout_s=state.settings.rerank_timeout_s,
        rerank_max_candidates=state.settings.rerank_max_candidates,
        rerank_auto_min_query_tokens=state.settings.rerank_auto_min_query_tokens,
        rerank_auto_min_intersection=state.settings.rerank_auto_min_intersection,
        top_k_default=state.settings.default_top_k,
        retrieval_candidates=state.settings.retrieval_candidates,
        bm25_top_k=state.settings.bm25_top_k,
        vector_top_k=state.settings.vector_top_k,
        rrf_k=state.settings.rrf_k,
        weight_bm25=state.settings.weight_bm25,
        weight_vector=state.settings.weight_vector,
        fusion_alpha=state.settings.fusion_alpha,
        max_chunks_per_doc=state.settings.max_chunks_per_doc,
        redact_uri_mode=state.settings.redact_uri_mode,
        enable_page_deduplication=state.settings.enable_page_deduplication,
        enable_parent_page_retrieval=state.settings.enable_parent_page_retrieval,
        adaptive_k_enabled=state.settings.adaptive_k_enabled,
        adaptive_k_min=state.settings.adaptive_k_min,
        adaptive_k_max=state.settings.adaptive_k_max,
    )
    REQS.labels(endpoint="/v1/search", status="200").inc()
    return r

