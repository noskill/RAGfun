from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import aio_pika
import httpx
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from app.clients import RetrievalClient, StorageClient
from app.logging_setup import setup_json_logging

logger = logging.getLogger("processor.worker")

CONSUMED = Counter("ingestion_tasks_consumed_total", "Consumed tasks", ["type", "status"])
RETRIED = Counter("ingestion_tasks_retried_total", "Retried tasks", ["type"])
FAILED = Counter("ingestion_tasks_failed_total", "Failed tasks", ["type"])
INFLIGHT = Gauge("ingestion_tasks_inflight", "Tasks currently processing", ["type"])
LAT = Histogram("ingestion_task_duration_seconds", "Task duration", ["type", "stage"])


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _parse_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)) or default)
    except Exception:
        return default


def _parse_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)) or default)
    except Exception:
        return default


def _delay_s(attempt: int) -> int:
    # conservative backoff: 5s, 30s, 2m, 10m, then 30m
    if attempt <= 0:
        return 5
    if attempt == 1:
        return 30
    if attempt == 2:
        return 120
    if attempt == 3:
        return 600
    return 1800


async def _patch_ingestion(storage: StorageClient, *, doc_id: str, ingestion: dict[str, Any]) -> None:
    # best-effort
    try:
        await storage.patch_extra(doc_id=doc_id, patch={"ingestion": ingestion})
    except Exception as e:
        logger.warning("patch_ingestion_failed", extra={"extra": {"doc_id": doc_id, "error": str(e)}})


async def handle_index(
    *,
    task: dict[str, Any],
    storage: StorageClient,
    doc_processor_url: str,
    doc_processor_timeout_s: float,
) -> None:
    doc_id = str(task.get("doc_id"))
    task_id = str(task.get("task_id") or "")
    attempt = int(task.get("attempt") or 0)
    now = time.time()

    # Exact deduplication: document-storage may flag byte-identical duplicates via storage_id.
    # We still index per-doc_id (doc_id is part of the retrieval contract), but we keep the info in ingestion metadata.
    try:
        meta = await storage.get_metadata(doc_id=doc_id)
        extra = (meta or {}).get("extra") if isinstance(meta, dict) else None
        dedup = (extra or {}).get("dedup") if isinstance(extra, dict) else None
        duplicate_of = (dedup or {}).get("duplicate_of") if isinstance(dedup, dict) else None
        if duplicate_of:
            await _patch_ingestion(
                storage,
                doc_id=doc_id,
                ingestion={
                    "state": "processing",
                    "type": "index",
                    "task_id": task_id,
                    "doc_id": doc_id,
                    "attempt": attempt,
                    "updated_at": now,
                    "stage": "duplicate_detected",
                    "result": {"duplicate_of": str(duplicate_of)},
                },
            )
    except Exception as e:
        # best-effort; if metadata can't be fetched, continue indexing
        logger.warning("dedup_metadata_check_failed", extra={"extra": {"doc_id": doc_id, "error": str(e)}})
    ingestion = {
        "state": "processing",
        "type": "index",
        "task_id": task_id,
        "doc_id": doc_id,
        "attempt": attempt,
        "updated_at": now,
        "stage": "doc_processor",
    }
    await _patch_ingestion(storage, doc_id=doc_id, ingestion=ingestion)

    payload = {"document": task.get("document") or {"doc_id": doc_id}, "refresh": bool(task.get("refresh"))}
    with LAT.labels("index", "doc_processor").time():
        async with httpx.AsyncClient(timeout=doc_processor_timeout_s) as client:
            r = await client.post(f"{doc_processor_url.rstrip('/')}/v1/process", json=payload)
            r.raise_for_status()
            resp = r.json()
    if resp.get("ok") is not True:
        try:
            logger.error(
                "doc_processor_failed",
                extra={
                    "extra": {
                        "doc_id": doc_id,
                        "task_id": task_id,
                        "attempt": attempt,
                        "error": resp.get("error"),
                        "detail": resp.get("detail"),
                    }
                },
            )
        except Exception:
            logger.error("doc_processor_failed", extra={"extra": {"doc_id": doc_id, "task_id": task_id}})
        raise RuntimeError(f"doc_processor_ok_false:{resp.get('error') or 'unknown'}")
    if resp.get("skipped") is True:
        now = time.time()
        ingestion = {
            "state": "done",
            "type": "index",
            "task_id": task_id,
            "doc_id": doc_id,
            "attempt": attempt,
            "updated_at": now,
            "stage": "skipped",
            "result": {
                "pages": resp.get("pages"),
                "chunks": resp.get("chunks"),
                "partial": resp.get("partial"),
                "degraded": resp.get("degraded"),
                "error": resp.get("error"),
                "detail": resp.get("detail"),
            },
        }
        await _patch_ingestion(storage, doc_id=doc_id, ingestion=ingestion)
        return
    retrieval = resp.get("retrieval") if isinstance(resp, dict) else None
    retrieval_ok = True
    retrieval_partial = False
    if isinstance(retrieval, dict):
        retrieval_ok = retrieval.get("ok") is True
        retrieval_partial = bool(retrieval.get("partial"))
    if not retrieval_ok or retrieval_partial:
        raise RuntimeError(f"doc_processor_retrieval_failed:ok={retrieval_ok},partial={retrieval_partial}")

    now = time.time()
    ingestion = {
        "state": "done",
        "type": "index",
        "task_id": task_id,
        "doc_id": doc_id,
        "attempt": attempt,
        "updated_at": now,
        "stage": "indexed",
        "result": {"pages": resp.get("pages"), "chunks": resp.get("chunks"), "partial": resp.get("partial"), "degraded": resp.get("degraded")},
    }
    await _patch_ingestion(storage, doc_id=doc_id, ingestion=ingestion)


async def handle_delete(
    *,
    task: dict[str, Any],
    storage: StorageClient,
    retrieval: RetrievalClient,
) -> None:
    doc_id = str(task.get("doc_id"))
    task_id = str(task.get("task_id") or "")
    attempt = int(task.get("attempt") or 0)
    now = time.time()
    await _patch_ingestion(
        storage,
        doc_id=doc_id,
        ingestion={
            "state": "processing",
            "type": "delete",
            "task_id": task_id,
            "doc_id": doc_id,
            "attempt": attempt,
            "updated_at": now,
            "stage": "delete_index",
        },
    )

    # 1) delete from retrieval
    with LAT.labels("delete", "retrieval_delete").time():
        await retrieval.index_delete(payload={"doc_id": doc_id, "refresh": True})

    # 2) best-effort delete from storage (this removes metadata too)
    with LAT.labels("delete", "storage_delete").time():
        try:
            await storage.delete_document(doc_id=doc_id)
        except httpx.HTTPStatusError as e:
            # Treat missing docs as success: delete is idempotent and users may request deletion
            # multiple times or the doc may have been removed already.
            if e.response is not None and e.response.status_code == 404:
                logger.info("storage_delete_not_found", extra={"extra": {"doc_id": doc_id}})
            else:
                raise

    now = time.time()
    await _patch_ingestion(
        storage,
        doc_id=doc_id,
        ingestion={
            "state": "done",
            "type": "delete",
            "task_id": task_id,
            "doc_id": doc_id,
            "attempt": attempt,
            "updated_at": now,
            "stage": "deleted",
        },
    )


async def main() -> None:
    log_level = _env("WORKER_LOG_LEVEL", "INFO") or "INFO"
    setup_json_logging(log_level)

    rabbit_url = _env("WORKER_RABBIT_URL", None)
    if not rabbit_url:
        raise SystemExit("WORKER_RABBIT_URL is required")

    queue_name = _env("WORKER_QUEUE", "ingestion.tasks") or "ingestion.tasks"
    retry_queue = _env("WORKER_RETRY_QUEUE", "ingestion.tasks.retry") or "ingestion.tasks.retry"
    dlq_queue = _env("WORKER_DLQ_QUEUE", "ingestion.tasks.dlq") or "ingestion.tasks.dlq"
    retry_ttl_ms = _parse_int("WORKER_RETRY_TTL_MS", 5000)

    max_attempts = _parse_int("WORKER_MAX_ATTEMPTS", 5)
    prefetch = _parse_int("WORKER_PREFETCH", 5)
    max_concurrency = _parse_int("WORKER_CONCURRENCY", prefetch)
    max_concurrency = max(1, int(max_concurrency))

    metrics_port = _parse_int("WORKER_METRICS_PORT", 8083)
    start_http_server(metrics_port)
    logger.info("metrics_started", extra={"extra": {"port": metrics_port}})

    storage_url = _env("WORKER_STORAGE_URL", _env("PROCESSOR_STORAGE_URL", "http://document-storage:8081")) or "http://document-storage:8081"
    retrieval_url = _env("WORKER_RETRIEVAL_URL", _env("PROCESSOR_RETRIEVAL_URL", "http://retrieval:8080")) or "http://retrieval:8080"
    storage_timeout_s = _parse_float("WORKER_STORAGE_TIMEOUT_S", 60.0)
    retrieval_timeout_s = _parse_float("WORKER_RETRIEVAL_TIMEOUT_S", 60.0)

    doc_processor_url = _env("WORKER_DOC_PROCESSOR_URL", "http://doc-processor:8082") or "http://doc-processor:8082"
    doc_processor_timeout_s = _parse_float("WORKER_DOC_PROCESSOR_TIMEOUT_S", 300.0)

    storage = StorageClient(base_url=storage_url, timeout_s=storage_timeout_s)
    retrieval = RetrievalClient(base_url=retrieval_url, timeout_s=retrieval_timeout_s)

    conn = await aio_pika.connect_robust(rabbit_url)
    channel = await conn.channel()
    await channel.set_qos(prefetch_count=prefetch)

    # Main queue
    q = await channel.declare_queue(queue_name, durable=True)

    # Retry queue: messages expire (per-message TTL) and dead-letter back into main queue.
    await channel.declare_queue(
        retry_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": "",
            "x-dead-letter-routing-key": queue_name,
            # Prefer queue-level TTL: more reliable than per-message expiration, and avoids plugin needs.
            "x-message-ttl": int(retry_ttl_ms),
        },
    )
    await channel.declare_queue(dlq_queue, durable=True)

    async def republish_retry(task: dict[str, Any], *, reason: str) -> None:
        t = dict(task)
        t["attempt"] = int(t.get("attempt") or 0) + 1
        body = json.dumps(t).encode("utf-8")
        msg = aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers={"reason": reason, "attempt": int(t["attempt"])},
        )
        await channel.default_exchange.publish(msg, routing_key=retry_queue)

    async def republish_dlq(task: dict[str, Any], *, reason: str) -> None:
        body = json.dumps(task).encode("utf-8")
        msg = aio_pika.Message(
            body=body,
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            headers={"reason": reason, "attempt": int(task.get("attempt") or 0)},
        )
        await channel.default_exchange.publish(msg, routing_key=dlq_queue)

    sem = asyncio.Semaphore(max_concurrency)
    tasks: set[asyncio.Task[None]] = set()

    async def _process_message(message: aio_pika.IncomingMessage) -> None:
        async with sem:
            async with message.process(ignore_processed=True):
                try:
                    task = json.loads(message.body.decode("utf-8"))
                except Exception:
                    # bad message: drop to DLQ
                    CONSUMED.labels(type="unknown", status="bad_json").inc()
                    return

                ttype = str(task.get("type") or "unknown")
                attempt = int(task.get("attempt") or 0)
                INFLIGHT.labels(ttype).inc()
                t0 = time.time()
                try:
                    if ttype == "index":
                        await handle_index(
                            task=task,
                            storage=storage,
                            doc_processor_url=doc_processor_url,
                            doc_processor_timeout_s=doc_processor_timeout_s,
                        )
                    elif ttype == "delete":
                        await handle_delete(task=task, storage=storage, retrieval=retrieval)
                    else:
                        raise RuntimeError(f"unknown_task_type:{ttype}")

                    CONSUMED.labels(type=ttype, status="ok").inc()
                except Exception as e:
                    CONSUMED.labels(type=ttype, status="error").inc()
                    FAILED.labels(type=ttype).inc()

                    doc_id = str(task.get("doc_id") or "")
                    task_id = str(task.get("task_id") or "")
                    now = time.time()
                    await _patch_ingestion(
                        storage,
                        doc_id=doc_id,
                        ingestion={
                            "state": "failed" if attempt >= max_attempts else "retrying",
                            "type": ttype,
                            "task_id": task_id,
                            "doc_id": doc_id,
                            "attempt": attempt,
                            "updated_at": now,
                            "last_error": f"{type(e).__name__}: {str(e)[:500]}",
                        },
                    )

                    if attempt >= max_attempts:
                        await republish_dlq(task, reason=str(e)[:200])
                    else:
                        RETRIED.labels(type=ttype).inc()
                        await republish_retry(task, reason=str(e)[:200])
                    logger.error("task_failed", extra={"extra": {"type": ttype, "attempt": attempt, "error": str(e)}})
                finally:
                    INFLIGHT.labels(ttype).dec()
                    _ = t0  # keep for future if we add overall histograms

    async with q.iterator() as queue_iter:
        async for message in queue_iter:
            task = asyncio.create_task(_process_message(message))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
            if len(tasks) >= max_concurrency:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)


if __name__ == "__main__":
    asyncio.run(main())
