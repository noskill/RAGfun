#!/usr/bin/env python3
import argparse
import hashlib
import mimetypes
import json
import os
import sys
from urllib.parse import quote

try:
    import requests
except Exception as e:
    print("This script requires 'requests'. Install with: pip install requests", file=sys.stderr)
    raise


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def jdump(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def base_url():
    return os.environ.get("GATE_URL", "http://localhost:8090").rstrip("/")


def opensearch_url():
    return os.environ.get("OPENSEARCH_URL", "http://localhost:9200").rstrip("/")


def opensearch_index():
    return (
        os.environ.get("OPENSEARCH_INDEX")
        or os.environ.get("RAG_OS_INDEX_ALIAS")
        or "rag_chunks"
    )


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv_list(val: str | None) -> list[str] | None:
    if not val:
        return None
    items = [v.strip() for v in val.split(",")]
    items = [v for v in items if v]
    return items or None


def cmd_upload(args):
    url = f"{base_url()}/v1/documents/upload"
    doc_id = args.doc_id or _hash_file(args.file)
    data = {
        "doc_id": doc_id,
    }
    # Optional fields
    # Default title to filename if not provided
    title = args.title if args.title is not None else os.path.basename(args.file)
    if title:
        data["title"] = title

    for key in [
        "uri",
        "source",
        "lang",
        "tags",
        "acl",
        "tenant_id",
        "project_id",
    ]:
        val = getattr(args, key)
        if val is not None:
            data[key] = val
    if args.refresh:
        data["refresh"] = "true"

    mime, _ = mimetypes.guess_type(args.file)
    with open(args.file, "rb") as f:
        if mime:
            files = {"file": (os.path.basename(args.file), f, mime)}
        else:
            files = {"file": (os.path.basename(args.file), f)}
        r = requests.post(url, data=data, files=files, timeout=args.timeout)
    try:
        payload = r.json()
    except Exception:
        payload = {"status_code": r.status_code, "text": r.text}
    jdump(payload)


def cmd_chat(args):
    url = f"{base_url()}/v1/chat"
    payload: dict[str, object] = {
        "query": args.query,
        "include_sources": not args.no_sources,
    }

    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.mode:
        payload["retrieval_mode"] = args.mode
    if args.rerank is not None:
        payload["rerank"] = args.rerank
    if args.use_adaptive_k is not None:
        payload["use_adaptive_k"] = args.use_adaptive_k

    filters: dict[str, object] = {}
    if args.source:
        filters["source"] = args.source
    if args.tags:
        filters["tags"] = _csv_list(args.tags)
    if args.lang:
        filters["lang"] = args.lang
    if args.doc_ids:
        filters["doc_ids"] = _csv_list(args.doc_ids)
    if args.tenant_id:
        filters["tenant_id"] = args.tenant_id
    if args.project_id:
        filters["project_id"] = args.project_id
    if args.project_ids:
        filters["project_ids"] = _csv_list(args.project_ids)
    if filters:
        payload["filters"] = filters

    if args.acl:
        payload["acl"] = _csv_list(args.acl) or []

    r = requests.post(url, json=payload, timeout=args.timeout)
    try:
        payload = r.json()
    except Exception:
        payload = {"status_code": r.status_code, "text": r.text}
    jdump(payload)


def cmd_list(args):
    url = f"{base_url()}/v1/documents"
    params = {
        "limit": args.limit,
        "offset": args.offset,
    }
    if args.source:
        params["source"] = args.source
    if args.tags:
        params["tags"] = args.tags
    if args.lang:
        params["lang"] = args.lang
    if args.collections:
        params["collections"] = args.collections
    r = requests.get(url, params=params, timeout=args.timeout)
    try:
        payload = r.json()
    except Exception:
        payload = {"status_code": r.status_code, "text": r.text}
    jdump(payload)


def cmd_dump_text(args):
    url = f"{opensearch_url()}/{opensearch_index()}/_search"
    must = [{"term": {"doc_id": args.doc_id}}]
    if args.page is not None:
        must.append({"term": {"locator.page": int(args.page)}})
    body = {
        "size": int(args.max_chunks),
        "query": {"bool": {"must": must}},
        "sort": [{"chunk_index": {"order": "asc"}}],
        "_source": ["chunk_index", "locator", "text"],
    }
    r = requests.post(url, json=body, timeout=args.timeout)
    try:
        payload = r.json()
    except Exception:
        payload = {"status_code": r.status_code, "text": r.text}
        jdump(payload)
        return

    hits = (payload.get("hits") or {}).get("hits") or []
    chunks = []
    for h in hits:
        src = h.get("_source") or {}
        text = src.get("text") or ""
        locator = src.get("locator") or {}
        page = locator.get("page")
        chunk_index = src.get("chunk_index", 0)
        chunks.append((page, chunk_index, text))

    if not chunks:
        print("")
        return

    has_pages = any(p is not None for p, _, _ in chunks)
    if not has_pages:
        out = "\n\n".join(t for _, _, t in chunks if t)
        print(out)
        return

    chunks.sort(key=lambda x: ((x[0] if x[0] is not None else 0), x[1]))
    current_page = None
    first = True
    for page, _, text in chunks:
        if page != current_page:
            if not first:
                print("\n")
            current_page = page
            print(f"=== page {current_page} ===")
            first = False
        if text:
            print(text)
            print("")


def cmd_status(args):
    # doc_id must be URL-encoded
    doc_id_enc = quote(args.doc_id, safe="")
    url = f"{base_url()}/v1/documents/{doc_id_enc}/status"
    r = requests.get(url, timeout=args.timeout)
    try:
        payload = r.json()
    except Exception:
        payload = {"status_code": r.status_code, "text": r.text}
    jdump(payload)


def build_parser():
    p = argparse.ArgumentParser(description="Minimal client for rag-gate document APIs")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")

    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Upload a document")
    up.add_argument("file", help="Path to file")
    up.add_argument("doc_id", nargs="?", help="Document ID (defaults to sha256 of file)")
    up.add_argument("--title")
    up.add_argument("--uri")
    up.add_argument("--source")
    up.add_argument("--lang")
    up.add_argument("--tags", help="Comma-separated tags")
    up.add_argument("--acl", help="Comma-separated ACL")
    up.add_argument("--tenant_id")
    up.add_argument("--project_id", help="Collection ID")
    up.add_argument("--refresh", action="store_true", help="Force index refresh")
    up.set_defaults(func=cmd_upload)

    ls = sub.add_parser("list", help="List documents")
    ls.add_argument("--source")
    ls.add_argument("--tags", help="Comma-separated tags")
    ls.add_argument("--lang")
    ls.add_argument("--collections", help="Comma-separated project_ids")
    ls.add_argument("--limit", type=int, default=100)
    ls.add_argument("--offset", type=int, default=0)
    ls.set_defaults(func=cmd_list)

    st = sub.add_parser("status", help="Get document status")
    st.add_argument("doc_id", help="Document ID")
    st.set_defaults(func=cmd_status)

    ch = sub.add_parser("chat", help="Ask a question (RAG) and return sources")
    ch.add_argument("query", help="User query")
    ch.add_argument("--top_k", type=int, help="Top-k chunks to retrieve")
    ch.add_argument("--mode", choices=["bm25", "vector", "hybrid"], help="Retrieval mode")
    ch.add_argument("--rerank", dest="rerank", action="store_true", help="Enable rerank")
    ch.add_argument("--no-rerank", dest="rerank", action="store_false", help="Disable rerank")
    ch.set_defaults(rerank=None)
    ch.add_argument("--use-adaptive-k", dest="use_adaptive_k", action="store_true", help="Enable adaptive-k")
    ch.add_argument("--no-adaptive-k", dest="use_adaptive_k", action="store_false", help="Disable adaptive-k")
    ch.set_defaults(use_adaptive_k=None)
    ch.add_argument("--no-sources", action="store_true", help="Disable sources in response")
    ch.add_argument("--source", help="Filter by source")
    ch.add_argument("--tags", help="Comma-separated tags")
    ch.add_argument("--lang", help="Filter by language")
    ch.add_argument("--doc-ids", dest="doc_ids", help="Comma-separated doc_ids")
    ch.add_argument("--tenant-id", dest="tenant_id", help="Tenant ID")
    ch.add_argument("--project-id", dest="project_id", help="Collection ID")
    ch.add_argument("--project-ids", dest="project_ids", help="Comma-separated project_ids")
    ch.add_argument("--acl", help="Comma-separated ACL")
    ch.set_defaults(func=cmd_chat)

    dt = sub.add_parser("dump-text", help="Dump extracted text by doc_id from OpenSearch")
    dt.add_argument("doc_id", help="Document ID")
    dt.add_argument("--page", type=int, help="Optional page number")
    dt.add_argument("--max-chunks", type=int, default=2000, help="Max chunks to fetch")
    dt.set_defaults(func=cmd_dump_text)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
