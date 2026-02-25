from __future__ import annotations

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    # Service
    service_name: str = "hybrid-retrieval"
    environment: str = "dev"
    log_level: str = "INFO"

    # OpenSearch
    os_url: AnyHttpUrl = Field(default="http://localhost:9200")
    os_username: str | None = None
    os_password: SecretStr | None = None
    os_index_alias: str = "rag_chunks"
    os_index_prefix: str = "rag_chunks_v"
    require_opensearch: bool = False

    # Qdrant
    qdrant_url: AnyHttpUrl = Field(default="http://localhost:6333")
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rag_chunks"
    vector_size: int = 768
    vector_distance: str = "cosine"  # cosine|dot|euclid
    require_qdrant: bool = False

    # Embeddings
    embedding_provider: str = "http"  # http|mock
    embedding_url: AnyHttpUrl | None = None
    embedding_model: str | None = None  # optional (e.g. OpenAI-compatible backends like Infinity)
    embedding_api_key: SecretStr | None = None
    embedding_dimensions: int | None = None
    embedding_timeout_s: float = 10.0
    embedding_batch_size: int = 32
    embedding_concurrency: int = 1

    # Chunking (mode=document only; mode=chunks uses doc-processor output)
    chunk_strategy: str = "semantic"  # semantic|token|semchunk
    chunk_max_tokens: int = 300
    chunk_overlap_tokens: int = 50
    chunk_encoding: str = "cl100k_base"  # tiktoken encoding for semchunk (and token/semantic)

    # Search: hybrid (BM25 + dense), configurable weights. RU/EN supported via embedder + BM25 analyzer.
    default_top_k: int = 10
    retrieval_candidates: int = 20  # candidates to retrieve before rerank (top-20 → rerank → top_k)
    bm25_top_k: int = 25  # per-source; should be >= retrieval_candidates for fusion pool
    vector_top_k: int = 25
    rrf_k: int = 60
    weight_bm25: float = 1.0
    weight_vector: float = 1.0
    max_chunks_per_doc: int = 3
    fusion_alpha: float = 0.75  # 1.0=rank-only (RRF), 0.0=score-only (normalized)

    # Rerank: swappable module. "http" = external service; "local" = in-process CPU (sentence-transformers).
    rerank_provider: str = "http"  # http|local
    rerank_mode: str = "disabled"  # disabled|always|auto
    rerank_url: AnyHttpUrl | None = None  # required when provider=http
    rerank_model: str | None = None  # optional for http (e.g. Infinity model id); HF model id for local
    rerank_device: str = "cpu"  # for local: cpu|cuda
    rerank_api_key: SecretStr | None = None
    rerank_timeout_s: float = 6.0
    rerank_max_candidates: int = 20  # align with retrieval_candidates for near-SOTA pipeline
    rerank_auto_min_query_tokens: int = 6
    rerank_auto_min_intersection: int = 2

    # Sources / URI redaction
    redact_uri_mode: str = "none"  # none|strip_query|strip_all

    # Page retrieval improvements
    enable_page_deduplication: bool = False  # Deduplicate chunks by (doc_id, page)
    enable_parent_page_retrieval: bool = False  # Return full pages instead of chunks

    # Adaptive-k: cut at steepest score drop instead of fixed top_k.
    adaptive_k_enabled: bool = False
    adaptive_k_min: int = 3
    adaptive_k_max: int = 24

    # Embedding enrichment (opt-in):
    # prepend lightweight contextual headers (title/uri/page/lang/source) to the text BEFORE embedding,
    # while keeping stored chunk text unchanged.
    # This usually improves vector retrieval for ambiguous references and multi-hop questions,
    # but requires re-embedding once enabled.
    embedding_contextual_headers_enabled: bool = False
    embedding_contextual_headers_max_chars: int = 400

    # Observability
    otel_enabled: bool = False
    otel_service_name: str = "hybrid-retrieval"

    def safe_summary(self) -> dict:
        """Configuration summary without secrets."""
        return {
            "service": {
                "name": self.service_name,
                "environment": self.environment,
            },
            "opensearch": {
                "url": str(self.os_url),
                "index_alias": self.os_index_alias,
                "index_prefix": self.os_index_prefix,
                "required": bool(self.require_opensearch),
                "username_set": self.os_username is not None,
            },
            "qdrant": {
                "url": str(self.qdrant_url),
                "collection": self.qdrant_collection,
                "vector_size": self.vector_size,
                "vector_distance": self.vector_distance,
                "required": bool(self.require_qdrant),
                "api_key_set": self.qdrant_api_key is not None,
            },
            "embeddings": {
                "provider": self.embedding_provider,
                "url": str(self.embedding_url) if self.embedding_url else None,
                "model": self.embedding_model,
                "api_key_set": self.embedding_api_key is not None,
                "dimensions": self.embedding_dimensions,
                "timeout_s": self.embedding_timeout_s,
                "batch_size": self.embedding_batch_size,
                "concurrency": self.embedding_concurrency,
                "contextual_headers_enabled": self.embedding_contextual_headers_enabled,
            },
            "chunking": {
                "strategy": self.chunk_strategy,
                "max_tokens": self.chunk_max_tokens,
                "overlap_tokens": self.chunk_overlap_tokens,
                "encoding": self.chunk_encoding,
            },
            "search": {
                "default_top_k": self.default_top_k,
                "retrieval_candidates": self.retrieval_candidates,
                "bm25_top_k": self.bm25_top_k,
                "vector_top_k": self.vector_top_k,
                "rrf_k": self.rrf_k,
                "weights": {"bm25": self.weight_bm25, "vector": self.weight_vector},
                "max_chunks_per_doc": self.max_chunks_per_doc,
            },
            "rerank": {
                "provider": self.rerank_provider,
                "mode": self.rerank_mode,
                "url": str(self.rerank_url) if self.rerank_url else None,
                "model": self.rerank_model,
                "device": self.rerank_device,
                "api_key_set": self.rerank_api_key is not None,
                "timeout_s": self.rerank_timeout_s,
                "max_candidates": self.rerank_max_candidates,
            },
            "sources": {
                "redact_uri_mode": self.redact_uri_mode,
            },
            "page_retrieval": {
                "enable_page_deduplication": self.enable_page_deduplication,
                "enable_parent_page_retrieval": self.enable_parent_page_retrieval,
            },
            "otel": {
                "enabled": self.otel_enabled,
                "service_name": self.otel_service_name,
            },
        }


def load_settings() -> Settings:
    s = Settings()
    if s.embedding_provider == "http" and s.embedding_url is None:
        raise ValueError("RAG_EMBEDDING_PROVIDER=http requires RAG_EMBEDDING_URL")
    if s.embedding_concurrency <= 0:
        raise ValueError("RAG_EMBEDDING_CONCURRENCY must be > 0")
    if s.embedding_dimensions is not None:
        if s.embedding_dimensions <= 0:
            raise ValueError("RAG_EMBEDDING_DIMENSIONS must be > 0")
        if s.vector_size != s.embedding_dimensions:
            raise ValueError("RAG_VECTOR_SIZE must match RAG_EMBEDDING_DIMENSIONS when set")
    if s.rerank_mode != "disabled":
        if s.rerank_provider == "http" and s.rerank_url is None:
            raise ValueError("RAG_RERANK_PROVIDER=http requires RAG_RERANK_URL when rerank is enabled")
        if s.rerank_provider not in ("http", "local"):
            raise ValueError("RAG_RERANK_PROVIDER must be 'http' or 'local'")
    if s.vector_size <= 0:
        raise ValueError("RAG_VECTOR_SIZE must be > 0")
    if s.chunk_max_tokens <= 0:
        raise ValueError("RAG_CHUNK_MAX_TOKENS must be > 0")
    if s.chunk_overlap_tokens < 0 or s.chunk_overlap_tokens >= s.chunk_max_tokens:
        raise ValueError("RAG_CHUNK_OVERLAP_TOKENS must be >=0 and < CHUNK_MAX_TOKENS")
    if s.chunk_strategy not in ("semantic", "token", "semchunk"):
        raise ValueError("RAG_CHUNK_STRATEGY must be 'semantic', 'token' or 'semchunk'")
    if s.retrieval_candidates < 1:
        raise ValueError("RAG_RETRIEVAL_CANDIDATES must be >= 1")
    return s

