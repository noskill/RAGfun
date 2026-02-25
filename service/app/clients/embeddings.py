from __future__ import annotations

import random
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential


class EmbeddingsClient:
    def __init__(
        self,
        provider: str,
        vector_size: int,
        url: str | None,
        model: str | None,
        api_key: str | None,
        dimensions: int | None,
        timeout_s: float,
    ):
        self.provider = provider
        self.vector_size = vector_size
        self.url = url
        self.model = model
        self.api_key = api_key
        self.dimensions = dimensions
        self.timeout_s = timeout_s
        # Reuse a single HTTP client to avoid per-request connection setup overhead
        # (especially visible during indexing batches and multi-query retrieval).
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Conservative pool sizes; Infinity (and similar embedding servers) benefit from keep-alive.
            limits = httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=30.0)
            self._client = httpx.AsyncClient(timeout=self.timeout_s, limits=limits)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _mock_embed_one(self, text: str) -> list[float]:
        # Deterministic pseudo-embedding: stable per text; good for local dev/tests.
        seed = int.from_bytes(text.encode("utf-8")[:16].ljust(16, b"\0"), "little", signed=False)
        rnd = random.Random(seed)
        return [rnd.uniform(-1.0, 1.0) for _ in range(self.vector_size)]

    @retry(wait=wait_random_exponential(multiplier=0.2, max=2.0), stop=stop_after_attempt(3))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "mock":
            return [self._mock_embed_one(t) for t in texts]

        if not self.url:
            raise RuntimeError("Embeddings URL is not configured")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {"input": texts}
        # Support OpenAI-compatible embedding backends (e.g. Infinity) that require a model id.
        if self.model:
            payload["model"] = self.model
        # OpenAI embeddings support a dimensions parameter (text-embedding-3+).
        if self.dimensions:
            payload["dimensions"] = int(self.dimensions)
        client = self._get_client()
        try:
            r = await client.post(self.url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        except Exception:
            # Reset the client on any failure to avoid keeping a broken connection in the pool.
            await self.aclose()
            raise

        # Supported response formats:
        # 1) Simple contract: { "vectors": [[...], ...] }
        vectors = data.get("vectors")
        if isinstance(vectors, list):
            return vectors

        # 2) OpenAI embeddings: { "data": [{"embedding":[...]}...], ... }
        oai_data = data.get("data")
        if isinstance(oai_data, list):
            out: list[list[float]] = []
            for it in oai_data:
                if not isinstance(it, dict):
                    continue
                emb = it.get("embedding")
                if isinstance(emb, list):
                    out.append(emb)
            if out:
                return out

        raise RuntimeError("Bad embeddings response: expected 'vectors' or OpenAI-style 'data[].embedding'")

