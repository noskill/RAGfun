from __future__ import annotations

import base64
import logging
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger("processor.clients")


class StorageClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def get_metadata(self, *, doc_id: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            # Use query param to avoid FastAPI path parsing issues with colons
            r = await client.get(f"{self._base_url}/v1/documents/by-id/metadata", params={"doc_id": doc_id})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()

    async def get_file(self, *, doc_id: str) -> tuple[bytes, str | None]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.get(f"{self._base_url}/v1/documents/by-id", params={"doc_id": doc_id})
            r.raise_for_status()
            ct = r.headers.get("content-type")
            return (r.content, ct)

    async def patch_extra(self, *, doc_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/documents/by-id/extra", json={"doc_id": doc_id, "patch": patch})
            r.raise_for_status()
            return r.json()

    async def delete_document(self, *, doc_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.delete(f"{self._base_url}/v1/documents/by-id", params={"doc_id": doc_id})
            r.raise_for_status()
            return r.json()


class RetrievalClient:
    def __init__(self, *, base_url: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    async def index_upsert(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/upsert", json=payload)
            r.raise_for_status()
            return r.json()

    async def index_delete(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/v1/index/delete", json=payload)
            r.raise_for_status()
            return r.json()


class VLMClient:
    """
    OpenAI-compatible chat completions client for multimodal extraction in vLLM.
    """

    def __init__(self, *, base_url: str, api_key: str | None, model: str, timeout_s: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s

    async def page_to_text(self, *, png_bytes: bytes) -> str:
        data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Convert this document page to clean text. "
                                "Preserve structure (headings, lists, tables) as Markdown. "
                                "Do not invent content. Output ONLY the converted text."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(f"{self._base_url}/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            j = r.json()
        try:
            return str(j["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            logger.error("vlm_unexpected_response", extra={"extra": {"keys": list(j.keys())}})
            raise RuntimeError("vlm_unexpected_response")


class LandingAIClient:
    """
    LandingAI ADE Parse client for PDF -> Markdown.
    """

    def __init__(
        self,
        *,
        parse_url: str,
        api_key: str | None,
        model: str,
        split: str | None,
        timeout_s: float,
    ) -> None:
        self._parse_url = parse_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._split = split
        self._timeout_s = timeout_s

    async def parse_pdf(self, *, pdf_bytes: bytes, filename: str | None) -> dict[str, Any]:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # LandingAI expects multipart form-data.
        data = {"model": self._model}
        if self._split:
            data["split"] = self._split
        files = {
            "document": (
                (filename or "document.pdf"),
                pdf_bytes,
                "application/pdf",
            )
        }

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            r = await client.post(self._parse_url, data=data, files=files, headers=headers)
            r.raise_for_status()
            return r.json()




