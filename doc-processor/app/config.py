from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROCESSOR_", extra="ignore")

    # Service
    service_name: str = "doc-processor"
    environment: str = "dev"
    log_level: str = "INFO"

    # Downstream services
    storage_url: AnyHttpUrl = Field(default="http://document-storage:8081")
    storage_timeout_s: float = 60.0

    retrieval_url: AnyHttpUrl = Field(default="http://retrieval:8080")
    retrieval_timeout_s: float = 60.0

    # VLM / Document parsing
    vlm_provider: Literal["vllm", "landing_ai", "tika"] = "vllm"

    # vLLM (OpenAI-compatible)
    vlm_base_url: AnyHttpUrl = Field(default="http://vllm-docling:8123/v1")
    vlm_api_key: SecretStr | None = None
    vlm_model: str = "ibm-granite/granite-docling-258M"
    vlm_timeout_s: float = 120.0

    # LandingAI ADE Parse
    landing_parse_url: AnyHttpUrl = Field(default="https://api.va.landing.ai/v1/ade/parse")
    landing_api_key: SecretStr | None = None
    landing_model: str = "dpt-2-latest"
    landing_split: str = "page"
    landing_timeout_s: float = 120.0

    # Apache Tika
    tika_url: AnyHttpUrl = Field(default="http://tika:9998")
    tika_timeout_s: float = 60.0

    # Limits
    max_pages: int = 25
    max_image_side_px: int = 1600

    # Chunking: strategy "semantic" (section-based, minimal overlap) or "fixed" (char-based with overlap)
    chunk_strategy: str = "semantic"  # semantic|fixed
    chunk_size_chars: int = 4000
    chunk_overlap_chars: int = 300

    def safe_summary(self) -> dict:
        return {
            "service": {"name": self.service_name, "environment": self.environment},
            "storage": {"url": str(self.storage_url), "timeout_s": self.storage_timeout_s},
            "retrieval": {"url": str(self.retrieval_url), "timeout_s": self.retrieval_timeout_s},
            "vlm": {
                "provider": self.vlm_provider,
                "base_url": str(self.vlm_base_url),
                "model": self.vlm_model,
                "api_key_set": self.vlm_api_key is not None,
                "timeout_s": self.vlm_timeout_s,
            },
            "landing_ai": {
                "parse_url": str(self.landing_parse_url),
                "model": self.landing_model,
                "split": self.landing_split,
                "api_key_set": self.landing_api_key is not None,
                "timeout_s": self.landing_timeout_s,
            },
            "tika": {"url": str(self.tika_url), "timeout_s": self.tika_timeout_s},
            "limits": {
                "max_pages": self.max_pages,
                "max_image_side_px": self.max_image_side_px,
                "chunk_strategy": self.chunk_strategy,
                "chunk_size_chars": self.chunk_size_chars,
                "chunk_overlap_chars": self.chunk_overlap_chars,
            },
        }


def load_settings() -> Settings:
    s = Settings()
    if s.chunk_strategy not in ("semantic", "fixed"):
        raise ValueError("PROCESSOR_CHUNK_STRATEGY must be 'semantic' or 'fixed'")
    if s.vlm_provider == "landing_ai":
        if s.landing_api_key is None or s.landing_api_key.get_secret_value().strip() == "":
            raise ValueError("PROCESSOR_LANDING_API_KEY is required when PROCESSOR_VLM_PROVIDER=landing_ai")
    return s
