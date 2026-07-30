"""Typed application configuration from the environment (pydantic-settings).

Override any field with an `APP_`-prefixed env var, e.g. `APP_LLM_BACKEND=openai`.
Fields are constrained (Literal / ge), so an invalid value (a typo'd backend, a
bogus log level) fails fast at startup with a clear ValidationError instead of
surfacing as a confusing error deep inside a request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_", env_file=".env", extra="ignore"
    )

    # Default to development; the Helm chart sets APP_ENV=production explicitly.
    env: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    llm_backend: Literal["mock", "openai"] = "mock"  # "openai" needs OPENAI_API_KEY
    vector_backend: Literal["numpy", "faiss", "qdrant"] = (
        "numpy"  # faiss/qdrant need the extra
    )
    default_top_k: int = Field(default=3, ge=1)
    # Input-contract bounds. Judgement calls sized to the reference pod's
    # 512Mi memory limit, not derivations; the tests pin that each bound exists
    # and bites, not its value.
    max_documents: int = Field(default=10_000, ge=1)
    max_document_chars: int = Field(default=8_000, ge=1)
    max_query_chars: int = Field(default=1_000, ge=1)
    max_top_k: int = Field(default=50, ge=1)
    max_request_bytes: int = Field(default=10_485_760, ge=1)  # 10 MiB
    # Chunking constants are DERIVED, not chosen — scripts/derive_chunking.py
    # measures them and commits chunking_derivation.json; a gate re-runs the
    # producer and pins these defaults against it.
    max_chunk_chars: int = Field(default=256, ge=1)
    chunk_overlap_chars: int = Field(default=83, ge=1)
    # LLM resilience: timeout, bounded retry, consecutive-failure breaker.
    # Judgement calls, APP_-overridable; the tests pin the behaviour of each
    # knob, not its value.
    llm_timeout_seconds: float = Field(default=5.0, gt=0)
    llm_retry_attempts: int = Field(default=1, ge=0)
    llm_breaker_failures: int = Field(default=3, ge=1)
    llm_breaker_reset_seconds: float = Field(default=30.0, gt=0)
    # When set, POST /index (the destructive corpus replace) requires X-API-Key.
    api_key: str = ""

    @model_validator(mode="after")
    def _production_requires_an_api_key(self) -> Settings:
        # The unsafe deploy the docs warn about must not start: dying at boot
        # with the fix beats serving an unauthenticated production data-plane.
        if self.env == "production" and not self.api_key:
            raise ValueError(
                "APP_ENV=production requires APP_API_KEY to be set: without it "
                "/index and /query are open to anyone who can reach the pod"
            )
        return self


def get_settings() -> Settings:
    return Settings()
