"""Typed application configuration from the environment (pydantic-settings).

Override any field with an `APP_`-prefixed env var, e.g. `APP_LLM_BACKEND=openai`.
Fields are constrained (Literal / ge), so an invalid value (a typo'd backend, a
bogus log level) fails fast at startup with a clear ValidationError instead of
surfacing as a confusing error deep inside a request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
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
    # When set, POST /index (the destructive corpus replace) requires X-API-Key.
    api_key: str = ""


def get_settings() -> Settings:
    return Settings()
