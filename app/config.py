"""Typed application configuration from the environment (pydantic-settings).

Override any field with an `APP_`-prefixed env var, e.g. `APP_LLM_BACKEND=openai`.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    env: str = "production"
    log_level: str = "INFO"
    llm_backend: str = "mock"          # set "openai" (+ OPENAI_API_KEY) in production
    vector_backend: str = "numpy"      # "faiss" / "qdrant" with the matching extra installed
    default_top_k: int = 3


def get_settings() -> Settings:
    return Settings()
