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

# What HTTP trims around a header value, so the comparison in app.main sees the
# same bytes the client's header does.
_HEADER_SPACE = " \t\r\n"


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
    # "hash" matches words (deterministic, no download); "semantic" matches
    # meaning via sentence-transformers and needs the extra.
    embedding_backend: Literal["hash", "semantic"] = "hash"
    default_top_k: int = Field(default=3, ge=1)
    # Input-contract bounds sized to the reference pod's 512Mi limit — judgement
    # calls, not derivations, and chunking multiplies them: vectors scale with
    # windows. The tests pin that each bound exists and bites, not its value.
    max_documents: int = Field(default=10_000, ge=1)
    max_document_chars: int = Field(default=8_000, ge=1)
    max_query_chars: int = Field(default=1_000, ge=1)
    max_top_k: int = Field(default=50, ge=1)
    max_request_bytes: int = Field(default=10_485_760, ge=1)  # 10 MiB
    # Both come from scripts/derive_chunking.py, which a gate re-runs against
    # these defaults: the overlap is measured over the corpus, the window is
    # arithmetic on the embedding model's own stated limit.
    max_chunk_chars: int = Field(default=254, ge=1)
    chunk_overlap_chars: int = Field(default=83, ge=1)
    # LLM resilience: timeout, bounded retry, consecutive-failure breaker.
    # Judgement calls, APP_-overridable; the tests pin the behaviour of each
    # knob, not its value.
    llm_timeout_seconds: float = Field(default=5.0, gt=0)
    llm_retry_attempts: int = Field(default=1, ge=0)
    llm_breaker_failures: int = Field(default=3, ge=1)
    llm_breaker_reset_seconds: float = Field(default=30.0, gt=0)
    # When set, both /index and /query require a matching X-API-Key.
    api_key: str = ""

    @model_validator(mode="after")
    def _the_key_is_taken_without_the_space_around_it(self) -> Settings:
        # The value arrives from a Secret or a .env line, where a trailing
        # newline is invisible; compared raw it matches nothing a client can
        # send, so every request 401s against a key correct in both places.
        #
        # Refused rather than stripped when there is nothing else to it: empty
        # is how this field says "no auth configured", so stripping a key of
        # only spaces down to empty would OPEN the data-plane instead of
        # closing it. Only the four bytes HTTP itself trims around a header
        # value are removed, so a key whose first character is some other
        # space stays a different key from one without it.
        if self.api_key and not self.api_key.strip(_HEADER_SPACE):
            raise ValueError(
                "APP_API_KEY is only whitespace: set a real key, or leave it "
                "unset to run without authentication"
            )
        self.api_key = self.api_key.strip(_HEADER_SPACE)
        return self

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

    @model_validator(mode="after")
    def _the_overlap_must_fit_inside_the_window(self) -> Settings:
        # Independent bounds, so this pairing is reachable from the environment.
        # It leaves the splitter unable to advance, which is a boot-time death.
        if self.chunk_overlap_chars >= self.max_chunk_chars:
            raise ValueError(
                f"APP_CHUNK_OVERLAP_CHARS ({self.chunk_overlap_chars}) must be "
                f"smaller than APP_MAX_CHUNK_CHARS ({self.max_chunk_chars})"
            )
        return self


def get_settings() -> Settings:
    return Settings()
