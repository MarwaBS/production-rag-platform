# ADR 004 — Vendor-neutral LLM protocol

**Status:** Accepted

**Scope:** implemented in this repo, through the published `rag-llm-infra` package.
`APP_LLM_BACKEND` selects the backend; the default is the deterministic mock, so the
service runs with no provider credential. The multi-provider fallback named under Why
exists in that package but is **not** wired in here.

## Context
Calling a single provider's SDK directly throughout the codebase creates vendor lock-in and
brittle test setups (tests that patch SDK internals by dotted path).

## Decision
Define a minimal `LLMProtocol` interface with swappable backends — the production provider, a
contract stub for a second vendor, and a deterministic mock for tests — selected via configuration.

## Why
The model vendor becomes a configuration choice rather than a code dependency; tests run against a
deterministic mock instead of patching SDK internals; and a future multi-provider fallback (route
to a second vendor when the primary budget is exhausted) becomes a small, localized change.

## Trade-off
One abstraction layer to maintain. The reference implementation — three real backends plus the
factory and conformance tests — is open-sourced at
[rag-llm-infra](https://github.com/MarwaBS/rag-llm-infra).
