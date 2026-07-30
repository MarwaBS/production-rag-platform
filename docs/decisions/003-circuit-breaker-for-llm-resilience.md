# ADR 003 — Circuit breaker for LLM resilience

**Status:** Accepted — rewritten 2026-07-29 to describe the shipped mechanism.
The earlier text of this ADR described retry-with-backoff and a daily cost
ceiling that were never implemented in this repository.

## Context

The LLM provider is this service's one critical external dependency, and it is
called inside a synchronous route: an unbounded provider call holds a
threadpool worker until the client gives up. The provider can rate-limit,
return server errors, or hang.

## Decision

Every provider call is wrapped in three mechanisms, all enforced in
`app/main.py` and pinned by `tests/test_generation_safety.py`:

- **Timeout** (`APP_LLM_TIMEOUT_SECONDS`, default 5): the call runs on an
  abandonable thread and the route stops waiting at the deadline.
- **Bounded retry** (`APP_LLM_RETRY_ATTEMPTS`, default 1): an immediate
  provider failure is retried at most this many times per request. A timeout
  is **not** retried — that would hold the worker for a second full window.
- **Circuit breaker** (`APP_LLM_BREAKER_FAILURES`, default 3;
  `APP_LLM_BREAKER_RESET_SECONDS`, default 30): after that many *consecutive*
  failed requests the provider is not called at all; once the reset window
  passes, one request probes it again, and a success closes the breaker.

Whatever fails — timeout, exhausted retries, open breaker — the caller gets
the same documented response: HTTP 503 with `{"error": "llm_unavailable"}`.
Never an unbounded wait, never an unshaped 500.

## Scope: what a breaker is worth at one replica

This service is single-replica by design (the corpus lives in process), so the
breaker does **not** do what a breaker does in a fleet: there is no cascade to
arrest and no sibling to protect. Its two real jobs here are **latency
shaping** — fail fast instead of hanging a synchronous route's worker — and
**provider backoff** — an erroring API is probed once per reset window instead
of being hammered on every request. The reset window is also the only retry
backoff; with one immediate retry per request at one replica, jitter would be
machinery without a job.

## Trade-off

A timed-out provider call cannot be killed; the abandoned thread runs to
completion in the background. That is bounded in practice because the breaker
opens after a few consecutive timeouts and stops creating new ones.

## Alternatives considered

- **Retry only, no breaker:** every request during an outage still pays the
  full timeout, which is exactly the hang this exists to shape.
- **External breaker (Istio / Envoy):** moves resilience to the service mesh;
  sensible only once multiple services share the dependency and need one
  common policy.
- **Provider fallback chain:** route to a second vendor when the primary
  trips — complementary, not a replacement. A budget-aware `FallbackLLM`
  (`+ BudgetExhausted`) implementing this is open-sourced in
  [rag-llm-infra](https://github.com/MarwaBS/rag-llm-infra) but is not wired
  in here.

## When to reconsider

Wire the fallback chain when a second vendor exists; move the policy to the
mesh if this stops being single-replica or the dependency becomes shared.
