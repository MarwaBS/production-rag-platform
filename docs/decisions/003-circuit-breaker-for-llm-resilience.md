# ADR 003 — Circuit breaker for LLM resilience

**Status:** Accepted

## Context
The LLM provider is the platform's most critical external dependency and can rate-limit, return
server errors, or time out.

## Decision
Wrap provider calls in retry-with-backoff plus a daily cost ceiling that degrades safely under
failure, rather than letting failures cascade through every request.

## Why
Failing fast and shedding load protects request latency, connection pools, and spend during a
provider incident, and recovers automatically when the provider does. The cost ceiling prevents a
runaway-spend failure mode independent of provider health.

## Trade-off
A small amount of resilience machinery to own and test — justified by the blast radius of an
unprotected single external dependency.

## Alternatives considered
- **Retry only, no breaker:** every request still pays the full timeout during an outage, and with
  no spend ceiling a stuck provider can run up cost.
- **External breaker (Istio / Envoy):** moves resilience to the service mesh, but can't see
  application-level signals such as token spend.
- **Provider fallback chain:** route to a second vendor when the primary trips — complementary, not
  a replacement. A budget-aware `FallbackLLM` (`+ BudgetExhausted`) implementing exactly this is
  open-sourced in [rag-llm-infra](https://github.com/MarwaBS/rag-llm-infra).

## When to reconsider
Add the multi-provider fallback chain above once a second vendor is wired in; relocate the breaker
to the mesh only if multiple services share the same external dependency and need one common policy.
