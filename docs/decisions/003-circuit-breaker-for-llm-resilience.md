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
