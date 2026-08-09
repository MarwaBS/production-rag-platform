# ADR 002. Pre-grounding over post-filtering

**Status:** Accepted

**Scope:** the pre-grounding decision is implemented in this repo; `/query` retrieves first
and builds the prompt from the retrieved windows alone. The post-generation check named under
Trade-off is **full system only**; nothing in `app/` inspects an answer after the model returns it.

## Context
Generated content must stay faithful to the user's own input.

## Decision
Retrieve supporting evidence and constrain the prompt **before** generation, rather than
generating freely and stripping unsupported content afterward.

## Why
Prevention beats detection. Post-hoc removal of unsupported claims tends to break coherence,
because the model may have woven them through the text. Constraining the prompt up front produces
faithful output on the first pass and yields a clean audit trail of which evidence informed each
section.

## Trade-off
Adds a retrieval step before each generation. The latency is small relative to the model call and
is well worth the faithfulness guarantee.

In the full system a lightweight post-generation check runs behind this as a safety net. This
repo ships no such check: what it enforces instead is that generation cannot proceed without
evidence; a query that retrieves nothing scoring above zero is answered `grounded: false`
with no model call at all.
