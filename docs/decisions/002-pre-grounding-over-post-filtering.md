# ADR 002 — Pre-grounding over post-filtering

**Status:** Accepted

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
is well worth the faithfulness guarantee. A lightweight post-generation check still runs as a
safety net.
