"""Sliding-window document splitter with overlap.

The overlap is the load-bearing property: a window boundary placed inside the
sentence that carries the answer would otherwise leave neither window
retrievable. Consecutive windows advance by (max_chars - overlap_chars), so any
span no longer than the overlap fits whole in at least one window, wherever the
boundary falls. The overlap width is therefore derived from measured sentence
lengths (scripts/derive_chunking.py), not chosen by taste.
"""

from __future__ import annotations

from typing import List


def chunk(text: str, *, max_chars: int, overlap_chars: int) -> List[str]:
    """Split `text` into windows of at most `max_chars`, consecutive windows
    sharing `overlap_chars` characters. Character-exact and deterministic; the
    full input is always covered (a run longer than the window is split, never
    dropped)."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be in [0, max_chars)")
    if len(text) <= max_chars:
        return [text]
    stride = max_chars - overlap_chars
    pieces: List[str] = []
    start = 0
    while True:
        pieces.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            return pieces
        start += stride
