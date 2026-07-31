"""Sliding-window splitter. Windows advance by (max_chars - overlap_chars), so
any span no longer than the overlap survives whole in at least one window —
the overlap width is derived from measured sentence lengths by
scripts/derive_chunking.py, not chosen."""

from __future__ import annotations

from typing import List


def chunk(text: str, *, max_chars: int, overlap_chars: int) -> List[str]:
    """Deterministic character windows; the full input is always covered."""
    # One condition covers both bounds: it implies a positive window and a
    # positive stride, without which the loop below never advances.
    if not 0 <= overlap_chars < max_chars:
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be in [0, max_chars) "
            f"with max_chars ({max_chars}) positive"
        )
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
