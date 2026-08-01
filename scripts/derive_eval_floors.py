"""Producer for the retrieval eval floors (eval_floors_derivation.json).

Baseline: the semantic backend on the zero-overlap paraphrase set — the one
instrument here that is not saturated (the literal set is solvable by token
matching on either backend). Floor rule: measured minus half a miss-quantum
(0.5/n); the pipeline is deterministic, so the first additional recall miss
trips it. MRR moves in finer steps than that margin, so recall is the binding
metric and the MRR floor guards larger collapses.

Needs the `semantic` extra. Run:
    python scripts/derive_eval_floors.py            # rewrite the committed file
    python scripts/derive_eval_floors.py --print    # print, no write (CI gate)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

os.environ["APP_EMBEDDING_BACKEND"] = "semantic"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import app.embedder as embedder  # noqa: E402
from evals.harness import CORPUS, PARAPHRASE_QUERIES, QUERIES, evaluate  # noqa: E402


def _measure(queries) -> dict[str, float]:
    top1, top3 = evaluate(k=1, queries=queries), evaluate(k=3, queries=queries)
    return {
        "recall_at_1": round(top1.recall_at_k, 4),
        "recall_at_3": round(top3.recall_at_k, 4),
        "mrr": round(top3.mrr, 4),
    }


def derive() -> dict:
    paraphrased = tuple((q, CORPUS[gold]) for q, gold in PARAPHRASE_QUERIES)
    semantic = {"literal": _measure(QUERIES), "paraphrase": _measure(paraphrased)}
    # Re-measure on the default hash backend for the recorded contrast — the
    # gap between these two paraphrase rows is the entire case for the
    # semantic extra.
    embedder._settings.embedding_backend = "hash"
    try:
        hash_measured = {
            "literal": _measure(QUERIES),
            "paraphrase": _measure(paraphrased),
        }
    finally:
        embedder._settings.embedding_backend = "semantic"
    n = len(paraphrased)
    baseline = semantic["paraphrase"]
    # The default path is gated too, and against its own measurement: floors
    # derived from the semantic instrument are far below what it actually does.
    default_baseline = hash_measured["literal"]
    default_n = len(QUERIES)
    return {
        "baseline": "semantic backend on the zero-overlap paraphrase set",
        "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
        "n_paraphrase_queries": n,
        "corpus_documents": len(CORPUS),
        "floor_rule": "measured minus half a miss-quantum (0.5/n)",
        "measured": {
            "semantic_paraphrase": baseline,
            "semantic_literal": semantic["literal"],
            "hash_paraphrase": hash_measured["paraphrase"],
            "hash_literal": hash_measured["literal"],
        },
        "derived_floors": {
            metric: round(value - 0.5 / n, 4) for metric, value in baseline.items()
        },
        "n_literal_queries": default_n,
        "derived_floors_default_path": {
            metric: round(value - 0.5 / default_n, 4)
            for metric, value in default_baseline.items()
        },
    }


def main() -> None:
    payload = json.dumps(derive(), indent=2) + "\n"
    if "--print" in sys.argv:
        sys.stdout.write(payload)
        return
    out = pathlib.Path(__file__).resolve().parent.parent / "eval_floors_derivation.json"
    out.write_text(payload, encoding="utf-8")
    sys.stdout.write(f"wrote {out}\n")


if __name__ == "__main__":
    main()
