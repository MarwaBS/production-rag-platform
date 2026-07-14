"""`python -m evals` — print the retrieval eval summary (the CI floor lives in
tests/test_eval.py)."""

from .harness import evaluate

if __name__ == "__main__":
    print(evaluate().summary())
