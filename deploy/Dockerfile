# syntax=docker/dockerfile:1
# Multi-stage build of the reference service: installs the published rag-llm-infra
# package + this app, runs non-root. `docker build` is exercised in CI on every push.

FROM python:3.12-slim-bookworm AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /install /usr/local
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
