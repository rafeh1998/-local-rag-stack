# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A self-hosted RAG question-answering service. FastAPI app in `app/`, local
models served by Ollama, a NumPy vector index on disk, Prometheus metrics.
Personal data is redacted **before** chunking/embedding, so it never reaches
the index or the model.

## Commands

```bash
make dev     # create .venv and install requirements-dev.txt
make test    # pytest -q (no model server needed)
make lint    # ruff check + ruff format --check
make up      # docker compose: Ollama + API + Prometheus
make ingest  # load sample_docs/handbook.md into a running stack
make ask Q="..."
```

Run a single test: `.venv/bin/pytest tests/test_chunking.py -q -k overlap`.
On Windows the venv binaries are under `.venv/Scripts/` instead of `.venv/bin/`.

CI (`.github/workflows/ci.yml`) runs ruff check, ruff format --check and pytest
on Python 3.12, then builds the Docker image. Keep both lint commands passing.

## Layout

- `app/main.py` — `create_app(llm, store, cfg)` factory and all endpoints
  (`/health`, `/documents`, `/ask`, `/metrics`). Module-level `app` is for uvicorn.
- `app/redact.py` — regex redaction to `[EMAIL]`, `[PHONE]`, `[IBAN]`. Pattern
  order matters (IBAN before PHONE).
- `app/chunking.py` — paragraph-aware chunking with character overlap.
- `app/store.py` — `VectorStore`: brute-force cosine search, persisted to `DATA_DIR`.
- `app/llm.py` — `LLMClient` protocol and `OllamaClient` (httpx).
- `app/config.py` — frozen `Settings` read from env vars (`OLLAMA_URL`,
  `CHAT_MODEL`, `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, ...).
- `tests/conftest.py` — `FakeLLM` (hashed bag-of-words embeddings, records every
  prompt) and the `client` fixture.

## Conventions

- Tests must never require Ollama or network access; use `FakeLLM` and inject it
  via `create_app(llm=..., store=...)`.
- Any change touching ingest or ask must keep the privacy guarantee: redact first,
  and add/keep a test that inspects `llm.prompts` for leaked personal data.
- Re-ingest only deletes the old source after the new embedding succeeds — keep
  that ordering.
- Ruff: line length 100, rules E, F, I, B, UP. Type hints on public functions.
- Keep dependencies minimal and pinned in `requirements*.txt`.
- Update the README (API table, test count) when behaviour changes.
