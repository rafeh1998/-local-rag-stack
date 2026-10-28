"""HTTP API: ingest documents and ask questions about them."""

import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from app.chunking import chunk_text
from app.config import Settings, settings
from app.llm import LLMClient, OllamaClient
from app.redact import redact
from app.store import VectorStore

REQUESTS = Counter("rag_requests_total", "API requests", ["endpoint", "status"])
REDACTIONS = Counter("rag_redactions_total", "Personal data items redacted", ["type"])
LATENCY = Histogram(
    "rag_ask_seconds", "End-to-end latency of /ask", buckets=(0.5, 1, 2, 5, 10, 30, 60)
)

PROMPT = """You are a helpful assistant for a company knowledge base.
Answer the question using ONLY the context below. If the answer is not in the
context, say you don't know. Keep the answer short and cite sources as [1], [2].

Context:
{context}

Question: {question}
Answer:"""


class IngestRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=200, examples=["handbook.md"])
    text: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    source: str
    chunks: int
    redacted: dict[str, int]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)


class Source(BaseModel):
    id: int
    source: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    redacted: dict[str, int]
    seconds: float


def create_app(
    llm: LLMClient | None = None,
    store: VectorStore | None = None,
    cfg: Settings = settings,
) -> FastAPI:
    llm = llm or OllamaClient(cfg.ollama_url, cfg.embed_model, cfg.chat_model, cfg.request_timeout)
    store = store if store is not None else VectorStore(cfg.data_dir)
    app = FastAPI(title="local-rag-stack", version="0.1.0")

    def _count_redactions(counts: dict[str, int]) -> None:
        for kind, n in counts.items():
            REDACTIONS.labels(kind).inc(n)

    @app.get("/health")
    def health() -> dict:
        ok = llm.healthy()
        REQUESTS.labels("health", "ok" if ok else "degraded").inc()
        return {"status": "ok" if ok else "degraded", "llm": ok, "chunks": len(store)}

    @app.get("/documents")
    def documents() -> dict:
        return {"sources": store.sources, "chunks": len(store)}

    @app.post("/documents", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        # Redact before chunking, so nothing personal is embedded or stored.
        clean = redact(req.text)
        _count_redactions(clean.counts)
        chunks = chunk_text(clean.text, cfg.chunk_size, cfg.chunk_overlap)
        try:
            vectors = llm.embed(chunks)
        except Exception as exc:
            REQUESTS.labels("ingest", "error").inc()
            raise HTTPException(502, f"embedding failed: {exc}") from exc
        # Only replace the old version once the new one is embedded.
        store.delete_source(req.source)
        store.add(req.source, chunks, vectors)
        REQUESTS.labels("ingest", "ok").inc()
        return IngestResponse(source=req.source, chunks=len(chunks), redacted=clean.counts)

    @app.delete("/documents/{source}")
    def delete(source: str) -> dict:
        removed = store.delete_source(source)
        if not removed:
            raise HTTPException(404, "unknown source")
        return {"source": source, "removed_chunks": removed}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        start = time.perf_counter()
        q = redact(req.question)
        _count_redactions(q.counts)
        if not len(store):
            REQUESTS.labels("ask", "empty").inc()
            raise HTTPException(409, "no documents ingested yet")
        try:
            hits = store.search(llm.embed([q.text])[0], req.top_k or cfg.top_k)
            context = "\n\n".join(f"[{i}] ({h.source})\n{h.text}" for i, h in enumerate(hits, 1))
            answer = llm.generate(PROMPT.format(context=context, question=q.text))
        except Exception as exc:
            REQUESTS.labels("ask", "error").inc()
            raise HTTPException(502, f"model call failed: {exc}") from exc
        elapsed = time.perf_counter() - start
        LATENCY.observe(elapsed)
        REQUESTS.labels("ask", "ok").inc()
        return AskResponse(
            answer=answer,
            sources=[
                Source(id=i, source=h.source, score=round(h.score, 4), excerpt=h.text[:200])
                for i, h in enumerate(hits, 1)
            ],
            redacted=q.counts,
            seconds=round(elapsed, 3),
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
