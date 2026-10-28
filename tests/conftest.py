"""Shared test fixtures. Tests never need a running model server."""

import hashlib
import re

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.store import VectorStore

DIM = 64


class FakeLLM:
    """Deterministic stand-in for Ollama.

    Embeddings are a hashed bag of words, so texts that share words end up
    close together, which is enough to test retrieval end to end. Every
    prompt is recorded so tests can check what would reach the model.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.up = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * DIM
            for word in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
            out.append(v)
        return out

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Fake answer [1]"

    def healthy(self) -> bool:
        return self.up


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def client(llm, tmp_path) -> TestClient:
    cfg = Settings(data_dir=tmp_path, chunk_size=200, chunk_overlap=20, top_k=2)
    return TestClient(create_app(llm=llm, store=VectorStore(tmp_path), cfg=cfg))
