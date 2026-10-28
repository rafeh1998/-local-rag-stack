"""Thin client for a local Ollama server."""

from typing import Protocol

import httpx


class LLMClient(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def generate(self, prompt: str) -> str: ...
    def healthy(self) -> bool: ...


class OllamaClient:
    def __init__(self, base_url: str, embed_model: str, chat_model: str, timeout: float) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout)
        self.embed_model = embed_model
        self.chat_model = chat_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        r = self._http.post("/api/embed", json={"model": self.embed_model, "input": texts})
        r.raise_for_status()
        return r.json()["embeddings"]

    def generate(self, prompt: str) -> str:
        r = self._http.post(
            "/api/generate",
            json={
                "model": self.chat_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    def healthy(self) -> bool:
        try:
            return self._http.get("/api/tags", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False
