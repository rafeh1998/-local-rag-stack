"""A small persistent vector store backed by NumPy.

For a few thousand chunks a brute-force cosine search is fast enough and
avoids running a separate database. The interface is small on purpose so
it can be swapped for Qdrant or pgvector later.
"""

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    source: str
    text: str


@dataclass
class Hit:
    source: str
    text: str
    score: float


class VectorStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir = data_dir
        self._lock = threading.Lock()
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        if data_dir is not None:
            self._load()

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def sources(self) -> list[str]:
        return sorted({c.source for c in self._chunks})

    def add(self, source: str, texts: list[str], vectors: list[list[float]]) -> int:
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have the same length")
        if not texts:
            return 0
        new = _normalise(np.asarray(vectors, dtype=np.float32))
        with self._lock:
            if self._vectors is not None and self._vectors.shape[1] != new.shape[1]:
                raise ValueError(
                    "embedding dimension changed; clear the index after switching models"
                )
            self._vectors = new if self._vectors is None else np.vstack([self._vectors, new])
            self._chunks.extend(Chunk(source, t) for t in texts)
            self._save()
        return len(texts)

    def delete_source(self, source: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.source != source]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._chunks = [self._chunks[i] for i in keep]
                self._vectors = self._vectors[keep] if keep else None
                self._save()
        return removed

    def search(self, query: list[float], k: int = 4) -> list[Hit]:
        with self._lock:
            if self._vectors is None or not self._chunks:
                return []
            q = _normalise(np.asarray([query], dtype=np.float32))[0]
            scores = self._vectors @ q
            top = np.argsort(-scores)[:k]
            return [
                Hit(self._chunks[i].source, self._chunks[i].text, float(scores[i])) for i in top
            ]

    def _save(self) -> None:
        if self._dir is None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        meta = [asdict(c) for c in self._chunks]
        (self._dir / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False))
        if self._vectors is None:
            (self._dir / "vectors.npy").unlink(missing_ok=True)
        else:
            np.save(self._dir / "vectors.npy", self._vectors)

    def _load(self) -> None:
        meta_path = self._dir / "chunks.json"
        vec_path = self._dir / "vectors.npy"
        if meta_path.exists() and vec_path.exists():
            self._chunks = [Chunk(**c) for c in json.loads(meta_path.read_text())]
            self._vectors = np.load(vec_path)


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
