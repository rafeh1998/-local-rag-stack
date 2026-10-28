"""Runtime configuration, read from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    chat_model: str = os.getenv("CHAT_MODEL", "qwen2.5:0.5b")
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "120"))


settings = Settings()
