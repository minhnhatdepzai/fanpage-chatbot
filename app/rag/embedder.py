"""Client embedding: gọi endpoint /embeddings (OpenAI-compatible) của model server. Có bản giả lập cho test."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from app.config import Settings
from app.observability.redaction import strip_diacritics


class EmbeddingError(RuntimeError):
    pass


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HttpEmbedder:
    def __init__(self, settings: Settings, *, batch_size: int = 32) -> None:
        self.model = settings.embedding_model_id
        self.dim = settings.embedding_dim
        self._url = settings.embedding_base_url.rstrip("/") + "/embeddings"
        self._key = settings.model_server_api_key.get_secret_value()
        self._timeout = settings.embedding_timeout_seconds
        self._batch = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for i in range(0, len(texts), self._batch):
                try:
                    r = await client.post(
                        self._url,
                        json={"model": self.model, "input": texts[i : i + self._batch]},
                        headers=headers,
                    )
                    r.raise_for_status()
                    data = sorted(r.json()["data"], key=lambda d: d["index"])
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    raise EmbeddingError(type(exc).__name__) from exc
                out.extend(d["embedding"] for d in data)
        if any(len(v) != self.dim for v in out):
            raise EmbeddingError(f"số chiều embedding khác cấu hình {self.dim}")
        return out


class HashingEmbedder:
    """Embedding giả lập xác định (túi từ đã bỏ dấu, băm vào ``dim`` chiều) - CHỈ dùng cho test."""

    def __init__(self, dim: int = 1024, model: str = "test-hashing") -> None:
        self.dim = dim
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", strip_diacritics(t.lower())):
                v[int(hashlib.md5(w.encode(), usedforsecurity=False).hexdigest(), 16) % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / n for x in v])
        return vecs
