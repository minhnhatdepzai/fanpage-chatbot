"""Tìm đoạn tài liệu liên quan (pgvector) và trả về dạng KnowledgeEntry để dùng chung khối "Nguồn tham khảo"."""

from __future__ import annotations

from app.config import Settings
from app.conversation.knowledge import KnowledgeEntry
from app.rag import store
from app.rag.embedder import Embedder


class DocSearch:
    def __init__(self, settings: Settings, embedder: Embedder) -> None:
        self.settings = settings
        self.embedder = embedder

    async def search(
        self, query: str, *, visibility: frozenset[str], top_k: int | None = None
    ) -> list[KnowledgeEntry]:
        vec = (await self.embedder.embed([query]))[0]
        hits = await store.search(
            query,
            vec,
            visibility=visibility,
            embedding_model=self.embedder.model,
            top_k=top_k or self.settings.doc_top_k,
            min_similarity=self.settings.doc_min_similarity,
        )
        return [
            KnowledgeEntry(f"doc:{h.document_id}:{h.chunk_id}", h.citation_title, h.source, h.content)
            for h in hits
        ]
