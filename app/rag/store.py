"""Lưu/tìm tài liệu trong PostgreSQL + pgvector (SQL thuần, không cần thư viện pgvector cho Python).

Tìm kiếm lai: vector (cosine, chỉ mục HNSW) + từ khóa (full-text trên văn bản đã bỏ dấu), gộp bằng Reciprocal Rank
Fusion. Cổng độ chính xác: chỉ nhận đoạn có độ tương đồng cosine >= ngưỡng; từ khóa chỉ để xếp hạng.
visibility luôn là điều kiện lọc trong SQL -> bot công khai không bao giờ đọc được tài liệu nội bộ.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from app.conversation.knowledge import terms
from app.observability.redaction import strip_diacritics
from app.storage.db import session_scope

PUBLIC = frozenset({"public"})
ALL = frozenset({"public", "internal"})
RRF_K = 60


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def normalize_for_search(t: str) -> str:
    return re.sub(r"\s+", " ", strip_diacritics(t.lower())).strip()


@dataclass(frozen=True)
class DocHit:
    chunk_id: int
    document_id: str
    title: str
    source: str
    content: str
    page_from: int | None
    page_to: int | None
    similarity: float
    score: float
    mime: str

    @property
    def citation_title(self) -> str:
        if self.mime == "application/pdf" and self.page_from:
            pages = (
                f"trang {self.page_from}"
                if self.page_from == self.page_to
                else f"trang {self.page_from}-{self.page_to}"
            )
            return f"{self.title} ({pages})"
        return self.title


async def find_document(sha256: str, visibility: str) -> dict[str, Any] | None:
    async with session_scope() as s:
        row = (
            (
                await s.execute(
                    text("SELECT id, title, chunks FROM kb_documents WHERE sha256 = :h AND visibility = :v"),
                    {"h": sha256, "v": visibility},
                )
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


async def insert_document(
    *,
    title: str,
    source: str,
    original_name: str,
    mime: str,
    sha256: str,
    visibility: str,
    pages: int,
    ocr_pages: int,
    embedding_model: str,
    chunks: list[dict[str, Any]],
    created_by: str,
    metadata: dict[str, Any] | None = None,
    replace: bool = False,
) -> uuid.UUID:
    doc_id = uuid.uuid4()
    async with session_scope() as s:
        if replace:
            await s.execute(
                text("DELETE FROM kb_documents WHERE sha256 = :h AND visibility = :v"),
                {"h": sha256, "v": visibility},
            )
        await s.execute(
            text(
                "INSERT INTO kb_documents (id, title, source, original_name, mime, sha256, visibility, pages, "
                "ocr_pages, chunks, embedding_model, metadata, created_by) VALUES (:id, :title, :source, :orig, :mime, "
                ":sha, :vis, :pages, :ocr, :n, :model, CAST(:meta AS jsonb), :by)"
            ),
            {
                "id": doc_id,
                "title": title,
                "source": source,
                "orig": original_name,
                "mime": mime,
                "sha": sha256,
                "vis": visibility,
                "pages": pages,
                "ocr": ocr_pages,
                "n": len(chunks),
                "model": embedding_model,
                "meta": json.dumps(metadata or {}, ensure_ascii=False),
                "by": created_by,
            },
        )
        for c in chunks:
            await s.execute(
                text(
                    "INSERT INTO kb_chunks (document_id, chunk_index, page_from, page_to, content, content_norm, "
                    "embedding) VALUES (:d, :i, :pf, :pt, :c, :cn, CAST(:e AS vector))"
                ),
                {
                    "d": doc_id,
                    "i": c["index"],
                    "pf": c["page_from"],
                    "pt": c["page_to"],
                    "c": c["text"],
                    "cn": normalize_for_search(c["text"]),
                    "e": vec_literal(c["embedding"]),
                },
            )
        await s.execute(
            text(
                "INSERT INTO audit_log (actor, action, target, details) VALUES (:a, 'doc_ingest', :t, CAST(:d AS jsonb))"
            ),
            {
                "a": created_by,
                "t": str(doc_id),
                "d": json.dumps(
                    {"title": title, "visibility": visibility, "chunks": len(chunks)}, ensure_ascii=False
                ),
            },
        )
    return doc_id


async def list_documents() -> list[dict[str, Any]]:
    async with session_scope() as s:
        rows = (
            (
                await s.execute(
                    text(
                        "SELECT id, title, source, original_name, visibility, pages, ocr_pages, chunks, embedding_model, "
                        "created_at FROM kb_documents ORDER BY created_at DESC"
                    )
                )
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


async def delete_document(doc_id: uuid.UUID, actor: str) -> bool:
    async with session_scope() as s:
        n = (await s.execute(text("DELETE FROM kb_documents WHERE id = :id"), {"id": doc_id})).rowcount
        if n:
            await s.execute(
                text("INSERT INTO audit_log (actor, action, target) VALUES (:a, 'doc_delete', :t)"),
                {"a": actor, "t": str(doc_id)},
            )
    return bool(n)


def _tsquery(query: str) -> str:
    words = {t for t in terms(query) if " " not in t and re.fullmatch(r"[a-z0-9]+", t)}
    return " | ".join(sorted(words))


async def search(
    query: str,
    query_vec: list[float],
    *,
    visibility: frozenset[str],
    embedding_model: str,
    top_k: int,
    min_similarity: float,
    candidates: int = 20,
) -> list[DocHit]:
    params = {"q": vec_literal(query_vec), "vis": list(visibility), "model": embedding_model, "k": candidates}
    async with session_scope() as s:
        await s.execute(text("SET LOCAL hnsw.ef_search = 100"))
        await s.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        vec_rows = (
            (
                await s.execute(
                    text(
                        "SELECT c.id, c.document_id, c.content, c.page_from, c.page_to, d.title, d.source, d.mime, "
                        "1 - (c.embedding <=> CAST(:q AS vector)) AS sim FROM kb_chunks c "
                        "JOIN kb_documents d ON d.id = c.document_id "
                        "WHERE d.visibility = ANY(:vis) AND d.embedding_model = :model "
                        "ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :k"
                    ),
                    params,
                )
            )
            .mappings()
            .all()
        )
        kw_rank: dict[int, int] = {}
        tsq = _tsquery(query)
        if tsq:
            kw_rows = (
                await s.execute(
                    text(
                        "SELECT c.id FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id "
                        "WHERE d.visibility = ANY(:vis) AND d.embedding_model = :model "
                        "AND to_tsvector('simple', c.content_norm) @@ to_tsquery('simple', :tsq) "
                        "ORDER BY ts_rank(to_tsvector('simple', c.content_norm), to_tsquery('simple', :tsq)) DESC "
                        "LIMIT :k"
                    ),
                    {**params, "tsq": tsq},
                )
            ).all()
            kw_rank = {r[0]: i for i, r in enumerate(kw_rows)}
    hits = []
    for i, r in enumerate(vec_rows):
        sim = float(r["sim"])
        if sim < min_similarity:
            continue
        score = 1 / (RRF_K + i) + (1 / (RRF_K + kw_rank[r["id"]]) if r["id"] in kw_rank else 0.0)
        hits.append(
            DocHit(
                r["id"],
                str(r["document_id"]),
                r["title"],
                r["source"],
                r["content"],
                r["page_from"],
                r["page_to"],
                round(sim, 4),
                score,
                r["mime"],
            )
        )
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
