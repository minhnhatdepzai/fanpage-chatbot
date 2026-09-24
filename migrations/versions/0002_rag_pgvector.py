"""RAG: pgvector + tài liệu nạp từ PDF/Word (kb_documents, kb_chunks)

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-24 01:40:00

Embedding 1024 chiều (AITeamVN/Vietnamese_Embedding, fine-tune từ BAAI/bge-m3). Đổi model khác số chiều -> migration mới.
visibility: public (bot Messenger/website được dùng) | internal (chỉ trợ lý nội bộ có xác thực).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE kb_documents (
            id uuid PRIMARY KEY,
            title text NOT NULL,
            source text NOT NULL,
            original_name text NOT NULL,
            mime varchar(128) NOT NULL,
            sha256 char(64) NOT NULL,
            visibility varchar(16) NOT NULL CHECK (visibility IN ('public', 'internal')),
            pages integer NOT NULL DEFAULT 0,
            ocr_pages integer NOT NULL DEFAULT 0,
            chunks integer NOT NULL DEFAULT 0,
            embedding_model varchar(160) NOT NULL,
            metadata jsonb,
            created_by varchar(64) NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_kb_documents_sha_vis UNIQUE (sha256, visibility)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE kb_chunks (
            id bigserial PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
            chunk_index integer NOT NULL,
            page_from integer,
            page_to integer,
            content text NOT NULL,
            content_norm text NOT NULL,
            embedding vector(1024) NOT NULL,
            CONSTRAINT uq_kb_chunks_doc_idx UNIQUE (document_id, chunk_index)
        )
        """
    )
    op.execute("CREATE INDEX ix_kb_chunks_embedding ON kb_chunks USING hnsw (embedding vector_cosine_ops)")
    op.execute("CREATE INDEX ix_kb_chunks_fts ON kb_chunks USING gin (to_tsvector('simple', content_norm))")
    op.execute("CREATE INDEX ix_kb_chunks_document ON kb_chunks (document_id)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS kb_chunks")
    op.execute("DROP TABLE IF EXISTS kb_documents")
