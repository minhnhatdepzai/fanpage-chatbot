"""Schema PostgreSQL (SQLAlchemy 2.0). Migration tương ứng nằm trong ``migrations/``."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# ----------------------------------------------------------------------------- hằng số trạng thái
class Role:
    user = "user"
    assistant = "assistant"
    human_agent = "human_agent"  # quản trị viên trả lời từ hộp thư Page


class InStatus:
    pending = "pending"  # chờ worker xử lý
    consumed = "consumed"  # đã thuộc về một turn
    skipped = "skipped"  # không cần trả lời (vd. handoff)


class OutStatus:
    pending = "pending"
    sending = "sending"  # đã ghi trước khi gọi Send API
    sent = "sent"
    uncertain = "uncertain"  # timeout sau khi đã gửi request -> không biết Meta nhận chưa
    failed = "failed"
    cancelled = "cancelled"  # bị chặn bởi kiểm tra lại trước khi gửi
    dry_run = "dry_run"  # SEND_MODE không cho phép gửi


class TurnStatus:
    running = "running"
    completed = "completed"
    failed = "failed"


class CandidateStatus:
    pending_review = "pending_review"
    approved = "approved"
    rejected = "rejected"


# ----------------------------------------------------------------------------- bảng
class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint("page_id", "psid", name="uq_conversations_page_psid"),
        Index(
            "ix_conversations_next_run_at",
            "next_run_at",
            postgresql_where=sql_text("next_run_at IS NOT NULL"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    page_id: Mapped[str] = mapped_column(String(64), nullable=False)
    psid: Mapped[str] = mapped_column(String(64), nullable=False)
    user_ref: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # cửa sổ nhắn tin: lần tương tác hợp lệ gần nhất của người dùng
    last_user_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_bot_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_disclosure_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_fallback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # human handoff
    handoff_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_text("false"))
    handoff_reason: Mapped[str | None] = mapped_column(String(64))
    handoff_by: Mapped[str | None] = mapped_column(String(64))
    handoff_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    bot_resumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # bộ nhớ ngắn hạn
    summary: Mapped[str | None] = mapped_column(Text)
    summary_upto_seq: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default=sql_text("0"))
    summary_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # lập lịch xử lý (hàng đợi trên PostgreSQL)
    next_run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_pending_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sql_text("0"))
    last_error: Mapped[str | None] = mapped_column(Text)


class Message(Base):
    """Bản ghi hội thoại: tin người dùng (inbound) và tin bot/quản trị viên (outbound)."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("uq_messages_mid", "mid", unique=True, postgresql_where=sql_text("mid IS NOT NULL")),
        Index(
            "uq_messages_idempotency_key",
            "idempotency_key",
            unique=True,
            postgresql_where=sql_text("idempotency_key IS NOT NULL"),
        ),
        Index("ix_messages_conv_seq", "conversation_id", "id"),
        Index("ix_messages_conv_status", "conversation_id", "role", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    kind: Mapped[str] = mapped_column(String(24), nullable=False, server_default=sql_text("'text'"))
    text: Mapped[str | None] = mapped_column(Text)
    attachment_types: Mapped[list[str] | None] = mapped_column(JSONB)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    mid: Mapped[str | None] = mapped_column(String(255))
    event_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="SET NULL"), index=True
    )
    # outbound
    chunk_index: Mapped[int | None] = mapped_column(Integer)
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    allow_during_handoff: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_text("false")
    )
    send_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sql_text("0"))
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    echo_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class Turn(Base):
    """Một lượt xử lý của worker (có thể gom nhiều tin người dùng gửi liên tiếp)."""

    __tablename__ = "turns"
    __table_args__ = (Index("ix_turns_conv_created", "conversation_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=sql_text("0"))
    mode: Mapped[str | None] = mapped_column(String(32))
    outcome: Mapped[str | None] = mapped_column(String(96))
    latest_inbound_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    graph_thread_id: Mapped[str | None] = mapped_column(String(128))
    checkpoint_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_text("false")
    )
    # phiên bản & đo lường
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    provider: Mapped[str | None] = mapped_column(String(32))
    model_name: Mapped[str | None] = mapped_column(String(160))
    adapter_version: Mapped[str | None] = mapped_column(String(128))
    model_latency_ms: Mapped[int | None] = mapped_column(Integer)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    check_flags: Mapped[list[str] | None] = mapped_column(JSONB)
    draft_text: Mapped[str | None] = mapped_column(Text)  # câu trả lời model sinh ra (trước khi chia tin)
    error: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[str | None] = mapped_column(String(64))


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    turn_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rating: Mapped[str | None] = mapped_column(String(8))  # good | bad
    corrected_reply: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TrainingCandidate(Base):
    """Mẫu chờ duyệt để đưa vào dữ liệu huấn luyện (đã che PII khi tạo)."""

    __tablename__ = "training_candidates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    turn_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("turns.id", ondelete="SET NULL"), index=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="SET NULL"), index=True
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # admin_correction | admin_good_rating
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    sample: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    rejected_reply: Mapped[str | None] = mapped_column(Text)  # câu trả lời gốc bị sửa (cho DPO)
    pii_redacted: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=sql_text("true"))
    pii_detected_before_redaction: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_text("false")
    )
    usage_rights_confirmed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=sql_text("false")
    )
    reviewer: Mapped[str | None] = mapped_column(String(64))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_notes: Mapped[str | None] = mapped_column(Text)
    exported_versions: Mapped[list[str] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DatasetExport(Base):
    """Lineage: phiên bản dữ liệu huấn luyện được xuất từ các mẫu đã duyệt."""

    __tablename__ = "dataset_exports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    num_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    candidate_ids: Mapped[list[int]] = mapped_column(JSONB, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, server_default=sql_text("'sft'"))
    notes: Mapped[str | None] = mapped_column(Text)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    actor: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    target: Mapped[str | None] = mapped_column(String(128))
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class WorkerHeartbeat(Base):
    __tablename__ = "worker_heartbeats"

    worker_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    info: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
