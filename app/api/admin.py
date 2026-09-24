"""Admin API tối thiểu (Bearer ADMIN_API_KEY). Chưa cấu hình khóa -> toàn bộ /admin bị khóa."""

from __future__ import annotations

import secrets
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.admin import service
from app.config import get_settings

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(
    authorization: Annotated[str | None, Header()] = None,
    x_admin_user: Annotated[str | None, Header()] = None,
) -> str:
    key = get_settings().admin_api_key.get_secret_value()
    if not key:
        raise HTTPException(status_code=503, detail="admin API disabled (ADMIN_API_KEY chưa cấu hình)")
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or not secrets.compare_digest(token.encode(), key.encode()):
        raise HTTPException(status_code=401, detail="unauthorized", headers={"WWW-Authenticate": "Bearer"})
    return (x_admin_user or "admin")[:64]


Admin = Annotated[str, Depends(require_admin)]


def _wrap(exc: Exception) -> HTTPException:
    if isinstance(exc, service.NotFound):
        return HTTPException(404, str(exc))
    if isinstance(exc, service.InvalidAction):
        return HTTPException(400, str(exc))
    raise exc


class RatingIn(BaseModel):
    rating: str = Field(pattern="^(good|bad)$")
    notes: str | None = None


class CorrectionIn(BaseModel):
    corrected_reply: str = Field(min_length=1, max_length=4000)
    notes: str | None = None


class ReviewIn(BaseModel):
    approve: bool
    usage_rights_confirmed: bool = False
    notes: str | None = None
    sample: dict[str, Any] | None = None


class HandoffIn(BaseModel):
    active: bool
    reason: str | None = None


class ExportIn(BaseModel):
    kind: str = Field(default="sft", pattern="^(sft|dpo)$")


@router.get("/turns")
async def turns(admin: Admin, status: str = "unrated", limit: int = 20) -> list[dict[str, Any]]:
    try:
        return await service.list_turns(status=status, limit=min(limit, 200))
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@router.post("/turns/{turn_id}/rating")
async def rate(turn_id: uuid.UUID, body: RatingIn, admin: Admin, request: Request) -> dict[str, Any]:
    try:
        return await service.rate_turn(turn_id, body.rating, admin, body.notes, request.app.state.tracer)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@router.post("/turns/{turn_id}/correction")
async def correct(turn_id: uuid.UUID, body: CorrectionIn, admin: Admin, request: Request) -> dict[str, Any]:
    try:
        return await service.correct_turn(
            turn_id, body.corrected_reply, admin, body.notes, request.app.state.tracer
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@router.get("/candidates")
async def candidates(admin: Admin, status: str = "pending_review", limit: int = 50) -> list[dict[str, Any]]:
    return await service.list_candidates(status, min(limit, 500))


@router.post("/candidates/{cand_id}/review")
async def review(cand_id: int, body: ReviewIn, admin: Admin) -> dict[str, Any]:
    try:
        return await service.review_candidate(
            cand_id,
            approve=body.approve,
            reviewer=admin,
            usage_rights_confirmed=body.usage_rights_confirmed,
            notes=body.notes,
            edited_sample=body.sample,
        )
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@router.post("/exports")
async def export(body: ExportIn, admin: Admin) -> dict[str, Any]:
    try:
        return await service.export_approved(reviewer=admin, kind=body.kind)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@router.get("/conversations")
async def conversations(admin: Admin, handoff_only: bool = False, limit: int = 20) -> list[dict[str, Any]]:
    return await service.list_conversations(min(limit, 200), handoff_only)


@router.get("/conversations/{conv_id}")
async def conversation(conv_id: uuid.UUID, admin: Admin) -> dict[str, Any]:
    try:
        return await service.get_conversation(conv_id)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@router.post("/conversations/{conv_id}/handoff")
async def handoff(conv_id: uuid.UUID, body: HandoffIn, admin: Admin) -> dict[str, Any]:
    try:
        return await service.set_handoff(conv_id, body.active, admin, body.reason)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


@router.delete("/conversations/{conv_id}")
async def delete(conv_id: uuid.UUID, admin: Admin, request: Request) -> dict[str, Any]:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    try:
        async with AsyncPostgresSaver.from_conn_string(get_settings().psycopg_conninfo) as saver:
            return await service.delete_conversation(conv_id, admin, saver, request.app.state.tracer)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


# ----------------------------------------------------------------------------- trợ lý nội bộ + tài liệu
class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    include_internal: bool = True


@router.post("/ask")
async def ask(body: AskIn, admin: Admin) -> dict[str, Any]:
    """Trợ lý nội bộ: trả lời dựa trên tài liệu đã nạp (public + internal), có trích nguồn."""
    from app.conversation.knowledge import load_knowledge
    from app.providers.llm import ProviderError, build_provider
    from app.rag import store
    from app.rag.assistant import ask as do_ask
    from app.rag.embedder import EmbeddingError, HttpEmbedder
    from app.rag.retriever import DocSearch

    s = get_settings()
    try:
        return await do_ask(
            body.question,
            settings=s,
            llm=build_provider(s),
            docs=DocSearch(s, HttpEmbedder(s)),
            kb=load_knowledge(s.knowledge_dir),
            visibility=store.ALL if body.include_internal else store.PUBLIC,
        )
    except (EmbeddingError, ProviderError) as exc:
        raise HTTPException(503, f"dịch vụ model chưa sẵn sàng: {type(exc).__name__}") from exc


@router.get("/docs")
async def documents(admin: Admin) -> list[dict[str, Any]]:
    from app.rag import store

    return await store.list_documents()
