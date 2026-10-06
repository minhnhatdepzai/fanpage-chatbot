"""API cho khung chat nhúng website (widget TypeScript ở web/widget, bản build phục vụ tại /web/widget.js).

    POST /web/session            -> {session_id, token}
    POST /web/messages           {session_id, token, text}   (lưu bền vững rồi mới trả 202, worker xử lý như Messenger)
    GET  /web/messages?session_id=&token=&after=<id>          -> tin trả lời mới + còn đang xử lý hay không
    GET  /web/widget.js, /web/demo

Bảo vệ: phiên có chữ ký HMAC, giới hạn độ dài tin, giới hạn tần suất theo IP và theo phiên, CORS chỉ cho các tên
miền trong WEB_ALLOWED_ORIGINS. Nội dung trả về là văn bản thuần (widget hiển thị bằng textContent, không innerHTML).
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.config import get_settings
from app.messenger.events import UserMessage
from app.storage.db import session_scope
from app.storage.repository import ingest_events
from app.web.channel import WEB_PAGE_ID, new_session, verify

router = APIRouter(prefix="/web", tags=["web"])
STATIC = Path(__file__).resolve().parents[1] / "web" / "static"
_IP_HITS: dict[str, deque[float]] = {}


def _secret() -> str:
    return get_settings().pseudonym_secret.get_secret_value()


def _enabled() -> None:
    if not get_settings().web_chat_enabled:
        raise HTTPException(404, "web chat disabled")


def _client_ip(request: Request) -> str:
    # API chỉ bind 127.0.0.1: truy cập từ ngoài đi qua Cloudflare Tunnel, header này do Cloudflare đặt
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")


def _rate_limit_ip(request: Request, kind: str, limit: int, window_s: float = 300.0) -> None:
    now = time.monotonic()
    q = _IP_HITS.setdefault(f"{kind}:{_client_ip(request)}", deque())
    while q and now - q[0] > window_s:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "Bạn gửi hơi nhanh, đợi một chút rồi nhắn tiếp nhé.")
    q.append(now)
    if len(_IP_HITS) > 10_000:  # chặn phình bộ nhớ
        _IP_HITS.clear()


class SessionOut(BaseModel):
    session_id: str
    token: str


class MessageIn(BaseModel):
    session_id: str = Field(max_length=64)
    token: str = Field(max_length=64)
    text: str = Field(min_length=1)


def _check_session(session_id: str, token: str) -> None:
    if not verify(_secret(), session_id, token):
        raise HTTPException(401, "phiên chat không hợp lệ")


@router.post("/session", response_model=SessionOut)
async def create_session(request: Request) -> SessionOut:
    _enabled()
    _rate_limit_ip(request, "session", get_settings().web_rate_limit_per_5min)
    sid, tok = new_session(_secret())
    return SessionOut(session_id=sid, token=tok)


@router.post("/messages", status_code=202)
async def post_message(body: MessageIn, request: Request) -> dict[str, Any]:
    _enabled()
    s = get_settings()
    _check_session(body.session_id, body.token)
    msg = body.text.strip()
    if not msg or len(msg) > s.web_max_message_chars:
        raise HTTPException(400, f"Tin nhắn cần từ 1 đến {s.web_max_message_chars} ký tự.")
    # nhiều người có thể chung một IP (trường, công ty) -> giới hạn theo IP rộng hơn giới hạn theo phiên
    _rate_limit_ip(request, "message", 3 * s.web_rate_limit_per_5min)
    await _ingest_text_message(body.session_id, msg, s)
    return {"accepted": True, "queue": await _queue_status_for_session(body.session_id)}


async def _ingest_text_message(session_id: str, msg: str, settings: Any | None = None) -> None:
    """Ghi một tin web vào cùng hàng đợi Messenger; dùng lại cho các phòng học chuyên biệt."""
    s = settings or get_settings()
    ev = UserMessage(
        page_id=WEB_PAGE_ID,
        psid=session_id,
        mid=f"web-{uuid.uuid4().hex}",
        ts=datetime.now(UTC),
        text=msg,
    )
    async with session_scope() as db:
        recent = (
            await db.execute(
                text(
                    "SELECT count(*) FROM messages m JOIN conversations c ON c.id = m.conversation_id "
                    "WHERE c.page_id = :p AND c.psid = :sid AND m.role = 'user' "
                    "AND m.created_at > now() - interval '5 minutes'"
                ),
                {"p": WEB_PAGE_ID, "sid": session_id},
            )
        ).scalar_one()
        if recent >= s.web_rate_limit_per_5min:
            raise HTTPException(429, "Bạn gửi hơi nhanh, đợi một chút rồi nhắn tiếp nhé.")
        await ingest_events(db, [ev], s)  # ghi bền vững TRƯỚC khi trả 202


async def _queue_status(db: Any, conv_id: Any) -> dict[str, Any]:
    """Vị trí công khai, không lộ định danh hay nội dung của người khác trong hàng đợi."""
    row = (
        (
            await db.execute(
                text(
                    """
                WITH live AS (
                    SELECT c.id, c.next_run_at,
                           (c.lease_owner IS NOT NULL AND c.lease_expires_at > now()) AS processing
                    FROM conversations c
                    WHERE c.next_run_at IS NOT NULL
                      AND (EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id
                                   AND m.role = 'user' AND m.status = 'pending')
                           OR EXISTS (SELECT 1 FROM turns t WHERE t.conversation_id = c.id
                                      AND t.status = 'running'))
                ), ranked AS (
                    SELECT id, processing,
                           row_number() OVER (ORDER BY processing DESC, next_run_at, id) AS position,
                           count(*) OVER () AS total,
                           count(*) FILTER (WHERE processing) OVER () AS active_jobs
                    FROM live
                ), capacity AS (
                    SELECT COALESCE(sum(slots), 0) AS slots FROM (
                        SELECT split_part(worker_id, ':', 1) AS host,
                               max(CASE WHEN info->>'concurrency' ~ '^[0-9]+$'
                                        THEN (info->>'concurrency')::int ELSE 0 END) AS slots
                        FROM worker_heartbeats WHERE last_seen > now() - interval '60 seconds'
                        GROUP BY split_part(worker_id, ':', 1)
                    ) live_workers
                )
                SELECT r.processing, r.position, r.total, r.active_jobs, capacity.slots
                FROM capacity LEFT JOIN ranked r ON r.id = :cid
                """
                ),
                {"cid": conv_id},
            )
        )
        .mappings()
        .one()
    )
    if row["position"] is None:
        return {
            "phase": "idle",
            "position": None,
            "total": 0,
            "active_jobs": 0,
            "worker_slots": int(row["slots"] or 0),
        }
    return {
        "phase": "processing" if row["processing"] else "queued",
        "position": int(row["position"]),
        "total": int(row["total"]),
        "active_jobs": int(row["active_jobs"]),
        "worker_slots": int(row["slots"] or 0),
    }


async def _queue_status_for_session(session_id: str) -> dict[str, Any]:
    async with session_scope() as db:
        conv_id = (
            await db.execute(
                text("SELECT id FROM conversations WHERE page_id = :p AND psid = :sid"),
                {"p": WEB_PAGE_ID, "sid": session_id},
            )
        ).scalar_one_or_none()
        if conv_id is None:
            return {"phase": "idle", "position": None, "total": 0, "active_jobs": 0, "worker_slots": 0}
        return await _queue_status(db, conv_id)


class ImageIn(BaseModel):
    session_id: str = Field(max_length=64)
    token: str = Field(max_length=64)
    image_base64: str = Field(min_length=16)
    caption: str = Field(default="", max_length=2000)


@router.post("/images", status_code=202)
async def post_image(body: ImageIn, request: Request) -> dict[str, Any]:
    """Ảnh từ widget: OCR/YOLO/VLM rồi lưu KẾT QUẢ đã che dữ liệu cá nhân, không lưu ảnh."""
    import base64
    import binascii

    from app.vision.client import VisionClient, VisionError
    from app.vision.context import sanitize_analysis

    _enabled()
    s = get_settings()
    if not s.vision_enabled:
        raise HTTPException(503, "Khung chat hiện chưa nhận ảnh.")
    _check_session(body.session_id, body.token)
    if len(body.image_base64) > s.vision_max_image_mb * 1_000_000 * 4 // 3 + 16:
        raise HTTPException(413, f"Ảnh vượt {s.vision_max_image_mb} MB.")
    caption = body.caption.strip()
    if len(caption) > s.web_max_message_chars:
        raise HTTPException(400, f"Chú thích tối đa {s.web_max_message_chars} ký tự.")
    _rate_limit_ip(request, "message", 3 * s.web_rate_limit_per_5min)
    try:
        data = base64.b64decode(body.image_base64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(400, "Ảnh không hợp lệ.") from exc
    try:
        analysis = sanitize_analysis(await VisionClient(s).analyze(data, question=caption or None))
    except VisionError as exc:
        if exc.status in (400, 413):
            raise HTTPException(
                400,
                "Ảnh không đọc được hoặc không đúng định dạng (JPEG/PNG/WEBP, tối đa "
                f"{s.vision_max_image_mb} MB).",
            ) from exc
        raise HTTPException(503, "Hiện chưa xử lý được ảnh, bạn thử lại sau hoặc gõ nội dung nhé.") from exc
    ev = UserMessage(
        page_id=WEB_PAGE_ID,
        psid=body.session_id,
        mid=f"web-{uuid.uuid4().hex}",
        ts=datetime.now(UTC),
        text=caption or None,
        attachment_types=["image"],
        images=[{"analysis": analysis}],
    )
    async with session_scope() as db:
        await ingest_events(db, [ev], s)
    return {"accepted": True, "queue": await _queue_status_for_session(body.session_id)}


@router.get("/messages")
async def get_messages(session_id: str, token: str, after: int = 0) -> dict[str, Any]:
    _enabled()
    _check_session(session_id, token)
    async with session_scope() as db:
        conv = (
            await db.execute(
                text("SELECT id FROM conversations WHERE page_id = :p AND psid = :sid"),
                {"p": WEB_PAGE_ID, "sid": session_id},
            )
        ).scalar_one_or_none()
        if conv is None:
            return {
                "messages": [],
                "pending": False,
                "queue": {"phase": "idle", "position": None, "total": 0, "active_jobs": 0, "worker_slots": 0},
            }
        rows = (
            (
                await db.execute(
                    text(
                        "SELECT m.id, m.text, m.sent_at, t.mode FROM messages m "
                        "LEFT JOIN turns t ON t.id = m.turn_id "
                        "WHERE m.conversation_id = :c AND m.role = 'assistant' "
                        "AND m.status = 'sent' AND m.id > :after ORDER BY m.id LIMIT 50"
                    ),
                    {"c": conv, "after": after},
                )
            )
            .mappings()
            .all()
        )
        pending = (
            await db.execute(
                text(
                    "SELECT EXISTS (SELECT 1 FROM messages WHERE conversation_id = :c AND role = 'user' "
                    "AND status = 'pending') OR EXISTS (SELECT 1 FROM turns WHERE conversation_id = :c "
                    "AND status = 'running')"
                ),
                {"c": conv},
            )
        ).scalar_one()
        queue = await _queue_status(db, conv)
    return {
        "messages": [
            {
                "id": r["id"],
                "text": r["text"],
                "mode": r["mode"] or "chat",
                "at": r["sent_at"].isoformat() if r["sent_at"] else None,
            }
            for r in rows
        ],
        "pending": bool(pending),
        "queue": queue,
    }


@router.get("/config")
async def widget_config() -> dict[str, Any]:
    s = get_settings()
    _enabled()
    return {
        "images": s.vision_enabled,
        "max_chars": s.web_max_message_chars,
        "max_image_mb": s.vision_max_image_mb,
    }


@router.get("/widget.js", include_in_schema=False)
async def widget_js() -> FileResponse:
    _enabled()
    return FileResponse(
        STATIC / "widget.js",
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=300"},
    )


@router.get("/demo", include_in_schema=False)
async def demo() -> HTMLResponse:
    _enabled()
    return HTMLResponse((STATIC / "demo.html").read_text(encoding="utf-8"))
