"""Webhook Messenger.

GET  /webhook : xác minh đăng ký (hub.mode, hub.verify_token, hub.challenge). Verify Token chỉ dùng cho bước
               này, KHÔNG phải cơ chế xác thực POST.
POST /webhook : xác minh X-Hub-Signature-256 trên raw body bằng App Secret, giới hạn kích thước body,
               kiểm tra Page ID, lưu sự kiện + lập lịch trong một transaction, rồi mới trả 200.
               DB lỗi -> 503 để Meta gửi lại (trùng lặp được chặn bằng message ID).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from app.config import get_settings
from app.messenger.events import parse_webhook_payload
from app.messenger.signature import verify_signature
from app.storage.db import session_scope
from app.storage.repository import ingest_events

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/webhook")
async def verify_webhook(request: Request) -> Response:
    s = get_settings()
    q = request.query_params
    mode, token, challenge = q.get("hub.mode"), q.get("hub.verify_token"), q.get("hub.challenge")
    expected = s.meta_verify_token.get_secret_value()
    if mode == "subscribe" and expected and token and challenge and hmac.compare_digest(token, expected):
        log.info("webhook_verified")
        return PlainTextResponse(challenge)
    log.warning("webhook_verify_rejected", extra={"mode": mode, "has_token": bool(token)})
    return PlainTextResponse("forbidden", status_code=403)


async def _read_body_limited(request: Request, limit: int) -> bytes | None:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/webhook")
async def receive_webhook(request: Request) -> Response:
    s = get_settings()
    body = await _read_body_limited(request, s.webhook_max_body_bytes)
    if body is None:
        return JSONResponse({"error": "payload_too_large"}, status_code=413)
    secret = s.meta_app_secret.get_secret_value()
    if not secret:
        log.error("webhook_rejected_no_app_secret - hãy cấu hình META_APP_SECRET")
        return JSONResponse({"error": "not_configured"}, status_code=503)
    if not verify_signature(secret, body, request.headers.get("x-hub-signature-256")):
        log.warning("webhook_bad_signature", extra={"has_header": "x-hub-signature-256" in request.headers})
        return JSONResponse({"error": "invalid_signature"}, status_code=401)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid_json"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "invalid_payload"}, status_code=400)
    events = parse_webhook_payload(payload, {s.meta_page_id} if s.meta_page_id else set())
    try:
        async with asyncio.timeout(4.0):  # Meta yêu cầu phản hồi trong <= 5 giây
            async with session_scope() as session:
                stats = await ingest_events(session, events, s)
    except Exception as exc:  # noqa: BLE001 - DB lỗi/timeout: KHÔNG ACK để Meta gửi lại
        log.error("webhook_persist_failed", extra={"error": type(exc).__name__})
        return JSONResponse({"error": "temporarily_unavailable"}, status_code=503)
    log.info(
        "webhook_received",
        extra={
            "events": stats.received,
            "stored": stats.stored_messages,
            "duplicates": stats.duplicates,
            "echo_confirmed": stats.echoes_confirmed,
            "human_agent": stats.human_agent_messages,
            "ignored": stats.ignored,
        },
    )
    return PlainTextResponse("EVENT_RECEIVED")
