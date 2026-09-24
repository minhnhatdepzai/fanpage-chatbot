"""Thông báo handoff cho quản trị viên qua webhook (Slack/Discord/Google Chat... tương thích JSON).

Chỉ khi thông báo gửi THÀNH CÔNG, bot mới được nói "đã báo cho quản trị viên".
Nội dung thông báo không chứa PSID hay nội dung tin nhắn (chỉ mã hội thoại nội bộ).
"""

from __future__ import annotations

import logging

import httpx

from app.config import Settings

log = logging.getLogger(__name__)


async def notify_handoff(settings: Settings, conversation_id: str, user_ref: str) -> bool:
    url = settings.handoff_notify_webhook_url.get_secret_value()
    if not url:
        return False
    msg = (
        f"[fanpage-chatbot] Người dùng {user_ref} yêu cầu gặp quản trị viên. "
        f"Hội thoại: {conversation_id}. Bot đã tạm dừng cho cuộc hội thoại này."
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=4.0)) as client:
            resp = await client.post(url, json={"text": msg, "content": msg})
        ok = 200 <= resp.status_code < 300
        if not ok:
            log.warning("handoff_notify_failed", extra={"status": resp.status_code})
        return ok
    except httpx.HTTPError as exc:
        log.warning("handoff_notify_failed", extra={"error": type(exc).__name__})
        return False
