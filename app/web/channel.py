"""Kênh chat website: hội thoại có page_id = "web", psid = mã phiên do server cấp (kèm chữ ký HMAC).

Tin nhắn web đi đúng hàng đợi/worker/graph như Messenger; khác ở bước giao: không gọi Send API, widget tự lấy tin
(``GET /web/messages``). Mã phiên không đoán được và có chữ ký -> người khác không đọc được hội thoại của mình.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

WEB_PAGE_ID = "web"


def is_web(page_id: str | None) -> bool:
    return page_id == WEB_PAGE_ID


def _key(secret: str) -> bytes:
    return hmac.new(secret.encode(), b"web-session-v1", hashlib.sha256).digest()


def new_session(secret: str) -> tuple[str, str]:
    sid = "w_" + secrets.token_urlsafe(18)
    return sid, sign(secret, sid)


def sign(secret: str, session_id: str) -> str:
    return hmac.new(_key(secret), session_id.encode(), hashlib.sha256).hexdigest()[:40]


def verify(secret: str, session_id: str, token: str) -> bool:
    if not (secret and session_id.startswith("w_") and 10 <= len(session_id) <= 64 and token):
        return False
    return hmac.compare_digest(sign(secret, session_id), token)
