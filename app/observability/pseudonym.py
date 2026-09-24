"""Định danh giả danh hóa (HMAC) cho log, trace và dữ liệu xuất.

PSID chỉ được lưu trong database để gửi tin; mọi nơi khác dùng ``user_ref``.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import secrets

log = logging.getLogger(__name__)
_ephemeral_key: bytes | None = None


def _key(secret: str) -> bytes:
    global _ephemeral_key
    if secret:
        return secret.encode("utf-8")
    if _ephemeral_key is None:
        log.warning(
            "PSEUDONYM_SECRET chưa cấu hình: dùng khóa tạm thời, user_ref sẽ đổi sau khi khởi động lại"
        )
        _ephemeral_key = secrets.token_bytes(32)
    return _ephemeral_key


def pseudonymize(secret: str, *parts: str, prefix: str = "u_", length: int = 20) -> str:
    msg = "\x1f".join(parts).encode("utf-8")
    digest = hmac.new(_key(secret), msg, hashlib.sha256).digest()
    return prefix + base64.b32encode(digest).decode("ascii").lower().rstrip("=")[:length]


def user_ref(secret: str, page_id: str, psid: str) -> str:
    """Định danh ổn định theo Page ID + PSID (PSID khác nhau giữa các Page)."""
    return pseudonymize(secret, page_id, psid, prefix="u_")
