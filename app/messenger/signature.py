"""Xác minh X-Hub-Signature-256 cho webhook POST.

Tài liệu Meta (Webhooks > Validate Payloads): chữ ký là HMAC-SHA256 của payload với App Secret,
header dạng ``sha256=<hex>``; Meta tính chữ ký trên *bản payload đã escape unicode* (``\\uXXXX``,
hex chữ thường). Thông thường raw body Meta gửi đã ở dạng escape nên tính trực tiếp trên raw bytes
là đủ; để an toàn với tiếng Việt, nếu raw body chứa ký tự UTF-8 thô, ta thử thêm bản đã escape.
Cả hai cách đều đòi hỏi biết App Secret nên không làm yếu xác thực.
"""

from __future__ import annotations

import hashlib
import hmac

PREFIX = "sha256="


def _escape_non_ascii(raw: bytes) -> bytes | None:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if text.isascii():
        return None
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp < 0x80:
            out.append(ch)
        elif cp <= 0xFFFF:
            out.append(f"\\u{cp:04x}")
        else:  # ngoài BMP (emoji) -> cặp surrogate như JSON
            cp -= 0x10000
            out.append(f"\\u{0xD800 + (cp >> 10):04x}\\u{0xDC00 + (cp & 0x3FF):04x}")
    return "".join(out).encode("ascii")


def compute_signature(app_secret: str, body: bytes) -> str:
    return PREFIX + hmac.new(app_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def verify_signature(app_secret: str, body: bytes, header_value: str | None) -> bool:
    """So sánh hằng thời gian; trả False nếu thiếu secret/header hoặc sai định dạng."""
    if not app_secret or not header_value or not header_value.startswith(PREFIX):
        return False
    received = header_value[len(PREFIX) :].strip().lower()
    if len(received) != 64:
        return False
    candidates = [body]
    escaped = _escape_non_ascii(body)
    if escaped is not None:
        candidates.append(escaped)
    ok = False
    for candidate in candidates:
        expected = hmac.new(app_secret.encode("utf-8"), candidate, hashlib.sha256).hexdigest()
        # không return sớm để thời gian không phụ thuộc vào vị trí khớp
        ok |= hmac.compare_digest(expected, received)
    return ok
