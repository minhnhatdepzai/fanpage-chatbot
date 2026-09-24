"""Che bí mật và thông tin cá nhân (PII) trước khi ghi log / trace / xuất dữ liệu.

Hai lớp:
- ``redact_secrets``: luôn áp dụng cho mọi dòng log. Che các giá trị bí mật đã biết
  (lấy từ cấu hình) và các mẫu token phổ biến.
- ``redact_pii``: che số điện thoại, email, số giấy tờ/số dài, số thẻ. Dùng cho nội dung
  hội thoại khi ghi trace, xuất dữ liệu huấn luyện. Regex không bắt được mọi PII
  (tên riêng, địa chỉ tự do) -> vẫn cần người duyệt.
"""

from __future__ import annotations

import re
import threading
import unicodedata

REDACTED = "[REDACTED]"

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Token truy cập Facebook/Meta (User/Page/App token đều bắt đầu bằng EAA)
    (re.compile(r"EAA[A-Za-z0-9]{20,}"), REDACTED),
    # Kaggle API token kiểu mới
    (re.compile(r"KGAT_[A-Za-z0-9]{16,}"), REDACTED),
    # Khóa Langfuse
    (re.compile(r"\b(?:pk|sk)-lf-[A-Za-z0-9\-]{8,}"), REDACTED),
    # Khóa kiểu sk-... (OpenAI/Anthropic/...)
    (re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_\-]{16,}"), REDACTED),
    # Authorization header
    (re.compile(r"(?i)\b(bearer|oauth)\s+[A-Za-z0-9._~+/=\-]{12,}"), r"\1 " + REDACTED),
    # Tham số nhạy cảm trong query string / form
    (
        re.compile(
            r"(?i)\b(access_token|input_token|hub\.verify_token|verify_token|appsecret_proof|"
            r"client_secret|app_secret|api_key|apikey|password|passwd|secret)=([^&\s\"']+)"
        ),
        r"\1=" + REDACTED,
    ),
    # Mật khẩu trong URL database
    (re.compile(r"(postgres(?:ql)?(?:\+\w+)?://[^:/\s@]+:)([^@\s]+)(@)"), r"\1" + REDACTED + r"\3"),
    # Chữ ký webhook
    (re.compile(r"sha256=[0-9a-fA-F]{64}"), "sha256=" + REDACTED),
]

_known_secrets: list[str] = []
_lock = threading.Lock()


def register_secrets(values: list[str]) -> None:
    """Đăng ký các giá trị bí mật thật (từ Settings) để che nguyên văn."""
    with _lock:
        for v in values:
            if v and len(v) >= 8 and v not in _known_secrets:
                _known_secrets.append(v)
        # thay chuỗi dài trước để tránh che dở dang
        _known_secrets.sort(key=len, reverse=True)


def clear_registered_secrets() -> None:
    with _lock:
        _known_secrets.clear()


def redact_secrets(text: str) -> str:
    if not text:
        return text
    for secret in _known_secrets:
        if secret in text:
            text = text.replace(secret, REDACTED)
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


# ---------------------------------------------------------------- PII
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
# Số thẻ: 13-19 chữ số, có thể cách nhau bởi dấu cách/gạch
_CARD = re.compile(r"(?<!\d)(?:\d[ \-]?){12,18}\d(?!\d)")
# Điện thoại Việt Nam: 0xxxxxxxxx / +84xxxxxxxxx, cho phép dấu cách, chấm, gạch
_PHONE_VN = re.compile(r"(?<![\d+])(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\d)")
# Chuỗi số dài (CCCD 12 số, CMND 9 số, số tài khoản...)
_LONG_NUMBER = re.compile(r"(?<!\d)\d{9,}(?!\d)")
_URL_QUERY = re.compile(r"(https?://[^\s?#]+)\?[^\s#]+")


def redact_pii(text: str) -> str:
    """Che PII phổ biến. Thứ tự quan trọng: email -> URL query -> thẻ -> điện thoại -> số dài."""
    if not text:
        return text
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL_QUERY.sub(r"\1?[...]", text)
    text = _CARD.sub(_card_or_phone, text)
    text = _PHONE_VN.sub("[SĐT]", text)
    text = _LONG_NUMBER.sub("[SỐ]", text)
    return text


def _card_or_phone(m: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return "[SỐ_THẺ]"
    return m.group(0)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_all(text: str) -> str:
    return redact_pii(redact_secrets(text))


def contains_pii(text: str) -> bool:
    return redact_pii(text) != text


# ---------------------------------------------------------------- tiện ích tiếng Việt
def nfc(text: str) -> str:
    """Chuẩn hóa Unicode NFC (giữ nguyên dấu tiếng Việt)."""
    return unicodedata.normalize("NFC", text) if text else text


VI_CHARS = frozenset("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")


def vietnamese_score(text: str) -> float:
    """Tỉ lệ chữ cái có dấu tiếng Việt (tín hiệu ngôn ngữ; KHÔNG dùng để loại tiếng Việt không dấu)."""
    letters = [c for c in (text or "").lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in VI_CHARS for c in letters) / len(letters)


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt để so khớp mẫu (không dùng để lưu trữ)."""
    t = unicodedata.normalize("NFD", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    return t.replace("đ", "d").replace("Đ", "D")
