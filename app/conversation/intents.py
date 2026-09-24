"""Nhận diện ý định bằng luật (xác định, kiểm thử được) cho các quyết định có hệ quả:
yêu cầu gặp người thật (handoff), câu hỏi danh tính và câu hỏi kiến thức/sự kiện (cần nguồn).

So khớp trên văn bản đã bỏ dấu + chữ thường nên hiểu cả tiếng Việt không dấu.
Thiết kế thận trọng: động từ "mạnh" (gặp, nói chuyện với, liên hệ...) đi với nhiều đối tượng;
động từ "yếu" (cần, muốn, xin) chỉ đi với đối tượng rõ nghĩa (người thật, admin, quản trị viên...)
để tránh bắt nhầm câu như "mình cần người yêu" hay "muốn làm nhân viên".
"""

from __future__ import annotations

import re

from app.observability.redaction import strip_diacritics

_STRONG_VERB = (
    r"(?:gap|noi chuyen(?: voi)?|chat voi|nhan tin (?:voi|cho)|lien he(?: voi)?|lien lac(?: voi)?|"
    r"ket noi(?: voi| den| toi)?|chuyen (?:sang|qua|cho|den|toi)|goi (?:cho )?|"
    r"talk to|speak (?:to|with)|connect me (?:to|with))"
)
_TARGET_ANY = (
    r"(?:nguoi that|admin|ad|quan tri(?: vien)?|qtv|nhan vien|tu van vien|chu (?:page|shop|trang)|"
    r"nguoi phu trach|nguoi ho tro|cham soc khach hang|cskh|human|real person|agent|staff|someone real)"
)
_WEAK_VERB = r"(?:can|muon|xin|cho (?:minh|toi|em|to|tui|anh|chi|tao))"
_TARGET_STRICT = r"(?:nguoi that|admin|quan tri vien|tu van vien|chu (?:page|shop|trang)|human|real person)"
_BOT_WORD = r"(?:bot|chatbot|robot|tro ly ao|may tu dong|tu dong)"
_DISLIKE = r"(?:(?:khong|ko|k) (?:muon|thich)|chan|ghet)"

_HANDOFF_PATTERNS = [
    re.compile(rf"\b{_STRONG_VERB}\b(?:\s+\w+){{0,2}}\s+{_TARGET_ANY}\b"),
    re.compile(rf"\b{_WEAK_VERB}\b(?:\s+\w+){{0,2}}\s+{_TARGET_STRICT}\b"),
    re.compile(rf"\b{_DISLIKE}\s+(?:noi chuyen|chat|nhan tin)\s+voi\s+(?:{_BOT_WORD}|may)\b"),
    re.compile(rf"\b{_DISLIKE}\s+{_BOT_WORD}\b"),
    re.compile(r"\b(?:live agent|human agent|real human)\b"),
    re.compile(r"\bco (?:admin|ad|nhan vien|quan tri vien|nguoi that)(?: nao)? (?:o do|khong|ko|k)\b"),
]
# Phủ định chỉ tính khi đối tượng là con người ("không cần gặp admin"), không phải bot.
_NEGATION = re.compile(
    rf"\b(?:khong|ko|k|chua|dung|khoi)\s+(?:can\s+|muon\s+)?{_STRONG_VERB}(?:\s+\w+){{0,2}}\s+{_TARGET_ANY}\b"
)
_IDENTITY = re.compile(
    r"\b(?:ban|em|anh|chi|day|may|ben kia|minh dang noi chuyen voi)\b(?:\s+\w+){0,3}\s+"
    r"(?:la|co phai|phai)\b(?:\s+\w+){0,3}\s+(?:nguoi that|nguoi|bot|ai|robot|may|chatbot|tro ly ao)\b"
    r"|\b(?:are you|is this) (?:a )?(?:bot|human|real|ai|robot)\b"
)
_EXPLICIT_REQUEST = re.compile(rf"\b{_STRONG_VERB}\b|\b{_WEAK_VERB}\b(?:\s+\w+){{0,2}}\s+{_TARGET_STRICT}\b")


def normalize(text: str) -> str:
    t = strip_diacritics(text.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_handoff_request(text: str | None) -> bool:
    if not text:
        return False
    t = normalize(text)
    if not t or _NEGATION.search(t):
        return False
    # "bạn là người thật à?" là câu hỏi danh tính, không phải yêu cầu gặp người thật
    if _IDENTITY.search(t) and not _EXPLICIT_REQUEST.search(t):
        return False
    return any(p.search(t) for p in _HANDOFF_PATTERNS)


def is_identity_question(text: str | None) -> bool:
    return bool(text) and bool(_IDENTITY.search(normalize(text)))


# Hỏi về khả năng của chính bot: trả lời theo danh sách tính năng trong prompt, không phải kiến thức cần nguồn.
_CAPABILITY = re.compile(
    r"\b(?:ban|em|bot|chatbot|tro ly)\b(?:\s+\w+){0,2}\s+(?:lam duoc|giup duoc|ho tro duoc|co the lam|co the giup|"
    r"biet lam|lam gi duoc|giup gi duoc|co chuc nang|co tinh nang|co kha nang)\b"
    r"|\b(?:chuc nang|tinh nang|kha nang)\s+(?:cua\s+)?(?:ban|em|bot|chatbot)\b"
    r"|\bwhat can you do\b"
)


def is_capability_question(text: str | None) -> bool:
    return bool(text) and bool(_CAPABILITY.search(normalize(text)))


# Câu hỏi đòi sự kiện/kiến thức -> câu trả lời phải có nguồn hoặc ghi chú "chưa kiểm chứng".
_KNOWLEDGE_CUE = re.compile(
    r"\b(?:la gi|la ai|nghia la|y nghia|tai sao|vi sao|vi dau|nhu the nao|the nao la|ra sao|"
    r"hoat dong (?:sao|the nao|nhu nao)|bao nhieu|bao gio|khi nao|nam nao|luc nao|o dau|ra doi|"
    r"phat minh|sang lap|tac gia|nguoi (?:tao ra|phat minh|sang lap)|khac (?:gi|nhau|nhau gi)|khac biet|"
    r"so sanh|giai thich|dinh nghia|nguyen ly|nguyen tac|cong thuc|co that|dung khong|dung ko|"
    r"nguon (?:nao|o dau|goc)|bang chung|thong ke|so lieu|ty le|ti le|"
    r"ke (?:ve|(?:cho )?(?:minh|toi|em|tui|to) (?:nghe|biet))|cho (?:minh|toi|em|tui) biet|thong tin ve|"
    r"tim hieu ve|"
    r"lich su|tieu su|moi nhat|cao nhat|nhieu nhat|lon nhat|dau tien|doanh thu|"
    r"(?:ra mat|phat hanh|cong bo|ra doi|ra) chua|"
    r"may (?:mon|nguoi|lan|nam|ngay|tinh|thanh pho|phan tram|diem)|"
    r"what is|what are|why|how (?:does|do|is|to|many|much)|when (?:did|was|is)|who (?:is|was|invented|created)|"
    r"explain|difference)\b"
)
# Hỏi thăm/nhắc lại chuyện riêng trong hội thoại: không phải câu hỏi kiến thức.
_PERSONAL = re.compile(
    r"\b(?:ten|tuoi|khoe|so thich|nha|que|nguoi yeu|tam trang|cam thay)\b.*\b(?:ban|cau|em|minh|toi|tui|to)\b"
    r"|\b(?:ban|cau|em|minh|toi|tui|to)\b.*\b(?:ten|tuoi|khoe|so thich|tam trang|cam thay)\b"
    r"|\b(?:ban|cau|em)\s+(?:the nao|sao roi|on khong|dang lam gi)\b"
)


def is_knowledge_question(text: str | None) -> bool:
    """Câu hỏi cần sự kiện/kiến thức (không phải chào hỏi, tâm sự, hỏi danh tính, nhắc chuyện riêng)."""
    if not text:
        return False
    t = normalize(text)
    if not t or is_identity_question(text) or is_capability_question(text) or _PERSONAL.search(t):
        return False
    return bool(_KNOWLEDGE_CUE.search(t))


GET_STARTED_PAYLOADS = {"GET_STARTED", "GET_STARTED_PAYLOAD", "get_started"}


def is_get_started(payload: str | None) -> bool:
    return bool(payload) and payload.strip() in GET_STARTED_PAYLOADS
