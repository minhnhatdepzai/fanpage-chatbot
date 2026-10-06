"""Nhận diện tác vụ Ngữ văn bằng luật xác định, hỗ trợ tiếng Việt có/không dấu.

Luật chỉ bắt các cụm đủ mạnh để câu hỏi thông thường có chữ ``phân tích`` hoặc
``bài viết`` không bị chuyển nhầm sang gia sư Ngữ văn.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

from app.conversation.intents import normalize
from app.writing_tutor.traditions import detect_creative_profile


@dataclass(frozen=True, slots=True)
class WritingTask:
    is_writing: bool
    kind: str = "none"
    output: str = "answer"
    grade: int | None = None
    requires_sources: bool = False
    creative_requested: bool = False
    tradition: str | None = None
    form: str | None = None
    tones: tuple[str, ...] = ()
    language: str | None = None
    target_words: int | None = None

    def to_state(self) -> dict[str, object]:
        state = asdict(self)
        state["tones"] = list(self.tones)
        return state


_GRADE = re.compile(r"\b(?:lop|grade)\s*(1[0-2]|[1-9])\b")
_REVISION = re.compile(
    r"\b(?:cham|nhan xet|sua|chua|viet lai|nang cap|cai thien|rut gon|mo rong)\b"
    r"(?:\s+\w+){0,5}\s+\b(?:bai van|doan van|bai viet|mo bai|than bai|ket bai)\b"
)
_OUTLINE = re.compile(r"\b(?:lap|len|viet|cho)\b(?:\s+\w+){0,3}\s+\b(?:dan y|outline)\b|\bdan y\b")
_READING = re.compile(
    r"\b(?:doc hieu|tra loi cau hoi doc hieu|phuong thuc bieu dat|bien phap tu tu|"
    r"ngoi ke|the tho|mach cam xuc|noi dung doan trich)\b"
)
_LITERARY = re.compile(
    r"\b(?:nghi luan van hoc|phan tich|cam nhan|binh giang|binh luan|so sanh)\b"
    r"(?:\s+\w+){0,8}\s+\b(?:bai tho|doan tho|cau tho|tac pham|truyen|truyen ngan|"
    r"doan trich|nhan vat|hinh tuong|chi tiet|van ban|tac gia|nghe thuat)\b"
    r"|\b(?:phan tich nhan vat|phan tich tac pham|cam nhan bai tho|cam nhan doan tho|"
    r"phan tich (?:hinh anh|chi tiet|bieu tuong|tinh huong)|"
    r"so sanh (?:hai )?cau tu viet|bi kich cua|hinh tuong (?:nhan vat )?|"
    r"so phan cua nhan vat|gia tri hien thuc|gia tri nhan dao)\b"
)
_LITERARY_ACTION = re.compile(r"\b(?:phan tich|cam nhan|binh giang|binh luan|nghi luan|so sanh)\b")
_LITERARY_CONTEXT = re.compile(
    r"\b(?:van hoc|truyen thuyet|than thoai|co tich|cau chuyen|tac pham|van ban|bai tho|doan tho|"
    r"cau tho|doan trich|truyen|truyen ngan|tieu thuyet|nhan vat|hinh tuong|chi tiet|bieu tuong|"
    r"tac gia|nguoi ke chuyen|nhan vat tru tinh|nghe thuat|thi phap)\b"
)
_GENERIC_LITERARY_ANALYSIS = re.compile(
    r"\bphan tich\s+(?:bai\s+)?nghi luan\b|\b(?:viet|lam)\b(?:\s+\w+){0,7}\s+\bbai phan tich\b"
)
_GENERIC_WRITING_REQUEST = re.compile(
    r"\b(?:viet|lam)\s+(?:cho\s+(?:toi|minh|em)\s+)?(?:mot\s+)?(?:bai\s+)?ve\s+\S+"
    r"|\b(?:viet|lam)\b(?:\s+\w+){0,6}\s+\bbai\b(?:\s+\w+){0,4}\s+"
    r"\b(?:ve|nghi luan|cam nhan|phan tich)\b"
)
_SOCIAL = re.compile(
    r"\b(?:nghi luan xa hoi|nghi luan ve mot tu tuong|nghi luan ve hien tuong|"
    r"ban luan ve|trinh bay suy nghi ve|suy nghi cua (?:em|ban) ve)\b"
)
_PARAGRAPH = re.compile(r"\b(?:viet|lam)\b(?:\s+\w+){0,4}\s+\b(?:doan van|doan nghi luan|paragraph)\b")
_FULL_ESSAY = re.compile(
    r"\b(?:viet|lam|soan)\b(?:\s+\w+){0,4}\s+\b(?:bai van|bai nghi luan|bai cam nhan|essay)\b"
    r"|\bwrite (?:an? )?(?:essay|paragraph)\b"
)
_CREATIVE = re.compile(
    r"\b(?:viet|ke|ta|lam|sang tac|tao|compose|write|create)\b(?:\s+\w+){0,10}\s+\b(?:bai van ta|van mieu ta|"
    r"van ke chuyen|cau chuyen|truyen ngan|tieu thuyet|kich ban|tan van|but ky|thu gui|nhat ky|bai tho|"
    r"tho|haiku|tanka|senryu|haibun|sijo|sonnet|villanelle|pantoum|ballad|ode|elegy|spoken word|"
    r"poem|story|short story|screenplay|novel)\b"
)
_COMPONENT = re.compile(
    r"\b(?:viet|goi y|huong dan|cho)\b(?:\s+\w+){0,4}\s+\b(?:mo bai|than bai|ket bai|"
    r"luan diem|cau chu de|cau chuyen doan|phan phan bien)\b"
)
_WORD_COUNT = re.compile(r"\b(\d{2,4})\s*(?:chu|tu|word|words)\b")
_SHORT_REQUEST = re.compile(r"\b(?:ngan gon|viet ngan|tom tat|suc tich|khong can dai)\b")
_WRITING_FOLLOWUP = re.compile(
    r"\b(?:phan tich|viet|lam|trien khai|mo rong|dao sau|sua)\b.*\b(?:dai|sau|chi tiet|ky|hon|them|chu|tu)\b"
    r"|\b(?:dai|sau|chi tiet|ky|ro|hay|bay bong)\s+hon\b"
    r"|\b(?:viet|lam)\b(?:\s+\w+){0,7}\s+\bbai phan tich\b"
    r"|\b(?:viet|lam)\b(?:\s+\w+){0,5}\s+\bbai\s+(?:nay|do|luon|day du|hoan chinh)\b"
)
_WORK_TITLE = re.compile(
    r"\b(?:truyện(?:\s+ngắn)?|tác\s+phẩm|văn\s+bản|bài\s+thơ|tiểu\s+thuyết|"
    r"bi\s+kịch(?:\s+của)?|nhân\s+vật|hình\s+tượng(?:\s+nhân\s+vật)?)\s+"
    r"[\"“”'‘’]?([^,.;:?!\n]{2,90})",
    re.IGNORECASE,
)
_TITLE_STOP = re.compile(
    r"\s+\b(?:của|do|trong|không\s+chỉ|tại\s+sao|vì\s+sao|phân\s+tích|giải\s+thích|"
    r"cho\s+thấy|qua\s+đó)\b.*$",
    re.IGNORECASE,
)


def literary_search_query(text: str) -> str:
    """Rút tên tác phẩm khỏi đề dài để công cụ tìm kiếm không lạc sang từ khóa chung như ``nhân vật``."""
    raw = (text or "").strip()
    match = _WORK_TITLE.search(raw)
    if not match:
        return raw[:600]
    title = _TITLE_STOP.sub("", match.group(1)).strip(" \t\"'“”‘’")
    words = title.split()
    if not title or len(words) > 10:
        return raw[:600]
    # Giữ phần câu hỏi để tìm đúng góc phân tích (nhân vật, chi tiết, thái độ...), còn tiêu đề đặt trong ngoặc kép
    # giúp máy tìm kiếm không lạc sang các tác phẩm có từ khóa chung.
    return f'"{title}" {raw[:450]}'


def requested_word_count(text: str | None) -> int | None:
    """Số chữ người dùng yêu cầu; chặn giá trị phi thực tế để bảo vệ độ trễ/context."""
    match = _WORD_COUNT.search(normalize(text or ""))
    return min(3200, max(100, int(match.group(1)))) if match else None


def continue_writing_task(previous_text: str | None, current_text: str | None) -> WritingTask | None:
    """Kế thừa chế độ Văn cho yêu cầu tiếp nối ngắn như ``phân tích dài 2000 chữ đi``."""
    current = (current_text or "").strip()
    if not current or len(current) > 240 or not _WRITING_FOLLOWUP.search(normalize(current)):
        return None
    previous = detect_writing_task(previous_text)
    if not previous.is_writing:
        return None
    return replace(previous, target_words=requested_word_count(current) or previous.target_words)


def detect_writing_task(text: str | None) -> WritingTask:
    """Trả về tác vụ viết; ``is_writing=False`` cho hội thoại không phải Ngữ văn."""
    if not text:
        return WritingTask(False)
    value = normalize(text)
    creative_profile = detect_creative_profile(value)
    grade_match = _GRADE.search(value)
    grade = int(grade_match.group(1)) if grade_match else None
    target_words = requested_word_count(text)

    if _REVISION.search(value):
        return WritingTask(True, "revision", "feedback", grade, False, target_words=target_words)
    if _READING.search(value):
        # Đọc hiểu thường dựa vào đoạn người dùng cung cấp; model không được bịa đoạn bị thiếu.
        return WritingTask(True, "reading_comprehension", "analysis", grade, False, target_words=target_words)
    literary_request = bool(
        _LITERARY.search(value)
        or _GENERIC_LITERARY_ANALYSIS.search(value)
        or (_LITERARY_ACTION.search(value) and _LITERARY_CONTEXT.search(value))
    )
    if literary_request:
        # Bài phân tích hoàn chỉnh cần đủ không gian cho đọc gần văn bản, nhiều
        # cách hiểu và liên hệ đời sống. Số chữ cụ thể/yêu cầu ngắn luôn thắng.
        literary_target = target_words or (500 if _SHORT_REQUEST.search(value) else 2700)
        return WritingTask(
            True,
            "literary_argument",
            "outline" if _OUTLINE.search(value) else "essay",
            grade,
            True,
            target_words=literary_target,
        )
    if _SOCIAL.search(value):
        return WritingTask(
            True,
            "social_argument",
            "outline" if _OUTLINE.search(value) else "essay",
            grade,
            True,
            target_words=target_words,
        )
    if _CREATIVE.search(value):
        is_poetry = bool(
            re.search(
                r"\b(?:tho|haiku|tanka|senryu|haibun|sijo|sonnet|villanelle|pantoum|ballad|ode|elegy|"
                r"spoken word|poem)\b",
                value,
            )
        )
        return WritingTask(
            True,
            "creative",
            "poem" if is_poetry else "creative_text",
            grade,
            False,
            True,
            creative_profile["tradition"],
            creative_profile["form"],
            creative_profile["tones"],
            creative_profile["language"],
            target_words,
        )
    if _OUTLINE.search(value):
        return WritingTask(True, "outline", "outline", grade, False, target_words=target_words)
    if _PARAGRAPH.search(value):
        return WritingTask(True, "paragraph", "paragraph", grade, False, target_words=target_words)
    if _FULL_ESSAY.search(value):
        return WritingTask(True, "essay", "essay", grade, False, target_words=target_words)
    if _COMPONENT.search(value):
        return WritingTask(True, "component", "component", grade, False, target_words=target_words)
    if _GENERIC_WRITING_REQUEST.search(value):
        return WritingTask(True, "essay", "essay", grade, False, target_words=target_words)
    return WritingTask(False)
