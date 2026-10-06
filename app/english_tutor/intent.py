"""Nhận diện bài Tiếng Anh mà không chuyển nhầm hội thoại tiếng Anh thông thường."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.conversation.intents import normalize


@dataclass(frozen=True, slots=True)
class EnglishTask:
    is_english: bool
    kind: str = "none"
    grade: int | None = None
    answer_language: str = "bilingual"
    requires_text: bool = False
    exam: str | None = None

    def to_state(self) -> dict[str, object]:
        return asdict(self)


_GRADE = re.compile(r"\b(?:lop|grade)\s*(1[0-2]|[1-9])\b")
_EXPLICIT = re.compile(
    r"\b(?:tieng anh|english|ielts|toeic|toefl|cefr|ngu phap|grammar|tu vung|vocabulary|"
    r"phat am|pronunciation|reading passage|listening test|speaking test)\b"
)
_EXERCISE = re.compile(
    r"\b(?:choose the correct|circle the correct|fill in the blanks?|complete(?: and explain| the sentences?)?|"
    r"correct form of the (?:verb|word)|find (?:and correct )?the (?:mistake|error)|"
    r"rewrite(?: the sentences?)?|word formation|sentence transformation|odd one out)\b"
)
_SOLVE_ENGLISH = re.compile(
    r"\b(?:giai|lam|chua|giai thich|dap an|solve|answer|explain)\b(?:\s+\w+){0,5}\s+"
    r"\b(?:bai|de|cau|exercise|test)\s+(?:tap\s+)?(?:tieng anh|english)\b"
)
_IELTS = re.compile(r"\b(?:ielts|task\s*[12]|band\s*[0-9]|academic writing|general training)\b")
_READING = re.compile(
    r"\b(?:reading|doc hieu|passage|true false not given|yes no not given|matching headings|"
    r"reading comprehension)\b"
)
_LISTENING = re.compile(r"\b(?:listening|luyen nghe|bai nghe|audio|transcript)\b")
_SPEAKING = re.compile(r"\b(?:speaking|luyen noi|cue card|part\s*[123]|hoi thoai tieng anh)\b")
_PRONUNCIATION = re.compile(r"\b(?:pronunciation|phat am|ipa|trong am|stress|minimal pairs?)\b")
_GRAMMAR = re.compile(
    r"\b(?:grammar|ngu phap|thi dong tu|chia dong tu|tense|conditional|relative clause|"
    r"passive voice|reported speech|preposition|article|word form|word formation|mixed conditional|find .* error)\b"
)
_VOCAB = re.compile(r"\b(?:vocabulary|tu vung|collocation|phrasal verb|idiom|synonym|antonym)\b")
_TRANSLATION = re.compile(
    r"\b(?:dich|translate|translation)\b(?:\s+\w+){0,6}\s+\b(?:tieng anh|english|tieng viet|vietnamese)\b"
    r"|\b(?:dich sang anh|dich sang viet|translate into)\b"
)
_WRITING = re.compile(
    r"\b(?:english writing|viet tieng anh|write (?:an?\s+)?(?:english\s+)?(?:essay|paragraph|letter|email|report)|"
    r"writing task|cham bai (?:viet )?tieng anh)\b"
)
_ENGLISH_ONLY = re.compile(
    r"\b(?:chi tra loi bang tieng anh|english only|only in english|"
    r"(?:giai thich|tra loi|viet)(?:\s+\w+){0,5}\s+bang tieng anh)\b"
)
_VI_ONLY = re.compile(r"\b(?:giai thich bang tieng viet|tra loi bang tieng viet|vietnamese explanation)\b")
_TEXT_REFERENCE = re.compile(
    r"\b(?:doan van|doan doc|passage|text|bai doc|audio|transcript|doan nay|bai nay|o tren|above)\b"
)


def detect_english_task(text: str | None) -> EnglishTask:
    """Chỉ bật khi có tín hiệu môn/kỹ năng hoặc khuôn bài tập Tiếng Anh rõ ràng."""
    if not text:
        return EnglishTask(False)
    value = normalize(text)
    if not (
        _EXPLICIT.search(value)
        or _EXERCISE.search(value)
        or _SOLVE_ENGLISH.search(value)
        or _WRITING.search(value)
    ):
        return EnglishTask(False)

    grade_match = _GRADE.search(value)
    grade = int(grade_match.group(1)) if grade_match else None
    language = "english" if _ENGLISH_ONLY.search(value) else "vietnamese" if _VI_ONLY.search(value) else "bilingual"
    exam = next((name.upper() for name in ("ielts", "toeic", "toefl") if name in value), None)

    if _IELTS.search(value):
        kind = "ielts"
    elif _READING.search(value):
        kind = "reading"
    elif _LISTENING.search(value):
        kind = "listening"
    elif _SPEAKING.search(value):
        kind = "speaking"
    elif _PRONUNCIATION.search(value):
        kind = "pronunciation"
    elif _TRANSLATION.search(value):
        kind = "translation"
    elif _WRITING.search(value):
        kind = "writing"
    elif _VOCAB.search(value):
        kind = "vocabulary"
    elif _GRAMMAR.search(value) or _EXERCISE.search(value):
        kind = "grammar"
    else:
        kind = "mixed"

    requires_text = kind in {"reading", "listening"} and bool(_TEXT_REFERENCE.search(value))
    return EnglishTask(True, kind, grade, language, requires_text, exam)
