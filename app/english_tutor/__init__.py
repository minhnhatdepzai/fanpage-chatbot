"""Gia sư Tiếng Anh: nhận diện bài tập và dựng chỉ dẫn theo kỹ năng."""

from app.english_tutor.intent import EnglishTask, detect_english_task
from app.english_tutor.prompt import ENGLISH_TEXT_REQUIRED_REPLY, build_english_instruction

__all__ = [
    "ENGLISH_TEXT_REQUIRED_REPLY",
    "EnglishTask",
    "build_english_instruction",
    "detect_english_task",
]
