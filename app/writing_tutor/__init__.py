"""Gia sư Ngữ văn: nhận diện tác vụ, hướng dẫn sinh bài và rubric đánh giá."""

from app.writing_tutor.intent import (
    WritingTask,
    continue_writing_task,
    detect_writing_task,
    literary_search_query,
    requested_word_count,
)
from app.writing_tutor.prompt import build_writing_instruction

__all__ = [
    "WritingTask",
    "build_writing_instruction",
    "continue_writing_task",
    "detect_writing_task",
    "literary_search_query",
    "requested_word_count",
]
