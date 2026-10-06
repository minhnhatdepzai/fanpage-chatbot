"""Keep detailed-answer guidance compatible with short greetings and grounding."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.conversation.knowledge import load_knowledge
from app.conversation.profile import FanpageProfile
from app.conversation.prompts import PROMPT_VERSION, build_system_prompt


def test_runtime_prompt_has_consistent_detailed_style_and_safety():
    prompt = build_system_prompt(FanpageProfile(), datetime.now(UTC), summary=None, needs_disclosure=True)
    assert PROMPT_VERSION == "edu-ent-vi-v19-evidence-bounded-writing"
    assert "220-380 từ" in prompt
    assert "2-4 ý thực sự hữu ích" in prompt
    assert "gọn 1-3 câu" in prompt
    assert "Toán lớp 10-12" in prompt
    assert "không bịa câu thơ" in prompt
    assert "1-4 câu" not in prompt
    assert "Độ chính xác luôn quan trọng hơn độ dài" in prompt
    assert "KHÔNG bịa" in prompt
    assert "tách rõ: sự kiện được nguồn ghi nhận" in prompt
    assert "Không quy kết động cơ cho cả" in prompt
    assert "nguồn yếu chỉ dùng như góc nhìn" in prompt
    assert "Không tiết lộ" in prompt
    assert "chưa có nguồn kiểm chứng" in prompt


@pytest.mark.parametrize("query", ["RAG khác fine-tuning thế nào?", "fine tuning là gì", "finetuning"])
def test_fine_tuning_reference_is_retrievable(query):
    kb = load_knowledge(Path(__file__).resolve().parents[1] / "config" / "knowledge")
    hits = kb.search(query, top_k=3)
    entry = next(hit.entry for hit in hits if hit.entry.id == "fine-tuning-basics")
    assert entry.source == "https://huggingface.co/docs/transformers/training"
    assert "không phải huấn luyện mô hình từ đầu" in entry.content
