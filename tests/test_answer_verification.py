"""Cổng kiểm định lần hai phải fail-closed; Văn sáng tạo được giữ, phân tích có nguồn được đối chiếu."""

from __future__ import annotations

from datetime import UTC, datetime

from app.config import get_settings
from app.conversation.graph import GraphContext, build_graph, run_turn
from app.conversation.knowledge import KnowledgeEntry
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.profile import FanpageProfile
from app.conversation.verification import (
    MATH_VERIFICATION_FAILED_REPLY,
    grounded_expansion_messages,
    grounded_verification_messages,
    parse_verification,
)
from app.providers.llm import FakeProvider


def test_verification_protocol_is_fail_closed():
    assert parse_verification("không phải JSON") is None
    assert parse_verification('{"verdict":"pass","answer":"ngắn","reason":"ok"}') is None
    assert (
        parse_verification(
            '{"verdict":"pass","answer":"Khẳng định đúng nhưng không có citation nào.","reason":"ok"}',
            source_count=1,
        )
        is None
    )
    valid = parse_verification(
        '{"verdict":"corrected","answer":"Kết quả đã sửa và có đủ bước kiểm tra [1].",'
        '"reason":"Bản nháp thiếu điều kiện"}',
        source_count=1,
    )
    assert valid is not None and valid.verdict == "corrected"


def test_literary_verifier_preserves_depth_and_distinguishes_inference():
    messages = grounded_verification_messages(
        "Phân tích Chí Phèo",
        "Bản nháp",
        [KnowledgeEntry("web:1", "Chí Phèo", "https://example.org", "Nội dung nguồn")],
        literary=True,
    )
    system = str(messages[0].content)
    assert "2.500-3.000 chữ" in system and "có thể hiểu" in system
    assert "không xuất hiện nguyên văn trong nguồn" in system
    assert "KHÔNG ĐƯỢC trả verdict insufficient" in system
    assert '"verdict":"pass|corrected"' in system


def test_literary_verifier_preserves_requested_long_form_depth():
    messages = grounded_verification_messages(
        "Phân tích bài thơ Lượm dài 2000 chữ",
        "Bản nháp",
        [KnowledgeEntry("web:1", "Lượm", "https://example.org", "Nội dung nguồn")],
        literary=True,
    )
    system = str(messages[0].content)
    assert "khoảng 2000 chữ" in system and "sai lệch tối đa 10%" in system
    assert "chi tiết/hình ảnh" in system and "kỹ thuật tự sự" in system


def test_literary_verifier_uses_runtime_default_target():
    messages = grounded_verification_messages(
        "Phân tích nhân vật trong tác phẩm",
        "Bản nháp",
        [KnowledgeEntry("web:1", "Tác phẩm", "https://example.org", "Nội dung nguồn")],
        literary=True,
        target_words=2700,
    )
    assert "khoảng 2700 chữ" in str(messages[0].content)


def test_grounded_expansion_requests_only_missing_analysis():
    messages = grounded_expansion_messages(
        "Phân tích Lượm khoảng 2000 chữ",
        "Bài hiện có [1].",
        [KnowledgeEntry("web:1", "Lượm", "https://example.org", "Nội dung nguồn")],
        target_words=2000,
        current_words=1150,
    )
    system = str(messages[0].content)
    assert "PHẦN BỔ SUNG" in system and "không chép lại" in system
    assert "khoảng 850 chữ" in system and "không kể lại cốt truyện" in system


def test_social_verifier_compares_sources_without_overtrusting_weak_sources():
    messages = grounded_verification_messages(
        "Nghị luận xã hội về mạng xã hội",
        "Bản nháp",
        [KnowledgeEntry("web:1", "Một bài báo", "https://example.org", "Nội dung nguồn")],
        social=True,
    )
    system = str(messages[0].content)
    assert "350-600 từ" in system and "đồng thuận và bất đồng" in system
    assert "blog hoặc bài mẫu" in system and "không được nâng thành bằng chứng chắc chắn" in system


async def _run(text: str, responder):  # type: ignore[no-untyped-def]
    now = datetime(2026, 10, 5, 9, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(responder)
    settings = get_settings().model_copy(update={"answer_verification_enabled": True})
    ctx = GraphContext(
        settings=settings, store=store, llm=provider, profile=FanpageProfile(), now=lambda: now
    )
    cid = store.ensure_conversation("PAGE1", "accuracy-test")
    store.add_user_message(cid, text, ts=now)
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)
    return result, provider


async def test_unsupported_math_cannot_be_certified_by_two_llm_answers():
    def responder(messages):  # type: ignore[no-untyped-def]
        if "bộ kiểm định Toán độc lập" in str(messages[0].content):
            return (
                '{"verdict":"corrected","answer":"Kết luận: chưa thể chứng minh từ dữ kiện đã cho. '
                'Kiểm tra: đề thiếu giả thiết cần thiết.","reason":"Bản nháp kết luận quá mức"}'
            )
        return "Đáp án chắc chắn là đúng."

    result, provider = await _run("Chứng minh tam giác ABC cân", responder)
    assert result["mode"] == "math_unverified"
    assert "math_exact_verification_required" in result["check_flags"]
    assert MATH_VERIFICATION_FAILED_REPLY in " ".join(result["reply_parts"])
    assert not provider.calls


async def test_invalid_math_verifier_output_refuses_instead_of_using_draft():
    replies = iter(["Đáp án x = 123.", "không phải JSON"])
    result, provider = await _run("Chứng minh định lý Fermat lớn", lambda _: next(replies))
    sent = " ".join(result["reply_parts"])
    assert MATH_VERIFICATION_FAILED_REPLY in sent
    assert "x = 123" not in sent
    assert "math_exact_verification_required" in result["check_flags"]
    assert not provider.calls


async def test_english_and_creative_writing_are_not_sent_to_math_verifier():
    english, en_provider = await _run(
        "Giải bài tiếng Anh: She ___ every day. A. go B. goes", lambda _: "B. goes, vì chủ ngữ số ít."
    )
    poem, poem_provider = await _run(
        "Tạo một bài thơ tự do về mưa", lambda _: "Mưa nghiêng qua cửa\nĐêm nở chậm."
    )
    assert english["mode"] == "english" and len(en_provider.calls) == 1
    assert poem["mode"] == "writing" and len(poem_provider.calls) == 1
