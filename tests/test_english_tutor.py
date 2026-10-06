"""Định tuyến gia sư Tiếng Anh và prompt giải bài theo kỹ năng."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.config import get_settings
from app.conversation.graph import GraphContext, build_graph, run_turn
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.output_check import OutputCheckContext, check_output
from app.conversation.profile import FanpageProfile
from app.english_tutor import build_english_instruction, detect_english_task
from app.providers.llm import FakeProvider


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Giải bài tiếng Anh: She ___ to school every day.", "mixed"),
        ("Giải thích ngữ pháp tiếng Anh về thì hiện tại hoàn thành", "grammar"),
        ("IELTS Writing Task 1: hướng dẫn viết overview", "ielts"),
        ("Làm Reading passage này và chỉ ra evidence", "reading"),
        ("Dịch sang tiếng Anh tự nhiên: Mình rất vui được gặp bạn", "translation"),
        ("Write an English essay about friendship", "writing"),
        ("Luyện Speaking IELTS Part 2", "ielts"),
        ("Giải bài: Choose the correct answer: She ___ every day.", "grammar"),
    ],
)
def test_detects_english_learning_tasks(text: str, kind: str):
    task = detect_english_task(text)
    assert task.is_english and task.kind == kind


def test_high_school_rewrite_and_conditional_are_detected():
    rewrite = detect_english_task(
        "Grade 12 rewrite: She started learning English three years ago. Begin with She has."
    )
    conditional = detect_english_task(
        "Complete and explain: If Nam had checked the map, he ___ lost now. (not/be)"
    )
    assert rewrite.is_english and rewrite.grade == 12
    assert conditional.is_english and conditional.kind == "grammar"


@pytest.mark.parametrize(
    "text",
    [
        "How are you today?",
        "Explain photosynthesis in simple words",
        "Giải phương trình 3x + 5 = 20",
        "Viết bài văn tả mẹ lớp 4",
    ],
)
def test_normal_chat_math_and_literature_are_not_english_subject(text: str):
    assert not detect_english_task(text).is_english


def test_english_prompt_requires_reasoning_and_no_fake_audio_or_band():
    task = detect_english_task("IELTS Writing Task 1, giải thích bằng tiếng Việt")
    instruction = build_english_instruction(task.to_state())
    assert "CHẾ ĐỘ GIA SƯ TIẾNG ANH" in instruction
    assert "không bịa con số" in instruction
    assert "không hứa điểm/band chính thức" in instruction.lower()
    assert "Giải thích bằng tiếng Việt" in instruction
    assert "bài nghe 3-6 phút" in instruction and "phiên âm IPA" in instruction
    assert '"goes" là /ɡoʊz/' in instruction and '"every" thường là /ˈev.ri/' in instruction
    assert "câu trần thuật trung tính thường hạ giọng" in instruction


def test_explicit_english_answer_language_is_respected():
    task = detect_english_task(
        "Giải thích thật kỹ bằng tiếng Anh câu She goes to school every day và luyện phát âm"
    )
    assert task.is_english and task.answer_language == "english"


def test_language_exercise_does_not_get_false_unsourced_warning():
    result = check_output(
        "B. goes. Chủ ngữ she là ngôi thứ ba số ít nên động từ thêm -es.",
        OutputCheckContext(
            needs_disclosure=False,
            is_first_bot_reply=False,
            user_text="Giải bài tiếng Anh: She ___ every day.",
            notifier_configured=False,
            knowledge_question=True,
            english_mode=True,
        ),
    )
    assert "unverified_note_added" not in result.flags


async def test_graph_routes_english_and_injects_specialized_prompt():
    now = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: "B. goes. Chủ ngữ ngôi thứ ba số ít nên động từ thêm -es.")
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "english-student")
    store.add_user_message(
        cid,
        "Giải và giải thích bài tiếng Anh: She ___ to school every day. A. go B. goes C. going D. went",
        ts=now,
    )
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "english"
    assert result["english_mode"] is True
    assert "CHẾ ĐỘ GIA SƯ TIẾNG ANH" in str(provider.calls[0][0].content)


async def test_missing_reading_passage_gets_deterministic_request():
    now = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: "A")
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "reader")
    store.add_user_message(cid, "Giải Reading passage ở trên và chọn đáp án", ts=now)
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "english_missing_text"
    assert not provider.calls
    assert "chưa nhận được đoạn đọc" in " ".join(result["reply_parts"])
