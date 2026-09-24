"""Hồi quy cho các lỗi thấy trong tin nhắn Messenger thật ngày 24/09/2026."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config import get_settings
from app.conversation.datecalc import answer_date_question
from app.conversation.graph import GraphContext, build_graph, run_turn
from app.conversation.knowledge import KnowledgeBase
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.output_check import OutputCheckContext, check_output
from app.conversation.profile import FanpageProfile
from app.conversation.prompts import build_system_prompt, capabilities_text
from app.providers.llm import FakeProvider

NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)  # Thứ Năm 24/09/2026 (giờ VN)


@pytest.mark.parametrize(
    ("q", "expect"),
    [
        ("20 ngày nữa là thứ mấy", "Thứ Tư, 14/10/2026"),  # bot từng trả lời sai "Thứ Sáu 6/10/2026"
        ("hôm nay thứ mấy", "Thứ Năm, 24/09/2026"),
        ("ngày mai là ngày nào", "Thứ Sáu, 25/09/2026"),
        ("2 tuần trước là ngày bao nhiêu", "Thứ Năm, 10/09/2026"),
        ("1 tháng nữa là thứ mấy", "Thứ Bảy, 24/10/2026"),
        ("ngày 2/9/2026 là thứ mấy", "Thứ Tư"),
        ("ngay 30 thang 2 la thu may", "không tồn tại"),
    ],
)
def test_date_questions_are_computed(q, expect):  # type: ignore[no-untyped-def]
    assert expect in answer_date_question(q, NOW)


@pytest.mark.parametrize("q", ["mai mình đi học", "20 ngày nữa mình thi", "RAG là gì", "hôm nay mình buồn"])
def test_non_date_questions_are_not_captured(q):  # type: ignore[no-untyped-def]
    assert answer_date_question(q, NOW) is None


def _ctx(user_text: str, **kw) -> OutputCheckContext:  # type: ignore[no-untyped-def]
    return OutputCheckContext(
        needs_disclosure=False, is_first_bot_reply=False, user_text=user_text, notifier_configured=False, **kw
    )


def test_unsourced_film_title_list_is_blocked():
    draft = "Mình gợi ý vài phim: *Đất nước*, *Tình ca số 9*, *Hà Nội 1945*. Chúc bạn xem vui!"
    r = check_output(draft, _ctx("Phim chiến tranh Việt Nam đi"))
    assert r.blocked and "unsourced_titles_blocked" in r.flags and "Tình ca số 9" not in r.text


def test_title_list_allowed_when_cited_or_from_question_or_image():
    cited = check_output(
        'Có "Mưa đỏ" và "Mùi cỏ cháy" [1].', _ctx("phim chiến tranh", sources=[("Mưa đỏ", "x")])
    )
    assert not cited.blocked
    echoed = check_output(
        '"Mưa đỏ" hay hơn "Mai" theo nhiều người.', _ctx('phim "Mưa đỏ" và "Mai" cái nào hay')
    )
    assert "unsourced_titles_blocked" not in echoed.flags
    ocr = check_output('Bìa sách ghi "Toán 6" và "Tập 1".', _ctx("sách này tên gì", has_image_context=True))
    assert not ocr.blocked


def test_capabilities_reflect_enabled_services():
    on = capabilities_text(vision=True, docs=True)
    assert "đọc chữ trong ảnh" in on and "tài liệu công khai" in on and "xem ảnh" not in on
    off = capabilities_text(vision=False, docs=False)
    assert "xem ảnh" in off and "đọc chữ trong ảnh" not in off
    prompt = build_system_prompt(
        FanpageProfile(), NOW, summary=None, needs_disclosure=False, vision_available=True
    )
    assert "Khả năng của bạn" in prompt and "đọc tệp PDF/Word" in prompt


async def _two_turns(responder, second_text: str):  # type: ignore[no-untyped-def]
    clock = [NOW]
    store = InMemoryConversationStore(now=lambda: clock[0])
    provider = FakeProvider(responder)
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        knowledge=KnowledgeBase(),
        now=lambda: clock[0],
    )
    cid = store.ensure_conversation("PAGE1", "u1")
    out = []
    for text in ["câu hỏi thứ nhất về học tập", second_text]:
        store.add_user_message(cid, text, ts=clock[0] - timedelta(seconds=1))
        tid = store.begin_turn(cid)
        r = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)
        store.apply_result(cid, tid, r)
        out.append(r)
        clock[0] += timedelta(seconds=30)
    return out, provider


async def test_verbatim_repeat_of_previous_reply_is_regenerated():
    same = "Đây là một câu trả lời khá dài về phương pháp học tập hiệu quả cho học sinh cấp ba nhé."
    answers = iter([same, same, "Câu trả lời mới đúng câu hỏi thứ hai."])
    (_, r2), p = await _two_turns(lambda m: next(answers), "câu hỏi thứ hai khác hẳn")
    assert "repeat_regenerated" in r2["check_flags"] and "Câu trả lời mới" in r2["reply_parts"][0]
    assert "LƯU Ý: bản nháp trước lặp lại" in str(p.calls[-1][0].content)


async def test_date_question_is_answered_without_llm():
    (_, r2), p = await _two_turns(
        lambda m: "ok nha, mình trả lời câu thứ nhất đây.", "20 ngày nữa là thứ mấy"
    )
    assert r2["mode"] == "date_calc" and len(p.calls) == 1 and "14/10/2026" in r2["reply_parts"][0]


@pytest.mark.db
async def test_image_only_message_waits_longer_for_caption(db):
    from sqlalchemy import text

    from app.messenger.events import UserMessage
    from app.storage.db import session_scope
    from app.storage.repository import ingest_events

    s = get_settings().model_copy(
        update={"debounce_seconds": 0.5, "debounce_attachment_seconds": 6.0, "debounce_max_seconds": 2.0}
    )
    img = UserMessage(
        page_id="PAGE1",
        psid="u9",
        mid="m-img",
        ts=datetime.now(UTC),
        text=None,
        attachment_types=["image"],
        images=[{"url": "https://scontent.xx.fbcdn.net/a.jpg"}],
    )
    async with session_scope() as db_:
        await ingest_events(db_, [img], s)
        wait = (
            await db_.execute(text("SELECT extract(epoch FROM next_run_at - now()) FROM conversations"))
        ).scalar()
    assert 5.0 < wait <= 6.1


async def test_new_image_data_is_in_current_user_message_not_only_system(http_mock):
    class V:
        async def analyze(self, data):  # type: ignore[no-untyped-def]
            return {}

    gpu = {"ocr": {"text": "GEFORCE RTX 5080", "lines": 1}, "objects": []}
    person = {
        "ocr": {"text": "", "lines": 0},
        "objects": [{"label_vi": "người", "conf": 0.92, "model": "coco"}],
    }
    clock = [NOW]
    store = InMemoryConversationStore(now=lambda: clock[0])
    p = FakeProvider(lambda m: f"trả lời {len(p.calls)}")
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=p,
        profile=FanpageProfile(),
        knowledge=KnowledgeBase(),
        vision=V(),
        now=lambda: clock[0],
    )
    cid = store.ensure_conversation("PAGE1", "u1")
    for a in (gpu, person):
        store.add_user_message(
            cid,
            None,
            attachment_types=["image"],
            images=[{"analysis": a}],
            ts=clock[0] - timedelta(seconds=1),
        )
        tid = store.begin_turn(cid)
        store.apply_result(
            cid, tid, await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)
        )
        clock[0] += timedelta(seconds=40)
    last_human = str(p.calls[-1][-1].content)
    assert "[Ảnh MỚI gửi kèm tin nhắn này" in last_human and "người (0.92)" in last_human
    assert "RTX 5080" not in last_human  # ảnh cũ chỉ nằm trong lịch sử


def test_stale_release_claim_without_source_is_removed():
    r = check_output(
        "Ảnh có vẻ là card RTX 5080 (chưa ra mắt theo thông tin hiện tại), hỗ trợ DLSS 4. Bạn cần gì thêm?",
        _ctx("đây là gì"),
    )
    assert "chưa ra mắt" not in r.text and "stale_claim_removed" in r.flags and "DLSS 4" in r.text


def test_capability_answer_with_numbers_gets_no_unverified_note():
    r = check_output(
        "Mình đọc chữ trong ảnh (tối đa 8 MB, 3 ảnh mỗi tin) và nhận diện khoảng 90 loại đồ vật.",
        _ctx("bạn làm được những gì"),
    )
    assert "unverified_note_added" not in r.flags


def test_leftover_example_parenthesis_is_cleaned():
    r = check_output(
        "Bạn nên xem trang chính thức của Bộ GD&ĐT (ví dụ https://fake.example.vn) để cập nhật.",
        _ctx("xem ở đâu"),
    )
    assert "(ví dụ" not in r.text and "fake.example" not in r.text


def test_stale_release_paragraph_removed_but_self_uncertainty_kept():
    draft = (
        "Ảnh có chữ GeForce RTX 5080 và DLSS 4.\n\nTuy nhiên, năm 2026, chưa có thông tin nào xác nhận dòng RTX "
        "5080 đã ra mắt. Dòng RTX 5000 là dự kiến cho tương lai.\n\nVì vậy, thông tin này có thể là giả định."
        "\n\nBạn muốn tìm hiểu DLSS 4 không?"
    )
    r = check_output(draft, _ctx("card này là loại nào"))
    assert "RTX 5080 và DLSS 4" in r.text and "Bạn muốn tìm hiểu" in r.text
    assert "dự kiến" not in r.text and "giả định" not in r.text and "chưa có thông tin" not in r.text
    assert "\n\n" in r.text
    kept = check_output(
        "Mình chưa có nguồn kiểm chứng về ngày phát hành phim này.", _ctx("phim X ra mắt khi nào")
    )
    assert "ngày phát hành" in kept.text and "stale_claim_removed" not in kept.flags


def test_image_related_answer_gets_image_note_instead_of_document_note():
    from app.conversation.prompts import IMAGE_UNVERIFIED_NOTE, UNVERIFIED_NOTE

    r = check_output(
        "Theo hệ thống nhận diện, có 1 người (độ tin cậy 0.92).", _ctx("có mấy người", recent_image=True)
    )
    assert r.text.endswith(IMAGE_UNVERIFIED_NOTE) and UNVERIFIED_NOTE not in r.text


def test_model_written_note_is_replaced_by_single_system_note():
    from app.conversation.prompts import IMAGE_UNVERIFIED_NOTE

    draft = f"Có 1 người (0.92).\n\n(Lưu ý: nhận diện từ ảnh có thể sai.)\n\n{IMAGE_UNVERIFIED_NOTE}"
    r = check_output(draft, _ctx("có mấy người", recent_image=True))
    assert r.text.count("Lưu ý") == 1 and r.text.endswith(IMAGE_UNVERIFIED_NOTE)


def test_removed_claims_keep_line_breaks_and_drop_dangling_lead_in():
    draft = (
        "Đây là card RTX 5080 của NVIDIA.\n\nTuy nhiên, cần lưu ý:\nSản phẩm này chưa được xác nhận ra mắt.\n"
        "Vì vậy có thể là bản quảng cáo hoặc chưa chính thức.\n\nBạn muốn tìm hiểu DLSS 4 không?"
    )
    r = check_output(draft, _ctx("card này là gì"))
    assert "cần lưu ý" not in r.text and "chưa chính thức" not in r.text
    assert r.text.startswith("Đây là card RTX 5080 của NVIDIA.\n\nBạn muốn tìm hiểu DLSS 4 không?")


async def test_image_followup_retrieves_sources_by_earlier_ocr_text(http_mock):
    from app.conversation.knowledge import load_knowledge

    class V:
        async def analyze(self, data):  # type: ignore[no-untyped-def]
            return {}

    gpu = {"ocr": {"text": "GEFORCE RTX\nnVIDIA\n5080\nDLSS 4", "lines": 4}, "objects": []}
    clock = [NOW]
    store = InMemoryConversationStore(now=lambda: clock[0])
    p = FakeProvider(lambda m: "Đây là card GeForce RTX 5080 [1].")
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=p,
        profile=FanpageProfile(),
        knowledge=load_knowledge(Path("config/knowledge")),
        vision=V(),
        now=lambda: clock[0],
    )
    cid = store.ensure_conversation("PAGE1", "u1")
    r = None
    for text_, imgs in ((None, [{"analysis": gpu}]), ("cái card lúc nãy ra mắt khi nào", [])):
        store.add_user_message(
            cid,
            text_,
            attachment_types=["image"] if imgs else [],
            images=imgs,
            ts=clock[0] - timedelta(seconds=1),
        )
        tid = store.begin_turn(cid)
        r = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)
        store.apply_result(cid, tid, r)
        clock[0] += timedelta(seconds=40)
    assert "nvidia-geforce-rtx-50" in [h["id"] for h in r["knowledge_hits"]]


def test_no_official_info_claim_is_stale_but_self_uncertainty_kept():
    draft = (
        "Mình không có thông tin mới về GPT-5. Các bản GPT-4 vẫn được dùng rộng rãi, nhưng không có thông tin chính "
        "thức về GPT-5."
    )
    r = check_output(draft, _ctx("GPT-5 ra mắt chưa"))
    assert "Mình không có thông tin mới về GPT-5." in r.text and "không có thông tin chính thức" not in r.text


def test_release_status_question_is_knowledge_question():
    from app.conversation.intents import is_knowledge_question

    assert is_knowledge_question("GPT-5 ra mắt chưa") and is_knowledge_question("phim đó phát hành chưa")


def test_cited_answer_still_loses_unsourced_stale_claim_and_derived_duration():
    src = [("RTX 50", "https://nvidianews.nvidia.com/x")]
    draft = (
        "RTX 5080 bán từ 30/1/2025 [1]. Đây là dòng card mới, chưa ra mắt rộng rãi tại Việt Nam [1]. "
        "Vậy nó đã ra mắt cách đây hơn 1 năm 8 tháng."
    )
    r = check_output(
        draft, _ctx("ra mắt khi nào", sources=src, source_text="GeForce RTX 5080 bán từ ngày 30/1/2025")
    )
    assert "30/1/2025" in r.text and "chưa ra mắt" not in r.text and "cách đây" not in r.text
    kept = check_output(
        "Phim dự kiến ra mắt tháng 12/2026 [1].",
        _ctx("phim ra mắt khi nào", sources=src, source_text="Phim dự kiến ra mắt tháng 12/2026."),
    )
    assert "dự kiến ra mắt" in kept.text
