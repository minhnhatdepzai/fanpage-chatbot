"""Độ chính xác: kho kiến thức có nguồn, truy xuất, trích nguồn, chặn link bịa, ghi chú khi chưa kiểm chứng."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.config import get_settings
from app.config.settings import GroundingMode
from app.conversation.graph import GraphContext, build_graph, run_turn
from app.conversation.intents import is_knowledge_question
from app.conversation.knowledge import load_knowledge
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.output_check import OutputCheckContext, check_output
from app.conversation.profile import FanpageProfile
from app.conversation.prompts import NO_SOURCE_REPLY, UNVERIFIED_NOTE
from app.providers.llm import FakeProvider

KB = load_knowledge(get_settings().knowledge_dir)
LORA_URL = "https://arxiv.org/abs/2106.09685"


def test_entries_without_source_or_with_duplicate_id_are_skipped():
    assert [e.id for e in KB.entries] == ["lora", "dpo"]
    assert KB.source_urls() == {LORA_URL, "https://arxiv.org/abs/2305.18290"}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("LoRA là gì?", ["lora"]),
        ("lora la gi", ["lora"]),
        ("DPO khác gì LoRA", ["dpo", "lora"]),
        ("reward model dùng để làm gì", ["dpo"]),
        ("giải thích sở thích của bạn", []),  # không khớp âm tiết rời rạc
        ("chào bạn", []),
        ("", []),
    ],
)
def test_search_requires_keyword_phrase(query, expected):  # type: ignore[no-untyped-def]
    assert sorted(h.entry.id for h in KB.search(query)) == sorted(expected)


@pytest.mark.parametrize(
    "text",
    [
        "LoRA là gì?",
        "tại sao trời xanh",
        "ai phát minh ra máy tính",
        "so sánh LoRA và QLoRA",
        "what is RAG",
        "kể về Charles Babbage đi",
        "kể mình nghe về Alan Turing",
        "cho mình biết lịch sử AI",
        "Việt Nam có bao nhiêu tỉnh thành?",
        "năm nay thi tốt nghiệp mấy môn",
        "phim việt nào doanh thu cao nhất",
        "model ai mới nhất là gì",
    ],
)
def test_knowledge_questions_detected(text):  # type: ignore[no-untyped-def]
    assert is_knowledge_question(text)


@pytest.mark.parametrize(
    "text",
    ["chào bạn", "hôm nay mình buồn quá", "tên mình là gì?", "bạn là ai", "bạn khỏe không", "cảm ơn nhé"],
)
def test_chitchat_is_not_knowledge_question(text):  # type: ignore[no-untyped-def]
    assert not is_knowledge_question(text)


def _ctx(**kw) -> OutputCheckContext:  # type: ignore[no-untyped-def]
    base = {
        "needs_disclosure": False,
        "is_first_bot_reply": False,
        "user_text": "LoRA là gì?",
        "notifier_configured": False,
        "sources": [("LoRA (2021)", LORA_URL)],
        "allowed_urls": {LORA_URL},
        "knowledge_question": True,
    }
    base.update(kw)
    return OutputCheckContext(**base)


def test_valid_citation_gets_system_source_footer():
    r = check_output("LoRA đóng băng trọng số gốc [1].", _ctx())
    assert r.text.startswith("LoRA đóng băng trọng số gốc [1].")
    assert f"Nguồn:\n[1] LoRA (2021): {LORA_URL}" in r.text
    assert "sources_attached" in r.flags and UNVERIFIED_NOTE not in r.text


def test_multi_source_writing_attaches_every_retrieved_source():
    sources = [
        ("Nguồn học thuật", "https://example.org/research"),
        ("Bài báo", "https://example.net/news"),
        ("Bài mẫu", "https://example.edu.vn/bai-mau"),
    ]
    r = check_output(
        "Có thể nhìn vấn đề từ nhiều phía [1].",
        _ctx(
            user_text="Nghị luận xã hội về mạng xã hội",
            sources=sources,
            allowed_urls={url for _, url in sources},
            writing_mode=True,
            attach_all_sources=True,
        ),
    )
    assert r.text
    assert all(url in r.text for _, url in sources)
    assert {"sources_attached", "all_sources_attached"} <= set(r.flags)
    assert UNVERIFIED_NOTE not in r.text


def test_invalid_citation_and_fabricated_link_are_removed():
    r = check_output(
        "LoRA giúp fine-tune rẻ hơn. Chi tiết xem https://fake-paper.example.com/lora nhé [3].", _ctx()
    )
    assert "[3]" not in r.text and "fake-paper" not in r.text
    assert {"invalid_citation_removed", "unverified_url_removed", "unverified_note_added"} <= set(r.flags)


def test_model_written_source_line_is_replaced_by_verified_footer():
    r = check_output("LoRA giảm tham số huấn luyện [1].\nNguồn: Wikipedia, blog ABC", _ctx())
    assert "Wikipedia" not in r.text and "blog ABC" not in r.text
    assert LORA_URL in r.text and "model_source_line_removed" in r.flags


def test_allowed_link_is_kept():
    r = check_output(f"Bạn đọc bài gốc ở {LORA_URL} nhé [1].", _ctx())
    assert r.text.count(LORA_URL) >= 1 and "unverified_url_removed" not in r.flags


def test_unsourced_knowledge_answer_gets_note_but_stated_uncertainty_does_not():
    r = check_output("Máy tính điện tử đầu tiên ra đời năm 1946.", _ctx(sources=[], allowed_urls=set()))
    assert r.text.endswith(UNVERIFIED_NOTE) and "unverified_note_added" in r.flags
    r2 = check_output("Mình chưa có nguồn kiểm chứng cho câu này nên không chắc.", _ctx(sources=[]))
    assert UNVERIFIED_NOTE not in r2.text and "uncertainty_stated" in r2.flags
    # nói "không có nguồn" nhưng vẫn khẳng định dài dòng -> vẫn phải có ghi chú
    long_claim = "Mình không có nguồn kiểm chứng. " + "Máy tính điện tử đầu tiên ra đời năm 1946 ở Mỹ. " * 4
    r3 = check_output(long_claim, _ctx(sources=[], allowed_urls=set()))
    assert r3.text.endswith(UNVERIFIED_NOTE)
    # khẳng định "chưa ra mắt" không nguồn (kiến thức cũ của model) -> bị bỏ hẳn
    stale = (
        "Mình không có nguồn kiểm chứng. Hiện chưa có thông tin chính thức nào xác nhận sản phẩm X ra mắt."
    )
    r4 = check_output(stale, _ctx(sources=[], allowed_urls=set()))
    assert "ra mắt" not in r4.text and "stale_claim_removed" in r4.flags


def test_fake_verification_claim_is_removed_and_note_added():
    draft = "Việt Nam có 63 tỉnh, thành phố.\n*(Lưu ý: Thông tin này được kiểm chứng từ nguồn chính thức của Chính phủ.)*"
    r = check_output(draft, _ctx(user_text="Việt Nam có bao nhiêu tỉnh?", sources=[], allowed_urls=set()))
    assert "kiểm chứng từ nguồn chính thức" not in r.text and "false_verification_claim_removed" in r.flags
    assert r.text.endswith(UNVERIFIED_NOTE)


def test_reply_that_is_only_a_fake_sourced_claim_becomes_fallback():
    r = check_output("Theo nghiên cứu [3], điều đó đúng.", _ctx(sources=[], allowed_urls=set()))
    assert r.text is None and "false_verification_claim_removed" in r.flags


def test_negated_verification_wording_is_kept():
    r = check_output("Ý này mình chưa được kiểm chứng nên không chắc.", _ctx(sources=[], allowed_urls=set()))
    assert "false_verification_claim_removed" not in r.flags and "chưa được kiểm chứng" in r.text


def test_unsourced_figures_get_note_even_without_question_cue():
    r = check_output(
        "Dân số khoảng 100 triệu người.", _ctx(user_text="kể nghe đi", knowledge_question=False, sources=[])
    )
    assert r.text.endswith(UNVERIFIED_NOTE)


def test_removed_citation_leaves_no_space_before_punctuation():
    r = check_output("Charles Babbage thiết kế máy vào thế kỷ 19 [4].", _ctx(sources=[], allowed_urls=set()))
    assert "thế kỷ 19." in r.text


def test_chitchat_reply_is_untouched():
    r = check_output(
        "Mình ở đây nghe bạn nè.", _ctx(user_text="buồn quá", knowledge_question=False, sources=[])
    )
    assert r.text == "Mình ở đây nghe bạn nè." and r.flags == []


# ------------------------------------------------------------------------------------------ graph
class Harness:
    def __init__(self, provider: FakeProvider, mode: GroundingMode = GroundingMode.annotate):
        self.clock = [datetime(2026, 9, 23, 10, 0, tzinfo=UTC)]
        self.store = InMemoryConversationStore(now=lambda: self.clock[0])
        self.provider = provider
        self.settings = get_settings().model_copy(
            update={"grounding_mode": mode, "answer_verification_enabled": False}
        )
        self.graph = build_graph()

    async def say(self, psid: str, text: str) -> dict:
        cid = self.store.ensure_conversation("PAGE1", psid)
        self.store.add_user_message(cid, text, ts=self.clock[0])
        tid = self.store.begin_turn(cid)
        ctx = GraphContext(
            settings=self.settings,
            store=self.store,
            llm=self.provider,
            profile=FanpageProfile(name="AI Test"),
            knowledge=KB,
            now=lambda: self.clock[0],
        )
        result = await run_turn(self.graph, turn_id=tid, conversation_id=cid, context=ctx)
        result["sent"] = self.store.apply_result(cid, tid, result)
        self.clock[0] += timedelta(seconds=30)
        return result


async def test_graph_puts_sources_in_prompt_and_attaches_footer():
    h = Harness(FakeProvider(lambda m: "LoRA đóng băng trọng số gốc và thêm ma trận hạng thấp [1]."))
    r = await h.say("u1", "LoRA là gì vậy?")
    system = str(h.provider.calls[-1][0].content)
    assert "[1] LoRA (2021)" in system and "ma trận hạng thấp" in system
    sent = "\n".join(r["sent"])
    assert LORA_URL in sent and "sources_attached" in r["check_flags"]


async def test_graph_unsourced_knowledge_question_gets_note_in_annotate_mode():
    h = Harness(FakeProvider(lambda m: "Máy tính đầu tiên là ENIAC."))
    r = await h.say("u1", "ai phát minh ra máy tính?")
    assert "Nguồn tham khảo cho tin nhắn này: (không có)" in str(h.provider.calls[-1][0].content)
    assert UNVERIFIED_NOTE in "\n".join(r["sent"])


async def test_graph_strict_mode_does_not_guess_without_source():
    h = Harness(FakeProvider(lambda m: "đoán bừa"), mode=GroundingMode.strict)
    r = await h.say("u1", "ai phát minh ra máy tính?")
    assert NO_SOURCE_REPLY in "\n".join(r["sent"]) and h.provider.calls == []
    r2 = await h.say("u1", "chào bạn")  # trò chuyện thường vẫn gọi model
    assert r2["action"] == "reply" and len(h.provider.calls) == 1


async def test_graph_follow_up_question_reuses_previous_topic_for_retrieval():
    h = Harness(FakeProvider(lambda m: "Ý bạn là LoRA nhỉ [1]."))
    await h.say("u1", "LoRA là gì?")
    r = await h.say("u1", "vậy nó hoạt động như thế nào?")
    assert [x["id"] for x in r["knowledge_hits"]] == ["lora"]


def test_markdown_is_stripped_for_messenger():
    r = check_output("### Ý chính\nVeRA dùng **một cặp ma trận chung** [1].", _ctx())
    assert "**" not in r.text and "###" not in r.text and "một cặp ma trận chung" in r.text


# ------------------------------------------------------------------------------------------ meta set-webhook
async def test_set_webhook_posts_expected_fields_and_never_prints_secrets():
    import httpx
    import respx

    from app.messenger.meta_check import set_webhook

    s = get_settings()
    base = f"{s.meta_graph_base_url}/{s.meta_graph_api_version}"
    with respx.mock(assert_all_called=True) as mock:
        app_route = mock.post(f"{base}/{s.meta_app_id}/subscriptions").mock(
            return_value=httpx.Response(200, json={"success": True})
        )
        mock.post(f"{base}/{s.meta_page_id}/subscribed_apps").mock(
            return_value=httpx.Response(
                403, json={"error": {"message": "needs pages_manage_metadata", "code": 200}}
            )
        )
        out = await set_webhook(s, "https://example.trycloudflare.com/webhook")
    sent = app_route.calls[0].request.content.decode()
    assert "object=page" in sent and "messages" in sent and "callback_url=https" in sent
    assert out["ok"] and out["page_subscription"]["status"] == 403
    dumped = str(out)
    for secret in s.secret_values():
        assert secret not in dumped


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hi! Có gì muốn hỏi về AI không?", False),  # nhắc AI như chủ đề -> chưa công bố
        ("AI là trí tuệ nhân tạo.", False),
        ("Mình là trợ lý AI tự động của fanpage.", True),
        ("Mình là AI nên không có cảm xúc như người.", True),
        ("Đây là chatbot của trang.", True),
        ("I'm an AI assistant.", True),
    ],
)
def test_ai_disclosure_detection(text, expected):  # type: ignore[no-untyped-def]
    from app.conversation.output_check import mentions_ai

    assert mentions_ai(text) is expected


def test_topic_mention_of_ai_still_gets_disclosure_when_needed():
    r = check_output("Hi! Có gì muốn hỏi về AI không?", _ctx(needs_disclosure=True, knowledge_question=False))
    assert r.text.startswith("Mình là trợ lý AI tự động") and "disclosure_added" in r.flags


def test_advice_to_check_official_sources_is_not_treated_as_verification_claim():
    r = check_output(
        "Mình không thể gửi link phim lậu. Bạn nên xem phim ở nguồn chính thức nhé.",
        _ctx(user_text="gửi link phim lậu", knowledge_question=False, sources=[], allowed_urls=set()),
    )
    assert "false_verification_claim_removed" not in r.flags and "nguồn chính thức" in r.text
    r2 = check_output("Theo nguồn chính thức, tỉ lệ là như vậy.", _ctx(sources=[], allowed_urls=set()))
    assert r2.text is None and "false_verification_claim_removed" in r2.flags


@pytest.mark.parametrize(
    "reply", ["Mình không cập nhật được thông tin này.", "Mình không có số liệu cụ thể về việc đó."]
)
def test_more_uncertainty_phrasings_are_recognised(reply):  # type: ignore[no-untyped-def]
    r = check_output(reply, _ctx(sources=[], allowed_urls=set()))
    assert "uncertainty_stated" in r.flags and UNVERIFIED_NOTE not in r.text


async def test_history_given_to_model_excludes_system_footer_and_note():
    h = Harness(FakeProvider(lambda m: "LoRA giữ nguyên trọng số gốc [1]."))
    await h.say("u1", "LoRA là gì?")
    await h.say("u1", "cảm ơn")
    history = [str(m.content) for m in h.provider.calls[-1][1:]]
    assert any("LoRA giữ nguyên trọng số gốc [1]." in t for t in history)
    assert not any("Nguồn:" in t or LORA_URL in t or UNVERIFIED_NOTE in t for t in history)


def test_lone_citation_line_is_joined_to_previous_line():
    r = check_output("Thí sinh cần:\n- Đạt trên 1,0 điểm mỗi môn\n\n[1]", _ctx())
    assert r.text.startswith("Thí sinh cần:\n- Đạt trên 1,0 điểm mỗi môn [1]")
