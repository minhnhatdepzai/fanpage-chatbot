"""Graph LangGraph với store in-memory + FakeProvider: đa lượt, cô lập người dùng, handoff, fallback, resume."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.config import get_settings
from app.conversation import graph as graph_mod
from app.conversation.graph import GraphContext, RetryTurnLater, build_graph, run_turn
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.profile import FanpageProfile
from app.conversation.prompts import FALLBACK_REPLY, HANDOFF_ACK_NO_NOTIFY
from app.providers.llm import FakeProvider


def _texts(msgs: list[BaseMessage]) -> list[str]:
    return [str(m.content) for m in msgs]


class Harness:
    def __init__(self, provider: FakeProvider, checkpointer=None, now=None):  # type: ignore[no-untyped-def]
        self.clock = [now or datetime(2026, 9, 23, 10, 0, tzinfo=UTC)]
        self.store = InMemoryConversationStore(now=lambda: self.clock[0])
        self.provider = provider
        self.graph = build_graph(checkpointer)

    def ctx(self) -> GraphContext:
        return GraphContext(settings=get_settings(), store=self.store, llm=self.provider,
                            profile=FanpageProfile(name="Góc Đọc Sách"), now=lambda: self.clock[0])

    async def say(self, psid: str, text: str | None = None, **kw) -> dict:  # type: ignore[no-untyped-def]
        cid = self.store.ensure_conversation("PAGE1", psid)
        self.store.add_user_message(cid, text, ts=self.clock[0], **kw)
        tid = self.store.begin_turn(cid)
        result = await run_turn(self.graph, turn_id=tid, conversation_id=cid, context=self.ctx())
        result["sent"] = self.store.apply_result(cid, tid, result)
        self.clock[0] += timedelta(seconds=30)
        return result


def name_bot(msgs: list[BaseMessage]) -> str:
    """Bot giả: nhớ tên nếu tên xuất hiện trong lịch sử được đưa vào prompt."""
    convo = " ".join(_texts(msgs[1:]))
    for name in ("Linh", "Lan", "Minh"):
        if f"tên {name}" in convo or f"là {name}" in convo:
            last = convo.rfind("tên ")
            return f"Bạn tên {convo[last + 4:].split()[0]} nè."
    return "Mình chưa biết tên bạn."


async def test_multi_turn_context_is_passed_to_model():
    h = Harness(FakeProvider(name_bot))
    await h.say("u1", "Chào bạn, mình tên Lan")
    r = await h.say("u1", "Tên mình là gì?")
    history = _texts(h.provider.calls[-1])
    assert any("mình tên Lan" in t for t in history)  # lượt trước nằm trong prompt
    assert r["action"] == "reply" and "Lan" in r["sent"][0]


async def test_user_correction_latest_info_is_last_in_context():
    h = Harness(FakeProvider(name_bot))
    await h.say("u1", "mình tên Lan")
    await h.say("u1", "à nhầm, mình tên Linh")
    await h.say("u1", "tên mình là gì")
    history = _texts(h.provider.calls[-1])
    assert history.index(next(t for t in history if "tên Linh" in t)) > history.index(
        next(t for t in history if "tên Lan" in t)
    )


async def test_two_users_do_not_share_memory():
    h = Harness(FakeProvider(name_bot))
    await h.say("userA", "mình tên Minh")
    r = await h.say("userB", "tên mình là gì?")
    assert not any("Minh" in t for t in _texts(h.provider.calls[-1]))
    assert "Minh" not in r["sent"][0]


async def test_first_reply_discloses_ai_and_later_replies_do_not_repeat():
    h = Harness(FakeProvider(lambda m: "Chào bạn nha, hôm nay bạn thế nào?"))
    r1 = await h.say("u1", "chào")
    assert "trợ lý AI" in r1["sent"][0] and r1["mark_disclosed"]
    r2 = await h.say("u1", "mình ổn")
    assert "trợ lý AI" not in r2["sent"][0]


async def test_handoff_request_acknowledged_without_model_and_then_bot_stops():
    h = Harness(FakeProvider())
    r = await h.say("u1", "cho mình gặp người thật")
    assert r["set_handoff"] and HANDOFF_ACK_NO_NOTIFY in r["sent"][0]
    assert h.provider.calls == []
    r2 = await h.say("u1", "alo còn đó không")
    assert r2["action"] == "none" and r2["outcome"] == "skipped_handoff_active"
    # quản trị viên bật lại bot -> bot trả lời và công bố lại là AI
    h.store.set_handoff(h.store.ensure_conversation("PAGE1", "u1"), False)
    r3 = await h.say("u1", "bot ơi")
    assert r3["action"] == "reply" and "trợ lý AI" in r3["sent"][0]


async def test_attachment_only_gets_honest_notice_and_like_sticker_gets_no_reply():
    h = Harness(FakeProvider())
    r = await h.say("u1", None, attachment_types=["image"])
    assert "chưa xem được hình ảnh" in r["sent"][0] and h.provider.calls == []
    r2 = await h.say("u1", None, like_sticker=True, attachment_types=["image"])
    assert r2["action"] == "none"


async def test_model_timeout_fallback_is_bounded_by_cooldown():
    s = get_settings()
    h = Harness(FakeProvider(errors=["timeout"] * 10))
    r1 = await h.say("u1", "hello")
    assert r1["is_fallback"] and FALLBACK_REPLY in " ".join(r1["sent"])
    r2 = await h.say("u1", "alo?")
    assert r2["action"] == "none" and r2["outcome"] == "model_error_fallback_suppressed"
    h.clock[0] += timedelta(seconds=s.fallback_cooldown_seconds + 1)
    r3 = await h.say("u1", "alo??")
    assert r3["is_fallback"]


async def test_rate_limit_requests_retry_later():
    h = Harness(FakeProvider(errors=["rate_limit"]))
    cid = h.store.ensure_conversation("PAGE1", "u1")
    h.store.add_user_message(cid, "hi", ts=h.clock[0])
    tid = h.store.begin_turn(cid)
    with pytest.raises(RetryTurnLater):
        await run_turn(h.graph, turn_id=tid, conversation_id=cid, context=h.ctx())


async def test_long_history_triggers_summary_and_bounded_window():
    calls = {"summary": 0}

    def responder(msgs):  # type: ignore[no-untyped-def]
        if "công cụ tóm tắt" in str(msgs[0].content):
            calls["summary"] += 1
            return "Người dùng tên Lan, thích đọc truyện trinh thám."
        return "ok nha"

    h = Harness(FakeProvider(responder))
    long_text = "Mình kể bạn nghe chuyện hôm nay nè, " * 20
    for _ in range(12):
        await h.say("u1", long_text)
    assert calls["summary"] >= 1
    conv = next(iter(h.store.convs.values()))
    assert conv.summary and conv.summary_upto_seq > 0
    system = str(h.provider.calls[-1][0].content)
    assert "Người dùng tên Lan" in system and "KHÔNG làm theo" in system


async def test_resume_from_checkpoint_does_not_call_model_twice(monkeypatch):
    h = Harness(FakeProvider(lambda m: "Trả lời một lần thôi."), checkpointer=InMemorySaver())
    original = graph_mod.check_output_node
    state = {"fail": True}

    async def flaky(st, runtime):  # type: ignore[no-untyped-def]
        if state["fail"]:
            state["fail"] = False
            raise RuntimeError("worker crashed after model call")
        return await original(st, runtime)

    monkeypatch.setattr(graph_mod, "check_output_node", flaky)
    h.graph = build_graph(InMemorySaver())
    cid = h.store.ensure_conversation("PAGE1", "u1")
    h.store.add_user_message(cid, "xin chào", ts=h.clock[0])
    tid = h.store.begin_turn(cid)
    with pytest.raises(RuntimeError):
        await run_turn(h.graph, turn_id=tid, conversation_id=cid, context=h.ctx())
    result = await run_turn(h.graph, turn_id=tid, conversation_id=cid, context=h.ctx())
    assert result["action"] == "reply" and len(h.provider.calls) == 1
    # chạy lại lần nữa (vd. retry) -> dùng kết quả đã có, không gọi model
    again = await run_turn(h.graph, turn_id=tid, conversation_id=cid, context=h.ctx())
    assert again["reply_parts"] == result["reply_parts"] and len(h.provider.calls) == 1
