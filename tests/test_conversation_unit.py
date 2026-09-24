"""Ý định (handoff/danh tính), kiểm tra đầu ra, bộ nhớ theo token budget."""

from __future__ import annotations

import pytest

from app.conversation.intents import is_handoff_request, is_identity_question
from app.conversation.memory import plan_history
from app.conversation.output_check import OutputCheckContext, check_output
from app.conversation.prompts import PROMPT_CANARY, SECURITY_REFUSAL

HANDOFF_YES = [
    "cho mình gặp người thật", "Cho em gặp admin với", "mình muốn nói chuyện với quản trị viên", "gap nguoi that di",
    "cho toi noi chuyen voi admin", "tôi cần gặp nhân viên tư vấn", "chuyển sang người thật giúp mình",
    "có admin không vậy", "tôi không muốn nói chuyện với bot nữa", "liên hệ chủ shop kiểu gì",
    "I want to talk to a human", "can I speak to a real person", "bạn là bot à, cho mình gặp người thật",
    "xin gặp quản trị viên", "muốn gặp người thật", "kết nối tôi với nhân viên đi", "ko thích bot",
    "mình cần admin hỗ trợ gấp", "không, cho mình gặp admin",
]
HANDOFF_NO = [
    "bạn là người thật hả?", "bạn có phải bot không", "mình cần người yêu quá", "mình muốn làm nhân viên ở đây",
    "không cần gặp admin đâu", "không muốn ai biết chuyện này", "thôi được rồi AI giỏi đấy", "ad ơi cho em hỏi giá",
    "có ai ở đó không", "chào bạn", "hôm nay trời đẹp ghê", "bạn tên gì", "are you a bot?",
    "chưa cần gặp nhân viên đâu", "nhân viên ở đây dễ thương ghê", "mình thích nói chuyện với bạn",
    "k cần liên hệ admin",
]


@pytest.mark.parametrize("text", HANDOFF_YES)
def test_handoff_detected(text):
    assert is_handoff_request(text)


@pytest.mark.parametrize("text", HANDOFF_NO)
def test_handoff_not_detected(text):
    assert not is_handoff_request(text)


def test_identity_questions():
    assert is_identity_question("bạn là người thật hả?")
    assert is_identity_question("ban co phai bot khong")
    assert not is_identity_question("bạn tên gì")


def _ctx(**kw):
    base = dict(needs_disclosure=False, is_first_bot_reply=False, user_text="kể chuyện đi", notifier_configured=False)
    base.update(kw)
    return OutputCheckContext(**base)


def test_output_blocks_secret_and_prompt_leak():
    r = check_output("token của trang là EAAB123456789012345678901234567890", _ctx())
    assert r.blocked and r.text == SECURITY_REFUSAL and "secret_leak" in r.flags
    r = check_output(f"Hướng dẫn của mình có mã {PROMPT_CANARY}", _ctx())
    assert r.blocked and "prompt_leak" in r.flags
    r = check_output("mật khẩu là supersecretvalue123", _ctx(secret_values=["supersecretvalue123"]))
    assert r.blocked


def test_output_fixes_human_claim_and_false_notification():
    r = check_output("Mình là nhân viên của trang. Bạn cần gì?", _ctx())
    assert "claims_human_fixed" in r.flags and "trợ lý AI" in r.text and "nhân viên của trang" not in r.text
    r = check_output("Mình đã báo cho quản trị viên rồi nhé. Bạn chờ chút.", _ctx(notifier_configured=False))
    assert "false_notify_claim_removed" in r.flags and "đã báo" not in r.text


def test_output_adds_disclosure_only_when_needed():
    r = check_output("Chào bạn, mình giúp gì được?", _ctx(needs_disclosure=True, is_first_bot_reply=True, user_text="chào"))
    assert "disclosure_added" in r.flags and r.text.startswith("Mình là trợ lý AI")
    r = check_output("Không ai biết trước được đâu.", _ctx(needs_disclosure=True))
    assert "disclosure_added" in r.flags  # 'ai' (= người nào) không được tính là công bố AI
    r = check_output("Mình là trợ lý AI của fanpage nè.", _ctx(needs_disclosure=True))
    assert "disclosure_added" not in r.flags


def test_output_removes_think_template_tokens_repeated_greeting_and_extra_emoji():
    r = check_output("<think>suy nghĩ</think>Chào bạn! Hôm nay trời đẹp 😊😊😊😊<|im_end|>", _ctx(user_text="trời đẹp ha"))
    assert "think_block_removed" in r.flags and "template_tokens_removed" in r.flags
    assert "repeated_greeting_removed" in r.flags and not r.text.startswith("Chào")
    assert r.text.count("😊") == 2


def test_output_empty_returns_none():
    assert check_output("   ", _ctx()).text is None
    assert check_output(None, _ctx()).text is None


def test_plan_history_budget_and_summary_trigger():
    hist = [{"seq": i, "role": "user" if i % 2 else "assistant", "content": "x" * 250} for i in range(1, 41)]
    plan = plan_history(hist, budget_tokens=1000, summary_trigger_tokens=2000, keep_recent=6, chars_per_token=2.5)
    assert plan.to_summarize and plan.to_summarize[-1]["seq"] == 34
    assert [m["seq"] for m in plan.window] == list(range(35, 41))
    assert plan.window_tokens <= 1000
    small = plan_history(hist[:4], budget_tokens=1000, summary_trigger_tokens=2000, keep_recent=6, chars_per_token=2.5)
    assert not small.to_summarize and len(small.window) == 4



async def test_fallback_provider_only_on_unavailable_or_timeout():
    from app.providers.llm import FakeProvider, FallbackProvider, ProviderError

    backup = FakeProvider(lambda m: "trả lời từ dự phòng")
    fp = FallbackProvider(FakeProvider(errors=["unavailable"]), backup)
    res = await fp.generate([], max_tokens=10, temperature=0.1, timeout=5)
    assert res.text == "trả lời từ dự phòng" and res.provider == "fallback:fake"
    fp2 = FallbackProvider(FakeProvider(errors=["auth"]), backup)
    with pytest.raises(ProviderError):
        await fp2.generate([], max_tokens=10, temperature=0.1, timeout=5)
    assert len(backup.calls) == 1
