"""LangGraph cho một lượt xử lý tin nhắn.

    load_state -> check_handoff -> prepare_history -> choose_response -> call_model -> check_output -> prepare_result

Graph KHÔNG gửi Messenger và không ghi DB: chỉ đọc ngữ cảnh và trả về quyết định. Lớp gửi tin (worker)
thực hiện side effect với idempotency key theo turn, nên graph resume/retry không thể gửi trùng.
Checkpointer (PostgreSQL) lưu tiến trình theo thread ``turn:<turn_id>`` để resume sau sự cố mà không
gọi lại model nếu bước đó đã xong.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.config import Settings
from app.config.settings import GroundingMode
from app.conversation import intents, prompts
from app.conversation.datecalc import answer_date_question
from app.conversation.knowledge import KnowledgeBase, KnowledgeEntry, extract_urls
from app.conversation.mathcalc import answer_math_question
from app.conversation.memory import format_for_summary, plan_history
from app.conversation.output_check import OutputCheckContext, check_output, mentions_ai
from app.conversation.profile import FanpageProfile
from app.conversation.state import ConversationStore, TurnState
from app.messenger.text import split_message
from app.observability.redaction import redact_pii
from app.observability.tracing import Observation
from app.providers.llm import ChatProvider, ProviderError, generate_with_retry
from app.vision.context import (
    NOTHING_RECOGNIZED_REPLY,
    current_image_note,
    image_context_block,
    nothing_recognized,
    ocr_text,
    sanitize_analysis,
)
from app.vision.fetch import fetch_image
from app.web.channel import is_web

log = logging.getLogger(__name__)


class RetryTurnLater(Exception):  # noqa: N818 - tên theo ngữ nghĩa
    """Yêu cầu worker lập lịch chạy lại lượt này sau ``delay_seconds`` (graph resume từ checkpoint)."""

    def __init__(self, delay_seconds: float, reason: str) -> None:
        super().__init__(reason)
        self.delay_seconds = delay_seconds
        self.reason = reason


@dataclass
class GraphContext:
    settings: Settings
    store: ConversationStore
    llm: ChatProvider
    profile: FanpageProfile
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)
    # tài liệu PDF/Word trong pgvector (None = tắt); kênh công khai chỉ được thấy tài liệu "public"
    docs: Any = None
    doc_visibility: frozenset[str] = frozenset({"public"})
    # dịch vụ ảnh (OCR + YOLO trên model server); None = tắt -> ảnh chỉ được báo là chưa xem được
    vision: Any = None
    notifier_configured: bool = False
    trace: Observation = field(default_factory=Observation)
    now: Callable[[], datetime] = lambda: datetime.now(UTC)


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


# ------------------------------------------------------------------------------------------ nodes
async def load_state(state: TurnState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    s = ctx.settings
    tc = await ctx.store.load_turn_context(state["conversation_id"], state["turn_id"], history_limit=80)
    now = ctx.now()
    gap = timedelta(hours=s.ai_disclosure_gap_hours)
    needs_disclosure = (
        tc.bot_messages_count == 0
        or tc.last_disclosure_at is None
        or (tc.last_bot_message_at is not None and now - tc.last_bot_message_at > gap)
        or (
            tc.bot_resumed_at is not None
            and (tc.last_disclosure_at is None or tc.bot_resumed_at > tc.last_disclosure_at)
        )
    )
    latest = max((i.event_ts for i in tc.inbound if i.event_ts), default=None)
    return {
        "page_id": tc.page_id,
        "user_ref": tc.user_ref,
        "inbound": [i.to_state() for i in tc.inbound],
        "history": [{"seq": h.seq, "role": h.role, "content": h.content} for h in tc.history],
        "summary": tc.summary,
        "summary_upto_seq": tc.summary_upto_seq,
        "handoff_active": tc.handoff_active,
        "needs_disclosure": needs_disclosure,
        "is_first_bot_reply": tc.bot_messages_count == 0,
        "last_fallback_at": tc.last_fallback_at.isoformat() if tc.last_fallback_at else None,
        "latest_inbound_at": latest.isoformat() if latest else None,
        "check_flags": [],
        "set_handoff": False,
        "is_fallback": False,
        "model_error": None,
        "draft": None,
    }


def _user_text(inbound: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in inbound:
        if item.get("text"):
            parts.append(item["text"])
        elif (
            item["kind"] == "postback" and item.get("payload") and not intents.is_get_started(item["payload"])
        ):
            parts.append(item["payload"])
    return "\n".join(parts).strip()


async def check_handoff(state: TurnState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    if state.get("handoff_active"):
        return {"route": "stop", "mode": "handoff_active", "outcome": "skipped_handoff_active"}
    text = _user_text(state.get("inbound", []))
    if any(intents.is_handoff_request(i.get("text")) for i in state.get("inbound", [])):
        if is_web(state.get("page_id")):
            return {
                "route": "template",
                "mode": "handoff_request_web",
                "template_reply": prompts.WEB_HANDOFF_REPLY,
                "user_text": text,
            }
        ack = (
            prompts.HANDOFF_ACK_NOTIFIED
            if runtime.context.notifier_configured
            else prompts.HANDOFF_ACK_NO_NOTIFY
        )
        return {
            "route": "template",
            "mode": "handoff_request",
            "template_reply": ack,
            "set_handoff": True,
            "user_text": text,
        }
    return {"route": "continue", "user_text": text}


async def prepare_history(state: TurnState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    if state.get("route") != "continue":
        return {}
    ctx = runtime.context
    s = ctx.settings
    plan = plan_history(
        state.get("history", []),
        budget_tokens=max(300, s.memory_max_context_tokens - 900),  # chừa chỗ cho system prompt + tin mới
        summary_trigger_tokens=s.memory_summary_trigger_tokens,
        keep_recent=s.memory_keep_recent_messages,
        chars_per_token=s.memory_chars_per_token,
    )
    update: dict[str, Any] = {"window": plan.window}
    if plan.to_summarize:
        span = ctx.trace.child(
            "summarize-history",
            as_type="generation",
            metadata={"messages": len(plan.to_summarize), "prompt_version": prompts.SUMMARY_PROMPT_VERSION},
        )
        try:
            previous = state.get("summary") or "(chưa có)"
            res = await generate_with_retry(
                ctx.llm,
                [
                    SystemMessage(prompts.SUMMARY_SYSTEM_PROMPT),
                    HumanMessage(
                        f"Tóm tắt hiện có:\n{previous}\n\nCác tin nhắn mới cần gộp vào:\n"
                        f"{format_for_summary(plan.to_summarize)}"
                    ),
                ],
                max_tokens=300,
                temperature=0.2,
                timeout=s.llm_timeout_seconds,
                max_retries=0,
            )
            summary = redact_pii(res.text.strip())[:2000]
            if summary:
                update["new_summary"] = summary
                update["new_summary_upto_seq"] = plan.to_summarize[-1]["seq"]
                update["summary"] = summary
            span.end(
                model=res.model,
                usage_details={"input": res.input_tokens or 0, "output": res.output_tokens or 0},
                output=span.content(summary),
            )
        except ProviderError as err:
            # Tóm tắt lỗi không chặn trả lời: giữ tóm tắt cũ, cửa sổ vẫn bị giới hạn theo token.
            span.end(level="WARNING", status_message=f"summary_failed:{err.kind}")
            update["check_flags"] = [*state.get("check_flags", []), "summary_failed"]
    return update


async def choose_response(state: TurnState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    if state.get("route") != "continue":
        return {}
    inbound = state.get("inbound", [])
    text = state.get("user_text", "")
    latest = _parse(state.get("latest_inbound_at"))
    max_age = runtime.context.settings.reply_max_age_seconds
    if latest and (runtime.context.now() - latest).total_seconds() > max_age:
        # tin đã quá hạn (vd. hệ thống ngừng lâu) -> không trả lời muộn, không tốn lượt gọi model
        return {"route": "stop", "mode": "stale", "outcome": "skipped_stale"}
    if inbound and all(i.get("like_sticker") for i in inbound):
        return {"route": "stop", "mode": "like_sticker", "outcome": "no_reply_needed"}
    kinds = [k for i in inbound for k in (i.get("attachment_types") or [])]
    if not text and any(
        i["kind"] == "postback" and intents.is_get_started(i.get("payload")) for i in inbound
    ):
        return {
            "route": "template",
            "mode": "welcome",
            "template_reply": prompts.welcome_message(runtime.context.profile),
        }
    ctx = runtime.context
    flags = list(state.get("check_flags", []))
    image_ctx: str | None = None
    image_note: str | None = None
    image_meta: dict[str, Any] = {}
    ocr_query = ""
    image_items = [(i["seq"], k, img) for i in inbound for k, img in enumerate(i.get("images") or [])][:3]
    analyzed: list[dict[str, Any]] = []
    if image_items and ctx.vision is not None:
        analyses = await _analyze_images(ctx, image_items, flags, analyzed)
        if analyses:
            image_meta = {
                "images": len(analyses),
                "ocr_lines": sum((a.get("ocr") or {}).get("lines", 0) for a in analyses),
                "objects": sum(len(a.get("objects") or []) for a in analyses),
            }
            if nothing_recognized(analyses):
                # không có chữ/vật thể đủ chắc -> KHÔNG gọi model (model hay bịa khi thiếu dữ kiện)
                return {
                    "route": "template",
                    "mode": "image_nothing_recognized",
                    "check_flags": flags,
                    "template_reply": NOTHING_RECOGNIZED_REPLY,
                    "image_meta": image_meta,
                    "analyzed_images": analyzed,
                }
            image_ctx = image_context_block(analyses)
            image_note = current_image_note(analyses)
            ocr_query = " ".join(ocr_text(a) for a in analyses)[:300]
    if not text and kinds and not image_ctx:
        return {
            "route": "template",
            "mode": "unsupported_attachment",
            "template_reply": prompts.attachment_notice(kinds),
            "check_flags": flags,
        }
    if not text and not image_ctx:
        return {"route": "stop", "mode": "empty", "outcome": "no_reply_needed"}
    image_update: dict[str, Any] = {
        "image_context": image_ctx,
        "image_note": image_note,
        "image_meta": image_meta,
        "analyzed_images": analyzed,
    }
    if not text:
        text = IMAGE_ONLY_USER_TEXT
        image_update["user_text"] = text
    math_answer = None if image_ctx else answer_math_question(text)
    if math_answer:  # phép tính số học: tính bằng code, không cần ghi chú nguồn
        return {"route": "template", "mode": "math_calc", "template_reply": math_answer, "check_flags": flags}
    date_answer = None if image_ctx else answer_date_question(text, ctx.now())
    if (
        date_answer
    ):  # ngày/thứ: tính bằng code (model từng tính sai), không gọi model, không cần ghi chú nguồn
        return {"route": "template", "mode": "date_calc", "template_reply": date_answer, "check_flags": flags}
    knowledge_q = intents.is_knowledge_question(text)
    if not image_ctx and _REFERS_TO_IMAGE.search(intents.normalize(text)):
        notes = [
            m["content"]
            for m in state.get("window", [])
            if m["role"] == "user" and "[Ảnh đã gửi" in m["content"]
        ]
        ocr_query = " ".join(n.split("[Ảnh đã gửi", 1)[1] for n in notes[-2:])[:600]
    entries = await _retrieve(ctx, f"{text}\n{ocr_query}".strip(), flags)
    if not entries and knowledge_q:
        # câu hỏi tiếp nối ("vậy nó hoạt động sao?") thường thiếu từ khóa -> ghép tin người dùng liền trước
        prev = next((m["content"] for m in reversed(state.get("window", [])) if m["role"] == "user"), "")
        if prev:
            entries = await _retrieve(ctx, f"{prev}\n{text}", flags)
    hit_dicts = [{"id": e.id, "title": e.title, "source": e.source, "content": e.content} for e in entries]
    update: dict[str, Any] = {
        "knowledge_question": knowledge_q,
        "knowledge_hits": hit_dicts,
        "check_flags": flags,
        **image_update,
    }
    if knowledge_q and not entries and ctx.settings.grounding_mode == GroundingMode.strict:
        return {
            **update,
            "route": "template",
            "mode": "no_verified_source",
            "template_reply": prompts.NO_SOURCE_REPLY,
        }
    return {**update, "route": "llm", "mode": "chat", "attachment_kinds": kinds}


IMAGE_ONLY_USER_TEXT = "(Người dùng gửi ảnh, không kèm câu hỏi.)"
# hỏi tiếp về ảnh đã gửi ("ảnh card lúc nãy là loại nào") -> tìm nguồn theo chữ đọc được trong ảnh đó
_REFERS_TO_IMAGE = re.compile(
    r"\b(?:anh|hinh|buc anh|tam anh|hinh anh|picture|image|photo|luc nay|vua roi|vua gui|hoi nay|ban nay|o tren)\b"
)


async def _analyze_images(
    ctx: GraphContext,
    items: list[tuple[int, int, dict[str, Any]]],
    flags: list[str],
    persist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ảnh website đã phân tích sẵn; ảnh Messenger tải từ CDN của Meta rồi gửi model server phân tích.
    items: (seq tin nhắn, vị trí ảnh, ảnh). Kết quả ảnh Messenger đưa vào ``persist`` để lưu (hỏi tiếp về ảnh)."""
    s = ctx.settings
    out: list[dict[str, Any]] = []
    for seq, idx, item in items:
        try:
            if isinstance(item.get("analysis"), dict):
                out.append(item["analysis"])
            elif isinstance(item.get("url"), str):
                data = await fetch_image(
                    item["url"],
                    allowed_suffixes=s.vision_allowed_image_hosts,
                    max_bytes=s.vision_max_image_mb * 1_000_000,
                )
                analysis = sanitize_analysis(await ctx.vision.analyze(data))
                out.append(analysis)
                persist.append({"seq": seq, "index": idx, "analysis": analysis})
        except Exception as exc:  # noqa: BLE001 - lỗi ảnh không được làm hỏng cả lượt trả lời
            log.warning("image_analysis_failed", extra={"error": type(exc).__name__})
            if "image_analysis_failed" not in flags:
                flags.append("image_analysis_failed")
    return out


async def _retrieve(ctx: GraphContext, query: str, flags: list[str]) -> list[KnowledgeEntry]:
    """Kho kiến thức YAML (khớp từ khóa, độ chính xác cao) trước, rồi đoạn tài liệu PDF/Word (pgvector)."""
    entries = [h.entry for h in ctx.knowledge.search(query, top_k=ctx.settings.knowledge_top_k)]
    if ctx.docs is not None:
        try:
            entries += await ctx.docs.search(query, visibility=ctx.doc_visibility)
        except Exception as exc:  # noqa: BLE001 - tìm tài liệu lỗi không được làm hỏng cả lượt trả lời
            log.warning("doc_search_failed", extra={"error": type(exc).__name__})
            if "doc_search_failed" not in flags:
                flags.append("doc_search_failed")
    return entries[: ctx.settings.knowledge_top_k + ctx.settings.doc_top_k]


def _hit_entries(state: TurnState) -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(h["id"], h["title"], h["source"], h["content"])
        for h in state.get("knowledge_hits") or []
    ]


def _strip_system_additions(text: str) -> str:
    """Bỏ danh sách "Nguồn:" và ghi chú do hệ thống gắn khỏi câu trả lời cũ trước khi đưa lại cho model.

    Nếu giữ, model bắt chước tự viết dòng nguồn/ghi chú ở lượt sau (bị lớp kiểm tra xóa, câu trả lời bị hụt).
    """
    text = text.replace(prompts.UNVERIFIED_NOTE, "").replace(prompts.IMAGE_UNVERIFIED_NOTE, "")
    text = re.sub(r"(?:\n\n\(Lưu ý: [^\n]*\))+\s*$", "", text.strip())  # ghi chú bản cũ đã lưu trong DB
    return text.split(f"\n\n{prompts.SOURCES_FOOTER_TITLE}\n", 1)[0].strip()


def _to_lc_messages(window: list[dict[str, Any]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in window:
        if m["role"] == "user":
            out.append(HumanMessage(m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(_strip_system_additions(m["content"])))
        elif m["role"] == "human_agent":
            out.append(AIMessage(f"(Quản trị viên của trang trả lời) {m['content']}"))
    return out


async def call_model(state: TurnState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    s = ctx.settings
    now = ctx.now()
    system = prompts.build_system_prompt(
        ctx.profile,
        now,
        summary=state.get("summary"),
        needs_disclosure=state.get("needs_disclosure", False),
        attachment_kinds=state.get("attachment_kinds") or None,
        sources=_hit_entries(state),
        image_context=state.get("image_context"),
        vision_available=ctx.vision is not None,
        docs_available=ctx.docs is not None,
    )
    messages: list[BaseMessage] = [SystemMessage(system), *_to_lc_messages(state.get("window", []))]
    human = state.get("user_text", "")
    if state.get("image_note"):
        human = f"{human}\n{state['image_note']}"
    messages.append(HumanMessage(human))
    # Tổng thời gian cho phép = không vượt quá hạn "tin quá hạn"
    latest = _parse(state.get("latest_inbound_at")) or now
    remaining = s.reply_max_age_seconds - (now - latest).total_seconds()
    deadline = max(5.0, min(remaining - 5, s.llm_timeout_seconds * (s.llm_max_retries + 1)))
    gen = ctx.trace.child(
        "chat-completion",
        as_type="generation",
        model=s.llm_model,
        model_parameters={"temperature": s.llm_temperature, "max_tokens": s.llm_max_output_tokens},
        input=ctx.trace.content([{"role": m.type, "content": m.content} for m in messages[1:]]),
        metadata={
            "prompt_version": prompts.PROMPT_VERSION,
            "window_messages": len(state.get("window", [])),
            "knowledge_hits": ",".join(h["id"] for h in state.get("knowledge_hits") or []) or "none",
        },
    )
    try:
        res = await generate_with_retry(
            ctx.llm,
            messages,
            max_tokens=s.llm_max_output_tokens,
            temperature=s.llm_temperature,
            timeout=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
            total_deadline=deadline,
        )
    except ProviderError as err:
        gen.end(level="ERROR", status_message=err.kind)
        if err.kind == "rate_limit" and remaining > 60:
            raise RetryTurnLater(30, "provider_rate_limited") from err
        return {"model_error": err.kind, "model_meta": {"provider": ctx.llm.name, "error": err.kind}}
    prev = next((m["content"] for m in reversed(state.get("window", [])) if m["role"] == "assistant"), "")
    repeat_flag: list[str] = []
    if prev and res.text and _too_similar(res.text, prev):
        # model 4B đôi khi chép lại nguyên câu trả lời trước (đã gặp với ảnh thứ hai) -> sinh lại một lần
        retry = [SystemMessage(system + REPEAT_WARNING), *messages[1:]]
        try:
            res2 = await generate_with_retry(
                ctx.llm,
                retry,
                max_tokens=s.llm_max_output_tokens,
                temperature=min(1.0, s.llm_temperature + 0.3),
                timeout=s.llm_timeout_seconds,
                max_retries=0,
            )
            if not _too_similar(res2.text, prev):
                res = res2
            repeat_flag = ["repeat_regenerated"]
        except ProviderError:
            repeat_flag = ["repeat_regenerate_failed"]
    gen.end(
        model=res.model,
        output=gen.content(res.text),
        usage_details={"input": res.input_tokens or 0, "output": res.output_tokens or 0},
        metadata={"adapter": res.adapter or "none", "latency_ms": res.latency_ms, "cost": "unknown"},
    )
    return {
        "check_flags": [*state.get("check_flags", []), *repeat_flag],
        "draft": res.text,
        "model_meta": {
            "provider": res.provider,
            "model": res.model,
            "adapter": res.adapter,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "latency_ms": res.latency_ms,
        },
    }


REPEAT_WARNING = (
    "\nLƯU Ý: bản nháp trước lặp lại nguyên văn câu trả lời ở lượt trước. Trả lời lại đúng tin nhắn MỚI NHẤT của "
    "người dùng (và ảnh mới nếu có), không chép lại câu trả lời cũ."
)


def _too_similar(a: str, b: str) -> bool:
    from difflib import SequenceMatcher

    note = prompts.UNVERIFIED_NOTE
    a = a.replace(note, "").strip()
    b = b.split("\n\nNguồn:")[0].replace(note, "").strip()
    if len(a) <= 40:
        return False
    m = SequenceMatcher(None, a, b, autojunk=False)
    # câu mới nằm gần như nguyên vẹn trong câu cũ (câu cũ có thể thêm lời giới thiệu/ghi chú) hoặc gần giống hệt
    return m.find_longest_match(0, len(a), 0, len(b)).size / len(a) > 0.9 or m.ratio() > 0.9


async def check_output_node(state: TurnState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    ctx = runtime.context
    s = ctx.settings
    flags = list(state.get("check_flags", []))
    if state.get("model_error"):
        last_fb = _parse(state.get("last_fallback_at"))
        if last_fb and (ctx.now() - last_fb).total_seconds() < s.fallback_cooldown_seconds:
            return {"action": "none", "outcome": "model_error_fallback_suppressed", "check_flags": flags}
        return {"template_reply": prompts.FALLBACK_REPLY, "is_fallback": True, "check_flags": flags}
    guard = ctx.trace.child("output-check", as_type="guardrail")
    hits = state.get("knowledge_hits") or []
    profile_text = "\n".join([ctx.profile.description, *ctx.profile.facts, ctx.profile.contact_hint])
    result = check_output(
        state.get("draft"),
        OutputCheckContext(
            needs_disclosure=state.get("needs_disclosure", False),
            is_first_bot_reply=state.get("is_first_bot_reply", False),
            user_text=state.get("user_text", ""),
            notifier_configured=ctx.notifier_configured,
            secret_values=s.secret_values(),
            sources=[(h["title"], h["source"]) for h in hits],
            allowed_urls={u for h in hits for u in extract_urls(h["source"])}
            | set(extract_urls(profile_text)),
            knowledge_question=bool(state.get("knowledge_question")),
            source_text=" ".join(h["content"] for h in hits),
            has_image_context=bool(state.get("image_context")),
            recent_image=any("[Ảnh đã gửi" in h["content"] for h in state.get("history", [])[-6:]),
        ),
    )
    flags += result.flags
    guard.end(metadata={"flags": ",".join(result.flags) or "none", "blocked": result.blocked})
    if result.text is None:
        return {"template_reply": prompts.FALLBACK_REPLY, "is_fallback": True, "check_flags": flags}
    return {"template_reply": result.text, "check_flags": flags}


async def prepare_result(state: TurnState, runtime: Runtime[GraphContext]) -> dict[str, Any]:
    s = runtime.context.settings
    if state.get("route") == "stop" or state.get("action") == "none":
        return {"action": "none", "reply_parts": [], "outcome": state.get("outcome") or "no_reply"}
    text = state.get("template_reply") or ""
    # Chính sách Messenger: công bố là dịch vụ tự động ở đầu hội thoại, sau thời gian dài vắng mặt,
    # hoặc khi chuyển từ người thật sang bot. Câu trả lời LLM đã được check_output xử lý;
    # ở đây phủ nốt các câu trả lời mẫu (handoff, fallback, tệp đính kèm).
    needs = bool(state.get("needs_disclosure"))
    if needs and not mentions_ai(text):
        text = f"{prompts.DISCLOSURE_SENTENCE} {text}"
    mark_disclosed = needs
    parts = split_message(text, s.messenger_max_chars_per_message, s.messenger_max_messages_per_reply)
    if not parts:
        return {"action": "none", "reply_parts": [], "outcome": "empty_reply"}
    outcome = (
        "fallback" if state.get("is_fallback") else ("handoff_ack" if state.get("set_handoff") else "reply")
    )
    return {"action": "reply", "reply_parts": parts, "outcome": outcome, "mark_disclosed": mark_disclosed}


# ------------------------------------------------------------------------------------------ routing
def _after_handoff(state: TurnState) -> str:
    return "prepare_history" if state.get("route") == "continue" else "prepare_result"


def _after_choose(state: TurnState) -> str:
    return "call_model" if state.get("route") == "llm" else "prepare_result"


def build_graph(checkpointer: Any = None):
    g = StateGraph(TurnState, context_schema=GraphContext)
    g.add_node("load_state", load_state)
    g.add_node("check_handoff", check_handoff)
    g.add_node("prepare_history", prepare_history)
    g.add_node("choose_response", choose_response)
    g.add_node("call_model", call_model)
    g.add_node("check_output", check_output_node)
    g.add_node("prepare_result", prepare_result)
    g.add_edge(START, "load_state")
    g.add_edge("load_state", "check_handoff")
    g.add_conditional_edges("check_handoff", _after_handoff, ["prepare_history", "prepare_result"])
    g.add_edge("prepare_history", "choose_response")
    g.add_conditional_edges("choose_response", _after_choose, ["call_model", "prepare_result"])
    g.add_edge("call_model", "check_output")
    g.add_edge("check_output", "prepare_result")
    g.add_edge("prepare_result", END)
    return g.compile(checkpointer=checkpointer)


def thread_id_for(turn_id: str) -> str:
    return f"turn:{turn_id}"


async def run_turn(graph, *, turn_id: str, conversation_id: str, context: GraphContext) -> dict[str, Any]:
    """Chạy (hoặc resume) graph cho một turn. Idempotent theo turn_id nhờ checkpointer."""
    config = {"configurable": {"thread_id": thread_id_for(turn_id)}}
    if graph.checkpointer is not None:
        snap = await graph.aget_state(config)
        if snap and snap.values and snap.values.get("action") and not snap.next:
            return dict(snap.values)  # đã chạy xong trước đó
        if snap and snap.next:
            out = await graph.ainvoke(None, config, context=context, durability="sync")
            return dict(out)
    # durability chỉ có nghĩa khi có checkpointer (LangGraph 1.2.12 lỗi nếu truyền khi không có)
    extra = {"durability": "sync"} if graph.checkpointer is not None else {}
    out = await graph.ainvoke(
        {"turn_id": turn_id, "conversation_id": conversation_id}, config, context=context, **extra
    )
    return dict(out)
