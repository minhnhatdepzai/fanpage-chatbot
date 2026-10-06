"""Xử lý một cuộc hội thoại đã được claim: tạo/resume turn -> graph -> ghi quyết định -> gửi -> hoàn tất."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

from app.config import Settings
from app.conversation import prompts
from app.conversation.graph import GraphContext, RetryTurnLater, run_turn
from app.conversation.knowledge import KnowledgeBase
from app.conversation.profile import FanpageProfile
from app.messenger.client import MessengerClient
from app.messenger.text import split_message
from app.observability.tracing import Tracer
from app.providers.llm import ChatProvider
from app.storage import repository as repo
from app.storage.db import session_scope
from app.storage.models import TurnStatus
from app.web.channel import is_web
from app.workers.notifier import notify_handoff
from app.workers.sender import gate_decision, send_allowed_for, send_turn

log = logging.getLogger(__name__)


@dataclass
class WorkerDeps:
    settings: Settings
    graph: Any
    llm: ChatProvider
    messenger: MessengerClient
    tracer: Tracer
    profile: FanpageProfile
    store: repo.PostgresConversationStore
    worker_id: str
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)
    docs: Any = None
    vision: Any = None
    web_search: Any = None


class LeaseLost(Exception):  # noqa: N818
    pass


async def _keep_lease(deps: WorkerDeps, conv_id: uuid.UUID, lost: asyncio.Event) -> None:
    interval = max(5, deps.settings.worker_lease_seconds // 3)
    while True:
        await asyncio.sleep(interval)
        if not await repo.renew_lease(conv_id, deps.worker_id, deps.settings.worker_lease_seconds):
            lost.set()
            return


async def _typing_on(deps: WorkerDeps, conv_id: uuid.UUID, turn_id: uuid.UUID) -> None:
    if not deps.settings.messenger_typing_indicator:
        return
    with contextlib.suppress(Exception):
        gate = await repo.send_gate(conv_id, turn_id)
        if is_web(gate.page_id):
            return
        if gate_decision(deps.settings, gate, False) is None and send_allowed_for(deps.settings, gate.psid):
            await asyncio.wait_for(deps.messenger.sender_action(gate.psid, "typing_on"), timeout=5)


async def _handoff_ack_parts(
    deps: WorkerDeps, conv_id: uuid.UUID, turn_id: uuid.UUID, result: dict[str, Any]
) -> list[str]:
    """Nếu có kênh thông báo: gửi thông báo trước; chỉ nói 'đã báo' khi gửi thành công (1 lần/turn)."""
    parts = list(result.get("reply_parts") or [])
    if not result.get("set_handoff") or not deps.settings.handoff_notify_webhook_url.get_secret_value():
        return parts
    async with session_scope() as s:
        from sqlalchemy import text

        done = (
            await s.execute(
                text("SELECT 1 FROM audit_log WHERE action = 'handoff_notify_ok' AND target = :t LIMIT 1"),
                {"t": str(turn_id)},
            )
        ).first()
    if done:
        return parts
    _, uref, _ = await repo.conversation_psid(conv_id)
    ok = await notify_handoff(deps.settings, str(conv_id), uref)
    async with session_scope() as s:
        await repo.audit(s, "worker", "handoff_notify_ok" if ok else "handoff_notify_failed", str(turn_id))
    if ok:
        return parts
    text_ = prompts.HANDOFF_ACK_NO_NOTIFY
    if result.get("needs_disclosure"):
        text_ = f"{prompts.DISCLOSURE_SENTENCE} {text_}"
    return split_message(
        text_, deps.settings.messenger_max_chars_per_message, deps.settings.messenger_max_messages_per_reply
    )


async def process_conversation(deps: WorkerDeps, conv_id: uuid.UUID, attempts: int) -> None:
    s = deps.settings
    started = time.perf_counter()
    turn = await repo.start_or_resume_turn(conv_id)
    if turn is None:
        await repo.release_success(conv_id, deps.worker_id)
        return
    turn_id, is_new = turn
    lost = asyncio.Event()
    keeper = asyncio.create_task(_keep_lease(deps, conv_id, lost))
    _, uref, _ = await repo.conversation_psid(conv_id)
    page_id, queue_wait_ms = await repo.turn_queue_metadata(conv_id, turn_id)
    channel = "web" if is_web(page_id) else "messenger"
    trace_id = deps.tracer.trace_id_for(str(turn_id))
    try:
        with deps.tracer.turn(
            turn_id=str(turn_id),
            session_id=str(conv_id),
            user_id=uref,
            version=prompts.PROMPT_VERSION,
            tags=[channel, s.app_env.value],
            metadata={
                "turn_id": str(turn_id),
                "resumed": not is_new,
                "attempts": attempts,
                "queue_wait_ms": queue_wait_ms,
                "channel": channel,
            },
            input_text=None,
        ) as trace:
            if is_new:
                asyncio.create_task(_typing_on(deps, conv_id, turn_id))
            ctx = GraphContext(
                settings=s,
                store=deps.store,
                llm=deps.llm,
                profile=deps.profile,
                knowledge=deps.knowledge,
                docs=deps.docs,
                vision=deps.vision,
                web_search=deps.web_search,
                notifier_configured=bool(s.handoff_notify_webhook_url.get_secret_value()),
                trace=trace,
            )
            result = await run_turn(
                deps.graph, turn_id=str(turn_id), conversation_id=str(conv_id), context=ctx
            )
            if lost.is_set():
                raise LeaseLost("lease lost during graph run")
            parts = (
                await _handoff_ack_parts(deps, conv_id, turn_id, result)
                if result.get("action") == "reply"
                else []
            )
            await repo.persist_turn_decision(conv_id, turn_id, result, parts)
            outcome = result.get("outcome") or "no_reply"
            if parts:
                tool = trace.child("messenger-send", as_type="tool", metadata={"parts": len(parts)})
                rep = await send_turn(
                    s,
                    deps.messenger,
                    conv_id=conv_id,
                    turn_id=turn_id,
                    worker_id=deps.worker_id,
                    mark_disclosed=bool(result.get("mark_disclosed")),
                )
                tool.end(metadata={"sent": rep.sent, "outcome": rep.outcome})
                if rep.outcome != "sent":
                    outcome = f"{outcome}:{rep.outcome}"
            trace.update(
                output=trace.content(" ".join(parts)) if parts else None,
                metadata={
                    "outcome": outcome,
                    "mode": result.get("mode"),
                    "flags": ",".join(result.get("check_flags") or []),
                },
            )
        total_ms = int((time.perf_counter() - started) * 1000)
        await repo.complete_turn(turn_id, outcome, TurnStatus.completed, total_ms, trace_id)
        await repo.release_success(conv_id, deps.worker_id)
        log.info(
            "turn_completed",
            extra={
                "turn_id": str(turn_id),
                "user_ref": uref,
                "outcome": outcome,
                "mode": result.get("mode"),
                "total_ms": total_ms,
                "parts": len(parts),
                "queue_wait_ms": queue_wait_ms,
                "channel": channel,
            },
        )
    except RetryTurnLater as rl:
        await repo.release_retry(conv_id, deps.worker_id, rl.delay_seconds, rl.reason)
        log.warning("turn_retry_later", extra={"turn_id": str(turn_id), "reason": rl.reason})
    except LeaseLost:
        log.warning("turn_lease_lost", extra={"turn_id": str(turn_id)})
    except Exception as exc:
        # lỗi hạ tầng/bug: thử lại có giới hạn, sau đó đánh dấu turn thất bại để không kẹt hội thoại
        err = f"{type(exc).__name__}"
        log.exception("turn_failed", extra={"turn_id": str(turn_id), "attempts": attempts})
        if attempts + 1 >= s.worker_max_attempts:
            with contextlib.suppress(Exception):
                await repo.cancel_pending_outbound(turn_id, "turn_failed")
                await repo.complete_turn(turn_id, f"failed:{err}", TurnStatus.failed, None, trace_id)
                await repo.release_success(conv_id, deps.worker_id)
        else:
            with contextlib.suppress(Exception):
                await repo.release_retry(conv_id, deps.worker_id, min(60, 2 ** (attempts + 1)), err)
    finally:
        keeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await keeper


def deps_with(deps: WorkerDeps, **kw: Any) -> WorkerDeps:
    return replace(deps, **kw)
