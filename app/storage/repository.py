"""Truy cập dữ liệu PostgreSQL cho webhook, worker và quản trị.

Hàng đợi = bảng ``conversations`` (cột next_run_at + lease). Webhook ghi tin nhắn và lập lịch trong
CÙNG một transaction rồi mới trả 200 cho Meta -> không có khoảng hở "đã ACK nhưng chưa enqueue".
Mỗi cuộc hội thoại (Page ID + PSID) chỉ được một worker xử lý tại một thời điểm (lease);
các cuộc hội thoại khác nhau chạy song song (FOR UPDATE SKIP LOCKED).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.conversation.state import HistoryItem, InboundItem, TurnContext
from app.messenger.events import Echo, Ignored, ParsedEvent, Postback, UserMessage, WindowOpener
from app.observability.pseudonym import user_ref as make_user_ref
from app.storage.db import session_scope
from app.storage.models import InStatus, OutStatus, Role, TurnStatus
from app.vision.context import history_content


# ============================================================================ ingest (webhook)
@dataclass
class IngestStats:
    received: int = 0
    stored_messages: int = 0
    duplicates: int = 0
    window_updates: int = 0
    echoes_confirmed: int = 0
    human_agent_messages: int = 0
    handoffs_from_human_reply: int = 0
    ignored: dict[str, int] = field(default_factory=dict)


_UPSERT_CONV = text(
    """
    INSERT INTO conversations (id, page_id, psid, user_ref, last_user_message_at)
    VALUES (:id, :page_id, :psid, :user_ref, :ts)
    ON CONFLICT (page_id, psid) DO UPDATE SET
        last_user_message_at = GREATEST(conversations.last_user_message_at, EXCLUDED.last_user_message_at),
        updated_at = now()
    RETURNING id
    """
)
_INSERT_MSG = text(
    """
    INSERT INTO messages (conversation_id, role, kind, text, attachment_types, payload, mid, event_ts, status)
    VALUES (:cid, :role, :kind, :text, CAST(:att AS JSONB), CAST(:payload AS JSONB), :mid, :ts, :status)
    ON CONFLICT (mid) WHERE mid IS NOT NULL DO NOTHING
    RETURNING id
    """
)
_SCHEDULE = text(
    """
    UPDATE conversations SET
        first_pending_at = COALESCE(first_pending_at, now()),
        next_run_at = LEAST(
            COALESCE(first_pending_at, now()) + make_interval(secs => :max_wait),
            now() + make_interval(secs => :debounce)
        )
    WHERE id = :cid
    """
)


def _debounce_for(ev: UserMessage | Postback, settings: Settings) -> dict[str, float]:
    wait, max_wait = settings.debounce_seconds, settings.debounce_max_seconds
    if isinstance(ev, UserMessage) and ev.attachment_types and not ev.text and not ev.is_like_sticker:
        wait = max(wait, settings.debounce_attachment_seconds)
        max_wait = max(max_wait, wait + 2)
    return {"debounce": wait, "max_wait": max_wait}


async def _upsert_conversation(
    s: AsyncSession, settings: Settings, page_id: str, psid: str, ts: datetime | None
) -> uuid.UUID:
    ref = make_user_ref(settings.pseudonym_secret.get_secret_value(), page_id, psid)
    row = await s.execute(
        _UPSERT_CONV, {"id": uuid.uuid4(), "page_id": page_id, "psid": psid, "user_ref": ref, "ts": ts}
    )
    return row.scalar_one()


def _json(v: Any) -> str | None:
    import json

    return None if v is None else json.dumps(v, ensure_ascii=False)


async def ingest_events(s: AsyncSession, events: list[ParsedEvent], settings: Settings) -> IngestStats:
    st = IngestStats(received=len(events))
    for ev in events:
        if isinstance(ev, UserMessage | Postback):
            cid = await _upsert_conversation(s, settings, ev.page_id, ev.psid, ev.ts)
            if isinstance(ev, UserMessage):
                kind = "quick_reply" if ev.quick_reply_payload else ("text" if ev.text else "attachment")
                payload = {"quick_reply": ev.quick_reply_payload} if ev.quick_reply_payload else None
                if ev.is_like_sticker:
                    payload = {"like_sticker": True}
                if ev.images:
                    payload = {**(payload or {}), "images": ev.images}
                params = {
                    "text": ev.text,
                    "att": _json(ev.attachment_types or None),
                    "payload": _json(payload),
                    "kind": kind,
                }
            else:
                params = {
                    "text": ev.title,
                    "att": None,
                    "payload": _json({"postback": ev.payload}),
                    "kind": "postback",
                }
            row = await s.execute(
                _INSERT_MSG,
                {
                    "cid": cid,
                    "role": Role.user,
                    "mid": ev.mid,
                    "ts": ev.ts,
                    "status": InStatus.pending,
                    **params,
                },
            )
            if row.scalar_one_or_none() is None:
                st.duplicates += 1  # Meta giao lại sự kiện -> không tạo lượt trả lời mới
                continue
            st.stored_messages += 1
            await s.execute(
                _SCHEDULE,
                {"cid": cid, **_debounce_for(ev, settings)},
            )
        elif isinstance(ev, WindowOpener):
            await _upsert_conversation(s, settings, ev.page_id, ev.psid, ev.ts)
            st.window_updates += 1
        elif isinstance(ev, Echo):
            await _handle_echo(s, settings, ev, st)
        elif isinstance(ev, Ignored):
            st.ignored[f"{ev.kind}:{ev.reason}"] = st.ignored.get(f"{ev.kind}:{ev.reason}", 0) + 1
    return st


async def _handle_echo(s: AsyncSession, settings: Settings, ev: Echo, st: IngestStats) -> None:
    if ev.metadata:
        res = await s.execute(
            text(
                """
                UPDATE messages SET echo_confirmed_at = now(),
                    status = CASE WHEN status IN ('sending', 'uncertain') THEN 'sent' ELSE status END,
                    sent_at = COALESCE(sent_at, now()),
                    mid = COALESCE(mid, :mid)
                WHERE idempotency_key = :key
                RETURNING id
                """
            ),
            {"key": ev.metadata, "mid": ev.mid},
        )
        if res.scalar_one_or_none() is not None:
            st.echoes_confirmed += 1
            return
    if settings.meta_app_id and ev.app_id == settings.meta_app_id:
        return  # tin của chính app này (không có metadata đối soát) -> bỏ qua
    # Tin do người (hộp thư Page) hoặc app khác gửi: lưu làm ngữ cảnh, có thể bật handoff.
    cid = await _upsert_conversation(s, settings, ev.page_id, ev.psid, None)
    row = await s.execute(
        _INSERT_MSG,
        {
            "cid": cid,
            "role": Role.human_agent,
            "kind": "text",
            "text": ev.text,
            "att": None,
            "payload": _json({"app_id": ev.app_id}),
            "mid": ev.mid,
            "ts": ev.ts,
            "status": OutStatus.sent,
        },
    )
    if row.scalar_one_or_none() is None:
        return
    st.human_agent_messages += 1
    if settings.auto_handoff_on_human_reply:
        res = await s.execute(
            text(
                """
                UPDATE conversations SET handoff_active = true, handoff_reason = 'human_replied',
                    handoff_by = 'human_reply', handoff_changed_at = now()
                WHERE id = :cid AND handoff_active = false
                RETURNING id
                """
            ),
            {"cid": cid},
        )
        if res.scalar_one_or_none() is not None:
            st.handoffs_from_human_reply += 1


# ============================================================================ hàng đợi (worker)
async def claim_conversation(worker_id: str, lease_seconds: int) -> tuple[uuid.UUID, int] | None:
    async with session_scope() as s:
        row = (
            await s.execute(
                text(
                    """
                    UPDATE conversations c
                    SET lease_owner = :w, lease_expires_at = now() + make_interval(secs => :lease)
                    WHERE c.id = (
                        SELECT id FROM conversations
                        WHERE next_run_at IS NOT NULL AND next_run_at <= now()
                          AND (lease_expires_at IS NULL OR lease_expires_at < now())
                        ORDER BY next_run_at
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    RETURNING c.id, c.attempts
                    """
                ),
                {"w": worker_id, "lease": lease_seconds},
            )
        ).first()
    return (row[0], row[1]) if row else None


async def renew_lease(conv_id: uuid.UUID, worker_id: str, lease_seconds: int) -> bool:
    async with session_scope() as s:
        row = await s.execute(
            text(
                """UPDATE conversations SET lease_expires_at = now() + make_interval(secs => :lease)
                   WHERE id = :cid AND lease_owner = :w RETURNING id"""
            ),
            {"cid": conv_id, "w": worker_id, "lease": lease_seconds},
        )
        return row.scalar_one_or_none() is not None


async def release_success(conv_id: uuid.UUID, worker_id: str) -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """
                UPDATE conversations SET lease_owner = NULL, lease_expires_at = NULL, attempts = 0, last_error = NULL,
                    next_run_at = CASE WHEN first_pending_at IS NOT NULL THEN COALESCE(next_run_at, now()) ELSE NULL END
                WHERE id = :cid AND lease_owner = :w
                """
            ),
            {"cid": conv_id, "w": worker_id},
        )


async def release_retry(conv_id: uuid.UUID, worker_id: str, delay_seconds: float, error: str) -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """
                UPDATE conversations SET lease_owner = NULL, lease_expires_at = NULL,
                    attempts = attempts + 1, last_error = :err,
                    next_run_at = now() + make_interval(secs => :delay)
                WHERE id = :cid AND lease_owner = :w
                """
            ),
            {"cid": conv_id, "w": worker_id, "delay": delay_seconds, "err": error[:500]},
        )


async def start_or_resume_turn(conv_id: uuid.UUID) -> tuple[uuid.UUID, bool] | None:
    """Trả về (turn_id, is_new). Turn mới "nhận" toàn bộ tin pending của cuộc hội thoại."""
    async with session_scope() as s:
        running = (
            await s.execute(
                text(
                    """SELECT id FROM turns WHERE conversation_id = :cid AND status = 'running'
                       ORDER BY created_at LIMIT 1"""
                ),
                {"cid": conv_id},
            )
        ).scalar_one_or_none()
        if running is not None:
            await s.execute(text("UPDATE turns SET attempts = attempts + 1 WHERE id = :t"), {"t": running})
            return running, False
        tid = uuid.uuid4()
        from app.conversation.graph import thread_id_for
        from app.conversation.prompts import PROMPT_VERSION

        await s.execute(
            text(
                """INSERT INTO turns (id, conversation_id, status, attempts, graph_thread_id, prompt_version)
                   VALUES (:t, :cid, 'running', 1, :th, :pv)"""
            ),
            {"t": tid, "cid": conv_id, "th": thread_id_for(str(tid)), "pv": PROMPT_VERSION},
        )
        rows = (
            await s.execute(
                text(
                    """UPDATE messages SET status = 'consumed', turn_id = :t
                       WHERE conversation_id = :cid AND role = 'user' AND status = 'pending'
                       RETURNING event_ts"""
                ),
                {"t": tid, "cid": conv_id},
            )
        ).all()
        await s.execute(
            text("UPDATE conversations SET first_pending_at = NULL WHERE id = :cid"), {"cid": conv_id}
        )
        if not rows:
            await s.execute(text("DELETE FROM turns WHERE id = :t"), {"t": tid})
            return None
        latest = max((r[0] for r in rows if r[0] is not None), default=None)
        await s.execute(
            text("UPDATE turns SET latest_inbound_at = :ts WHERE id = :t"), {"ts": latest, "t": tid}
        )
        return tid, True


class PostgresConversationStore:
    """Cài đặt ``ConversationStore`` cho graph (chỉ đọc)."""

    async def load_turn_context(self, conversation_id: str, turn_id: str, history_limit: int) -> TurnContext:
        async with session_scope() as s:
            c = (
                (
                    await s.execute(
                        text(
                            """SELECT id, page_id, user_ref, summary, summary_upto_seq, handoff_active,
                                  last_user_message_at, last_bot_message_at, last_disclosure_at,
                                  last_fallback_at, bot_resumed_at
                           FROM conversations WHERE id = :cid"""
                        ),
                        {"cid": conversation_id},
                    )
                )
                .mappings()
                .one()
            )
            inbound_rows = (
                (
                    await s.execute(
                        text(
                            """SELECT id, kind, text, attachment_types, payload, event_ts FROM messages
                           WHERE turn_id = :t AND role = 'user' ORDER BY id"""
                        ),
                        {"t": turn_id},
                    )
                )
                .mappings()
                .all()
            )
            first_seq = min((r["id"] for r in inbound_rows), default=None)
            hist_rows = (
                (
                    await s.execute(
                        text(
                            """SELECT id, role, text, attachment_types, payload FROM messages
                           WHERE conversation_id = :cid AND id > :upto AND (:first IS NULL OR id < :first)
                             AND ((role = 'user' AND status IN ('consumed', 'skipped'))
                                  OR (role = 'assistant' AND status IN ('sent', 'uncertain'))
                                  OR role = 'human_agent')
                           ORDER BY id DESC LIMIT :lim"""
                        ),
                        {
                            "cid": conversation_id,
                            "upto": c["summary_upto_seq"],
                            "first": first_seq,
                            "lim": history_limit,
                        },
                    )
                )
                .mappings()
                .all()
            )
            bot_count = (
                await s.execute(
                    text(
                        """SELECT count(*) FROM messages WHERE conversation_id = :cid AND role = 'assistant'
                           AND status IN ('sent', 'uncertain')"""
                    ),
                    {"cid": conversation_id},
                )
            ).scalar_one()
        inbound = []
        for r in inbound_rows:
            payload = r["payload"] or {}
            inbound.append(
                InboundItem(
                    seq=r["id"],
                    kind=r["kind"],
                    text=r["text"],
                    attachment_types=list(r["attachment_types"] or []),
                    payload=payload.get("postback") or payload.get("quick_reply"),
                    event_ts=r["event_ts"],
                    like_sticker=bool(payload.get("like_sticker")),
                    images=list(payload.get("images") or []),
                )
            )
        history = [
            HistoryItem(
                seq=r["id"],
                role=r["role"],
                content=history_content(r["text"], r["attachment_types"], r["payload"]),
            )
            for r in reversed(hist_rows)
        ]
        return TurnContext(
            conversation_id=str(c["id"]),
            page_id=c["page_id"],
            user_ref=c["user_ref"],
            inbound=inbound,
            history=history,
            summary=c["summary"],
            summary_upto_seq=int(c["summary_upto_seq"] or 0),
            handoff_active=bool(c["handoff_active"]),
            last_user_message_at=c["last_user_message_at"],
            last_bot_message_at=c["last_bot_message_at"],
            last_disclosure_at=c["last_disclosure_at"],
            last_fallback_at=c["last_fallback_at"],
            bot_resumed_at=c["bot_resumed_at"],
            bot_messages_count=int(bot_count),
        )


# ============================================================================ kết quả turn + outbox
async def persist_turn_decision(
    conv_id: uuid.UUID, turn_id: uuid.UUID, result: dict[str, Any], parts: list[str]
) -> None:
    """Ghi quyết định của graph (idempotent): tin outbound pending, handoff, tóm tắt, metadata turn."""
    meta = result.get("model_meta") or {}
    async with session_scope() as s:
        for i, part in enumerate(parts):
            await s.execute(
                text(
                    """
                    INSERT INTO messages (conversation_id, role, kind, text, status, turn_id, chunk_index,
                                          idempotency_key, allow_during_handoff)
                    VALUES (:cid, 'assistant', 'text', :text, 'pending', :t, :i, :key, :allow)
                    ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING
                    """
                ),
                {
                    "cid": conv_id,
                    "text": part,
                    "t": turn_id,
                    "i": i,
                    "key": f"{turn_id}:{i}",
                    "allow": bool(result.get("set_handoff")),
                },
            )
        for item in result.get("analyzed_images") or []:
            # thay URL ảnh (link CDN có thời hạn) bằng kết quả phân tích đã che dữ liệu cá nhân; không lưu ảnh
            await s.execute(
                text(
                    """UPDATE messages SET payload = jsonb_set(payload, CAST(:path AS text[]), CAST(:a AS jsonb))
                       WHERE id = :id AND conversation_id = :cid AND role = 'user'
                         AND jsonb_typeof(payload -> 'images') = 'array'"""
                ),
                {
                    "path": f"{{images,{int(item['index'])}}}",
                    "id": int(item["seq"]),
                    "cid": conv_id,
                    "a": _json({"analysis": item["analysis"]}),
                },
            )
        if result.get("set_handoff"):
            await s.execute(
                text(
                    """UPDATE conversations SET handoff_active = true, handoff_reason = 'user_requested',
                           handoff_by = 'user', handoff_changed_at = now()
                       WHERE id = :cid AND handoff_active = false"""
                ),
                {"cid": conv_id},
            )
        if result.get("new_summary") and result.get("new_summary_upto_seq"):
            await s.execute(
                text(
                    """UPDATE conversations SET summary = :sm, summary_upto_seq = :upto, summary_updated_at = now()
                       WHERE id = :cid AND summary_upto_seq < :upto"""
                ),
                {"cid": conv_id, "sm": result["new_summary"], "upto": result["new_summary_upto_seq"]},
            )
        if result.get("is_fallback"):
            await s.execute(
                text("UPDATE conversations SET last_fallback_at = now() WHERE id = :cid"), {"cid": conv_id}
            )
        if result.get("mode") == "handoff_active":
            await s.execute(
                text("UPDATE messages SET status = 'skipped' WHERE turn_id = :t AND role = 'user'"),
                {"t": turn_id},
            )
        await s.execute(
            text(
                """
                UPDATE turns SET mode = :mode, provider = :prov, model_name = :model, adapter_version = :adapter,
                    model_latency_ms = :lat, input_tokens = :itok, output_tokens = :otok,
                    check_flags = CAST(:flags AS JSONB), draft_text = :draft, error = :err
                WHERE id = :t
                """
            ),
            {
                "t": turn_id,
                "mode": result.get("mode"),
                "prov": meta.get("provider"),
                "model": meta.get("model"),
                "adapter": meta.get("adapter"),
                "lat": meta.get("latency_ms"),
                "itok": meta.get("input_tokens"),
                "otok": meta.get("output_tokens"),
                "flags": _json(result.get("check_flags") or []),
                "draft": result.get("draft"),
                "err": result.get("model_error"),
            },
        )


async def outbound_for_turn(turn_id: uuid.UUID) -> list[dict[str, Any]]:
    async with session_scope() as s:
        rows = (
            (
                await s.execute(
                    text(
                        """SELECT id, text, status, chunk_index, idempotency_key, allow_during_handoff, send_attempts
                       FROM messages WHERE turn_id = :t AND role = 'assistant' ORDER BY chunk_index"""
                    ),
                    {"t": turn_id},
                )
            )
            .mappings()
            .all()
        )
    return [dict(r) for r in rows]


@dataclass
class SendGate:
    psid: str
    handoff_active: bool
    last_user_message_at: datetime | None
    latest_inbound_at: datetime | None
    recent_bot_messages: int
    db_now: datetime
    page_id: str = ""


async def send_gate(conv_id: uuid.UUID, turn_id: uuid.UUID) -> SendGate:
    """Đọc lại điều kiện gửi NGAY trước khi gửi (công việc có thể đã chờ trong hàng đợi)."""
    async with session_scope() as s:
        r = (
            (
                await s.execute(
                    text(
                        """SELECT c.psid, c.page_id, c.handoff_active, c.last_user_message_at, t.latest_inbound_at,
                              now() AS db_now,
                              (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id AND m.role = 'assistant'
                                 AND m.status IN ('sent', 'sending', 'uncertain')
                                 AND m.sent_at > now() - interval '60 seconds') AS recent
                       FROM conversations c JOIN turns t ON t.conversation_id = c.id
                       WHERE c.id = :cid AND t.id = :t"""
                    ),
                    {"cid": conv_id, "t": turn_id},
                )
            )
            .mappings()
            .one()
        )
    return SendGate(
        r["psid"],
        r["handoff_active"],
        r["last_user_message_at"],
        r["latest_inbound_at"],
        r["recent"],
        r["db_now"],
        r["page_id"],
    )


async def mark_outbound(
    msg_id: int, status: str, *, error: str | None = None, mid: str | None = None
) -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """UPDATE messages SET status = :st, error = :err, mid = COALESCE(:mid, mid),
                       sent_at = CASE WHEN :mark_sent THEN COALESCE(sent_at, now()) ELSE sent_at END
                   WHERE id = :id"""
            ),
            {
                "id": msg_id,
                "st": status,
                "err": error,
                "mid": mid,
                "mark_sent": status in ("sent", "uncertain"),
            },
        )


async def claim_outbound_for_send(msg_id: int, conv_id: uuid.UUID, worker_id: str) -> bool:
    """pending -> sending (nguyên tử, chỉ khi worker còn giữ lease). Commit TRƯỚC khi gọi Send API."""
    async with session_scope() as s:
        row = await s.execute(
            text(
                """
                UPDATE messages SET status = 'sending', send_attempts = send_attempts + 1
                WHERE id = :id AND status = 'pending'
                  AND EXISTS (SELECT 1 FROM conversations c WHERE c.id = :cid AND c.lease_owner = :w
                              AND c.lease_expires_at > now())
                RETURNING id
                """
            ),
            {"id": msg_id, "cid": conv_id, "w": worker_id},
        )
        return row.scalar_one_or_none() is not None


async def after_sent(conv_id: uuid.UUID, mark_disclosed: bool) -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """UPDATE conversations SET last_bot_message_at = now(),
                       last_disclosure_at = CASE WHEN :d THEN now() ELSE last_disclosure_at END
                   WHERE id = :cid"""
            ),
            {"cid": conv_id, "d": mark_disclosed},
        )


async def complete_turn(
    turn_id: uuid.UUID,
    outcome: str,
    status: str = TurnStatus.completed,
    total_ms: int | None = None,
    trace_id: str | None = None,
) -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """UPDATE turns SET status = :st, outcome = :oc, completed_at = now(),
                       total_latency_ms = COALESCE(:ms, total_latency_ms), trace_id = COALESCE(:tr, trace_id)
                   WHERE id = :t"""
            ),
            {"t": turn_id, "st": status, "oc": outcome, "ms": total_ms, "tr": trace_id},
        )


async def cancel_pending_outbound(turn_id: uuid.UUID, reason: str) -> int:
    async with session_scope() as s:
        res = await s.execute(
            text(
                """UPDATE messages SET status = 'cancelled', error = :r
                   WHERE turn_id = :t AND role = 'assistant' AND status = 'pending' RETURNING id"""
            ),
            {"t": turn_id, "r": reason},
        )
        return len(res.all())


async def conversation_psid(conv_id: uuid.UUID) -> tuple[str, str, bool]:
    async with session_scope() as s:
        r = (
            await s.execute(
                text("SELECT psid, user_ref, handoff_active FROM conversations WHERE id = :cid"),
                {"cid": conv_id},
            )
        ).one()
    return r[0], r[1], r[2]


async def turn_queue_metadata(conv_id: uuid.UUID, turn_id: uuid.UUID) -> tuple[str, int]:
    """Kênh và thời gian chờ của lượt; chỉ metadata, không đọc nội dung tin nhắn."""
    async with session_scope() as s:
        row = (
            await s.execute(
                text(
                    """SELECT c.page_id,
                              GREATEST(0, (extract(epoch FROM (now() - min(m.created_at))) * 1000)::bigint)
                       FROM conversations c JOIN messages m ON m.conversation_id = c.id
                       WHERE c.id = :cid AND m.turn_id = :tid AND m.role = 'user'
                       GROUP BY c.page_id"""
                ),
                {"cid": conv_id, "tid": turn_id},
            )
        ).one()
    return str(row[0]), int(row[1])


async def write_heartbeat(worker_id: str, info: dict[str, Any]) -> None:
    async with session_scope() as s:
        await s.execute(
            text(
                """INSERT INTO worker_heartbeats (worker_id, last_seen, info) VALUES (:w, now(), CAST(:i AS JSONB))
                   ON CONFLICT (worker_id) DO UPDATE SET last_seen = now(), info = EXCLUDED.info"""
            ),
            {"w": worker_id, "i": _json(info)},
        )


async def audit(
    s: AsyncSession, actor: str, action: str, target: str | None, details: dict[str, Any] | None = None
) -> None:
    await s.execute(
        text(
            "INSERT INTO audit_log (actor, action, target, details) VALUES (:a, :ac, :t, CAST(:d AS JSONB))"
        ),
        {"a": actor, "ac": action, "t": target, "d": _json(details)},
    )
