"""Store hội thoại in-memory: dùng cho test tự động và bộ đánh giá (chạy đúng graph production
mà không cần PostgreSQL/Messenger). Mô phỏng các quy tắc chính của repository PostgreSQL."""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.conversation.state import HistoryItem, InboundItem, TurnContext
from app.vision.context import history_content


@dataclass
class _Conv:
    id: str
    page_id: str
    user_ref: str
    summary: str | None = None
    summary_upto_seq: int = 0
    handoff_active: bool = False
    last_user_message_at: datetime | None = None
    last_bot_message_at: datetime | None = None
    last_disclosure_at: datetime | None = None
    last_fallback_at: datetime | None = None
    bot_resumed_at: datetime | None = None


@dataclass
class _Msg:
    seq: int
    conv_id: str
    role: str
    text: str | None
    status: str
    kind: str = "text"
    attachment_types: list[str] = field(default_factory=list)
    payload: dict[str, Any] | None = None
    event_ts: datetime | None = None
    turn_id: str | None = None


class InMemoryConversationStore:
    def __init__(self, now=lambda: datetime.now(UTC)) -> None:  # type: ignore[no-untyped-def]
        self.now = now
        self.convs: dict[str, _Conv] = {}
        self.msgs: list[_Msg] = []
        self._seq = itertools.count(1)

    def ensure_conversation(self, page_id: str, psid: str) -> str:
        key = f"{page_id}:{psid}"
        if key not in self.convs:
            self.convs[key] = _Conv(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)), page_id=page_id, user_ref=f"u_{psid}"
            )
        return self.convs[key].id

    def _conv(self, conv_id: str) -> _Conv:
        return next(c for c in self.convs.values() if c.id == conv_id)

    def add_user_message(
        self,
        conv_id: str,
        text: str | None = None,
        *,
        attachment_types: list[str] | None = None,
        postback: str | None = None,
        like_sticker: bool = False,
        ts: datetime | None = None,
        images: list[dict[str, Any]] | None = None,
    ) -> int:
        ts = ts or self.now()
        payload: dict[str, Any] | None = None
        kind = "text" if text else "attachment"
        if postback:
            payload, kind = {"postback": postback}, "postback"
        if like_sticker:
            payload = {"like_sticker": True}
        if images:
            payload = {**(payload or {}), "images": images}
        seq = next(self._seq)
        self.msgs.append(
            _Msg(seq, conv_id, "user", text, "pending", kind, list(attachment_types or []), payload, ts)
        )
        c = self._conv(conv_id)
        c.last_user_message_at = max(filter(None, [c.last_user_message_at, ts]))
        return seq

    def add_human_agent_message(self, conv_id: str, text: str) -> None:
        self.msgs.append(_Msg(next(self._seq), conv_id, "human_agent", text, "sent"))

    def begin_turn(self, conv_id: str) -> str | None:
        pending = [
            m for m in self.msgs if m.conv_id == conv_id and m.role == "user" and m.status == "pending"
        ]
        if not pending:
            return None
        tid = str(uuid.uuid4())
        for m in pending:
            m.status, m.turn_id = "consumed", tid
        return tid

    async def load_turn_context(self, conversation_id: str, turn_id: str, history_limit: int) -> TurnContext:
        c = self._conv(conversation_id)
        inbound = [m for m in self.msgs if m.turn_id == turn_id and m.role == "user"]
        first = min((m.seq for m in inbound), default=None)
        hist = [
            m
            for m in self.msgs
            if m.conv_id == conversation_id
            and m.seq > c.summary_upto_seq
            and (first is None or m.seq < first)
            and (
                (m.role == "user" and m.status in ("consumed", "skipped"))
                or (m.role == "assistant" and m.status in ("sent", "uncertain"))
                or m.role == "human_agent"
            )
        ][-history_limit:]
        bot_count = sum(
            1
            for m in self.msgs
            if m.conv_id == conversation_id and m.role == "assistant" and m.status == "sent"
        )
        return TurnContext(
            conversation_id=c.id,
            page_id=c.page_id,
            user_ref=c.user_ref,
            inbound=[
                InboundItem(
                    seq=m.seq,
                    kind=m.kind,
                    text=m.text,
                    attachment_types=m.attachment_types,
                    payload=(m.payload or {}).get("postback"),
                    event_ts=m.event_ts,
                    like_sticker=bool((m.payload or {}).get("like_sticker")),
                    images=list((m.payload or {}).get("images") or []),
                )
                for m in inbound
            ],
            history=[
                HistoryItem(m.seq, m.role, history_content(m.text, m.attachment_types, m.payload))
                for m in hist
            ],
            summary=c.summary,
            summary_upto_seq=c.summary_upto_seq,
            handoff_active=c.handoff_active,
            last_user_message_at=c.last_user_message_at,
            last_bot_message_at=c.last_bot_message_at,
            last_disclosure_at=c.last_disclosure_at,
            last_fallback_at=c.last_fallback_at,
            bot_resumed_at=c.bot_resumed_at,
            bot_messages_count=bot_count,
        )

    def apply_result(self, conv_id: str, turn_id: str, result: dict[str, Any]) -> list[str]:
        """Áp dụng quyết định của graph như worker + coi như gửi thành công. Trả về các tin đã 'gửi'."""
        c = self._conv(conv_id)
        now = self.now()
        if result.get("set_handoff"):
            c.handoff_active = True
        if result.get("new_summary") and (result.get("new_summary_upto_seq") or 0) > c.summary_upto_seq:
            c.summary, c.summary_upto_seq = result["new_summary"], result["new_summary_upto_seq"]
        if result.get("mode") == "handoff_active":
            for m in self.msgs:
                if m.turn_id == turn_id and m.role == "user":
                    m.status = "skipped"
        if result.get("is_fallback"):
            c.last_fallback_at = now
        for item in result.get("analyzed_images") or []:
            msg = next(x for x in self.msgs if x.seq == item["seq"])
            msg.payload["images"][item["index"]] = {"analysis": item["analysis"]}
        parts = list(result.get("reply_parts") or []) if result.get("action") == "reply" else []
        for p in parts:
            self.msgs.append(_Msg(next(self._seq), conv_id, "assistant", p, "sent", turn_id=turn_id))
        if parts:
            c.last_bot_message_at = now
            if result.get("mark_disclosed"):
                c.last_disclosure_at = now
        return parts

    def set_handoff(self, conv_id: str, active: bool) -> None:
        c = self._conv(conv_id)
        c.handoff_active = active
        if not active:
            c.bot_resumed_at = self.now()

    def delete_conversation(self, conv_id: str) -> None:
        self.msgs = [m for m in self.msgs if m.conv_id != conv_id]
        self.convs = {k: v for k, v in self.convs.items() if v.id != conv_id}
