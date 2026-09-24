"""Phân tích payload webhook Messenger thành các sự kiện có kiểu.

Payload chuẩn: ``{"object": "page", "entry": [{"id": PAGE_ID, "messaging": [event, ...]}]}``.
Một request có thể chứa nhiều entry và nhiều event.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.observability.redaction import nfc

# Nút "Like" (ngón cái) của Messenger gửi dưới dạng sticker với các id cố định.
LIKE_STICKER_IDS = {369239263222822, 369239343222814, 369239383222810}


MAX_IMAGES_PER_MESSAGE = 3


def _ts(ms: Any) -> datetime:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return datetime.now(UTC)


@dataclass(slots=True)
class UserMessage:
    page_id: str
    psid: str
    mid: str
    ts: datetime
    text: str | None
    attachment_types: list[str] = field(default_factory=list)
    quick_reply_payload: str | None = None
    sticker_id: int | None = None
    # ảnh kèm theo: [{"url": ...}] (Messenger) hoặc [{"analysis": {...}}] (website, đã phân tích sẵn)
    images: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_like_sticker(self) -> bool:
        return self.sticker_id in LIKE_STICKER_IDS and not self.text


@dataclass(slots=True)
class Postback:
    page_id: str
    psid: str
    mid: str
    ts: datetime
    title: str | None
    payload: str | None


@dataclass(slots=True)
class Echo:
    """Tin do Page gửi đi (bởi app này, app khác, hoặc người trả lời từ hộp thư Page)."""

    page_id: str
    psid: str
    mid: str
    ts: datetime
    text: str | None
    app_id: str | None
    metadata: str | None


@dataclass(slots=True)
class WindowOpener:
    """Tương tác của người dùng mở/gia hạn cửa sổ 24h nhưng không cần trả lời (reaction, referral...)."""

    page_id: str
    psid: str
    ts: datetime
    kind: str


@dataclass(slots=True)
class Ignored:
    kind: str
    reason: str
    page_id: str | None = None


ParsedEvent = UserMessage | Postback | Echo | WindowOpener | Ignored


def _fallback_mid(prefix: str, psid: str, ts_ms: Any, extra: Any) -> str:
    raw = json.dumps([psid, ts_ms, extra], ensure_ascii=False, sort_keys=True)
    return f"{prefix}:{hashlib.sha256(raw.encode()).hexdigest()[:32]}"


def parse_webhook_payload(payload: dict[str, Any], expected_page_ids: set[str]) -> list[ParsedEvent]:
    events: list[ParsedEvent] = []
    if payload.get("object") != "page":
        return [Ignored(kind="object", reason=f"object={payload.get('object')!r}")]
    for entry in payload.get("entry") or []:
        if not isinstance(entry, dict):
            continue
        page_id = str(entry.get("id", ""))
        if expected_page_ids and page_id not in expected_page_ids:
            events.append(Ignored(kind="entry", reason="unexpected_page", page_id=page_id))
            continue
        if entry.get("standby"):
            # Handover protocol: app này không phải primary receiver -> không xử lý.
            events.append(Ignored(kind="standby", reason="standby_channel", page_id=page_id))
        for ev in entry.get("messaging") or []:
            if isinstance(ev, dict):
                events.append(_parse_messaging_event(page_id, ev))
    return events


def _parse_messaging_event(page_id: str, ev: dict[str, Any]) -> ParsedEvent:
    sender = str((ev.get("sender") or {}).get("id", ""))
    recipient = str((ev.get("recipient") or {}).get("id", ""))
    ts_ms = ev.get("timestamp")
    ts = _ts(ts_ms)

    if "message" in ev and isinstance(ev["message"], dict):
        msg = ev["message"]
        if msg.get("is_echo"):
            if sender != page_id:
                return Ignored(kind="echo", reason="sender_not_page", page_id=page_id)
            return Echo(
                page_id=page_id,
                psid=recipient,
                mid=str(msg.get("mid") or _fallback_mid("echo", recipient, ts_ms, msg.get("text"))),
                ts=ts,
                text=nfc(msg["text"]) if isinstance(msg.get("text"), str) else None,
                app_id=str(msg["app_id"]) if msg.get("app_id") is not None else None,
                metadata=msg.get("metadata") if isinstance(msg.get("metadata"), str) else None,
            )
        if recipient != page_id:
            return Ignored(kind="message", reason="recipient_not_page", page_id=page_id)
        if msg.get("is_deleted"):
            return Ignored(kind="message", reason="deleted", page_id=page_id)
        attachments = msg.get("attachments") or []
        att_types = [str(a.get("type", "unknown")) for a in attachments if isinstance(a, dict)]
        images = [
            {"url": a["payload"]["url"]}
            for a in attachments
            if isinstance(a, dict)
            and a.get("type") == "image"
            and isinstance(a.get("payload"), dict)
            and isinstance(a["payload"].get("url"), str)
            and not a["payload"].get("sticker_id")  # sticker cũng là "image" -> bỏ qua
        ][:MAX_IMAGES_PER_MESSAGE]
        sticker_id = msg.get("sticker_id")
        if not images and (sticker_id or "sticker" in att_types):
            # sticker được gửi dưới dạng "image" -> không để bot nói "chưa xem được hình ảnh" với sticker
            att_types = [t for t in att_types if t != "image"] or ["sticker"]
        text = msg.get("text") if isinstance(msg.get("text"), str) else None
        qr = msg.get("quick_reply") or {}
        return UserMessage(
            page_id=page_id,
            psid=sender,
            mid=str(msg.get("mid") or _fallback_mid("msg", sender, ts_ms, text)),
            ts=ts,
            text=nfc(text).strip() if text else None,
            attachment_types=att_types,
            quick_reply_payload=qr.get("payload") if isinstance(qr, dict) else None,
            sticker_id=int(sticker_id)
            if isinstance(sticker_id, int | str) and str(sticker_id).isdigit()
            else None,
            images=images,
        )

    if "postback" in ev and isinstance(ev["postback"], dict):
        if recipient != page_id:
            return Ignored(kind="postback", reason="recipient_not_page", page_id=page_id)
        pb = ev["postback"]
        return Postback(
            page_id=page_id,
            psid=sender,
            mid=str(pb.get("mid") or _fallback_mid("pb", sender, ts_ms, pb.get("payload"))),
            ts=ts,
            title=nfc(pb["title"]) if isinstance(pb.get("title"), str) else None,
            payload=pb.get("payload") if isinstance(pb.get("payload"), str) else None,
        )

    for kind in ("reaction", "referral", "optin"):
        if kind in ev:
            if recipient != page_id:
                return Ignored(kind=kind, reason="recipient_not_page", page_id=page_id)
            return WindowOpener(page_id=page_id, psid=sender, ts=ts, kind=kind)

    for kind in (
        "delivery",
        "read",
        "account_linking",
        "policy_enforcement",
        "app_roles",
        "pass_thread_control",
        "take_thread_control",
        "request_thread_control",
        "message_edit",
        "game_play",
    ):
        if kind in ev:
            return Ignored(kind=kind, reason="not_needed", page_id=page_id)
    return Ignored(kind="unknown", reason="unsupported_event", page_id=page_id)
