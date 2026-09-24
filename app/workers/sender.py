"""Gửi các tin outbound của một turn qua Send API với kiểm tra lại điều kiện ngay trước khi gửi.

Bất biến an toàn:
- Mỗi chunk có idempotency key ``<turn_id>:<i>``; chuyển pending -> sending là nguyên tử và được commit
  TRƯỚC khi gọi API; worker chỉ gửi khi còn giữ lease.
- Gặp chunk ở trạng thái sending/uncertain (worker trước chết giữa chừng hoặc timeout) -> đánh dấu
  uncertain và DỪNG các chunk còn lại (không retry mù). Echo webhook (metadata = idempotency key) sẽ
  xác nhận lại nếu Meta thực sự đã nhận.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import timedelta

from app.config import SendMode, Settings
from app.messenger.client import MessengerClient, SendOutcome
from app.storage import repository as repo
from app.storage.models import OutStatus
from app.web.channel import is_web

log = logging.getLogger(__name__)


@dataclass
class SendReport:
    sent: int = 0
    outcome: str = "sent"


def gate_decision(settings: Settings, gate: repo.SendGate, allow_during_handoff: bool) -> str | None:
    """Trả về lý do không được gửi (None = được gửi)."""
    if gate.handoff_active and not allow_during_handoff:
        return "cancelled_handoff"
    if gate.last_user_message_at is None:
        return "cancelled_no_user_interaction"
    window = timedelta(hours=settings.messaging_window_hours) - timedelta(
        minutes=settings.messaging_window_safety_minutes
    )
    if gate.db_now - gate.last_user_message_at > window:
        return "cancelled_window_closed"
    if (
        gate.latest_inbound_at
        and (gate.db_now - gate.latest_inbound_at).total_seconds() > settings.reply_max_age_seconds
    ):
        return "cancelled_stale"
    if gate.recent_bot_messages >= settings.max_bot_messages_per_minute:
        return "cancelled_rate_limited_locally"
    return None


def send_allowed_for(settings: Settings, psid: str) -> bool:
    if settings.messenger_send_mode == SendMode.disabled:
        return False
    if settings.messenger_send_mode == SendMode.allowlist:
        return psid in settings.messenger_allowed_psids
    return True


async def send_turn(
    settings: Settings,
    client: MessengerClient,
    *,
    conv_id: uuid.UUID,
    turn_id: uuid.UUID,
    worker_id: str,
    mark_disclosed: bool,
) -> SendReport:
    report = SendReport()
    rows = await repo.outbound_for_turn(turn_id)
    for row in rows:
        status = row["status"]
        if status in (OutStatus.sent, OutStatus.dry_run):
            report.sent += status == OutStatus.sent
            continue
        if status in (OutStatus.sending, OutStatus.uncertain):
            await repo.mark_outbound(
                row["id"], OutStatus.uncertain, error="unknown_delivery_after_interruption"
            )
            await repo.cancel_pending_outbound(turn_id, "previous_chunk_uncertain")
            report.outcome = "uncertain_delivery"
            return report
        if status in (OutStatus.failed, OutStatus.cancelled):
            report.outcome = status
            return report
        # --- pending: kiểm tra lại điều kiện gửi
        gate = await repo.send_gate(conv_id, turn_id)
        reason = gate_decision(settings, gate, bool(row["allow_during_handoff"]))
        if reason:
            await repo.cancel_pending_outbound(turn_id, reason)
            report.outcome = reason
            return report
        if is_web(gate.page_id):
            # kênh website: không có API ngoài để gọi, widget tự lấy tin -> giao ngay, không retry, không trùng
            await repo.mark_outbound(row["id"], OutStatus.sent, mid=f"web:{row['idempotency_key']}")
            await repo.after_sent(conv_id, mark_disclosed)
            report.sent += 1
            continue
        if not send_allowed_for(settings, gate.psid):
            await repo.mark_outbound(
                row["id"], OutStatus.dry_run, error=f"send_mode={settings.messenger_send_mode}"
            )
            report.outcome = "dry_run"
            continue
        # --- gửi với retry có giới hạn cho lỗi chắc chắn chưa gửi / lỗi tạm thời
        attempt = 0
        while True:
            if not await repo.claim_outbound_for_send(row["id"], conv_id, worker_id):
                report.outcome = "lease_lost_or_already_sending"
                return report
            res = await client.send_text(gate.psid, row["text"], metadata=row["idempotency_key"])
            attempt += 1
            if res.outcome == SendOutcome.sent:
                await repo.mark_outbound(row["id"], OutStatus.sent, mid=res.message_id)
                await repo.after_sent(conv_id, mark_disclosed)
                report.sent += 1
                break
            if res.outcome == SendOutcome.uncertain:
                await repo.mark_outbound(row["id"], OutStatus.uncertain, error=res.detail)
                await repo.cancel_pending_outbound(turn_id, "previous_chunk_uncertain")
                report.outcome = "uncertain_delivery"
                return report
            if (
                res.outcome in (SendOutcome.retryable, SendOutcome.rate_limited)
                and attempt < settings.messenger_send_max_attempts
            ):
                await repo.mark_outbound(row["id"], OutStatus.pending, error=f"{res.outcome}:{res.detail}")
                await asyncio.sleep(1.5 * attempt * (3 if res.outcome == SendOutcome.rate_limited else 1))
                gate = await repo.send_gate(conv_id, turn_id)
                reason = gate_decision(settings, gate, bool(row["allow_during_handoff"]))
                if reason:
                    await repo.cancel_pending_outbound(turn_id, reason)
                    report.outcome = reason
                    return report
                continue
            # lỗi vĩnh viễn / hết lượt thử
            await repo.mark_outbound(
                row["id"],
                OutStatus.failed,
                error=f"{res.outcome}:{res.error_code}:{res.error_subcode}:{res.detail}",
            )
            await repo.cancel_pending_outbound(turn_id, f"previous_chunk_{res.outcome}")
            if res.outcome == SendOutcome.auth_error:
                log.error("page_token_invalid_or_expired - cần tạo lại Page Access Token")
            report.outcome = f"send_failed_{res.outcome}"
            return report
    return report
