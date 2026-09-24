"""Tác vụ định kỳ trong worker: dọn checkpoint cũ, tự bật lại bot sau handoff (nếu cấu hình),
xóa dữ liệu quá hạn lưu trữ (nếu cấu hình)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from app.storage.db import session_scope

if TYPE_CHECKING:
    from app.workers.processor import WorkerDeps

log = logging.getLogger(__name__)


async def prune_checkpoints(checkpointer: Any, retention_hours: float, batch: int = 200) -> int:
    async with session_scope() as s:
        rows = (
            await s.execute(
                text(
                    """SELECT id, graph_thread_id FROM turns
                       WHERE status <> 'running' AND NOT checkpoint_deleted AND graph_thread_id IS NOT NULL
                         AND COALESCE(completed_at, created_at) < now() - make_interval(secs => :sec)
                       ORDER BY created_at LIMIT :lim"""
                ),
                {"sec": retention_hours * 3600, "lim": batch},
            )
        ).all()
    for turn_id, thread in rows:
        await checkpointer.adelete_thread(thread)
        async with session_scope() as s:
            await s.execute(text("UPDATE turns SET checkpoint_deleted = true WHERE id = :t"), {"t": turn_id})
    return len(rows)


async def auto_resume_handoffs(hours: float) -> int:
    if hours <= 0:
        return 0
    async with session_scope() as s:
        res = await s.execute(
            text(
                """UPDATE conversations SET handoff_active = false, handoff_by = 'auto_resume',
                       handoff_changed_at = now(), bot_resumed_at = now()
                   WHERE handoff_active AND handoff_changed_at < now() - make_interval(secs => :sec)
                   RETURNING id"""
            ),
            {"sec": hours * 3600},
        )
        return len(res.all())


async def purge_expired(deps: WorkerDeps, checkpointer: Any) -> int:
    days = deps.settings.data_retention_days
    if days <= 0:
        return 0
    from app.admin.service import delete_conversation

    async with session_scope() as s:
        ids = (
            (
                await s.execute(
                    text(
                        """SELECT id FROM conversations
                       WHERE COALESCE(last_user_message_at, created_at) < now() - make_interval(days => :d)
                         AND lease_owner IS NULL LIMIT 100"""
                    ),
                    {"d": days},
                )
            )
            .scalars()
            .all()
        )
    for cid in ids:
        await delete_conversation(cid, "retention_policy", checkpointer, deps.tracer)
    return len(ids)


async def housekeeping_loop(deps: WorkerDeps, checkpointer: Any, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=deps.settings.housekeeping_interval_seconds)
        if stop.is_set():
            return
        try:
            pruned = await prune_checkpoints(checkpointer, deps.settings.checkpoint_retention_hours)
            resumed = await auto_resume_handoffs(deps.settings.handoff_auto_resume_hours)
            purged = await purge_expired(deps, checkpointer)
            async with session_scope() as s:
                await s.execute(
                    text("DELETE FROM worker_heartbeats WHERE last_seen < now() - interval '1 day'")
                )
            if pruned or resumed or purged:
                log.info(
                    "housekeeping",
                    extra={"checkpoints_pruned": pruned, "handoffs_resumed": resumed, "purged": purged},
                )
        except Exception:
            log.exception("housekeeping_failed")
