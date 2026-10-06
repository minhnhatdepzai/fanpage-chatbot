"""Tiến trình worker: claim hội thoại từ hàng đợi PostgreSQL và xử lý song song (tuần tự trong từng hội thoại).

uv run botctl worker          # hoặc: uv run python -m app.workers.main
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import uuid

from app.config import Settings, get_settings
from app.conversation.graph import build_graph
from app.conversation.knowledge import load_knowledge
from app.conversation.profile import load_profile
from app.messenger.client import MessengerClient
from app.observability.logging_setup import configure_logging
from app.observability.tracing import build_tracer
from app.providers.llm import build_provider
from app.rag.embedder import HttpEmbedder
from app.rag.retriever import DocSearch
from app.storage import repository as repo
from app.storage.db import dispose_db, init_db
from app.vision.client import VisionClient
from app.websearch import build_web_search
from app.workers.housekeeping import housekeeping_loop
from app.workers.processor import WorkerDeps, process_conversation

log = logging.getLogger(__name__)


async def _slot(deps: WorkerDeps, stop: asyncio.Event, slot: int) -> None:
    s = deps.settings
    while not stop.is_set():
        try:
            claimed = await repo.claim_conversation(deps.worker_id, s.worker_lease_seconds)
        except Exception:  # DB tạm lỗi -> chờ rồi thử lại
            log.exception("claim_failed")
            await asyncio.sleep(2)
            continue
        if claimed is None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=s.worker_poll_interval_seconds)
            continue
        conv_id, attempts = claimed
        await process_conversation(deps, conv_id, attempts)


async def _heartbeat(deps: WorkerDeps, stop: asyncio.Event) -> None:
    while not stop.is_set():
        with contextlib.suppress(Exception):
            await repo.write_heartbeat(
                deps.worker_id, {"pid": os.getpid(), "concurrency": deps.settings.worker_concurrency}
            )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=15)


async def run_worker(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    configure_logging(s.log_level, s.secret_values())
    init_db(s)
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        s.psycopg_conninfo,
        min_size=1,
        max_size=max(2, s.worker_concurrency + 1),
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
    deps = WorkerDeps(
        settings=s,
        graph=build_graph(checkpointer),
        llm=build_provider(s),
        messenger=MessengerClient(s),
        tracer=build_tracer(s),
        profile=load_profile(s.fanpage_profile_path),
        store=repo.PostgresConversationStore(),
        worker_id=worker_id,
        knowledge=load_knowledge(s.knowledge_dir),
        docs=DocSearch(s, HttpEmbedder(s)) if s.rag_docs_enabled else None,
        vision=VisionClient(s) if s.vision_enabled else None,
        web_search=build_web_search(s),
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)
    log.info(
        "worker_started",
        extra={
            "worker_id": worker_id,
            "concurrency": s.worker_concurrency,
            "provider": s.llm_provider.value,
            "send_mode": str(s.messenger_send_mode),
            "tracing": deps.tracer.enabled,
            "knowledge_entries": len(deps.knowledge),
            "grounding_mode": s.grounding_mode.value,
            "web_search": deps.web_search is not None,
            "vision_vlm": bool(getattr(deps.vision, "vlm_enabled", False)),
        },
    )
    tasks = [asyncio.create_task(_slot(deps, stop, i)) for i in range(s.worker_concurrency)]
    tasks.append(asyncio.create_task(_heartbeat(deps, stop)))
    tasks.append(asyncio.create_task(housekeeping_loop(deps, checkpointer, stop)))
    await stop.wait()
    log.info("worker_stopping")
    # chờ các lượt đang xử lý kết thúc (tối đa lease); lease hết hạn thì worker khác sẽ tiếp quản
    done, pending = await asyncio.wait(tasks, timeout=s.worker_lease_seconds)
    for t in pending:
        t.cancel()
    await deps.messenger.aclose()
    deps.tracer.shutdown()
    await pool.close()
    await dispose_db()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
