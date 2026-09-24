"""Chạy thử câu hỏi qua ĐÚNG graph production (model, kho kiến thức, tài liệu, dịch vụ ảnh thật) mà không gửi Messenger.

    uv run python scripts/replay_questions.py "câu hỏi 1" "câu hỏi 2"
    uv run python scripts/replay_questions.py --thread "câu 1" "câu hỏi tiếp"   # cùng một hội thoại

Mỗi câu một hội thoại mới (mặc định) hoặc chung một hội thoại (--thread). Dữ liệu chỉ ở bộ nhớ, không ghi DB.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta

from app.config import get_settings
from app.conversation.graph import GraphContext, build_graph, run_turn
from app.conversation.knowledge import load_knowledge
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.profile import load_profile
from app.providers.llm import build_provider
from app.rag.embedder import HttpEmbedder
from app.rag.retriever import DocSearch
from app.storage.db import init_db
from app.vision.client import VisionClient


async def main(questions: list[str], thread: bool) -> None:
    s = get_settings()
    init_db(s)
    clock = [datetime.now(UTC)]
    store = InMemoryConversationStore(now=lambda: clock[0])
    ctx = GraphContext(
        settings=s, store=store, llm=build_provider(s), profile=load_profile(s.fanpage_profile_path),
        knowledge=load_knowledge(s.knowledge_dir), docs=DocSearch(s, HttpEmbedder(s)) if s.rag_docs_enabled else None,
        vision=VisionClient(s) if s.vision_enabled else None, now=lambda: clock[0],
    )
    graph = build_graph()
    for i, q in enumerate(questions):
        cid = store.ensure_conversation("REPLAY", "thread" if thread else f"q{i}")
        store.add_user_message(cid, q, ts=clock[0] - timedelta(seconds=1))
        tid = store.begin_turn(cid)
        r = await run_turn(graph, turn_id=tid, conversation_id=cid, context=ctx)
        store.apply_result(cid, tid, r)
        clock[0] += timedelta(seconds=30)
        print("=" * 100)
        print(f"Q: {q}\n   mode={r.get('mode')} hits={[h['id'] for h in r.get('knowledge_hits') or []]} "
              f"flags={r.get('check_flags')}")
        print("\n".join(r.get("reply_parts") or ["(không trả lời)"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--thread", action="store_true", help="các câu thuộc cùng một hội thoại")
    a = ap.parse_args()
    asyncio.run(main(a.questions, a.thread))
