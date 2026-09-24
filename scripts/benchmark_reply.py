"""Measure real web → queue → worker → model → reply latency using synthetic questions.

    .venv/bin/python scripts/benchmark_reply.py --label before

Creates isolated web conversations (no Messenger messages), saves only these synthetic
questions/replies and timing to runs/benchmarks/<label>.json. Does not delete user data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx
from sqlalchemy import text

from app.config import get_settings
from app.storage.db import dispose_db, init_db, session_scope

QUESTIONS = [
    "Chào bạn, hôm nay mình muốn trò chuyện một chút.",
    "RAG khác fine-tuning thế nào?",
    "Mình hay quên bài đã học, có cách nào ôn tập hiệu quả hơn không?",
]


async def benchmark() -> list[dict]:
    settings = get_settings()
    init_db(settings)
    results = []
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=15) as client:
            for question in QUESTIONS:
                response = await client.post("/web/session")
                response.raise_for_status()
                session = response.json()
                start = time.perf_counter()
                response = await client.post("/web/messages", json={**session, "text": question})
                response.raise_for_status()
                while time.perf_counter() - start < 90:
                    await asyncio.sleep(0.2)
                    response = await client.get("/web/messages", params=session)
                    response.raise_for_status()
                    body = response.json()
                    if not body["messages"] or body["pending"]:
                        continue
                    elapsed = time.perf_counter() - start
                    async with session_scope() as db:
                        row = (
                            (
                                await db.execute(
                                    text(
                                        "SELECT t.model_latency_ms,t.total_latency_ms,t.output_tokens,t.check_flags,"
                                        "t.trace_id,t.prompt_version,t.outcome FROM turns t "
                                        "JOIN conversations c ON c.id=t.conversation_id WHERE c.psid=:sid "
                                        "AND c.page_id='web' ORDER BY t.created_at DESC LIMIT 1"
                                    ),
                                    {"sid": session["session_id"]},
                                )
                            )
                            .mappings()
                            .first()
                        )
                    reply = "\n\n".join(m["text"] for m in body["messages"])
                    record = {
                        "question": question,
                        "reply": reply,
                        "chars": len(reply),
                        "parts": len(body["messages"]),
                        "wall_seconds": round(elapsed, 2),
                        **dict(row or {}),
                    }
                    results.append(record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    break
                else:
                    raise TimeoutError("Synthetic turn did not complete within 90 seconds")
        return results
    finally:
        await dispose_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("label must use letters, digits, hyphens or underscores")
    target = Path("runs/benchmarks") / f"{args.label}.json"
    if target.exists():
        parser.error(f"Refusing to overwrite {target}")
    result = asyncio.run(benchmark())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
