"""Tích hợp với PostgreSQL thật (docker compose up -d postgres). Meta Graph API luôn được mock.

Chứng minh: ACK chỉ sau khi lưu bền vững, chống trùng, lọc Page, tuần tự theo hội thoại, resume sau
crash không gửi trùng, handoff chặn tin đang chờ, không gửi tin quá hạn, timeout gửi -> uncertain,
giữ ngữ cảnh sau restart, xóa dữ liệu kể cả checkpoint, admin API cần xác thực.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import text

from app.admin import service
from app.config import get_settings
from app.conversation.graph import build_graph, thread_id_for
from app.conversation.profile import FanpageProfile
from app.main import app
from app.messenger.client import MessengerClient
from app.observability.tracing import Tracer
from app.providers.llm import FakeProvider
from app.storage import repository as repo
from app.storage.db import session_scope
from app.workers.processor import WorkerDeps, process_conversation
from tests.helpers import echo_event, msg_event, now_ms, payload, signed

pytestmark = pytest.mark.db
SEND_URL = "https://graph.facebook.com/v26.0/PAGE1/messages"


@pytest.fixture(autouse=True)
def _tracer():  # lifespan không chạy dưới ASGITransport
    app.state.tracer = Tracer()


async def post(obj, *, headers_override=None, raw=None):  # type: ignore[no-untyped-def]
    body, headers = signed(obj)
    if raw is not None:
        body = raw
    headers.update(headers_override or {})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        return await c.post("/webhook", content=body, headers=headers)


async def get(path: str, **kw):  # type: ignore[no-untyped-def]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        return await c.get(path, **kw)


@asynccontextmanager
async def worker(provider: FakeProvider, worker_id: str = "w1"):  # type: ignore[no-untyped-def]
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    s = get_settings()
    async with AsyncPostgresSaver.from_conn_string(s.psycopg_conninfo) as saver:
        deps = WorkerDeps(
            settings=s, graph=build_graph(saver), llm=provider, messenger=MessengerClient(s), tracer=Tracer(),
            profile=FanpageProfile(name="Góc Đọc Sách"), store=repo.PostgresConversationStore(), worker_id=worker_id,
        )
        try:
            yield deps, saver
        finally:
            await deps.messenger.aclose()


async def run_once(deps: WorkerDeps) -> bool:
    claimed = await repo.claim_conversation(deps.worker_id, 60)
    if claimed is None:
        return False
    await process_conversation(deps, *claimed)
    return True


def mock_send(http_mock, *, side_effect=None):  # type: ignore[no-untyped-def]
    route = http_mock.post(SEND_URL)
    if side_effect is not None:
        route.side_effect = side_effect
    else:
        route.respond(200, json={"recipient_id": "u", "message_id": f"m_{uuid.uuid4().hex[:8]}"})
    return route


async def scalar(sql: str, **params):  # type: ignore[no-untyped-def]
    async with session_scope() as s:
        return (await s.execute(text(sql), params)).scalar()


# ============================================================ webhook
async def test_get_verification_ok_and_rejected():
    ok = await get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "verify-token-test-123",
                                        "hub.challenge": "12345"})
    assert ok.status_code == 200 and ok.text == "12345"
    bad = await get("/webhook", params={"hub.mode": "subscribe", "hub.verify_token": "sai", "hub.challenge": "1"})
    assert bad.status_code == 403
    no_mode = await get("/webhook", params={"hub.verify_token": "verify-token-test-123", "hub.challenge": "1"})
    assert no_mode.status_code == 403


async def test_post_signature_missing_or_wrong_is_rejected(db):
    obj = payload(("PAGE1", [msg_event("u1", "m1", "hi")]))
    assert (await post(obj, headers_override={"X-Hub-Signature-256": ""})).status_code == 401
    assert (await post(obj, headers_override={"X-Hub-Signature-256": "sha256=" + "0" * 64})).status_code == 401
    assert await scalar("SELECT count(*) FROM messages") == 0


async def test_post_body_too_large_rejected(db):
    big = b"{" + b" " * (get_settings().webhook_max_body_bytes + 10) + b"}"
    r = await post(payload(), raw=big)
    assert r.status_code == 413


async def test_multiple_events_stored_and_duplicates_do_not_schedule_again(db):
    obj = payload(("PAGE1", [msg_event("u1", "m1", "chào"), msg_event("u2", "m2", "hello")]),
                  ("PAGE1", [msg_event("u1", "m3", "bạn ơi")]))
    r = await post(obj)
    assert r.status_code == 200 and r.text == "EVENT_RECEIVED"
    assert await scalar("SELECT count(*) FROM messages") == 3
    assert await scalar("SELECT count(*) FROM conversations WHERE next_run_at IS NOT NULL") == 2
    # Meta giao lại đúng payload đó
    await post(obj)
    assert await scalar("SELECT count(*) FROM messages") == 3


async def test_other_page_and_echo_not_scheduled_human_echo_sets_handoff(db):
    await post(payload(("OTHER", [msg_event("u9", "x1", "hi", page="OTHER")])))
    assert await scalar("SELECT count(*) FROM conversations") == 0
    await post(payload(("PAGE1", [echo_event("u1", "e1", "tin của bot", app_id="APP123")])))
    assert await scalar("SELECT count(*) FROM messages") == 0
    await post(payload(("PAGE1", [echo_event("u1", "e2", "Chào bạn, mình là admin", app_id="263902037430900")])))
    assert await scalar("SELECT count(*) FROM messages WHERE role = 'human_agent'") == 1
    assert await scalar("SELECT handoff_active FROM conversations") is True
    assert await scalar("SELECT count(*) FROM conversations WHERE next_run_at IS NOT NULL") == 0


async def test_database_failure_is_not_acknowledged(db, monkeypatch):
    async def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise ConnectionError("db down")

    monkeypatch.setattr("app.api.webhook.ingest_events", boom)
    r = await post(payload(("PAGE1", [msg_event("u1", "m1", "hi")])))
    assert r.status_code == 503 and r.text != "EVENT_RECEIVED"


async def test_admin_api_requires_auth(db):
    assert (await get("/admin/turns")).status_code == 401
    assert (await get("/admin/turns", headers={"Authorization": "Bearer wrong"})).status_code == 401
    ok = await get("/admin/turns", headers={"Authorization": "Bearer admin-key-test-0123456789abcdef"})
    assert ok.status_code == 200 and ok.json() == []


# ============================================================ worker
async def test_end_to_end_reply_sent_exactly_once(db, http_mock):
    send = mock_send(http_mock)
    await post(payload(("PAGE1", [msg_event("u1", "m1", "Chào bạn")])))
    async with worker(FakeProvider(lambda m: "Chào bạn nha!")) as (deps, _):
        assert await run_once(deps)
        assert not await run_once(deps)  # không còn việc
        await post(payload(("PAGE1", [msg_event("u1", "m1", "Chào bạn")])))  # Meta gửi lại
        assert not await run_once(deps)
    assert send.call_count == 1
    sent = send.calls[0].request
    body = __import__("json").loads(sent.content)
    assert body["messaging_type"] == "RESPONSE" and body["recipient"]["id"] == "u1"
    assert "trợ lý AI" in body["message"]["text"]  # lượt đầu công bố là AI
    assert body["message"]["metadata"].endswith(":0")  # idempotency key để đối soát echo
    assert sent.headers["authorization"].startswith("Bearer ") and "access_token" not in str(sent.url)
    assert await scalar("SELECT status FROM messages WHERE role = 'assistant'") == "sent"
    assert await scalar("SELECT outcome FROM turns") == "reply"


async def test_same_conversation_is_claimed_by_one_worker_at_a_time(db):
    await post(payload(("PAGE1", [msg_event("u1", "m1", "a"), msg_event("u2", "m2", "b")])))
    c1 = await repo.claim_conversation("wA", 60)
    c2 = await repo.claim_conversation("wB", 60)
    c3 = await repo.claim_conversation("wC", 60)
    assert c1 and c2 and c1[0] != c2[0] and c3 is None  # 2 hội thoại song song, không ai lấy trùng
    await post(payload(("PAGE1", [msg_event("u1", "m3", "c")])))  # tin mới khi hội thoại đang bị giữ
    assert await repo.claim_conversation("wD", 60) is None


async def test_crash_mid_send_resumes_without_duplicate(db, http_mock):
    send = mock_send(http_mock)
    await post(payload(("PAGE1", [msg_event("u1", "m1", "hi")])))
    provider = FakeProvider(lambda m: "Chào nha")
    async with worker(provider, "w1") as (deps, _):
        conv_id, _att = await repo.claim_conversation("w1", 60)
        turn_id, is_new = await repo.start_or_resume_turn(conv_id)
        from app.conversation.graph import GraphContext, run_turn

        ctx = GraphContext(settings=deps.settings, store=deps.store, llm=provider, profile=deps.profile)
        result = await run_turn(deps.graph, turn_id=str(turn_id), conversation_id=str(conv_id), context=ctx)
        await repo.persist_turn_decision(conv_id, turn_id, result, result["reply_parts"])
        # "crash" giữa lúc gửi: chunk đã chuyển sang sending, không biết Meta đã nhận chưa
        async with session_scope() as s:
            await s.execute(text("UPDATE messages SET status = 'sending' WHERE role = 'assistant'"))
            await s.execute(text("UPDATE conversations SET lease_expires_at = now() - interval '1 second'"))
    async with worker(provider, "w2") as (deps2, _):
        assert await run_once(deps2)
    assert send.call_count == 0  # không gửi lại mù quáng
    assert len(provider.calls) == 1  # graph resume từ checkpoint, không gọi lại model
    assert await scalar("SELECT status FROM messages WHERE role = 'assistant'") == "uncertain"
    # echo từ Meta xác nhận tin đã tới -> đối soát thành sent
    key = await scalar("SELECT idempotency_key FROM messages WHERE role = 'assistant'")
    await post(payload(("PAGE1", [echo_event("u1", "e9", "Chào nha", app_id="APP123", metadata=key)])))
    assert await scalar("SELECT status FROM messages WHERE role = 'assistant'") == "sent"


async def test_handoff_blocks_reply_waiting_in_queue(db, http_mock):
    send = mock_send(http_mock)
    await post(payload(("PAGE1", [msg_event("u1", "m1", "kể chuyện đi")])))
    provider = FakeProvider(lambda m: "Ngày xửa ngày xưa...")
    async with worker(provider) as (deps, _):
        conv_id, _ = await repo.claim_conversation("w1", 60)
        turn_id, _ = await repo.start_or_resume_turn(conv_id)
        from app.conversation.graph import GraphContext, run_turn
        from app.workers.sender import send_turn

        ctx = GraphContext(settings=deps.settings, store=deps.store, llm=provider, profile=deps.profile)
        result = await run_turn(deps.graph, turn_id=str(turn_id), conversation_id=str(conv_id), context=ctx)
        await repo.persist_turn_decision(conv_id, turn_id, result, result["reply_parts"])
        await service.set_handoff(conv_id, True, "admin", "quản trị viên nhận xử lý")  # xảy ra trong lúc chờ gửi
        rep = await send_turn(deps.settings, deps.messenger, conv_id=conv_id, turn_id=turn_id, worker_id="w1",
                              mark_disclosed=True)
    assert rep.outcome == "cancelled_handoff" and send.call_count == 0
    assert await scalar("SELECT status FROM messages WHERE role = 'assistant'") == "cancelled"


async def test_stale_message_is_not_answered(db, http_mock):
    send = mock_send(http_mock)
    old = int((datetime.now(UTC) - timedelta(hours=2)).timestamp() * 1000)
    await post(payload(("PAGE1", [msg_event("u1", "m1", "còn ai không", ts=old)])))
    provider = FakeProvider()
    async with worker(provider) as (deps, _):
        assert await run_once(deps)
    assert send.call_count == 0 and provider.calls == []
    assert await scalar("SELECT outcome FROM turns") == "skipped_stale"


async def test_send_timeout_marks_uncertain_and_does_not_retry(db, http_mock):
    send = mock_send(http_mock, side_effect=httpx.ReadTimeout("timeout"))
    await post(payload(("PAGE1", [msg_event("u1", "m1", "hi")])))
    async with worker(FakeProvider(lambda m: "Chào")) as (deps, _):
        await run_once(deps)
    assert send.call_count == 1
    assert await scalar("SELECT status FROM messages WHERE role = 'assistant'") == "uncertain"
    assert "uncertain_delivery" in await scalar("SELECT outcome FROM turns")


async def test_rate_limited_send_is_retried_boundedly(db, http_mock, monkeypatch):
    monkeypatch.setattr("app.workers.sender.asyncio.sleep", _no_sleep)
    responses = iter([httpx.Response(400, json={"error": {"code": 613, "message": "rate"}}),
                      httpx.Response(200, json={"recipient_id": "u1", "message_id": "m_ok"})])
    send = mock_send(http_mock, side_effect=lambda req: next(responses))
    await post(payload(("PAGE1", [msg_event("u1", "m1", "hi")])))
    async with worker(FakeProvider(lambda m: "Chào")) as (deps, _):
        await run_once(deps)
    assert send.call_count == 2
    assert await scalar("SELECT status FROM messages WHERE role = 'assistant'") == "sent"


async def _no_sleep(*_a, **_k):  # type: ignore[no-untyped-def]
    return None


async def test_context_survives_worker_restart(db, http_mock):
    mock_send(http_mock)

    def bot(msgs):  # type: ignore[no-untyped-def]
        convo = " ".join(str(m.content) for m in msgs[1:])
        return "Bạn tên Lan nè." if "mình tên Lan" in convo and "tên mình là gì" in convo else "Ok nha."

    await post(payload(("PAGE1", [msg_event("u1", "m1", "mình tên Lan")])))
    async with worker(FakeProvider(bot), "w1") as (deps, _):
        await run_once(deps)
    # "khởi động lại": engine DB, graph, store, provider mới hoàn toàn
    from app.storage.db import dispose_db, init_db

    await dispose_db()
    init_db(get_settings())
    await post(payload(("PAGE1", [msg_event("u1", "m2", "tên mình là gì")])))
    provider2 = FakeProvider(bot)
    async with worker(provider2, "w2") as (deps2, _):
        await run_once(deps2)
    assert any("mình tên Lan" in str(m.content) for m in provider2.calls[-1])
    assert await scalar("SELECT text FROM messages WHERE role = 'assistant' ORDER BY id DESC LIMIT 1") == "Bạn tên Lan nè."


async def test_delete_conversation_removes_messages_summary_and_checkpoints(db, http_mock):
    mock_send(http_mock)
    await post(payload(("PAGE1", [msg_event("u1", "m1", "số của mình 0912345678")])))
    async with worker(FakeProvider(lambda m: "Ok")) as (deps, saver):
        await run_once(deps)
        conv_id = await scalar("SELECT id FROM conversations")
        turn_id = await scalar("SELECT id FROM turns")
        async with session_scope() as s:
            await s.execute(text("UPDATE conversations SET summary = 'tóm tắt riêng tư'"))
        cfg = {"configurable": {"thread_id": thread_id_for(str(turn_id))}}
        assert await saver.aget_tuple(cfg) is not None
        details = await service.delete_conversation(conv_id, "admin", saver, Tracer())
        assert await saver.aget_tuple(cfg) is None
    assert details["messages"] == 2 and details["checkpoint_threads"] == 1
    for table in ("conversations", "messages", "turns"):
        assert await scalar(f"SELECT count(*) FROM {table}") == 0  # noqa: S608
    audit = await scalar("SELECT details::text FROM audit_log WHERE action = 'delete_conversation'")
    assert "0912345678" not in audit and "tóm tắt" not in audit


# ============================================================ vòng phản hồi
async def test_feedback_review_export_pipeline(db, http_mock, tmp_path):
    mock_send(http_mock)
    await post(payload(("PAGE1", [msg_event("u1", "m1", "Gợi ý sách hay đi, sđt mình 0912345678")])))
    async with worker(FakeProvider(lambda m: "Bạn thử đọc Nhà Giả Kim nhé.")) as (deps, _):
        await run_once(deps)
    turn_id = await scalar("SELECT id FROM turns")
    good = await service.rate_turn(turn_id, "good", "admin", None)
    corr = await service.correct_turn(turn_id, "Bạn thử đọc 'Hoàng tử bé' nha, ngắn và ý nghĩa.", "admin", None)
    cands = await service.list_candidates()
    assert {c["id"] for c in cands} == {good["candidate_id"], corr["candidate_id"]}
    assert all("0912345678" not in str(c["sample"]) for c in cands)  # PII đã được che
    with pytest.raises(service.InvalidAction):  # chưa có mẫu approved
        await service.export_approved(reviewer="admin", out_root=tmp_path)
    with pytest.raises(service.InvalidAction):  # phải xác nhận quyền sử dụng
        await service.review_candidate(corr["candidate_id"], approve=True, reviewer="a", usage_rights_confirmed=False, notes=None)
    await service.review_candidate(corr["candidate_id"], approve=True, reviewer="a", usage_rights_confirmed=True, notes=None)
    await service.review_candidate(good["candidate_id"], approve=False, reviewer="a", usage_rights_confirmed=False, notes=None)
    sft = await service.export_approved(reviewer="admin", out_root=tmp_path)
    assert sft["records"] == 1
    rec = __import__("json").loads((tmp_path / sft["dataset_version"] / "data.jsonl").read_text().splitlines()[0])
    assert rec["messages"][-1] == {"role": "assistant", "content": "Bạn thử đọc 'Hoàng tử bé' nha, ngắn và ý nghĩa."}
    dpo = await service.export_approved(reviewer="admin", out_root=tmp_path, kind="dpo")
    pair = __import__("json").loads((tmp_path / dpo["dataset_version"] / "data.jsonl").read_text().splitlines()[0])
    assert pair["chosen"].startswith("Bạn thử đọc 'Hoàng tử bé'") and "Nhà Giả Kim" in pair["rejected"]
    assert await scalar("SELECT count(*) FROM dataset_exports") == 2


async def test_now_ms_helper_sanity():
    assert now_ms() > 1_700_000_000_000
