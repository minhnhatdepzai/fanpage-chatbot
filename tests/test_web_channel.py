"""Kênh chat website: phiên có chữ ký, lưu bền vững, worker xử lý như Messenger nhưng giao tại chỗ (không gọi Meta)."""

from __future__ import annotations

import httpx
import pytest

from app.config import get_settings
from app.conversation.prompts import WEB_HANDOFF_REPLY
from app.main import app
from app.observability.tracing import Tracer
from app.providers.llm import FakeProvider
from app.web.channel import new_session, sign, verify
from tests.test_db_flows import run_once, scalar, worker


def test_session_signature():
    secret = "s" * 32
    sid, tok = new_session(secret)
    assert verify(secret, sid, tok)
    assert not verify(secret, sid, sign("other-secret-000000000000", sid))
    assert not verify(secret, "w_" + "x" * 20, tok)
    assert not verify("", sid, tok)


@pytest.fixture(autouse=True)
def _tracer():
    app.state.tracer = Tracer()


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.mark.db
async def test_web_message_roundtrip_without_messenger(db, http_mock):
    async with client() as c:
        sess = (await c.post("/web/session")).json()
        assert (await c.post("/web/messages", json={**sess, "token": "sai", "text": "hi"})).status_code == 401
        assert (await c.post("/web/messages", json={**sess, "text": "x" * 5000})).status_code == 400
        r = await c.post("/web/messages", json={**sess, "text": "chào bạn"})
        assert r.status_code == 202
        assert r.json()["queue"]["phase"] == "queued"
        assert r.json()["queue"]["position"] == 1
        q = {"session_id": sess["session_id"], "token": sess["token"]}
        waiting = (await c.get("/web/messages", params=q)).json()
        assert waiting["messages"] == [] and waiting["pending"] is True
        assert waiting["queue"]["phase"] == "queued" and waiting["queue"]["position"] == 1
        async with worker(FakeProvider(lambda m: "Chào bạn nha!")) as (deps, _):
            assert await run_once(deps)
        body = (await c.get("/web/messages", params=q)).json()
        assert body["pending"] is False and len(body["messages"]) == 1
        assert body["queue"]["phase"] == "idle" and body["queue"]["position"] is None
        assert "trợ lý AI" in body["messages"][0]["text"] and "Chào bạn nha!" in body["messages"][0]["text"]
        assert body["messages"][0]["mode"] == "chat"
        after = {**q, "after": body["messages"][0]["id"]}
        assert (await c.get("/web/messages", params=after)).json()["messages"] == []
        other = (await c.post("/web/session")).json()  # phiên khác không thấy tin của phiên này
        assert (await c.get("/web/messages", params={k: other[k] for k in ("session_id", "token")})).json()[
            "messages"
        ] == []
    assert not http_mock.calls  # không có request nào ra ngoài (Meta)
    assert await scalar("SELECT page_id FROM conversations") == "web"


@pytest.mark.db
async def test_public_queue_reports_each_sessions_position_without_other_user_data(db):
    async with client() as c:
        first = (await c.post("/web/session")).json()
        second = (await c.post("/web/session")).json()
        one = (await c.post("/web/messages", json={**first, "text": "câu hỏi thứ nhất"})).json()
        two = (await c.post("/web/messages", json={**second, "text": "câu hỏi thứ hai"})).json()
        assert one["queue"]["phase"] == "queued" and one["queue"]["position"] == 1
        assert two["queue"]["phase"] == "queued" and two["queue"]["position"] == 2
        assert two["queue"]["total"] == 2
        assert set(two["queue"]) == {"phase", "position", "total", "active_jobs", "worker_slots"}


@pytest.mark.db
async def test_web_handoff_points_to_messenger_and_keeps_bot_on(db):
    async with client() as c:
        sess = (await c.post("/web/session")).json()
        await c.post("/web/messages", json={**sess, "text": "cho mình gặp quản trị viên"})
        async with worker(FakeProvider(lambda m: "không được gọi")) as (deps, _):
            assert await run_once(deps)
        msgs = (await c.get("/web/messages", params={k: sess[k] for k in ("session_id", "token")})).json()[
            "messages"
        ]
    assert WEB_HANDOFF_REPLY in msgs[0]["text"]
    assert await scalar("SELECT handoff_active FROM conversations") is False


@pytest.mark.db
async def test_web_rate_limit_per_session(db, monkeypatch):
    from app.api import web as web_api

    monkeypatch.setattr(web_api, "_IP_HITS", {})
    s = get_settings()
    async with client() as c:
        sess = (await c.post("/web/session")).json()
        codes = [
            (await c.post("/web/messages", json={**sess, "text": f"tin {i}"})).status_code
            for i in range(s.web_rate_limit_per_5min + 1)
        ]
    assert codes[:-1] == [202] * s.web_rate_limit_per_5min and codes[-1] == 429


async def test_widget_assets_served():
    async with client() as c:
        js = await c.get("/web/widget.js")
        assert js.status_code == 200 and "attachShadow" in js.text and "innerHTML" not in js.text
        assert "/web/widget.js" in (await c.get("/web/demo")).text


async def test_math_and_literature_practice_pages_serve_grade_1_to_12_bank():
    async with client() as c:
        math = await c.get("/web/math")
        literature = await c.get("/web/literature")
        bank = await c.get("/web/math/study-bank.js")
    assert math.status_code == literature.status_code == bank.status_code == 200
    assert 'id="grade-strip"' in math.text and 'data-grade="12"' in math.text
    assert 'id="literature-submission"' in literature.text
    assert 'id="long-writing"' in literature.text and "2.500+ chữ" in literature.text
    script = await _read_static_math_js()
    assert "practice-explain" in script
    assert "requestLongLiteratureAnswer" in script and "tối thiểu 2500 chữ" in script
    assert "grade: 1" in bank.text and "grade: 12" in bank.text
    assert bank.text.count("grade:") >= 48


async def _read_static_math_js() -> str:
    async with client() as c:
        return (await c.get("/web/math/app.js")).text


@pytest.mark.db
async def test_long_literature_submission_uses_chatbot_writing_revision_flow(db):
    async with client() as c:
        session = (await c.post("/web/session")).json()
        response = await c.post(
            "/web/literature/submit",
            json={
                **session,
                "grade": 9,
                "prompt": "Viết bài nghị luận về lòng biết ơn trong đời sống hôm nay.",
                "answer": "Em cho rằng lòng biết ơn cần thể hiện bằng hành động. " * 45,
            },
        )
        assert response.status_code == 202
        assert response.json()["mode"] == "writing_revision"
        async with worker(
            FakeProvider(lambda _: "Bài có luận điểm rõ; cần thêm phản đề và sửa liên kết.")
        ) as (deps, _):
            assert await run_once(deps)
        messages = (
            await c.get(
                "/web/messages",
                params={key: session[key] for key in ("session_id", "token")},
            )
        ).json()["messages"]
    assert messages[0]["mode"] == "writing"
    assert "cần thêm phản đề" in messages[0]["text"]
    saved = await scalar("SELECT text FROM messages WHERE role = 'user'")
    assert "Chấm, nhận xét và sửa bài văn lớp 9" in saved
    assert "BÀI LÀM CỦA HỌC SINH" in saved and len(saved) > 1000
