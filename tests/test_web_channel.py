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
        q = {"session_id": sess["session_id"], "token": sess["token"]}
        assert (await c.get("/web/messages", params=q)).json() == {"messages": [], "pending": True}
        async with worker(FakeProvider(lambda m: "Chào bạn nha!")) as (deps, _):
            assert await run_once(deps)
        body = (await c.get("/web/messages", params=q)).json()
        assert body["pending"] is False and len(body["messages"]) == 1
        assert "trợ lý AI" in body["messages"][0]["text"] and "Chào bạn nha!" in body["messages"][0]["text"]
        after = {**q, "after": body["messages"][0]["id"]}
        assert (await c.get("/web/messages", params=after)).json()["messages"] == []
        other = (await c.post("/web/session")).json()  # phiên khác không thấy tin của phiên này
        assert (await c.get("/web/messages", params={k: other[k] for k in ("session_id", "token")})).json()[
            "messages"
        ] == []
    assert not http_mock.calls  # không có request nào ra ngoài (Meta)
    assert await scalar("SELECT page_id FROM conversations") == "web"


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
