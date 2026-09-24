"""Ảnh end-to-end (không cần GPU): webhook -> tải ảnh an toàn -> phân tích (giả lập) -> graph -> trả lời trung thực."""

from __future__ import annotations

import base64
import io
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from PIL import Image

from app.config import get_settings
from app.conversation.graph import IMAGE_ONLY_USER_TEXT, GraphContext, build_graph, run_turn
from app.conversation.knowledge import KnowledgeBase
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.profile import FanpageProfile
from app.messenger.events import UserMessage, parse_webhook_payload
from app.providers.llm import FakeProvider
from app.vision.context import NOTHING_RECOGNIZED_REPLY
from app.vision.fetch import ImageFetchError, fetch_image, host_allowed

CDN = "https://scontent.xx.fbcdn.net/v/t1/abc.jpg"
HOSTS = ["fbcdn.net", "fbsbx.com"]


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="JPEG")
    return buf.getvalue()


# ------------------------------------------------------------------------------------------ webhook
def test_webhook_image_urls_are_captured_and_stickers_ignored():
    payload = {
        "object": "page",
        "entry": [
            {
                "id": "PAGE1",
                "time": 1,
                "messaging": [
                    {
                        "sender": {"id": "u1"},
                        "recipient": {"id": "PAGE1"},
                        "timestamp": 1,
                        "message": {
                            "mid": "m1",
                            "attachments": [
                                {"type": "image", "payload": {"url": CDN}},
                                {
                                    "type": "image",
                                    "payload": {"url": CDN + "?s", "sticker_id": 369239263222822},
                                },
                                {"type": "video", "payload": {"url": "https://video.xx.fbcdn.net/v.mp4"}},
                            ],
                        },
                    }
                ],
            }
        ],
    }
    ev = parse_webhook_payload(payload, {"PAGE1"})[0]
    assert isinstance(ev, UserMessage) and ev.images == [{"url": CDN}]
    assert ev.attachment_types == ["image", "image", "video"]


# ------------------------------------------------------------------------------------------ tải ảnh
def test_host_allowlist():
    assert host_allowed(CDN, HOSTS) and host_allowed("https://lookaside.fbsbx.com/x", HOSTS)
    assert not host_allowed("http://scontent.xx.fbcdn.net/x", HOSTS)  # không phải https
    assert not host_allowed("https://fbcdn.net.evil.com/x", HOSTS)
    assert not host_allowed("https://169.254.169.254/latest", HOSTS)


async def test_fetch_image_ok_and_rejections(http_mock):
    http_mock.get(CDN).respond(200, content=_jpeg(), headers={"content-type": "image/jpeg"})
    assert (await fetch_image(CDN, allowed_suffixes=HOSTS, max_bytes=10**6)).startswith(b"\xff\xd8")
    http_mock.get(CDN.replace("abc", "big")).respond(
        200, content=b"x" * 5000, headers={"content-type": "image/jpeg"}
    )
    with pytest.raises(ImageFetchError, match="dung lượng"):
        await fetch_image(CDN.replace("abc", "big"), allowed_suffixes=HOSTS, max_bytes=1000)
    http_mock.get(CDN.replace("abc", "html")).respond(
        200, content=b"<html>", headers={"content-type": "text/html"}
    )
    with pytest.raises(ImageFetchError, match="không phải ảnh"):
        await fetch_image(CDN.replace("abc", "html"), allowed_suffixes=HOSTS, max_bytes=1000)
    http_mock.get(CDN.replace("abc", "redir")).respond(
        302, headers={"location": "https://internal.example.com/secret"}
    )
    with pytest.raises(ImageFetchError, match="danh sách cho phép"):
        await fetch_image(CDN.replace("abc", "redir"), allowed_suffixes=HOSTS, max_bytes=1000)
    with pytest.raises(ImageFetchError, match="danh sách cho phép"):
        await fetch_image("https://example.com/a.jpg", allowed_suffixes=HOSTS, max_bytes=1000)


# ------------------------------------------------------------------------------------------ graph
class FakeVision:
    def __init__(self, analysis=None, fail: bool = False):  # type: ignore[no-untyped-def]
        self.analysis = analysis
        self.fail = fail
        self.calls = 0

    async def analyze(self, data: bytes):  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.fail:
            raise RuntimeError("model server down")
        return self.analysis


HOMEWORK = {
    "ocr": {"text": "Câu 1: Tính 125 + 378. Liên hệ 0912345678", "lines": 2, "mean_conf": 0.9},
    "objects": [{"label_vi": "sách", "conf": 0.8, "model": "coco", "box": [0, 0, 1, 1]}],
}


async def _turn(vision, text, images, responder, http_mock):  # type: ignore[no-untyped-def]
    http_mock.get(CDN).respond(200, content=_jpeg(), headers={"content-type": "image/jpeg"})
    clock = [datetime(2026, 9, 24, 10, 0, tzinfo=UTC)]
    store = InMemoryConversationStore(now=lambda: clock[0])
    provider = FakeProvider(responder)
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(name="AI Test"),
        knowledge=KnowledgeBase(),
        vision=vision,
        now=lambda: clock[0],
    )
    cid = store.ensure_conversation("PAGE1", "u1")
    store.add_user_message(
        cid, text, attachment_types=["image"], images=images, ts=clock[0] - timedelta(seconds=1)
    )
    tid = store.begin_turn(cid)
    return await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx), provider


async def test_image_with_text_goes_to_llm_with_redacted_ocr(http_mock):
    r, p = await _turn(
        FakeVision(HOMEWORK), "giải giúp mình", [{"url": CDN}], lambda m: "125 + 378 = 503.", http_mock
    )
    system = str(p.calls[-1][0].content)
    assert "Ảnh 1:" in system and "Tính 125 + 378" in system and "0912345678" not in system
    assert "sách (0.80)" in system and r["image_meta"]["images"] == 1
    assert "503" in "\n".join(r["reply_parts"])


async def test_image_only_message_uses_placeholder_question(http_mock):
    r, p = await _turn(
        FakeVision(HOMEWORK), None, [{"url": CDN}], lambda m: "Ảnh có bài toán cộng.", http_mock
    )
    human = str(p.calls[-1][-1].content)
    assert human.startswith(IMAGE_ONLY_USER_TEXT) and "[Ảnh MỚI gửi kèm tin nhắn này" in human
    assert r["action"] == "reply"


async def test_nothing_recognized_does_not_call_llm(http_mock):
    empty = {"ocr": {"text": "", "lines": 0}, "objects": []}
    r, p = await _turn(
        FakeVision(empty), "tổng số chấm là bao nhiêu", [{"url": CDN}], lambda m: "21", http_mock
    )
    assert p.calls == [] and NOTHING_RECOGNIZED_REPLY in "\n".join(r["reply_parts"])


async def test_vision_failure_falls_back_to_honest_notice(http_mock):
    r, p = await _turn(FakeVision(fail=True), None, [{"url": CDN}], lambda m: "x", http_mock)
    assert p.calls == [] and "chưa xem được hình ảnh" in "\n".join(r["reply_parts"])
    assert "image_analysis_failed" in r["check_flags"]


async def test_preanalyzed_web_image_is_used_without_fetch(http_mock):
    vision = FakeVision(HOMEWORK)
    r, p = await _turn(vision, "đây là gì", [{"analysis": HOMEWORK}], lambda m: "Bài toán cộng.", http_mock)
    assert vision.calls == 0 and "Ảnh 1:" in str(p.calls[-1][0].content)


# ------------------------------------------------------------------------------------------ API
@pytest.mark.db
async def test_web_image_upload_stores_sanitized_analysis(db, monkeypatch):
    from app.main import app
    from app.observability.tracing import Tracer
    from app.vision.client import VisionClient
    from tests.test_db_flows import scalar

    async def fake_analyze(self, data):  # type: ignore[no-untyped-def]
        assert data.startswith(b"\xff\xd8")
        return HOMEWORK

    monkeypatch.setattr(VisionClient, "analyze", fake_analyze)
    app.state.tracer = Tracer()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/web/config")).json()["images"] is False  # test tắt VISION_ENABLED
        monkeypatch.setattr(get_settings(), "vision_enabled", True)
        sess = (await c.post("/web/session")).json()
        img = base64.b64encode(_jpeg()).decode()
        assert (
            await c.post("/web/images", json={**sess, "token": "x", "image_base64": img})
        ).status_code == 401
        r = await c.post("/web/images", json={**sess, "image_base64": img, "caption": "giải giúp"})
        assert r.status_code == 202
    stored = await scalar("SELECT payload::text FROM messages WHERE role = 'user'")
    assert "Tính 125 + 378" in stored and "0912345678" not in stored and '"box"' not in stored


def test_model_server_vision_endpoints():
    from fastapi.testclient import TestClient

    from serving.server import create_app

    class FakeRuntime:
        base_model_id, revision, adapter_id, adapter_method = "m", "r", None, None

        def vram_gb(self) -> float:
            return 0.0

    class V:
        custom_id = "edu"

        def analyze(self, img):  # type: ignore[no-untyped-def]
            return {"size": list(img.size)}

        def ocr(self, img):  # type: ignore[no-untyped-def]
            return {"text": "abc"}

    key = get_settings().model_server_api_key.get_secret_value()
    h = {"Authorization": f"Bearer {key}"}
    c = TestClient(create_app(FakeRuntime(), "m", None, V()))  # type: ignore[arg-type]
    img = base64.b64encode(_jpeg()).decode()
    assert c.post("/v1/vision/analyze", json={"image_base64": img}, headers=h).json() == {"size": [32, 32]}
    assert c.post("/v1/vision/ocr", json={"image_base64": img}, headers=h).json()["text"] == "abc"
    assert c.post("/v1/vision/analyze", json={"image_base64": "!!notbase64!!"}, headers=h).status_code == 400
    assert c.post("/v1/vision/analyze", json={"image_base64": img}).status_code == 401
    no_vision = TestClient(create_app(FakeRuntime(), "m", None, None))  # type: ignore[arg-type]
    assert no_vision.post("/v1/vision/analyze", json={"image_base64": img}, headers=h).status_code == 503


# ------------------------------------------------------------------------------------------ hỏi tiếp, ca lỗi
async def test_follow_up_question_about_image_has_image_context(http_mock):
    http_mock.get(CDN).respond(200, content=_jpeg(), headers={"content-type": "image/jpeg"})
    clock = [datetime(2026, 9, 24, 10, 0, tzinfo=UTC)]
    store = InMemoryConversationStore(now=lambda: clock[0])
    provider = FakeProvider(lambda m: "Câu 1 là phép cộng 125 + 378.")
    vision = FakeVision(HOMEWORK)
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(name="AI Test"),
        knowledge=KnowledgeBase(),
        vision=vision,
        now=lambda: clock[0],
    )
    graph = build_graph()
    cid = store.ensure_conversation("PAGE1", "u1")
    for text, images in [("bài này là gì", [{"url": CDN}]), ("vậy kết quả là bao nhiêu?", None)]:
        store.add_user_message(
            cid,
            text,
            attachment_types=["image"] if images else None,
            images=images,
            ts=clock[0] - timedelta(seconds=1),
        )
        tid = store.begin_turn(cid)
        store.apply_result(cid, tid, await run_turn(graph, turn_id=tid, conversation_id=cid, context=ctx))
        clock[0] += timedelta(seconds=30)
    history = "\n".join(str(m.content) for m in provider.calls[-1][1:])
    assert "[Ảnh đã gửi" in history and "Tính 125 + 378" in history and "0912345678" not in history
    assert vision.calls == 1  # không phân tích lại ảnh ở lượt hỏi tiếp
    assert store.msgs[0].payload["images"] == [{"analysis": sanitize_for_test(HOMEWORK)}]  # URL đã được thay


def sanitize_for_test(a):  # type: ignore[no-untyped-def]
    from app.vision.context import sanitize_analysis

    return sanitize_analysis(a)


def test_many_objects_summary_is_capped_and_scope_is_stated():
    from app.vision.context import image_context_block

    objs = [{"label_vi": f"vật {i}", "conf": 0.9 - i / 100} for i in range(30)]
    block = image_context_block([{"ocr": {"text": ""}, "objects": objs}])
    assert "vật 11" in block and "vật 12" not in block  # tối đa 12 loại
    assert "CHỈ biết các loại đã học" in block and "không có trong danh sách KHÔNG có nghĩa" in block


@pytest.mark.db
async def test_web_image_errors_are_reported_honestly(db, monkeypatch):
    from app.main import app
    from app.observability.tracing import Tracer
    from app.vision.client import VisionClient, VisionError

    async def bad(self, data):  # type: ignore[no-untyped-def]
        raise VisionError("HTTP 400", status=400)

    monkeypatch.setattr(VisionClient, "analyze", bad)
    monkeypatch.setattr(get_settings(), "vision_enabled", True)
    app.state.tracer = Tracer()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        sess = (await c.post("/web/session")).json()
        r = await c.post(
            "/web/images", json={**sess, "image_base64": base64.b64encode(b"not an image!!").decode()}
        )
        assert r.status_code == 400 and "không đọc được" in r.json()["detail"]
        big = "A" * (get_settings().vision_max_image_mb * 1_400_000)
        assert (await c.post("/web/images", json={**sess, "image_base64": big})).status_code == 413


@pytest.mark.db
async def test_messenger_image_end_to_end_persists_analysis_and_answers(db, http_mock):
    from app.storage import repository as repo
    from tests.helpers import payload, signed
    from tests.test_db_flows import SEND_URL, post, scalar, worker

    http_mock.get(CDN).respond(200, content=_jpeg(), headers={"content-type": "image/jpeg"})
    send = http_mock.post(SEND_URL).respond(200, json={"recipient_id": "u1", "message_id": "m_out"})
    ev = {
        "sender": {"id": "u1"},
        "recipient": {"id": "PAGE1"},
        "timestamp": int(datetime.now(UTC).timestamp() * 1000),
        "message": {
            "mid": "m_img",
            "text": "giải giúp mình",
            "attachments": [{"type": "image", "payload": {"url": CDN}}],
        },
    }
    assert (await post(payload(("PAGE1", [ev])))).status_code == 200
    async with worker(FakeProvider(lambda m: "125 + 378 = 503.")) as (deps, _):
        deps.vision = FakeVision(HOMEWORK)
        claimed = await repo.claim_conversation(deps.worker_id, 60)
        from app.workers.processor import process_conversation

        await process_conversation(deps, *claimed)
    assert (
        send.call_count == 1
        and "503" in __import__("json").loads(send.calls[0].request.content)["message"]["text"]
    )
    stored = await scalar("SELECT payload::text FROM messages WHERE mid = 'm_img'")
    assert "Tính 125 + 378" in stored and "fbcdn" not in stored and "0912345678" not in stored
    del signed


async def test_capability_statement_only_when_vision_available(http_mock):
    from app.conversation.prompts import VISION_CAPABILITY

    _, p = await _turn(FakeVision(HOMEWORK), "bạn đọc được hình không", [], lambda m: "ok", http_mock)
    assert VISION_CAPABILITY.strip() in str(p.calls[-1][0].content)
    _, p2 = await _turn(None, "bạn đọc được hình không", [], lambda m: "ok", http_mock)
    assert VISION_CAPABILITY.strip() not in str(p2.calls[-1][0].content)


def test_sticker_only_message_is_typed_as_sticker_not_image():
    from app.messenger.events import UserMessage, parse_webhook_payload

    msg = {
        "mid": "m2",
        "sticker_id": 1234,
        "attachments": [{"type": "image", "payload": {"url": CDN, "sticker_id": 1234}}],
    }
    ev = parse_webhook_payload(
        {
            "object": "page",
            "entry": [
                {
                    "id": "PAGE1",
                    "time": 1,
                    "messaging": [
                        {"sender": {"id": "u1"}, "recipient": {"id": "PAGE1"}, "timestamp": 1, "message": msg}
                    ],
                }
            ],
        },
        {"PAGE1"},
    )[0]
    assert isinstance(ev, UserMessage) and ev.attachment_types == ["sticker"] and ev.images == []
