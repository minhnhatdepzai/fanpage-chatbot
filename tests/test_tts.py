"""TTS neural: danh sách giọng đóng, prosody có giới hạn và API không lưu nội dung."""

from __future__ import annotations

import httpx
import pytest

from app.main import app
from app.tts.service import TTSServiceError, _text_chunks, _without_id3_prefix, resolve_prosody


def test_tts_prosody_combines_preset_and_user_adjustments_safely():
    assert resolve_prosody("podcast", 8, 2, -3) == ("+0%", "+0Hz", "-3%")
    assert resolve_prosody("clear", -30, -20, 20) == ("-50%", "-20Hz", "+22%")
    with pytest.raises(TTSServiceError):
        resolve_prosody("unknown", 0, 0, 0)


def test_long_tts_text_is_split_on_sentence_boundaries():
    chunks = _text_chunks(("Một câu đủ rõ. " * 500).strip(), limit=240)
    assert len(chunks) > 2 and all(len(chunk) <= 240 for chunk in chunks)
    assert " ".join(chunks).count("Một câu đủ rõ.") == 500


def test_mp3_join_strips_only_followup_id3_header():
    header = b"ID3\x04\x00\x00\x00\x00\x00\x03abc"
    frames = b"\xff\xfbMP3"
    assert _without_id3_prefix(header + frames) == frames
    assert _without_id3_prefix(frames) == frames


async def test_tts_voices_and_authenticated_synthesis(monkeypatch):
    from app.api import tts as api

    captured: dict[str, object] = {}

    async def fake_synthesize(text: str, **kwargs):  # type: ignore[no-untyped-def]
        captured.update({"text": text, **kwargs})
        return b"\xff\xfbFAKE-MP3"

    monkeypatch.setattr(api, "synthesize_speech", fake_synthesize)
    monkeypatch.setattr(api, "_rate_limit_ip", lambda *args, **kwargs: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        voice_response = await client.get("/web/tts/voices")
        session = (await client.post("/web/session")).json()
        payload = {
            **session,
            "text": "Hello. This is a pronunciation lesson.",
            "voice": "en-GB-SoniaNeural",
            "style": "podcast",
            "rate": -3,
            "pitch": 2,
            "volume": 0,
        }
        unauthorized = await client.post("/web/tts/synthesize", json={**payload, "token": "wrong"})
        response = await client.post("/web/tts/synthesize", json=payload)

    body = voice_response.json()
    assert voice_response.status_code == 200 and body["enabled"] is True
    assert {voice["locale"] for voice in body["voices"]} == {"vi-VN", "en-US", "en-GB"}
    assert len(body["voices"]) == 6 and len(body["styles"]) == 4
    assert unauthorized.status_code == 401
    assert response.status_code == 200 and response.headers["content-type"] == "audio/mpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content.startswith(b"\xff\xfb")
    assert captured["text"] == payload["text"] and captured["voice"] == "en-GB-SoniaNeural"


async def test_tts_rejects_unknown_voice(monkeypatch):
    from app.api import tts as api

    monkeypatch.setattr(api, "_rate_limit_ip", lambda *args, **kwargs: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        session = (await client.post("/web/session")).json()
        response = await client.post(
            "/web/tts/synthesize",
            json={**session, "text": "hello", "voice": "not-a-real-voice", "style": "teacher"},
        )
    assert response.status_code == 503
    assert "chưa được hỗ trợ" in response.json()["detail"]
