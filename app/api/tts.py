"""API TTS cho phòng học Văn và Tiếng Anh."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.api.web import _check_session, _enabled, _rate_limit_ip
from app.config import get_settings
from app.tts import TTS_STYLES, TTS_VOICES, TTSServiceError, synthesize_speech

log = logging.getLogger("app.tts")
router = APIRouter(prefix="/web/tts", tags=["text-to-speech"])


class TTSIn(BaseModel):
    session_id: str = Field(max_length=64)
    token: str = Field(max_length=64)
    text: str = Field(min_length=1, max_length=24_000)
    voice: str = Field(max_length=64)
    style: Literal["teacher", "podcast", "story", "clear"] = "podcast"
    rate: int = Field(default=0, ge=-30, le=30)
    pitch: int = Field(default=0, ge=-20, le=20)
    volume: int = Field(default=0, ge=-30, le=20)


@router.get("/voices")
async def voices() -> dict[str, object]:
    _enabled()
    settings = get_settings()
    return {
        "enabled": settings.tts_enabled,
        "provider": "edge-neural",
        "voices": [
            {
                "id": voice.id,
                "label": voice.label,
                "language": voice.language,
                "locale": voice.locale,
                "accent": voice.accent,
                "gender": voice.gender,
                "sample": voice.sample,
            }
            for voice in TTS_VOICES
        ],
        "styles": [{"id": key, **value} for key, value in TTS_STYLES.items()],
    }


@router.post("/synthesize")
async def synthesize(body: TTSIn, request: Request) -> Response:
    _enabled()
    settings = get_settings()
    if not settings.tts_enabled:
        raise HTTPException(503, "Giọng đọc neural đang tắt.")
    _check_session(body.session_id, body.token)
    if len(body.text) > settings.tts_max_chars:
        raise HTTPException(413, f"Bài đọc tối đa {settings.tts_max_chars} ký tự.")
    _rate_limit_ip(request, "tts", 2 * settings.web_rate_limit_per_5min)
    started = time.perf_counter()
    try:
        audio = await asyncio.wait_for(
            synthesize_speech(
                body.text,
                voice=body.voice,
                style=body.style,
                rate=body.rate,
                pitch=body.pitch,
                volume=body.volume,
                max_bytes=settings.tts_max_audio_mb * 1_000_000,
            ),
            timeout=settings.tts_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(504, "Tạo bài đọc quá thời gian; hãy thử đoạn ngắn hơn.") from exc
    except TTSServiceError as exc:
        raise HTTPException(503, str(exc)) from exc
    log.info(
        "tts_completed",
        extra={
            "voice": body.voice,
            "style": body.style,
            "characters": len(body.text),
            "audio_bytes": len(audio),
            "ms": int((time.perf_counter() - started) * 1000),
        },
    )
    return Response(
        audio,
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-store",
            "X-TTS-Provider": "edge-neural",
            "X-TTS-Voice": body.voice,
            "X-TTS-Characters": str(len(body.text)),
        },
    )
