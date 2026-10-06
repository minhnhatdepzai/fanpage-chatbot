"""TTS neural có danh sách giọng đóng và giới hạn tài nguyên chặt.

Không lưu văn bản hoặc MP3. ``edge-tts`` gọi dịch vụ đọc trực tuyến của Microsoft Edge;
UI có Web Speech fallback khi dịch vụ này không khả dụng.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any


class TTSServiceError(RuntimeError):
    """TTS không tạo được âm thanh hợp lệ."""


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    id: str
    label: str
    language: str
    locale: str
    accent: str
    gender: str
    sample: str


TTS_VOICES = (
    VoiceProfile(
        "vi-VN-HoaiMyNeural",
        "Hoài My — Việt Nam",
        "vi",
        "vi-VN",
        "Việt Nam",
        "female",
        "Chào bạn, hôm nay chúng ta sẽ cùng tìm hiểu bài học này thật rõ ràng nhé.",
    ),
    VoiceProfile(
        "vi-VN-NamMinhNeural",
        "Nam Minh — Việt Nam",
        "vi",
        "vi-VN",
        "Việt Nam",
        "male",
        "Chào bạn, mình sẽ giải thích từng ý theo một trình tự thật dễ theo dõi.",
    ),
    VoiceProfile(
        "en-US-JennyNeural",
        "Jenny — American English",
        "en",
        "en-US",
        "Mỹ",
        "female",
        "Hello! Let's explore this lesson one clear step at a time.",
    ),
    VoiceProfile(
        "en-US-GuyNeural",
        "Guy — American English",
        "en",
        "en-US",
        "Mỹ",
        "male",
        "Hello! Let's explore this lesson one clear step at a time.",
    ),
    VoiceProfile(
        "en-GB-SoniaNeural",
        "Sonia — British English",
        "en",
        "en-GB",
        "Anh",
        "female",
        "Hello! Let's explore this lesson one clear step at a time.",
    ),
    VoiceProfile(
        "en-GB-RyanNeural",
        "Ryan — British English",
        "en",
        "en-GB",
        "Anh",
        "male",
        "Hello! Let's explore this lesson one clear step at a time.",
    ),
)

TTS_STYLES: dict[str, dict[str, Any]] = {
    "teacher": {"label": "Giáo viên", "rate": -5, "pitch": 0, "volume": 0},
    "podcast": {"label": "Podcast", "rate": -8, "pitch": -2, "volume": 0},
    "story": {"label": "Kể chuyện", "rate": -12, "pitch": 3, "volume": 1},
    "clear": {"label": "Chậm & rõ", "rate": -20, "pitch": 0, "volume": 2},
}

_VOICE_IDS = {voice.id for voice in TTS_VOICES}
_TTS_CHUNK_CHARS = 1200


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def resolve_prosody(style: str, rate: int, pitch: int, volume: int) -> tuple[str, str, str]:
    preset = TTS_STYLES.get(style)
    if preset is None:
        raise TTSServiceError("Kiểu đọc chưa được hỗ trợ.")
    final_rate = _clamp(int(preset["rate"]) + rate, -50, 50)
    final_pitch = _clamp(int(preset["pitch"]) + pitch, -50, 50)
    final_volume = _clamp(int(preset["volume"]) + volume, -50, 30)
    return f"{final_rate:+d}%", f"{final_pitch:+d}Hz", f"{final_volume:+d}%"


def _text_chunks(text: str, limit: int = _TTS_CHUNK_CHARS) -> list[str]:
    """Chia ở biên câu/từ để TTS dài không mắc kẹt trong một request streaming khổng lồ."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        pieces = []
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit + 1)
            cut = cut if cut > limit // 2 else limit
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > limit:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def _without_id3_prefix(audio: bytes) -> bytes:
    """Bỏ ID3v2 ở đầu mảnh sau trước khi nối MP3; frame âm thanh được giữ nguyên."""
    if len(audio) < 10 or audio[:3] != b"ID3":
        return audio
    size = sum((audio[6 + index] & 0x7F) << (21 - index * 7) for index in range(4))
    end = min(len(audio), 10 + size)
    return audio[end:]


async def synthesize_speech(
    text: str,
    *,
    voice: str,
    style: str,
    rate: int = 0,
    pitch: int = 0,
    volume: int = 0,
    max_bytes: int = 12_000_000,
) -> bytes:
    """Tạo MP3 trong bộ nhớ, không ghi tệp và không chấp nhận voice tùy ý."""
    if voice not in _VOICE_IDS:
        raise TTSServiceError("Giọng đọc chưa được hỗ trợ.")
    clean = " ".join(text.split())
    if not clean:
        raise TTSServiceError("Không có nội dung để đọc.")
    rate_value, pitch_value, volume_value = resolve_prosody(style, rate, pitch, volume)
    try:
        import edge_tts

        semaphore = asyncio.Semaphore(8)

        async def synthesize_chunk(part: str) -> bytes:
            async with semaphore:
                communicate = edge_tts.Communicate(
                    part,
                    voice,
                    rate=rate_value,
                    pitch=pitch_value,
                    volume=volume_value,
                )
                chunk_audio = bytearray()
                async for chunk in communicate.stream():
                    if chunk.get("type") != "audio":
                        continue
                    chunk_audio.extend(chunk.get("data") or b"")
                    if len(chunk_audio) > max_bytes:
                        raise TTSServiceError("Bài đọc quá dài để tạo âm thanh an toàn.")
                if not chunk_audio:
                    raise TTSServiceError("Dịch vụ chưa trả về dữ liệu âm thanh.")
                return bytes(chunk_audio)

        rendered = await asyncio.gather(*(synthesize_chunk(part) for part in _text_chunks(clean)))
        audio = bytearray(rendered[0])
        for chunk_audio in rendered[1:]:
            audio.extend(_without_id3_prefix(chunk_audio))
        if len(audio) > max_bytes:
            raise TTSServiceError("Bài đọc quá dài để tạo âm thanh an toàn.")
    except TTSServiceError:
        raise
    except Exception as exc:  # mạng/provider không được làm hỏng API chính
        raise TTSServiceError("Dịch vụ giọng đọc đang tạm thời không phản hồi.") from exc
    return bytes(audio)
