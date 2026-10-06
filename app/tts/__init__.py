"""Neural text-to-speech for the web learning rooms."""

from app.tts.service import TTS_STYLES, TTS_VOICES, TTSServiceError, synthesize_speech

__all__ = ["TTS_STYLES", "TTS_VOICES", "TTSServiceError", "synthesize_speech"]
