"""Lớp model provider (qua LangChain). Worker chỉ biết interface ``ChatProvider``.

- ``openai_compatible``: model server local của dự án (Transformers + PEFT, xem serving/),
  hoặc bất kỳ endpoint tương thích OpenAI Chat Completions (vLLM, Ollama, OpenAI...).
- ``anthropic``: tùy chọn, cần cài extra ``anthropic`` và đặt LLM_MODEL rõ ràng.
- ``fake``: dùng trong test/đánh giá hệ thống (không gọi mạng).

Lưu ý: adapter PEFT (LoRA/VeRA) chỉ gắn được vào model open-weight chạy local; không gắn được vào
API model đóng nếu nhà cung cấp không hỗ trợ.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import BaseMessage

from app.config import LLMProvider, Settings

log = logging.getLogger(__name__)


class ProviderError(Exception):
    """kind: timeout | rate_limit | unavailable | auth | bad_response"""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail

    @property
    def retryable(self) -> bool:
        return self.kind in {"timeout", "unavailable"}


@dataclass(slots=True)
class LLMResult:
    text: str
    provider: str
    model: str
    adapter: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int


class ChatProvider(Protocol):
    name: str

    async def generate(
        self, messages: list[BaseMessage], *, max_tokens: int, temperature: float, timeout: float
    ) -> LLMResult: ...


def _map_exception(exc: BaseException) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    name = type(exc).__name__
    if isinstance(exc, asyncio.TimeoutError | TimeoutError) or "Timeout" in name:
        return ProviderError("timeout", name)
    if "RateLimit" in name or getattr(exc, "status_code", None) == 429:
        return ProviderError("rate_limit", name)
    if (
        "Authentication" in name
        or "PermissionDenied" in name
        or getattr(exc, "status_code", None) in (401, 403)
    ):
        return ProviderError("auth", name)
    status = getattr(exc, "status_code", None)
    if "Connection" in name or (isinstance(status, int) and status >= 500):
        return ProviderError("unavailable", name)
    return ProviderError("bad_response", name)


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        settings: Settings,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        name: str | None = None,
    ) -> None:
        from langchain_openai import ChatOpenAI

        self._settings = settings
        self._model = model or settings.llm_model
        if name:
            self.name = name
        self._llm = ChatOpenAI(
            model=self._model,
            base_url=base_url or settings.llm_base_url,
            api_key=api_key
            or settings.llm_api_key.get_secret_value()
            or settings.model_server_api_key.get_secret_value()
            or "none",
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # retry do lớp này kiểm soát (có giới hạn tổng thời gian)
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            use_responses_api=False,
        )

    async def generate(
        self, messages: list[BaseMessage], *, max_tokens: int, temperature: float, timeout: float
    ) -> LLMResult:
        start = time.perf_counter()
        try:
            msg = await asyncio.wait_for(
                self._llm.ainvoke(messages, max_tokens=max_tokens, temperature=temperature), timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001
            raise _map_exception(exc) from exc
        latency = int((time.perf_counter() - start) * 1000)
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        usage = getattr(msg, "usage_metadata", None) or {}
        meta = getattr(msg, "response_metadata", None) or {}
        fingerprint = meta.get("system_fingerprint")
        return LLMResult(
            text=text,
            provider=self.name,
            model=str(meta.get("model_name") or self._model),
            adapter=fingerprint if fingerprint and fingerprint != "base" else None,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            latency_ms=latency,
        )


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - phụ thuộc extra
            raise RuntimeError("Cần cài extra: uv sync --extra anthropic") from exc
        if not settings.anthropic_api_key.get_secret_value():
            raise RuntimeError("Thiếu ANTHROPIC_API_KEY")
        self._settings = settings
        self._llm = ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

    async def generate(
        self, messages: list[BaseMessage], *, max_tokens: int, temperature: float, timeout: float
    ) -> LLMResult:
        start = time.perf_counter()
        try:
            msg = await asyncio.wait_for(
                self._llm.ainvoke(messages, max_tokens=max_tokens, temperature=temperature), timeout=timeout
            )
        except Exception as exc:  # noqa: BLE001
            raise _map_exception(exc) from exc
        usage = getattr(msg, "usage_metadata", None) or {}
        text = (
            msg.content
            if isinstance(msg.content, str)
            else "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
        )
        return LLMResult(
            text=text,
            provider=self.name,
            model=self._settings.llm_model,
            adapter=None,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


class FallbackProvider:
    """Gọi provider chính; nếu KHÔNG kết nối được / hết thời gian (unavailable, timeout) thì dùng provider dự phòng.

    Dùng cho: model server (bf16 + adapter) là chính, Ollama Q4 là dự phòng khi model server khởi động lại/lỗi.
    Không chuyển khi lỗi rate_limit/auth/bad_response (lỗi đó cần xử lý, không che đi).
    """

    def __init__(self, primary: ChatProvider, fallback: ChatProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+fallback"

    async def generate(
        self, messages: list[BaseMessage], *, max_tokens: int, temperature: float, timeout: float
    ) -> LLMResult:
        try:
            return await self.primary.generate(
                messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
            )
        except ProviderError as err:
            if err.kind not in {"unavailable", "timeout"}:
                raise
            log.warning("llm_fallback", extra={"kind": err.kind, "fallback": self.fallback.name})
        res = await self.fallback.generate(
            messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout
        )
        res.provider = f"fallback:{self.fallback.name}"
        return res


Responder = Callable[[list[BaseMessage]], str | Awaitable[str]]


class FakeProvider:
    """Provider giả lập cho test: trả lời theo hàm ``responder`` hoặc ném lỗi theo kịch bản."""

    name = "fake"

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        errors: list[str] | None = None,
        delay_seconds: float = 0.0,
        model: str = "fake-model",
    ) -> None:
        self.responder = responder or (lambda msgs: "Mình đây, bạn cần gì nè?")
        self.errors = list(errors or [])
        self.delay_seconds = delay_seconds
        self.model = model
        self.calls: list[list[BaseMessage]] = []

    async def generate(
        self, messages: list[BaseMessage], *, max_tokens: int, temperature: float, timeout: float
    ) -> LLMResult:
        self.calls.append(list(messages))
        if self.errors:
            raise ProviderError(self.errors.pop(0), "scripted")
        if self.delay_seconds:
            try:
                await asyncio.wait_for(asyncio.sleep(self.delay_seconds), timeout=timeout)
            except TimeoutError as exc:
                raise ProviderError("timeout", "scripted delay") from exc
        out = self.responder(messages)
        if asyncio.iscoroutine(out) or isinstance(out, Awaitable):
            out = await out  # type: ignore[misc]
        return LLMResult(str(out), self.name, self.model, None, 10, 10, 1)


async def generate_with_retry(
    provider: ChatProvider,
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    temperature: float,
    timeout: float,
    max_retries: int,
    total_deadline: float | None = None,
) -> LLMResult:
    """Retry có giới hạn cho lỗi tạm thời (timeout/unavailable). Không retry rate_limit/auth."""
    deadline = time.monotonic() + (total_deadline or timeout * (max_retries + 1))
    attempt = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderError("timeout", "total deadline exceeded")
        try:
            return await provider.generate(
                messages, max_tokens=max_tokens, temperature=temperature, timeout=min(timeout, remaining)
            )
        except ProviderError as err:
            attempt += 1
            if not err.retryable or attempt > max_retries:
                raise
            backoff = min(2.0 * attempt, max(0.0, deadline - time.monotonic() - 1))
            log.warning("llm_retry", extra={"kind": err.kind, "attempt": attempt})
            await asyncio.sleep(backoff)


def build_provider(settings: Settings) -> ChatProvider:
    if settings.llm_provider == LLMProvider.openai_compatible:
        primary = OpenAICompatibleProvider(settings)
        if settings.llm_fallback_base_url and settings.llm_fallback_model:
            # model dự phòng (vd. Ollama) không nhận khóa của model server
            backup = OpenAICompatibleProvider(
                settings,
                base_url=settings.llm_fallback_base_url,
                model=settings.llm_fallback_model,
                api_key="none",
                name="ollama",
            )
            return FallbackProvider(primary, backup)
        return primary
    if settings.llm_provider == LLMProvider.anthropic:
        return AnthropicProvider(settings)
    return FakeProvider()
