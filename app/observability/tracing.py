"""Tracing qua Langfuse (SDK v4, OpenTelemetry). Tích hợp có thể tắt hoàn toàn.

Bảo vệ dữ liệu:
- ``user_id`` = user_ref (HMAC), ``session_id`` = conversation UUID nội bộ -> không có PSID.
- Mặc định CHỈ gửi metadata (độ dài, cờ kiểm tra, phiên bản, token, latency). Bật
  ``LANGFUSE_CAPTURE_CONTENT=true`` để gửi nội dung, khi đó nội dung luôn qua ``redact_all``.
- Mọi lỗi Langfuse bị nuốt và ghi cảnh báo: chatbot không phụ thuộc Langfuse.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from app.config import Settings
from app.observability.redaction import redact_all

log = logging.getLogger(__name__)
_warned: set[str] = set()


def _warn_once(key: str, exc: BaseException) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning("langfuse_error", extra={"op": key, "error": type(exc).__name__})


def _mask(*, data: Any, **_: Any) -> Any:
    """Hook ``mask`` của Langfuse: che bí mật + PII trong mọi chuỗi (phòng thủ nhiều lớp)."""
    if isinstance(data, str):
        return redact_all(data)
    if isinstance(data, dict):
        return {k: _mask(data=v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return [_mask(data=v) for v in data]
    return data


class Observation:
    """Bọc một observation Langfuse; mọi method đều an toàn khi Langfuse tắt/lỗi."""

    def __init__(self, obs: Any = None, capture_content: bool = False) -> None:
        self._obs = obs
        self._capture = capture_content

    def content(self, value: Any) -> Any:
        return _mask(data=value) if self._capture else None

    def child(self, name: str, as_type: str = "span", **kwargs: Any) -> Observation:
        if self._obs is None:
            return Observation()
        try:
            return Observation(
                self._obs.start_observation(name=name, as_type=as_type, **kwargs), self._capture
            )
        except Exception as exc:  # noqa: BLE001
            _warn_once("child", exc)
            return Observation()

    def update(self, **kwargs: Any) -> None:
        if self._obs is None:
            return
        try:
            self._obs.update(**kwargs)
        except Exception as exc:  # noqa: BLE001
            _warn_once("update", exc)

    def end(self, **kwargs: Any) -> None:
        if self._obs is None:
            return
        try:
            if kwargs:
                self._obs.update(**kwargs)
            self._obs.end()
        except Exception as exc:  # noqa: BLE001
            _warn_once("end", exc)


class Tracer:
    """Mặc định no-op. ``LangfuseTracer`` ghi đè khi được cấu hình."""

    enabled = False
    capture_content = False

    def trace_id_for(self, turn_id: str) -> str | None:
        return None

    @contextlib.contextmanager
    def turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        user_id: str,
        version: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_text: str | None,
    ) -> Iterator[Observation]:
        yield Observation()

    def score(
        self,
        *,
        trace_id: str | None,
        name: str,
        value: float | str,
        data_type: str,
        comment: str | None = None,
    ) -> None:
        return None

    def delete_traces(self, trace_ids: list[str]) -> bool:
        return False

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class LangfuseTracer(Tracer):
    enabled = True

    def __init__(self, settings: Settings) -> None:
        from langfuse import Langfuse

        self.capture_content = settings.langfuse_capture_content
        self._client = Langfuse(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            tracing_enabled=True,
            sample_rate=settings.langfuse_sample_rate,
            environment=settings.app_env.value,
            mask=_mask,
            flush_interval=5,
            timeout=5,
        )

    def trace_id_for(self, turn_id: str) -> str | None:
        try:
            from langfuse import Langfuse

            return Langfuse.create_trace_id(seed=turn_id)
        except Exception as exc:  # noqa: BLE001
            _warn_once("trace_id", exc)
            return None

    @contextlib.contextmanager
    def turn(
        self,
        *,
        turn_id: str,
        session_id: str,
        user_id: str,
        version: str,
        tags: list[str],
        metadata: dict[str, Any],
        input_text: str | None,
    ) -> Iterator[Observation]:
        from langfuse import propagate_attributes

        root: Observation = Observation()
        stack = contextlib.ExitStack()
        try:
            stack.enter_context(
                propagate_attributes(
                    user_id=user_id,
                    session_id=session_id,
                    version=version,
                    tags=tags,
                    trace_name="messenger-turn",
                    metadata={k: str(v) for k, v in metadata.items()},
                )
            )
            trace_id = self.trace_id_for(turn_id)
            obs = self._client.start_observation(
                trace_context={"trace_id": trace_id} if trace_id else None,
                name="messenger-turn",
                as_type="chain",
                input=_mask(data=input_text) if self.capture_content else None,
                metadata=metadata,
                version=version,
            )
            root = Observation(obs, self.capture_content)
        except Exception as exc:  # noqa: BLE001
            _warn_once("turn_start", exc)
        try:
            yield root
        finally:
            root.end()
            with contextlib.suppress(Exception):
                stack.close()

    def score(
        self,
        *,
        trace_id: str | None,
        name: str,
        value: float | str,
        data_type: str,
        comment: str | None = None,
    ) -> None:
        if not trace_id:
            return
        try:
            self._client.create_score(
                trace_id=trace_id, name=name, value=value, data_type=data_type, comment=comment
            )
        except Exception as exc:  # noqa: BLE001
            _warn_once("score", exc)

    def delete_traces(self, trace_ids: list[str]) -> bool:
        ids = [t for t in trace_ids if t]
        if not ids:
            return True
        try:
            self._client.api.trace.delete_multiple(trace_ids=ids)
            return True
        except Exception as exc:  # noqa: BLE001
            _warn_once("delete", exc)
            return False

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:  # noqa: BLE001
            _warn_once("flush", exc)

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception as exc:  # noqa: BLE001
            _warn_once("shutdown", exc)


def build_tracer(settings: Settings) -> Tracer:
    if not settings.langfuse_enabled:
        return Tracer()
    if not (
        settings.langfuse_public_key.get_secret_value() and settings.langfuse_secret_key.get_secret_value()
    ):
        log.warning("langfuse_enabled_but_missing_keys -> tracing disabled")
        return Tracer()
    try:
        return LangfuseTracer(settings)
    except Exception as exc:  # noqa: BLE001
        _warn_once("init", exc)
        return Tracer()
