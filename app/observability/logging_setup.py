"""Logging JSON một dòng/bản ghi, luôn che bí mật trước khi ghi."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import UTC, datetime
from typing import Any

from app.observability.redaction import redact_secrets, register_secrets

_STANDARD_ATTRS = set(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None).__dict__.keys()) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            # Chỉ giữ loại và thông điệp lỗi + traceback đã che bí mật
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["exc"] = "".join(traceback.format_exception(*record.exc_info))[-4000:]
        line = json.dumps(payload, ensure_ascii=False, default=str)
        return redact_secrets(line)


class _DropQueryString(logging.Filter):
    """Phòng hờ: bỏ query string khỏi access log (GET webhook chứa hub.verify_token)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args and isinstance(record.args, tuple) and len(record.args) >= 3:
            args = list(record.args)
            if isinstance(args[2], str) and "?" in args[2]:
                args[2] = args[2].split("?", 1)[0]
                record.args = tuple(args)
        return True


def configure_logging(level: str = "INFO", secret_values: list[str] | None = None) -> None:
    if secret_values:
        register_secrets(secret_values)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # Thư viện HTTP có thể log URL đầy đủ -> chỉ cho WARNING trở lên
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "langfuse", "opentelemetry"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True
    logging.getLogger("uvicorn.access").addFilter(_DropQueryString())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
