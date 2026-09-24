"""Tiện ích dựng payload webhook Messenger cho test."""

from __future__ import annotations

import json
import time
from typing import Any

from app.messenger.signature import compute_signature

PAGE = "PAGE1"
SECRET = "test-app-secret-0123456789"


def now_ms() -> int:
    return int(time.time() * 1000)


def msg_event(psid: str, mid: str, text: str | None = None, *, page: str = PAGE, ts: int | None = None,
              attachments: list[dict] | None = None, sticker_id: int | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"mid": mid}
    if text is not None:
        message["text"] = text
    if attachments:
        message["attachments"] = attachments
    if sticker_id:
        message["sticker_id"] = sticker_id
    return {"sender": {"id": psid}, "recipient": {"id": page}, "timestamp": ts or now_ms(), "message": message}


def echo_event(psid: str, mid: str, text: str, *, app_id: str | None, metadata: str | None = None,
               page: str = PAGE) -> dict[str, Any]:
    m: dict[str, Any] = {"mid": mid, "text": text, "is_echo": True}
    if app_id:
        m["app_id"] = int(app_id) if app_id.isdigit() else app_id
    if metadata:
        m["metadata"] = metadata
    return {"sender": {"id": page}, "recipient": {"id": psid}, "timestamp": now_ms(), "message": m}


def payload(*entries: tuple[str, list[dict]]) -> dict[str, Any]:
    return {"object": "page", "entry": [{"id": pid, "time": now_ms(), "messaging": evs} for pid, evs in entries]}


def signed(body_obj: dict[str, Any], secret: str = SECRET, *, ascii_only: bool = False) -> tuple[bytes, dict[str, str]]:
    body = json.dumps(body_obj, ensure_ascii=ascii_only).encode("utf-8")
    return body, {"X-Hub-Signature-256": compute_signature(secret, body), "Content-Type": "application/json"}
