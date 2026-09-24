"""Client Send API (Graph API) với phân loại lỗi để quyết định retry an toàn.

- Token gửi qua header ``Authorization: Bearer`` (không đặt trong URL -> không lọt vào access log).
- Lỗi kết nối TRƯỚC khi gửi request -> an toàn để thử lại.
- Timeout/ngắt kết nối SAU khi đã gửi -> ``uncertain`` (không biết Meta đã nhận chưa) -> không retry mù.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from app.config import Settings
from app.observability.redaction import redact_secrets

log = logging.getLogger(__name__)

RATE_LIMIT_CODES = {4, 17, 32, 613}
TEMPORARY_CODES = {1200}
AUTH_CODES = {190}
WINDOW_CLOSED = {(10, 2018278)}


class SendOutcome(StrEnum):
    sent = "sent"
    retryable = "retryable"  # chắc chắn chưa gửi hoặc Meta báo lỗi tạm thời
    rate_limited = "rate_limited"
    uncertain = "uncertain"  # có thể đã gửi
    window_closed = "window_closed"
    auth_error = "auth_error"  # token hỏng/hết hạn -> cần người vận hành
    permanent = "permanent"


@dataclass(slots=True)
class SendResult:
    outcome: SendOutcome
    message_id: str | None = None
    http_status: int | None = None
    error_code: int | None = None
    error_subcode: int | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.outcome == SendOutcome.sent


def classify_graph_error(http_status: int, body: Any) -> SendResult:
    err = body.get("error") if isinstance(body, dict) else None
    if not isinstance(err, dict):
        if http_status == 429:
            return SendResult(SendOutcome.rate_limited, http_status=http_status, detail="HTTP 429")
        if http_status >= 500:
            return SendResult(SendOutcome.uncertain, http_status=http_status, detail=f"HTTP {http_status}")
        return SendResult(SendOutcome.permanent, http_status=http_status, detail=f"HTTP {http_status}")
    code = err.get("code")
    sub = err.get("error_subcode")
    detail = redact_secrets(str(err.get("message", ""))[:300])
    res = SendResult(
        SendOutcome.permanent, http_status=http_status, error_code=code, error_subcode=sub, detail=detail
    )
    if (code, sub) in WINDOW_CLOSED:
        res.outcome = SendOutcome.window_closed
    elif code in AUTH_CODES:
        res.outcome = SendOutcome.auth_error
    elif code in RATE_LIMIT_CODES or http_status == 429:
        res.outcome = SendOutcome.rate_limited
    elif code in TEMPORARY_CODES or err.get("is_transient") is True:
        res.outcome = SendOutcome.retryable
    elif http_status >= 500:
        res.outcome = SendOutcome.uncertain
    return res


class MessengerClient:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        timeout = httpx.Timeout(settings.messenger_send_timeout_seconds, connect=5.0, pool=5.0, write=10.0)
        self._http = http or httpx.AsyncClient(timeout=timeout)
        self._owns_http = http is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._settings.meta_page_access_token.get_secret_value()}"}

    async def _post(self, body: dict[str, Any]) -> SendResult:
        url = self._settings.send_api_url
        try:
            resp = await self._http.post(url, json=body, headers=self._headers())
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            return SendResult(SendOutcome.retryable, detail=f"not_sent:{type(exc).__name__}")
        except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError, httpx.ReadError) as exc:
            return SendResult(SendOutcome.uncertain, detail=f"maybe_sent:{type(exc).__name__}")
        except httpx.HTTPError as exc:
            return SendResult(SendOutcome.uncertain, detail=f"maybe_sent:{type(exc).__name__}")
        try:
            data = resp.json()
        except ValueError:
            data = None
        if resp.status_code == 200 and isinstance(data, dict) and "error" not in data:
            return SendResult(SendOutcome.sent, message_id=data.get("message_id"), http_status=200)
        return classify_graph_error(resp.status_code, data)

    async def send_text(self, psid: str, text: str, metadata: str | None = None) -> SendResult:
        message: dict[str, Any] = {"text": text}
        if metadata:
            # Meta trả lại chuỗi này trong webhook message_echoes -> đối soát trạng thái "uncertain"
            message["metadata"] = metadata[:1000]
        body = {"recipient": {"id": psid}, "messaging_type": "RESPONSE", "message": message}
        return await self._post(body)

    async def sender_action(self, psid: str, action: str) -> bool:
        """typing_on / typing_off / mark_seen - best effort, lỗi không ảnh hưởng luồng chính."""
        res = await self._post({"recipient": {"id": psid}, "sender_action": action})
        if not res.ok:
            log.debug("sender_action_failed", extra={"action": action, "outcome": res.outcome})
        return res.ok
