"""Kiểm tra cấu hình Meta bằng Graph API mà không in bất kỳ bí mật nào.

Trả lời các câu hỏi: token trong .env là loại gì (PAGE/USER/APP)? thuộc app nào, Page nào? có quyền
``pages_messaging`` không? App Secret có đúng không? app đã cấu hình webhook (callback, fields) chưa?
Page đã subscribe app chưa (cần ``pages_manage_metadata`` để đọc qua API)?
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from app.config import Settings
from app.observability.redaction import redact_secrets

REQUIRED_WEBHOOK_FIELDS = {"messages", "messaging_postbacks"}
RECOMMENDED_WEBHOOK_FIELDS = {"message_echoes", "message_reactions"}


def _ts(v: Any) -> str:
    if isinstance(v, int) and v > 0:
        return datetime.fromtimestamp(v, UTC).isoformat()
    return "không hết hạn" if v == 0 else str(v)


async def _get(
    client: httpx.AsyncClient, url: str, params: dict[str, Any], bearer: str | None = None
) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {bearer}"} if bearer else {}
    try:
        r = await client.get(url, params=params, headers=headers)
        body = r.json() if r.content else {}
    except (httpx.HTTPError, ValueError) as exc:
        return 0, {"error": {"message": type(exc).__name__}}
    if isinstance(body, dict) and "error" in body:
        e = body["error"]
        body = {
            "error": {
                k: (redact_secrets(str(e.get(k))) if k == "message" else e.get(k))
                for k in ("message", "type", "code", "error_subcode")
            }
        }
    return r.status_code, body


async def check_meta(settings: Settings) -> dict[str, Any]:
    base = f"{settings.meta_graph_base_url}/{settings.meta_graph_api_version}"
    page_token = settings.meta_page_access_token.get_secret_value()
    app_secret = settings.meta_app_secret.get_secret_value()
    report: dict[str, Any] = {
        "graph_api_version": settings.meta_graph_api_version,
        "findings": [],
        "missing": [],
    }
    if not page_token:
        report["missing"].append("META_PAGE_ACCESS_TOKEN")
        return report
    async with httpx.AsyncClient(timeout=15) as client:
        # 1) loại token + quyền (debug_token; dùng app token nếu có App Secret, không thì chính token)
        app_token = f"{settings.meta_app_id}|{app_secret}" if (app_secret and settings.meta_app_id) else None
        status, dbg = await _get(
            client,
            f"{base}/debug_token",
            {"input_token": page_token, "access_token": app_token or page_token},
        )
        data = dbg.get("data", {}) if isinstance(dbg, dict) else {}
        if app_token and "error" in dbg:
            report["app_secret_status"] = (
                f"KHÔNG hợp lệ hoặc không khớp META_APP_ID ({dbg['error'].get('message')})"
            )
            status, dbg = await _get(
                client, f"{base}/debug_token", {"input_token": page_token, "access_token": page_token}
            )
            data = dbg.get("data", {})
        elif app_token:
            report["app_secret_status"] = "hợp lệ (app access token được Graph API chấp nhận)"  # noqa: S105
        else:
            report["app_secret_status"] = (
                "chưa cấu hình -> webhook POST sẽ bị từ chối (không xác minh được chữ ký)"  # noqa: S105
            )
            report["missing"].append("META_APP_SECRET")
        report["token"] = {
            "type": data.get("type"),
            "is_valid": data.get("is_valid"),
            "app_id": data.get("app_id"),
            "application": data.get("application"),
            "page_id": data.get("profile_id"),
            "scopes": data.get("scopes"),
            "expires_at": _ts(data.get("expires_at")),
            "data_access_expires_at": _ts(data.get("data_access_expires_at")),
            "error": dbg.get("error") if "error" in dbg else None,
        }
        tok = report["token"]
        if tok["type"] != "PAGE":
            report["findings"].append(
                "Token KHÔNG phải Page Access Token -> không gửi Messenger thay Page được."
            )
        if "pages_messaging" not in (tok["scopes"] or []):
            report["findings"].append("Thiếu quyền pages_messaging -> không gửi được tin nhắn.")
        if settings.meta_page_id and tok["page_id"] and tok["page_id"] != settings.meta_page_id:
            report["findings"].append("META_PAGE_ID khác Page của token.")
        if settings.meta_app_id and tok["app_id"] and tok["app_id"] != settings.meta_app_id:
            report["findings"].append("META_APP_ID khác app của token.")
        report["findings"].append(
            "Token Page chỉ dùng cho Messenger; KHÔNG dùng để gọi mô hình AI (cần cấu hình LLM riêng)."
        )
        # 2) webhook của app (cần app token)
        if app_token and "hợp lệ" in report["app_secret_status"] and settings.meta_app_id:
            s2, subs = await _get(
                client, f"{base}/{settings.meta_app_id}/subscriptions", {"access_token": app_token}
            )
            pages = [x for x in subs.get("data", []) if x.get("object") == "page"] if s2 == 200 else []
            if pages:
                fields = {f.get("name") for f in pages[0].get("fields", []) if isinstance(f, dict)}
                report["app_webhook"] = {
                    "callback_url": pages[0].get("callback_url"),
                    "active": pages[0].get("active"),
                    "fields": sorted(fields),
                    "missing_required_fields": sorted(REQUIRED_WEBHOOK_FIELDS - fields),
                    "missing_recommended_fields": sorted(RECOMMENDED_WEBHOOK_FIELDS - fields),
                }
            else:
                report["app_webhook"] = {
                    "configured": False,
                    "detail": subs.get("error") or "chưa có subscription object=page",
                }
        # 3) Page đã subscribe app chưa (cần pages_manage_metadata)
        s3, sub_apps = await _get(client, f"{base}/me/subscribed_apps", {}, bearer=page_token)
        if s3 == 200:
            report["page_subscribed_apps"] = [
                {"id": a.get("id"), "name": a.get("name"), "subscribed_fields": a.get("subscribed_fields")}
                for a in sub_apps.get("data", [])
            ]
        else:
            report["page_subscribed_apps"] = (
                "không đọc được bằng token hiện tại (cần pages_manage_metadata) - kiểm tra trong App Dashboard"
            )
    return report


async def set_webhook(settings: Settings, callback_url: str) -> dict[str, Any]:
    """Trỏ webhook (object=page) của app tới ``callback_url`` + đăng ký Page với app. Không in bí mật.

    Meta gọi GET xác minh tới ``callback_url`` ngay khi đặt, nên API phải đang chạy và truy cập được qua HTTPS.
    Bước Page -> app (``subscribed_apps``) cần quyền ``pages_manage_metadata``; nếu token thiếu quyền thì làm
    trong App Dashboard (Messenger -> Settings -> Webhooks -> Add Subscriptions).
    """
    base = f"{settings.meta_graph_base_url}/{settings.meta_graph_api_version}"
    app_secret = settings.meta_app_secret.get_secret_value()
    if not (settings.meta_app_id and app_secret and settings.meta_verify_token.get_secret_value()):
        return {"ok": False, "detail": "thiếu META_APP_ID / META_APP_SECRET / META_VERIFY_TOKEN"}
    fields = ",".join(sorted(REQUIRED_WEBHOOK_FIELDS | RECOMMENDED_WEBHOOK_FIELDS))
    out: dict[str, Any] = {"callback_url": callback_url, "fields": fields}

    def _summ(r: httpx.Response) -> Any:
        try:
            body = r.json()
        except ValueError:
            return {"status": r.status_code}
        if isinstance(body, dict) and "error" in body:
            e = body["error"]
            return {
                "status": r.status_code,
                "error": redact_secrets(str(e.get("message"))),
                "code": e.get("code"),
            }
        return {"status": r.status_code, "result": body}

    async with httpx.AsyncClient(timeout=30) as client:
        r1 = await client.post(
            f"{base}/{settings.meta_app_id}/subscriptions",
            data={
                "object": "page",
                "callback_url": callback_url,
                "verify_token": settings.meta_verify_token.get_secret_value(),
                "fields": fields,
                "access_token": f"{settings.meta_app_id}|{app_secret}",
            },
        )
        out["app_webhook"] = _summ(r1)
        r2 = await client.post(
            f"{base}/{settings.meta_page_id or 'me'}/subscribed_apps",
            data={"subscribed_fields": fields},
            headers={"Authorization": f"Bearer {settings.meta_page_access_token.get_secret_value()}"},
        )
        out["page_subscription"] = _summ(r2)
    out["ok"] = r1.status_code == 200
    return out
