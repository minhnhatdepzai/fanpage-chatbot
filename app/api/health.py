"""/health (tiến trình sống) và /ready (đủ dependency bắt buộc để nhận webhook)."""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import get_settings
from app.config.settings import PROJECT_ROOT
from app.storage.db import get_sessionmaker

log = logging.getLogger(__name__)
router = APIRouter()


@lru_cache(maxsize=1)
def alembic_head() -> str | None:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    return ScriptDirectory.from_config(cfg).get_current_head()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def _db_checks() -> dict[str, Any]:
    out: dict[str, Any] = {}
    async with get_sessionmaker()() as session:
        await session.execute(text("SET statement_timeout = 2000"))
        await session.execute(text("SELECT 1"))
        out["database"] = "ok"
        version = (
            await session.execute(text("SELECT version_num FROM alembic_version"))
        ).scalar_one_or_none()
        head = alembic_head()
        out["migrations"] = "ok" if version == head else f"pending (db={version}, head={head})"
        has_ckpt = (
            await session.execute(text("SELECT to_regclass('public.checkpoint_migrations') IS NOT NULL"))
        ).scalar_one()
        out["langgraph_checkpointer"] = "ok" if has_ckpt else "missing (chạy: botctl db migrate)"
        hb = (
            await session.execute(
                text("SELECT max(last_seen) > now() - interval '60 seconds' FROM worker_heartbeats")
            )
        ).scalar_one()
        out["worker_heartbeat"] = "ok" if hb else "no recent heartbeat (thông tin, không chặn readiness)"
    return out


@router.get("/ready")
async def ready() -> JSONResponse:
    s = get_settings()
    checks: dict[str, Any] = {}
    try:
        checks.update(await asyncio.wait_for(_db_checks(), timeout=3.0))
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {type(exc).__name__}"
    checks["meta_app_secret"] = "ok" if s.meta_app_secret.get_secret_value() else "missing"
    checks["meta_verify_token"] = "ok" if s.meta_verify_token.get_secret_value() else "missing"
    checks["meta_page_id"] = "ok" if s.meta_page_id else "missing"
    required = [
        "database",
        "migrations",
        "langgraph_checkpointer",
        "meta_app_secret",
        "meta_verify_token",
        "meta_page_id",
    ]
    ok = all(checks.get(k) == "ok" for k in required)
    checks["meta_page_access_token"] = (
        "configured" if s.meta_page_access_token.get_secret_value() else "missing"
    )
    checks["send_mode"] = str(s.messenger_send_mode)
    return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)
