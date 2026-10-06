"""FastAPI app: webhook Messenger, health/ready, admin API.

uv run botctl api        # hoặc: uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from app.api import admin, health, math_tutor, tts, web, webhook
from app.config import get_settings
from app.observability.logging_setup import configure_logging
from app.observability.tracing import build_tracer
from app.storage.db import dispose_db, init_db

log = logging.getLogger("app.http")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    configure_logging(s.log_level, s.secret_values())
    init_db(s)
    app.state.tracer = build_tracer(s)
    log.info("api_started", extra={"env": s.app_env.value, "graph_api": s.meta_graph_api_version})
    yield
    app.state.tracer.shutdown()
    await dispose_db()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title="fanpage-chatbot",
        lifespan=lifespan,
        docs_url="/docs" if s.app_env.value != "production" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if s.app_env.value != "production" else None,
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        response = await call_next(request)
        # KHÔNG ghi query string (GET /webhook chứa hub.verify_token) và không ghi body
        log.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": int((time.perf_counter() - start) * 1000),
            },
        )
        return response

    app.include_router(health.router)
    app.include_router(webhook.router)
    app.include_router(admin.router)
    app.include_router(math_tutor.router)
    app.include_router(math_tutor.literature_router)
    app.include_router(math_tutor.pages_router)
    app.include_router(tts.router)
    app.include_router(web.router)
    app.mount(
        "/task1-coach-pages",
        StaticFiles(directory=math_tutor.STATIC / "task1-coach-pages", html=True),
        name="task1-coach-pages",
    )
    if s.web_allowed_origins:
        from fastapi.middleware.cors import CORSMiddleware

        # chỉ các tên miền được liệt kê mới gọi được API từ trình duyệt; không gửi cookie
        app.add_middleware(
            CORSMiddleware,
            allow_origins=s.web_allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            allow_credentials=False,
        )
    return app


app = create_app()
