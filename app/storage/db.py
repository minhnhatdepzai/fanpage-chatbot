"""Engine/session SQLAlchemy async (driver psycopg 3)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.sqlalchemy_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_pool_size,
        pool_pre_ping=True,
        pool_timeout=settings.db_connect_timeout_seconds,
        connect_args={"connect_timeout": int(max(1, settings.db_connect_timeout_seconds))},
    )


def init_db(settings: Settings) -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        _engine = create_engine(settings)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    if _sessionmaker is None:
        raise RuntimeError("init_db() chưa được gọi")
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Một transaction: commit khi thành công, rollback khi lỗi."""
    async with get_sessionmaker()() as session:
        async with session.begin():
            yield session


async def dispose_db() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None


async def ping(timeout_seconds: float = 3.0) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(text(f"SET LOCAL statement_timeout = {int(timeout_seconds * 1000)}"))
        await session.execute(text("SELECT 1"))
