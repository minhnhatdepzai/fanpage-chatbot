"""Cấu hình test: KHÔNG đọc .env thật, dùng database riêng ``chatbot_test``, mọi gọi Graph API đều mock.

Test cần PostgreSQL (đánh dấu ``@pytest.mark.db``) tự bỏ qua nếu không kết nối được:
    docker compose up -d postgres
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ---- phải chạy TRƯỚC khi import bất kỳ module app nào
os.environ["BOT_ENV_FILE"] = ""
ROOT = Path(__file__).resolve().parents[1]


def _test_database_url() -> str:
    if os.environ.get("TEST_DATABASE_URL"):
        return os.environ["TEST_DATABASE_URL"]
    env = ROOT / ".env"
    if env.exists():  # chỉ đọc DATABASE_URL để suy ra DB test trên cùng server
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("DATABASE_URL="):
                return re.sub(r"/[^/?]+(\?|$)", r"/chatbot_test\1", line.split("=", 1)[1].strip(), count=1)
    return "postgresql+psycopg://chatbot:chatbot@127.0.0.1:55432/chatbot_test"


TEST_ENV = {
    "APP_ENV": "test",
    "META_APP_ID": "APP123",
    "META_APP_SECRET": "test-app-secret-0123456789",
    "META_PAGE_ID": "PAGE1",
    "META_PAGE_ACCESS_TOKEN": "EAATESTTOKEN00000000000000000000000000",
    "META_VERIFY_TOKEN": "verify-token-test-123",
    "ADMIN_API_KEY": "admin-key-test-0123456789abcdef",
    "PSEUDONYM_SECRET": "pseudonym-secret-test-0123456789",
    "MODEL_SERVER_API_KEY": "model-key-test-123456",
    "LLM_PROVIDER": "fake",
    "MESSENGER_SEND_MODE": "all",
    "MESSENGER_ALLOWED_PSIDS": "",
    "LANGFUSE_ENABLED": "false",
    "DEBOUNCE_SECONDS": "0",
    "DEBOUNCE_MAX_SECONDS": "0",
    "DEBOUNCE_ATTACHMENT_SECONDS": "0",
    "MESSENGER_TYPING_INDICATOR": "false",
    "HANDOFF_NOTIFY_WEBHOOK_URL": "",
    "DATABASE_URL": _test_database_url(),
    "FANPAGE_PROFILE_PATH": str(ROOT / "tests" / "fixtures" / "fanpage_profile.yaml"),
    "KNOWLEDGE_DIR": str(ROOT / "tests" / "fixtures" / "knowledge"),
    "GROUNDING_MODE": "annotate",
    "RAG_DOCS_ENABLED": "false",
    "VISION_ENABLED": "false",
}
os.environ.update(TEST_ENV)

import pytest  # noqa: E402

from app.config import get_settings, reset_settings_cache  # noqa: E402

reset_settings_cache()

TRUNCATE = (
    "TRUNCATE feedback, training_candidates, dataset_exports, audit_log, worker_heartbeats, messages, turns, "
    "conversations, kb_documents, kb_chunks RESTART IDENTITY CASCADE"
)


def _ensure_test_db() -> str | None:
    """Tạo DB test nếu chưa có + migrate. Trả về lý do lỗi (None = OK)."""
    import psycopg

    s = get_settings()
    conninfo = s.psycopg_conninfo
    admin = re.sub(r"/chatbot_test(\?|$)", r"/postgres\1", conninfo)
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'chatbot_test'").fetchone()
            if not exists:
                conn.execute("CREATE DATABASE chatbot_test")
    except Exception as exc:  # noqa: BLE001
        return f"PostgreSQL không sẵn sàng: {type(exc).__name__}"
    from alembic import command
    from alembic.config import Config

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.attributes["skip_logging_config"] = True
    command.upgrade(cfg, "head")
    import asyncio

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async def setup() -> None:
        async with AsyncPostgresSaver.from_conn_string(conninfo) as saver:
            await saver.setup()

    asyncio.run(setup())
    return None


_DB_STATE: dict[str, str | None] = {}


@pytest.fixture(scope="session")
def db_available() -> None:
    if "reason" not in _DB_STATE:
        _DB_STATE["reason"] = _ensure_test_db()
    if _DB_STATE["reason"]:
        pytest.skip(_DB_STATE["reason"])


@pytest.fixture
async def db(db_available):  # type: ignore[no-untyped-def]
    """DB sạch cho mỗi test (TRUNCATE) + engine async."""
    from sqlalchemy import text

    from app.storage.db import dispose_db, init_db, session_scope

    await dispose_db()
    init_db(get_settings())
    async with session_scope() as s:
        await s.execute(text(TRUNCATE))
        await s.execute(text("TRUNCATE checkpoints, checkpoint_blobs, checkpoint_writes"))
    yield
    await dispose_db()


@pytest.fixture
def settings():  # type: ignore[no-untyped-def]
    return get_settings()


@pytest.fixture(autouse=True)
def http_mock():  # type: ignore[no-untyped-def]
    """Mọi request HTTP ra ngoài (qua httpx) phải được mock; request không khớp sẽ lỗi -> không chạm Meta thật."""
    import respx

    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(host="127.0.0.1").pass_through()
        router.route(host="localhost").pass_through()
        yield router
