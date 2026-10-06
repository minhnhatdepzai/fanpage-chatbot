"""CLI vận hành: ``uv run botctl --help``.

Không lệnh nào in token/secret. Các lệnh quản trị thao tác trực tiếp trên database (chạy trên máy chủ).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

import typer

from app.config import get_settings

app = typer.Typer(help="Vận hành fanpage-chatbot", no_args_is_help=True)
db_app = typer.Typer(help="Database & migration", no_args_is_help=True)
meta_app = typer.Typer(help="Kiểm tra cấu hình Meta/Messenger", no_args_is_help=True)
review_app = typer.Typer(help="Xem và chấm các lượt trả lời", no_args_is_help=True)
cand_app = typer.Typer(help="Duyệt mẫu dữ liệu huấn luyện", no_args_is_help=True)
data_app = typer.Typer(help="Xuất dữ liệu đã duyệt", no_args_is_help=True)
conv_app = typer.Typer(help="Hội thoại: handoff, xóa dữ liệu", no_args_is_help=True)
model_app = typer.Typer(help="Registry adapter: status/register/promote/rollback", no_args_is_help=True)
kb_app = typer.Typer(help="Kho kiến thức có nguồn (config/knowledge)", no_args_is_help=True)
docs_app = typer.Typer(help="Tài liệu PDF/Word cho RAG (pgvector)", no_args_is_help=True)
vision_app = typer.Typer(help="Ảnh: OCR + nhận diện vật thể (chạy cục bộ)", no_args_is_help=True)
for sub, name in [
    (db_app, "db"),
    (meta_app, "meta"),
    (review_app, "review"),
    (cand_app, "candidates"),
    (data_app, "data"),
    (conv_app, "conversations"),
    (model_app, "model"),
    (kb_app, "knowledge"),
    (docs_app, "docs"),
    (vision_app, "vision"),
]:
    app.add_typer(sub, name=name)


def _print(obj: Any) -> None:
    typer.echo(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def _run(coro_factory) -> Any:  # type: ignore[no-untyped-def]
    from app.observability.logging_setup import configure_logging
    from app.storage.db import dispose_db, init_db

    s = get_settings()
    configure_logging("WARNING", s.secret_values())

    async def runner() -> Any:
        init_db(s)
        try:
            return await coro_factory()
        finally:
            await dispose_db()

    return asyncio.run(runner())


# ----------------------------------------------------------------------------- tiến trình
@app.command()
def api(host: str = typer.Option(None), port: int = typer.Option(None)) -> None:
    """Chạy FastAPI (webhook + admin)."""
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "app.main:app",
        host=host or s.api_host,
        port=port or s.api_port,
        log_config=None,
        access_log=False,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )


@app.command()
def worker() -> None:
    """Chạy worker xử lý hội thoại."""
    from app.workers.main import main

    main()


# ----------------------------------------------------------------------------- db
@db_app.command("migrate")
def db_migrate() -> None:
    """alembic upgrade head + tạo bảng checkpoint của LangGraph."""
    from alembic import command
    from alembic.config import Config

    from app.config.settings import PROJECT_ROOT

    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    command.upgrade(cfg, "head")

    async def setup_checkpointer() -> None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(get_settings().psycopg_conninfo) as saver:
            await saver.setup()

    asyncio.run(setup_checkpointer())
    typer.echo("migrate: OK (alembic head + langgraph checkpointer)")


# ----------------------------------------------------------------------------- meta
@meta_app.command("check")
def meta_check() -> None:
    """Xác định loại token, quyền, App Secret, trạng thái webhook (không in bí mật)."""
    from app.messenger.meta_check import check_meta

    _print(asyncio.run(check_meta(get_settings())))


@meta_app.command("set-webhook")
def meta_set_webhook(
    callback_url: str = typer.Argument(
        ..., help="URL HTTPS công khai tới /webhook, vd. https://xxx.trycloudflare.com/webhook"
    ),
) -> None:
    """Trỏ webhook của app tới URL mới (dùng mỗi khi URL tunnel đổi) + đăng ký Page với app."""
    from app.messenger.meta_check import set_webhook

    if not callback_url.startswith("https://") or not callback_url.rstrip("/").endswith("/webhook"):
        raise typer.BadParameter("callback_url phải là https://.../webhook")
    _print(asyncio.run(set_webhook(get_settings(), callback_url.rstrip("/"))))


@meta_app.command("send-test")
def meta_send_test(
    psid: str = typer.Option(..., help="PSID tài khoản KIỂM THỬ (phải có trong MESSENGER_ALLOWED_PSIDS)"),
    text: str = typer.Option("Tin nhắn kiểm thử từ fanpage-chatbot."),
) -> None:
    """Gửi 1 tin thử - chỉ tới PSID nằm trong allowlist."""
    from app.messenger.client import MessengerClient

    s = get_settings()
    if psid not in s.messenger_allowed_psids:
        raise typer.BadParameter("PSID không nằm trong MESSENGER_ALLOWED_PSIDS - từ chối gửi.")

    async def go() -> Any:
        client = MessengerClient(s)
        try:
            return await client.send_text(psid, text)
        finally:
            await client.aclose()

    res = asyncio.run(go())
    _print(
        {
            "outcome": res.outcome,
            "message_id": res.message_id,
            "error_code": res.error_code,
            "error_subcode": res.error_subcode,
            "detail": res.detail,
        }
    )


# ----------------------------------------------------------------------------- review
@review_app.command("list")
def review_list(
    status: str = typer.Option("unrated", help="unrated|flagged|bad|errors|all"), limit: int = 20
) -> None:
    from app.admin import service

    _print(_run(lambda: service.list_turns(status=status, limit=limit)))


@review_app.command("rate")
def review_rate(
    turn_id: uuid.UUID,
    rating: str = typer.Argument(..., help="good|bad"),
    notes: str = typer.Option(None),
    reviewer: str = typer.Option("cli-admin"),
) -> None:
    from app.admin import service

    _print(_run(lambda: service.rate_turn(turn_id, rating, reviewer, notes)))


@review_app.command("correct")
def review_correct(
    turn_id: uuid.UUID,
    text: str = typer.Option(..., help="Câu trả lời đúng"),
    notes: str = typer.Option(None),
    reviewer: str = typer.Option("cli-admin"),
) -> None:
    from app.admin import service

    _print(_run(lambda: service.correct_turn(turn_id, text, reviewer, notes)))


@cand_app.command("list")
def cand_list(status: str = typer.Option("pending_review"), limit: int = 50) -> None:
    from app.admin import service

    _print(_run(lambda: service.list_candidates(status, limit)))


@cand_app.command("approve")
def cand_approve(
    cand_id: int,
    confirm_usage_rights: bool = typer.Option(False, "--confirm-usage-rights"),
    notes: str = typer.Option(None),
    reviewer: str = typer.Option("cli-admin"),
) -> None:
    from app.admin import service

    _print(
        _run(
            lambda: service.review_candidate(
                cand_id,
                approve=True,
                reviewer=reviewer,
                usage_rights_confirmed=confirm_usage_rights,
                notes=notes,
            )
        )
    )


@cand_app.command("reject")
def cand_reject(
    cand_id: int, notes: str = typer.Option(None), reviewer: str = typer.Option("cli-admin")
) -> None:
    from app.admin import service

    _print(
        _run(
            lambda: service.review_candidate(
                cand_id, approve=False, reviewer=reviewer, usage_rights_confirmed=False, notes=notes
            )
        )
    )


@data_app.command("export")
def data_export(
    kind: str = typer.Option("sft", help="sft|dpo"), reviewer: str = typer.Option("cli-admin")
) -> None:
    from app.admin import service

    _print(_run(lambda: service.export_approved(reviewer=reviewer, kind=kind)))


# ----------------------------------------------------------------------------- conversations
@conv_app.command("list")
def conv_list(
    limit: int = 20,
    handoff_only: bool = False,
    show_psid: bool = typer.Option(False, help="Hiện PSID (để thêm tài khoản kiểm thử vào allowlist)"),
) -> None:
    from app.admin import service

    _print(_run(lambda: service.list_conversations(limit, handoff_only, include_psid=show_psid)))


@conv_app.command("show")
def conv_show(conv_id: uuid.UUID) -> None:
    from app.admin import service

    _print(_run(lambda: service.get_conversation(conv_id)))


@conv_app.command("handoff")
def conv_handoff(
    conv_id: uuid.UUID,
    state: str = typer.Argument(..., help="on|off"),
    reason: str = typer.Option(None),
    actor: str = typer.Option("cli-admin"),
) -> None:
    """Bật (on) / tắt (off = bật lại bot) handoff cho một hội thoại."""
    from app.admin import service

    if state not in ("on", "off"):
        raise typer.BadParameter("state phải là on hoặc off")
    _print(_run(lambda: service.set_handoff(conv_id, state == "on", actor, reason)))


@conv_app.command("delete")
def conv_delete(
    conv_id: uuid.UUID, yes: bool = typer.Option(False, "--yes"), actor: str = typer.Option("cli-admin")
) -> None:
    """Xóa toàn bộ dữ liệu hội thoại (tin nhắn, tóm tắt, checkpoint, phản hồi, mẫu, trace)."""
    from app.admin import service
    from app.observability.tracing import build_tracer

    if not yes:
        raise typer.BadParameter("Thêm --yes để xác nhận xóa vĩnh viễn")

    async def go() -> Any:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        tracer = build_tracer(get_settings())
        try:
            async with AsyncPostgresSaver.from_conn_string(get_settings().psycopg_conninfo) as saver:
                return await service.delete_conversation(conv_id, actor, saver, tracer)
        finally:
            tracer.shutdown()

    _print(_run(go))


# ----------------------------------------------------------------------------- model registry
@model_app.command("status")
def model_status() -> None:
    from serving import registry

    _print(registry.load(get_settings().adapter_registry_path))


@model_app.command("register")
def model_register(adapter_dir: Path) -> None:
    from serving import registry

    _print({"registered": registry.register(get_settings().adapter_registry_path, adapter_dir)})


@model_app.command("attach-eval")
def model_attach_eval(adapter_id: str, report: Path) -> None:
    from serving import registry

    registry.attach_eval(get_settings().adapter_registry_path, adapter_id, report)
    _print({"adapter_id": adapter_id, "eval_report": str(report)})


@model_app.command("promote")
def model_promote(
    adapter_id: str,
    reason: str = typer.Option(...),
    by: str = typer.Option("cli-admin"),
    force: bool = typer.Option(False, help="Bỏ qua cổng đánh giá (không khuyến nghị)"),
) -> None:
    """Triển khai adapter (cần báo cáo đánh giá PASS). Sau đó reload model server."""
    from serving import registry

    _print(
        registry.promote(get_settings().adapter_registry_path, adapter_id, by=by, reason=reason, force=force)
    )
    typer.echo(
        "Nhớ reload model server: curl -X POST -H 'Authorization: Bearer ...' http://127.0.0.1:8100/admin/reload"
    )


@model_app.command("rollback")
def model_rollback(reason: str = typer.Option(...), by: str = typer.Option("cli-admin")) -> None:
    from serving import registry

    _print(registry.rollback(get_settings().adapter_registry_path, by=by, reason=reason))
    typer.echo("Nhớ reload model server để áp dụng.")


# ----------------------------------------------------------------------------- kho kiến thức
@kb_app.command("list")
def kb_list() -> None:
    """Liệt kê các mục đã nạp (mục thiếu nguồn bị bỏ qua)."""
    from app.conversation.knowledge import load_knowledge

    kb = load_knowledge(get_settings().knowledge_dir)
    _print(
        [{"id": e.id, "title": e.title, "source": e.source, "keywords": list(e.keywords)} for e in kb.entries]
    )


@kb_app.command("search")
def kb_search(query: str, top_k: int = typer.Option(None)) -> None:
    """Thử truy xuất: câu hỏi này sẽ kèm những nguồn nào và có bị coi là câu hỏi kiến thức không."""
    from app.conversation.intents import is_knowledge_question
    from app.conversation.knowledge import load_knowledge

    s = get_settings()
    kb = load_knowledge(s.knowledge_dir)
    hits = kb.search(query, top_k=top_k or s.knowledge_top_k)
    _print(
        {
            "knowledge_question": is_knowledge_question(query),
            "grounding_mode": s.grounding_mode.value,
            "hits": [{"id": h.entry.id, "score": h.score, "source": h.entry.source} for h in hits],
        }
    )


# ----------------------------------------------------------------------------- tài liệu (RAG) + trợ lý nội bộ
def _ocr_client() -> Any:
    from app.vision.client import VisionClient

    return VisionClient(get_settings()).ocr_image


@docs_app.command("ingest")
def docs_ingest(
    paths: list[Path] = typer.Argument(..., help="Tệp .pdf / .docx / .txt / .md (hoặc thư mục)"),
    visibility: str = typer.Option(
        "internal", help="public (bot công khai được dùng) | internal (chỉ nội bộ)"
    ),
    title: str = typer.Option(None, help="Tiêu đề hiển thị khi trích nguồn (mặc định: tên tệp)"),
    source: str = typer.Option(None, help="Nguồn hiển thị, vd. URL gốc của tài liệu"),
    ocr: bool = typer.Option(True, help="OCR các trang scan qua model server"),
    allow_pii_public: bool = typer.Option(False, help="Cho phép tài liệu có dữ liệu cá nhân ở kênh public"),
    replace: bool = typer.Option(False, help="Nạp lại nếu tài liệu đã có"),
    actor: str = typer.Option("cli-admin"),
) -> None:
    """Nạp tài liệu vào pgvector: đọc -> OCR trang scan -> làm sạch -> chia đoạn -> embedding."""
    from app.rag.embedder import HttpEmbedder
    from app.rag.ingest import ingest_file
    from app.rag.parsing import SUPPORTED, DocumentError

    s = get_settings()
    files: list[Path] = []
    for p in paths:
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED) if p.is_dir() else [p]
    ocr_fn = _ocr_client() if ocr else None

    async def go() -> list[dict[str, Any]]:
        out = []
        for f in files:
            try:
                out.append(
                    await ingest_file(
                        f,
                        settings=s,
                        embedder=HttpEmbedder(s),
                        visibility=visibility,
                        actor=actor,
                        title=title if len(files) == 1 else None,
                        source=source if len(files) == 1 else None,
                        ocr=ocr_fn,
                        allow_pii_public=allow_pii_public,
                        replace=replace,
                    )
                )
            except DocumentError as exc:
                out.append({"file": f.name, "status": "error", "error": str(exc)})
        return out

    _print(_run(go))


@docs_app.command("list")
def docs_list() -> None:
    from app.rag import store

    _print(_run(store.list_documents))


@docs_app.command("delete")
def docs_delete(
    doc_id: uuid.UUID, yes: bool = typer.Option(False, "--yes"), actor: str = "cli-admin"
) -> None:
    """Xóa tài liệu và mọi đoạn/embedding của nó."""
    from app.rag import store

    if not yes:
        raise typer.BadParameter("Thêm --yes để xác nhận xóa")
    _print({"deleted": _run(lambda: store.delete_document(doc_id, actor))})


@docs_app.command("search")
def docs_search(
    query: str,
    visibility: str = typer.Option("public", help="public | all"),
    top_k: int = 5,
    min_similarity: float = typer.Option(0.0, help="0 = hiện cả đoạn dưới ngưỡng để hiệu chỉnh"),
) -> None:
    """Xem đoạn tài liệu nào được truy xuất và điểm tương đồng (dùng để hiệu chỉnh DOC_MIN_SIMILARITY)."""
    from app.rag import store
    from app.rag.embedder import HttpEmbedder

    s = get_settings()
    emb = HttpEmbedder(s)

    async def go() -> list[dict[str, Any]]:
        vec = (await emb.embed([query]))[0]
        hits = await store.search(
            query,
            vec,
            visibility=store.ALL if visibility == "all" else store.PUBLIC,
            embedding_model=emb.model,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        return [
            {
                "similarity": h.similarity,
                "passes_threshold": h.similarity >= s.doc_min_similarity,
                "title": h.citation_title,
                "text": h.content[:160],
            }
            for h in hits
        ]

    _print(_run(go))


@app.command()
def ask(question: str, visibility: str = typer.Option("all", help="all | public")) -> None:
    """Trợ lý nội bộ: hỏi đáp dựa trên tài liệu đã nạp (có trích nguồn)."""
    from app.conversation.knowledge import load_knowledge
    from app.providers.llm import build_provider
    from app.rag import store
    from app.rag.assistant import ask as do_ask
    from app.rag.embedder import HttpEmbedder
    from app.rag.retriever import DocSearch

    s = get_settings()

    async def go() -> dict[str, Any]:
        return await do_ask(
            question,
            settings=s,
            llm=build_provider(s),
            docs=DocSearch(s, HttpEmbedder(s)),
            kb=load_knowledge(s.knowledge_dir),
            visibility=store.ALL if visibility == "all" else store.PUBLIC,
        )

    res = _run(go)
    typer.echo(res["answer"])
    typer.echo(json.dumps({k: res[k] for k in ("flags", "prompt_version")}, ensure_ascii=False))


# ----------------------------------------------------------------------------- ảnh (cục bộ)
def _vision_runtime() -> Any:
    from serving.vision import VisionRuntime

    s = get_settings()
    return VisionRuntime(
        coco_weights=str(s.vision_coco_weights),
        custom_weights=str(s.vision_custom_weights) if s.vision_custom_weights else None,
        min_conf=s.vision_min_conf,
        custom_min_conf=s.vision_custom_min_conf,
        ocr_min_conf=s.vision_ocr_min_conf,
    )


@vision_app.command("analyze")
def vision_analyze(image: Path) -> None:
    """Chỉ phân tích: chữ OCR + vật thể nhận diện (không gọi model ngôn ngữ)."""
    from serving.vision import load_image

    s = get_settings()
    res = _vision_runtime().analyze(load_image(image.read_bytes(), s.vision_max_image_mb * 1_000_000))
    _print(res)


@vision_app.command("ask")
def vision_ask(
    image: Path, question: str = typer.Argument("", help="Câu hỏi kèm ảnh (có thể bỏ trống)")
) -> None:
    """Hỏi về một ảnh: OCR + YOLO -> ngữ cảnh ảnh + nguồn tham khảo -> model ngôn ngữ -> kiểm tra đầu ra."""
    from datetime import UTC, datetime

    from langchain_core.messages import HumanMessage, SystemMessage

    from app.conversation.knowledge import extract_urls, load_knowledge
    from app.conversation.output_check import OutputCheckContext, check_output
    from app.conversation.profile import load_profile
    from app.conversation.prompts import build_system_prompt
    from app.providers.llm import build_provider, generate_with_retry
    from app.rag.embedder import HttpEmbedder
    from app.rag.retriever import DocSearch
    from app.vision.context import NOTHING_RECOGNIZED_REPLY, image_context_block, nothing_recognized, ocr_text
    from serving.vision import load_image, summarize_objects

    s = get_settings()
    image_bytes = image.read_bytes()
    analysis = _vision_runtime().analyze(load_image(image_bytes, s.vision_max_image_mb * 1_000_000))
    user_text = question.strip() or "(Người dùng gửi ảnh, không kèm câu hỏi.)"
    query = f"{question}\n{ocr_text(analysis)[:300]}".strip()

    async def go() -> dict[str, Any]:
        if s.vision_vlm_enabled:
            from app.vision.client import VisionClient, VisionError
            from app.vision.context import sanitize_analysis

            try:
                analysis["vlm"] = await VisionClient(s).describe(image_bytes, question=question or None)
                analysis.update(sanitize_analysis(analysis))
            except VisionError as exc:
                typer.echo(f"[cảnh báo] VLM không phản hồi: {exc}", err=True)
        if nothing_recognized([analysis]):
            return {"answer": NOTHING_RECOGNIZED_REPLY, "flags": ["image_nothing_recognized"]}
        kb = load_knowledge(s.knowledge_dir)
        entries = [h.entry for h in kb.search(query, top_k=s.knowledge_top_k)]
        try:
            entries += await DocSearch(s, HttpEmbedder(s)).search(query, visibility=frozenset({"public"}))
        except Exception as exc:  # noqa: BLE001 - tài liệu là phần thêm, lỗi thì bỏ qua
            typer.echo(f"[cảnh báo] không tìm được tài liệu: {type(exc).__name__}", err=True)
        system = build_system_prompt(
            load_profile(s.fanpage_profile_path),
            datetime.now(UTC),
            summary=None,
            needs_disclosure=False,
            sources=entries,
            image_context=image_context_block([analysis]),
            vision_available=True,
            rich_vision_available="vlm" in analysis,
            docs_available=True,
        )
        res = await generate_with_retry(
            build_provider(s),
            [SystemMessage(system), HumanMessage(user_text)],
            max_tokens=s.llm_max_output_tokens,
            temperature=s.llm_temperature,
            timeout=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
        )
        checked = check_output(
            res.text,
            OutputCheckContext(
                needs_disclosure=False,
                is_first_bot_reply=False,
                user_text=user_text,
                notifier_configured=False,
                secret_values=s.secret_values(),
                sources=[(e.title, e.source) for e in entries],
                allowed_urls={u for e in entries for u in extract_urls(e.source)},
                knowledge_question=bool(question),
                has_image_context=True,
            ),
        )
        return {"answer": checked.text, "flags": checked.flags}

    out = _run(go)
    typer.echo(
        f"[ảnh] chữ đọc được: {analysis['ocr']['lines']} dòng (tin cậy TB {analysis['ocr']['mean_conf']}); "
        f"vật thể: {summarize_objects(analysis['objects']) or 'không có'}; {analysis['ms']} ms"
    )
    typer.echo(out["answer"] or "(không có câu trả lời)")
    typer.echo(json.dumps({"flags": out["flags"]}, ensure_ascii=False))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
