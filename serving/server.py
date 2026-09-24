"""Model server tương thích OpenAI Chat Completions (chạy trên host có GPU).

    uv run python -m serving.server                          # base + adapter active trong registry
    uv run python -m serving.server --adapter artifacts/adapters/<id>   # ép dùng adapter (đánh giá)
    uv run python -m serving.server --no-adapter             # chỉ base model

Worker gọi qua LangChain ``ChatOpenAI(base_url=http://127.0.0.1:8100/v1)``. Bảo vệ bằng Bearer
MODEL_SERVER_API_KEY. Trả ``system_fingerprint`` = adapter id (hoặc "base") để truy vết phiên bản.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.observability.logging_setup import configure_logging
from serving import registry
from serving.runtime import ModelRuntime

log = logging.getLogger("serving")
STATE: dict[str, Any] = {}


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, Any]]


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: list[str] | str | None = None
    seed: int | None = None
    stream: bool = False
    n: int = Field(default=1, ge=1, le=1)


def _auth(authorization: str | None = Header(default=None)) -> None:
    key = get_settings().model_server_api_key.get_secret_value()
    if not key:
        return
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token.encode(), key.encode()):
        raise HTTPException(401, "unauthorized")


def _flatten(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


MAX_EMBED_INPUTS = 64
MAX_EMBED_CHARS = 8000


class ImageRequest(BaseModel):
    image_base64: str


def _decode_image(req: ImageRequest, max_mb: int) -> Any:
    import base64
    import binascii

    from serving.vision import ImageError, load_image

    if len(req.image_base64) > max_mb * 1_000_000 * 4 // 3 + 16:
        raise HTTPException(413, f"ảnh vượt {max_mb} MB")
    try:
        data = base64.b64decode(req.image_base64, validate=True)
        return load_image(data, max_mb * 1_000_000)
    except (binascii.Error, ValueError) as exc:  # ImageError là ValueError
        raise HTTPException(400, str(exc) if isinstance(exc, ImageError) else "base64 không hợp lệ") from exc


def create_app(runtime: ModelRuntime, served_name: str, embedder: Any = None, vision: Any = None) -> FastAPI:
    app = FastAPI(title="fanpage-chatbot model server", docs_url=None, redoc_url=None)
    STATE.update(runtime=runtime, served_name=served_name, embedder=embedder, vision=vision)
    max_image_mb = get_settings().vision_max_image_mb

    def model_label() -> str:
        return f"{served_name}+{runtime.adapter_id}" if runtime.adapter_id else served_name

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": runtime.base_model_id, "revision": runtime.revision,
                "adapter": runtime.adapter_id or "base", "method": runtime.adapter_method, "vram_gb": runtime.vram_gb(),
                "embedding_model": embedder.model_id if embedder else None,
                "vision": {"custom_model": vision.custom_id} if vision else None}

    @app.post("/v1/vision/analyze", dependencies=[Depends(_auth)])
    async def vision_analyze(req: ImageRequest) -> dict[str, Any]:
        """Ảnh -> chữ OCR + vật thể nhận diện. Ảnh chỉ xử lý trong bộ nhớ."""
        if vision is None:
            raise HTTPException(503, "dịch vụ ảnh chưa bật trên model server")
        img = _decode_image(req, max_image_mb)
        return await asyncio.to_thread(vision.analyze, img)

    @app.post("/v1/vision/ocr", dependencies=[Depends(_auth)])
    async def vision_ocr(req: ImageRequest) -> dict[str, Any]:
        if vision is None:
            raise HTTPException(503, "dịch vụ ảnh chưa bật trên model server")
        img = _decode_image(req, max_image_mb)
        return await asyncio.to_thread(vision.ocr, img)

    @app.post("/v1/embeddings", dependencies=[Depends(_auth)])
    async def embeddings(req: EmbeddingRequest) -> dict[str, Any]:
        if embedder is None:
            raise HTTPException(503, "embedding chưa bật trên model server")
        texts = [req.input] if isinstance(req.input, str) else req.input
        if not texts or len(texts) > MAX_EMBED_INPUTS or any(len(t) > MAX_EMBED_CHARS for t in texts):
            raise HTTPException(400, f"cần 1-{MAX_EMBED_INPUTS} đoạn, mỗi đoạn <= {MAX_EMBED_CHARS} ký tự")
        vecs, n_tokens = await asyncio.to_thread(embedder.embed, texts)
        return {"object": "list", "model": embedder.model_id,
                "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
                "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens}}

    @app.get("/v1/models", dependencies=[Depends(_auth)])
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "local"}]}

    @app.post("/v1/chat/completions", dependencies=[Depends(_auth)])
    async def chat(req: ChatRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(400, "stream chưa được hỗ trợ")
        msgs = [{"role": m.role, "content": _flatten(m.content)} for m in req.messages]
        max_new = req.max_completion_tokens or req.max_tokens or 400
        stop = [req.stop] if isinstance(req.stop, str) else req.stop
        use_adapter = not (req.model or "").endswith("@base")  # "<tên>@base": bỏ qua adapter (đánh giá A/B)
        try:
            res = await asyncio.to_thread(
                runtime.generate, msgs, max_new_tokens=min(max_new, 2048), temperature=req.temperature,
                top_p=req.top_p, seed=req.seed, stop=stop, use_adapter=use_adapter,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_label() if use_adapter else served_name,
            "system_fingerprint": (runtime.adapter_id or "base") if use_adapter else "base",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": res.text},
                         "finish_reason": res.finish_reason}],
            "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
                      "total_tokens": res.prompt_tokens + res.completion_tokens},
        }

    @app.post("/admin/reload", dependencies=[Depends(_auth)])
    def reload() -> dict[str, Any]:
        """Nạp lại adapter active theo registry (sau promote/rollback)."""
        aid, path = registry.active_adapter(get_settings().adapter_registry_path)
        if aid and path:
            runtime.load_adapter(path, aid)
        else:
            runtime.unload_adapter()
        return {"adapter": runtime.adapter_id or "base"}

    return app


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=s.model_server_host)
    ap.add_argument("--port", type=int, default=s.model_server_port)
    ap.add_argument("--adapter", default=None, help="đường dẫn adapter (bỏ qua registry)")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--no-embeddings", action="store_true", help="không nạp model embedding (RAG tài liệu tắt)")
    ap.add_argument("--no-vision", action="store_true", help="không nạp OCR/YOLO (ảnh chỉ được báo là chưa xem được)")
    args = ap.parse_args()
    configure_logging(s.log_level, s.secret_values())
    adapter_path = adapter_id = None
    if args.adapter:
        adapter_path, adapter_id = args.adapter, args.adapter.rstrip("/").split("/")[-1]
    elif not args.no_adapter:
        adapter_id, adapter_path = registry.active_adapter(s.adapter_registry_path)
    runtime = ModelRuntime(s.base_model_id, s.base_model_revision, adapter_path=adapter_path, adapter_id=adapter_id)
    embedder = None
    if not args.no_embeddings:
        from serving.embedding import EmbeddingRuntime

        embedder = EmbeddingRuntime(s.embedding_model_id, s.embedding_model_revision)
    vision = None
    if s.vision_enabled and not args.no_vision:
        from serving.vision import VisionRuntime

        vision = VisionRuntime(
            coco_weights=str(s.vision_coco_weights),
            custom_weights=str(s.vision_custom_weights) if s.vision_custom_weights else None,
            min_conf=s.vision_min_conf, custom_min_conf=s.vision_custom_min_conf, ocr_min_conf=s.vision_ocr_min_conf,
        )
    import uvicorn

    uvicorn.run(
        create_app(runtime, s.served_model_name, embedder, vision),
        host=args.host, port=args.port, log_config=None, access_log=False,
    )


if __name__ == "__main__":
    main()
