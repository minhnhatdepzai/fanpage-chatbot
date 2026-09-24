"""Client dịch vụ ảnh trên model server: OCR (chữ trong ảnh) + nhận diện vật thể (YOLO).

Ảnh chỉ xử lý trong bộ nhớ, không lưu xuống đĩa; kết quả OCR có thể chứa dữ liệu cá nhân nên lớp gọi phải che trước khi
ghi log/trace.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import httpx

from app.config import Settings


class VisionError(RuntimeError):
    def __init__(self, detail: str, status: int | None = None) -> None:
        super().__init__(detail)
        self.status = status  # 400/413 = ảnh không hợp lệ; None/5xx = dịch vụ lỗi


class VisionClient:
    def __init__(self, settings: Settings) -> None:
        self._base = settings.embedding_base_url.rstrip("/")  # cùng model server
        self._key = settings.model_server_api_key.get_secret_value()
        self._timeout = settings.vision_timeout_seconds

    async def _post(self, path: str, image_bytes: bytes, **extra: Any) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        payload = {"image_base64": base64.b64encode(image_bytes).decode(), **extra}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                r = await client.post(f"{self._base}{path}", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise VisionError(type(exc).__name__) from exc
        if r.status_code != 200:
            raise VisionError(f"HTTP {r.status_code}", status=r.status_code)
        try:
            return r.json()
        except ValueError as exc:
            raise VisionError("bad_json") from exc

    async def analyze(self, image_bytes: bytes) -> dict[str, Any]:
        return await self._post("/vision/analyze", image_bytes)

    async def ocr_image(self, image: Any) -> str:
        """Ảnh PIL (vd. trang PDF scan) -> văn bản."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        return (await self._post("/vision/ocr", buf.getvalue()))["text"]
