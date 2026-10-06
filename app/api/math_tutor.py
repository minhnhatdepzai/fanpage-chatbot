"""Trang gia sư toán và API giải có kiểm chứng."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.api.web import _check_session, _enabled, _ingest_text_message, _rate_limit_ip
from app.config import get_settings
from app.english_tutor import detect_english_task
from app.math_tutor import MathTutorError, looks_like_math, solve_math
from app.writing_tutor import detect_writing_task

router = APIRouter(prefix="/web/math", tags=["math-tutor"])
literature_router = APIRouter(prefix="/web/literature", tags=["literature-tutor"])
pages_router = APIRouter(tags=["study-pages"])
STATIC = Path(__file__).resolve().parents[1] / "web" / "static"


def _study_page() -> HTMLResponse:
    """HTML luôn mới; địa chỉ CSS/JS thay theo nội dung để tránh lệch backend/frontend."""
    html = (STATIC / "math.html").read_text(encoding="utf-8")
    for filename, route in (
        ("math.js", "app.js"),
        ("study-bank.js", "study-bank.js"),
        ("math-scenes.js", "scenes.js"),
        ("math.css", "style.css"),
        ("math-upload.css", "upload.css"),
        ("math-subjects.css", "subjects.css"),
    ):
        version = hashlib.sha256((STATIC / filename).read_bytes()).hexdigest()[:12]
        html = re.sub(rf"/web/math/{re.escape(route)}\?v=[^\"']+", f"/web/math/{route}?v={version}", html)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


class MathProblemIn(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    curriculum: Literal["vi", "world"] = "vi"
    grade: int | None = Field(default=None, ge=1, le=12)


class MathImageIn(MathProblemIn):
    question: str = Field(default="Hãy đọc và giải bài toán trong ảnh.", max_length=1200)
    image_base64: str = Field(min_length=16)


class LiteratureSubmissionIn(BaseModel):
    session_id: str = Field(max_length=64)
    token: str = Field(max_length=64)
    grade: int = Field(ge=1, le=12)
    prompt: str = Field(min_length=5, max_length=2400)
    answer: str = Field(min_length=20, max_length=12000)


@router.get("", include_in_schema=False)
async def math_page() -> HTMLResponse:
    _enabled()
    return _study_page()


@pages_router.get("/web/chat", include_in_schema=False)
async def chat_page() -> HTMLResponse:
    _enabled()
    return _study_page()


@pages_router.get("/web/literature", include_in_schema=False)
async def literature_page() -> HTMLResponse:
    _enabled()
    return _study_page()


@pages_router.get("/web/english", include_in_schema=False)
async def english_page() -> FileResponse:
    _enabled()
    return FileResponse(STATIC / "english.html", media_type="text/html")


@router.get("/app.js", include_in_schema=False)
async def math_js() -> FileResponse:
    _enabled()
    return FileResponse(
        STATIC / "math.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"}
    )


@router.get("/style.css", include_in_schema=False)
async def math_css() -> FileResponse:
    _enabled()
    return FileResponse(STATIC / "math.css", media_type="text/css")


@router.get("/scenes.js", include_in_schema=False)
async def math_scenes_js() -> FileResponse:
    _enabled()
    return FileResponse(
        STATIC / "math-scenes.js", media_type="application/javascript", headers={"Cache-Control": "no-cache"}
    )


@router.get("/upload.css", include_in_schema=False)
async def math_upload_css() -> FileResponse:
    _enabled()
    return FileResponse(
        STATIC / "math-upload.css", media_type="text/css", headers={"Cache-Control": "no-cache"}
    )


@router.get("/subjects.css", include_in_schema=False)
async def math_subjects_css() -> FileResponse:
    _enabled()
    return FileResponse(STATIC / "math-subjects.css", media_type="text/css")


@router.get("/study-bank.js", include_in_schema=False)
async def study_bank_js() -> FileResponse:
    _enabled()
    return FileResponse(STATIC / "study-bank.js", media_type="application/javascript")


@literature_router.post("/submit", status_code=202)
async def submit_literature(body: LiteratureSubmissionIn, request: Request) -> dict[str, Any]:
    """Đưa bài làm dài vào đúng hàng đợi chatbot để gia sư Văn chấm, sửa và hướng dẫn phát triển."""
    _enabled()
    settings = get_settings()
    _check_session(body.session_id, body.token)
    _rate_limit_ip(request, "literature-submission", settings.web_rate_limit_per_5min)
    message = (
        f"Chấm, nhận xét và sửa bài văn lớp {body.grade} dưới đây theo đúng đề.\n\n"
        f"ĐỀ BÀI:\n{body.prompt.strip()}\n\n"
        "BÀI LÀM CỦA HỌC SINH:\n"
        f"<<<\n{body.answer.strip()}\n>>>\n\n"
        "YÊU CẦU CHẤM:\n"
        "1. Chấm theo thang 10 và giải thích điểm theo các tiêu chí: đáp ứng đề, luận điểm/nội dung, "
        "dẫn chứng và phân tích, bố cục/liên kết, diễn đạt/chính tả, sáng tạo.\n"
        "2. Chỉ ra điểm tốt bằng dẫn chứng ngắn từ bài làm; sau đó nêu lỗi cụ thể và cách sửa.\n"
        "3. Đề xuất 3-5 hướng phát triển ý sâu hơn, gồm ít nhất một phản đề hoặc liên hệ đời sống nếu phù hợp.\n"
        "4. Viết một bản sửa tham khảo giữ giọng của học sinh; không bịa dẫn chứng tác phẩm và không nhận bản sửa "
        "là bài nguyên gốc của học sinh."
    )
    await _ingest_text_message(body.session_id, message, settings)
    return {"accepted": True, "mode": "writing_revision", "grade": body.grade}


@router.post("/solve")
async def solve(body: MathProblemIn, request: Request) -> dict[str, Any]:
    _enabled()
    _rate_limit_ip(request, "math", 3 * get_settings().web_rate_limit_per_5min)
    try:
        return solve_math(body.question, curriculum=body.curriculum, grade=body.grade)
    except MathTutorError as exc:
        raise HTTPException(
            422,
            {
                "message": str(exc),
                "safe": True,
                "hint": "Hãy nhập phép tính, phương trình một ẩn, hoặc hệ tuyến tính; ví dụ: 3x + 5 = 20.",
            },
        ) from exc


@router.post("/route")
async def route_message(body: MathProblemIn, request: Request) -> dict[str, Any]:
    """Phân biệt đề toán với hội thoại thường; chỉ trả structured solution khi kiểm chứng được."""
    _enabled()
    _rate_limit_ip(request, "math-route", 3 * get_settings().web_rate_limit_per_5min)
    if not looks_like_math(body.question):
        english = detect_english_task(body.question)
        if english.is_english:
            return {
                "mode": "english",
                "math_detected": False,
                "notice": "Đã nhận diện bài Tiếng Anh và chuyển tới gia sư theo kỹ năng.",
            }
        writing = detect_writing_task(body.question)
        if writing.is_writing:
            return {
                "mode": "writing",
                "math_detected": False,
                "notice": "Đã nhận diện yêu cầu Ngữ văn/viết và chuyển tới gia sư phù hợp.",
            }
        return {"mode": "chat", "math_detected": False}
    try:
        solution = solve_math(body.question, curriculum=body.curriculum, grade=body.grade)
    except MathTutorError as exc:
        return {
            "mode": "math_review",
            "math_detected": True,
            "notice": (
                f"Đề nằm ngoài bộ giải xác định ({exc}); lời giải chỉ được gửi sau một lượt kiểm định độc lập."
            ),
        }
    return {"mode": "math", "math_detected": True, "solution": solution}


def _math_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    candidates += [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    candidates += [
        match.group(1).strip() for match in re.finditer(r"(?:câu|bài)\s*\d*\s*[:.]\s*(.+)", text, re.I)
    ]
    # Ảnh giao diện thường lặp lại cùng phép tính cạnh hình minh họa. Tách từng mảnh toán thay vì đưa toàn bộ OCR
    # nhiễu vào parser. Với đẳng thức chỉ gồm số, lấy vế trái để trả chính kết quả phép tính.
    atom = r"(?:\d+(?:[.,]\d+)?|[xyz])"
    for match in re.finditer(rf"{atom}(?:\s*[+\-*/^=]\s*{atom})+", text, re.I):
        fragment = match.group(0).strip()
        if "=" in fragment and not re.search(r"[xyz]", fragment, re.I):
            candidates.append(fragment.split("=", 1)[0].strip())
        candidates.append(fragment)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


@router.post("/read-image")
async def read_image(body: MathImageIn, request: Request) -> dict[str, Any]:
    """Đọc đề từ ảnh; chỉ trả đáp án nếu bộ giải xác định kiểm chứng được."""
    from app.vision.client import VisionClient, VisionError
    from app.vision.context import sanitize_analysis

    _enabled()
    settings = get_settings()
    if not settings.vision_enabled:
        raise HTTPException(503, "Hệ thống đọc ảnh đang tắt.")
    _rate_limit_ip(request, "math-image", settings.web_rate_limit_per_5min)
    if len(body.image_base64) > settings.vision_max_image_mb * 1_000_000 * 4 // 3 + 16:
        raise HTTPException(413, f"Ảnh vượt {settings.vision_max_image_mb} MB.")
    try:
        image = base64.b64decode(body.image_base64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(400, "Ảnh không hợp lệ.") from exc
    try:
        analysis = sanitize_analysis(await VisionClient(settings).analyze(image, question=body.question))
    except VisionError as exc:
        raise HTTPException(503, "Hiện chưa đọc được ảnh; hãy thử ảnh rõ hơn hoặc gõ lại đề.") from exc
    vlm = analysis.get("vlm") or {}
    extracted = str(vlm.get("visible_text") or (analysis.get("ocr") or {}).get("text") or "").strip()
    for candidate in _math_candidates(extracted):
        try:
            result = solve_math(candidate, curriculum=body.curriculum, grade=body.grade)
        except MathTutorError:
            continue
        result["image_analysis"] = {
            "extracted_text": extracted,
            "ocr_lines": (analysis.get("ocr") or {}).get("lines", 0),
            "vlm_used": bool(vlm),
        }
        result["warnings"] = ["Đề được chép tự động từ ảnh; hãy đối chiếu lại phần văn bản đã nhận dạng."]
        return result
    raise HTTPException(
        422,
        {
            "message": "Mình đã đọc ảnh nhưng chưa tách được biểu thức để kiểm chứng an toàn.",
            "safe": True,
            "extracted_text": extracted,
            "hint": "Hãy kiểm tra phần chữ nhận dạng rồi gõ lại phép tính/phương trình vào ô chat.",
        },
    )
