from __future__ import annotations

import base64

import httpx
import pytest

from app.config import get_settings
from app.main import app
from app.math_tutor import MathTutorError, solve_math


@pytest.mark.parametrize(
    ("question", "answer", "visual"),
    [
        ("27 + 18", "45", "number_line"),
        ("3/4 + 1/8", "7/8", "fraction"),
        ("(-8/9)^3", "-512/729", "formula"),
        ("3x + 5 = 20", "x = 5", "balance"),
        ("x^2 - 5x + 6 = 0", "x ∈ {2, 3}", "graph"),
        ("x + y = 7; x - y = 1", "x = 4, y = 3", "coordinate_system"),
        ("đồ thị y=4x+9", "y = 4*x + 9", "graph"),
    ],
)
def test_verified_math_answers(question: str, answer: str, visual: str):
    result = solve_math(question)
    assert result["status"] == "verified"
    assert result["verification"]["passed"] is True
    assert result["answer"] == answer
    assert result["visual"]["type"] == visual


@pytest.mark.parametrize(
    "question",
    [
        "1 / 0",
        "__import__('os').system('id')",
        "x^20 = 1",
        "hãy chứng minh định lý Fermat lớn",
    ],
)
def test_unsafe_or_unsupported_input_abstains(question: str):
    with pytest.raises(MathTutorError):
        solve_math(question)


def test_curriculum_and_grade_are_explicit():
    result = solve_math("2 + 3", curriculum="world", grade=1)
    assert result["grade"] == 1
    assert "corestandards.org" in result["sources"][0]["url"]


def test_graph_request_returns_exact_points_and_is_detected_as_math():
    from app.math_tutor import looks_like_math

    result = solve_math("Vẽ đồ thị y = 4x + 9", grade=9)
    points = result["visual"]["points"]
    assert looks_like_math("đồ thị y=4x+9") is True
    assert result["verification"]["method"] == "exact_symbolic_sampling"
    assert result["visual"]["expression"] == "4*x + 9"
    assert {"x": 0, "y": 9.0} in points
    assert {"x": -2, "y": 1.0} in points


def test_short_draw_command_returns_verified_graph():
    from app.math_tutor import looks_like_math

    result = solve_math("vẽ y=6x-9", grade=9)
    assert looks_like_math("vẽ y=6x-9") is True
    assert result["answer"] == "y = 6*x - 9"
    assert result["visual"]["type"] == "graph"
    assert result["verification"]["method"] == "exact_symbolic_sampling"
    assert {"x": 0, "y": -9.0} in result["visual"]["points"]


@pytest.mark.parametrize(
    ("question", "answer", "topic", "method"),
    [
        ("Đạo hàm của x^3 - 3x + 1", "d/dx = 3*x^2 - 3", "Đạo hàm", "symbolic_differentiation"),
        ("Tích phân từ 0 đến 1 của x^2 dx", "1/3", "Tích phân xác định", "antiderivative_and_bounds"),
        ("lim x->0 sin(x)/x", "1", "Giới hạn", "symbolic_limit"),
        (
            "Giải bất phương trình x^2 - 5x + 6 >= 0",
            "x ∈ (-∞, 2] ∪ [3, ∞)",
            "Bất phương trình một ẩn",
            "exact_sign_analysis",
        ),
        ("C(10,3)", "120", "Số học và biểu thức", "exact_simplification"),
        ("A(5,2)", "20", "Số học và biểu thức", "exact_simplification"),
        ("Giải phương trình 3^(x-1) = 27", "x = 4", "Phương trình mũ", "exact_power_and_substitution"),
    ],
)
def test_high_school_exact_math_families(question: str, answer: str, topic: str, method: str):
    result = solve_math(question, grade=12)
    assert result["status"] == "verified"
    assert result["answer"] == answer
    assert result["topic"] == topic
    assert result["verification"] == {"engine": "sympy-safe-ast", "passed": True, "method": method}
    assert result["grade"] == 12


@pytest.mark.parametrize(
    "question",
    [
        "Đạo hàm của open('/etc/passwd')",
        "Tích phân từ 0 đến 1 của __import__('os') dx",
        "lim x->0 eval(x)",
        "C(5,-1)",
    ],
)
def test_high_school_math_keeps_safe_ast_boundary(question: str):
    with pytest.raises(MathTutorError):
        solve_math(question, grade=12)


@pytest.mark.parametrize(
    ("question", "answer", "visual"),
    [
        ("Trên bàn có 6 viên bi rồi đặt thêm 2 viên. Có tất cả bao nhiêu viên?", "8", "number_line"),
        ("Lan có 12 cái kẹo rồi cho đi 5 cái. Lan còn lại bao nhiêu?", "7", "number_line"),
        ("Có 4 hộp, mỗi hộp 3 cây bút. Có tất cả bao nhiêu bút?", "12", "groups"),
        ("12 viên kẹo chia đều cho 3 người. Mỗi người có bao nhiêu?", "4", "groups"),
        ("Hình chữ nhật có chiều dài 8 cm, chiều rộng 5 cm. Tính diện tích.", "40", "rectangle"),
        ("A rectangle has length 7 and width 3. Find its perimeter.", "20", "rectangle"),
    ],
)
def test_deterministic_word_problem_families(question: str, answer: str, visual: str):
    result = solve_math(question)
    assert result["answer"] == answer
    assert result["visual"]["type"] == visual
    assert result["verification"]["engine"] == "deterministic-word-rules"


@pytest.mark.parametrize(
    ("question", "answer", "visual", "minimum_steps"),
    [
        (
            "Một hộp có 5 bi đỏ, 4 bi xanh, 3 bi vàng. Lấy ngẫu nhiên 2 viên không hoàn lại. "
            "Tính xác suất hai viên khác màu. Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả "
            "và tạo trực quan hóa nếu phù hợp.",
            "47/66",
            "urn_probability",
            4,
        ),
        (
            "Khảo sát và vẽ đồ thị hàm số y = x^3 - 3x + 2, nêu cực trị và giao điểm với các trục. "
            "Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.",
            "y = x^3 - 3*x + 2",
            "graph",
            5,
        ),
        (
            "Tính tích phân từ 0 đến 1 của (3x^2 + 2x + 1) dx, trình bày nguyên hàm và kiểm tra "
            "bằng đạo hàm. Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.",
            "3",
            "integral_area",
            4,
        ),
        (
            "Trong mặt phẳng Oxy, cho A(-2;3), B(4;-1). Tìm trung điểm M và độ dài AB, giải thích "
            "bằng hệ trục tọa độ. Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.",
            "M = (1; 1), AB = 2*sqrt(13)",
            "coordinate_segment",
            4,
        ),
        (
            "Giải hệ phương trình: 2x + y = 7 và x - y = 2. Trình bày từng bước và kiểm tra lại nghiệm. "
            "Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.",
            "x = 3, y = 1",
            "coordinate_system",
            4,
        ),
        (
            "Giải phương trình 3(2x - 1) - 5 = 4x + 8, nêu điều kiện và thế nghiệm để kiểm tra. "
            "Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.",
            "x = 8",
            "balance",
            3,
        ),
        (
            "4 học sinh làm 60 tấm thiệp trong 3 giờ với năng suất như nhau. 6 học sinh làm trong 5 giờ "
            "được bao nhiêu tấm? Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.",
            "150",
            "rate_grid",
            4,
        ),
        (
            "Hàng đầu có 18 ghế, mỗi hàng sau hơn hàng trước 2 ghế. Tính số ghế hàng 15 và tổng số ghế "
            "của 15 hàng. Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.",
            "Hàng 15: 46 ghế; tổng 15 hàng: 480 ghế",
            "arithmetic_sequence",
            5,
        ),
    ],
)
def test_reported_math_failures_return_verified_steps_and_visuals(
    question: str, answer: str, visual: str, minimum_steps: int
):
    from app.math_tutor import looks_like_math

    assert looks_like_math(question) is True
    result = solve_math(question)
    assert result["status"] == "verified" and result["verification"]["passed"] is True
    assert result["answer"] == answer
    assert result["visual"]["type"] == visual
    assert len(result["steps"]) >= minimum_steps


async def test_math_page_and_api_are_served(monkeypatch):
    from app.api import math_tutor as api

    monkeypatch.setattr(api, "_rate_limit_ip", lambda *args, **kwargs: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        page = await client.get("/web/math")
        chat_page = await client.get("/web/chat")
        literature_page = await client.get("/web/literature")
        english_page = await client.get("/web/english")
        coach_app = await client.get("/task1-coach-pages/index.html")
        coach_css = await client.get(
            "/task1-coach-pages/_expo/static/css/styles-9dfbbbb4c32267d231b72d08cbe1fb53.css"
        )
        script = await client.get("/web/math/app.js")
        motion_css = await client.get("/web/math/upload.css")
        response = await client.post(
            "/web/math/solve", json={"question": "4x = 28", "curriculum": "vi", "grade": 7}
        )
        unsafe = await client.post("/web/math/solve", json={"question": "x = open('/etc/passwd')"})
        math_route = await client.post("/web/math/route", json={"question": "3x + 5 = 20"})
        chat_route = await client.post("/web/math/route", json={"question": "Transformer là gì?"})
        english_route = await client.post(
            "/web/math/route", json={"question": "Giải bài tiếng Anh: She ___ every day."}
        )
        writing_route = await client.post(
            "/web/math/route", json={"question": "Lập dàn ý nghị luận xã hội lớp 9 về lòng biết ơn"}
        )
    assert page.status_code == 200 and "StudyScope" in page.text
    assert "Phòng thi Toán" in page.text and "Phòng thi Ngữ văn" in page.text
    assert "Phòng thi Tiếng Anh" in page.text and "BỘ ĐỀ LUYỆN TẬP" in page.text
    assert chat_page.status_code == 200 and 'data-route="chat"' in chat_page.text
    assert literature_page.status_code == 200 and "2.500–3.000 chữ" in literature_page.text
    assert english_page.status_code == 200 and "/task1-coach-pages/index.html" in english_page.text
    assert coach_app.status_code == 200 and "Task 1 Coach" in coach_app.text
    assert coach_css.status_code == 200 and len(coach_css.content) > 50_000
    assert script.status_code == 200 and "setupSolutionAnimation" in script.text
    assert "visual-trace" in script.text and "step-state" in script.text
    assert "urn_probability" in script.text and "rate_grid" in script.text
    assert "arithmetic_sequence" in script.text and "visual-grow" in script.text
    assert "coordinate_segment" in script.text and "visual-move" in script.text
    assert "buildPodcastPlayer" in script.text and "/web/tts/synthesize" in script.text
    assert "english-podcast" in script.text and "literature-podcast" in script.text
    assert "written-answer" in script.text and "FULL WRITTEN ANSWER" in script.text
    assert "pageConfig" in script.text and "location.pathname.endsWith" in script.text
    assert "reply.mode || body.mode" in script.text and "Nguồn:" in script.text
    assert motion_css.status_code == 200 and "@keyframes visualDraw" in motion_css.text
    assert "@keyframes visualMove" in motion_css.text
    assert "@keyframes visualGrow" in motion_css.text
    assert "prefers-reduced-motion" in motion_css.text
    assert response.status_code == 200 and response.json()["answer"] == "x = 7"
    assert unsafe.status_code == 422 and unsafe.json()["detail"]["safe"] is True
    assert math_route.json()["mode"] == "math" and math_route.json()["solution"]["answer"] == "x = 5"
    assert chat_route.json() == {"mode": "chat", "math_detected": False}
    assert english_route.json()["mode"] == "english"
    assert writing_route.json()["mode"] == "writing"


async def test_math_image_is_ocr_then_deterministically_verified(monkeypatch):
    from app.api import math_tutor as api
    from app.vision import client as vision_module

    class FakeVision:
        def __init__(self, settings):  # type: ignore[no-untyped-def]
            pass

        async def analyze(self, image: bytes, *, question: str | None = None):  # type: ignore[no-untyped-def]
            assert image == b"fake-image" and question
            return {"ocr": {"text": "Câu 1: 5x + 5 = 30", "lines": 1}, "objects": []}

    monkeypatch.setattr(api, "_rate_limit_ip", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api, "get_settings", lambda: get_settings().model_copy(update={"vision_enabled": True})
    )
    monkeypatch.setattr(vision_module, "VisionClient", FakeVision)
    payload = {
        "question": "Giải đề trong ảnh",
        "image_base64": base64.b64encode(b"fake-image").decode(),
        "curriculum": "vi",
        "grade": 7,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/web/math/read-image", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "x = 5" and body["verification"]["passed"] is True
    assert body["image_analysis"]["extracted_text"] == "Câu 1: 5x + 5 = 30"


async def test_math_image_extracts_repeated_arithmetic_from_noisy_ocr(monkeypatch):
    from app.api import math_tutor as api
    from app.vision import client as vision_module

    class FakeVision:
        def __init__(self, settings):  # type: ignore[no-untyped-def]
            pass

        async def analyze(self, image: bytes, *, question: str | None = None):  # type: ignore[no-untyped-def]
            return {
                "ocr": {"text": "4+3=7, 4 chấm + 3 chấm, 4+3=7, 4 3 7", "lines": 1},
                "objects": [],
            }

    monkeypatch.setattr(api, "_rate_limit_ip", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        api, "get_settings", lambda: get_settings().model_copy(update={"vision_enabled": True})
    )
    monkeypatch.setattr(vision_module, "VisionClient", FakeVision)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/web/math/read-image",
            json={"image_base64": base64.b64encode(b"fake-image").decode(), "grade": 1},
        )
    assert response.status_code == 200
    assert response.json()["answer"] == "7"
