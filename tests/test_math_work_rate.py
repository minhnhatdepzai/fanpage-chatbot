"""Hồi quy đề máy xúc thật và dữ kiện chưa được mô hình hóa, dùng đáp án tính độc lập."""

import random
from fractions import Fraction

import httpx
import pytest

from app.main import app
from app.math_tutor import MathTutorError, solve_math
from tests.test_answer_verification import _run

EXCAVATOR = (
    "Một đội máy xúc được thuê đào 20000 m³ đất để mở rộng hồ Dầu Tiếng. "
    "Ban đầu đội dự định mỗi ngày đào một lượng đất cố định để hoàn thành công việc, "
    "nhưng khi đã đào được 5000 m³ thì đội được tăng cường thêm máy móc nên mỗi ngày "
    "đào thêm được 100m³, do đó đã hoàn thành công việc trong 35 ngày. "
    "Hỏi ban đầu đội dự định mỗi ngày đào bao nhiêu mét khối đất?"
)


@pytest.mark.parametrize(
    "question",
    [
        EXCAVATOR,
        EXCAVATOR.replace("20000", "20.000").replace("5000", "5.000"),
        EXCAVATOR.replace("m³", "m3"),
        EXCAVATOR.replace("m³", "mét khối"),
        EXCAVATOR
        + " Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp.",
    ],
)
def test_real_excavation_question_is_rate_not_addition(question):
    solved = solve_math(question)
    assert solved["values"] == {
        "initial_rate": "500",
        "later_rate": "600",
        "first_days": "10",
        "second_days": "25",
    }
    assert solved["verification"]["checked_data"] == 4
    assert solved["verification"]["residuals"] == ["0"] * 4
    assert solved["visual"]["type"] == "work_timeline"
    assert len(solved["steps"]) == 7
    # Original quantities/time checked independently of the production polynomial.
    rate = Fraction(solved["values"]["initial_rate"])
    assert Fraction(5000, rate) + Fraction(15000, rate + 100) == 35


def test_two_stage_variants_recover_independently_constructed_rate():
    rng = random.Random(20261006)
    for _ in range(40):
        initial, increase, d1, d2 = [rng.randint(1, 20) * 10 for _ in range(4)]
        first = initial * d1
        total = first + (initial + increase) * d2
        question = (
            f"Một đội máy xúc đào {total} m³ đất. Ban đầu dự định mỗi ngày đào một lượng cố định. "
            f"Khi đã đào được {first} m³ thì mỗi ngày đào thêm được {increase} m³. "
            f"Đội hoàn thành công việc trong {d1 + d2} ngày. Hỏi ban đầu mỗi ngày đào bao nhiêu?"
        )
        values = solve_math(question)["values"]
        assert Fraction(values["initial_rate"]) == initial
        assert Fraction(values["later_rate"]) == initial + increase
        assert Fraction(values["first_days"]) == d1
        assert Fraction(values["second_days"]) == d2


@pytest.mark.parametrize(
    "question",
    [
        "Lan có 7 quả táo, mẹ cho thêm 5 quả rồi Lan ăn 3 quả. Hỏi còn bao nhiêu?",
        "Lan có 7 quả táo trong 2 giỏ, mẹ cho thêm 5 quả. Hỏi có bao nhiêu?",
        "Lan có 7 quả táo, mẹ cho thêm 5 quả. Đã bán 2 quả. Hỏi còn bao nhiêu?",
        "Lan có 7 quả táo, mẹ cho thêm 5 quả cam. Hỏi có bao nhiêu quả táo?",
        "13 viên kẹo chia đều cho 3 người. Mỗi người được bao nhiêu?",
        "Hình chữ nhật có chiều dài 8 cm, chiều rộng 5 cm. Tính diện tích và chu vi.",
        "Hình chữ nhật có chiều dài -8 cm, chiều rộng 5 cm. Tính diện tích.",
        "Một hộp có 5 bi đỏ, 4 bi xanh, 3 bi vàng. Lấy ngẫu nhiên 2 viên không hoàn lại. Bỏ thêm 1 viên trước khi lấy. Tính xác suất hai viên khác màu.",
        "Có 48 chiếc bút chia đều vào 6 hộp rồi bán 2 hộp. Hỏi còn bao nhiêu bút?",
        "Thư viện có 38 truyện, cho mượn 16 rồi nhận thêm 9 truyện và mất 4 truyện. Còn bao nhiêu?",
        EXCAVATOR.replace("35 ngày", "35 ngày và nghỉ 2 ngày"),
        EXCAVATOR.replace("35 ngày", "0 ngày"),
        EXCAVATOR.replace("5000 m³", "25000 m³"),
        EXCAVATOR.replace("35 ngày", "35 giờ"),
    ],
)
def test_no_verified_answer_when_facts_do_not_fit_supported_model(question):
    with pytest.raises(MathTutorError):
        solve_math(question)


async def test_math_route_refuses_unmodeled_data_without_ai_final_answer(monkeypatch):
    from app.api import math_tutor as api

    monkeypatch.setattr(api, "_rate_limit_ip", lambda *a, **kw: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        exact = await client.post("/web/math/route", json={"question": EXCAVATOR})
        incomplete = await client.post("/web/math/route", json={"question": EXCAVATOR + " Nghỉ 2 ngày."})
    assert exact.json()["solution"]["values"]["initial_rate"] == "500"
    assert incomplete.json()["mode"] == "math_unverified"
    assert "solution" not in incomplete.json()
    result, llm = await _run(EXCAVATOR + " Nghỉ 2 ngày.", lambda _: "Đáp án chắc chắn là 5100.")
    assert result["mode"] == "math_unverified" and not llm.calls
    assert "5100" not in " ".join(result["reply_parts"])


async def test_widget_chat_uses_same_exact_work_solver():
    result, llm = await _run(EXCAVATOR, lambda _: "Đáp án là 5100.")
    assert result["mode"] == "math_calc" and not llm.calls
    assert "500 m³/ngày" in " ".join(result["reply_parts"])


def test_ocr_does_not_take_one_expression_from_unverified_word_problem():
    from app.api.math_tutor import _math_candidates

    question = EXCAVATOR + " Một phép tính ghi bên cạnh: 5000 + 100."
    assert _math_candidates(question) == [question]
