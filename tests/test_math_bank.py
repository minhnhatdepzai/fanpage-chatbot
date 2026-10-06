"""Kho đề thực tế là cổng phát hành: đủ câu hỏi phụ, đáp án độc lập, và dữ liệu hình."""

from __future__ import annotations

import json
import random
import re
from fractions import Fraction
from pathlib import Path

import httpx
import pytest

from app.main import app
from app.math_tutor import MathTutorError, looks_like_math, solve_math

BANK = Path(__file__).resolve().parents[1] / "app/web/static/study-bank.js"
PROMPTS = re.findall(r'prompt: "([^"\n]+)"', BANK.read_text().split("literature:")[0])
EXPECTED = [
    "12",
    "14 > 9; 8 + 3 < 12.",
    "31",
    "22",
    "Mỗi hộp: 8; cần thêm: 16",
    "Đã ăn: 3/8; còn lại: 5/8",
    "30",
    "3/2",
    "53 nghìn đồng",
    "Thể tích: 18/5 m³; nước: 2880 lít",
    "-1",
    "12 nhóm; mỗi nhóm 3 nam và 4 nữ",
    "18/5 km",
    "150",
    "x = 8",
    "12*x",
    "x = 3, y = 1",
    "BC = 10 cm; AH = 24/5 cm",
    "y = x^2 - 4*x + 3",
    "M = (1; 1), AB = 2*sqrt(13)",
    "Hàng 15: 46 ghế; tổng 15 hàng: 480 ghế",
    "47/66",
    "y = x^3 - 3*x + 2",
    "3",
]
SUFFIX = " Hãy giải thích thật rõ từng bước, kiểm tra lại kết quả và tạo trực quan hóa nếu phù hợp."


@pytest.mark.parametrize("index", range(24))
@pytest.mark.parametrize("suffix", ["", SUFFIX])
def test_real_bank_answers_are_complete(index: int, suffix: str):
    assert len(PROMPTS) == len(EXPECTED) == 24
    question = PROMPTS[index] + suffix
    assert looks_like_math(question)
    solved = solve_math(question)
    assert solved["answer"] == EXPECTED[index]
    assert solved["steps"] and solved["visual"]["type"]
    assert solved["verification"]["passed"]
    json.dumps(solved, allow_nan=False)


def test_generated_variants_match_independent_integer_and_fraction_oracles():
    rng = random.Random(20261006)
    for _ in range(60):
        first, change, rows = rng.randint(5, 30), rng.randint(1, 6), rng.randint(4, 24)
        q = f"Hàng đầu có {first} ghế, mỗi hàng sau hơn hàng trước {change} ghế. Tính số ghế hàng {rows} và tổng số ghế của {rows} hàng."
        solved = solve_math(q)
        seats = [first + i * change for i in range(rows)]
        assert solved["visual"]["nth_value"] == seats[-1]
        assert int(solved["visual"]["total"]) == sum(seats)
        a, b, c = (rng.randint(2, 14) for _ in range(3))
        q = f"Một hộp có {a} bi đỏ, {b} bi xanh, {c} bi vàng. Lấy ngẫu nhiên 2 viên không hoàn lại. Tính xác suất hai viên khác màu."
        # Ordered sequential draws give an independent oracle to unordered pair counting.
        probability = sum(
            Fraction(x, a + b + c) * Fraction(y, a + b + c - 1)
            for i, x in enumerate([a, b, c])
            for j, y in enumerate([a, b, c])
            if i != j
        )
        assert solve_math(q)["answer"] == str(probability)
        total, removed, added = rng.randint(40, 90), rng.randint(1, 30), rng.randint(1, 20)
        q = f"Thư viện lớp có {total} truyện, cho mượn {removed} truyện rồi nhận thêm {added} truyện. Thư viện còn bao nhiêu truyện?"
        assert solve_math(q)["answer"] == str(total - removed + added)
        groups, each, additional = rng.randint(2, 10), rng.randint(2, 10), rng.randint(1, 5)
        q = f"Có {groups * each} chiếc bút chia đều vào {groups} hộp. Mỗi hộp có bao nhiêu bút? Nếu thêm {additional} hộp như vậy thì cần thêm bao nhiêu bút?"
        assert solve_math(q)["values"] == {"per_group": str(each), "additional": str(additional * each)}


def test_integral_does_not_apply_ftc_across_singularities():
    with pytest.raises(MathTutorError):
        solve_math("Tích phân từ -1 đến 1 của 1/x^2 dx")


@pytest.mark.parametrize(
    "question",
    [
        "Có 49 chiếc bút chia đều vào 6 hộp. Mỗi hộp có bao nhiêu bút?",
        "Có -48 chiếc bút chia đều vào 6 hộp. Mỗi hộp có bao nhiêu bút?",
        "Có 48 chiếc bút chia đều vào 0 hộp. Mỗi hộp có bao nhiêu bút?",
        "Bản đồ tỉ lệ 1:0, hai điểm cách nhau 2 cm. Tính khoảng cách thực.",
    ],
)
def test_word_problem_does_not_verify_impossible_discrete_or_scale_data(question):
    with pytest.raises(MathTutorError):
        solve_math(question)


async def test_page_uses_content_versions_to_prevent_stale_renderer():
    import hashlib

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        page = await client.get("/web/math")
        scene = await client.get("/web/math/scenes.js")
        script = await client.get("/web/math/app.js")
    assert page.headers["cache-control"] == "no-store"
    assert script.headers["cache-control"] == "no-cache"
    assert f"app.js?v={hashlib.sha256(script.content).hexdigest()[:12]}" in page.text
    assert f"scenes.js?v={hashlib.sha256(scene.content).hexdigest()[:12]}" in page.text
    assert page.text.index("scenes.js?v=") < page.text.index("app.js?v=")
