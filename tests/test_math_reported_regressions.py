from __future__ import annotations

import httpx
import pytest

from app.main import app
from app.math_tutor import looks_like_math, solve_math

MIXTURE = (
    "Nếu trộn dung dịch muối nồng độ 10% với dung dịch muối nồng độ 60% "
    "để được 250 ml dung dịch muối nồng độ 40% thì cần lấy bao nhiêu mililít dung dịch mỗi loại?"
)
VEHICLES = (
    "Một ô tô và một xe máy cùng khởi hành từ địa điểm A và đi đến địa điểm B. "
    "Do vận tốc của ô tô lớn hơn vận tốc của xe máy là 20 km/h nên ô tô đến B "
    "sớm hơn xe máy 30 phút. Biết quãng đường AB dài 60 km, tính vận tốc của mỗi xe."
)


@pytest.mark.parametrize(
    ("question", "answer", "visual", "method"),
    [
        (
            MIXTURE,
            "Dung dịch 10%: 100 ml; dung dịch 60%: 150 ml",
            "mixture",
            "volume_and_solute_conservation",
        ),
        (
            VEHICLES,
            "Xe máy: 40 km/h; ô tô: 60 km/h",
            "motion_comparison",
            "distance_time_equation_and_substitution",
        ),
    ],
)
def test_reported_word_problems_are_routed_solved_and_visualized(
    question: str, answer: str, visual: str, method: str
):
    assert looks_like_math(question)
    solved = solve_math(question)
    assert solved["answer"] == answer
    assert solved["visual"]["type"] == visual
    assert solved["verification"]["method"] == method
    assert len(solved["steps"]) >= 6


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("(x + 2)(x2 – x + 3) = x3 + 8;", "x ∈ {-2, 1}"),
        ("(x2 – 3x)2 – (x – 4)2 = 0.", "x ∈ {2, 1 - sqrt(5), 1 + sqrt(5)}"),
        ("(x^2 – 3x)^2 – (x – 4)^2 = 0.", "x ∈ {2, 1 - sqrt(5), 1 + sqrt(5)}"),
    ],
)
def test_lost_superscripts_are_recovered_and_solution_has_written_algebra(
    question: str, answer: str
):
    solved = solve_math(question)
    details = "\n".join(step["detail"] for step in solved["steps"])
    assert solved["answer"] == answer
    assert "Phân tích thành nhân tử" in [step["title"] for step in solved["steps"]]
    assert "= 0" in details
    assert "vế trái − vế phải = 0" in details
    assert solved["visual"]["markers"]


async def test_public_route_uses_deterministic_math_instead_of_general_chat(monkeypatch):
    from app.api import math_tutor as api

    monkeypatch.setattr(api, "_rate_limit_ip", lambda *args, **kwargs: None)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        mixture = await client.post("/web/math/route", json={"question": MIXTURE})
        vehicles = await client.post("/web/math/route", json={"question": VEHICLES})
    assert mixture.json()["mode"] == "math"
    assert vehicles.json()["mode"] == "math"
