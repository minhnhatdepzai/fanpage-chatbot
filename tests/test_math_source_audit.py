from __future__ import annotations

from pathlib import Path

import pytest

from app.math_tutor import MathTutorError, solve_math
from app.math_tutor.validation import validate_verified_solution
from evaluation.math_source_audit import discover_links, evaluate_cases, load_cases

ROOT = Path(__file__).resolve().parents[1]


def test_discovery_keeps_only_same_grade_math_links_and_deduplicates():
    html = """
      <a href="/toan-6/bai-1.jsp"><span>Toán lớp 6</span> Bài 1</a>
      <a href="/toan-6/bai-1.jsp">Bài Toán lớp 6 trùng</a>
      <a href="/toan-7/bai-1.jsp">Toán lớp 7</a>
      <a href="https://evil.example/toan-6">Toán lớp 6 ngoài nguồn</a>
      <a href="/soan-van-6/">Soạn văn lớp 6</a>
    """
    links = discover_links(html, "https://www.vietjack.com/series/lop-6.jsp", 6)
    assert len(links) == 1
    assert links[0].url == "https://www.vietjack.com/toan-6/bai-1.jsp"
    assert links[0].grade == 6


def test_draft_audit_has_one_passing_case_per_grade():
    cases = load_cases(ROOT / "evaluation/datasets/math_vietjack_audit_draft_v0.yaml")
    report = evaluate_cases(cases)
    assert report["summary"]["total"] == 12
    assert report["summary"]["passed"] == 12
    assert set(report["summary"]["by_grade"]) == {str(grade) for grade in range(1, 13)}


def test_visual_contract_rejects_non_finite_or_missing_payload():
    valid = solve_math("27 + 18")
    valid["visual"]["change"] = float("nan")
    with pytest.raises(MathTutorError, match="không hữu hạn"):
        validate_verified_solution(valid)

    valid = solve_math("27 + 18")
    del valid["visual"]["change"]
    with pytest.raises(MathTutorError, match="thiếu dữ kiện"):
        validate_verified_solution(valid)


def test_proportional_notebook_price_is_exact_and_visualized():
    solved = solve_math(
        "Mua 4 quyển vở hết 28 nghìn đồng. Với cùng đơn giá, 7 quyển vở giá bao nhiêu nghìn đồng?",
        grade=7,
    )
    assert solved["answer"] == "49 nghìn đồng"
    assert solved["values"] == {"unit_price": "7", "target_total": "49"}
    assert solved["verification"]["method"] == "unit_rate_and_cross_product"
    assert solved["visual"]["type"] == "bar_chart"
