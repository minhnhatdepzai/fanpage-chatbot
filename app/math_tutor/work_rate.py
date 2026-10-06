"""Bài năng suất hai giai đoạn: mô hình đủ dữ kiện, thế lại thời gian và khối lượng."""

from __future__ import annotations

import re
from typing import Any

import sympy as sp


def solve_two_stage_work(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    from app.math_tutor.solver import (
        MathTutorError,
        _fold_text,
        _normalize,
        _pretty,
        _require_number_coverage,
        _source,
    )

    text = _fold_text(_normalize(question)).replace("m3", "m³")
    if not (
        "moi ngay" in text
        and re.search(r"\bdao\b", text)
        and re.search(r"\b(?:du dinh|du kien|ke hoach)\b", text)
    ):
        return None
    number = r"(\d{1,3}(?:\.\d{3})+|\d+(?:[.,]\d+)?)"
    volume = r"\s*(?:m³|met khoi)"
    total_match = re.search(rf"(?:dao|san xuat|lam|tong(?: khoi luong)?(?: la|:)?)\s+{number}{volume}", text)
    first_match = re.search(
        rf"(?:khi|sau khi)\s+(?:da\s+)?(?:dao|lam|san xuat)\s+(?:duoc\s+)?{number}{volume}", text
    )
    increment_match = re.search(
        rf"moi ngay\s+(?:dao|lam|san xuat)\s+(?:them\s+(?:duoc\s+)?|tang\s+them\s+){number}{volume}", text
    )
    days_match = re.search(rf"hoan thanh\s+(?:cong viec\s+)?(?:trong|sau)\s+{number}\s+ngay", text)
    if not all([total_match, first_match, increment_match, days_match]):
        raise MathTutorError(
            "Bài năng suất chưa tách đủ tổng khối lượng, mốc thay đổi, mức tăng mỗi ngày và tổng thời gian."
        )
    if not re.search(r"(?:hoi|tinh|tim).*?ban dau.*?(?:moi ngay|nang suat)", text):
        raise MathTutorError("Chưa xác định yêu cầu tìm năng suất ban đầu; cần làm rõ đại lượng muốn tính.")
    matches = [total_match, first_match, increment_match, days_match]
    _require_number_coverage(text, matches)

    def quantity(match: re.Match[str]) -> sp.Rational:
        raw = match.group(1)
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
            raw = raw.replace(".", "")
        return sp.Rational(raw.replace(",", "."))

    total, first, increase, days = [quantity(m) for m in matches]
    if not 0 < first < total or increase <= 0 or days <= 0:
        raise MathTutorError("Khối lượng mỗi giai đoạn, mức tăng và tổng thời gian phải dương.")
    x = sp.Symbol("x", real=True)
    remaining = total - first
    equation = first / x + remaining / (x + increase) - days
    polynomial = sp.expand(days * x * (x + increase) - first * (x + increase) - remaining * x)
    roots = sp.solve(polynomial, x)
    valid = [r for r in roots if r.is_positive is True and sp.simplify(equation.subs(x, r)) == 0]
    if len(valid) != 1:
        raise MathTutorError("Chưa xác định duy nhất một năng suất dương thỏa mọi dữ kiện.")
    rate = valid[0]
    faster = rate + increase
    first_days = sp.simplify(first / rate)
    second_days = sp.simplify(remaining / faster)
    residuals = [
        sp.simplify(rate * first_days - first),
        sp.simplify(faster * second_days - remaining),
        sp.simplify(first_days + second_days - days),
        sp.simplify(first + remaining - total),
    ]
    if any(r != 0 for r in residuals):
        raise MathTutorError("Nghiệm không vượt qua kiểm tra độc lập tất cả điều kiện của đề.")
    rejected = [r for r in roots if r not in valid]
    equation_text = f"{_pretty(first)}/x + {_pretty(remaining)}/(x + {_pretty(increase)}) = {_pretty(days)}"
    return {
        "status": "verified",
        "problem": question,
        "answer": f"Ban đầu: {_pretty(rate)} m³/ngày; sau tăng cường: {_pretty(faster)} m³/ngày",
        "answer_latex": sp.latex(rate),
        "topic": "Năng suất hai giai đoạn",
        "grade": grade or 9,
        "steps": [
            {
                "title": "Đọc đủ dữ kiện",
                "detail": f"Tổng {_pretty(total)} m³, đã đào {_pretty(first)} m³ thì tăng {_pretty(increase)} m³/ngày; toàn bộ kéo dài {_pretty(days)} ngày.",
            },
            {
                "title": "Đặt ẩn và điều kiện",
                "detail": f"Gọi x > 0 là năng suất ban đầu (m³/ngày). Sau tăng cường là x + {_pretty(increase)}; không cộng mức tăng này vào khối lượng đã đào.",
            },
            {
                "title": "Tính thời gian mỗi giai đoạn",
                "detail": f"Còn lại {_pretty(total)} − {_pretty(first)} = {_pretty(remaining)} m³. Thời gian = khối lượng ÷ năng suất: t₁ = {_pretty(first)}/x, t₂ = {_pretty(remaining)}/(x + {_pretty(increase)}).",
            },
            {"title": "Lập phương trình tổng thời gian", "detail": equation_text + "."},
            {
                "title": "Giải và chọn nghiệm hợp lệ",
                "detail": f"Nhân hai vế với x(x + {_pretty(increase)}), được {_pretty(polynomial)} = 0. Nghiệm x = {_pretty(rate)}; loại {', '.join(_pretty(r) for r in rejected) or 'không có nghiệm khác'} vì không thỏa điều kiện năng suất dương.",
            },
            {
                "title": "Thế lại từng giai đoạn",
                "detail": f"Giai đoạn 1: {_pretty(first)} ÷ {_pretty(rate)} = {_pretty(first_days)} ngày. Giai đoạn 2: {_pretty(remaining)} ÷ {_pretty(faster)} = {_pretty(second_days)} ngày. Tổng {_pretty(first_days)} + {_pretty(second_days)} = {_pretty(days)} ngày, đúng đề.",
            },
            {
                "title": "Đối chiếu khối lượng và kết luận",
                "detail": f"{_pretty(rate)} × {_pretty(first_days)} + {_pretty(faster)} × {_pretty(second_days)} = {_pretty(total)} m³. Ban đầu đào {_pretty(rate)} m³/ngày.",
            },
        ],
        "values": {
            "initial_rate": _pretty(rate),
            "later_rate": _pretty(faster),
            "first_days": _pretty(first_days),
            "second_days": _pretty(second_days),
        },
        "visual": {
            "type": "work_timeline",
            "total": float(total),
            "first": float(first),
            "remaining": float(remaining),
            "initial_rate": float(rate),
            "later_rate": float(faster),
            "first_days": float(first_days),
            "second_days": float(second_days),
            "days": float(days),
        },
        "verification": {
            "engine": "exact-two-stage-work",
            "passed": True,
            "method": "positive_root_and_all_constraints",
            "equation": equation_text,
            "residuals": [str(r) for r in residuals],
            "checked_data": 4,
        },
        "sources": [_source(curriculum)],
        "warnings": [],
    }
