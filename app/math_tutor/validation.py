"""Hợp đồng phát hành cho lời giải toán đã kiểm chứng và kế hoạch trực quan hóa.

Bộ giải chỉ được trả trạng thái ``verified`` khi cả đáp số, các bước và payload dùng
để dựng SVG đều đủ dữ kiện cơ bản. Đây là lớp chặn lỗi hiển thị; nó không thay thế
chứng minh toán học riêng của từng họ bài.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from app.math_tutor.solver import MathTutorError

_VISUAL_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "work_timeline": (
        "total",
        "first",
        "remaining",
        "days",
        "initial_rate",
        "later_rate",
        "first_days",
        "second_days",
    ),
    "arithmetic_sequence": ("first", "difference", "nth", "nth_value", "count", "total", "rows"),
    "groups": ("groups", "per_group"),
    "right_triangle": ("ab", "ac", "bc", "ah"),
    "cuboid": ("length", "width", "height", "fill", "litres"),
    "bar_chart": ("bars",),
    "number_journey": ("positions",),
    "number_comparison": ("markers",),
    "scale_map": ("map_cm", "factor", "km"),
    "integral_area": ("expression", "lower", "upper", "points", "result"),
    "fraction_sum": ("numerator", "denominator"),
    "formula": ("expression", "result"),
    "fraction": ("numerator", "denominator"),
    "number_line": ("start", "change"),
    "urn_probability": ("groups", "draw", "favorable", "total", "probability"),
    "rate_grid": (
        "base_people",
        "base_hours",
        "base_output",
        "target_people",
        "target_hours",
        "target_output",
    ),
    "rectangle": ("length", "width", "result"),
    "balance": ("left", "right", "solution"),
    "graph": ("points",),
    "coordinate_system": ("lines", "solution"),
    "coordinate_segment": ("points",),
    "mixture": (
        "low_percent",
        "high_percent",
        "target_percent",
        "low_volume",
        "high_volume",
        "total_volume",
    ),
    "motion_comparison": (
        "distance",
        "slower_speed",
        "faster_speed",
        "slower_time",
        "faster_time",
        "saved_minutes",
    ),
}


def _is_missing(value: Any) -> bool:
    return value is None or isinstance(value, str) and not value.strip()


def _assert_finite(value: Any, path: str = "visual") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MathTutorError(f"Dữ liệu trực quan không hữu hạn tại {path}.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _validate_point(point: Any, path: str) -> None:
    if not isinstance(point, Mapping) or not {"x", "y"}.issubset(point):
        raise MathTutorError(f"Điểm tọa độ tại {path} thiếu x hoặc y.")
    _assert_finite(point, path)


def validate_visual_plan(visual: Any) -> None:
    """Từ chối payload có thể tạo hình trống, NaN hoặc thiếu dữ kiện bắt buộc."""
    if not isinstance(visual, Mapping):
        raise MathTutorError("Lời giải chưa có kế hoạch trực quan hóa hợp lệ.")
    kind = visual.get("type")
    required = _VISUAL_REQUIRED_FIELDS.get(kind)
    if required is None:
        raise MathTutorError(f"Loại trực quan hóa '{kind}' chưa có bộ dựng được kiểm tra.")
    missing = [field for field in required if field not in visual or _is_missing(visual[field])]
    if missing:
        raise MathTutorError("Kế hoạch trực quan hóa thiếu dữ kiện: " + ", ".join(missing) + ".")
    _assert_finite(visual)

    if kind in {"fraction", "fraction_sum"} and int(visual["denominator"]) <= 0:
        raise MathTutorError("Mẫu số trong hình phân số phải lớn hơn 0.")
    if kind == "groups" and (float(visual["groups"]) <= 0 or float(visual["per_group"]) <= 0):
        raise MathTutorError("Hình nhóm cần số nhóm và số phần tử dương.")
    if kind == "bar_chart" and not visual["bars"]:
        raise MathTutorError("Biểu đồ cột cần ít nhất một cột.")
    if kind in {"graph", "coordinate_segment", "integral_area"}:
        points = visual["points"]
        if not isinstance(points, list) or len(points) < 2:
            raise MathTutorError("Đồ thị cần ít nhất hai điểm hữu hạn.")
        for index, point in enumerate(points):
            _validate_point(point, f"visual.points[{index}]")
    if kind == "coordinate_system":
        lines = visual["lines"]
        if not isinstance(lines, list) or not lines:
            raise MathTutorError("Hệ trục tọa độ cần ít nhất một đường.")
        for line_index, points in enumerate(lines):
            if not isinstance(points, list) or len(points) < 2:
                raise MathTutorError(f"Đường {line_index + 1} cần ít nhất hai điểm.")
            for point_index, point in enumerate(points):
                _validate_point(point, f"visual.lines[{line_index}][{point_index}]")
    if kind == "work_timeline":
        if not math.isclose(
            float(visual["first"]) + float(visual["remaining"]),
            float(visual["total"]),
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise MathTutorError("Hai giai đoạn của hình công việc không cộng đúng tổng khối lượng.")
        if not math.isclose(
            float(visual["first_days"]) + float(visual["second_days"]),
            float(visual["days"]),
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise MathTutorError("Hai giai đoạn của hình công việc không cộng đúng tổng thời gian.")
    if kind == "mixture":
        if not math.isclose(
            float(visual["low_volume"]) + float(visual["high_volume"]),
            float(visual["total_volume"]),
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise MathTutorError("Hai phần dung dịch không cộng đúng thể tích sau khi trộn.")
        weighted = (
            float(visual["low_percent"]) * float(visual["low_volume"])
            + float(visual["high_percent"]) * float(visual["high_volume"])
        ) / float(visual["total_volume"])
        if not math.isclose(weighted, float(visual["target_percent"]), rel_tol=0, abs_tol=1e-9):
            raise MathTutorError("Hình pha trộn không bảo toàn nồng độ.")
    if kind == "motion_comparison":
        if min(
            float(visual["distance"]),
            float(visual["slower_speed"]),
            float(visual["faster_speed"]),
        ) <= 0:
            raise MathTutorError("Hình chuyển động cần quãng đường và vận tốc dương.")
        saved = (float(visual["slower_time"]) - float(visual["faster_time"])) * 60
        if not math.isclose(saved, float(visual["saved_minutes"]), rel_tol=0, abs_tol=1e-9):
            raise MathTutorError("Hình chuyển động không khớp độ chênh thời gian.")


def validate_verified_solution(solution: dict[str, Any]) -> dict[str, Any]:
    """Kiểm tra hợp đồng chung trước khi một lời giải được phép mang nhãn verified."""
    verification = solution.get("verification")
    if solution.get("status") != "verified" or not isinstance(verification, Mapping):
        raise MathTutorError("Lời giải chưa có trạng thái kiểm chứng hợp lệ.")
    if verification.get("passed") is not True:
        raise MathTutorError("Lời giải chưa vượt qua bước kiểm chứng độc lập.")
    if _is_missing(solution.get("answer")):
        raise MathTutorError("Lời giải chưa có đáp án.")
    steps = solution.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise MathTutorError("Lời giải cần ít nhất hai bước rõ ràng.")
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping) or _is_missing(step.get("title")) or _is_missing(step.get("detail")):
            raise MathTutorError(f"Bước giải {index + 1} thiếu tiêu đề hoặc nội dung.")
    validate_visual_plan(solution.get("visual"))
    return solution
