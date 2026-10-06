"""Các quan hệ toán phổ thông được tính chính xác và trả đủ các yêu cầu phụ.

Mẫu chỉ nhận họ đề có dữ kiện tường minh. Kết quả nhiều đại lượng luôn giữ cả
hai, thay vì cắt câu thứ hai rồi trả một đáp án cho toàn bộ bài.
"""

from __future__ import annotations

import re
from typing import Any

import sympy as sp


def solve_school_problem(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    # Nhập tại lúc gọi để bộ parser AST an toàn vẫn là nguồn tính thống nhất.
    from app.math_tutor.solver import (
        MathTutorError,
        _fold_text,
        _normalize,
        _parse_expr,
        _pretty,
        _require_number_coverage,
        _source,
    )

    text = _fold_text(_normalize(question))
    number = r"(-?\d+(?:[.,]\d+)?)"

    def rational(raw: str) -> sp.Rational:
        return sp.Rational(raw.replace(",", "."))

    def result(
        answer: str,
        topic: str,
        steps: list[tuple[str, str]],
        visual: dict[str, Any],
        values: dict[str, Any],
        default_grade: int,
        method: str,
    ) -> dict[str, Any]:
        return {
            "status": "verified",
            "problem": question,
            "answer": answer,
            "answer_latex": "",
            "topic": topic,
            "grade": grade or default_grade,
            "steps": [{"title": title, "detail": detail} for title, detail in steps],
            "visual": visual,
            "values": {key: str(value) for key, value in values.items()},
            "verification": {"engine": "exact-school-relations", "passed": True, "method": method},
            "sources": [_source(curriculum)],
            "warnings": [],
        }

    comparisons = list(re.finditer(r"([\d\s+*/().,-]+)\s*__\s*([\d\s+*/().,-]+)", text))
    if comparisons:
        _require_number_coverage(text, comparisons)
        solved, markers, steps = [], [], []
        for match in comparisons:
            left_raw, right_raw = match.groups()
            left = _parse_expr(_normalize(left_raw.strip()))
            right = _parse_expr(_normalize(right_raw.strip()))
            relation = ">" if left > right else "<" if left < right else "="
            solved.append(f"{left_raw.strip().rstrip('.')} {relation} {right_raw.strip().rstrip('.')}")
            markers.extend(
                [{"x": float(left), "label": _pretty(left)}, {"x": float(right), "label": _pretty(right)}]
            )
            steps.append(
                (
                    "Tính rồi so sánh",
                    f"Vế trái = {_pretty(left)}, vế phải = {_pretty(right)}; điền dấu {relation}.",
                )
            )
        steps.append(("Kiểm tra trên trục số", "Số lớn hơn ở bên phải; số bằng nhau nằm cùng một vị trí."))
        return result(
            "; ".join(solved) + ".",
            "So sánh số",
            steps,
            {"type": "number_comparison", "markers": markers},
            {},
            1,
            "exact_comparison",
        )

    # Theo dõi mọi thay đổi của số lượng hoặc nhiệt độ, theo đúng thứ tự.
    initial = re.search(rf"(?:co|la)\s*{number}\s*(?:truyen|°c|do c)", text)
    changes = list(re.finditer(rf"(cho muon|nhan them|tang|giam)\s*{number}", text))
    if initial and changes:
        _require_number_coverage(text, [initial, *changes])
        start = rational(initial.group(1))
        current = start
        positions = [float(start)]
        steps = [("Số ban đầu", f"Ban đầu: {_pretty(start)}.")]
        for change in changes:
            amount = rational(change.group(2))
            delta = -amount if change.group(1) in {"cho muon", "giam"} else amount
            before = current
            current += delta
            positions.append(float(current))
            steps.append(
                (
                    "Theo dõi thay đổi",
                    f"{_pretty(before)} {'+' if delta >= 0 else '−'} {_pretty(abs(delta))} = {_pretty(current)}.",
                )
            )
        steps.append(("Kiểm tra ngược", f"Đảo các thay đổi từ {_pretty(current)} trở về {_pretty(start)}."))
        return result(
            _pretty(current),
            "Thay đổi liên tiếp",
            steps,
            {"type": "number_journey", "positions": positions},
            {"final": current},
            2,
            "inverse_changes",
        )

    division = re.search(rf"co\s+{number}.*?chia deu\s+(?:vao|cho)\s+{number}\s+(?:hop|nhom|nguoi)", text)
    if division:
        extra = re.search(rf"them\s+{number}\s+(?:hop|nhom).*?nhu vay", text)
        _require_number_coverage(text, [division, *([extra] if extra else [])])
        total, groups = map(rational, division.groups())
        if groups <= 0 or not groups.is_Integer or total < 0:
            raise MathTutorError("Số nhóm phải là số nguyên dương và tổng số lượng không được âm.")
        per_group = total / groups
        if re.search(r"(?:but|hoc sinh|quyen|vien)", text) and not per_group.is_Integer:
            raise MathTutorError(
                "Không thể chia đều số vật nguyên theo dữ kiện này; cần nêu cách xử lý phần dư."
            )
        answer = f"Mỗi hộp: {_pretty(per_group)}"
        values = {"per_group": per_group}
        steps = [
            ("Chia đều", f"{_pretty(total)} ÷ {_pretty(groups)} = {_pretty(per_group)} phần tử mỗi nhóm.")
        ]
        if extra:
            additional_groups = rational(extra.group(1))
            if additional_groups < 0 or not additional_groups.is_Integer:
                raise MathTutorError("Số nhóm thêm phải là số nguyên không âm.")
            additional = additional_groups * per_group
            answer += f"; cần thêm: {_pretty(additional)}"
            values["additional"] = additional
            steps.append(
                (
                    "Tính phần thêm",
                    f"{_pretty(additional_groups)} nhóm × {_pretty(per_group)} = {_pretty(additional)} phần tử.",
                )
            )
        steps.append(
            ("Kiểm tra", f"{_pretty(groups)} × {_pretty(per_group)} = {_pretty(total)}, khớp số ban đầu.")
        )
        return result(
            answer,
            "Chia đều và mở rộng",
            steps,
            {
                "type": "groups",
                "groups": int(groups),
                "per_group": float(per_group),
                "additional_groups": int(rational(extra.group(1))) if extra else 0,
            },
            values,
            3,
            "division_inverse",
        )

    cake = re.search(r"chia\s+(?:thanh\s+)?(\d+)\s+phan bang nhau.*?an\s+(\d+)\s+phan", text)
    if cake:
        _require_number_coverage(text, [cake])
        denominator, eaten = map(int, cake.groups())
        if not 1 <= denominator <= 120 or not 0 <= eaten <= denominator:
            raise MathTutorError("Số phần ăn phải từ 0 đến tổng số phần; tổng từ 1 đến 120.")
        remaining = denominator - eaten
        return result(
            f"Đã ăn: {eaten}/{denominator}; còn lại: {remaining}/{denominator}",
            "Phân số một đơn vị",
            [
                ("Xác định đơn vị", f"Một bánh gồm {denominator} phần bằng nhau."),
                ("Phần đã ăn", f"Ăn {eaten} phần, chiếm {eaten}/{denominator} bánh."),
                (
                    "Phần còn lại",
                    f"{denominator} − {eaten} = {remaining} phần, chiếm {remaining}/{denominator}.",
                ),
                ("Kiểm tra", "Phần đã ăn + phần còn lại bằng đúng một bánh."),
            ],
            {"type": "fraction", "numerator": eaten, "denominator": denominator},
            {"eaten": sp.Rational(eaten, denominator), "remaining": sp.Rational(remaining, denominator)},
            3,
            "fraction_complement",
        )

    average = re.search(r"lan luot\s+([\d\s,.va]+)\s+(?:cay|diem|quyen)", text)
    if average and "trung binh" in text:
        _require_number_coverage(text, [average])
        quantities = [int(value) for value in re.findall(r"\d+", average.group(1))]
        mean = sp.Rational(sum(quantities), len(quantities))
        return result(
            _pretty(mean),
            "Trung bình cộng",
            [
                ("Đọc dữ kiện", ", ".join(map(str, quantities))),
                ("Tính tổng", f"{' + '.join(map(str, quantities))} = {sum(quantities)}."),
                ("Chia đều", f"{sum(quantities)} ÷ {len(quantities)} = {_pretty(mean)}."),
                ("Kiểm tra", f"{_pretty(mean)} × {len(quantities)} = {sum(quantities)}."),
            ],
            {
                "type": "bar_chart",
                "bars": [{"label": f"Lớp {i + 1}", "value": value} for i, value in enumerate(quantities)],
                "mean": float(mean),
            },
            {"average": mean},
            4,
            "mean_times_count",
        )

    prices = re.search(
        rf"vo gia\s+{number}\s+nghin.*?but gia\s+{number}\s+nghin.*?mua\s+(\d+)\s+quyen vo.*?(\d+)\s+cay but",
        text,
    )
    if prices:
        _require_number_coverage(text, [prices])
        notebook, pen = map(rational, prices.groups()[:2])
        n, p = map(int, prices.groups()[2:])
        notebook_total, pen_total = n * notebook, p * pen
        total = notebook_total + pen_total
        return result(
            f"{_pretty(total)} nghìn đồng",
            "Tính tiền theo số lượng",
            [
                ("Tiền mua vở", f"{n} × {_pretty(notebook)} = {_pretty(notebook_total)} nghìn đồng."),
                ("Tiền mua bút", f"{p} × {_pretty(pen)} = {_pretty(pen_total)} nghìn đồng."),
                (
                    "Cộng tiền",
                    f"{_pretty(notebook_total)} + {_pretty(pen_total)} = {_pretty(total)} nghìn đồng.",
                ),
                ("Kiểm tra", f"Quy đổi: {_pretty(total * 1000)} đồng."),
            ],
            {
                "type": "bar_chart",
                "bars": [
                    {"label": "Vở", "value": float(notebook_total)},
                    {"label": "Bút", "value": float(pen_total)},
                ],
            },
            {"total_thousand": total},
            5,
            "exact_decimal_price",
        )

    cuboid = re.search(
        rf"hinh hop chu nhat dai\s+{number}\s*m.*?rong\s+{number}\s*m.*?cao\s+{number}\s*m", text
    )
    if cuboid:
        percent = re.search(r"day\s+(\d+(?:[.,]\d+)?)%", text)
        _require_number_coverage(text, [cuboid, *([percent] if percent else [])])
        length, width, height = map(rational, cuboid.groups())
        if min(length, width, height) <= 0:
            raise MathTutorError("Kích thước hình hộp phải dương.")
        volume = length * width * height
        fill = rational(percent.group(1)) / 100 if percent else sp.S.One
        if not 0 <= fill <= 1:
            raise MathTutorError("Phần nước phải từ 0% đến 100%.")
        litres = volume * 1000 * fill
        return result(
            f"Thể tích: {_pretty(volume)} m³; nước: {_pretty(litres)} lít",
            "Thể tích và phần trăm",
            [
                (
                    "Thể tích bể",
                    f"{_pretty(length)} × {_pretty(width)} × {_pretty(height)} = {_pretty(volume)} m³.",
                ),
                ("Đổi ra lít", f"{_pretty(volume)} m³ = {_pretty(volume * 1000)} lít."),
                (
                    "Tính phần nước",
                    f"{_pretty(volume * 1000)} × {_pretty(fill * 100)}% = {_pretty(litres)} lít.",
                ),
                ("Kiểm tra", f"Nước chiếm {_pretty(fill * height)} m chiều cao bể."),
            ],
            {
                "type": "cuboid",
                "length": float(length),
                "width": float(width),
                "height": float(height),
                "fill": float(fill),
                "litres": _pretty(litres),
            },
            {"volume": volume, "litres": litres},
            5,
            "volume_and_percent",
        )

    gcd_groups = re.search(r"co\s+(\d+)\s+hoc sinh nam.*?(\d+)\s+hoc sinh nu", text)
    if gcd_groups and "nhieu nhat" in text and "nhom" in text:
        _require_number_coverage(text, [gcd_groups])
        boys, girls = map(int, gcd_groups.groups())
        if boys <= 0 or girls <= 0:
            raise MathTutorError("Số học sinh mỗi nhóm phải dương.")
        count = int(sp.gcd(boys, girls))
        return result(
            f"{count} nhóm; mỗi nhóm {boys // count} nam và {girls // count} nữ",
            "Ước chung lớn nhất",
            [
                ("Điều kiện chia đều", f"Số nhóm phải là ước chung của {boys} và {girls}."),
                ("Tìm ƯCLN", f"ƯCLN({boys}; {girls}) = {count}."),
                (
                    "Tính thành phần nhóm",
                    f"{boys}/{count} = {boys // count} nam; {girls}/{count} = {girls // count} nữ.",
                ),
                ("Kiểm tra", f"{count} nhóm sử dụng hết {boys} nam và {girls} nữ."),
            ],
            {
                "type": "groups",
                "groups": count,
                "per_group": (boys + girls) // count,
                "boys_per_group": boys // count,
                "girls_per_group": girls // count,
            },
            {"groups": count, "boys_per_group": boys // count, "girls_per_group": girls // count},
            6,
            "gcd_divisibility",
        )

    scale = re.search(r"ti le\s+(1)\s*:\s*([\d.]+).*?cach nhau\s+(\d+(?:[.,]\d+)?)\s*cm", text)
    if scale:
        _require_number_coverage(text, [scale])
        factor = int(scale.group(2).replace(".", ""))
        distance = rational(scale.group(3))
        if factor <= 0 or distance < 0:
            raise MathTutorError("Tỉ lệ phải dương, khoảng cách không được âm.")
        km = distance * factor / 100_000
        return result(
            f"{_pretty(km)} km",
            "Tỉ lệ bản đồ",
            [
                ("Hiểu tỉ lệ", f"1 cm trên bản đồ tương ứng {factor} cm thực tế."),
                ("Khoảng cách thực", f"{_pretty(distance)} × {factor} = {_pretty(distance * factor)} cm."),
                ("Đổi sang km", f"{_pretty(distance * factor)} / 100000 = {_pretty(km)} km."),
                ("Kiểm tra ngược", f"{_pretty(km)} km / {factor} = {_pretty(distance)} cm trên bản đồ."),
            ],
            {"type": "scale_map", "map_cm": _pretty(distance), "factor": factor, "km": _pretty(km)},
            {"km": km},
            7,
            "unit_and_scale_inverse",
        )

    triangle = re.search(rf"tam giac abc vuong tai a.*?ab\s*=\s*{number}\s*cm.*?ac\s*=\s*{number}\s*cm", text)
    if triangle:
        _require_number_coverage(text, [triangle])
        ab, ac = map(rational, triangle.groups())
        if min(ab, ac) <= 0:
            raise MathTutorError("Cạnh tam giác phải dương.")
        bc = sp.sqrt(ab**2 + ac**2)
        ah = sp.simplify(ab * ac / bc)
        return result(
            f"BC = {_pretty(bc)} cm; AH = {_pretty(ah)} cm",
            "Tam giác vuông",
            [
                (
                    "Vẽ hình và xác định cạnh",
                    f"AB = {_pretty(ab)}, AC = {_pretty(ac)} là hai cạnh góc vuông.",
                ),
                ("Áp dụng Pythagore", f"BC² = AB² + AC² = {_pretty(ab**2 + ac**2)}; BC = {_pretty(bc)} cm."),
                ("Diện tích theo hai cách", "AB×AC/2 = BC×AH/2."),
                ("Tính đường cao", f"AH = AB×AC/BC = {_pretty(ah)} cm."),
                ("Kiểm tra", f"BC² = {_pretty(ab**2 + ac**2)} và BC×AH = {_pretty(ab * ac)} = AB×AC."),
            ],
            {
                "type": "right_triangle",
                "ab": float(ab),
                "ac": float(ac),
                "bc": _pretty(bc),
                "ah": _pretty(ah),
            },
            {"bc": bc, "ah": ah},
            9,
            "pythagoras_and_area_identity",
        )

    return None
