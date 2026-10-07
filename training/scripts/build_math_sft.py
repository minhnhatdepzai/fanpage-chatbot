"""Sinh dữ liệu SFT Toán 1-12 từ bộ giải xác định, không sao chép đề nguồn.

Mỗi đáp án được tạo bởi ``solve_math`` sau khi thế ngược/đối chiếu và sau khi
payload trực quan hóa vượt hợp đồng backend. Tập này dạy cách trình bày; runtime
vẫn phải dùng bộ giải xác định để chốt đáp số.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from app.math_tutor import solve_math

DEFAULT_OUTPUT = Path("data/processed/math_1_12_sft_v1.yaml")


def _question(grade: int, index: int) -> str:
    if grade == 1:
        left = 10 + index // 9
        right = 1 + index % 9
        return f"Tính {left} + {right}. Giải thích bằng trục số."
    if grade == 2:
        start = 20 + index % 60
        removed = 1 + index * 5 % 18
        return f"Lan có {start} cái kẹo rồi cho đi {removed} cái. Lan còn lại bao nhiêu?"
    if grade == 3:
        start = 40 + index * 7 % 80
        removed = 2 + index * 3 % 25
        added = 1 + index * 5 % 20
        return (
            f"Thư viện lớp có {start} truyện, cho mượn {removed} truyện rồi nhận thêm {added} truyện. "
            "Thư viện còn bao nhiêu truyện?"
        )
    if grade == 4:
        groups = 2 + index // 50
        each = 2 + index // 5 % 10
        additional = 1 + index % 5
        return (
            f"Có {groups * each} chiếc bút chia đều vào {groups} hộp. Mỗi hộp có bao nhiêu bút? "
            f"Nếu thêm {additional} hộp như vậy thì cần thêm bao nhiêu bút?"
        )
    if grade == 5:
        denominator = 20 + index // 19
        eaten = 1 + index % 19
        return (
            f"Một chiếc bánh chia {denominator} phần bằng nhau, đã ăn {eaten} phần. "
            "Hỏi đã ăn và còn lại bao nhiêu phần chiếc bánh?"
        )
    if grade == 6:
        morning = -10 + index // 96
        increase = 2 + index // 8 % 12
        decrease = 1 + index % 8
        return (
            f"Nhiệt độ buổi sáng là {morning} độ C, trưa tăng {increase} độ rồi tối giảm {decrease} độ. "
            "Nhiệt độ buổi tối là bao nhiêu?"
        )
    if grade == 7:
        count = 2 + index % 8
        unit = 3 + index * 5 % 17
        target = count + 1 + index * 7 % 8
        return (
            f"Mua {count} quyển vở hết {count * unit} nghìn đồng. Với cùng đơn giá, "
            f"{target} quyển vở giá bao nhiêu nghìn đồng?"
        )
    if grade == 8:
        if index % 2:
            variant = index // 2
            low = 5 + 5 * (variant % 6)
            high = 55 + 5 * (variant % 6)
            total = 202 + 2 * variant
            target = (low + high) // 2
            return (
                f"Trộn dung dịch muối nồng độ {low}% với dung dịch muối nồng độ {high}% "
                f"để được {total} ml dung dịch muối nồng độ {target}%. "
                "Cần lấy bao nhiêu mililít dung dịch mỗi loại?"
            )
        root_a = 1 + index // 12
        root_b = 20 + index % 12
        return f"Giải phương trình x^2 - {root_a + root_b}x + {root_a * root_b} = 0."
    if grade == 9:
        if index % 2:
            variant = index // 2
            speed_difference = 10 + 10 * (variant % 5)
            multiplier = 2 + variant // 5
            motorcycle_speed = speed_difference * multiplier
            distance = multiplier * (motorcycle_speed + speed_difference)
            return (
                "Một ô tô và một xe máy cùng đi từ A đến B. Vận tốc của ô tô "
                f"lớn hơn vận tốc của xe máy là {speed_difference} km/h nên ô tô đến B "
                f"sớm hơn xe máy 60 phút. Biết quãng đường AB dài {distance} km, "
                "tính vận tốc của mỗi xe."
            )
        x_value = -5 + index % 14
        y_value = -4 + index * 5 % 13
        return f"Giải hệ phương trình: 2x + y = {2 * x_value + y_value} và x - y = {x_value - y_value}."
    if grade == 10:
        red = 2 + index % 9
        blue = 2 + index * 3 % 8
        yellow = 1 + index * 5 % 7
        return (
            f"Một hộp có {red} bi đỏ, {blue} bi xanh, {yellow} bi vàng. "
            "Lấy ngẫu nhiên 2 viên không hoàn lại. Tính xác suất hai viên khác màu."
        )
    if grade == 11:
        first = 5 + index % 30
        difference = 1 + index * 3 % 7
        rows = 5 + index * 5 % 30
        return (
            f"Hàng đầu có {first} ghế, mỗi hàng sau hơn hàng trước {difference} ghế. "
            f"Tính số ghế hàng {rows} và tổng số ghế của {rows} hàng."
        )
    variant = index // 2
    upper = 1 + variant % 10
    coefficient = 1 + variant // 10
    if index % 2:
        return f"Đạo hàm của {coefficient}x^3 - {upper}x + 1."
    return f"Tích phân từ 0 đến {upper} của {coefficient}x^2 + x + 1 dx"


def _answer(solution: dict[str, Any]) -> str:
    lines = [f"Kết quả: {solution['answer']}", ""]
    for index, step in enumerate(solution["steps"], 1):
        lines.append(f"Bước {index} — {step['title']}: {step['detail']}")
    lines.extend(
        [
            "",
            "Kiểm tra độc lập: đáp số đã được tính lại bằng số chính xác hoặc đại số ký hiệu; "
            f"phương pháp kiểm tra là {solution['verification']['method']}.",
            f"Trực quan hóa phù hợp: {solution['visual']['type']}.",
        ]
    )
    return "\n".join(lines)


def build_examples(*, per_grade: int, seed: int) -> list[dict[str, Any]]:
    if per_grade < 20:
        raise ValueError("Mỗi lớp cần ít nhất 20 mẫu để có train và validation riêng.")
    examples: list[dict[str, Any]] = []
    for grade in range(1, 13):
        grade_questions: set[str] = set()
        for index in range(per_grade):
            question = _question(grade, index)
            if question in grade_questions:
                raise ValueError(f"Câu hỏi sinh bị trùng ở lớp {grade}: {question}")
            grade_questions.add(question)
            solved = solve_math(question, grade=grade)
            examples.append(
                {
                    "id": f"math_g{grade:02d}_{index:04d}",
                    "category": f"math_grade_{grade}",
                    "split": "val" if index % 10 == 9 else "train",
                    "provenance": "synthetic_deterministic",
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": _answer(solved)},
                    ],
                }
            )
    random.Random(seed).shuffle(examples)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-grade", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    examples = build_examples(per_grade=args.per_grade, seed=args.seed)
    document = {
        "version": 1,
        "synthetic": True,
        "generator": "training/scripts/build_math_sft.py",
        "seed": args.seed,
        "note": (
            "Bài tương đương được sinh và kiểm chứng xác định; không sao chép đề VietJack. "
            "Không dùng tập này làm benchmark phát hành."
        ),
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8"
    )
    counts = Counter(example["split"] for example in examples)
    print(f"output={args.output} total={len(examples)} train={counts['train']} val={counts['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
