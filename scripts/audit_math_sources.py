#!/usr/bin/env python3
"""Kiểm kê nguồn VietJack và chạy tập đánh giá Toán đã được duyệt."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.math_source_audit import (
    dump_report,
    evaluate_cases,
    inventory_sources,
    load_cases,
    load_source_specs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory", help="Lấy metadata URL/tiêu đề, không sao chép nội dung bài."
    )
    inventory.add_argument(
        "--sources", type=Path, default=Path("evaluation/datasets/vietjack_sources_v1.yaml")
    )
    inventory.add_argument("--output", type=Path)
    inventory.add_argument("--delay", type=float, default=0.25)

    evaluate = subparsers.add_parser("evaluate", help="Chạy các ca có đáp án tham chiếu đã được duyệt.")
    evaluate.add_argument(
        "--cases", type=Path, default=Path("evaluation/datasets/math_vietjack_audit_draft_v0.yaml")
    )
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--strict", action="store_true", help="Trả mã lỗi nếu còn ca chưa đạt.")

    args = parser.parse_args()
    if args.command == "inventory":
        report = inventory_sources(load_source_specs(args.sources), delay_seconds=max(0, args.delay))
    else:
        report = evaluate_cases(load_cases(args.cases))
    print(dump_report(report, args.output), end="")
    return int(args.command == "evaluate" and args.strict and report["summary"]["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
