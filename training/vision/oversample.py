"""Tăng mẫu các nhóm ít ảnh CHỈ trong tập train (lặp đường dẫn trong danh sách, không chép ảnh).

    uv run python -m training.vision.oversample --data data/processed/yolo-edu-ent-v2 --factor dice=3 guitar=5 chess=2

Tạo data_oversampled.yaml (train = train_oversampled.txt; val/test giữ nguyên để đánh giá không bị lệch).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--factor", nargs="+", default=["dice=3", "guitar=5", "chess=2"])
    args = ap.parse_args()
    factors = {k: int(v) for k, v in (f.split("=") for f in args.factor)}
    lines = []
    for img in sorted((args.data / "train" / "images").glob("*.jpg")):
        k = next((p for p in factors if img.name.startswith(p + "_")), None)
        lines += [str(img.resolve())] * (factors[k] if k else 1)
    (args.data / "train_oversampled.txt").write_text("\n".join(lines) + "\n")
    base = (args.data / "data.yaml").read_text()
    (args.data / "data_oversampled.yaml").write_text(
        base.replace("train: train/images", "train: train_oversampled.txt")
    )
    print(f"train entries: {len(lines)} (factors {factors})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
