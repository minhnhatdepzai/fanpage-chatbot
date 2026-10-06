"""Train bộ nhận dạng toán trên CPU, giữ tập đánh giá tách theo mẫu diễn đạt.

uv run python -m training.scripts.train_math_router

Đây là logistic regression cho phân luồng, không phải adapter Qwen/LoRA.
Không sử dụng hội thoại khách hàng. Export JSON được runtime đọc trực tiếp.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from app.math_tutor.router_model import DIMENSIONS, MODEL_PATH, features


def main() -> None:
    rng = np.random.default_rng(20261006)
    positive = [
        "Lan mua {a} quyển sách giá {b} nghìn, phải trả bao nhiêu?",
        "Có {a} chiếc kẹo phân đều cho {b} bạn. Mỗi bạn nhận bao nhiêu?",
        "Hàng ghế đầu {a} chỗ, hàng kế tiếp tăng {b} chỗ. Đếm tổng.",
        "Hình chữ nhật dài {a}, rộng {b}, tính diện tích và chu vi.",
        "Nhiệt độ {a} độ tăng {b} độ thì bằng mấy?",
        "Rút gọn x^2 + {a}x - {b} và giải phương trình bằng không.",
        "Một nhóm làm {a} sản phẩm trong {b} giờ. Tính năng suất.",
        "Trung bình cộng của {a}, {b} và 15 bằng bao nhiêu?",
        "Một chiếc bánh có {a} phần, ăn {b} phần thì còn bao nhiêu?",
        "Tính {a} + {b} và minh họa trục số.",
        "Tam giác có cạnh {a} và {b}, tính cạnh huyền.",
        "Lấy 2 viên bi từ {a} viên đỏ và {b} viên xanh. Xác suất khác màu?",
    ]
    negative = [
        "Viết bài văn {a} chữ về những năm tháng tuổi {b}.",
        "Năm {a}, Việt Nam có sự kiện lịch sử gì?",
        "Dịch đoạn văn bài {a} trang {b} sang tiếng Anh.",
        "So sánh điện thoại {a} với phiên bản {b}.",
        "Kể truyện về {a} người bạn có {b} sở thích.",
        "Tôi học lớp {a}, muốn biết lịch thi ngày {b}.",
        "Phân tích {a} câu thơ và viết {b} đoạn văn.",
        "Chào bạn, tôi có {a} câu hỏi về Chí Phèo và bài {b}.",
        "Tóm tắt {a} chương đầu của cuốn sách trong {b} đoạn.",
        "Giải thích tiếng Anh câu {a} trong đề {b}.",
        "Gợi ý {a} món ăn cho {b} người.",
        "Tin hôm nay ngày {a} tháng {b} có gì mới?",
    ]
    heldout_positive = [
        "{a} cây bút giá {b} đồng mỗi cây, hết bao nhiêu tiền?",
        "Phân {a} quả táo cho {b} học sinh như nhau. Tìm số quả mỗi em.",
        "Một hình vuông cạnh {a} cm có chu vi bao nhiêu? Cho biết số cạnh là {b}.",
        "Nhiệt kế chỉ {a} độ rồi giảm {b} độ. Tìm nhiệt độ cuối.",
    ]
    heldout_negative = [
        "Phân tích nhân vật trong {a} chương đầu rồi viết {b} đoạn nghị luận.",
        "Dịch câu số {a} trang {b} từ Anh sang Việt giúp em.",
        "Tôi muốn nghe {a} bài thơ về tuổi {b}.",
        "Lịch sử năm {a} thế kỷ {b} có những thay đổi gì?",
    ]
    records = []

    def build(templates: list[str], label: int, count: int, split: str) -> tuple[np.ndarray, np.ndarray]:
        rows = []
        for i in range(count):
            text = templates[i % len(templates)].format(
                a=int(rng.integers(2, 90)), b=int(rng.integers(2, 30))
            )
            vector = np.zeros(DIMENSIONS, dtype=np.float64)
            for index, value in features(text).items():
                vector[index] = value
            rows.append(vector)
            records.append(
                {
                    "input": text,
                    "expectedOutput": label,
                    "metadata": {"split": split, "synthetic": True, "template": i % len(templates)},
                }
            )
        return np.array(rows), np.full(count, label, dtype=np.float64)

    px, py = build(positive, 1, 600, "train")
    nx, ny = build(negative, 0, 600, "train")
    tx = np.concatenate([px, nx])
    ty = np.concatenate([py, ny])
    ex, ey = build(heldout_positive, 1, 120, "test")
    ox, oy = build(heldout_negative, 0, 120, "test")
    eval_x = np.concatenate([ex, ox])
    eval_y = np.concatenate([ey, oy])
    weights = np.zeros(DIMENSIONS)
    bias = 0.0
    history = []
    for epoch in range(240):
        predictions = 1 / (1 + np.exp(-np.clip(tx @ weights + bias, -60, 60)))
        error = predictions - ty
        weights -= 1.4 * (tx.T @ error / len(ty) + 0.0001 * weights)
        bias -= 1.4 * float(error.mean())
        if epoch % 30 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(
                        -np.mean(ty * np.log(predictions + 1e-9) + (1 - ty) * np.log(1 - predictions + 1e-9))
                    ),
                }
            )
    confidence = 1 / (1 + np.exp(-np.clip(eval_x @ weights + bias, -60, 60)))
    accuracy = float(((confidence >= 0.5) == eval_y).mean())
    passed = accuracy >= 0.95
    out = Path("artifacts/math-quality")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = MODEL_PATH if passed else out / "math-router-v1-candidate.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model = {
        "kind": "logistic-regression-math-intent",
        "dimensions": DIMENSIONS,
        "weights": weights.tolist(),
        "bias": bias,
        "training_examples": 1200,
        "heldout_examples": 240,
        "heldout_accuracy": accuracy,
        "seed": 20261006,
        "optimizer_steps": 240,
        "loss_history": history,
        "purpose": "routing_only_not_answer_generation",
    }
    checkpoint.write_text(json.dumps(model, ensure_ascii=False))
    loaded = json.loads(checkpoint.read_text())
    assert np.allclose(eval_x @ np.array(loaded["weights"]) + loaded["bias"], eval_x @ weights + bias)
    (out / "router-dataset.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    report = {key: value for key, value in model.items() if key not in {"weights", "bias"}}
    report.update(
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        promoted=passed,
        gate_accuracy=0.95,
        train_template_count=len(positive) + len(negative),
        heldout_template_count=len(heldout_positive) + len(heldout_negative),
        reload_verified=True,
        synthetic=True,
        license="project-authored synthetic templates",
        limitations="Held-out wording in covered families; not universal mathematics accuracy.",
    )
    (out / "router-training-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
