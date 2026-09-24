"""Fine-tune YOLO (Ultralytics, giấy phép AGPL-3.0) trên bộ yolo-edu-ent-v1.

    uv run python -m training.vision.train_yolo --epochs 80

Khởi đầu từ trọng số COCO; lưu vào artifacts/vision/<run>/ kèm meta (dataset, lớp, chỉ số trên tập test).
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

DATA = Path("data/processed/yolo-edu-ent-v1")
OUT = Path("artifacts/vision")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--data", type=Path, default=DATA, help="thư mục dataset (có data.yaml, manifest.json)")
    ap.add_argument("--name", default="edu-ent", help="tiền tố tên lần chạy")
    ap.add_argument("--yaml", default="data.yaml", help="tệp cấu hình trong --data (vd. data_oversampled.yaml)")
    args = ap.parse_args()
    data = args.data

    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # không gửi thống kê sử dụng về Ultralytics
    OUT.mkdir(parents=True, exist_ok=True)
    run = f"{args.name}-{Path(args.model).stem}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    model = YOLO(str(OUT / args.model) if (OUT / args.model).exists() else args.model)
    model.train(data=str(data / args.yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                patience=args.patience, seed=20260924, deterministic=True, project=str(OUT.resolve()), name=run,
                workers=4, plots=False, verbose=False)
    best = OUT / run / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(data / "data.yaml"), split="test", plots=False, verbose=False)
    per_class = {model.names[int(c)]: round(float(m), 4) for c, m in zip(metrics.box.ap_class_index, metrics.box.maps[metrics.box.ap_class_index], strict=True)}
    meta = {
        "run": run,
        "base_weights": args.model,
        "dataset": json.loads((data / "manifest.json").read_text()),
        "test": {"mAP50": round(float(metrics.box.map50), 4), "mAP50_95": round(float(metrics.box.map), 4),
                 "per_class_mAP50_95": per_class},
        "args": {k: str(v) for k, v in vars(args).items()},
        "license_note": "Ultralytics YOLO: AGPL-3.0 (dịch vụ thương mại cần công bố mã nguồn hoặc giấy phép Enterprise)",
        "finished_at": datetime.now(UTC).isoformat(),
    }
    (OUT / run / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    shutil.copy(best, OUT / f"{run}.pt")
    print(json.dumps({"run": run, "test": meta["test"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
