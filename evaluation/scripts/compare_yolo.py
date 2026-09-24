"""So sánh các model YOLO (có thể khác bộ lớp) trên CÙNG ảnh test, theo nhóm nguồn.

    uv run python -m evaluation.scripts.compare_yolo --data data/processed/yolo-edu-ent-v2 \
        --model artifacts/vision/a.pt --model artifacts/vision/b.pt --group st dice cards

Với mỗi model và nhóm (tiền tố tên ảnh: st_, cr_, dice_, guitar_, chess_, cards_), tạo bộ đánh giá tạm trong
runs/yolo-eval/: nhãn được ánh xạ sang chỉ số lớp của model theo TÊN lớp; lớp model không biết bị bỏ khỏi nhãn.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, type=Path)
    ap.add_argument("--group", nargs="+", default=["st", "cr", "dice", "guitar", "chess", "cards"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.001, help="ngưỡng tin cậy khi chấm (0.001 = đo mAP chuẩn)")
    args = ap.parse_args()

    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    src_names = yaml.safe_load((args.data / "data.yaml").read_text())["names"]
    tmp = Path("runs/yolo-eval")
    results: dict[str, dict[str, dict[str, float]]] = {}
    for mpath in args.model:
        model = YOLO(str(mpath))
        name_to_idx = {n: i for i, n in model.names.items()}
        results[mpath.stem] = {}
        for g in args.group:
            d = tmp / mpath.stem / g
            shutil.rmtree(d, ignore_errors=True)
            (d / "images").mkdir(parents=True)
            (d / "labels").mkdir(parents=True)
            n = 0
            for img in sorted((args.data / args.split / "images").glob(f"{g}_*.jpg")):
                lab = args.data / args.split / "labels" / f"{img.stem}.txt"
                out = []
                for line in lab.read_text().splitlines():
                    c, *rest = line.split()
                    cname = src_names[int(c)]
                    if cname in name_to_idx:
                        out.append(" ".join([str(name_to_idx[cname]), *rest]))
                if not out:
                    continue
                (d / "images" / img.name).symlink_to(img.resolve())
                (d / "labels" / f"{img.stem}.txt").write_text("\n".join(out) + "\n")
                n += 1
            if not n:
                results[mpath.stem][g] = {"images": 0}
                continue
            (d / "data.yaml").write_text(yaml.safe_dump(
                {"path": str(d.resolve()), "train": "images", "val": "images", "names": model.names},
                allow_unicode=True))
            m = model.val(data=str(d / "data.yaml"), split="val", conf=args.conf, plots=False, verbose=False)
            results[mpath.stem][g] = {"images": n, "mAP50": round(float(m.box.map50), 4),
                                      "mAP50_95": round(float(m.box.map), 4), "P": round(float(m.box.mp), 4),
                                      "R": round(float(m.box.mr), 4)}
    print(json.dumps(results, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
