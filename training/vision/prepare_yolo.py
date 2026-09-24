"""Gộp dataset Kaggle thành một bộ YOLO cho model nhận diện "giáo dục + giải trí".

    uv run python -m training.vision.prepare_yolo

Nguồn (đã kiểm tra giấy phép khi tải bằng training.data.kaggle_download --kind vision):
  - abdullahsami10/stationary-dataset (CC BY 4.0): nhãn YOLO, 4 lớp, KHÔNG có data.yaml. Tên lớp do Claude xác định
    ngày 24/09/2026 bằng cách xem ảnh cắt theo từng lớp: 0 bút, 1 tẩy, 2 gọt bút chì, 3 thước kẻ.
  - josephnelson/six-sided-dice-images-and-bounding-boxes (CC0-1.0): nhãn Pascal VOC, lớp "1".."6" = mặt xúc xắc.
Chia tập theo NHÓM ảnh gần trùng (dHash, Hamming <= 6) để bản tăng cường của cùng một ảnh không rơi vào 2 tập.
Ảnh lớn được thu nhỏ (cạnh dài <= 1280). Ghi manifest (nguồn, giấy phép, seed, số lượng, sha256).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

RAW = Path("data/raw/kaggle")
OUT = Path("data/processed/yolo-edu-ent-v1")
SEED = 20260924
MAX_SIDE = 1280
CLASSES = ["but", "tay", "got_but_chi", "thuoc_ke", "xuc_xac_1", "xuc_xac_2", "xuc_xac_3", "xuc_xac_4",
           "xuc_xac_5", "xuc_xac_6"]
# Tên tiếng Việt hiển thị cho người dùng / prompt
VI_NAMES = {"but": "bút", "tay": "tẩy (gôm)", "got_but_chi": "gọt bút chì", "thuoc_ke": "thước kẻ",
            **{f"xuc_xac_{i}": f"xúc xắc mặt {i}" for i in range(1, 7)}}
STATIONERY = RAW / "abdullahsami10__stationary-dataset/v1/files/Stationary Dataset"
DICE = RAW / "josephnelson__six-sided-dice-images-and-bounding-boxes/v1/files/export"


def dhash(img: Image.Image, size: int = 8) -> int:
    g = img.convert("L").resize((size + 1, size))
    px = list(g.tobytes())  # ảnh "L": mỗi byte là một điểm ảnh
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | (px[r * (size + 1) + c] > px[r * (size + 1) + c + 1])
    return bits


def group_near_duplicates(hashes: list[int], max_dist: int = 6) -> list[int]:
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i] ^ hashes[j]).count("1") <= max_dist:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(hashes))]


def _save(img: Image.Image, labels: list[str], split: str, name: str) -> None:
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    img.save(OUT / split / "images" / f"{name}.jpg", quality=92)
    (OUT / split / "labels" / f"{name}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))


def normalize_yolo_line(line: str) -> str | None:
    """Dòng YOLO chuẩn (5 số) giữ nguyên; dòng đa giác (class x1 y1 x2 y2 ...) -> hộp bao. Dataset gốc trộn cả hai
    loại trong cùng tệp, Ultralytics khi đó BỎ QUA cả ảnh ("labels mix segment and detection rows")."""
    parts = line.split()
    if len(parts) == 5:
        return line
    if len(parts) >= 7 and len(parts) % 2 == 1:
        xs = [float(v) for v in parts[1::2]]
        ys = [float(v) for v in parts[2::2]]
        x1, x2, y1, y2 = max(0.0, min(xs)), min(1.0, max(xs)), max(0.0, min(ys)), min(1.0, max(ys))
        return f"{parts[0]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"
    return None


def voc_to_yolo(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()  # noqa: S314 - tệp nhãn cục bộ đã tải từ nguồn đã kiểm tra
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    out = []
    for obj in root.findall("object"):
        cls = CLASSES.index(f"xuc_xac_{obj.findtext('name').strip()}")
        b = obj.find("bndbox")
        x1, x2 = float(b.findtext("xmin")), float(b.findtext("xmax"))
        y1, y2 = float(b.findtext("ymin")), float(b.findtext("ymax"))
        out.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    rng = random.Random(SEED)
    counts: dict[str, dict[str, int]] = {}

    # 1) đồ dùng học tập: giữ nguyên chia tập của tác giả (train/valid/test) nhưng kiểm tra trùng lặp chéo tập
    items = []
    polygons = 0
    for split, target in [("train", "train"), ("valid", "val"), ("test", "test")]:
        for img_p in sorted((STATIONERY / split / "images").glob("*.jpg")):
            lab = STATIONERY / split / "labels" / f"{img_p.stem}.txt"
            raw = [x.strip() for x in lab.read_text().splitlines() if x.strip()] if lab.exists() else []
            lines = [n for n in (normalize_yolo_line(x) for x in raw) if n]
            polygons += sum(len(x.split()) > 5 for x in raw)
            items.append((target, img_p, lines))
    hashes = [dhash(Image.open(p)) for _, p, _ in items]
    groups = group_near_duplicates(hashes)
    group_split: dict[int, str] = {}
    leaked = 0
    for (target, img_p, lines), g in zip(items, groups, strict=True):
        split = group_split.setdefault(g, target)  # ảnh gần trùng ảnh đã có -> theo tập của nhóm (chống rò rỉ)
        leaked += split != target
        _save(Image.open(img_p), lines, split, f"st_{img_p.stem}")
        counts.setdefault("stationery", {}).setdefault(split, 0)
        counts["stationery"][split] += 1

    # 2) xúc xắc: chia 80/10/10 theo nhóm gần trùng
    xmls = sorted(DICE.glob("*.xml"))
    imgs = [DICE / (x.name[: -len(".xml")] + ".jpg") for x in xmls]
    dgroups = group_near_duplicates([dhash(Image.open(p)) for p in imgs])
    uniq = sorted(set(dgroups))
    rng.shuffle(uniq)
    n = len(uniq)
    assign = {g: ("train" if i < 0.8 * n else "val" if i < 0.9 * n else "test") for i, g in enumerate(uniq)}
    for x, p, g in zip(xmls, imgs, dgroups, strict=True):
        split = assign[g]
        _save(Image.open(p), voc_to_yolo(x), split, f"dice_{hashlib.sha1(p.name.encode(), usedforsecurity=False).hexdigest()[:12]}")
        counts.setdefault("dice", {}).setdefault(split, 0)
        counts["dice"][split] += 1

    data_yaml = OUT / "data.yaml"
    data_yaml.write_text(
        f"path: {OUT.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    manifest = {
        "name": "yolo-edu-ent-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "classes": CLASSES,
        "vi_names": VI_NAMES,
        "sources": [
            {"ref": "abdullahsami10/stationary-dataset", "license": "CC BY 4.0",
             "url": "https://www.kaggle.com/datasets/abdullahsami10/stationary-dataset",
             "note": "class names inferred by visual inspection (dataset ships no data.yaml)",
             "stationery_images_moved_between_splits_as_near_duplicates": leaked,
             "polygon_rows_converted_to_boxes": polygons},
            {"ref": "josephnelson/six-sided-dice-images-and-bounding-boxes", "license": "CC0-1.0",
             "url": "https://www.kaggle.com/datasets/josephnelson/six-sided-dice-images-and-bounding-boxes",
             "dice_groups": n, "dice_images": len(imgs)},
        ],
        "counts": counts,
        "split_policy": "near-duplicate groups (dHash hamming<=6) never cross splits",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"counts": counts, "moved_near_duplicates": leaked, "dice_groups": n, "polygons": polygons},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
