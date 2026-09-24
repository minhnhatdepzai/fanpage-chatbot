"""Bộ YOLO v2 "giáo dục + giải trí" gộp 6 nguồn Kaggle (đã kiểm tra giấy phép khi tải, --kind vision).

    uv run python -m training.vision.prepare_yolo_v2

Giáo dục:  stationary-dataset (CC BY 4.0), objects-in-the-classroom (MIT)
Giải trí:  six-sided-dice (CC0), guitar-detection (Kaggle ghi CC0; nguồn Roboflow ghi CC BY 4.0 -> ghi công theo CC BY),
           chess-piece-object-detection (CC0), playing-cards-object-detection (CC0, lấy mẫu vì có 20.000 ảnh tổng hợp).
Lớp trùng nghĩa giữa các nguồn được hợp nhất (vd. "eraser" và lớp 1 của stationery -> "tay").
Chống rò rỉ: gom nhóm theo ảnh gốc (tên trước ".rf." của Roboflow) + dHash gần trùng; một nhóm chỉ thuộc một tập.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from training.vision.prepare_yolo import dhash, group_near_duplicates, normalize_yolo_line, voc_to_yolo

RAW = Path("data/raw/kaggle")
OUT = Path("data/processed/yolo-edu-ent-v2")
SEED = 20260924
MAX_SIDE = 1280
CARD_SAMPLE = {"train": 2500, "val": 300, "test": 300}

SUITS = {"c": "nhép", "d": "rô", "h": "cơ", "s": "bích"}
RANKS = {"A": "Át", "J": "J", "Q": "Q", "K": "K"}
CHESS_PIECE = {"bishop": "tượng", "king": "vua", "knight": "mã", "pawn": "tốt", "queen": "hậu", "rook": "xe"}
CHESS_COLOR = {"black": "đen", "white": "trắng"}
CLASSROOM = {
    "table": ("ban", "bàn"), "chair": ("ghe", "ghế"), "whiteboard": ("bang_trang", "bảng trắng"),
    "bookshelf": ("ke_sach", "kệ sách"), "clock": ("dong_ho", "đồng hồ"), "wall-magazine": ("bao_tuong", "báo tường"),
    "trash-can": ("thung_rac", "thùng rác"), "eraser": ("tay", "tẩy (gôm)"), "sharpener": ("got_but_chi", "gọt bút chì"),
    "pen": ("but", "bút"), "book": ("sach", "sách"), "ruler": ("thuoc_ke", "thước kẻ"), "scissor": ("keo", "kéo"),
    "fan": ("quat", "quạt"), "laptop": ("laptop", "laptop"), "remote-control": ("dieu_khien", "điều khiển từ xa"),
    "bag": ("cap_tui", "cặp/túi"), "pants": ("quan", "quần"), "shoes": ("giay", "giày"), "hat": ("mu", "mũ"),
}


def _card_vi(code: str) -> str:
    rank, suit = code[:-1], code[-1]
    return f"lá bài {RANKS.get(rank, rank)} {SUITS[suit]}"


def build_classes() -> tuple[list[str], dict[str, str]]:
    vi: dict[str, str] = {k: v for k, v in CLASSROOM.values()}
    vi.update({f"xuc_xac_{i}": f"xúc xắc mặt {i}" for i in range(1, 7)})
    vi["dan_guitar"] = "đàn guitar"
    for piece, p_vi in CHESS_PIECE.items():
        for color, c_vi in CHESS_COLOR.items():
            vi[f"co_vua_{piece}_{color}"] = f"quân {p_vi} {c_vi} (cờ vua)"
    for rank in ["A", *[str(n) for n in range(2, 11)], "J", "Q", "K"]:
        for suit in SUITS:
            vi[f"bai_{rank}{suit}"] = _card_vi(f"{rank}{suit}")
    names = list(vi)
    return names, vi


CLASSES, VI_NAMES = build_classes()
IDX = {c: i for i, c in enumerate(CLASSES)}


def _save(img_path: Path, labels: list[str], split: str, name: str) -> None:
    img = Image.open(img_path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
    img.save(OUT / split / "images" / f"{name}.jpg", quality=92)
    (OUT / split / "labels" / f"{name}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))


def _remap(lines: list[str], mapping: dict[int, str]) -> list[str]:
    out = []
    for raw in lines:
        norm = normalize_yolo_line(raw)
        if norm is None:
            continue
        cls, *rest = norm.split()
        out.append(" ".join([str(IDX[mapping[int(cls)]]), *rest]))
    return out


def _read_labels(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _source_key(p: Path) -> str:
    return p.stem.split(".rf.")[0]  # Roboflow: nhiều bản tăng cường của một ảnh gốc chung tiền tố


def assign_groups(paths: list[Path], preferred: list[str], rng: random.Random, dedupe: bool,
                  ratios: tuple[float, float] = (0.8, 0.9)) -> tuple[list[str], int]:
    """Gom nhóm (tiền tố Roboflow + dHash gần trùng) rồi gán tập. preferred='' -> tự chia theo ratios.
    Trả về tập cho từng ảnh và số ảnh phải đổi tập so với chia sẵn của tác giả (dấu hiệu rò rỉ ở nguồn)."""
    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_key: dict[str, int] = {}
    for i, p in enumerate(paths):
        k = _source_key(p)
        if k in by_key:
            parent[find(i)] = find(by_key[k])
        else:
            by_key[k] = i
    if dedupe:
        for i, g in enumerate(group_near_duplicates([dhash(Image.open(p)) for p in paths])):
            parent[find(i)] = find(g)
    roots = [find(i) for i in range(len(paths))]
    split_of: dict[int, str] = {}
    if all(preferred):
        votes: dict[int, Counter[str]] = {}
        for r, s in zip(roots, preferred, strict=True):
            votes.setdefault(r, Counter())[s] += 1
        # nhóm trải qua nhiều tập -> đặt vào tập "hiếm" nhất theo thứ tự test > val > train (giữ tập đánh giá sạch)
        for r, c in votes.items():
            split_of[r] = next(s for s in ("test", "val", "train") if c[s]) if len(c) > 1 else next(iter(c))
    else:
        uniq = sorted(set(roots))
        rng.shuffle(uniq)
        n = len(uniq)
        split_of = {g: ("train" if i < ratios[0] * n else "val" if i < ratios[1] * n else "test")
                    for i, g in enumerate(uniq)}
    splits = [split_of[r] for r in roots]
    moved = sum(1 for a, b in zip(splits, preferred, strict=True) if b and a != b)
    return splits, moved


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    rng = random.Random(SEED)
    report: dict[str, dict] = {}

    def add(source: str, items: list[tuple[Path, list[str], str]], dedupe: bool) -> None:
        paths = [p for p, _, _ in items]
        splits, moved = assign_groups(paths, [s for *_, s in items], rng, dedupe)
        counts: Counter[str] = Counter()
        for i, ((p, labels, _), split) in enumerate(zip(items, splits, strict=True)):
            _save(p, labels, split, f"{source}_{i:05d}")
            counts[split] += 1
        report[source] = {"images": dict(counts), "moved_between_author_splits": moved}

    # 1) đồ dùng học tập (4 lớp, không có data.yaml; tên lớp xác định bằng mắt - xem prepare_yolo.py)
    st = RAW / "abdullahsami10__stationary-dataset/v1/files/Stationary Dataset"
    st_map = {0: "but", 1: "tay", 2: "got_but_chi", 3: "thuoc_ke"}
    add("st", [(p, _remap(_read_labels(st / s / "labels" / f"{p.stem}.txt"), st_map), t)
               for s, t in [("train", "train"), ("valid", "val"), ("test", "test")]
               for p in sorted((st / s / "images").glob("*.jpg"))], dedupe=True)
    # 2) đồ vật lớp học (20 lớp)
    cr = RAW / "aryakrisnaputra__objects-in-the-classroom/v1/files/objects in the classroom/data"
    cr_names = list(CLASSROOM)
    cr_map = {i: CLASSROOM[n][0] for i, n in enumerate(cr_names)}
    add("cr", [(p, _remap(_read_labels(cr / "labels" / s / f"{p.stem}.txt"), cr_map), s)
               for s in ("train", "val", "test")
               for p in sorted((cr / "images" / s).iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        dedupe=True)
    # 3) xúc xắc (VOC) - tự chia
    dice = RAW / "josephnelson__six-sided-dice-images-and-bounding-boxes/v1/files/export"
    old_classes = ["but", "tay", "got_but_chi", "thuoc_ke", *[f"xuc_xac_{i}" for i in range(1, 7)]]
    items = []
    for x in sorted(dice.glob("*.xml")):
        lines = []
        for ln in voc_to_yolo(x):
            c, *rest = ln.split()
            lines.append(" ".join([str(IDX[old_classes[int(c)]]), *rest]))
        items.append((dice / (x.name[: -len(".xml")] + ".jpg"), lines, ""))
    add("dice", items, dedupe=True)
    # 4) guitar: tập test của tác giả KHÔNG có nhãn -> chỉ dùng train+valid, tự chia
    gt = next(RAW.glob("sagarnildass__guitar-detection-dataset/v*/files/Guitar-Detection-2"))
    add("guitar", [(p, _remap(_read_labels(gt / s / "labels" / f"{p.stem}.txt"), {0: "dan_guitar"}), "")
                   for s in ("train", "valid") for p in sorted((gt / s / "images").glob("*.jpg"))], dedupe=True)
    # 5) cờ vua (12 lớp)
    ch = next(RAW.glob("cookiemonsteryum__chess-piece-object-detection/v*/files"))
    ch_map = {i: f"co_vua_{n.split()[0]}_{n.split()[1]}" for i, n in enumerate(
        ["bishop black", "bishop white", "king black", "king white", "knight black", "knight white", "pawn black",
         "pawn white", "queen black", "queen white", "rook black", "rook white"])}
    add("chess", [(p, _remap(_read_labels(ch / s / "labels" / f"{p.stem}.txt"), ch_map), t)
                  for s, t in [("train", "train"), ("valid", "val"), ("test", "test")]
                  for p in sorted((ch / s / "images").glob("*.jpg"))], dedupe=True)
    # 6) bài tây (52 lớp) - lấy mẫu ngẫu nhiên theo từng tập của tác giả; ảnh tổng hợp, không dHash
    cd = next(RAW.glob("andy8744__playing-cards-object-detection-dataset/v*/files"))
    cd_names = ["10c", "10d", "10h", "10s", "2c", "2d", "2h", "2s", "3c", "3d", "3h", "3s", "4c", "4d", "4h", "4s",
                "5c", "5d", "5h", "5s", "6c", "6d", "6h", "6s", "7c", "7d", "7h", "7s", "8c", "8d", "8h", "8s", "9c",
                "9d", "9h", "9s", "Ac", "Ad", "Ah", "As", "Jc", "Jd", "Jh", "Js", "Kc", "Kd", "Kh", "Ks", "Qc", "Qd",
                "Qh", "Qs"]
    cd_map = {i: f"bai_{n}" for i, n in enumerate(cd_names)}
    items = []
    for s, t in [("train", "train"), ("valid", "val"), ("test", "test")]:
        imgs = sorted((cd / s / "images").glob("*.jpg"))
        for p in rng.sample(imgs, min(CARD_SAMPLE[t], len(imgs))):
            items.append((p, _remap(_read_labels(cd / s / "labels" / f"{p.stem}.txt"), cd_map), t))
    add("cards", items, dedupe=False)

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    manifest = {
        "name": "yolo-edu-ent-v2", "created_at": datetime.now(UTC).isoformat(), "seed": SEED,
        "classes": CLASSES, "vi_names": VI_NAMES, "sources": report,
        "licenses": {
            "abdullahsami10/stationary-dataset": "CC BY 4.0",
            "aryakrisnaputra/objects-in-the-classroom": "MIT",
            "josephnelson/six-sided-dice-images-and-bounding-boxes": "CC0-1.0",
            "sagarnildass/guitar-detection-dataset": "Kaggle: CC0-1.0; Roboflow source: CC BY 4.0 (attribute)",
            "cookiemonsteryum/chess-piece-object-detection": "CC0-1.0",
            "andy8744/playing-cards-object-detection-dataset": "CC0-1.0",
        },
        "split_policy": "groups = Roboflow source prefix + dHash near-duplicates; a group never crosses splits",
        "card_sample": CARD_SAMPLE,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"classes": len(CLASSES), "sources": report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
