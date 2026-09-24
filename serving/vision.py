"""Phân tích ảnh cục bộ: OCR chữ (EasyOCR, Apache-2.0) + nhận diện vật thể (Ultralytics YOLO, AGPL-3.0).

Model "nhìn" ảnh theo nghĩa hẹp: chỉ trả về CHỮ đọc được và DANH SÁCH VẬT THỂ nhận diện được (kèm độ tin cậy).
Mô hình ngôn ngữ trả lời dựa trên đúng các thông tin này, không mô tả thêm những gì không có.

Bảo vệ đầu vào: chỉ nhận JPEG/PNG/WEBP, giới hạn số điểm ảnh (chống decompression bomb), thu nhỏ trước khi xử lý.
Ảnh chỉ xử lý trong bộ nhớ, không ghi xuống đĩa.
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path
from typing import Any

MAX_PIXELS = 40_000_000
MAX_SIDE = 2048
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
UNSURE_BELOW = 0.6  # vật thể dưới ngưỡng này được ghi "chưa chắc" trong prompt

# 80 lớp COCO -> tiếng Việt
COCO_VI = {
    "person": "người",
    "bicycle": "xe đạp",
    "car": "ô tô",
    "motorcycle": "xe máy",
    "airplane": "máy bay",
    "bus": "xe buýt",
    "train": "tàu hỏa",
    "truck": "xe tải",
    "boat": "thuyền",
    "traffic light": "đèn giao thông",
    "fire hydrant": "trụ cứu hỏa",
    "stop sign": "biển dừng",
    "parking meter": "đồng hồ đỗ xe",
    "bench": "ghế băng",
    "bird": "chim",
    "cat": "mèo",
    "dog": "chó",
    "horse": "ngựa",
    "sheep": "cừu",
    "cow": "bò",
    "elephant": "voi",
    "bear": "gấu",
    "zebra": "ngựa vằn",
    "giraffe": "hươu cao cổ",
    "backpack": "ba lô",
    "umbrella": "ô (dù)",
    "handbag": "túi xách",
    "tie": "cà vạt",
    "suitcase": "vali",
    "frisbee": "đĩa ném",
    "skis": "ván trượt tuyết",
    "snowboard": "ván trượt tuyết đơn",
    "sports ball": "bóng",
    "kite": "diều",
    "baseball bat": "gậy bóng chày",
    "baseball glove": "găng bóng chày",
    "skateboard": "ván trượt",
    "surfboard": "ván lướt sóng",
    "tennis racket": "vợt tennis",
    "bottle": "chai",
    "wine glass": "ly rượu",
    "cup": "cốc",
    "fork": "nĩa",
    "knife": "dao",
    "spoon": "thìa",
    "bowl": "bát",
    "banana": "chuối",
    "apple": "táo",
    "sandwich": "bánh mì kẹp",
    "orange": "cam",
    "broccoli": "súp lơ xanh",
    "carrot": "cà rốt",
    "hot dog": "xúc xích kẹp bánh mì",
    "pizza": "pizza",
    "donut": "bánh donut",
    "cake": "bánh ngọt",
    "chair": "ghế",
    "couch": "ghế sofa",
    "potted plant": "chậu cây",
    "bed": "giường",
    "dining table": "bàn ăn",
    "toilet": "bồn cầu",
    "tv": "tivi",
    "laptop": "laptop",
    "mouse": "chuột máy tính",
    "remote": "điều khiển từ xa",
    "keyboard": "bàn phím",
    "cell phone": "điện thoại",
    "microwave": "lò vi sóng",
    "oven": "lò nướng",
    "toaster": "máy nướng bánh mì",
    "sink": "bồn rửa",
    "refrigerator": "tủ lạnh",
    "book": "sách",
    "clock": "đồng hồ",
    "vase": "bình hoa",
    "scissors": "kéo",
    "teddy bear": "gấu bông",
    "hair drier": "máy sấy tóc",
    "toothbrush": "bàn chải đánh răng",
}


class ImageError(ValueError):
    pass


def load_image(data: bytes, max_bytes: int) -> Any:
    from PIL import Image

    if len(data) > max_bytes:
        raise ImageError(f"ảnh {len(data) / 1e6:.1f} MB vượt giới hạn {max_bytes / 1e6:.0f} MB")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        img = Image.open(io.BytesIO(data))
        if img.format not in ALLOWED_FORMATS:
            raise ImageError(f"định dạng {img.format} không hỗ trợ (chỉ JPEG/PNG/WEBP)")
        img.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise ImageError(f"không đọc được ảnh: {type(exc).__name__}") from exc
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


class VisionRuntime:
    def __init__(
        self,
        *,
        coco_weights: str,
        custom_weights: str | None = None,
        ocr_langs: tuple[str, ...] = ("vi", "en"),
        min_conf: float = 0.4,
        custom_min_conf: float = 0.6,
        ocr_min_conf: float = 0.4,
        gpu: bool = True,
    ) -> None:
        import easyocr
        from ultralytics import YOLO, settings

        settings.update({"sync": False})  # không gửi thống kê về Ultralytics
        self._lock = threading.Lock()
        self.ocr_min_conf = ocr_min_conf
        self.reader = easyocr.Reader(list(ocr_langs), gpu=gpu, verbose=False)
        # (tên, model, tên tiếng Việt, ngưỡng tin cậy). Model tự train chỉ biết vài lớp -> ngưỡng cao hơn để ít báo
        # nhầm trên ảnh ngoài lĩnh vực (đo trên tập test: precision 0.77 @0.35 -> 0.89 @0.6).
        self.models: list[tuple[str, Any, dict[str, str], float]] = [
            ("coco", YOLO(coco_weights), COCO_VI, min_conf)
        ]
        self.custom_id = None
        if custom_weights:
            names_vi = {}
            meta = Path(custom_weights).with_suffix("").parent / Path(custom_weights).stem / "meta.json"
            if meta.exists():
                names_vi = json.loads(meta.read_text())["dataset"].get("vi_names", {})
            self.models.append(("edu_ent", YOLO(custom_weights), names_vi, custom_min_conf))
            self.custom_id = Path(custom_weights).stem

    def ocr(self, img: Any) -> dict[str, Any]:
        import numpy as np

        with self._lock:
            lines = self.reader.readtext(np.array(img), detail=1, paragraph=False)
        # sắp theo dòng (y rồi x); bỏ mảnh tin cậy thấp. Đo trên trang scan tiếng Việt: dòng 0.4-0.6 vẫn là chữ đúng
        # (dấu làm giảm điểm), chữ rác trên ảnh đường phố phần lớn < 0.4.
        items = sorted(
            (
                (box, text, float(conf))
                for box, text, conf in lines
                if float(conf) >= self.ocr_min_conf
                and text.strip()
                # mảnh ngắn (< 6 ký tự) dễ là nhiễu -> cần tin cậy cao hơn
                and (len(text.strip()) >= 6 or float(conf) >= 0.6)
            ),
            key=lambda t: (round(min(p[1] for p in t[0]) / 20), min(p[0] for p in t[0])),
        )
        dropped = sum(1 for _, t, _ in lines if t.strip()) - len(items)
        text = "\n".join(t for _, t, _ in items)
        mean = sum(c for *_, c in items) / len(items) if items else 0.0
        return {"text": text, "lines": len(items), "mean_conf": round(mean, 3), "dropped_low_conf": dropped}

    def detect(self, img: Any) -> list[dict[str, Any]]:
        out = []
        with self._lock:
            for tag, model, names_vi, conf in self.models:
                res = model.predict(img, conf=conf, verbose=False)[0]
                for b in res.boxes:
                    label = res.names[int(b.cls)]
                    out.append(
                        {
                            "label": label,
                            "label_vi": names_vi.get(label, label),
                            "conf": round(float(b.conf), 3),
                            "box": [round(float(x)) for x in b.xyxy[0].tolist()],
                            "model": tag,
                        }
                    )
        return dedupe_across_models(sorted(out, key=lambda o: -o["conf"]))

    def analyze(self, img: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        ocr = self.ocr(img)
        objects = self.detect(img)
        return {
            "width": img.size[0],
            "height": img.size[1],
            "ocr": ocr,
            "objects": objects,
            "custom_model": self.custom_id,
            "ms": int((time.perf_counter() - t0) * 1000),
        }


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def dedupe_across_models(objects: list[dict[str, Any]], iou: float = 0.5) -> list[dict[str, Any]]:
    """COCO và model tự train có lớp trùng nghĩa (ghế, sách, laptop...): cùng tên tiếng Việt + hộp chồng nhau
    -> chỉ giữ một (độ tin cậy cao hơn), tránh đếm một vật hai lần. Đầu vào đã sắp theo độ tin cậy giảm dần."""
    kept: list[dict[str, Any]] = []
    for o in objects:
        if not any(
            k["label_vi"] == o["label_vi"] and k["model"] != o["model"] and _iou(k["box"], o["box"]) > iou
            for k in kept
        ):
            kept.append(o)
    return kept


def summarize_objects(objects: list[dict[str, Any]], max_items: int = 12) -> str:
    """ "người x2 (0.91), sách (0.66)" - gộp cùng nhãn, giữ độ tin cậy cao nhất."""
    agg: dict[str, list[float]] = {}
    for o in objects:
        agg.setdefault(o["label_vi"], []).append(o["conf"])
    parts = [
        f"{k}{f' x{len(v)}' if len(v) > 1 else ''} ({max(v):.2f}{', chưa chắc' if max(v) < UNSURE_BELOW else ''})"
        for k, v in sorted(agg.items(), key=lambda kv: -max(kv[1]))
    ]
    return ", ".join(parts[:max_items])
