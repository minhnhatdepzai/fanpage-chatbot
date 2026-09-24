"""Khảo sát dataset hội thoại tiếng Việt trên Kaggle (chỉ đọc metadata, không tải dữ liệu).

    uv run python -m training.data.kaggle_search --out data/kaggle/search_results.json

Không hardcode slug: mọi ứng viên đều lấy từ kết quả tìm kiếm thật tại thời điểm chạy.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from training.data.kaggle_auth import get_kaggle_api

DEFAULT_QUERIES = [
    "vietnamese conversation",
    "vietnamese dialogue",
    "vietnamese dialog",
    "vietnamese chatbot",
    "vietnamese chat",
    "vietnamese conversational",
    "vietnamese multi-turn",
    "vietnamese sharegpt",
    "vietnamese instruction",
    "vietnamese daily dialog",
    "vietnamese customer support",
    "vietnamese messenger",
    "tieng viet hoi thoai",
    "hội thoại tiếng việt",
    "multilingual conversation",
    "multilingual dialogue",
    "multilingual chat",
]


# Dataset ảnh cho nhận diện vật thể (YOLO) theo chủ đề fanpage: giáo dục, giải trí, đời sống
VISION_QUERIES = [
    "school supplies object detection",
    "stationery object detection yolo",
    "classroom object detection yolo",
    "book detection yolo",
    "musical instruments detection yolo",
    "musical instrument object detection",
    "board game detection yolo",
    "toys object detection yolo",
    "vietnamese food detection",
    "vietnamese food yolo",
    "vietnamese banknote detection",
    "vietnamese yolo dataset",
]
PRESETS = {"conversation": DEFAULT_QUERIES, "vision": VISION_QUERIES}


def _get(d: Any, name: str) -> Any:
    return getattr(d, name, None)


def search(api, queries: list[str], pages: int = 2) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for q in queries:
        for page in range(1, pages + 1):
            try:
                res = api.dataset_list(search=q, sort_by="votes", page=page) or []
            except Exception as exc:  # noqa: BLE001 - ghi nhận lỗi truy vấn, tiếp tục
                print(f"[warn] query={q!r} page={page}: {type(exc).__name__}")
                break
            if not res:
                break
            for d in res:
                if d is None:
                    continue
                ref = str(_get(d, "ref"))
                row = seen.setdefault(
                    ref,
                    {
                        "ref": ref,
                        "url": f"https://www.kaggle.com/datasets/{ref}",
                        "title": _get(d, "title"),
                        "subtitle": _get(d, "subtitle"),
                        "owner": _get(d, "owner_name") or _get(d, "creator_name"),
                        "license": _get(d, "license_name"),
                        "total_bytes": _get(d, "total_bytes"),
                        "last_updated": str(_get(d, "last_updated")),
                        "version": _get(d, "current_version_number"),
                        "downloads": _get(d, "download_count"),
                        "votes": _get(d, "vote_count"),
                        "usability": _get(d, "usability_rating"),
                        "matched_queries": [],
                    },
                )
                if q not in row["matched_queries"]:
                    row["matched_queries"].append(q)
    return sorted(seen.values(), key=lambda r: (-(r["votes"] or 0), r["ref"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/kaggle/search_results.json"))
    ap.add_argument("--pages", type=int, default=2)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="conversation")
    args = ap.parse_args()
    api = get_kaggle_api(get_settings().kaggle_json_path)
    queries = PRESETS[args.preset]
    rows = search(api, queries, pages=args.pages)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"searched_at": datetime.now(UTC).isoformat(), "queries": queries, "results": rows}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} dataset -> {args.out}")
    for r in rows:
        size_mb = (r["total_bytes"] or 0) / 1e6
        print(f"- {r['ref']} | {r['license']} | {size_mb:.1f}MB | votes={r['votes']} | {r['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
