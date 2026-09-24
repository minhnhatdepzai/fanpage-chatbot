"""Chia train/validation/test theo NHÓM (cùng cây hội thoại + gần trùng) và ghi manifest.

    uv run python -m training.data.split --inputs data/interim/oasst1_vi.jsonl --name oasst1-vi

- Mọi đoạn của cùng một hội thoại/nhóm gần trùng nằm trong đúng một tập.
- Kiểm tra rò rỉ gần trùng giữa các tập sau khi chia (lượt user đầu + toàn văn).
- Lưu seed, checksum từng tệp và của cả phiên bản vào ``manifest.json``.
- Tập test KHÔNG được dùng cho few-shot, RAG hay huấn luyện.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.data.quality import jaccard, near_duplicate_groups, shingles
from training.data.safe_extract import sha256_file


def load_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        with p.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def _first_user(conv: dict) -> str:
    return next((m["content"] for m in conv["messages"] if m["role"] == "user"), "")


def _full_text(conv: dict) -> str:
    return " ".join(m["content"] for m in conv["messages"])


def group_split(
    rows: list[dict], ratios: tuple[float, float, float], seed: int, near_dup_threshold: float
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    by_id = {r["conversation_id"]: r for r in rows}
    # nhóm 1: group_id gốc (cùng cây); nhóm 2: gần trùng theo lượt user đầu tiên
    dup_rep = near_duplicate_groups({cid: _first_user(r) for cid, r in by_id.items()}, near_dup_threshold)
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for cid, r in by_id.items():
        union(f"c:{cid}", f"g:{r.get('group_id', cid)}")
        union(f"c:{cid}", f"d:{dup_rep[cid]}")
    groups: dict[str, list[str]] = {}
    for cid in by_id:
        groups.setdefault(find(f"c:{cid}"), []).append(cid)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(rows)
    targets = {"train": ratios[0] * n, "validation": ratios[1] * n, "test": ratios[2] * n}
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    # gán nhóm lần lượt vào tập đang thiếu nhiều nhất (ưu tiên val/test có tối thiểu 1 nhóm)
    for k in keys:
        name = max(splits, key=lambda s: targets[s] - len(splits[s]))
        splits[name].extend(by_id[c] for c in sorted(groups[k]))
    stats = {
        "groups": len(groups),
        "merged_by_near_duplicate": sum(1 for g in groups.values() if len(g) > 1),
        "near_dup_threshold": near_dup_threshold,
    }
    return splits, stats


def leakage_report(splits: dict[str, list[dict]], threshold: float) -> dict[str, Any]:
    """Tìm cặp gần trùng giữa các tập khác nhau (so lượt user đầu và toàn văn)."""
    feats = {
        name: [(r["conversation_id"], shingles(_first_user(r)), shingles(_full_text(r))) for r in rows]
        for name, rows in splits.items()
    }
    names = list(splits)
    leaks: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for ca, fa, ta in feats[names[i]]:
                for cb, fb, tb in feats[names[j]]:
                    s1, s2 = jaccard(fa, fb), jaccard(ta, tb)
                    if s1 >= threshold or s2 >= threshold:
                        leaks.append(
                            {"a": f"{names[i]}:{ca}", "b": f"{names[j]}:{cb}", "first_user": s1, "full": s2}
                        )
    ids = {name: {r["conversation_id"] for r in rows} for name, rows in splits.items()}
    overlap_ids = sum(len(ids[a] & ids[b]) for a in ids for b in ids if a < b)
    return {"near_duplicate_pairs_across_splits": leaks, "overlapping_conversation_ids": overlap_ids}


def write_version(
    splits: dict[str, list[dict]], name: str, out_root: Path, seed: int, inputs: list[Path], extra: dict
) -> Path:
    content = hashlib.sha256()
    for s in ("train", "validation", "test"):
        for r in splits[s]:
            content.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())
    version = f"{name}-{datetime.now(UTC):%Y%m%d}-{content.hexdigest()[:8]}"
    out = out_root / version
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for s, rows in splits.items():
        p = out / f"{s}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[s] = {
            "path": p.name,
            "conversations": len(rows),
            "assistant_turns": sum(1 for r in rows for m in r["messages"] if m["role"] == "assistant"),
            "sha256": sha256_file(p),
        }
    manifest = {
        "dataset_version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in inputs],
        "sources": sorted({r["source"] for rs in splits.values() for r in rs}),
        "licenses": sorted({r["license"] for rs in splits.values() for r in rs}),
        "files": files,
        "policy": "test split must never be used for training, few-shot prompts or RAG",
        **extra,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--near-dup-threshold", type=float, default=0.8)
    args = ap.parse_args()
    rows = [r for r in load_jsonl(args.inputs) if r.get("approved_for_training")]
    splits, stats = group_split(rows, tuple(args.ratios), args.seed, args.near_dup_threshold)
    leaks = leakage_report(splits, args.near_dup_threshold)
    if leaks["overlapping_conversation_ids"] or leaks["near_duplicate_pairs_across_splits"]:
        print(json.dumps(leaks, ensure_ascii=False, indent=2))
        raise SystemExit("Phát hiện rò rỉ giữa các tập -> dừng, không ghi phiên bản dữ liệu.")
    out = write_version(
        splits, args.name, args.out_root, args.seed, args.inputs, {"grouping": stats, "leakage": leaks}
    )
    print(f"dataset_version_dir={out}")
    print((out / "manifest.json").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
