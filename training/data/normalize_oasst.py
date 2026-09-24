"""Chuẩn hóa phần tiếng Việt của OASST1 (bản Kaggle, Apache-2.0) sang định dạng hội thoại chuẩn.

    uv run python -m training.data.normalize_oasst \
        --raw data/raw/kaggle/mkaur1141__openassistant-conversations-dataset-oasst1/v1 \
        --out data/interim/oasst1_vi.jsonl

Mỗi cây hội thoại (message_tree_id) -> đúng MỘT hội thoại theo nhánh tốt nhất
(assistant có rank thấp nhất; prompter ưu tiên nhánh đi sâu nhất). Không nối các cây khác nhau.
Báo cáo số lượng trước/sau từng bước lọc được ghi vào ``<out>.report.json``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.observability.redaction import redact_pii
from training.data.quality import clean_text, content_hash, looks_like_spam, vietnamese_score

TOXICITY_MAX = 0.8
ROLE_MAP = {"prompter": "user", "assistant": "assistant"}
IDENTITY_CONTAMINATION = re.compile(r"open\s?assistant|laion", re.IGNORECASE)


def _toxicity(v: Any) -> float:
    if isinstance(v, str) and v.strip().startswith("{"):
        try:
            d = ast.literal_eval(v)
            return float(d.get("toxicity", 0.0)) if isinstance(d, dict) else 0.0
        except (ValueError, SyntaxError):
            return 0.0
    return 0.0


def _best_path(root: str, children: dict[str, list[dict]], by_id: dict[str, dict]) -> list[dict]:
    """Chọn nhánh: assistant rank tốt nhất (rank NaN xếp sau); prompter chọn nhánh sâu nhất."""

    def depth(mid: str) -> int:
        kids = children.get(mid, [])
        return 1 + (max(depth(k["message_id"]) for k in kids) if kids else 0)

    path = [by_id[root]]
    cur = root
    while children.get(cur):
        kids = children[cur]
        if kids[0]["role"] == "assistant":
            kids = sorted(
                kids, key=lambda k: (pd.isna(k["rank"]), k["rank"] if not pd.isna(k["rank"]) else 0)
            )
            nxt = kids[0]
        else:
            nxt = max(kids, key=lambda k: depth(k["message_id"]))
        path.append(nxt)
        cur = nxt["message_id"]
    return path


def normalize(raw_dir: Path, source_license: str) -> tuple[list[dict], dict[str, Any]]:
    manifest = json.loads((raw_dir / "download_manifest.json").read_text())
    files = raw_dir / "files"
    frames = [pd.read_csv(files / f["path"]) for f in manifest["files"] if f["path"].endswith(".csv")]
    df = pd.concat(frames, ignore_index=True)
    report: dict[str, Any] = {"source": manifest["ref"], "source_version": manifest["version"], "steps": []}

    def step(name: str, n: int, note: str = "") -> None:
        report["steps"].append({"step": name, "count": int(n), "note": note})

    step("raw_messages_all_languages", len(df))
    vi = df[df["lang"] == "vi"].copy()
    step("messages_lang_vi", len(vi))
    step("trees_lang_vi", vi["message_tree_id"].nunique())

    vi = vi[(vi["deleted"] == False) & (vi["synthetic"] == False)]  # noqa: E712
    step("after_drop_deleted_synthetic", len(vi))
    vi = vi[vi["review_result"] != False]  # noqa: E712  (giữ True/NaN: NaN = chưa review xong)
    step("after_drop_review_rejected", len(vi))
    vi["text"] = vi["text"].fillna("").map(clean_text)
    vi = vi[vi["text"].str.len() > 0]
    step("after_drop_empty", len(vi))

    records = vi.to_dict("records")
    by_id = {r["message_id"]: r for r in records}
    children: dict[str, list[dict]] = defaultdict(list)
    roots: list[str] = []
    for r in records:
        pid = r["parent_id"]
        if isinstance(pid, str) and pid in by_id:
            children[pid].append(r)
        elif not isinstance(pid, str) or pd.isna(pid):
            roots.append(r["message_id"])
        # tin có parent bị loại -> nhánh con bị cắt (không nối sang nơi khác)
    step("root_prompts_kept", len(roots))

    convs: list[dict] = []
    dropped = defaultdict(int)
    seen_hash: set[str] = set()
    pii_hits = 0
    for root in roots:
        path = _best_path(root, children, by_id)
        msgs = [{"role": ROLE_MAP[m["role"]], "content": m["text"]} for m in path if m["role"] in ROLE_MAP]
        # bắt buộc xen kẽ user/assistant, bắt đầu bằng user, kết thúc bằng assistant
        if not msgs or msgs[0]["role"] != "user":
            dropped["not_starting_with_user"] += 1
            continue
        if any(msgs[i]["role"] == msgs[i + 1]["role"] for i in range(len(msgs) - 1)):
            dropped["roles_not_alternating"] += 1
            continue
        while msgs and msgs[-1]["role"] != "assistant":
            msgs.pop()
        if len(msgs) < 2:
            dropped["no_assistant_reply"] += 1
            continue
        if max(_toxicity(m.get("detoxify")) for m in path) > TOXICITY_MAX:
            dropped[f"toxicity_gt_{TOXICITY_MAX:.1f}"] += 1
            continue
        if any(looks_like_spam(m["content"]) for m in msgs):
            dropped["spam_pattern"] += 1
            continue
        # Trợ lý tự xưng "Open Assistant" -> sẽ dạy model sai danh tính (bot phải là trợ lý AI của fanpage)
        if any(m["role"] == "assistant" and IDENTITY_CONTAMINATION.search(m["content"]) for m in msgs):
            dropped["assistant_identity_contamination"] += 1
            continue
        joined = " ".join(m["content"] for m in msgs)
        # 'vi' của OASST1 là nhãn người dùng chọn -> kiểm tra lại tín hiệu tiếng Việt (có dấu)
        if vietnamese_score(joined) < 0.03:
            dropped["low_vietnamese_signal"] += 1
            continue
        red = [{"role": m["role"], "content": redact_pii(m["content"])} for m in msgs]
        if red != msgs:
            pii_hits += 1
        h = content_hash(red)
        if h in seen_hash:
            dropped["exact_duplicate"] += 1
            continue
        seen_hash.add(h)
        convs.append(
            {
                "conversation_id": f"oasst1-vi-{root}",
                "group_id": str(path[0]["message_tree_id"]),
                "messages": red,
                "source": f"kaggle:{manifest['ref']}@v{manifest['version']} (upstream: OpenAssistant/oasst1)",
                "license": source_license,
                "language": "vi",
                "approved_for_training": True,
                "approval_basis": "license:apache-2.0 + automated QC (deleted/synthetic/review/toxicity/spam/dup/PII)",
                "synthetic": False,
                "num_turns": sum(1 for m in red if m["role"] == "user"),
            }
        )
    for k, v in dropped.items():
        step(f"dropped:{k}", v)
    step("conversations_with_pii_redacted", pii_hits)
    step("conversations_final", len(convs))
    report["multi_turn_conversations"] = sum(1 for c in convs if c["num_turns"] >= 2)
    return convs, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/interim/oasst1_vi.jsonl"))
    ap.add_argument("--license", default="apache-2.0")
    args = ap.parse_args()
    convs, report = normalize(args.raw, args.license)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for c in convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    Path(str(args.out) + ".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
