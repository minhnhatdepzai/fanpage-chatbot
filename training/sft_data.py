"""Dựng dữ liệu SFT từ ``training/datasets/*.yaml`` với ĐÚNG system prompt production.

Mỗi ví dụ khai báo ``sources`` (id trong kho kiến thức) -> khối "Nguồn tham khảo" được dựng như lúc bot chạy
thật, nên model học đúng định dạng [n] mà lớp kiểm tra đầu ra xử lý. Thời điểm trong prompt cố định để dữ liệu
tái lập được (checksum ổn định).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from app.conversation.knowledge import KnowledgeBase
from app.conversation.profile import FanpageProfile
from app.conversation.prompts import build_system_prompt
from app.english_tutor import detect_english_task
from app.writing_tutor import detect_writing_task

PROMPT_TIME = datetime(2026, 9, 24, 10, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))


class DatasetError(ValueError):
    pass


def load_examples(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    examples = data.get("examples") or []
    seen: set[str] = set()
    for ex in examples:
        if ex["id"] in seen:
            raise DatasetError(f"trùng id {ex['id']}")
        seen.add(ex["id"])
        roles = [m["role"] for m in ex["messages"]]
        if not roles or roles[0] != "user" or roles[-1] != "assistant":
            raise DatasetError(f"{ex['id']}: hội thoại phải bắt đầu bằng user và kết thúc bằng assistant")
        if any(a == b for a, b in zip(roles, roles[1:], strict=False)):
            raise DatasetError(f"{ex['id']}: user/assistant phải xen kẽ")
    return examples


def render(ex: dict[str, Any], kb: KnowledgeBase, profile: FanpageProfile) -> list[dict[str, str]]:
    by_id = {e.id: e for e in kb.entries}
    missing = [s for s in ex.get("sources") or [] if s not in by_id]
    if missing:
        raise DatasetError(f"{ex['id']}: nguồn không có trong kho kiến thức: {missing}")
    first_user = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
    writing_task = detect_writing_task(first_user)
    english_task = detect_english_task(first_user)
    system = build_system_prompt(
        profile,
        PROMPT_TIME,
        summary=None,
        needs_disclosure=bool(ex.get("disclosure")),
        sources=[by_id[s] for s in ex.get("sources") or []],
        writing_task=writing_task.to_state() if writing_task.is_writing else None,
        english_task=english_task.to_state() if english_task.is_english else None,
    )
    return [{"role": "system", "content": system}, *ex["messages"]]


def build_splits(
    paths: Path | list[Path], kb: KnowledgeBase, profile: FanpageProfile
) -> tuple[dict[str, list[list[dict[str, str]]]], dict[str, Any]]:
    """Gộp một hoặc nhiều tệp SFT, nhưng vẫn lưu checksum của từng tệp để tái lập lần train."""
    dataset_paths = [paths] if isinstance(paths, Path) else paths
    if not dataset_paths:
        raise DatasetError("cần ít nhất một dataset")
    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in dataset_paths:
        for ex in load_examples(path):
            if ex["id"] in seen:
                raise DatasetError(f"trùng id giữa các dataset: {ex['id']}")
            seen.add(ex["id"])
            examples.append(ex)
    splits: dict[str, list[list[dict[str, str]]]] = {"train": [], "val": []}
    for ex in examples:
        splits[ex.get("split", "train")].append(render(ex, kb, profile))
    files = [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in dataset_paths
    ]
    combined_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    dataset_name = "+".join(path.stem for path in dataset_paths)
    manifest = {
        "dataset_files": files,
        "sha256": combined_sha,
        "dataset_version": f"{dataset_name}-{combined_sha[:8]}",
        "counts": {k: len(v) for k, v in splits.items()},
        "categories": _count(examples, "category"),
        "synthetic": True,
        "knowledge_entries_used": sorted({s for ex in examples for s in ex.get("sources") or []}),
        "rendered_sha256": hashlib.sha256(json.dumps(splits, ensure_ascii=False).encode()).hexdigest(),
    }
    return splits, manifest


def _count(examples: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for ex in examples:
        out[ex.get(key, "?")] = out.get(ex.get(key, "?"), 0) + 1
    return out
