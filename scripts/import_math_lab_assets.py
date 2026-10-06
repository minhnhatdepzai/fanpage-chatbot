#!/usr/bin/env python3
"""Nhập tài sản huấn luyện Math Lab sang ``artifacts/math_lab_import``.

Không tự đăng ký adapter vào runtime. Script giữ nguyên cấu trúc, bỏ virtualenv/cache,
và tạo manifest SHA-256 để lần nhập có thể kiểm toán.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_SOURCE = Path("/home/admin123/Downloads/EduVisionAI-Math-Lab-FE-update-and-remove-")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = PROJECT_ROOT / "artifacts" / "math_lab_import"
IGNORED_DIRS = {".venv", "__pycache__", ".pytest_cache", "node_modules"}


def _ignore(_: str, names: list[str]) -> set[str]:
    return {name for name in names if name in IGNORED_DIRS or name.endswith(".pyc")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_assets(source: Path, destination: Path) -> dict[str, object]:
    training_source = source / "training"
    if not training_source.is_dir():
        raise SystemExit(f"Không tìm thấy thư mục training: {training_source}")
    destination.mkdir(parents=True, exist_ok=True)
    imported: list[str] = []
    for name in ("math_lab", "arithmetic_quiz"):
        src = training_source / name
        dst = destination / name
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=_ignore)
        imported.append(name)

    for relative in (
        Path("frontend/public/data/visual-model.json"),
        Path("docs/GRADE_1_9_COVERAGE.md"),
        Path("docs/TEACHER_READINESS_ASSESSMENT.md"),
    ):
        src = source / relative
        if src.is_file():
            dst = destination / "project_evidence" / relative.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    files = [path for path in destination.rglob("*") if path.is_file() and path.name != "import-manifest.json"]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "imported_at": datetime.now(UTC).isoformat(),
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "excluded": sorted(IGNORED_DIRS),
        "runtime_policy": {
            "qwen3_0_6b_gsm8k_lora": "rejected: holdout exact match 14/50, no improvement, incompatible base",
            "multilingual_visual_selector_lora": "accepted for visualization selection only",
            "answers": "must be computed outside the learned visual selector",
        },
        "trees": imported,
        "files": [
            {
                "path": str(path.relative_to(destination)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(files)
        ],
    }
    manifest_path = destination / "import-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    manifest = import_assets(args.source, args.destination)
    total_bytes = sum(int(item["bytes"]) for item in manifest["files"])  # type: ignore[index]
    print(json.dumps({"files": len(manifest["files"]), "bytes": total_bytes, "destination": str(args.destination)}))


if __name__ == "__main__":
    main()
