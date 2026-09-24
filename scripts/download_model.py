"""Tải base model từ Hugging Face Hub theo revision cố định, có giới hạn dung lượng.

    uv run python scripts/download_model.py                       # dùng BASE_MODEL_ID/REVISION trong .env
    uv run python scripts/download_model.py --max-gb 20 --dry-run

Chỉ tải safetensors + config + tokenizer (không tải *.bin, gguf, onnx...).
"""

from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi, snapshot_download

from app.config import get_settings

ALLOW = [
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.jinja",
    "tokenizer.model",
    "*.tiktoken",
    "LICENSE*",
    "README.md",
]


def main() -> int:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=s.base_model_id)
    ap.add_argument("--revision", default=s.base_model_revision)
    ap.add_argument("--max-gb", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    info = HfApi().model_info(args.model, revision=args.revision, files_metadata=True)
    lic = (info.card_data or {}).get("license") if info.card_data else None
    total = 0
    for sib in info.siblings or []:
        name = sib.rfilename
        if any(__import__("fnmatch").fnmatch(name, pat) for pat in ALLOW):
            total += sib.size or 0
    gb = total / 1e9
    print(
        f"model={args.model} revision={info.sha} license={lic} gated={info.gated} download_size={gb:.2f} GB"
    )
    if info.sha != args.revision:
        print(f"CẢNH BÁO: revision yêu cầu {args.revision} khác sha thực {info.sha}", file=sys.stderr)
    if gb > args.max_gb:
        print(f"Dừng: dung lượng {gb:.2f} GB vượt giới hạn --max-gb {args.max_gb}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    path = snapshot_download(args.model, revision=args.revision, allow_patterns=ALLOW)
    print(f"downloaded_to={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
