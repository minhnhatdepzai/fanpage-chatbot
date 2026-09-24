"""Tải một dataset Kaggle ĐÃ ĐƯỢC DUYỆT giấy phép vào ``data/raw`` (bản gốc, không sửa).

    uv run python -m training.data.kaggle_download mkaur1141/openassistant-conversations-dataset-oasst1 \
        --expect-license apache-2.0 --max-mb 400

- Kiểm tra giấy phép thực tế trên Kaggle khớp ``--expect-license`` trước khi tải.
- Kiểm tra dung lượng trước khi tải; không dùng ``--unzip`` của CLI mà giải nén an toàn.
- Không chạy notebook/script đi kèm dataset (chỉ cho phép csv/json/jsonl/parquet/txt/md/tsv/xlsx).
- Ghi ``download_manifest.json`` (nguồn, phiên bản, giấy phép, checksum).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from training.data.kaggle_auth import get_kaggle_api
from training.data.safe_extract import (
    VISION_LIMITS,
    ExtractLimits,
    copy_plain_file,
    safe_extract_zip,
    sha256_file,
)

_LICENSE_ALIASES = {"cc0:-public-domain": "cc0-1.0", "attribution-4.0-international-(cc-by-4.0)": "cc-by-4.0"}


def _norm_license(name: str | None) -> str:
    n = (name or "").strip().lower().replace(" ", "-")
    return _LICENSE_ALIASES.get(n, n)


def download(ref: str, expect_license: str, max_mb: float, root: Path, kind: str = "text") -> Path:
    api = get_kaggle_api(get_settings().kaggle_json_path)
    owner, slug = ref.split("/", 1)
    # metadata thật (giấy phép, phiên bản, dung lượng)
    matches = [
        d for d in (api.dataset_list(search=slug, user=owner) or []) if d is not None and str(d.ref) == ref
    ]
    if not matches:
        raise SystemExit(f"Không tìm thấy dataset {ref} qua API tìm kiếm")
    d = matches[0]
    with tempfile.TemporaryDirectory() as tmp:
        api.dataset_metadata(ref, path=tmp)
        info = json.loads((Path(tmp) / "dataset-metadata.json").read_text()).get("info", {})
    licenses = [_norm_license(x.get("name")) for x in info.get("licenses") or []]
    if _norm_license(expect_license) not in licenses:
        raise SystemExit(f"Giấy phép thực tế {licenses} không khớp --expect-license {expect_license}; dừng.")
    size_mb = (d.total_bytes or 0) / 1e6
    if size_mb > max_mb:
        raise SystemExit(f"Dataset {size_mb:.1f} MB vượt giới hạn --max-mb {max_mb}; dừng.")
    version = d.current_version_number
    out_dir = root / f"{owner}__{slug}" / f"v{version}"
    if (out_dir / "download_manifest.json").exists():
        print(f"Đã có {out_dir}, bỏ qua tải lại.")
        return out_dir
    staging = root / ".staging" / f"{owner}__{slug}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    api.dataset_download_files(ref, path=str(staging), force=True, quiet=True, unzip=False)
    downloaded = [p for p in staging.iterdir() if p.is_file()]
    if len(downloaded) != 1:
        raise SystemExit(f"Kỳ vọng 1 tệp tải về, nhận {len(downloaded)}")
    archive = downloaded[0]
    files_dir = out_dir / "files"
    if archive.suffix == ".zip":
        base = VISION_LIMITS if kind == "vision" else ExtractLimits()
        files = safe_extract_zip(archive, files_dir, replace(base, max_total_bytes=int(max_mb * 4 * 1e6)))
    else:
        files = [copy_plain_file(archive, files_dir)]
    manifest = {
        "ref": ref,
        "url": f"https://www.kaggle.com/datasets/{ref}",
        "title": info.get("title"),
        "owner": owner,
        "license": licenses,
        "user_specified_sources": info.get("userSpecifiedSources"),
        "version": version,
        "last_updated": str(d.last_updated),
        "downloaded_at": datetime.now(UTC).isoformat(),
        "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": sha256_file(archive)},
        "files": files,
    }
    (out_dir / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--expect-license", required=True)
    ap.add_argument("--max-mb", type=float, default=400)
    ap.add_argument("--root", type=Path, default=Path("data/raw/kaggle"))
    ap.add_argument("--kind", choices=["text", "vision"], default="text", help="vision: cho phép ảnh + nhãn")
    args = ap.parse_args()
    download(args.ref, args.expect_license, args.max_mb, args.root, args.kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
