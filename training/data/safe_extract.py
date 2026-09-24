"""Giải nén ZIP an toàn: chống path traversal, symlink, zip bomb, giới hạn dung lượng."""

from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class UnsafeArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExtractLimits:
    max_total_bytes: int = 2 * 1024**3  # 2 GiB sau giải nén
    max_files: int = 2000
    max_ratio: float = 200.0  # tỉ lệ nén tối đa cho mỗi tệp (chống zip bomb)
    # bỏ qua (không giải nén) thay vì dừng: vd. labels.cache của Ultralytics là pickle -> không bao giờ nạp
    skip_suffixes: tuple[str, ...] = ()
    allowed_suffixes: tuple[str, ...] = (
        ".csv",
        ".json",
        ".jsonl",
        ".parquet",
        ".txt",
        ".md",
        ".tsv",
        ".xlsx",
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _safe_member_path(dest: Path, name: str) -> Path:
    if "\x00" in name:
        raise UnsafeArchiveError(f"tên tệp chứa ký tự NUL: {name!r}")
    pp = PurePosixPath(name.replace("\\", "/"))
    if pp.is_absolute() or any(part in ("..", "") for part in pp.parts) or ":" in pp.parts[0]:
        raise UnsafeArchiveError(f"đường dẫn không an toàn trong archive: {name!r}")
    target = (dest / Path(*pp.parts)).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise UnsafeArchiveError(f"path traversal: {name!r}")
    return target


# Dataset ảnh (nhận diện vật thể): ảnh + nhãn (YOLO .txt, Pascal VOC .xml, COCO .json, data.yaml). Vẫn KHÔNG cho
# .py/.ipynb/.sh/tệp thực thi -> không bao giờ chạy mã đi kèm dataset.
VISION_LIMITS = ExtractLimits(
    max_total_bytes=4 * 1024**3,
    max_files=80000,
    skip_suffixes=(".cache",),
    allowed_suffixes=(".jpg", ".jpeg", ".png", ".webp", ".bmp", ".txt", ".xml", ".json", ".yaml", ".yml", ".csv", ".md"),
)


def safe_extract_zip(zip_path: Path, dest: Path, limits: ExtractLimits = ExtractLimits()) -> list[dict]:
    """Giải nén từng tệp (stream) sau khi kiểm tra toàn bộ archive. Trả về danh sách tệp + sha256."""
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > limits.max_files:
            raise UnsafeArchiveError(f"quá nhiều tệp: {len(infos)} > {limits.max_files}")
        total = 0
        plan: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in infos:
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise UnsafeArchiveError(f"archive chứa symlink: {info.filename!r}")
            target = _safe_member_path(dest, info.filename)
            if target.suffix.lower() in limits.skip_suffixes:
                continue
            if target.suffix.lower() not in limits.allowed_suffixes:
                raise UnsafeArchiveError(f"loại tệp không cho phép: {info.filename!r}")
            if info.compress_size and info.file_size / max(1, info.compress_size) > limits.max_ratio:
                raise UnsafeArchiveError(f"tỉ lệ nén bất thường (zip bomb?): {info.filename!r}")
            total += info.file_size
            if total > limits.max_total_bytes:
                raise UnsafeArchiveError(f"tổng dung lượng giải nén vượt {limits.max_total_bytes} bytes")
            plan.append((info, target))
        out: list[dict] = []
        for info, target in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with zf.open(info) as src, target.open("wb") as dst:
                while block := src.read(1 << 20):
                    written += len(block)
                    if written > info.file_size:  # kích thước khai báo sai
                        raise UnsafeArchiveError(f"tệp lớn hơn kích thước khai báo: {info.filename!r}")
                    dst.write(block)
            out.append(
                {"path": str(target.relative_to(dest)), "bytes": written, "sha256": sha256_file(target)}
            )
    return out


def copy_plain_file(src: Path, dest_dir: Path) -> dict:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    shutil.copy2(src, target)
    return {"path": target.name, "bytes": target.stat().st_size, "sha256": sha256_file(target)}
