"""Xác thực Kaggle an toàn, không sao chép credential vào dự án.

Kaggle CLI 2.x (đã đọc mã nguồn kaggle 2.2.4):
- thư mục cấu hình lấy từ ``KAGGLE_CONFIG_DIR`` NGAY LÚC IMPORT; nếu không có sẽ tự tạo
  ``~/.kaggle`` hoặc ``~/.config/kaggle``;
- thứ tự xác thực: access token (KAGGLE_API_TOKEN / access_token) -> khóa legacy trong
  kaggle.json (username/key) -> OAuth.

Vì vậy module này đặt ``KAGGLE_CONFIG_DIR`` = thư mục chứa kaggle.json *trước* khi import
``kaggle`` và không bao giờ in giá trị ``key``.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class KaggleCredentialError(RuntimeError):
    pass


@dataclass(frozen=True)
class CredentialCheck:
    path: Path
    exists: bool
    valid_json: bool
    has_username: bool
    has_key: bool
    world_or_group_readable: bool

    @property
    def ok(self) -> bool:
        return self.exists and self.valid_json and self.has_username and self.has_key

    def describe(self) -> str:
        if not self.exists:
            return (
                f"Không tìm thấy {self.path}. Tải kaggle.json tại https://www.kaggle.com/settings (mục API)."
            )
        if not self.valid_json:
            return f"{self.path} không phải JSON hợp lệ. Tải lại tệp từ trang Settings của Kaggle."
        if not (self.has_username and self.has_key):
            return f"{self.path} thiếu trường 'username' hoặc 'key'."
        msg = f"{self.path}: hợp lệ (có username, key)."
        if self.world_or_group_readable:
            msg += f" CẢNH BÁO: tệp đang đọc được bởi người dùng khác -> nên chạy: chmod 600 {self.path}"
        return msg


def check_kaggle_json(path: Path) -> CredentialCheck:
    """Kiểm tra cấu trúc kaggle.json mà không đọc/in giá trị ra ngoài."""
    exists = path.is_file()
    valid = has_user = has_key = readable = False
    if exists:
        mode = path.stat().st_mode
        readable = bool(mode & (stat.S_IRGRP | stat.S_IROTH))
        try:
            data: Any = json.loads(path.read_text(encoding="utf-8"))
            valid = isinstance(data, dict)
            if valid:
                has_user = isinstance(data.get("username"), str) and bool(data["username"])
                has_key = isinstance(data.get("key"), str) and bool(data["key"])
        except (OSError, ValueError):
            valid = False
    return CredentialCheck(path, exists, valid, has_user, has_key, readable)


def get_kaggle_api(kaggle_json_path: Path):
    """Trả về ``KaggleApi`` đã xác thực bằng kaggle.json tại đường dẫn cho trước."""
    check = check_kaggle_json(kaggle_json_path)
    if not check.ok:
        raise KaggleCredentialError(check.describe())
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_json_path.parent)
    # Không để biến token kiểu mới (nếu có trong môi trường) chen trước tệp đã chỉ định.
    os.environ.pop("KAGGLE_API_TOKEN", None)
    from kaggle.api.kaggle_api_extended import KaggleApi  # import sau khi đặt biến môi trường

    api = KaggleApi()
    if Path(api.config) != kaggle_json_path:
        raise KaggleCredentialError(f"Kaggle đang đọc cấu hình ở {api.config}, không phải {kaggle_json_path}")
    try:
        api.authenticate()
    except SystemExit as exc:  # thư viện gọi exit(1) khi không có credential
        raise KaggleCredentialError("Kaggle từ chối xác thực. Kiểm tra lại kaggle.json.") from exc
    return api
