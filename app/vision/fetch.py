"""Tải ảnh người dùng gửi qua Messenger (URL CDN của Meta) một cách an toàn.

- Chỉ HTTPS và chỉ các host thuộc danh sách cho phép (mặc định *.fbcdn.net, *.fbsbx.com) -> chống SSRF.
- Tự xử lý chuyển hướng (tối đa 3 lần), mỗi lần đều kiểm tra lại host.
- Đọc dạng stream, dừng khi vượt giới hạn dung lượng; chỉ nhận Content-Type image/*.
"""

from __future__ import annotations

from urllib.parse import urljoin, urlparse

import httpx


class ImageFetchError(RuntimeError):
    pass


def host_allowed(url: str, allowed_suffixes: list[str]) -> bool:
    u = urlparse(url)
    host = (u.hostname or "").lower()
    return u.scheme == "https" and any(host == s or host.endswith("." + s) for s in allowed_suffixes)


async def fetch_image(
    url: str, *, allowed_suffixes: list[str], max_bytes: int, timeout: float = 20.0
) -> bytes:
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for _ in range(4):
            if not host_allowed(url, allowed_suffixes):
                raise ImageFetchError("host ảnh không nằm trong danh sách cho phép")
            async with client.stream("GET", url) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code != 200:
                    raise ImageFetchError(f"HTTP {r.status_code}")
                if not r.headers.get("content-type", "").lower().startswith("image/"):
                    raise ImageFetchError("không phải ảnh")
                buf = bytearray()
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    if len(buf) > max_bytes:
                        raise ImageFetchError("ảnh vượt giới hạn dung lượng")
                return bytes(buf)
    raise ImageFetchError("chuyển hướng quá nhiều lần")
