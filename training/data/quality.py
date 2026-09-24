"""Kiểm tra chất lượng dữ liệu hội thoại + phát hiện gần trùng.

Nguyên tắc: không coi tiếng lóng, lỗi gõ, không dấu hay hội thoại cảm xúc là dữ liệu xấu.
Chỉ loại: rỗng, lỗi cấu trúc, trùng lặp, spam rõ ràng, độc hại mức cao, sai ngôn ngữ.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable

from app.observability.redaction import strip_diacritics, vietnamese_score  # noqa: F401 (re-export)

_WS = re.compile(r"[ \t ]+")
_MANY_NL = re.compile(r"\n{3,}")
_URL_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$")
_REPEAT = re.compile(r"(.)\1{15,}")


def clean_text(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = "\n".join(_WS.sub(" ", line).rstrip() for line in t.split("\n"))
    t = _MANY_NL.sub("\n\n", t)
    return t.strip()


def looks_like_spam(text: str) -> bool:
    return bool(_URL_ONLY.match(text)) or bool(_REPEAT.search(text))


def content_hash(messages: Iterable[dict]) -> str:
    h = hashlib.sha256()
    for m in messages:
        h.update(m["role"].encode())
        h.update(b"\x1f")
        h.update(_canon(m["content"]).encode())
        h.update(b"\x1e")
    return h.hexdigest()


def _canon(text: str) -> str:
    t = strip_diacritics(text.lower())
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def shingles(text: str, k: int = 5) -> set[str]:
    t = _canon(text)
    if len(t) <= k:
        return {t} if t else set()
    return {t[i : i + k] for i in range(len(t) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicate_groups(texts: dict[str, str], threshold: float = 0.8) -> dict[str, str]:
    """Gom nhóm id có văn bản gần trùng (union-find, O(n^2) - đủ cho vài chục nghìn mẫu ngắn).

    Trả về map id -> id đại diện của nhóm.
    """
    ids = list(texts)
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sh = {i: shingles(texts[i]) for i in ids}
    # chặn theo 1 shingle hiếm để giảm số cặp so sánh
    buckets: dict[str, list[str]] = {}
    for i in ids:
        for s in sorted(sh[i])[:8]:
            buckets.setdefault(s, []).append(i)
    compared: set[tuple[str, str]] = set()
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                key = (a, b) if a < b else (b, a)
                if key in compared:
                    continue
                compared.add(key)
                if jaccard(sh[a], sh[b]) >= threshold:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra
    return {i: find(i) for i in ids}
