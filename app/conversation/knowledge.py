"""Kho kiến thức có nguồn (quản trị viên kiểm duyệt) + truy xuất từ khóa để trả lời có căn cứ.

Mỗi mục BẮT BUỘC có ``source`` (URL hoặc tên tài liệu kiểm chứng được); mục thiếu nguồn bị bỏ qua khi nạp.
Truy xuất trên văn bản đã bỏ dấu (hiểu cả tiếng Việt không dấu), xác định, không cần GPU:
- một mục chỉ được chọn khi câu hỏi chứa NGUYÊN VẸN ít nhất một cụm từ khóa của mục (``keywords``; thiếu thì
  lấy từ tiêu đề). Khớp từng âm tiết rời rạc dễ sai với tiếng Việt ("giải thích" ≠ "sở thích") - trích nhầm
  nguồn còn tệ hơn không có nguồn;
- các mục được chọn xếp hạng bằng BM25 (từ đơn + cặp từ liền kề).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.observability.redaction import strip_diacritics

log = logging.getLogger(__name__)

_URL = re.compile(r"https?://[^\s<>\"')\]]+")
# Từ chức năng (đã bỏ dấu) - chỉ loại khỏi từ đơn, vẫn giữ trong cặp từ.
_STOPWORDS = frozenset(
    """
    la gi cua va co khong ko k nhu the nao cho voi mot nhung cac duoc trong thi ma minh ban toi tui em anh chi
    oi a nhe nha ha vay sao de den tu ve o khi do day nay kia hay hoac rat qua lam cai con biet giup hoi
    xin cam on noi ra vao len xuong nua roi chua da dang se bi boi vi neu tai tren duoi ay u uh ok okay
    what is are the a an of to in for and or how why does do who when
    """.split()
)
_TITLE_WEIGHT = 2
_KEYWORD_WEIGHT = 3
_BM25_K1 = 1.2
_BM25_B = 0.75


def _normalize(text: str) -> list[str]:
    t = strip_diacritics(text.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return t.split()


def terms(text: str) -> list[str]:
    """Từ đơn (bỏ từ chức năng, dài >= 2) + cặp từ liền kề."""
    words = _normalize(text)
    uni = [w for w in words if w not in _STOPWORDS and len(w) >= 2]
    bi = [
        f"{a} {b}"
        for a, b in zip(words, words[1:], strict=False)
        if not (a in _STOPWORDS and b in _STOPWORDS)
    ]
    return uni + bi


@dataclass(frozen=True)
class KnowledgeEntry:
    id: str
    title: str
    source: str
    content: str
    keywords: tuple[str, ...] = ()

    def match_phrases(self) -> list[str]:
        if self.keywords:
            return [p for p in (" ".join(_normalize(k)) for k in self.keywords) if p]
        return [w for w in _normalize(self.title) if w not in _STOPWORDS and len(w) >= 3]


@dataclass(frozen=True)
class KnowledgeHit:
    entry: KnowledgeEntry
    score: float


@dataclass
class KnowledgeBase:
    entries: list[KnowledgeEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._tf: list[Counter[str]] = []
        for e in self.entries:
            tf: Counter[str] = Counter()
            for t in terms(e.content):
                tf[t] += 1
            for t in terms(e.title):
                tf[t] += _TITLE_WEIGHT
            for kw in e.keywords:
                for t in terms(kw):
                    tf[t] += _KEYWORD_WEIGHT
            self._tf.append(tf)
        self._df: Counter[str] = Counter(t for tf in self._tf for t in tf)
        lengths = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(lengths) / len(lengths)) if lengths else 1.0
        self._phrases = [e.match_phrases() for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def search(self, query: str, *, top_k: int = 3) -> list[KnowledgeHit]:
        """Các mục có cụm từ khóa xuất hiện trong câu hỏi, xếp theo BM25, tối đa ``top_k``."""
        if not self.entries or not query.strip():
            return []
        padded = f" {' '.join(_normalize(query))} "
        q = set(terms(query))
        n = len(self.entries)
        scored: list[KnowledgeHit] = []
        for entry, tf, phrases in zip(self.entries, self._tf, self._phrases, strict=True):
            if not any(f" {p} " in padded for p in phrases):
                continue
            doc_len = sum(tf.values())
            score = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                score += (
                    idf
                    * f
                    * (_BM25_K1 + 1)
                    / (f + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / self._avg_len))
                )
            scored.append(KnowledgeHit(entry, round(score, 3)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def source_urls(self) -> set[str]:
        return {u.rstrip(".,;") for e in self.entries for u in _URL.findall(e.source)}


def extract_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?") for u in _URL.findall(text or "")]


def load_knowledge(directory: Path) -> KnowledgeBase:
    """Nạp mọi ``*.yaml``/``*.yml`` trong thư mục. Mục thiếu id/title/content/source bị bỏ qua (ghi log)."""
    if not directory.is_dir():
        log.info("knowledge_dir_missing", extra={"path": str(directory)})
        return KnowledgeBase()
    entries: list[KnowledgeEntry] = []
    seen: set[str] = set()
    for path in sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for raw in data.get("entries") or []:
            if not isinstance(raw, dict):
                continue
            item = {k: str(raw.get(k) or "").strip() for k in ("id", "title", "source", "content")}
            missing = [k for k, v in item.items() if not v]
            if missing:
                log.warning(
                    "knowledge_entry_skipped", extra={"file": path.name, "id": item["id"], "missing": missing}
                )
                continue
            if item["id"] in seen:
                log.warning("knowledge_entry_duplicate", extra={"file": path.name, "id": item["id"]})
                continue
            seen.add(item["id"])
            kws = tuple(str(k).strip() for k in (raw.get("keywords") or []) if str(k).strip())
            entries.append(KnowledgeEntry(item["id"], item["title"], item["source"], item["content"], kws))
    log.info("knowledge_loaded", extra={"entries": len(entries), "path": str(directory)})
    return KnowledgeBase(entries)
