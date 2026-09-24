"""Chia câu trả lời thành các tin nhắn nằm trong giới hạn ký tự của Send API.

Send API giới hạn text 2000 ký tự (xem docs/meta-setup.md về nguồn xác minh). Ta đếm theo
UTF-16 code unit (emoji ngoài BMP = 2) và mặc định dùng ngưỡng thấp hơn (1900) để an toàn.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…。])\s+|\n+")


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_message(text: str, max_chars: int, max_parts: int) -> list[str]:
    """Chia theo đoạn -> câu -> từ; cắt bớt nếu vượt ``max_parts`` (thêm dấu …)."""
    text = text.strip()
    if not text:
        return []
    if utf16_len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        if utf16_len(para) <= max_chars:
            pieces.append(para)
            continue
        for sent in [s.strip() for s in _SENTENCE_END.split(para) if s.strip()]:
            if utf16_len(sent) <= max_chars:
                pieces.append(sent)
            else:
                pieces.extend(_split_words(sent, max_chars))
    parts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if utf16_len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = piece
    if current:
        parts.append(current)
    if len(parts) > max_parts:
        parts = parts[:max_parts]
        last = parts[-1]
        while utf16_len(last) + 1 > max_chars:
            last = last[:-1]
        parts[-1] = last.rstrip() + "…"
    return parts


def _hard_cut(word: str, max_chars: int) -> tuple[str, str]:
    """Cắt phần đầu dài nhất của ``word`` vừa ``max_chars`` đơn vị UTF-16 (không tách đôi emoji)."""
    used = 0
    for i, ch in enumerate(word):
        used += utf16_len(ch)
        if used > max_chars:
            return word[:i], word[i:]
    return word, ""


def _split_words(sentence: str, max_chars: int) -> list[str]:
    out: list[str] = []
    current = ""
    for word in sentence.split(" "):
        while utf16_len(word) > max_chars:  # "từ" quá dài (URL, chuỗi emoji...)
            if current:
                out.append(current)
                current = ""
            head, word = _hard_cut(word, max_chars)
            out.append(head)
        candidate = f"{current} {word}" if current else word
        if utf16_len(candidate) <= max_chars:
            current = candidate
        else:
            out.append(current)
            current = word
    if current:
        out.append(current)
    return out
