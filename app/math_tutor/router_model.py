"""Bộ nhận dạng đề toán nhỏ, được huấn luyện riêng; không quyết định đáp án."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[2] / "config" / "models" / "math-router-v1.json"
DIMENSIONS = 1024


def features(text: str) -> dict[int, float]:
    folded = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    folded = re.sub(r"\d+(?:[.,]\d+)?", "#", folded)
    tokens = re.findall(r"[a-z]+|#|[+*/=<>]", folded)
    terms = tokens + [" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    vector: dict[int, float] = {}
    for term in terms:
        index = int.from_bytes(hashlib.sha256(term.encode()).digest()[:4], "big") % DIMENSIONS
        vector[index] = 1.0
    norm = math.sqrt(sum(value**2 for value in vector.values())) or 1
    return {index: value / norm for index, value in vector.items()}


@lru_cache(maxsize=1)
def _load() -> dict:
    try:
        return json.loads(MODEL_PATH.read_text())
    except (OSError, ValueError):
        return {}


def math_probability(text: str) -> float:
    model = _load()
    if not model or len(model.get("weights", [])) != DIMENSIONS:
        return 0.0
    score = model["bias"] + sum(model["weights"][index] * value for index, value in features(text).items())
    return 1 / (1 + math.exp(-max(-60, min(60, score))))
