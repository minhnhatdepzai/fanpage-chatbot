"""Kiểm tra cơ học cho các thể thơ có cấu trúc rõ trước khi gửi.

Không chấm chất lượng nghệ thuật. Bộ kiểm tra chỉ bắt lỗi đếm dòng/tiếng và điệp khúc mà code có thể xác minh;
những yếu tố như thanh luật nguyên ngữ, nhịp iambic hay chiều sâu hình ảnh vẫn cần đánh giá riêng.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.observability.redaction import strip_diacritics


@dataclass(frozen=True, slots=True)
class CreativeValidation:
    valid: bool
    issues: tuple[str, ...]
    line_counts: tuple[int, ...] = ()


_AI_PREFIX = re.compile(r"^\s*(?:mình|tôi|em)\s+là\s+trợ lý ai tự động[^.!?\n]*[.!?]?\s*", re.I)
_COMMENTARY = re.compile(
    r"^[\[(（* _-]*(?:ghi chú\s*:|chú thích\s*:|nguồn\s*:|note\s*:|sources?\s*:|---+)", re.I
)
_WORD = re.compile(r"[\wÀ-ỹ]+(?:['’][\wÀ-ỹ]+)?", re.UNICODE)


def _verse_lines(text: str) -> list[str]:
    text = _AI_PREFIX.sub("", text.strip(), count=1)
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("*_`> ")
        if not line:
            continue
        if _COMMENTARY.match(line):
            break
        # Bỏ nhan đề rõ ràng, nhưng không đoán một câu thơ ngắn là nhan đề.
        if not lines and re.match(r"^(?:nhan đề|tựa đề|title)\s*:", strip_diacritics(line.lower())):
            continue
        lines.append(line)
    return lines


def _syllables(line: str) -> int:
    # Với tiếng Việt, mỗi token chữ thường tương ứng gần một tiếng; đủ để kiểm 5-7-5/lục bát cơ học.
    return len(_WORD.findall(line))


def _line_target(lines: list[str], target: tuple[int, ...], label: str) -> CreativeValidation:
    counts = tuple(_syllables(line) for line in lines)
    issues: list[str] = []
    if len(lines) != len(target):
        issues.append(f"{label} cần {len(target)} dòng, bản nháp có {len(lines)} dòng")
    if len(lines) == len(target) and counts != target:
        issues.append(
            f"số tiếng từng dòng cần {'-'.join(map(str, target))}, hiện là {'-'.join(map(str, counts))}"
        )
    return CreativeValidation(not issues, tuple(issues), counts)


def validate_creative_output(
    form: str | None, text: str, *, strict_output_only: bool = False
) -> CreativeValidation:
    """Kiểm tra hình thức đã nhận diện; form không có luật cơ học trả về hợp lệ."""
    lines = _verse_lines(text)
    issues: list[str] = []
    if strict_output_only and any(_COMMENTARY.match(raw.strip().strip("*_`> ")) for raw in text.splitlines()):
        issues.append("người dùng yêu cầu chỉ đưa tác phẩm nhưng bản nháp còn phần ghi chú/nguồn")

    if form in {"haiku", "senryu"}:
        base = _line_target(lines, (5, 7, 5), form)
    elif form == "tanka":
        base = _line_target(lines, (5, 7, 5, 7, 7), "tanka")
    elif form == "sijo":
        counts = tuple(_syllables(line) for line in lines)
        local = []
        if len(lines) != 3:
            local.append(f"sijo cần 3 dòng dài, bản nháp có {len(lines)} dòng")
        elif any(not 14 <= count <= 16 for count in counts):
            local.append(f"bản chuyển thể sijo cần khoảng 14-16 tiếng mỗi dòng, hiện là {counts}")
        base = CreativeValidation(not local, tuple(local), counts)
    elif form == "jueju":
        base = CreativeValidation(
            len(lines) == 4, () if len(lines) == 4 else (f"tuyệt cú cần 4 dòng, hiện có {len(lines)}",)
        )
    elif form == "lushi":
        base = CreativeValidation(
            len(lines) == 8, () if len(lines) == 8 else (f"luật thi cần 8 dòng, hiện có {len(lines)}",)
        )
    elif form in {"sonnet_shakespearean", "sonnet_petrarchan"}:
        base = CreativeValidation(
            len(lines) == 14, () if len(lines) == 14 else (f"sonnet cần 14 dòng, hiện có {len(lines)}",)
        )
    elif form == "villanelle":
        local = []
        if len(lines) != 19:
            local.append(f"villanelle cần 19 dòng, hiện có {len(lines)}")
        elif lines[0].casefold() != lines[-1].casefold() or lines[2].casefold() != lines[-2].casefold():
            local.append("hai câu điệp mở đầu chưa trở lại ở hai dòng kết")
        base = CreativeValidation(not local, tuple(local))
    elif form == "luc_bat":
        counts = tuple(_syllables(line) for line in lines)
        target = tuple(6 if i % 2 == 0 else 8 for i in range(len(lines)))
        local = []
        if len(lines) < 2 or len(lines) % 2:
            local.append("lục bát cần số dòng chẵn, tối thiểu một cặp 6-8")
        if counts != target:
            local.append(f"số tiếng cần luân phiên 6-8, hiện là {counts}")
        base = CreativeValidation(not local, tuple(local), counts)
    else:
        base = CreativeValidation(True, ())

    all_issues = (*base.issues, *issues)
    return CreativeValidation(not all_issues, all_issues, base.line_counts)


_OUTPUT_ONLY = re.compile(
    r"\b(?:chi dua|chi viet|chi tra|khong giai thich|khong ghi chu|output only|only the (?:poem|story|text))\b"
)


def requests_output_only(normalized_user_text: str) -> bool:
    return bool(_OUTPUT_ONLY.search(normalized_user_text))


def repair_contract(form: str | None) -> str:
    contracts = {
        "haiku": "Đúng 3 dòng. Đếm theo từ cách nhau bằng khoảng trắng: dòng 1 đúng 5, dòng 2 đúng 7, dòng 3 đúng 5 tiếng.",
        "senryu": "Đúng 3 dòng. Đếm theo từ cách nhau bằng khoảng trắng: dòng 1 đúng 5, dòng 2 đúng 7, dòng 3 đúng 5 tiếng.",
        "tanka": "Đúng 5 dòng, số tiếng lần lượt chính xác 5-7-5-7-7.",
        "sijo": "Đúng 3 dòng dài; mỗi dòng 14-16 tiếng và có một chỗ ngắt ý tự nhiên gần giữa dòng.",
        "jueju": "Đúng 4 dòng thơ, không thêm nhan đề hoặc lời bình nếu người dùng không yêu cầu.",
        "lushi": "Đúng 8 dòng thơ; các cặp dòng giữa phải có quan hệ đối/parallel rõ.",
        "sonnet_shakespearean": "Đúng 14 dòng: 3 quatrain và 1 couplet kết; giữ volta và hướng tới vần ABAB CDCD EFEF GG.",
        "sonnet_petrarchan": "Đúng 14 dòng: octave rồi sestet, có volta rõ giữa hai phần.",
        "villanelle": "Đúng 19 dòng: 5 tercet rồi 1 quatrain; dòng 1 và 3 trở lại luân phiên và là hai dòng kết.",
        "luc_bat": "Số dòng chẵn; số tiếng luân phiên chính xác 6-8 từ đầu đến cuối.",
    }
    return contracts.get(form or "", "Giữ đúng hình thức mà yêu cầu gốc đã nêu.")


def validation_distance(result: CreativeValidation) -> int:
    """Điểm lỗi để giữ bản ít sai hơn nếu mọi lượt sửa đều thất bại."""
    return len(result.issues) * 100 + sum(abs(value - 6) for value in result.line_counts)


def constrained_shape(form: str | None) -> tuple[int, ...] | None:
    return {
        "haiku": (5, 7, 5),
        "senryu": (5, 7, 5),
        "tanka": (5, 7, 5, 7, 7),
        "sijo": (15, 15, 15),
    }.get(form or "")


def parse_constrained_lines(raw: str, form: str | None) -> str | None:
    """Đọc JSON hoặc dòng ``D1=từng|tiếng``; cắt phần dư, không bao giờ tự bịa phần thiếu."""
    shape = constrained_shape(form)
    if not shape:
        return None
    lines: object = None
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        try:
            lines = json.loads(match.group(0))["lines"]
        except (json.JSONDecodeError, KeyError, TypeError):
            lines = None
    if not isinstance(lines, list):
        slot_lines = []
        for value in raw.splitlines():
            value = re.sub(r"^\s*(?:D|dòng|dong|line)\s*\d+\s*[=:]\s*", "", value.strip())
            if "|" in value:
                slot_lines.append(value.split("|"))
        lines = slot_lines
    if len(lines) != len(shape):
        return None
    rendered: list[str] = []
    for items, expected in zip(lines, shape, strict=True):
        if not isinstance(items, list):
            return None
        # Model nhỏ đôi khi đặt cả dòng vào một phần tử; tách lại theo token rồi khóa đúng số ô.
        tokens = [word for item in items for word in _WORD.findall(str(item))]
        if len(tokens) < expected:
            return None
        rendered.append(" ".join(tokens[:expected]))
    text = "\n".join(rendered)
    return text if validate_creative_output(form, text).valid else None
