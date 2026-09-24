"""Trả lời câu hỏi ngày/thứ bằng tính toán xác định (theo giờ Việt Nam) - không để model tự tính (model từng tính sai:
"20 ngày nữa" từ 24/09/2026 ra 6/10 thay vì 14/10).

Hỗ trợ: "hôm nay/ngày mai/hôm qua là thứ mấy/ngày nào", "N ngày|tuần|tháng|năm nữa/trước là thứ mấy/ngày nào",
"ngày 2/9(/2026) là thứ mấy". Không khớp -> None (để luồng thường xử lý).
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.conversation.intents import normalize

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]
_ASKS_DATE = re.compile(r"\b(thu may|ngay may|ngay nao|ngay bao nhieu|ngay gi|la ngay|thu bao nhieu)\b")
_OFFSET = re.compile(r"\b(\d{1,4})\s*(ngay|tuan|thang|nam)\s*(nua|toi|sau|tiep theo|truoc|qua)\b")
# văn bản đã chuẩn hóa: "/" và "-" thành khoảng trắng -> "ngay 2 9 2026" hoặc "ngay 2 thang 9 nam 2026"
_EXPLICIT = re.compile(r"\bngay\s+(\d{1,2})\s+(?:thang\s+)?(\d{1,2})(?:\s+(?:nam\s+)?(\d{4}))?\b")


def fmt(d: date) -> str:
    return f"{WEEKDAYS[d.weekday()]}, {d:%d/%m/%Y}"


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def answer_date_question(text: str | None, now: datetime) -> str | None:
    if not text:
        return None
    t = normalize(text)
    if not _ASKS_DATE.search(t):
        return None
    today = now.astimezone(VN_TZ).date()
    m = _OFFSET.search(t)
    if m:
        n, unit, direction = int(m.group(1)), m.group(2), m.group(3)
        if n > 3650 and unit == "ngay":
            return None
        sign = -1 if direction in ("truoc", "qua") else 1
        if unit == "ngay":
            target = today + timedelta(days=sign * n)
        elif unit == "tuan":
            target = today + timedelta(weeks=sign * n)
        elif unit == "thang":
            target = _add_months(today, sign * n)
        else:
            target = _add_months(today, sign * 12 * n)
        unit_vi = {"ngay": "ngày", "tuan": "tuần", "thang": "tháng", "nam": "năm"}[unit]
        dir_vi = "trước" if sign < 0 else "nữa"
        return (
            f"Hôm nay là {fmt(today)}, nên {n} {unit_vi} {dir_vi} là {fmt(target)} (tính theo giờ Việt Nam)."
        )
    m = _EXPLICIT.search(t)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else today.year
        try:
            target = date(year, month, day)
        except ValueError:
            return f"Ngày {day}/{month}/{year} không tồn tại trong lịch, bạn kiểm tra lại giúp mình nhé."
        return f"Ngày {target:%d/%m/%Y} là {WEEKDAYS[target.weekday()]}."
    if re.search(r"\bngay mai\b|\bmai la\b", t):
        return f"Ngày mai là {fmt(today + timedelta(days=1))}."
    if re.search(r"\bhom qua\b", t):
        return f"Hôm qua là {fmt(today - timedelta(days=1))}."
    if re.search(r"\bhom nay\b|\bbay gio\b", t):
        return f"Hôm nay là {fmt(today)} (theo giờ Việt Nam)."
    return None
