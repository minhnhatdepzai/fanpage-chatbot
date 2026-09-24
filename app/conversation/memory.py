"""Bộ nhớ ngắn hạn: chọn cửa sổ lịch sử theo ngân sách token và quyết định khi nào tóm tắt.

Ước lượng token = số ký tự / ``chars_per_token`` (mặc định 2.5, thận trọng cho tiếng Việt
với tokenizer Qwen3; đã đo thực tế - xem docs/architecture.md). Có thể đổi qua cấu hình.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MESSAGE_OVERHEAD_TOKENS = 6  # token điều khiển của chat template cho mỗi message


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return math.ceil(len(text or "") / max(0.5, chars_per_token)) + MESSAGE_OVERHEAD_TOKENS


@dataclass(frozen=True)
class HistoryPlan:
    window: list[dict]  # tin sẽ đưa nguyên văn vào prompt (cũ -> mới)
    to_summarize: list[dict]  # tin cũ nên gộp vào tóm tắt
    window_tokens: int
    unsummarized_tokens: int


def plan_history(
    history: list[dict],
    *,
    budget_tokens: int,
    summary_trigger_tokens: int,
    keep_recent: int,
    chars_per_token: float,
) -> HistoryPlan:
    """``history``: các tin CHƯA được tóm tắt (cũ -> mới), mỗi tin có 'content'.

    - Luôn giữ nguyên văn ``keep_recent`` tin gần nhất nếu vừa ngân sách.
    - Nếu tổng token chưa tóm tắt vượt ``summary_trigger_tokens`` -> đề xuất tóm tắt phần cũ.
    - Cửa sổ được lấy từ mới về cũ cho tới khi hết ngân sách (không cắt giữa một tin).
    """
    costs = [estimate_tokens(m["content"], chars_per_token) for m in history]
    total = sum(costs)
    window: list[dict] = []
    used = 0
    for msg, cost in zip(reversed(history), reversed(costs), strict=True):
        if used + cost > budget_tokens:
            break
        window.append(msg)
        used += cost
    window.reverse()
    to_summarize: list[dict] = []
    if total > summary_trigger_tokens and len(history) > keep_recent:
        cut = len(history) - keep_recent
        to_summarize = history[:cut]
        # sau khi tóm tắt, cửa sổ chỉ cần phần gần đây
        window = [m for m in window if m in history[cut:]]
        used = sum(estimate_tokens(m["content"], chars_per_token) for m in window)
    return HistoryPlan(
        window=window, to_summarize=to_summarize, window_tokens=used, unsummarized_tokens=total
    )


def format_for_summary(messages: list[dict]) -> str:
    role_name = {"user": "Người dùng", "assistant": "Trợ lý", "human_agent": "Quản trị viên"}
    return "\n".join(f"{role_name.get(m['role'], m['role'])}: {m['content']}" for m in messages)
