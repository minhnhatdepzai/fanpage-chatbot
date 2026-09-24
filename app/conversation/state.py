"""State của graph (chỉ kiểu dữ liệu đơn giản -> checkpoint tuần tự hóa an toàn) và ngữ cảnh lượt."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Protocol, TypedDict


class TurnState(TypedDict, total=False):
    # định danh (đầu vào)
    turn_id: str
    conversation_id: str
    # tải từ store
    page_id: str
    user_ref: str
    inbound: list[dict[str, Any]]
    history: list[dict[str, Any]]
    summary: str | None
    summary_upto_seq: int
    handoff_active: bool
    needs_disclosure: bool
    is_first_bot_reply: bool
    last_fallback_at: str | None
    latest_inbound_at: str | None
    # quyết định
    route: str  # stop | template | llm
    mode: str
    template_reply: str | None
    set_handoff: bool
    window: list[dict[str, Any]]
    new_summary: str | None
    new_summary_upto_seq: int | None
    user_text: str
    attachment_kinds: list[str]
    # độ chính xác: nguồn tham khảo đã truy xuất (id, title, source, content) + loại câu hỏi
    knowledge_question: bool
    knowledge_hits: list[dict[str, Any]]
    # ảnh: khối ngữ cảnh (OCR đã che dữ liệu cá nhân + vật thể) và thống kê gọn cho trace
    image_context: str | None
    image_note: str | None  # dữ liệu ảnh mới, gắn vào tin nhắn hiện tại khi gọi model
    image_meta: dict[str, Any]
    analyzed_images: list[dict[str, Any]]  # [{"seq", "index", "analysis"}] -> lưu vào tin nhắn để hỏi tiếp
    # model
    draft: str | None
    model_meta: dict[str, Any]
    model_error: str | None
    # kết quả cho lớp gửi tin
    check_flags: list[str]
    reply_parts: list[str]
    action: str  # reply | none
    outcome: str
    is_fallback: bool
    mark_disclosed: bool


@dataclass
class InboundItem:
    seq: int
    kind: str  # text | attachment | postback | quick_reply
    text: str | None
    attachment_types: list[str] = field(default_factory=list)
    payload: str | None = None
    event_ts: datetime | None = None
    like_sticker: bool = False
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_state(self) -> dict[str, Any]:
        d = asdict(self)
        d["event_ts"] = self.event_ts.isoformat() if self.event_ts else None
        return d


@dataclass
class HistoryItem:
    seq: int
    role: str  # user | assistant | human_agent
    content: str


@dataclass
class TurnContext:
    conversation_id: str
    page_id: str
    user_ref: str
    inbound: list[InboundItem]
    history: list[HistoryItem]
    summary: str | None = None
    summary_upto_seq: int = 0
    handoff_active: bool = False
    last_user_message_at: datetime | None = None
    last_bot_message_at: datetime | None = None
    last_disclosure_at: datetime | None = None
    last_fallback_at: datetime | None = None
    bot_resumed_at: datetime | None = None
    bot_messages_count: int = 0


class ConversationStore(Protocol):
    async def load_turn_context(
        self, conversation_id: str, turn_id: str, history_limit: int
    ) -> TurnContext: ...
