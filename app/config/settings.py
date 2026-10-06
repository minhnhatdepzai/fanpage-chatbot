"""Cấu hình ứng dụng (đọc từ biến môi trường / .env).

Bí mật dùng ``SecretStr`` để không vô tình in ra khi repr/log.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# BOT_ENV_FILE="" -> không đọc tệp .env nào (test tự động dùng cách này để không chạm bí mật thật).
_ENV_FILE = os.environ.get("BOT_ENV_FILE", str(PROJECT_ROOT / ".env")) or None


class AppEnv(StrEnum):
    dev = "dev"
    test = "test"
    production = "production"


class SendMode(StrEnum):
    """Chế độ gửi tin Messenger.

    - ``disabled``: không gọi Send API, chỉ ghi lại câu trả lời (dry-run).
    - ``allowlist``: chỉ gửi tới PSID trong ``MESSENGER_ALLOWED_PSIDS`` (smoke test).
    - ``all``: gửi cho mọi người nhắn tới Page (production).
    """

    disabled = "disabled"
    allowlist = "allowlist"
    all = "all"


class GroundingMode(StrEnum):
    """Cách xử lý câu hỏi kiến thức/sự kiện khi KHÔNG có nguồn kiểm chứng trong kho kiến thức.

    - ``annotate``: model vẫn trả lời nhưng hệ thống bắt buộc gắn ghi chú "chưa có nguồn kiểm chứng".
    - ``strict``: không gọi model, trả lời mẫu rằng chưa có nguồn nên không trả lời đoán.
    """

    annotate = "annotate"
    strict = "strict"


class WebSearchMode(StrEnum):
    """Khi nào dùng nguồn web (chỉ có hiệu lực khi WEB_SEARCH_ENABLED=true)."""

    auto = "auto"  # câu hỏi cần thông tin mới hoặc kho nội bộ không có kết quả
    always = "always"  # mọi câu hỏi kiến thức


class LLMProvider(StrEnum):
    openai_compatible = "openai_compatible"
    anthropic = "anthropic"
    fake = "fake"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Ứng dụng ---
    app_env: AppEnv = AppEnv.dev
    log_level: str = "INFO"
    # Mặc định KHÔNG ghi nội dung tin nhắn vào log.
    log_content: bool = False
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- Database ---
    database_url: SecretStr = SecretStr("postgresql+psycopg://chatbot:chatbot@127.0.0.1:55432/chatbot")
    db_pool_size: int = 10
    db_connect_timeout_seconds: float = 5.0

    # --- Meta / Messenger ---
    meta_app_id: str = ""
    meta_app_secret: SecretStr = SecretStr("")
    meta_page_id: str = ""
    meta_page_access_token: SecretStr = SecretStr("")
    meta_verify_token: SecretStr = SecretStr("")
    meta_graph_api_version: str = "v26.0"
    meta_graph_base_url: str = "https://graph.facebook.com"
    webhook_max_body_bytes: int = 1_000_000

    messenger_send_mode: SendMode | None = None  # None -> mặc định theo APP_ENV
    messenger_allowed_psids: Annotated[list[str], NoDecode] = Field(default_factory=list)
    messenger_max_chars_per_message: int = 1900  # giới hạn Send API: 2000 ký tự (xem docs)
    messenger_max_messages_per_reply: int = 3
    messenger_send_timeout_seconds: float = 15.0
    messenger_send_max_attempts: int = 3
    messenger_typing_indicator: bool = True
    # Echo của tin do con người gửi từ hộp thư Page -> tự bật handoff.
    auto_handoff_on_human_reply: bool = True

    # --- Chính sách hội thoại ---
    messaging_window_hours: float = 24.0
    messaging_window_safety_minutes: float = 5.0
    reply_max_age_seconds: int = 600  # tin quá hạn hơn mức này thì không trả lời nữa
    debounce_seconds: float = 0.5
    debounce_max_seconds: float = 2.0
    # tin chỉ có ảnh/tệp: chờ thêm để gộp câu hỏi gửi ngay sau (người dùng hay gửi ảnh rồi mới gõ câu hỏi)
    debounce_attachment_seconds: float = 6.0
    max_bot_messages_per_minute: int = 6
    ai_disclosure_gap_hours: float = 24.0
    fallback_cooldown_seconds: int = 600
    handoff_auto_resume_hours: float = 0.0  # 0 = không tự bật lại bot
    handoff_notify_webhook_url: SecretStr = SecretStr("")
    fanpage_profile_path: Path = PROJECT_ROOT / "config" / "fanpage_profile.yaml"

    # --- Độ chính xác: kho kiến thức có nguồn ---
    knowledge_dir: Path = PROJECT_ROOT / "config" / "knowledge"
    knowledge_top_k: int = 3
    # Mặc định fail-closed: câu hỏi kiến thức không có nguồn sẽ không được model đoán.
    grounding_mode: GroundingMode = GroundingMode.strict

    # --- RAG tài liệu PDF/Word (pgvector) ---
    rag_docs_enabled: bool = True
    embedding_model_id: str = "AITeamVN/Vietnamese_Embedding"
    embedding_model_revision: str = "dea33aa1ab339f38d66ae0a40e6c40e0a9249568"
    embedding_dim: int = 1024
    embedding_base_url: str = "http://127.0.0.1:8100/v1"  # endpoint /embeddings của model server
    embedding_timeout_seconds: float = 20.0
    doc_top_k: int = 3
    doc_min_similarity: float = (
        0.40  # cosine; hiệu chỉnh 24/09/2026: liên quan 0.32-0.59, không liên quan <= 0.31 (docs/rag.md)
    )
    docs_max_file_mb: int = 50
    docs_max_pages: int = 500
    chunk_max_chars: int = 900
    chunk_overlap_chars: int = 150

    # --- Khung chat website ---
    web_chat_enabled: bool = True
    # tên miền được nhúng widget (CORS), phân tách dấu phẩy, vd. https://example.vn,https://www.example.vn
    web_allowed_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    web_max_message_chars: int = 1000
    web_rate_limit_per_5min: int = 20

    # --- Ảnh: OCR + nhận diện vật thể (model server) ---
    vision_enabled: bool = True
    vision_timeout_seconds: float = 60.0
    vision_max_image_mb: int = 8
    vision_coco_weights: Path = PROJECT_ROOT / "artifacts" / "vision" / "yolo26s.pt"
    vision_custom_weights: Path | None = None  # model YOLO tự train (artifacts/vision/<run>.pt)
    vision_min_conf: float = 0.4  # YOLO COCO
    vision_custom_min_conf: float = 0.6  # YOLO tự train (precision 0.89 trên tập test ở ngưỡng này)
    vision_ocr_min_conf: float = 0.4
    # VLM đa phương thức (OpenAI-compatible, ví dụ Ollama/vLLM chạy Qwen3-VL). Tắt mặc định vì ảnh có thể
    # chứa dữ liệu riêng tư; chỉ bật khi endpoint và chính sách lưu/chuyển dữ liệu đã được người vận hành duyệt.
    vision_vlm_enabled: bool = False
    vision_vlm_base_url: str = "http://127.0.0.1:11434/v1"
    vision_vlm_api_key: SecretStr = SecretStr("")
    vision_vlm_model: str = "qwen3-vl:8b"
    vision_vlm_timeout_seconds: float = 90.0
    vision_vlm_max_output_tokens: int = 700
    # Máy GPU nhỏ chạy đồng thời text LLM + VLM: yêu cầu Ollama dỡ VLM sau mỗi ảnh để tránh OOM ở lượt trả lời chữ.
    # Chỉ bật khi VISION_VLM_BASE_URL là Ollama do chính deployment này quản lý.
    vision_vlm_ollama_unload_after_request: bool = False
    # host được phép tải ảnh Messenger (hậu tố tên miền); chống SSRF
    vision_allowed_image_hosts: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["fbcdn.net", "fbsbx.com"]
    )

    # --- Tra cứu web có nguồn (Brave LLM Context API) ---
    web_search_enabled: bool = True
    web_search_mode: WebSearchMode = WebSearchMode.auto
    brave_search_api_key: SecretStr = SecretStr("")
    web_search_timeout_seconds: float = 30.0
    # Đủ rộng để Văn/NLXH đối chiếu nhiều góc nhìn, nhưng vẫn hữu hạn để không làm tràn context/độ trễ.
    web_search_max_results: int = 8
    web_search_max_context_tokens: int = 4096
    web_search_country: str = "vn"
    web_search_language: str = "vi"
    # Metasearch không cần API key (Google/Bing/Brave/DuckDuckGo... tùy backend khả dụng), sau đó mới rơi về Wiki.
    web_search_ddgs_fallback: bool = True
    # Không cần API key: Wikipedia là fallback cho kiến thức phổ thông có URL; không thay thế nguồn tin mới/chính thức.
    web_search_wikipedia_fallback: bool = True

    # --- Worker ---
    worker_concurrency: int = 4
    worker_poll_interval_seconds: float = 0.2
    worker_lease_seconds: int = 90
    worker_max_attempts: int = 4
    checkpoint_retention_hours: float = 72.0
    data_retention_days: int = 0  # 0 = không tự xóa
    housekeeping_interval_seconds: int = 600

    # --- LLM provider ---
    llm_provider: LLMProvider = LLMProvider.openai_compatible
    llm_base_url: str = "http://127.0.0.1:8100/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "qwen3-4b-instruct-2507"
    llm_timeout_seconds: float = 180.0
    llm_max_retries: int = 1
    llm_temperature: float = 0.3  # thấp để giảm bịa; model card Qwen gợi ý 0.7 cho hội thoại tự do
    llm_top_p: float = 0.8
    llm_max_output_tokens: int = 900
    # Bài Ngữ văn dài trên web; Messenger vẫn bị chia/cắt theo giới hạn riêng ở lớp giao tin.
    writing_max_output_tokens: int = 4096
    # Phần giải thích Tiếng Anh dùng cho chế độ đọc/podcast cần đủ chiều sâu như Ngữ văn.
    english_max_output_tokens: int = 1400
    # TTS neural chỉ nhận văn bản câu trả lời; không lưu âm thanh hay nội dung sau request.
    tts_enabled: bool = True
    tts_timeout_seconds: float = 120.0
    tts_max_chars: int = 24_000
    tts_max_audio_mb: int = 36
    # Kiểm định lần hai chỉ cho Toán ngoài miền xác định và câu kiến thức có nguồn; không áp dụng sáng tác/Văn/Anh.
    answer_verification_enabled: bool = True
    answer_verification_max_tokens: int = 4096
    answer_verification_timeout_seconds: float = 180.0
    anthropic_api_key: SecretStr = SecretStr("")
    # model dự phòng khi model server không phản hồi (vd. Ollama Q4: http://127.0.0.1:11434/v1). Trống = tắt.
    llm_fallback_base_url: str = ""
    llm_fallback_model: str = ""

    # --- Memory ---
    memory_max_context_tokens: int = 3000
    memory_summary_trigger_tokens: int = 2200
    memory_keep_recent_messages: int = 8
    memory_chars_per_token: float = 2.5  # ước lượng thận trọng cho tiếng Việt

    # --- Quản trị / riêng tư ---
    admin_api_key: SecretStr = SecretStr("")
    pseudonym_secret: SecretStr = SecretStr("")

    # --- Langfuse (tùy chọn) ---
    langfuse_enabled: bool = False
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    langfuse_base_url: str = "https://cloud.langfuse.com"
    # Mặc định chỉ gửi metadata; bật để gửi nội dung ĐÃ redaction.
    langfuse_capture_content: bool = False
    langfuse_sample_rate: float = 1.0

    # --- Model server local (serving/) ---
    model_server_host: str = "127.0.0.1"
    model_server_port: int = 8100
    model_server_api_key: SecretStr = SecretStr("")
    base_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    base_model_revision: str = "cdbee75f17c01a7cc42f958dc650907174af0554"
    served_model_name: str = "qwen3-4b-instruct-2507"
    adapter_registry_path: Path = PROJECT_ROOT / "artifacts" / "registry.json"

    # --- Kaggle (chỉ đường dẫn, không chứa credential) ---
    kaggle_json_path: Path = Path.home() / ".kaggle" / "kaggle.json"

    @field_validator(
        "messenger_allowed_psids", "web_allowed_origins", "vision_allowed_image_hosts", mode="before"
    )
    @classmethod
    def _split_csv(cls, v: object) -> object:
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @model_validator(mode="after")
    def _defaults_and_production_guard(self) -> Settings:
        if self.messenger_send_mode is None:
            self.messenger_send_mode = (
                SendMode.all if self.app_env == AppEnv.production else SendMode.allowlist
            )
        if self.app_env == AppEnv.production:
            missing = [
                name
                for name in (
                    "meta_app_secret",
                    "meta_page_access_token",
                    "meta_verify_token",
                    "admin_api_key",
                    "pseudonym_secret",
                )
                if not getattr(self, name).get_secret_value()
            ]
            if not self.meta_page_id:
                missing.append("meta_page_id")
            if missing:
                raise ValueError(f"APP_ENV=production nhưng thiếu cấu hình bắt buộc: {', '.join(missing)}")
            if len(self.admin_api_key.get_secret_value()) < 24:
                raise ValueError("ADMIN_API_KEY quá ngắn (cần >= 24 ký tự)")
        return self

    # --- tiện ích ---
    @property
    def sqlalchemy_url(self) -> str:
        return self.database_url.get_secret_value()

    @property
    def psycopg_conninfo(self) -> str:
        """URL dạng libpq cho psycopg/LangGraph (bỏ hậu tố driver của SQLAlchemy)."""
        url = self.database_url.get_secret_value()
        for prefix in ("postgresql+psycopg://", "postgresql+asyncpg://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql://" + url[len(prefix) :]
        return url

    @property
    def send_api_url(self) -> str:
        return (
            f"{self.meta_graph_base_url}/{self.meta_graph_api_version}/{self.meta_page_id or 'me'}/messages"
        )

    def secret_values(self) -> list[str]:
        """Mọi giá trị bí mật đang cấu hình (để lớp log/trace che đi)."""
        values: list[str] = []
        for name, field in type(self).model_fields.items():
            val = getattr(self, name)
            if isinstance(val, SecretStr):
                raw = val.get_secret_value()
                if raw and len(raw) >= 8:
                    values.append(raw)
                    # che cả phần mật khẩu trong DATABASE_URL
                    if name == "database_url" and "@" in raw and ":" in raw.split("@", 1)[0]:
                        pw = raw.split("@", 1)[0].rsplit(":", 1)[-1]
                        if len(pw) >= 6:
                            values.append(pw)
            del field
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
