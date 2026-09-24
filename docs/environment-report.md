# Báo cáo môi trường (khảo sát ngày 23/09/2026)

Đây là báo cáo lịch sử; ID tài khoản và đường dẫn cá nhân đã được lược bỏ trước khi xuất bản.
Không dùng báo cáo này để suy ra tình trạng xác thực hiện tại.

Mọi mục dưới đây đều đã được **kiểm chứng bằng lệnh thật** trên máy. Mục chưa kiểm chứng được ghi rõ.

## 1. Phần cứng và hệ điều hành

| Hạng mục | Giá trị | Cách kiểm chứng |
|---|---|---|
| OS | Ubuntu 24.04.5 LTS, kernel 7.0.0-34 | `/etc/os-release`, `uname -a` |
| CPU | 16 luồng | `nproc` |
| RAM | 31 GiB (+ swap 31 GiB) | `free -h` |
| Đĩa trống | 765 GB trên `/` | `df -h` |
| GPU | **NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM, compute capability 12.0 (Blackwell, sm_120)** | `nvidia-smi --query-gpu=...` |
| NVIDIA driver | 580.178.04 (hỗ trợ CUDA 13.0) | `nvidia-smi` |
| CUDA toolkit | 13.0 (V13.0.88) tại `/usr/local/cuda-13.0` | `nvcc --version` |
| CUDA runtime của PyTorch | 13.0 (PyTorch 2.14.0 từ PyPI kéo `cuda-toolkit==13.0.3`) | `torch.version.cuda` |
| sm_120 trong PyTorch | Có (`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`) | `torch.cuda.get_arch_list()` |
| bf16 trên GPU | Hoạt động (nhân ma trận 2048x2048 bf16 thành công) | script kiểm tra |

**Phân biệt 3 thành phần CUDA:** driver 580 (khả năng tối đa CUDA 13.0) ≠ CUDA toolkit hệ thống 13.0 (chỉ cần cho biên dịch, dự án không dùng) ≠ CUDA runtime đi kèm wheel PyTorch (13.0.3, tự cài qua pip). Không cài lại driver hay toolkit.

**Giới hạn phần cứng:** 16 GB VRAM đủ cho model ~4B ở bf16 để suy luận (~8,5 GB) **hoặc** huấn luyện LoRA/VeRA (có gradient checkpointing), nhưng không nên chạy model server và huấn luyện cùng lúc. Model 7–9B cần lượng tử hóa 4-bit.

## 2. Phần mềm

| Công cụ | Phiên bản |
|---|---|
| Python | 3.12.3 (hệ thống) — dự án dùng `requires-python >=3.12,<3.13` |
| uv | 0.12.18 |
| Docker Engine / Compose | 29.8.1 / v5.5.1 (user thuộc nhóm `docker`) |
| NVIDIA Container Toolkit | **Chưa cài** (Docker runtimes: chỉ `runc`) → model server chạy trên host |
| git | có; chưa cấu hình `user.name` |

Phiên bản thư viện chính đã khóa trong `uv.lock` (kiểm tra PyPI ngày 23/09/2026): fastapi 0.141.1, uvicorn 0.53.0, SQLAlchemy 2.0.54, psycopg 3.3.6, alembic 1.20.0, langchain-core 1.6.4, langchain-openai 1.6.4, langgraph 1.2.12, langgraph-checkpoint-postgres 3.1.2, langfuse 4.15.4, torch 2.14.0, transformers 5.17.0, peft 0.21.0, datasets 5.0.1, accelerate 1.15.0, kaggle 2.2.4.

## 3. Cổng mạng đang dùng trên máy

Đã có dịch vụ ở 8081, 21118 (RustDesk), 5037 (adb), 631 (CUPS)… Dự án dùng: **8000** (API), **8100** (model server), **55432** (PostgreSQL, chỉ bind 127.0.0.1) — không xung đột.

## 4. Thông tin xác thực (chỉ kiểm tra cấu trúc, KHÔNG in giá trị)

### Meta
Token bạn gửi đã được lưu vào `.env` (quyền 600, gitignore). Kết quả `debug_token` qua Graph API v26.0:

| Thuộc tính | Giá trị |
|---|---|
| Loại token | **PAGE** (Page Access Token), hợp lệ, không hết hạn |
| Data access hết hạn | 22/12/2026 (cần đăng nhập lại app để gia hạn trước ngày này) |
| App | App riêng trên máy phát triển (ID đã lược bỏ) |
| Page ID | ID đã lược bỏ |
| Quyền | `pages_messaging`, `public_profile` |

Kết luận:
- Token **gửi được tin Messenger** thay Page (có `pages_messaging`).
- Token **không gọi được mô hình AI** — mô hình chạy riêng (model server local hoặc API khác).
- Token **không có `pages_manage_metadata`** → không đăng ký/đọc webhook subscription của Page qua API (làm trong App Dashboard).
- **Thiếu App Secret** → chưa xác minh được chữ ký webhook; `/ready` báo `meta_app_secret: missing`.
- ⚠️ Token này và Kaggle API token (KGAT_…) đã được dán vào khung chat → nên coi là đã lộ, cần thay khóa đã chia sẻ trước khi sử dụng tiếp.

### Kaggle
- `~/Downloads/kaggle.json` tồn tại, JSON hợp lệ, có `username` và `key` (khóa legacy 32 ký tự hex).
- Quyền tệp đang là **664** (người khác trên máy đọc được) → khuyến nghị `chmod 600 ~/Downloads/kaggle.json`.
- `~/.kaggle` không tồn tại. Xác thực thành công bằng khóa legacy qua `KAGGLE_CONFIG_DIR=~/Downloads` (không sao chép tệp).
- Kaggle CLI 2.2.4 tự tạo thư mục cấu hình khi import nếu thiếu `KAGGLE_CONFIG_DIR`; một thư mục rỗng `~/.config/kaggle` bị tạo trong lúc khảo sát đã được xóa.
- Token KGAT_ bạn dán **không cần dùng** và không được ghi vào đâu.

### Biến môi trường khác
Không có API key LLM nào (OpenAI/Anthropic/Gemini/…) trong môi trường → runtime mặc định là **model open-weight chạy local**.

## 5. Model khả dụng

| Ứng viên | Giấy phép | Tham số | Nhận xét |
|---|---|---|---|
| **Qwen/Qwen3-4B-Instruct-2507** @ `cdbee75f…0554` | Apache-2.0, không gated | 4,02 B | **Chọn.** Chỉ xử lý văn bản, không có chế độ thinking, ChatML sạch, vừa 16 GB ở bf16 cho cả suy luận lẫn LoRA/VeRA; đã tải (8,06 GB) |
| Qwen/Qwen3.5-4B | Apache-2.0 | 4,66 B | Đa phương thức, kiến trúc mới `qwen3_5` → rủi ro cao hơn với PEFT; ứng viên để thử sau |
| google/gemma-4-E4B-it | Apache-2.0 | 8,0 B (tổng) | Kiến trúc any-to-any phức tạp hơn |
| aisingapore/Apertus-SEA-LION-v4-8B-IT | MIT | 8,05 B | Có tiếng Việt chính thức nhưng 8B bf16 = 16 GB → cần 4-bit |

Đo tokenizer Qwen3 trên tiếng Việt: ~3,36 ký tự/token (có dấu), ~2,42 (không dấu) → ngân sách bộ nhớ dùng ước lượng thận trọng 2,5.

## 6. Tài liệu đã đối chiếu

- Graph API changelog: phiên bản hiện hành **v26.0** (29/07/2026); v25.0 hết hạn 29/07/2028.
- Messenger Webhooks: xác minh GET, chữ ký `X-Hub-Signature-256`, **yêu cầu trả 200 trong ≤ 5 giây**, Meta gửi lại khi lỗi, sau 1 giờ lỗi liên tục có thể hủy đăng ký webhook.
- Messenger Policy: cửa sổ 24 giờ; bot tự động phải **công bố là dịch vụ tự động** ở đầu hội thoại, sau thời gian dài vắng mặt, và khi chuyển từ người sang bot.
- Send API / message_echoes: `metadata` được trả lại trong echo; `app_id` của echo từ hộp thư Page là app Page Inbox.
- **Chưa đọc trực tiếp được** trang tham chiếu Send API (trang render bằng JavaScript; WebFetch/curl chỉ nhận khung trang). Giới hạn 2000 ký tự/tin lấy từ nguồn thứ cấp → dự án chia tin ở ngưỡng 1900 (UTF-16).
- LangGraph 1.2.12, Langfuse 4.15.4, PEFT 0.21.0, Kaggle 2.2.4: API được **kiểm tra trực tiếp trên bản đã cài** (`inspect.signature`, đọc mã nguồn).
