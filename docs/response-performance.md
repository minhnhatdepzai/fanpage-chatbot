# Tốc độ và độ chi tiết của câu trả lời

Đã áp dụng và đo trên máy local ngày 24/09/2026:

- Thời gian gom tin nhắn: 2 giây → 0,5 giây, tối đa 8 → 2 giây.
- Chu kỳ lấy công việc của worker: 0,5 → 0,2 giây.
- Trần đầu ra LLM: 400 → 900 token (không bắt buộc dùng hết).
- Prompt: câu hỏi kiến thức/hướng dẫn mặc định khoảng 120–220 từ, có giải thích,
  ví dụ/bước thực hành phù hợp; chào hỏi và xác nhận vẫn ngắn.
- Giữ model Qwen + VeRA, kiểm tra an toàn, nguồn trích dẫn, timeout và cơ chế fallback.
  Không huấn luyện lại model trong lần điều chỉnh này.

## Đo thực tế

`scripts/benchmark_reply.py` tạo hội thoại web riêng và chạy qua API → queue →
worker → model → kiểm tra đầu ra → trả lời. Không gửi tin thử vào Messenger.
Đây là mẫu nhỏ trên máy local, không phải benchmark tải đồng thời hay cam kết độ trễ.
Thời gian chưa bao gồm đường truyền Facebook và thiết bị người dùng.

| Câu hỏi | Trước: giây / ký tự | Sau: giây / ký tự |
| --- | --- | --- |
| Chào hỏi | 5,53 / 162 | 3,88 / 194 |
| RAG khác fine-tuning | 5,52 / 348 | 13,69 / 1.320 |
| Cách ôn bài | 5,53 / 224 | 12,25 / 1.154 |

Kết quả gốc nằm trong `runs/benchmarks/reply-before-v6.json` và
`runs/benchmarks/reply-after-v6-grounded.json`. Bản sau ghi nhận prompt
`edu-ent-vi-v7-detailed-capabilities` vì workspace còn có chỉnh sửa prompt song song;
giữ nguyên các chỉnh sửa đó. Không coi chênh lệch là tác động riêng của một thay đổi.
Tính ngẫu nhiên của model và tải GPU cũng có thể làm số đo dao động.

Giảm thời gian chờ giúp bắt đầu xử lý sớm hơn, nhưng câu trả lời dài vẫn mất nhiều
thời gian sinh hơn. Chưa triển khai streaming: nội dung vẫn được kiểm tra trước khi gửi.
Langfuse xác nhận generation dùng trần 900 token; chỉ gửi metadata, không bật thu thập
nội dung hội thoại. Các cờ `long_reply` vẫn được giữ để theo dõi, không chặn câu trả lời.

Lượt thử phát hiện bot nhầm fine-tuning là huấn luyện từ đầu. Đã thêm tài liệu
[Hugging Face](https://huggingface.co/docs/transformers/training) vào kho kiến thức;
lượt thử sau diễn giải đúng rằng fine-tuning tiếp tục từ mô hình pretrained.
Đây không phải bảo đảm mọi câu trả lời đều chính xác; vẫn cần đánh giá với nhiều câu hỏi.

Chạy lại (dùng nhãn mới để không ghi đè kết quả):

```bash
.venv/bin/python scripts/benchmark_reply.py --label ten-lan-do-moi
```

Tinh chỉnh qua `.env` nếu cần: `DEBOUNCE_SECONDS`, `DEBOUNCE_MAX_SECONDS`,
`WORKER_POLL_INTERVAL_SECONDS`, `LLM_MAX_OUTPUT_TOKENS`. Khởi động lại API/worker
sau khi đổi. Không cần đổi URL webhook hoặc nạp lại model chỉ vì thay đổi các mục này.
