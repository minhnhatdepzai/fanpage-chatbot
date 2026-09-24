# Tiến độ

## Kiểm tra trước khi xuất bản GitHub — 24/09/2026

- Hoàn thiện README cho cài máy mới, cấu hình, khởi động, Messenger/web, model, RAG, train, Docker và sao lưu.
- Kiểm tra lại: `uv run --no-sync pytest -q` → **246 passed** (có DB); Ruff sạch; widget **3 passed**, typecheck/build thành công.
- Dịch vụ local khi kiểm tra: model, Ollama, API, worker, tunnel đang chạy; readiness true, heartbeat ok.
- Rà soát danh sách commit: không kèm .env, khóa thực trong .env/Kaggle, node_modules, artifact hay hội thoại.
- Clone mới dùng base model; adapter và YOLO custom phải train lại hoặc khôi phục riêng.
- Docker profile app và cài mới toàn bộ trên máy sạch chưa được kiểm thử end-to-end trong lần xuất bản này.

## Báo cáo trước đó (24/09/2026, 07:30 giờ VN)

Trạng thái chỉ ghi "đạt" khi có bằng chứng chạy thật. Mock ≠ tích hợp thật.

## Đã đạt

| Hạng mục | Bằng chứng |
|---|---|
| Hạ tầng: PostgreSQL, migration, LangGraph checkpointer, `/health`, `/ready` | `/ready` = 200, mọi check `ok` |
| Model local Qwen3-4B-Instruct-2507 (bf16, ~8 GB VRAM) qua server tương thích OpenAI | `/health` cổng 8100; sinh câu trả lời ~2–5 s |
| Webhook: GET verify, chữ ký sai/thiếu bị từ chối, trùng lặp không tạo lượt mới | test tự động + thử qua tunnel công khai (verify 200 / token sai 403 / POST không ký 401) |
| Luồng đầy đủ webhook → DB → worker → LangGraph → model → kiểm tra đầu ra → cổng gửi | webhook ký thật với PSID giả ngoài allowlist: sinh câu trả lời có nguồn, trạng thái `dry_run` (không gửi), gửi lại y hệt không tạo tin mới; dữ liệu thử đã xóa |
| Gửi Messenger thật | phiên trước: `meta send-test` tới PSID allowlist → `sent` |
| Độ chính xác: kho kiến thức có nguồn + truy xuất + gắn nguồn + chặn link/nguồn bịa + ghi chú chưa kiểm chứng + chế độ `strict` | 40 test trong `tests/test_grounding.py`; chạy thử với model thật (xem README) |
| Bộ nhớ nhiều lượt, cô lập người dùng, handoff, fallback có giới hạn, resume không gọi model lại | test tự động (`tests/test_graph.py`, `tests/test_db_flows.py`) |
| Pipeline Kaggle | đã tải OASST1 (bản Kaggle), lọc tiếng Việt → `data/processed/oasst1-vi-20260923-87ed8c64` (37 hội thoại, 26/6/5, có manifest) |

Kiểm thử: `uv run pytest -q` → 196 passed; widget `npm test` 3/3; `uv run ruff check .` sạch.
(`ruff format --check` còn báo 15 tệp chưa format — chưa chạy format hàng loạt để tránh diff lớn.)

## Chưa kiểm chứng / đang chặn

- **Lượt Messenger thật end-to-end (người dùng nhắn → bot trả lời)**: hệ thống đang chạy, nhưng webhook của app vẫn
  trỏ tới URL tunnel cũ đã chết. Cần chủ app chạy `uv run botctl meta set-webhook <URL>/webhook` (URL in ra bởi
  `./scripts/run_local.sh start`) hoặc sửa Callback URL trong App Dashboard. Chưa biết Page đã "Add subscriptions"
  với app hay chưa (token không có `pages_manage_metadata` nên không đọc được).
- Tunnel là Cloudflare quick tunnel: URL đổi mỗi lần khởi động lại → phải đặt lại webhook. Muốn URL cố định: named
  tunnel (cần tài khoản Cloudflare + domain) hoặc bật Tailscale Funnel cho tailnet.

## Fine-tune giáo dục / giải trí (24/09/2026)

- Kho kiến thức: thêm `config/knowledge/tin-tuc-2026.yaml` (12 mục: thi THPT 2026, phổ điểm, miễn học phí, Luật
  Nhà giáo, SGK thống nhất, 34 tỉnh thành, kỹ thuật học hiệu quả, phòng vé phim Việt, model AI tháng 9/2026), mỗi mục
  đối chiếu trang nguồn ngày 24/09/2026. Tổng 19 mục. Tin tức cũ đi nhanh -> cần cập nhật định kỳ.
- Không dùng OASST-vi để huấn luyện: câu trả lời không có nguồn, có chỗ sai sự thật (dạy model khẳng định bừa).
- Dữ liệu: `training/datasets/edu_ent_sft_v1.yaml` - 72 hội thoại tổng hợp (Claude soạn), 56 train / 16 val, dựng
  với đúng system prompt production + nguồn thật. Loss chỉ trên token assistant (đã kiểm tra che mặt nạ).
- Huấn luyện (RTX 5060 Ti 16 GB, bf16, gradient checkpointing, 21 bước):
  | | tham số train | val loss (3.059 ->) | thời gian | VRAM đỉnh |
  |---|---|---|---|---|
  | LoRA r=16 | 33,0 M (0,81%) | 1.425 | 321 s | 12,3 GB |
  | VeRA r=256 | 1,17 M (0,03%) | 2.013 | 385 s | 11,8 GB |
- Đánh giá `evaluation/datasets/edu_ent_eval_v1.yaml` (50 kịch bản tổng hợp, không trùng dữ liệu train), qua đúng
  pipeline production, seed cố định, chấm bằng luật (`evaluation/scripts/compare.py`):
  | | đạt | tín hiệu bịa | trích nguồn (sourced) | độ dài TB | p50 / p95 |
  |---|---|---|---|---|---|
  | baseline | 43/50 | 7 | 100% | 214 ký tự | 1,76 / 3,24 s |
  | LoRA | 46/50 | 3 | 100% | 180 | 1,97 / 3,08 s |
  | VeRA | 46/50 | 3 | 100% | 161 | 1,98 / 2,96 s |
  Lần chạy đầu phát hiện lỗi ở chính guardrail/harness (lịch sử chứa footer nguồn, bộ lọc "nguồn chính thức" bắt
  nhầm, nhận diện "không chắc" thiếu) -> đã sửa cho cả production rồi chạy lại cả 3 model.
- Chọn **VeRA** (`vera-edu-ent-v1-20260923-175339`, đang active): LoRA mâu thuẫn nguồn ở e09 (nói "chưa có thông tin"
  dù nguồn ghi Thanh Hóa không sáp nhập). Rollback: `uv run botctl model rollback --reason "..."` rồi restart model.
- Giới hạn còn lại: model vẫn có thể thêm chi tiết nhỏ không có trong nguồn (vd. nói phim "đang chiếu"); chênh lệch
  3/50 kịch bản là nhỏ; cùng một người soạn dữ liệu train và bộ đánh giá -> có thể thiên lệch chung; cần người duyệt.


## Đợt 24/09/2026: RAG tài liệu, widget website, Ollama 4-bit, ảnh (OCR + YOLO)

**RAG PDF/Word + pgvector** (đạt, có bằng chứng)
- Postgres chuyển sang `pgvector/pgvector:0.8.6-pg17-bookworm` bằng dump/restore sang volume mới (collation musl->glibc
  khác nhau); số dòng khớp; volume cũ `pgdata` giữ để rollback; bản dump ở `runs/backup/`.
- Embedding `AITeamVN/Vietnamese_Embedding` (fine-tune bge-m3, Apache-2.0, 1024 chiều) chạy trong model server
  (`/v1/embeddings`). bge-m3 gốc không nạp vì repo chỉ có trọng số pickle `.bin`.
- `botctl docs ingest|list|delete|search`, `botctl ask`, `POST /admin/ask`. Tài liệu `internal` chỉ trợ lý nội bộ
  thấy (lọc trong SQL, có test). Tài liệu public có dữ liệu cá nhân bị từ chối nạp.
- Đã nạp: Quy chế thi tốt nghiệp THPT (TT 24/2024, PDF 82 trang, 200 đoạn; 6 trang scan chưa OCR - nạp lại với
  `--replace --ocr` khi cần) + tài liệu mẫu nội bộ (Word). Ngưỡng tương đồng hiệu chỉnh 0,40 (liên quan 0,32-0,59,
  không liên quan <= 0,31). Đối chiếu câu trả lời với văn bản gốc: Điều 37 (phúc khảo), Điều 45 (công nhận TN) khớp.

**Widget website (TypeScript)** - `web/widget` (TS 7.0.2 + esbuild), phục vụ ở `/web/widget.js`, demo `/web/demo`.
Kênh "web" đi chung hàng đợi/worker/graph; phiên ký HMAC; giới hạn tần suất; CORS theo `WEB_ALLOWED_ORIGINS`.
Đã chạy thật trong Chromium headless: gửi/nhận, link nguồn an toàn, lịch sử sau khi tải lại, không lỗi console.

**Ollama + Qwen3-4B Q4_K_M** (v0.34.3, cài ở `~/.local/ollama`, kiểm SHA-256). So sánh 50 kịch bản:
| | đạt | tín hiệu bịa | p50 / p95 | VRAM |
|---|---|---|---|---|
| bf16 gốc | 41/50 | 9 | 1,9 / 3,5 s | ~8 GB |
| bf16 + VeRA (đang chạy) | 46/50 | 2 | 2,1 / 3,1 s | ~8 GB |
| Ollama Q4_K_M | 40/50 | 11 | 0,55 / 1,34 s | 3,9 GB (kèm KV 8k) |
Ollama không chạy được VeRA -> dùng làm **dự phòng tự động** (`LLM_FALLBACK_*`) khi model server lỗi (đã thử: 206 ms).

**Ảnh: OCR + YOLO - CHỈ CHẠY CỤC BỘ (chưa gắn vào Messenger/website theo yêu cầu "làm local trước")**
- `botctl vision analyze <ảnh>`, `botctl vision ask <ảnh> "câu hỏi"`.
- OCR EasyOCR (vi+en): ảnh chữ tổng hợp khớp 0,989; trang scan thật đọc tốt, sai vài dấu.
- YOLO26s (COCO, tên lớp dịch tiếng Việt) + YOLO26n tự train trên Kaggle:
  `abdullahsami10/stationary-dataset` (CC BY 4.0; tên lớp không có sẵn - xác định bằng xem ảnh: bút, tẩy, gọt bút chì,
  thước kẻ) + `josephnelson/six-sided-dice-images-and-bounding-boxes` (CC0; mặt xúc xắc 1-6). Chia tập theo nhóm ảnh
  gần trùng; phát hiện 17 ảnh trùng giữa các tập của tác giả và 23 tệp nhãn trộn đa giác (Ultralytics bỏ qua cả ảnh).
- Tập test đã sửa (94 ảnh, 190 vật thể): v1 mAP50 0,856 / mAP50-95 0,696; v2 (nhãn đa giác -> hộp) 0,835 / 0,663.
  Chọn v1 theo tập validation (0,811 vs 0,798 - gần như ngang). Ngưỡng 0,6: precision 0,88, recall 0,67.
- Chốt chặn chống bịa: ảnh không có chữ và không có vật thể đủ chắc -> trả lời mẫu, KHÔNG gọi model (trước đó model
  bịa "tổng xúc xắc là 21"). Chữ OCR bị che dữ liệu cá nhân; chữ trong ảnh coi là dữ liệu (chống chèn lệnh).
- Giấy phép: Ultralytics AGPL-3.0 (dịch vụ thương mại: công bố mã nguồn hoặc mua Enterprise).

**Chưa làm tiếp (chờ bạn quyết)**: gắn ảnh vào Messenger (tải ảnh từ CDN Meta, giới hạn host) và widget
(`/web/images`); OCR 6 trang scan của quy chế; dữ liệu YOLO còn nhỏ (~500 ảnh, 2 chủ đề).


## Đợt 24/09/2026 (sáng): YOLO v3 + ảnh trên Messenger/website

**Dữ liệu v2 (6 nguồn Kaggle, 91 lớp, 8.323 ảnh)**: stationery (CC BY 4.0), objects-in-the-classroom (MIT), dice (CC0),
guitar (Kaggle CC0 / Roboflow CC BY 4.0), chess (CC0), playing cards (CC0, lấy mẫu 3.100/20.000). Phát hiện ở nguồn:
588 ảnh classroom và 20 ảnh chess/stationery trùng giữa train/val/test của tác giả -> gom nhóm (tiền tố Roboflow +
dHash) và dồn về một tập. `labels.cache` (pickle) trong bộ guitar bị bỏ qua khi giải nén. Tập test của guitar không có
nhãn -> chỉ dùng train+valid.

**Huấn luyện**: v2 (60 epoch) chỉ đạt mAP50 0,797 vì `optimizer=auto` của Ultralytics chọn AdamW lr 0,000105 khi số bước
< 10.000; xúc xắc sụp (mAP50-95 0,15). v3: tăng mẫu train (xúc xắc x3, guitar x5, cờ vua x2) + 100 epoch -> MuSGD lr 0,01.

| Test (1.149 ảnh) | mAP50 | mAP50-95 |
|---|---|---|
| v2 | 0,797 | 0,666 |
| **v3 (đang dùng)** | **0,930** | **0,803** |

So với bản đang chạy trước (v1, 10 lớp) trên ảnh CẢ HAI chưa từng train: đồ dùng học tập (27 ảnh) ngang nhau (mAP50
0,858 vs 0,856; P/R ở 0,6: 0,861/0,743 vs 0,855/0,743); xúc xắc (5 ảnh - mẫu nhỏ) v3 tốt hơn (mAP50-95 0,550 -> 0,725;
P@0,6 0,75 -> 1,00). -> triển khai v3, giữ v1/v2 để rollback.

**Tích hợp**: webhook bắt URL ảnh -> worker tải (chỉ host Meta, ≤ 8 MB, image/*) -> model server `/v1/vision/analyze`
-> khối ngữ cảnh ảnh (nói rõ giới hạn) -> LLM -> kiểm tra đầu ra -> Send API. Widget: `/web/images` phân tích ngay,
lưu kết quả đã che dữ liệu cá nhân. Kết quả được lưu vào tin nhắn (URL CDN bị xoá) -> câu hỏi tiếp có ngữ cảnh ảnh.

**Kiểm chứng**
- Tự động: 196 test Python (gồm luồng Messenger có ảnh qua DB với ảnh/phân tích giả lập) + 3 test widget.
- VRAM khi tải đồng thời (3 ảnh 4000x5300 + 32 embedding + 2 lượt LLM): đỉnh 13,15 / 16,3 GB, tất cả HTTP 200.
- Website THẬT (Chromium + toàn bộ dịch vụ đang chạy): bài tập -> "125 + 378 = 503"; hỏi tiếp "câu 2" -> "28 cm" (dùng
  ngữ cảnh ảnh lượt trước); ảnh lá bài -> "3 cơ, 5 cơ x2, 10 rô" (khớp nhãn thật); ảnh xe buýt hỏi biển số -> nói không
  nhận diện được; tệp hỏng -> thông báo tiếng Việt; API: ảnh hỏng 400, ảnh > 8 MB 413.
- Messenger THẬT: xem mục cuối (chờ tin nhắn có ảnh từ tài khoản thử).

## Đợt 24/09/2026 (trưa): rà tin nhắn thật trên Messenger và sửa lỗi

Lỗi thấy trong hội thoại thật (tài khoản thử) và cách sửa:

| Lỗi thật | Nguyên nhân | Sửa |
|---|---|---|
| Đã đọc được ảnh nhưng lượt sau nói "Mình không thể đọc hình ảnh" | ảnh và câu hỏi bị tách thành 2 lượt; prompt không nói bot có OCR/YOLO | tin chỉ có ảnh chờ thêm 6 giây để gộp với câu hỏi đi kèm (`DEBOUNCE_ATTACHMENT_SECONDS`); prompt nêu rõ khả năng đọc ảnh |
| Ảnh thứ 2 (người, ghế, laptop) bị trả lời bằng nội dung ảnh 1 (RTX 5080) | kết quả ảnh mới chỉ nằm trong system prompt, ghi chú ảnh cũ nằm ngay trong lịch sử -> model 4B bám ảnh cũ | dữ liệu ảnh MỚI được gắn thẳng vào tin nhắn hiện tại (`current_image_note`) |
| "RTX 5080 chưa ra mắt", "RTX 5000 dự kiến cho tương lai", "có thể là giả" | kiến thức cũ của model | `output_check` bỏ câu "chưa ra mắt/dự kiến/không có thông tin chính thức..." khi nguồn không nói vậy (giữ câu bot tự nhận "mình không có thông tin"); thêm mục kiến thức có nguồn: RTX 50 Series (nvidianews.nvidia.com), GPT-5 (Wikipedia) |
| Ghi chú "Lưu ý" lặp 2 lần | model chép ghi chú của lượt trước | bỏ ghi chú khỏi lịch sử trước khi gửi model; bỏ "Lưu ý" do model tự viết, chỉ giữ 1 ghi chú của hệ thống |
| Ghi chú "chưa có nguồn trong tài liệu của trang" cho câu trả lời về ảnh | ghi chú chung | câu trả lời về ảnh dùng ghi chú riêng: dựa trên nhận diện tự động, có thể sót/nhầm |
| Sticker bị trả lời "chưa xem được hình ảnh" | sticker được Meta gửi dưới dạng `image` | sticker được phân loại là `sticker` |
| "50 nhân 50" kèm ghi chú "chưa có nguồn" | model tự tính | phép tính số học tính bằng code (`app/conversation/mathcalc.py`), không gọi model |
| "ra mắt cách đây hơn 1 năm 8 tháng" (đúng là 1 năm 7 tháng) | model tự tính khoảng thời gian | bỏ câu tự tính khoảng thời gian không có nguồn |
| "(ví dụ )" còn sót sau khi bỏ link giả | dọn dấu vết chỉ chạy khi đổi số nguồn | dọn cả khi chỉ bỏ link |

**Đã kiểm chứng (sau khi khởi động lại API + worker)**
- 246 test Python, ruff sạch.
- Chạy lại qua graph thật (Qwen3-4B + VeRA, model server, kho kiến thức, pgvector) với phân tích ảnh THẬT lấy từ DB
  (ảnh RTX 5080 rồi ảnh người/ghế): ảnh 2 được mô tả đúng; hỏi tiếp "có mấy người" -> 1 người (0,92); hỏi lại "card lúc
  nãy" -> RTX 5080, trích nguồn NVIDIA, không còn "chưa ra mắt".
- Website THẬT qua URL Cloudflare: ảnh "GEFORCE RTX 5080" -> đúng + nguồn; ảnh bài tập thứ 2 -> 19 và 100; hỏi tiếp
  "câu 2" -> 100; hỏi "card lúc nãy ra mắt khi nào" -> 30/1/2025; "50 nhân 50" -> 2.500 (tính bằng code).

**Còn chờ kiểm chứng**: Messenger thật với code mới (gửi ảnh kèm câu hỏi, ảnh thứ hai khác, câu hỏi tiếp). Model 4B
vẫn có thể thêm chi tiết không có trong ảnh/nguồn (vd. đoán ý nghĩa mã "OC 1688"); prompt đã cấm, nhưng không chặn được
100%.

## Chưa làm (theo lộ trình ban đầu)
- Giai đoạn 8: lịch chạy vòng cải thiện (các lệnh review/candidates/data export đã có, chạy thủ công).
- Langfuse: code tích hợp có sẵn nhưng đang tắt (`LANGFUSE_ENABLED=false`), chưa thử với server Langfuse thật.
- Chưa có commit git nào.

## Việc an toàn cần làm

- Page Access Token, Kaggle API token và App Secret đã từng xuất hiện trong chat/ảnh chụp → nên tạo lại (Meta: App
  settings → Basic → App secret → Reset; tạo lại Page token; Kaggle: Settings → API → Expire token) rồi cập nhật `.env`.
- `chmod 600 ~/Downloads/kaggle.json`.

## Thay đổi phiên này

- Mới: `app/conversation/knowledge.py`, `config/knowledge/ai-co-ban.yaml`, `tests/test_grounding.py`,
  `tests/fixtures/knowledge/`, `scripts/run_local.sh`, lệnh `botctl knowledge list|search`, `botctl meta set-webhook`.
- Sửa: prompt `ai-tutor-vi-v3-grounded` (quy tắc chính xác + khối nguồn), `output_check` (nguồn, link, câu tự nhận
  kiểm chứng, ghi chú, bỏ markdown), `intents.is_knowledge_question`, graph (truy xuất, chế độ strict), settings
  (`KNOWLEDGE_DIR`, `KNOWLEDGE_TOP_K`, `GROUNDING_MODE`, `LLM_TEMPERATURE` 0.7 → 0.3).
- `serving/runtime.py`: sửa lỗi Transformers 5 trả `BatchEncoding` từ `apply_chat_template` (do phiên Codex trước).
- `.env`: `MESSENGER_SEND_MODE` all → allowlist (chỉ trả lời PSID kiểm thử).
- Cài `cloudflared` 2026.9.1 vào `~/.local/bin` (đã đối chiếu SHA-256 với release chính thức).
