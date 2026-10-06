# Tiến độ

## Cổng giảm sai số Toán — 05/10/2026

- Giữ hai mức tin cậy riêng: `verified` chỉ dành cho kết quả đi qua bộ giải xác định; `math_review` dành cho bài
  ngoài miền. Nhánh `math_review` bắt buộc sinh lời giải rồi gọi một lượt kiểm định độc lập ở nhiệt độ 0 để giải lại,
  kiểm tra điều kiện, nghiệm ngoại lai và phép thế ngược.
- Giao thức kiểm định fail-closed: chỉ nhận JSON có verdict `pass|corrected|insufficient`; lỗi model, JSON hỏng hoặc
  thiếu căn cứ đều loại bỏ bản nháp và trả lời rằng chưa thể kiểm chứng, không chốt đáp số có thể sai.
- Kiến thức thực tế mặc định dùng `GROUNDING_MODE=strict`: không có nguồn thì không cho model đoán. Câu có nguồn được
  đối chiếu lần hai theo từng khẳng định và citation. Cổng này không áp dụng cho sáng tác Ngữ văn hay bài Tiếng Anh.
- Langfuse tiếp tục nhận trace metadata `answer-verification` khi được cấu hình; môi trường hiện không có credential,
  nên chưa tạo annotation queue/dataset live. Đánh giá cục bộ dùng code-based checks cho giao thức và routing.

## Mở rộng lớp 10–12 — 05/10/2026

- Bộ giải xác định thêm đạo hàm, tích phân xác định, giới hạn, bất phương trình một ẩn, phương trình mũ đưa được về
  cùng cơ số, tổ hợp/chỉnh hợp và các hàm lượng giác/logarit an toàn. Kết quả đều mang phương pháp kiểm chứng; đầu
  vào vẫn qua AST whitelist, không dùng `eval`/`sympify` trực tiếp.
- Router nhận thêm khuôn bài THPT như `complete and explain`, `rewrite` và so sánh câu thơ tự viết; prompt production
  v13 yêu cầu nêu điều kiện, trình bày phép biến đổi, tự kiểm tra, không bịa passage/audio/trích dẫn còn thiếu.
- Thêm 22 hội thoại THPT tổng hợp tự soạn (16 train, 6 validation). Bộ gộp có 115 train / 36 validation, manifest
  `edu_ent_sft_v1+edu_ent_sft_v2_rich+writing_sft_v1+english_tutor_sft_v1+high_school_10_12_sft_v1-15b09c11`.
- Train thật `vera-edu-ent-v5-high-school-20261005-073853`: 40 bước, 615,7 giây, validation loss 3,3050 → 2,3427,
  peak VRAM 15,14 GB; nạp lại trên base sạch thành công, loss mẫu 1,2150 hữu hạn.
- Eval tách biệt 13 tình huống: base 9/13, v5 8/13, cả hai 0 tín hiệu bịa; 4/4 bài đi qua bộ giải xác định đạt ở
  cả hai. V5 **FAIL gate** (yêu cầu tốt hơn baseline ít nhất 2), đã register `candidate` và không promote. Production
  tiếp tục dùng VeRA v1; năng lực Toán xác định/router/prompt mới triển khai độc lập và đã qua test.
- Kiểm thử cuối: `pytest -q` **334 passed**, Ruff sạch; local và URL public đều trả đúng tích phân `1/3`, health xác
  nhận production đang dùng `vera-edu-ent-v1-20260923-175339`.

## Gia sư Tiếng Anh + giao diện đa môn — 05/10/2026

- StudyScope có điều hướng trực quan theo Toán, Ngữ văn, Tiếng Anh và Kiến thức. Đây là bộ lọc câu hỏi mẫu; router
  vẫn tự nhận biết từ nội dung, đúng với yêu cầu người học không phải chọn chế độ trước.
- Thêm `app/english_tutor/`: nhận diện ngữ pháp, từ vựng, Reading, Listening, Speaking, phát âm, dịch, Writing và
  IELTS; có trả lời song ngữ, giải phương án nhiễu, bám evidence trong passage, không giả nghe audio và không cấp
  band chính thức. Thiếu passage/transcript/ảnh đề thì trả lời xác định, không gọi model để đoán.
- Đối chiếu repo `task1-coach-pages` commit `7fcac40`: dùng các trục Vocabulary, Grammar, paraphrase, Writing Task 1,
  Reading, Listening và Speaking làm phạm vi; giữ nguyên tuyên bố provenance rằng bài tự soạn không phải đề IELTS/
  Cambridge chính thức. Không sao chép bundle build hoặc nội dung không có giấy phép rõ vào runtime.
- Dữ liệu Tiếng Anh v1 gồm 24 hội thoại tổng hợp tự soạn (18 train, 6 validation). Bộ gộp toàn năng lực có 99 train,
  30 validation, manifest `edu_ent_sft_v1+edu_ent_sft_v2_rich+writing_sft_v1+english_tutor_sft_v1-a56e3540`.
- Train thật `vera-edu-ent-v4-english-20261005-060531`: 32 bước, 435,6 giây, validation loss 3,2258 → 2,2915,
  peak VRAM 14,83 GB; nạp lại từ base sạch thành công, loss mẫu 1,3428 hữu hạn.
- Eval tách biệt 10 tình huống: base 3/10, v4 4/10, cả hai 0 tín hiệu bịa. Gate yêu cầu ứng viên tốt hơn ít nhất
  2 tình huống, nên v4 **FAIL gate** và chỉ được register ở trạng thái `candidate`; production giữ VeRA v1. Phần
  router/prompt/guard v12 vẫn được triển khai độc lập vì có unit test và không phụ thuộc adapter v4.
- Prompt v12 và output guard coi bài ngôn ngữ dựa trên câu/passage người dùng cung cấp là tác vụ học tập, không gắn
  cảnh báo “chưa có nguồn” sai ngữ cảnh; câu hỏi về quy định IELTS mới vẫn đi RAG/web và nguồn chính thức.

## Sáng tác nguyên bản đa truyền thống — 05/10/2026

- Prompt production v11 tách rõ “sáng tác mới” khỏi “phân tích tác phẩm có thật”: nhánh sáng tác được tự tạo câu thơ,
  tiêu đề, nhân vật, lời thoại và dữ kiện hư cấu; nhánh phân tích vẫn bắt buộc bám văn bản/nguồn.
- Thêm hồ sơ sở thích có cấu trúc trong state/trace: hình thức, truyền thống, ngôn ngữ đầu ra và tối đa ba sắc thái.
  Các dạng được hướng dẫn trực tiếp gồm thơ Trung Hoa cổ điển, haiku/tanka/senryu/haibun, sijo, sonnet,
  villanelle, pantoum, ballad, ode, elegy, spoken word, thơ tự do và lục bát.
- Output guard phân biệt hư cấu với khẳng định sự kiện: tiêu đề/con số/câu thoại được sáng tác không bị chặn như tên
  tác phẩm hay số liệu bịa; lọc bí mật, URL lạ và các rào chắn hệ thống vẫn giữ nguyên.
- Quy tắc đa văn hóa ưu tiên đặc trưng hình thức, tránh khuôn mẫu dân tộc; quy tắc âm vị Nhật/Hàn/Hán khi viết bằng
  tiếng Việt phải được coi là chuyển thể. Phân tích đa chiều phải có căn cứ văn bản, không tạo phản biện giả.
- Thêm bộ kiểm tra hình thức trước khi gửi: haiku/senryu/tanka/sijo/lục bát và các dạng cố định được đếm dòng/tiếng;
  nếu hai lượt biên tập model vẫn sai, fallback điền ô rồi cắt phần dư. Smoke model thật xác nhận haiku 5-7-5.

## Gia sư Ngữ văn + StudyScope thống nhất — 05/10/2026

- Thêm bộ định tuyến xác định cho viết bài/đoạn, lập dàn ý, nghị luận xã hội, nghị luận văn học, đọc hiểu, viết sáng
  tạo và chấm sửa. Câu “phân tích dữ liệu bán hàng” vẫn là chat thường; Toán vẫn đi bộ giải xác định.
- Prompt production v10 có hướng dẫn theo cấp lớp 1–12, cấu trúc luận điểm-dẫn chứng-phân tích, phản biện, sửa bài giữ
  giọng học sinh và rào chắn không bịa câu thơ/chi tiết tác phẩm. Bài Ngữ văn có trần 1.400 token riêng.
- Giao diện `/web/math` đổi tên hiển thị thành StudyScope, thêm câu hỏi mẫu Ngữ văn nhưng vẫn giữ API/URL tương thích.
- Thêm khung chương trình có nguồn Bộ GD&ĐT, không sao chép toàn văn sách giáo khoa/tác phẩm còn bản quyền. Phân tích
  sát văn bản dùng đoạn học sinh cung cấp hoặc nguồn truy xuất; thiếu ngữ liệu phải hỏi lại.
- Dữ liệu SFT Ngữ văn v1 gồm 12 mẫu tổng hợp; sau khi gộp toàn bộ có 81 train, 24 validation. Đây không phải kho toàn
  bộ tác phẩm. Rubric đánh giá gồm đúng đề, bố cục, lập luận, toàn vẹn dẫn chứng, phù hợp cấp lớp và giá trị học tập.
- Train thật ứng viên `vera-edu-ent-v3-writing-20261004-192436`: 28 bước, 350,2 giây, validation loss 3,1860 →
  2,1430, peak VRAM 13,59 GB; nạp lại từ base sạch thành công, loss mẫu 1,3779 hữu hạn.
- Eval tách biệt 8 tình huống: base 6/8; VeRA v1 6/8, 0 tín hiệu bịa; VeRA v3 6/8 và 1 tín hiệu trích dẫn không hợp
  lệ. V3 **FAIL gate**, nên không promote; production tiếp tục dùng VeRA v1 cùng prompt/routing mới. Cần dữ liệu do
  giáo viên duyệt và đánh giá thủ công trước lần fine-tune tiếp theo.

## Trả lời sâu + fine-tune VeRA v2 — 05/10/2026

- Prompt production v9: câu kiến thức mặc định 220-380 từ, trả lời trực tiếp rồi mở rộng bằng nguyên lý/ví dụ/so
  sánh/hành động; chào hỏi và yêu cầu "ngắn" vẫn có luật 1-3 câu. Tin mới dùng web/RAG có nguồn, không dạy model nhớ
  tin tức.
- Dữ liệu SFT v2 là dữ liệu tổng hợp: gộp 72 hội thoại v1 với 21 hội thoại bổ sung; 73 train / 20 validation. Manifest
  lưu SHA-256 của từng file. Nội dung AI soạn là mục tiêu hành vi, không được coi là ground truth con người.
- Train thật `vera-edu-ent-v2-rich-20261004-180414`: 24 bước, 288,3 giây, 1.170.432 tham số train; validation loss
  2,7975 → 1,8142; peak VRAM 13,31 GB. Reload từ base sạch thành công, loss mẫu validation 1,4089 hữu hạn.
- Eval tách biệt 17 kịch bản, chạy cùng process cho base/v1/v2. Base 14/17; adapter active v1 14/17 (10/10 câu sâu,
  citation 100%, 0 tín hiệu bịa); v2 14/17 (9/10 câu sâu, citation 90%, 1 tín hiệu bịa). V2 **FAIL gate** vì cần
  tối thiểu 16/17 và không được giảm citation, nên không register/promote; production vẫn là VeRA v1.
- Kết quả quan trọng: prompt mới + adapter v1 hiện có đã nâng nhóm câu kiến thức sâu lên 10/10, trung bình 1.469 ký
  tự trong lần so sánh chung. Adapter v2 không chứng minh được tốt hơn nên được giữ làm artifact thử nghiệm.
- Web grounding đã có code nhưng chưa thể cung cấp dữ liệu mới liên tục khi chưa có Brave API key. Không giả vờ rằng
  model tự cập nhật hoặc đã tra web.

## Hiểu ảnh đa phương thức + web grounding — 05/10/2026

- Thêm VLM OpenAI-compatible vào pipeline ảnh, đã bật local bằng `qwen3-vl:8b` trên Ollama. VLM cung cấp mô tả cảnh,
  màu sắc, vị trí, quan hệ, số lượng và câu trả lời theo câu hỏi; OCR/YOLO vẫn chạy để đối chiếu và fallback.
- Không ghi/lưu pixel ảnh; chỉ lưu phân tích đã che PII. Ảnh không gửi lên Langfuse. Với GPU 16 GB, cấu hình local dỡ
  VLM khỏi Ollama sau mỗi ảnh để text LLM không OOM.
- Smoke thật trên ảnh bài tây: phân tích ảnh 18,18–18,39 s; lượt đầy đủ ảnh → text LLM khoảng 25,07 s; text model chính
  + VeRA trả lời sau khi VLM được dỡ. VLM/YOLO có lúc mâu thuẫn (VLM đếm 2–3, YOLO tạo 4 phát hiện và VLM đọc sai
  chất lá), nên prompt bắt buộc nêu mâu thuẫn, không quảng cáo khả năng “nhìn đúng mọi thứ”.
- Thêm Brave LLM Context client: query che PII, nguồn web đi qua cùng citation/output guardrail, mode `auto|always`,
  lỗi web không làm hỏng lượt chat. Contract và graph đã test bằng mock; **chưa chạy API Brave thật vì chưa có key**.
- Langfuse synthetic trace `6bf5eb2415c9b8a04c3805bf0c64c407` đã kiểm tra: `describe-image` là generation,
  `grounding-retrieval` là retriever, output check là guardrail, content capture vẫn tắt.
- Kiểm thử cuối: `ruff check .` sạch; `pytest -q` → **251 passed**.
- Không fine-tune trọng số VLM trong đợt này: chưa có tập ảnh/câu hỏi/đáp án được người duyệt và tập test tách biệt.
  Model đa phương thức tạo năng lực nhìn ảnh ngay; feedback được review mới được phép đưa vào lần QLoRA/LoRA sau.

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

## Đợt 05/10/2026: MathScope, nhập Math Lab và tra cứu web fallback

- Thêm trang `/web/math` và API `/web/math/solve`, `/web/math/read-image`. Bộ giải dùng AST allowlist + SymPy,
  tính chính xác và thế ngược nghiệm; không chạy `eval`/`sympify` trên đầu vào. Phạm vi đã kiểm thử: số học, phân số,
  căn, phương trình đa thức một ẩn đến bậc 4, hệ tuyến tính 2-3 ẩn, một số họ bài lời văn cộng/trừ/nhóm/chia đều,
  diện tích/chu vi hình chữ nhật.
- SVG phía trình duyệt: trục số, phân số, nhóm vật, cân bằng, đồ thị/hệ tọa độ, hình chữ nhật; tải được tệp SVG.
- Ảnh đề: OCR/VLM chép đề, bộ giải xác định mới quyết định đáp án. Thử ảnh thật từ Math Lab: đọc đúng bài 6 viên bi
  thêm 2 viên, trả 8, `deterministic-word-rules/pass=true`, kèm văn bản OCR để đối chiếu.
- Nhập 7.469 tệp / 396 MB từ EduVisionAI vào `artifacts/math_lab_import` với SHA-256. Selector trực quan được giữ cho
  vai trò chọn view; LoRA GSM8K Qwen3-0.6B giữ làm bằng chứng nhưng khóa khỏi runtime vì base/adapter cùng 14/50 và
  checkpoint tự đánh dấu `accepted_for_runtime=false`.
- Khi không có Brave key, MediaWiki API hoạt động làm fallback có URL cho kiến thức phổ thông. Đã thử live truy vấn
  “định lý Pythagore” và nhận 4 kết quả. Tin mới/luật/giá/lịch vẫn cần Brave và nguồn chính thức.
- Kiểm chứng cuối: Ruff sạch, JavaScript parse sạch, **275 test Python qua**; `/ready=true`, DB và heartbeat worker OK;
  model Qwen3-4B + VeRA v1, API, worker và Cloudflare tunnel chạy dưới systemd user transient units.
- Bổ sung phân luồng tự động thống nhất: `/web/math/route` chỉ giữ câu toán đã kiểm chứng ở MathScope; câu hỏi thường
  dùng `/web/messages` và cùng session chatbot/RAG/web. Graph chính cũng dùng MathScope cho phương trình/lời văn trước
  khi gọi LLM. Kiểm tra end-to-end: “Transformer hoạt động như thế nào?” trả lời dài kèm nguồn arXiv; “Giải phương
  trình 3x + 5 = 20” trả `x = 5` và thế ngược. Tổng test sau thay đổi: **280 passed**.
