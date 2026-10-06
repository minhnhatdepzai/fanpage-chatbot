# fanpage-chatbot

Chatbot AI tiếng Việt cho Facebook Fanpage Messenger và khung chat website: FastAPI (webhook/API) → PostgreSQL
(hàng đợi bền vững + pgvector) → worker → LangGraph → Qwen3-4B-Instruct-2507 + adapter VeRA chạy local trên GPU
(dự phòng: Ollama Q4) → kiểm tra đầu ra → Send API / widget. Hiểu ảnh bằng OCR + YOLO và VLM Qwen3-VL tùy chọn;
tra cứu web có nguồn qua Brave LLM Context hoặc DDGS + MediaWiki khi không có API key.

Mục tiêu: giảm câu trả lời thiếu căn cứ bằng kho kiến thức có nguồn
(`config/knowledge/`) hoặc thông tin fanpage (`config/fanpage_profile.yaml`), tự gắn link nguồn khi trích dẫn;
câu hỏi kiến thức không có nguồn thì bị gắn ghi chú "chưa có nguồn kiểm chứng" (hoặc từ chối trả lời đoán ở chế độ
`strict`). Xem [Độ chính xác](#độ-chính-xác-và-nguồn).

Tiến độ chi tiết và việc còn dở: [docs/progress.md](docs/progress.md). Môi trường máy: [docs/environment-report.md](docs/environment-report.md).

## Mục lục

- [Cài đặt từ máy mới](#cài-đặt-từ-máy-mới)
- [Chạy bot](#chạy-bot)
- [Độ chính xác và nguồn](#độ-chính-xác-và-nguồn)
- [Fine-tune và đánh giá](#fine-tune-và-đánh-giá)
- [Tài liệu PDF/Word](#tài-liệu-pdfword-rag-pgvector-và-trợ-lý-nội-bộ)
- [StudyScope: Toán, Ngữ văn và Tiếng Anh](#studyscope-toán-ngữ-văn-và-tiếng-anh)
- [Khung chat website](#khung-chat-website-typescript)
- [Ảnh: OCR + YOLO + VLM](#ảnh-ocr--yolo--vlm)
- [Tra cứu web có nguồn](#tra-cứu-web-có-nguồn)
- [Ollama dự phòng](#ollama-model-4-bit-dự-phòng)
- [Docker cho API/worker](#docker-cho-apiworker)
- [Dataset Kaggle](#dataset-kaggle)
- [Vận hành, sao lưu và xử lý lỗi](#vận-hành-sao-lưu-và-xử-lý-lỗi)
- [Cấu trúc mã nguồn](#cấu-trúc-mã-nguồn)

## Cài đặt từ máy mới

### 1. Yêu cầu

Luồng triển khai đã chạy trên Ubuntu 24.04, Python 3.12, GPU NVIDIA RTX 5060 Ti 16 GB và RAM 32 GB.
Nên có GPU 16 GB để chạy đồng thời LLM bf16, embedding và xử lý ảnh; huấn luyện cần dừng model server để giải phóng VRAM.
Dung lượng tải riêng Qwen khoảng 8 GB; cần thêm chỗ cho môi trường Python/CUDA, embedding, OCR, YOLO và dataset.
Chuẩn bị ít nhất 40 GB trống cho chạy thử, nhiều hơn khi huấn luyện. Windows nên dùng WSL2 có GPU passthrough;
chưa kiểm thử Windows/macOS. `run_local.sh` và registry adapter sử dụng tiện ích Linux.

| Thành phần | Vai trò |
|---|---|
| Git, curl, Python 3.12 và uv | lấy mã nguồn, cài thư viện theo `uv.lock` |
| Docker Engine + Compose | PostgreSQL 17 có pgvector |
| NVIDIA driver tương thích PyTorch trong lockfile | suy luận/huấn luyện GPU trên host |
| Node.js 24 + npm | build và test widget TypeScript; không cần nếu dùng bundle đã commit |
| cloudflared | tunnel HTTPS để Meta gọi webhook; không bắt buộc khi chỉ thử localhost |
| edge-tts | giọng neural Việt, Mỹ và Anh cho phòng đọc Văn/Tiếng Anh; cần Internet, có Web Speech fallback |
| ddgs | metasearch web không cần API key; kết quả vẫn qua lọc URL, grounding và kiểm định nguồn |
| Tài khoản Meta Developer và quyền quản trị Page | kết nối Messenger |
| Ollama, Kaggle, Langfuse | tùy chọn, không bắt buộc để chạy luồng chính |

Cài Docker theo [hướng dẫn Ubuntu chính thức](https://docs.docker.com/engine/install/ubuntu/).
Cài cloudflared theo [trang tải chính thức](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/).
Cài uv bằng [installer chính thức](https://docs.astral.sh/uv/getting-started/installation/):

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | sh
# Mở terminal mới nếu lệnh uv chưa có trong PATH
uv --version
docker compose version
docker info
nvidia-smi
```

`docker info` phải chạy được bằng tài khoản hiện tại vì script không gọi sudo.
Không cần NVIDIA Container Toolkit cho cách chạy mặc định: model chạy trên host, chỉ PostgreSQL chạy trong Docker.

### 2. Lấy mã nguồn và cài thư viện

```bash
git clone https://github.com/minhnhatdepzai/fanpage-chatbot.git
cd fanpage-chatbot
uv python install 3.12
uv sync --frozen --extra ml --extra data
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Repo public có thể clone không cần đăng nhập; quyền push chỉ dành cho tài khoản được cấp quyền.
`ml` cài torch/transformers/PEFT/OCR/YOLO; `data` chỉ cần cho Kaggle. API/worker dùng model ở máy khác chỉ cần
`uv sync --frozen`. Sau khi đã cài GPU extras, các lệnh kiểm tra dưới đây dùng `--no-sync` để giữ môi trường đã cài.
Không chạy lại `uv sync` thiếu extras nếu vẫn cần thư viện GPU.

### 3. Cấu hình .env

```bash
cp .env.example .env       # chỉ làm lần đầu, không ghi đè .env đang dùng
chmod 600 .env
python3 -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Chạy lệnh sinh khóa riêng cho từng biến cần bí mật; mở `.env` bằng trình soạn thảo và điền:

| Biến | Cách điền |
|---|---|
| `POSTGRES_PASSWORD` | mật khẩu mới cho PostgreSQL; chuỗi URL-safe giúp tránh lỗi URL |
| `DATABASE_URL` | `postgresql+psycopg://chatbot:MẬT_KHẨU@127.0.0.1:55432/chatbot`, mật khẩu khớp dòng trên |
| `META_APP_ID`, `META_PAGE_ID` | ID app và Page thực tế |
| `META_APP_SECRET`, `META_PAGE_ACCESS_TOKEN` | lấy trong cấu hình app Meta |
| `META_VERIFY_TOKEN` | khóa tự sinh, nhập cùng giá trị khi xác minh webhook |
| `MODEL_SERVER_API_KEY` | khóa tự sinh dùng giữa ứng dụng và model server |
| `ADMIN_API_KEY` | khóa tự sinh tối thiểu 24 ký tự cho `/admin`; để trống sẽ khóa admin API |
| `PSEUDONYM_SECRET` | khóa tự sinh để ký phiên web và giả danh người dùng; giữ ổn định khi chuyển máy |
| `MESSENGER_SEND_MODE` | bắt đầu bằng `allowlist`; `disabled` không gửi; `all` cho phép mọi người dùng đủ điều kiện |
| `MESSENGER_ALLOWED_PSIDS` | PSID tài khoản thử, phân tách dấu phẩy |
| `LLM_FALLBACK_BASE_URL`, `LLM_FALLBACK_MODEL` | để trống cả hai nếu chưa cài Ollama |
| `WEB_ALLOWED_ORIGINS` | origin website nhúng widget, gồm scheme và port nếu có, không kèm đường dẫn |

Sửa `config/fanpage_profile.yaml` thành thông tin Page của bạn. Kho YAML ở `config/knowledge/` là nội dung mẫu;
kiểm tra nguồn và thời điểm trước khi dùng. `.env.example` liệt kê các tùy chọn còn lại, giá trị mặc định đầy đủ ở
`app/config/settings.py`. Không cần key OpenAI khi chạy Qwen local.

Chỉ thử web thì có thể để trống các trường Meta thực tế, nhưng `/ready` vẫn báo thiếu cấu hình Messenger;
kiểm tra `/health`, model và luồng demo riêng. Không đặt thông tin giả rồi coi `/ready` là bằng chứng Messenger hoạt động.

### 4. Chuẩn bị model và widget

```bash
uv run --no-sync python scripts/download_model.py --dry-run
uv run --no-sync python scripts/download_model.py
npm --prefix web/widget ci
npm --prefix web/widget run build
```

Lệnh tải khóa revision Qwen theo `.env`, giới hạn mặc định 20 GB. Embedding tiếng Việt và OCR/YOLO tải khi model server
khởi động lần đầu, cần mạng và có thể mất vài phút. Nếu script start hết thời gian chờ, xem log model rồi gọi start lại.
Bundle `app/web/static/widget.js` đã có trong repo nên bước npm chỉ bắt buộc khi sửa widget.

**Repo không chứa model đã train, registry đang active, dữ liệu hội thoại, tài liệu PDF/Word hay .env.**
Clone mới chạy base Qwen (không có adapter VeRA) và YOLO COCO, do đó chất lượng có thể khác máy gốc.
Để dùng lại adapter/YOLO riêng, sao chép artifact từ bản sao lưu tin cậy hoặc tự train theo các mục bên dưới.
Đừng đặt `VISION_CUSTOM_WEIGHTS` tới file không tồn tại.

### 5. Khởi động và kiểm tra

```bash
./scripts/run_local.sh start
./scripts/run_local.sh status
curl -fsS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
curl -fsS http://127.0.0.1:8100/health
```

Mở **http://127.0.0.1:8000/web/demo**, gửi câu hỏi rồi chờ worker trả lời.
`/health` chỉ chứng minh tiến trình sống; `/ready` kiểm tra DB/migration/cấu hình Meta, **không kiểm tra suy luận model**
và heartbeat worker không chặn readiness. Vì vậy cần xem `worker_heartbeat` và thử hội thoại để xác nhận luồng đầy đủ.

Nếu muốn chạy từng thành phần hoặc không dùng tunnel, mở ba terminal ở thư mục repo sau khi chạy migrate:

```bash
docker compose up -d --wait postgres
uv run --no-sync botctl db migrate
# Terminal 1:
uv run --no-sync python -m serving.server
# Terminal 2:
uv run --no-sync botctl api
# Terminal 3:
uv run --no-sync botctl worker
```

Các tiến trình tự chạy bằng tay phải dừng bằng Ctrl+C; script chỉ quản lý PID do chính script tạo.
Script dùng cổng mặc định 8000/8100 cho health và tunnel; nếu đổi cổng hãy chạy từng thành phần và tunnel tương ứng.

## Chạy bot

Yêu cầu: Docker, `uv`, GPU NVIDIA ≥ 10 GB VRAM (model ~8 GB), `cloudflared` (tunnel HTTPS miễn phí), `.env` đã điền
(xem `.env.example`).

```bash
./scripts/run_local.sh start      # postgres + migrate + model server + API + worker + tunnel, in URL webhook
uv run botctl meta set-webhook https://<xxx>.trycloudflare.com/webhook   # URL tunnel đổi mỗi lần start
./scripts/run_local.sh status
./scripts/run_local.sh stop
```

Sau đó nhắn tin vào Fanpage bằng tài khoản có PSID trong `MESSENGER_ALLOWED_PSIDS`
(`MESSENGER_SEND_MODE=allowlist` chỉ trả lời các tài khoản này; `all` = trả lời mọi người).

Log: `runs/logs/{model,api,worker,tunnel}.log`. Kiểm tra cấu hình Meta (không in bí mật): `uv run botctl meta check`.

### Meta Dashboard (một lần)

| Giá trị | Lấy ở đâu | Dùng để |
|---|---|---|
| `META_APP_SECRET` | App settings → Basic → App secret | xác minh chữ ký webhook `X-Hub-Signature-256` |
| `META_PAGE_ACCESS_TOKEN` | Messenger → Messenger API Settings → Generate token | **gửi** tin nhắn (không gọi được model AI) |
| `META_VERIFY_TOKEN` | tự đặt chuỗi ngẫu nhiên | chỉ dùng lúc Meta xác minh URL webhook (GET) |

`botctl meta set-webhook` đặt callback URL + các field `messages, messaging_postbacks, message_echoes,
message_reactions` cho app, rồi thử đăng ký Page với app. Bước đăng ký Page cần quyền `pages_manage_metadata`; nếu token
không có quyền này và lệnh báo lỗi ở `page_subscription` và bot không nhận được tin, vào
Messenger → Messenger API Settings → Webhooks → chọn Page → **Add subscriptions** và tích 4 field trên.

App ở chế độ Development chỉ nhận tin từ tài khoản có vai trò trong app (admin/developer/tester).

## Độ chính xác và nguồn

- **Kho kiến thức** `config/knowledge/*.yaml`: mỗi mục có `id`, `title`, `source` (bắt buộc), `keywords`, `content`.
  Mục thiếu nguồn bị bỏ qua. 28 mục (kiến thức AI + tin giáo dục, học tập, phim Việt, model AI tháng 9/2026, GPT-5,
  RTX 50 Series, lỗi CUDA OOM) do Claude soạn từ trang nguồn — **quản trị viên nên rà soát**.
- **Truy xuất**: một mục chỉ được đưa vào prompt khi câu hỏi chứa nguyên vẹn một cụm `keywords` của mục (không dấu
  cũng được), xếp hạng BM25. Thử: `uv run botctl knowledge search "vera khác gì lora"`.
- **Kiểm tra đầu ra** (`app/conversation/output_check.py`), không phụ thuộc model:
  - số nguồn `[n]` hợp lệ → hệ thống tự gắn danh sách `Nguồn:` với link thật; `[n]` không tồn tại bị xóa;
  - link model tự viết mà không có trong kho → xóa; dòng "Nguồn: ..." model tự bịa → xóa;
  - câu tự nhận "đã được kiểm chứng", "theo nguồn chính thức/Wikipedia/thống kê…" khi không trích nguồn thật → xóa;
  - câu hỏi kiến thức (hoặc câu trả lời có số liệu) không trích nguồn → gắn ghi chú "chưa có nguồn đã kiểm chứng"
    (câu trả lời về ảnh: ghi chú "dựa trên nhận diện tự động"); ghi chú model tự viết bị bỏ để không lặp;
  - câu "chưa ra mắt / dự kiến / không có thông tin chính thức…" khi nguồn không nói vậy → xóa (kiến thức cũ của
    model); câu tự tính khoảng thời gian ("cách đây 1 năm 8 tháng") không có nguồn → xóa.
- **Tính bằng code, không gọi model**: ngày/thứ (`datecalc.py`, giờ Việt Nam) và phép tính số học đơn giản
  (`mathcalc.py`, vd. "50 nhân 50 bằng mấy").
- `GROUNDING_MODE=strict`: câu hỏi kiến thức không có nguồn → không gọi model, trả lời rằng chưa có nguồn.
- `LLM_TEMPERATURE=0.3` để giảm bịa.

**Giới hạn thật**: model 4B vẫn có thể diễn đạt sai *trong* câu trả lời có nguồn, hoặc nêu kiến thức cũ (ví dụ số
tỉnh thành đã thay đổi) — khi đó ghi chú cảnh báo được gắn nhưng câu sai vẫn hiện. Dùng `strict` và bổ sung kho kiến thức để giảm trả lời thiếu nguồn; chế độ này cũng không bảo đảm mọi diễn giải đều đúng. Nhận diện "câu hỏi kiến thức" dựa trên luật, có thể sót.

## Fine-tune và đánh giá

```bash
./scripts/run_local.sh stop                        # giải phóng GPU (model server)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python -m training.scripts.train_sft --config training/configs/vera-edu-ent-v2-rich.yaml
uv run python -m evaluation.scripts.compare --adapter artifacts/adapters/<id> --scenarios evaluation/datasets/edu_ent_eval_v2_rich.yaml
uv run botctl model register artifacts/adapters/<id>
uv run botctl model attach-eval <id> evaluation/results/<run>/report-<id>.json
uv run botctl model promote <id> --reason "..."    # chỉ đạt khi gate PASS
uv run botctl model rollback --reason "..."
```

Fine-tune dạy phong cách và hành vi có căn cứ; **tin tức mới phải lấy từ web/RAG hoặc thêm vào
`config/knowledge/`** (có nguồn), không nhồi vào trọng số. Dataset v2 gộp v1 với tập bổ sung trả lời sâu; manifest lưu
checksum của từng tệp. Kết quả lần chạy gần nhất, gồm cả adapter thử nghiệm không qua gate:
[docs/progress.md](docs/progress.md).

## Tài liệu PDF/Word (RAG, pgvector) và trợ lý nội bộ

```bash
uv run botctl docs ingest tai-lieu.pdf --visibility public --source "https://link-goc"   # bot công khai được dùng
uv run botctl docs ingest quy-trinh.docx --visibility internal                         # chỉ trợ lý nội bộ
uv run botctl docs search "câu hỏi" --visibility all      # xem đoạn nào được tìm thấy + điểm tương đồng
uv run botctl ask "thí sinh được phúc khảo trong bao lâu"  # trợ lý nội bộ (hoặc POST /admin/ask có ADMIN_API_KEY)
```

Đọc PDF (pypdfium2, trang scan được OCR), Word (python-docx) → làm sạch → chia đoạn có số trang → embedding tiếng Việt
(`AITeamVN/Vietnamese_Embedding`, 1024 chiều) → pgvector (HNSW) + tìm từ khóa, gộp RRF. Chỉ nhận đoạn có độ tương
đồng ≥ `DOC_MIN_SIMILARITY` (0,40, đã hiệu chỉnh). Tài liệu `internal` bị lọc ngay trong SQL với kênh công khai.
Tài liệu public có dấu hiệu dữ liệu cá nhân bị từ chối nạp.

## StudyScope: Toán, Ngữ văn và Tiếng Anh

Sau khi chạy API, giao diện được tách thành bốn URL độc lập:

- `/web/chat`: chatbot tổng quát, vẫn tự nhận diện đề Toán/Văn/Anh.
- `/web/math`: phòng thi Toán lớp 1–12, ngân hàng đề và SVG lời giải chuyển động.
- `/web/literature`: phòng thi Ngữ văn, đọc hiểu, nghị luận dài, sáng tác và audio lesson.
- `/web/english`: khung StudyScope chứa nguyên bản build `task1-coach-pages` do người dùng cung cấp.

Router phía sau vẫn tự nhận biết câu hỏi thường, đề Toán, yêu cầu Ngữ văn và bài Tiếng Anh; URL chỉ tổ chức trải
nghiệm học tập, không ép sai môn. Trang Toán giải biểu thức, phân số, phương trình đa thức một ẩn
đến bậc 4 và hệ tuyến tính 2-3 phương trình; mỗi nghiệm được thế ngược trước khi trả về. Hình SVG (trục số, phân số,
nhóm đồ vật, cân bằng và đồ thị) được dựng trực tiếp từ dữ liệu nghiệm và có thể tải xuống.

Đây là chatbot thống nhất, không phải ô chỉ nhận toán: `/web/math/route` tự phân luồng từng lượt. Đề toán đã hỗ trợ
đi vào bộ giải/SVG; câu hỏi bình thường đi qua cùng hàng đợi chatbot, lịch sử, RAG và web search như widget. Ví dụ
“Transformer là gì?” trả lời đa lĩnh vực có nguồn, còn “3x + 5 = 20” trả nghiệm đã thế ngược.

Yêu cầu như “lập dàn ý”, “viết đoạn/bài văn”, “nghị luận xã hội”, “phân tích thơ/truyện”, “đọc hiểu” và “chấm sửa
bài” được chuyển sang gia sư Ngữ văn (`app/writing_tutor/`). Luật nhận diện hỗ trợ tiếng Việt có/không dấu và cấp
lớp 1–12, nhưng không bắt nhầm các câu như “phân tích dữ liệu bán hàng”. Gia sư điều chỉnh mức diễn đạt theo lớp,
đưa bài mẫu kèm gợi ý cá nhân hóa, giữ giọng học sinh khi sửa bài và không tự bịa câu thơ/chi tiết tác phẩm.

Nhánh **sáng tác mới** được phân biệt với phân tích tác phẩm: bot được phép tự tạo câu thơ, tiêu đề, nhân vật, lời
thoại và chi tiết hư cấu. Bộ nhận diện giữ các lựa chọn người dùng về ngôn ngữ, sắc thái và hình thức; có hướng dẫn
riêng cho các dạng phổ biến như tuyệt cú/luật thi, haiku/tanka/senryu/haibun, sijo, sonnet, villanelle, pantoum,
ballad, ode, elegy, spoken word, thơ tự do và lục bát. “Phong cách Trung/Nhật/Hàn/Âu/Mỹ” không được biến thành
khuôn mẫu dân tộc; với thể thơ phụ thuộc âm vị nguyên ngữ, bản tiếng Việt phải được hiểu là phỏng theo/chuyển thể.

Kho `config/knowledge/ngu-van.yaml` chỉ lưu khung chương trình có nguồn chính thức. Repo không sao chép toàn văn sách
giáo khoa hay tác phẩm còn bản quyền: khi phân tích sát văn bản, học sinh cần gửi đoạn trích; nếu không có ngữ liệu hoặc
nguồn truy xuất, bot phải hỏi lại hay diễn giải thay vì giả trích dẫn. `training/datasets/writing_sft_v1.yaml` là 12
mẫu tổng hợp dạy hành vi, không phải “toàn bộ văn học lớp 1–12” và không được coi là ground truth giáo viên.

Gia sư Tiếng Anh (`app/english_tutor/`) nhận diện ngữ pháp, từ vựng, đọc hiểu, nghe dựa trên transcript, nói, phát âm,
dịch, viết và IELTS. Trắc nghiệm phải có đáp án kèm quy tắc/căn cứ; Reading chỉ dùng passage được gửi; Writing Task 1
không được tự đặt số; chấm band chỉ là ước lượng theo rubric. `task1-coach-pages` do người dùng cung cấp được dùng để
phục vụ nguyên bản tại `/task1-coach-pages/` và được nhúng trong `/web/english`. Bundle, font, mascot, audio synthetic,
`practice-sources.md` và `THIRD_PARTY_NOTICES.txt` được giữ cùng nhau để không làm mất provenance. Ứng dụng không nhận
bài tự soạn là đề Cambridge/IELTS chính thức. Nguồn kiến thức runtime nằm trong `config/knowledge/tieng-anh.yaml`.

Đã train thật ứng viên `vera-edu-ent-v4-english-20261005-060531` trên bộ gộp 99 train / 30 validation: 32 bước,
validation loss 3,2258 → 2,2915, lưu/nạp lại thành công. Tuy nhiên eval Tiếng Anh tách biệt chỉ tăng 3/10 → 4/10,
không đạt gate yêu cầu +2 kịch bản; adapter được đăng ký ở trạng thái `candidate` và **không promote**. Production
vẫn dùng VeRA v1 cùng router/prompt v12 mới; chi tiết xem `docs/progress.md`.

Mở rộng THPT thêm bộ giải ký hiệu cho đạo hàm, giới hạn, tích phân xác định, bất phương trình, phương trình mũ và
tổ hợp/chỉnh hợp, cùng dữ liệu Toán–Văn–Anh lớp 10–12. Ứng viên
`vera-edu-ent-v5-high-school-20261005-073853` được train thật 40 bước trên 115 train / 36 validation; validation loss
giảm 3,3050 → 2,3427 và checkpoint nạp lại thành công. Tuy nhiên eval tách biệt chỉ đạt 8/13 so với base 9/13,
nên v5 chỉ ở trạng thái `candidate`, không promote. Production tiếp tục dùng VeRA v1; các bộ giải xác định và router
v13 vẫn hoạt động độc lập vì được kiểm thử bằng mã và thế ngược kết quả.

Người dùng có thể tải ảnh đề PNG/JPEG/WEBP. OCR/VLM chỉ chép lại đề; `POST /web/math/read-image` vẫn bắt buộc bộ giải
xác định tính và kiểm chứng đáp án. Nếu không tách/kiểm chứng được, hệ thống trả lỗi an toàn và hiện phần chữ đã nhận
dạng thay vì đoán. Xem phạm vi, chính sách checkpoint và lệnh nhập 396 MB tài sản Math Lab tại
[docs/math-tutor.md](docs/math-tutor.md).

Không có hệ thống nào bảo đảm đúng mọi bài lớp 1-12. Nhãn cấp lớp và liên kết chương trình là metadata sư phạm;
phạm vi đúng đã xác minh được công bố rõ. Câu ngoài phạm vi hiện tại phải đi qua bộ đánh giá mới trước khi quảng bá.

## Khung chat website (TypeScript)

Mã nguồn `web/widget/` (TypeScript + esbuild, không thư viện runtime). Build: `cd web/widget && npm ci && npm run build`
(ra `app/web/static/widget.js`), test: `npm test`. Nhúng:

```html
<script src="https://<địa-chỉ-api>/web/widget.js" data-title="AI Test" data-color="#0866ff" defer></script>
```

Thêm tên miền vào `WEB_ALLOWED_ORIGINS`. Demo: `/web/demo`. Tin nhắn web đi chung hàng đợi/worker/kiểm tra đầu ra với
Messenger; phiên ký HMAC; giới hạn tần suất; hiển thị bằng `textContent` (chống XSS). Gửi ảnh được (nút 🖼).

Hàng đợi dùng trực tiếp PostgreSQL hiện có, không giữ tác vụ trong RAM: mỗi cuộc hội thoại được xử lý tuần tự,
nhiều người dùng chạy song song theo `WORKER_CONCURRENCY` (mặc định 4), và `FOR UPDATE SKIP LOCKED` tránh hai worker
nhận cùng một lượt. `POST /web/messages` cùng `GET /web/messages` trả thêm `queue` gồm `phase`, `position`, `total`,
`active_jobs`, `worker_slots`; giao diện trang học và widget hiển thị vị trí chờ rồi tự chuyển sang “đang xử lý”.
API tuyệt đối không trả định danh hay nội dung của những người khác trong hàng đợi. Khi tăng concurrency, phải đo
VRAM/độ trễ model trước vì nhiều worker không đồng nghĩa GPU có thể sinh nhiều đáp án đồng thời nhanh hơn.

Trang công khai khi chạy Quick Tunnel là `https://<quick-tunnel>/web/math`. Đây là URL tạm và sẽ đổi khi tunnel
khởi động lại. Muốn URL cố định cho khách, cấu hình named Cloudflare Tunnel cùng tên miền do bạn sở hữu; backend,
model và PostgreSQL vẫn chạy trên máy này. Trước khi gửi link, kiểm tra cả `/ready`, worker heartbeat và một lượt
chat thật — HTTP 200 của trang không chứng minh model đã phản hồi.

## Ảnh: OCR + YOLO + VLM

Người dùng gửi ảnh (Messenger hoặc widget) → OCR đọc chữ + YOLO nhận diện vật thể + VLM tùy chọn mô tả cảnh, màu sắc,
vị trí, quan hệ và số lượng → kết quả cùng câu hỏi và ngữ cảnh hội thoại đưa cho LLM → trả lời trên đúng kênh. Kết quả
phân tích (đã che dữ liệu cá nhân, không có ảnh/toạ độ) được lưu vào tin nhắn nên **hỏi tiếp về ảnh vừa gửi** vẫn có
ngữ cảnh. OCR/YOLO vẫn là fallback và là tín hiệu đối chiếu khi VLM sai.

Repo không hứa “nhìn được mọi thứ”: VLM vẫn có thể nhầm vật nhỏ, màu trong ánh sáng xấu và số lượng khi vật bị che.
Không dùng VLM để nhận diện danh tính hoặc suy đoán thuộc tính nhạy cảm. `VISION_VLM_ENABLED=false` mặc định vì bật nó
sẽ gửi ảnh tới `VISION_VLM_BASE_URL`; nên trỏ vào Ollama/vLLM local nếu ảnh người dùng không được phép ra ngoài máy.

Ví dụ chạy Qwen3-VL đã có trong Ollama:

```bash
ollama pull qwen3-vl:8b
# .env
VISION_VLM_ENABLED=true
VISION_VLM_BASE_URL=http://127.0.0.1:11434/v1
VISION_VLM_MODEL=qwen3-vl:8b
VISION_VLM_OLLAMA_UNLOAD_AFTER_REQUEST=true  # nên bật trên GPU 16 GB khi text LLM cũng nằm trên GPU
```

**Model trên máy phát triển khi đánh giá** (clone mới chưa có trọng số custom):
- YOLO26s COCO (80 lớp đồ vật thông dụng, tên dịch tiếng Việt), ngưỡng 0,4.
- YOLO26n tự train **v3** `artifacts/vision/edu-ent-v3-yolo26n-20260924-003644.pt`, 91 lớp, ngưỡng 0,6:
  đồ dùng học tập & lớp học (20: bút, tẩy, gọt bút chì, thước kẻ, sách, bàn, ghế, bảng trắng, kệ sách, cặp...),
  mặt xúc xắc 1-6, 12 quân cờ vua, 52 lá bài tây, đàn guitar (còn yếu).
- Tập test (1.149 ảnh, không dùng khi train): mAP50 0,930 / mAP50-95 0,803. Ở ngưỡng 0,6, precision theo nhóm:
  lớp học 0,92 · bài tây 0,97 · đồ dùng học tập 0,88 · cờ vua 0,83 · xúc xắc 0,81 · guitar 1,00 nhưng recall 0,14.

Khi VLM tắt, giới hạn cũ vẫn áp dụng: OCR chỉ đọc chữ; YOLO chỉ biết các lớp trên, không mô tả bối cảnh/màu sắc.
Ảnh không có chữ, vật thể hoặc mô tả VLM đủ dùng → trả lời mẫu, không gọi model để tránh đoán.
Ảnh Messenger chỉ tải từ `*.fbcdn.net`, `*.fbsbx.com`; ảnh > 8 MB hoặc không phải JPEG/PNG/WEBP bị từ chối.

```bash
uv run botctl vision analyze anh.jpg            # xem chữ + vật thể
uv run botctl vision ask anh.jpg "câu hỏi"       # hỏi thử trên máy
# train lại
uv run python -m training.vision.prepare_yolo_v2
uv run python -m training.vision.oversample --data data/processed/yolo-edu-ent-v2
uv run python -m training.vision.train_yolo --data data/processed/yolo-edu-ent-v2 --yaml data_oversampled.yaml --name edu-ent-v3 --epochs 100
uv run python -m evaluation.scripts.compare_yolo --data data/processed/yolo-edu-ent-v2 --model <cũ>.pt --model <mới>.pt --conf 0.6
```

**Rollback YOLO**: đặt `VISION_CUSTOM_WEIGHTS` tới bản trọng số cũ đã lưu, rồi chạy `./scripts/run_local.sh stop` và `./scripts/run_local.sh start`. Bỏ biến này để chỉ dùng COCO; `VISION_ENABLED=false` để tắt ảnh. Restart ngắt tạm thời dịch vụ.

Phụ thuộc Ultralytics có điều khoản giấy phép riêng; xem LICENSE đi kèm phiên bản đã cài trước khi phân phối/triển khai.

## Tra cứu web có nguồn

Bot có thể tự tra web cho câu hỏi cần thông tin mới hoặc khi kho YAML/pgvector không đủ. Khi có key, tích hợp dùng
Brave LLM Context: chỉ gửi query đã che PII, nhận đoạn nội dung + URL, đưa vào cùng khối `Nguồn tham khảo`, rồi lớp
kiểm tra đầu ra xóa mọi link model tự viết ngoài danh sách kết quả. Nội dung web là dữ liệu không tin cậy và không
được phép điều khiển prompt/công cụ.

```bash
# Tạo key tại Brave Search API, không dán key vào chat/log
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MODE=auto       # auto | always
BRAVE_SEARCH_API_KEY=<secret>
```

`auto` tra khi câu hỏi có tính cập nhật hoặc kho nội bộ không có kết quả; nghị luận văn học và nghị luận xã hội luôn
tra web. Câu phân tích tác phẩm có tên cụ thể tạo truy vấn tập trung theo tên tác phẩm để tránh từ chối chỉ vì kho
nội bộ chưa có bài đó. Mặc định mỗi lượt lấy tối đa 8 kết quả web, tổng hợp cả nguồn chính thống lẫn các góc nhìn từ
báo, blog/bài mẫu nếu công cụ tìm thấy, và công khai toàn bộ URL đã đưa vào lượt trả lời. Nguồn yếu không bị giấu,
nhưng chỉ được dùng như góc nhìn để so sánh/phản biện; không được nâng thành dữ kiện chắc chắn. `always`
tra mọi câu hỏi kiến thức. Nếu API
lỗi, bot vẫn dùng kho nội bộ và gắn cờ `web_search_failed`; không giả vờ đã tra cứu. Tra web không làm model tự học
trọng số. Câu trả lời tốt/xấu vẫn phải được quản trị viên review trước khi đưa vào dataset huấn luyện.
Không có `BRAVE_SEARCH_API_KEY`, `WEB_SEARCH_DDGS_FALLBACK=true` dùng DDGS metasearch và kết hợp
`WEB_SEARCH_WIKIPEDIA_FALLBACK=true` để lấy thêm trang bách khoa sát tiêu đề; mọi kết quả vẫn giữ URL. Wikipedia
không thay thế nguồn chính thức cho tin mới, luật, giá, lịch hoặc số liệu đang thay đổi;
các câu đó cần Brave và nên đối chiếu nguồn gốc. Đây là truy xuất tại lúc hỏi nên nội dung mới không phải đợi một đợt
train theo giờ/ngày.

## Ollama (model 4-bit, dự phòng)

`serving/ollama/Modelfile` (Qwen3-4B-Instruct-2507 Q4_K_M, `num_ctx 8192`). `run_local.sh` tự chạy Ollama nếu có
`~/.local/ollama/bin/ollama`. Khi model server lỗi, bot tự chuyển sang Ollama (`LLM_FALLBACK_*`). Ollama không chạy
được adapter VeRA nên chỉ dùng dự phòng (so sánh chất lượng/độ trễ/VRAM: docs/progress.md).

Để thiết lập fallback sau khi cài Ollama, chạy Ollama server trước rồi tạo model:

```bash
ollama create fanpage-qwen3-4b-q4 -f serving/ollama/Modelfile
ollama list
```

Đặt `LLM_FALLBACK_BASE_URL=http://127.0.0.1:11434/v1` và `LLM_FALLBACK_MODEL=fanpage-qwen3-4b-q4`.
Nếu Ollama đã chạy bằng systemd thì để dịch vụ đó quản lý. Nếu dùng bản cài riêng ở đường dẫn khác,
đặt `OLLAMA_BIN=/duong/dan/ollama` trước khi gọi script start. Fallback này chỉ thay LLM, không thay embedding/OCR/YOLO.

## Docker cho API/worker

Luồng mặc định đã kiểm thử là PostgreSQL trong Docker, còn model/API/worker trên host.
Profile `app` cung cấp API và worker trong container; **model GPU vẫn chạy trên host**.
Profile này chưa được xác nhận end-to-end trong lần xuất bản đầu tiên.

Dừng API/worker đang chiếm cổng trước khi chuyển. Model server cần nghe ở địa chỉ container truy cập được:

```bash
uv run --no-sync python -m serving.server --host 0.0.0.0
```

Giữ `MODEL_SERVER_API_KEY` mạnh và chỉ cho mạng Docker tin cậy truy cập cổng 8100. Trong `.env` dành cho profile này đặt:

```dotenv
LLM_BASE_URL_FOR_DOCKER=http://host.docker.internal:8100/v1
EMBEDDING_BASE_URL=http://host.docker.internal:8100/v1
# Chỉ đặt khi dùng fallback; Ollama cũng phải cho phép kết nối từ container:
# LLM_FALLBACK_BASE_URL=http://host.docker.internal:11434/v1
```

```bash
docker compose --profile app up -d --build
docker compose --profile app ps
docker compose --profile app logs --tail=100 api worker migrate
```

Compose thay `DATABASE_URL` bằng hostname `postgres`; localhost trong container không phải host.
Endpoint ảnh dùng `LLM_BASE_URL` trong cấu hình hiện tại. Nếu model ở máy khác, đặt cả URL LLM và embedding tới máy đó.
Dockerfile không cài extra `ml`/`data`; train và lệnh phân tích ảnh trực tiếp cần chạy trên host có extra `ml`.

## Dataset Kaggle

Không cần Kaggle để chạy bot hoặc train SFT từ YAML có sẵn. Cần Kaggle khi muốn tải lại dataset ảnh hoặc OASST1.
Đặt credential riêng ở `~/.kaggle/kaggle.json`, `chmod 600` hoặc khai báo `KAGGLE_JSON_PATH` trong `.env`.
Không chép credential vào repo. Cài extra `data` như bước setup.

Ví dụ tải dữ liệu văn bản:

```bash
uv run --no-sync python -m training.data.kaggle_download \
  mkaur1141/openassistant-conversations-dataset-oasst1 --expect-license apache-2.0 --max-mb 400
uv run --no-sync python -m training.data.normalize_oasst --help
```

Để dựng lại dataset YOLO v2/v3, tải đủ sáu nguồn dưới đây trước khi chạy `prepare_yolo_v2`.
Mỗi lệnh kiểm tra giấy phép metadata; nếu không khớp, dừng để kiểm tra thay vì bỏ qua.
Giới hạn dung lượng là ngân sách tải, không phải dung lượng chính xác của dataset.

```bash
uv run --no-sync python -m training.data.kaggle_download abdullahsami10/stationary-dataset --expect-license cc-by-4.0 --kind vision --max-mb 100
uv run --no-sync python -m training.data.kaggle_download aryakrisnaputra/objects-in-the-classroom --expect-license mit --kind vision --max-mb 500
uv run --no-sync python -m training.data.kaggle_download josephnelson/six-sided-dice-images-and-bounding-boxes --expect-license cc0-1.0 --kind vision --max-mb 700
uv run --no-sync python -m training.data.kaggle_download sagarnildass/guitar-detection-dataset --expect-license cc0-1.0 --kind vision --max-mb 300
uv run --no-sync python -m training.data.kaggle_download cookiemonsteryum/chess-piece-object-detection --expect-license cc0-1.0 --kind vision --max-mb 200
uv run --no-sync python -m training.data.kaggle_download andy8744/playing-cards-object-detection-dataset --expect-license cc0-1.0 --kind vision --max-mb 1500
```

Manifest tải nằm trong `data/raw/kaggle/`; pipeline xử lý đặt dữ liệu trong `data/processed/`, không commit.
Script chuẩn bị ảnh có đường dẫn tới layout/version dataset cụ thể; phiên bản upstream khác có thể cần cập nhật đường dẫn.
Nguồn guitar có khác biệt metadata Kaggle/Roboflow được ghi trong manifest của script; giữ thông tin nguồn khi sử dụng.

### Khôi phục adapter đã train

Huấn luyện SFT bằng YAML trong repo không cần dataset tải riêng. Các báo cáo trong `docs/progress.md` là kết quả
lịch sử, không bảo đảm model mới train có cùng điểm số. Sau train, dùng ID và đường dẫn được lệnh in ra để register,
attach-eval và promote như phần fine-tune. Sau promote phải khởi động lại model server để nạp adapter mới.

Nếu chuyển máy, sao chép thư mục adapter đầy đủ (trọng số, config, tokenizer nếu có, `training_meta.json`) và báo cáo
đánh giá tương ứng qua kênh riêng. Chạy register/attach-eval/promote lại trên máy đích để tránh đường dẫn registry cũ.
Custom YOLO cần file `.pt` và metadata `.json` cùng tên nếu có; giữ cả thư mục run làm bản sao lưu.

## Vận hành, sao lưu và xử lý lỗi

### Kiểm thử

```bash
docker compose up -d --wait postgres
uv run --no-sync pytest -q
uv run --no-sync ruff check .
npm --prefix web/widget ci
npm --prefix web/widget test
npm --prefix web/widget run build
```

Kết quả kiểm tra ngày 24/09/2026: **246 test Python passed**, lint sạch, **3 test widget passed**, typecheck/build thành công.
Test DB dùng `chatbot_test` riêng và làm sạch dữ liệu trong DB đó; không trỏ `TEST_DATABASE_URL` tới DB thật.
Nếu PostgreSQL không truy cập được, test DB có thể skip; số test passed khi đó không chứng minh luồng DB đã chạy.
Các HTTP request bên ngoài trong test được mock; kết quả này không thay thế test Messenger với tài khoản thật.

### Lệnh quản trị

```bash
uv run --no-sync botctl --help
uv run --no-sync botctl meta check
uv run --no-sync botctl conversations list
uv run --no-sync botctl conversations show <conversation-id>
uv run --no-sync botctl conversations handoff <conversation-id> on
uv run --no-sync botctl conversations handoff <conversation-id> off
uv run --no-sync botctl review list
uv run --no-sync botctl review rate <turn-id> good
uv run --no-sync botctl review correct <turn-id> --text "Câu trả lời đã sửa"
uv run --no-sync botctl candidates list
uv run --no-sync botctl candidates approve <candidate-id> --confirm-usage-rights
uv run --no-sync botctl candidates reject <candidate-id>
uv run --no-sync botctl data export --kind sft
uv run --no-sync botctl knowledge search "vera khác gì lora"
uv run --no-sync botctl model status
```

`--confirm-usage-rights` chỉ dùng khi đã kiểm tra quyền sử dụng mẫu. Export không tự train hay tự promote model.
`conversations delete <id> --yes` xóa dữ liệu hội thoại; `docs delete --help` hướng dẫn xóa tài liệu đã ingest.
`AUTO_HANDOFF_ON_HUMAN_REPLY=true` dừng bot khi quản trị viên trả lời; bật lại bằng handoff off.

### API và quan sát

| Endpoint | Mục đích |
|---|---|
| `GET /health`, `GET /ready` | sức khỏe tiến trình và các điều kiện nhận webhook |
| `GET /webhook`, `POST /webhook` | xác minh Meta và nhận event có chữ ký |
| `GET /web/demo`, `GET /web/widget.js` | demo và bundle nhúng |
| `POST /web/session` | tạo session ID và token cho web |
| `POST /web/messages`, `GET /web/messages`, `POST /web/images` | chat/poll/ảnh với session token |
| `/admin/*` | quản trị; dùng `Authorization: Bearer <ADMIN_API_KEY>` |
| Model `:8100/v1/chat/completions`, `/v1/embeddings`, `/v1/vision/analyze` | endpoint nội bộ có model API key |

Schema request cụ thể nằm ở `app/api/` và `serving/server.py`.
Log local ở `runs/logs/`; `.env` mặc định tắt nội dung log. Langfuse là tùy chọn: đặt `LANGFUSE_ENABLED=true`,
`LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` của project muốn dùng.
`LANGFUSE_CAPTURE_CONTENT=false` giữ chế độ metadata; xem phần triển khai hiện có trong `app/observability/`.

### Sao lưu và chuyển máy

```bash
mkdir -p runs/backup
chmod 700 runs/backup
# Dump bao gồm hội thoại, feedback, tài liệu/embedding và checkpoint; giữ riêng tư.
(umask 077; docker compose exec -T postgres pg_dump -U chatbot -d chatbot -Fc > runs/backup/chatbot.dump)
```

Lưu thêm `.env`, artifact model, registry và báo cáo đánh giá ở nơi riêng có kiểm soát truy cập.
Kiểm thử restore trên môi trường riêng trước khi thay DB đang dùng. Ví dụ restore vào DB mới, trống:

```bash
docker compose exec -T postgres createdb -U chatbot chatbot_restore
docker compose exec -T postgres pg_restore -U chatbot -d chatbot_restore --no-owner < runs/backup/chatbot.dump
```

Đổi `DATABASE_URL` sang DB đã khôi phục trên môi trường đích rồi chạy migrate. Với profile Docker cần chỉnh cả
`DATABASE_URL` override trong Compose. Đừng chạy `docker compose down -v` nếu cần giữ dữ liệu: `-v` xóa volume.
`run_local.sh stop` giữ PostgreSQL và dữ liệu; sau khởi động lại máy cần chạy start và cập nhật tunnel webhook.

### Lỗi thường gặp

| Triệu chứng | Kiểm tra/cách xử lý |
|---|---|
| `ModuleNotFoundError: torch`/`easyocr` | `uv sync --frozen --extra ml --extra data`, chạy lại |
| CUDA không có hoặc OOM | kiểm tra `nvidia-smi`, torch CUDA; dừng tiến trình GPU khác hoặc dừng model trước khi train |
| Model khởi động lâu | tải lần đầu cần mạng; xem `runs/logs/model.log`; thử chạy server trực tiếp |
| Không cần ảnh/embedding | `VISION_ENABLED=false`, `RAG_DOCS_ENABLED=false`; chạy server tay với `--no-vision --no-embeddings` để bỏ nạp |
| DB authentication failed | mật khẩu URL phải khớp mật khẩu đã khởi tạo volume; đổi `.env` không tự đổi mật khẩu trong DB cũ |
| `/ready` báo pending/missing checkpoint | chạy `uv run --no-sync botctl db migrate` |
| API lên nhưng không trả lời | kiểm tra heartbeat, `runs/logs/worker.log`, model health, handoff và send mode |
| Meta verify thất bại | URL HTTPS phải truy cập được; verify token phải khớp; kiểm tra `/health` qua tunnel |
| Webhook có tin nhưng bot không gửi | kiểm tra Page token, quyền app/tài khoản, PSID allowlist, handoff và thời hạn trả lời |
| `page_subscription` lỗi | token thiếu quyền đăng ký Page: cấu hình subscriptions trong Meta Dashboard |
| Widget không kết nối từ website khác | kiểm tra `WEB_ALLOWED_ORIGINS`, HTTPS, script URL và API/worker |
| Model/embedding không tới được từ Docker | localhost trong container là container; dùng host gateway và bind model phù hợp |
| Tìm tài liệu không có kết quả | kiểm tra đã ingest, visibility, embedding model và `DOC_MIN_SIMILARITY` qua lệnh docs search |
| Repo clone không có VeRA/YOLO custom | artifact bị loại khỏi Git; train lại hoặc khôi phục từ bản sao lưu riêng |

### Giới hạn và dữ liệu không đưa lên GitHub

`.gitignore` loại `.env`, credential, log, DB dump trong `runs/`, model weights, dữ liệu thô/xử lý/tài liệu và kết quả
đánh giá chứa hội thoại. Repo chứa mã nguồn, mẫu cấu hình, widget bundle, dữ liệu tổng hợp và metadata nguồn dataset.
Không có giấy phép riêng được tự gán cho mã nguồn dự án; kiểm tra điều khoản phụ thuộc/model/dataset trước khi phân phối.
Báo cáo môi trường là ghi nhận lịch sử, không phải trạng thái hiện tại của token hay dịch vụ.

## Cấu trúc mã nguồn

```text
app/
  api/             Webhook, web chat, admin, health
  conversation/    LangGraph, memory, truy xuất YAML, kiểm tra đầu ra
  messenger/       Xác minh, gọi Graph API, chính sách gửi
  rag/             Đọc/chia tài liệu, embedding, tìm kiếm pgvector
  storage/         SQLAlchemy, repository, schema
  workers/         Hàng đợi bền vững, xử lý lượt, housekeeping
  observability/   Log, che dữ liệu nhạy cảm, Langfuse
  web/             Phiên web và bundle widget
serving/           Qwen, embedding, OCR/YOLO, registry adapter, Ollama
training/          SFT LoRA/VeRA, pipeline Kaggle, train YOLO
evaluation/        Dataset tổng hợp và script so sánh
tests/             Unit/DB/flow tests với HTTP mock
web/widget/        Nguồn TypeScript, test, lockfile npm
config/            Thông tin Page và kho kiến thức YAML
migrations/        Alembic, pgvector
scripts/           Start/stop, tải model, benchmark/replay
docs/              Tiến độ, môi trường, hiệu năng, riêng tư
```
