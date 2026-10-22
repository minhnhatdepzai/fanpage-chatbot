# Đóng góp cho fanpage-chatbot

Cảm ơn bạn muốn cải thiện dự án. Backend dùng FastAPI, PostgreSQL/pgvector và
LangGraph; `web/widget/` là widget TypeScript; các trang StudyScope nằm trong
`app/web/static/`. Model server, train và đánh giá nằm trong `serving/`,
`training/` và `evaluation/`.

## Thiết lập môi trường

Cần Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 24, Docker Engine và
Docker Compose. GPU chỉ cần khi chạy model local hoặc train.

```bash
git clone https://github.com/minhnhatdepzai/fanpage-chatbot.git
cd fanpage-chatbot
cp .env.example .env
uv sync --frozen
cd web/widget && npm ci && cd ../..
docker compose up -d --wait postgres
```

Điền các secret cục bộ trong `.env`; không đưa `.env`, token, hội thoại, tài
liệu riêng, model weights hoặc dữ liệu train chưa rõ giấy phép vào commit.

## Quy trình sửa mã

1. Tạo branch từ `main`, mỗi pull request tập trung vào một thay đổi.
2. Thêm test hồi quy khi sửa logic xử lý tin, bộ giải, RAG hoặc migration.
3. Chạy `make check`. Test có marker `db` tự dùng database `chatbot_test` trên
   PostgreSQL local; các HTTP call tới Meta đều bị mock.
4. Khi sửa widget, commit lại `app/web/static/widget.js` do `npm run build` sinh ra.
5. Khi sửa container, chạy `make container-smoke`.
6. Pull request cần nêu hành vi đổi, migration/dữ liệu bị ảnh hưởng và lệnh test
   đã chạy. Mọi tuyên bố về độ chính xác, chất lượng model hoặc độ trễ cần kèm
   tập đánh giá và số đo có thể lặp lại.

CI chạy Ruff, toàn bộ test Python với PostgreSQL/pgvector, test và
build widget, kiểm tra Compose, build image và smoke test API/Toán. Pull request
phải qua CI trước khi merge.

## Container và model

Image ứng dụng chỉ chứa API, worker và migration; nó không chứa model weights,
Torch/CUDA hay dữ liệu riêng. Profile `app` kết nối model server đang chạy trên
host qua `host.docker.internal`. Xem phần Docker trong `README.md` để chạy image
local hoặc image đã phát hành trên GHCR.

Mã nguồn dùng giấy phép MIT. Dữ liệu, tài liệu, giọng nói, checkpoint và model
bên ngoài vẫn tuân theo giấy phép và quyền sử dụng riêng của từng nguồn.
