# StudyScope: gia sư đa môn, Toán có kiểm chứng

## Kiểm tra kho đề và bản sửa 2026-10-06

Đối chiếu chính 24 đề đang hiển thị trong `study-bank.js`, không dùng một bộ ví dụ thay thế. Bản kiểm tra ban đầu
phát hiện 12 đề chưa được giải, cùng hai đáp án bị thiếu thao tác/yêu cầu phụ (truyện cho mượn và bút cần thêm).
`school.py` bổ sung các quan hệ phổ thông có dữ kiện tường minh; câu hỏi nhiều đại lượng trả đầy đủ trong `values`.
Kho đề hiện có đáp án chuẩn cho cả 24 đề, kiểm tra cả đề gốc lẫn câu yêu cầu giải thích đính kèm, và 240 biến thể
số liệu đối chiếu với phép cộng trực tiếp hoặc xác suất lấy tuần tự độc lập. Toàn bộ suite: **453 passed**.

`math-scenes.js` dựng hàng ghế thật, nhóm vật, tam giác đúng tỉ lệ, đường cao, bể nước theo kích thước,
biểu đồ trung bình, đổi đơn vị bản đồ, vùng tích phân và trục số nhiều biến đổi. Bài hội trường vẽ đủ 480 ghế
trên 15 hàng; hình thay đổi theo bước bên trái, có chọn hàng, tạm dừng, phát lại và phóng lớn. Khi giới hạn số
vật hiển thị, hình ghi rõ là minh họa một phần. Hình chữ nhật và đoạn thẳng tọa độ dùng thang đo đúng tỉ lệ.
Nhánh `math_review` chỉ có sơ đồ lập luận, nhãn tham khảo; không giả làm hình học hay kiểm chứng ký hiệu.

HTML trả `Cache-Control: no-store`, URL CSS/JS gắn hash nội dung để tránh renderer cũ không hiểu kiểu hình mới.
Bộ kiểm tra Chromium thật `scripts/check_math_web.py` chạy 24 đề qua API và renderer, kiểm tra chọn bước có hình
nhìn thấy, rồi bấm nút Giải thích thật, phát lại/tạm dừng/tiếp tục/phóng lớn và kiểm tra viewport 390px không tràn.
Ảnh và báo cáo nằm tại `artifacts/math-quality/`, không phải chứng nhận đúng với mọi đề ngoài kho.

Đã thực sự huấn luyện **bộ phân loại định tuyến nhỏ**, không phải fine-tune Qwen:
`training/scripts/train_math_router.py` dùng 1.200 mẫu tổng hợp tự biên soạn, 240 mẫu đánh giá với cách diễn đạt
khác tập train; cập nhật optimizer 240 bước, lưu checkpoint và kiểm tra nạp lại. Accuracy giữ lại 87,5%, thấp
hơn ngưỡng phát hành 95%, nên checkpoint ứng viên **không được đưa vào runtime**. Báo cáo, dữ liệu và hash có
tại `artifacts/math-quality/router-training-report.json`. Thay dữ kiện trên cùng một mẫu câu không được quảng bá
thành đa dạng ngôn ngữ hay bằng chứng model có thể giải mọi bài toán. Quy trình theo skill Langfuse áp dụng
ca lỗi thực tế → dữ liệu hồi quy → tiêu chí phát hành; không tự động tải nội dung người dùng vào dataset live.

Các trang chạy tại `GET /web/chat`, `/web/math`, `/web/literature` và `/web/english`; API Toán có cấu trúc chạy tại
`POST /web/math/solve`. Trang Tiếng Anh nhúng nguyên bản build Task 1 Coach do người dùng cung cấp; notice và nguồn
đi kèm nằm dưới `app/web/static/task1-coach-pages/`. Ảnh đề toán được gửi tới
`POST /web/math/read-image`: OCR/VLM chỉ chép đề, sau đó cùng bộ giải xác định mới tính và thế ngược đáp án.

Trang không ép mọi câu thành toán. `POST /web/math/route` nhận biết đề toán trước: bài thuộc miền kiểm chứng trả cấu
trúc lời giải/SVG; Ngữ văn và Tiếng Anh được gắn nhãn gia sư tương ứng trước khi gửi qua `/web/messages`; lời chào và
câu hỏi kiến thức đi vào chatbot đa lĩnh vực. Tất cả giữ session, lịch sử, RAG, web search và nguồn như widget. Đề
toán ngoài miền xác định được chuyển sang nhánh kiểm định độc lập `math_review`.

Giao diện có URL và ngân hàng đề riêng cho Toán, Ngữ văn; chatbot tổng quát nằm ở trang riêng. Việc vào một phòng
không thay đổi nguyên tắc an toàn: bộ định tuyến vẫn xác định môn từ chính nội dung người dùng nhập. Yêu cầu như
`Vẽ đồ thị y = 4x + 9` được bộ giải ký hiệu lấy mẫu chính xác; trình duyệt dựng hệ trục, lưới, đường hàm và các điểm
bằng SVG với thang trục tự co giãn. Khi nhận lời giải Toán có cấu trúc, giao diện trình bày timeline các bước ở bên
trái và SVG chuyển động ở bên phải; từng bước lần lượt xuất hiện, có điều khiển tạm dừng/phát lại và tự tắt chuyển
động khi trình duyệt bật `prefers-reduced-motion`.

## Phòng đọc neural cho Ngữ văn và Tiếng Anh

Câu trả lời được định tuyến là `writing` hoặc `english` có thêm player dạng audio lesson/podcast. Backend
`POST /web/tts/synthesize` sử dụng `edge-tts` 7.2.x để tạo MP3 trong bộ nhớ và không ghi nội dung/âm thanh xuống
đĩa. Endpoint yêu cầu session web hợp lệ, giới hạn ký tự, kích thước MP3, thời gian xử lý và tần suất theo IP.

Các giọng được whitelist thay vì nhận tên model tùy ý:

- Việt Nam: `vi-VN-HoaiMyNeural`, `vi-VN-NamMinhNeural`.
- Mỹ: `en-US-JennyNeural`, `en-US-GuyNeural`.
- Anh: `en-GB-SoniaNeural`, `en-GB-RyanNeural`.

Người học chọn preset Giáo viên, Podcast, Kể chuyện hoặc Chậm & rõ, rồi chỉnh thêm tốc độ, cao độ và âm lượng.
Phần “nhấn nhá” dùng prosody mà Edge hỗ trợ; không gửi SSML tùy ý. Trong lúc phát, giao diện có waveform, phụ đề
theo câu, seek, tạm dừng, dừng và tải MP3. Khi neural TTS không truy cập được, Web Speech API của trình duyệt là
fallback và UI nói rõ đang dùng giọng hệ thống. Neural TTS là dịch vụ trực tuyến, vì vậy cần kết nối Internet.

Prompt Văn/Anh mặc định tạo phần giải thích có mạch đủ cho một bài nghe khoảng 3–6 phút khi người dùng không yêu cầu
ngắn. Yêu cầu “chỉ đáp án/chỉ đưa tác phẩm” vẫn được ưu tiên, tránh phá hợp đồng đầu ra của đề bài.

## Nguyên tắc chính xác

- Đầu vào không được đưa thẳng vào `eval`, `sympify` hay mã Python. Bộ phân tích AST chỉ cho phép số, `x/y/z`,
  `pi`, `sqrt`, `sin/cos/tan`, `log/exp/abs`, tổ hợp/chỉnh hợp, ngoặc và các toán tử đã liệt kê.
- Phép tính dùng số hữu tỉ/ký hiệu chính xác. Nghiệm phương trình và hệ phương trình đều được thế ngược trước khi
  trả về `status=verified`.
- LLM/VLM không được quyền ghi đè đáp án đã kiểm chứng. Câu nằm ngoài miền hỗ trợ sẽ trả lỗi an toàn thay vì đoán.
- Đề Toán ngoài miền bộ giải xác định được gắn `math_review`: model tạo bản nháp, sau đó một lượt kiểm định độc lập
  giải lại và kiểm tra điều kiện/nghiệm ngoại lai. Kiểm định lỗi, trả sai giao thức hoặc không đủ căn cứ thì hệ thống
  bỏ bản nháp và từ chối chốt đáp số. Nhánh này không được dùng nhãn `status=verified`.
- Hình minh họa là SVG sinh từ chính dữ liệu nghiệm: trục số, nhóm vật, phân số, cân bằng đại số hoặc đồ thị.

Phạm vi bản hiện tại: biểu thức số học, căn và hàm sơ cấp; phương trình đa thức một ẩn đến bậc 4; hệ tuyến tính 2-3
phương trình; phương trình mũ đưa chính xác được về cùng cơ số; bất phương trình một ẩn; đạo hàm; giới hạn; tích
phân xác định; tổ hợp/chỉnh hợp. Cấp lớp 1-12 là nhãn sư phạm; không phải tuyên bố rằng mọi bài toán trên thế giới
đều đã được bao phủ.

## Tài sản fine-tuning được nhập

Chạy:

```bash
uv run python scripts/import_math_lab_assets.py
```

Script nhập hai cây `training/math_lab` và `training/arithmetic_quiz` từ EduVisionAI, ngoại trừ môi trường ảo và cache,
vào `artifacts/math_lab_import/`. Mỗi tệp có SHA-256 trong `import-manifest.json`; `artifacts/` được gitignore để
không đẩy trọng số/dataset lớn lên Git.

Chính sách runtime dựa trên bằng chứng đi kèm checkpoint:

- `multilingual-visual-selector-lora`: 1.065 mẫu test giữ lại, accuracy 1.0; chỉ được chọn loại hình minh họa.
- `qwen3-0.6b-gsm8k-lora-100step`: base và adapter cùng 14/50 trên GSM8K test; `accepted_for_runtime=false`. Artifact
  vẫn được bảo tồn để kiểm toán nhưng tuyệt đối không được quảng bá hoặc nạp vào model Qwen3-4B đang phục vụ.

## Kiến thức mới và nguồn

“Cập nhật mỗi giờ” được hiểu là tra cứu web tại thời điểm hỏi, không phải tự huấn luyện model mỗi giờ. Tra cứu bật
bằng `WEB_SEARCH_ENABLED=true`: Brave cung cấp web rộng khi có key; DDGS metasearch kết hợp MediaWiki là fallback
không-key cho kiến thức phổ thông và tác phẩm có tên cụ thể; mỗi kết quả giữ URL trong ngữ cảnh. Fine-tuning chỉ
chạy từ dataset có version/hash và phải qua tập đánh
giá tách biệt trước khi thăng hạng.

Nguồn chương trình tham chiếu hiện hiển thị trên trang:

- Việt Nam: Thông tư 32/2018/TT-BGDĐT trên Cơ sở dữ liệu quốc gia về văn bản pháp luật; khi cập nhật nội dung cần
  xét thêm các văn bản sửa đổi hiện hành, gồm Thông tư 17/2025/TT-BGDĐT.
- Quốc tế: Common Core Mathematics K-12. “Chương trình thế giới” không phải một chương trình duy nhất; Cambridge và
  IB cần được thêm thành profile riêng thay vì trộn chuẩn.

## Đánh giá trước khi mở rộng

Theo dõi tối thiểu: exact-answer, tỷ lệ thế ngược thành công, bất biến SVG, đầy đủ bước giải, nguồn hợp lệ/còn mới,
gắn cấp lớp/chương trình và tỷ lệ từ chối đúng. Langfuse chỉ ghi metadata khi
`LANGFUSE_CAPTURE_CONTENT=false`; không tạo dataset/evaluator live tự động nếu chưa duyệt rubric và dữ liệu riêng tư.
