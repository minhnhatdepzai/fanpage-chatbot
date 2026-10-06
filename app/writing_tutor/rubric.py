"""Rubric cục bộ cho bộ đánh giá Ngữ văn; chưa tự động đẩy lên Langfuse."""

from __future__ import annotations

WRITING_RUBRIC = {
    "version": "writing-rubric-v1",
    "scale": "0-4",
    "criteria": [
        {"id": "task_adherence", "label": "Đúng đề, thể loại và độ dài"},
        {"id": "structure", "label": "Bố cục và liên kết"},
        {"id": "development", "label": "Luận điểm, phân tích và chiều sâu"},
        {"id": "evidence_integrity", "label": "Dẫn chứng đúng, không bịa trích dẫn"},
        {"id": "language_grade_fit", "label": "Diễn đạt phù hợp cấp lớp"},
        {"id": "learning_value", "label": "Giúp học sinh hiểu và tự chỉnh sửa"},
        {"id": "form_fidelity", "label": "Đúng đặc trưng hình thức/truyền thống đã yêu cầu"},
        {"id": "preference_alignment", "label": "Bám ngôn ngữ, giọng điệu và sở thích người dùng"},
        {"id": "originality", "label": "Tác phẩm mới, không sao chép hay giả trích dẫn"},
    ],
    "hard_fail": [
        "Bịa câu thơ, câu văn, tác giả hoặc chi tiết tác phẩm",
        "Chép toàn văn tác phẩm còn bản quyền không do người dùng cung cấp",
        "Khẳng định số liệu/sự kiện không có nguồn như sự thật",
        "Dùng khuôn mẫu văn hóa thay cho đặc trưng hình thức hoặc ngữ cảnh cụ thể",
    ],
}
