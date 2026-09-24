import pytest

from app.conversation.mathcalc import answer_math_question


@pytest.mark.parametrize(
    ("q", "expect"),
    [
        ("50 nhân 50 bằng mấy", "50 × 50 = 2.500."),
        ("50x50=?", "50 × 50 = 2.500."),
        ("tính 12,5 + 7", "12,5 + 7 = 19,5."),
        ("100 chia 4 bằng bao nhiêu", "100 : 4 = 25."),
        ("(3 + 4) x 5 là bao nhiêu", "(3 + 4) × 5 = 35."),
        ("2^10 = ?", "2 ^ 10 = 1.024."),
        ("1.000.000 trừ 1 bằng mấy", "1.000.000 − 1 = 999.999."),
        ("10 chia 3 bằng mấy", "10 : 3 = 3,333333 (làm tròn 6 chữ số thập phân)."),
        ("5 chia 0 bằng mấy", "Phép tính có chia cho 0 nên không có kết quả."),
    ],
)
def test_arithmetic_is_computed(q, expect):
    assert answer_math_question(q) == expect


@pytest.mark.parametrize(
    "q",
    [
        "2x + 3 = 7 thì x bằng mấy",
        "50 nhân 50",  # không hỏi kết quả
        "năm 2026 có bao nhiêu ngày",
        "3 quả táo cộng 2 quả táo bằng mấy",
        "9^999999 bằng mấy",
        "hôm nay là thứ mấy",
    ],
)
def test_non_arithmetic_is_left_to_normal_flow(q):
    assert answer_math_question(q) is None
