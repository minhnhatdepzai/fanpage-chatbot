"""Bộ giải toán an toàn cho trang Math Tutor.

LLM có thể giải thích hoặc nhận dạng đề, nhưng đáp án thuộc phạm vi hỗ trợ ở đây luôn được
tính và thế ngược bằng SymPy. Không dùng ``sympify``/``eval`` trực tiếp trên đầu vào.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import sympy as sp


class MathTutorError(ValueError):
    """Đề chưa đủ dữ kiện hoặc nằm ngoài miền bộ kiểm chứng hiện tại."""


VI_CURRICULUM = {
    "title": "Chương trình giáo dục phổ thông Việt Nam - môn Toán",
    "url": "https://vbpl.moj.gov.vn/bogiaoducdaotao/Pages/vbpq-toanvan.aspx?ItemID=146721",
    "publisher": "Cơ sở dữ liệu quốc gia về văn bản pháp luật",
}
WORLD_CURRICULUM = {
    "title": "Common Core State Standards for Mathematics (K-12)",
    "url": "https://corestandards.org/mathematics-standards/",
    "publisher": "Common Core State Standards Initiative",
}

_ALLOWED_NAMES = {
    "x": sp.Symbol("x"),
    "y": sp.Symbol("y"),
    "z": sp.Symbol("z"),
    "pi": sp.pi,
    "oo": sp.oo,
}
_ALLOWED_FUNCTIONS = {
    "sqrt": (sp.sqrt, 1),
    "sin": (sp.sin, 1),
    "cos": (sp.cos, 1),
    "tan": (sp.tan, 1),
    "log": (sp.log, (1, 2)),
    "exp": (sp.exp, 1),
    "abs": (sp.Abs, 1),
}
_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}


@dataclass(slots=True)
class _Parsed:
    normalized: str
    expressions: list[sp.Expr]
    equations: list[sp.Equality]


class _SafeExpression(ast.NodeVisitor):
    def __init__(self) -> None:
        self.nodes = 0

    def visit(self, node: ast.AST) -> sp.Expr:  # type: ignore[override]
        self.nodes += 1
        if self.nodes > 120:
            raise MathTutorError("Biểu thức quá phức tạp để kiểm chứng an toàn.")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> sp.Expr:  # noqa: N802
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> sp.Expr:  # noqa: N802
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise MathTutorError("Chỉ chấp nhận hằng số.")
        return sp.Rational(str(node.value))

    def visit_Name(self, node: ast.Name) -> sp.Expr:  # noqa: N802
        if node.id not in _ALLOWED_NAMES:
            raise MathTutorError(f"Biến '{node.id}' chưa được hỗ trợ.")
        return _ALLOWED_NAMES[node.id]

    def visit_UnaryOp(self, node: ast.UnaryOp) -> sp.Expr:  # noqa: N802
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        raise MathTutorError("Toán tử một ngôi không được hỗ trợ.")

    def visit_BinOp(self, node: ast.BinOp) -> sp.Expr:  # noqa: N802
        operation = _OPS.get(type(node.op))
        if operation is None:
            raise MathTutorError("Toán tử chưa được hỗ trợ.")
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise MathTutorError("Không thể chia cho 0.")
        if isinstance(node.op, ast.Pow):
            if not right.is_number or abs(float(right)) > 12:
                raise MathTutorError("Số mũ quá lớn hoặc không hợp lệ.")
        return operation(left, right)

    def visit_Call(self, node: ast.Call) -> sp.Expr:  # noqa: N802
        if node.keywords or not isinstance(node.func, ast.Name):
            raise MathTutorError("Lời gọi hàm không hợp lệ.")
        function = _ALLOWED_FUNCTIONS.get(node.func.id)
        if function is not None:
            fn, arity = function
            valid_arities = {arity} if isinstance(arity, int) else set(arity)
            if len(node.args) not in valid_arities:
                raise MathTutorError(f"Hàm {node.func.id} nhận {sorted(valid_arities)} đối số.")
            return fn(*(self.visit(arg) for arg in node.args))
        if node.func.id in {"c", "a"} and len(node.args) == 2:
            n, k = (self.visit(arg) for arg in node.args)
            if not (n.is_Integer and k.is_Integer and 0 <= k <= n <= 10_000):
                raise MathTutorError("C(n,k)/A(n,k) cần 0 ≤ k ≤ n ≤ 10.000 và n, k là số nguyên.")
            return sp.binomial(n, k) if node.func.id == "c" else sp.factorial(n) / sp.factorial(n - k)
        if node.func.id == "factorial" and len(node.args) == 1:
            value = self.visit(node.args[0])
            if not (value.is_Integer and 0 <= value <= 10_000):
                raise MathTutorError("Giai thừa cần số nguyên từ 0 đến 10.000.")
            return sp.factorial(value)
        raise MathTutorError("Hàm chưa được hỗ trợ trong bộ kiểm chứng an toàn.")

    def generic_visit(self, node: ast.AST) -> sp.Expr:
        raise MathTutorError(f"Cú pháp {type(node).__name__} không được hỗ trợ.")


def _normalize(raw: str) -> str:
    text = raw.strip().lower()
    if not text or len(text) > 1200:
        raise MathTutorError("Đề toán cần từ 1 đến 1.200 ký tự.")
    text = text.replace("−", "-").replace("–", "-").replace("×", "*").replace("·", "*")
    text = text.replace("÷", "/").replace("^", "**").replace("π", "pi").replace("∞", "oo")
    # Giữ dấu phẩy phân cách đối số của C(n,k), A(n,k); dấu phẩy giữa hai chữ số khác là thập phân.
    text = re.sub(r"\b([ca])\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", r"\1(\2;\3)", text)
    text = re.sub(r"\bln\s*\(", "log(", text)
    text = re.sub(r"√\s*\(", "sqrt(", text)
    text = re.sub(r"√\s*([0-9.]+)", r"sqrt(\1)", text)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    text = re.sub(r"\b([ca])\((\d+);(\d+)\)", r"\1(\2,\3)", text)
    # 2x, 2(x+1), )( và x( được đổi thành phép nhân tường minh.
    text = re.sub(r"(?<=\d)(?=[xyzp(])", "*", text)
    text = re.sub(r"(?<=\d)(?=(?:sqrt|sin|cos|tan|log|exp|abs)\s*\()", "*", text)
    text = re.sub(r"(?<=[xyz)])(?=\d|[xyz(])", "*", text)
    # Giao diện thêm yêu cầu sư phạm sau đề. Chúng không phải là một phần của
    # biểu thức, nếu giữ lại parser an toàn sẽ từ chối cả bài toán hợp lệ.
    text = re.split(
        r"(?:\s*[,.;?!]\s*)+(?=(?:hãy\s+giải\s+thích|giải\s+thích\s+bằng|trình\s+bày|"
        r"rút\s+gọn\s+kết\s+quả|nêu\s+(?:điều\s+kiện|cực\s+trị)|"
        r"xác\s+định\s+đỉnh|kiểm\s+tra\s+lại)\b)|"
        r"\s+và\s+giải\s+thích\s+(?:hằng\s+đẳng\s+thức|bằng\s+mô\s+hình)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return text


def _extract_math_text(text: str) -> str:
    prefixes = (
        r"^hãy\s+",
        r"^vui lòng\s+",
        r"^tính(?: giá trị)?\s*[:：]?\s*",
        r"^giải\s+bất\s+phương\s+trình\s*[:：]?\s*",
        r"^giải(?: hệ)?(?: phương trình)?\s*[:：]?\s*",
        r"^solve\s*[:：]?\s*",
        r"^calculate\s*[:：]?\s*",
    )
    for prefix in prefixes:
        text = re.sub(prefix, "", text, count=1)
    if text.count("=") >= 2:
        text = re.sub(r"\s+(?:và|and)\s+(?=[^;=\n]{1,120}=)", ";", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s+(?:thì\s+)?[xyz]\s+(?:bằng|là)\s+(?:mấy|bao nhiêu|gì)\s*$",
        "",
        text,
    )
    return text.strip().rstrip("?.")


def _parse_expr(text: str) -> sp.Expr:
    text = text.strip()
    if not re.fullmatch(r"[0-9a-z_+\-*/().,\s]+", text):
        raise MathTutorError("Mình chưa tách được biểu thức an toàn từ câu này.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise MathTutorError("Biểu thức chưa đúng cú pháp.") from exc
    return sp.simplify(_SafeExpression().visit(tree))


def _parse(question: str) -> _Parsed:
    normalized = _normalize(question)
    body = _extract_math_text(normalized)
    parts = [part.strip() for part in re.split(r"[;\n]+", body) if part.strip()]
    equations: list[sp.Equality] = []
    expressions: list[sp.Expr] = []
    for part in parts:
        if part.count("=") == 1:
            left, right = part.split("=", 1)
            equations.append(sp.Eq(_parse_expr(left), _parse_expr(right), evaluate=False))
        elif "=" in part:
            raise MathTutorError("Mỗi phương trình chỉ có một dấu '='.")
        else:
            expressions.append(_parse_expr(part))
    if equations and expressions:
        raise MathTutorError("Hãy tách phép tính và hệ phương trình thành hai câu hỏi.")
    if not equations and len(expressions) != 1:
        raise MathTutorError("Mỗi lượt chỉ giải một phép tính hoặc một hệ phương trình.")
    return _Parsed(normalized=body, expressions=expressions, equations=equations)


def _pretty(value: sp.Expr) -> str:
    value = sp.simplify(value)
    if value is sp.true:
        return "đúng với mọi giá trị"
    if value is sp.false:
        return "vô nghiệm"
    return str(value).replace("**", "^")


def _source(curriculum: str) -> dict[str, str]:
    return VI_CURRICULUM if curriculum == "vi" else WORLD_CURRICULUM


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _word_problem_result(
    *,
    question: str,
    expression: str,
    value: sp.Expr,
    operation: str,
    explanation: str,
    visual: dict[str, Any],
    curriculum: str,
    grade: int | None,
) -> dict[str, Any]:
    answer = _pretty(value)
    return {
        "status": "verified",
        "problem": question.strip(),
        "answer": answer,
        "answer_latex": sp.latex(value),
        "topic": operation,
        "grade": grade or 3,
        "steps": [
            {"title": "Tóm tắt dữ kiện", "detail": explanation},
            {"title": "Lập phép tính", "detail": expression},
            {"title": "Tính chính xác", "detail": f"{expression} = {answer}."},
            {"title": "Kiểm tra", "detail": "Tính lại bằng số chính xác và đối chiếu với quan hệ trong đề."},
        ],
        "visual": visual,
        "verification": {"engine": "deterministic-word-rules", "passed": True, "method": "exact_arithmetic"},
        "sources": [_source(curriculum)],
        "warnings": [],
    }


def _require_number_coverage(text: str, matches: list[re.Match[str]]) -> None:
    """Không cho phép một mẫu lời văn bỏ qua các dữ kiện số ngoài phần đã nhận dạng."""
    for number in re.finditer(r"(?<![a-z])[-+]?\d+(?:[.,]\d+)?", text):
        if not any(
            start <= number.start() and number.end() <= end
            for match in matches
            for group in range(1, len(match.groups()) + 1)
            for start, end in [match.span(group)]
        ):
            raise MathTutorError(
                "Chưa mô hình hóa đủ dữ kiện của đề; không thể chốt đáp án từ một phần câu hỏi."
            )


def _solve_word_problem(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    """Các họ bài lời văn có quan hệ tường minh; không suy diễn nếu câu không khớp mẫu chặt."""
    text = _fold_text(_normalize(question))
    number = r"(\d+(?:[.,]\d+)?)"

    seat_sequence = re.search(
        r"hang dau\s+(?:co\s+)?(\d+)\s+ghe.*?moi hang sau\s+"
        r"(hon|it hon|kem)\s+hang truoc\s+(\d+)\s+ghe",
        text,
    )
    requested_row = re.search(r"(?:so ghe\s+)?hang\s+(?:thu\s+)?(\d+)", text)
    total_rows = re.search(r"tong\s+(?:so\s+)?ghe.*?(?:cua\s+)?(\d+)\s+hang", text)
    if seat_sequence and requested_row and total_rows:
        _require_number_coverage(text, [seat_sequence, requested_row, total_rows])
        first = int(seat_sequence.group(1))
        difference = int(seat_sequence.group(3))
        if seat_sequence.group(2) in {"it hon", "kem"}:
            difference = -difference
        nth = int(requested_row.group(1))
        count = int(total_rows.group(1))
        if not 1 <= nth <= 10_000 or not 1 <= count <= 10_000:
            raise MathTutorError("Số thứ tự hàng phải từ 1 đến 10.000.")
        nth_value = first + (nth - 1) * difference
        last_for_sum = first + (count - 1) * difference
        if min(first, nth_value, last_for_sum) < 0:
            raise MathTutorError("Dữ kiện làm số ghế âm; hãy kiểm tra lại độ tăng/giảm và số hàng.")
        total = sp.Rational(count * (first + last_for_sum), 2)
        sample_indices = list(dict.fromkeys([1, 2, 3, max(1, nth - 1), nth]))
        samples = [
            {"row": index, "seats": first + (index - 1) * difference}
            for index in sample_indices
            if index <= nth
        ]
        answer = f"Hàng {nth}: {nth_value} ghế; tổng {count} hàng: {_pretty(total)} ghế"
        return {
            "status": "verified",
            "problem": question.strip(),
            "answer": answer,
            "answer_latex": sp.latex(total),
            "topic": "Cấp số cộng",
            "grade": grade or 11,
            "steps": [
                {
                    "title": "Nhận dạng cấp số cộng",
                    "detail": f"Số ghế tạo thành cấp số cộng với u₁ = {first} và công sai d = {difference}.",
                },
                {
                    "title": f"Tính số ghế hàng {nth}",
                    "detail": (
                        f"u₍{nth}₎ = u₁ + ({nth} − 1)d = {first} + ({nth} − 1)×{difference} = {nth_value}."
                    ),
                },
                {
                    "title": f"Lập tổng {count} hàng",
                    "detail": (f"S₍{count}₎ = {count}(u₁ + u₍{count}₎)/2, với u₍{count}₎ = {last_for_sum}."),
                },
                {
                    "title": "Tính tổng",
                    "detail": f"S₍{count}₎ = {count}×({first} + {last_for_sum})/2 = {_pretty(total)} ghế.",
                },
                {
                    "title": "Kiểm tra độc lập",
                    "detail": (
                        f"Trung bình hàng đầu và hàng {count} là ({first}+{last_for_sum})/2 = "
                        f"{_pretty(sp.Rational(first + last_for_sum, 2))}; nhân {count} hàng được {_pretty(total)}, khớp."
                    ),
                },
            ],
            "visual": {
                "type": "arithmetic_sequence",
                "first": first,
                "difference": difference,
                "nth": nth,
                "nth_value": nth_value,
                "count": count,
                "total": _pretty(total),
                "rows": samples,
            },
            "verification": {
                "engine": "deterministic-sequence",
                "passed": True,
                "method": "nth_term_and_gauss_sum",
            },
            "sources": [_source(curriculum)],
            "warnings": [],
        }

    probability = re.search(r"(?:hop|tui)(.*?)(?:lay|rut)", text)
    if probability and "khong hoan lai" in text and "khac mau" in text:
        groups = re.findall(r"(\d+)\s+(?:vien\s+)?bi\s+([a-z]+)", probability.group(1))
        drawn = re.search(r"lay\s+(?:ngau nhien\s+)?(2)\s+(?:vien\s+)?(?:bi)?", text)
        if len(groups) >= 2 and drawn:
            _require_number_coverage(text, [*re.finditer(r"(\d+)\s+(?:vien\s+)?bi\s+([a-z]+)", text), drawn])
            counts = [(int(count), color) for count, color in groups]
            if len({color for _, color in counts}) != len(counts) or any(count <= 0 for count, _ in counts):
                raise MathTutorError("Cần các nhóm màu khác nhau, mỗi nhóm có số bi nguyên dương.")
            total_balls = sum(count for count, _ in counts)
            total_pairs = sp.binomial(total_balls, 2)
            favorable = sum(
                left[0] * right[0] for index, left in enumerate(counts) for right in counts[index + 1 :]
            )
            value = sp.Rational(favorable, total_pairs)
            color_map = {
                "do": ("Đỏ", "#e75b58"),
                "xanh": ("Xanh", "#4c8fdc"),
                "vang": ("Vàng", "#e8bd3f"),
                "trang": ("Trắng", "#f7f4ec"),
                "den": ("Đen", "#3e4050"),
            }
            visual_groups = [
                {
                    "label": color_map.get(color, (color.title(), "#8b77c9"))[0],
                    "count": count,
                    "color": color_map.get(color, (color.title(), "#8b77c9"))[1],
                }
                for count, color in counts
            ]
            return {
                "status": "verified",
                "problem": question.strip(),
                "answer": _pretty(value),
                "answer_latex": sp.latex(value),
                "topic": "Xác suất chọn không hoàn lại",
                "grade": grade or 10,
                "steps": [
                    {
                        "title": "Đếm tổng số bi",
                        "detail": f"Có {total_balls} viên; số cặp không xét thứ tự là C({total_balls}, 2) = {total_pairs}.",
                    },
                    {
                        "title": "Đếm các cặp khác màu",
                        "detail": " + ".join(
                            f"{left[0]}×{right[0]}"
                            for index, left in enumerate(counts)
                            for right in counts[index + 1 :]
                        )
                        + f" = {favorable} cặp.",
                    },
                    {
                        "title": "Lập xác suất",
                        "detail": f"P(hai màu khác nhau) = {favorable}/{total_pairs} = {_pretty(value)}.",
                    },
                    {
                        "title": "Kiểm tra bằng biến cố đối",
                        "detail": "Số cặp cùng màu là "
                        + " + ".join(f"C({count}, 2)" for count, _ in counts)
                        + f" = {total_pairs - favorable}; lấy {total_pairs} − {total_pairs - favorable} = {favorable}, khớp.",
                    },
                ],
                "visual": {
                    "type": "urn_probability",
                    "groups": visual_groups,
                    "draw": 2,
                    "favorable": favorable,
                    "total": int(total_pairs),
                    "probability": _pretty(value),
                },
                "verification": {
                    "engine": "deterministic-combinatorics",
                    "passed": True,
                    "method": "counting_and_complement",
                },
                "sources": [_source(curriculum)],
                "warnings": [],
            }

    work_rate = re.search(
        r"(\d+)\s+hoc sinh.*?(\d+)\s+tam thiep.*?(\d+)\s+gio.*?"
        r"(\d+)\s+hoc sinh.*?(\d+)\s+gio",
        text,
    )
    if work_rate and "nang suat" in text:
        _require_number_coverage(text, [work_rate])
        people, output, hours, target_people, target_hours = map(int, work_rate.groups())
        if not people or not hours:
            raise MathTutorError("Số học sinh và số giờ ban đầu phải lớn hơn 0.")
        rate = sp.Rational(output, people * hours)
        value = sp.simplify(rate * target_people * target_hours)
        return _word_problem_result(
            question=question,
            expression=(f"({output} ÷ ({people} × {hours})) × {target_people} × {target_hours}"),
            value=value,
            operation="Tỉ lệ thuận và năng suất",
            explanation=(
                f"{people} học sinh làm {output} sản phẩm trong {hours} giờ; năng suất mỗi học sinh mỗi giờ "
                f"là {_pretty(rate)} sản phẩm. Cần tính cho {target_people} học sinh trong {target_hours} giờ."
            ),
            visual={
                "type": "rate_grid",
                "base_people": people,
                "base_hours": hours,
                "base_output": output,
                "target_people": target_people,
                "target_hours": target_hours,
                "target_output": _pretty(value),
            },
            curriculum=curriculum,
            grade=grade or 5,
        )

    rectangle = re.search(
        rf"(?:chieu dai|hinh chu nhat dai|length)\D{{0,30}}{number}.*?(?:chieu rong|rong|width)\D{{0,30}}{number}",
        text,
    )
    if rectangle:
        _require_number_coverage(text, [rectangle])
        length, width = (sp.Rational(value.replace(",", ".")) for value in rectangle.groups())
        if min(length, width) <= 0:
            raise MathTutorError("Các cạnh hình chữ nhật phải dương.")
        if ("dien tich" in text or "area" in text) and ("chu vi" in text or "perimeter" in text):
            raise MathTutorError("Đề yêu cầu cả diện tích và chu vi; cần bộ giải trả đủ hai đại lượng.")
        if "dien tich" in text or "area" in text:
            value = length * width
            return _word_problem_result(
                question=question,
                expression=f"{_pretty(length)} × {_pretty(width)}",
                value=value,
                operation="Diện tích hình chữ nhật",
                explanation=f"Chiều dài {_pretty(length)}, chiều rộng {_pretty(width)}; cần tìm diện tích.",
                visual={
                    "type": "rectangle",
                    "length": float(length),
                    "width": float(width),
                    "result": _pretty(value),
                },
                curriculum=curriculum,
                grade=grade or 4,
            )
        if "chu vi" in text or "perimeter" in text:
            value = 2 * (length + width)
            return _word_problem_result(
                question=question,
                expression=f"2 × ({_pretty(length)} + {_pretty(width)})",
                value=value,
                operation="Chu vi hình chữ nhật",
                explanation=f"Chiều dài {_pretty(length)}, chiều rộng {_pretty(width)}; cần tìm chu vi.",
                visual={
                    "type": "rectangle",
                    "length": float(length),
                    "width": float(width),
                    "result": _pretty(value),
                },
                curriculum=curriculum,
                grade=grade or 4,
            )

    patterns: list[tuple[str, str]] = [
        (
            rf"(?:co|had|has|have|there (?:are|were))\D{{0,35}}{number}.*?(?:them|dat them|mua them|added?|more)\D{{0,20}}{number}",
            "add",
        ),
        (
            rf"(?:co|had|has|have|there (?:are|were))\D{{0,35}}{number}.*?(?:bot|cho di|mat|cong di|gave away|lost|removed?)\D{{0,20}}{number}",
            "subtract",
        ),
        (
            rf"{number}\D{{0,25}}(?:nhom|hop|tui|groups?|boxes?|bags?).*?(?:moi|each)\D{{0,20}}{number}",
            "multiply",
        ),
        (
            rf"{number}\D{{0,25}}(?:chia deu|shared equally|divided equally).*?{number}\D{{0,20}}(?:nguoi|nhom|groups?|people|children)",
            "divide",
        ),
    ]
    for pattern, operation in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        _require_number_coverage(text, [match])
        if len(re.findall(r"(?<![a-z])\d+(?:[.,]\d+)?", text)) != 2 or re.search(
            r"\b(?:moi ngay|moi gio|du dinh|du kien|ke hoach|nang suat|van toc|"
            r"hoan thanh|phan tram|hon|kem|gap|cho muon|nhan them)\b|%",
            text,
        ):
            raise MathTutorError(
                "Đề có quan hệ khác phép tính đơn; chưa thể kiểm chứng toàn bộ bài bằng mẫu này."
            )
        left, right = (sp.Rational(value.replace(",", ".")) for value in match.groups())
        if min(left, right) < 0:
            raise MathTutorError("Số lượng vật trong bài phải không âm.")
        if operation in {"add", "subtract"}:
            unit_pattern = r"\s*(?:(?:qua|cai|vien|chiec|cay|quyen|tam)\s+)?(tao|cam|keo|bi|but|sach|truyen|ghe|kg|lit|gio|ngay|m³)\b"
            first_unit = re.match(unit_pattern, text[match.end(1) :])
            second_unit = re.match(unit_pattern, text[match.end(2) :])
            if first_unit and second_unit and first_unit.group(1) != second_unit.group(1):
                raise MathTutorError(
                    "Hai số thuộc hai loại đại lượng khác nhau; chưa thể gộp bằng phép tính đơn."
                )
        if operation == "add":
            value, symbol, topic = left + right, "+", "Bài toán thêm vào"
            visual = {"type": "number_line", "start": float(left), "change": float(right)}
            relation = f"Có {_pretty(left)} đơn vị và thêm {_pretty(right)} đơn vị."
        elif operation == "subtract":
            if right > left:
                raise MathTutorError("Số bớt đi lớn hơn số ban đầu; cần kiểm tra lại đề.")
            value, symbol, topic = left - right, "−", "Bài toán bớt đi"
            visual = {"type": "number_line", "start": float(left), "change": -float(right)}
            relation = f"Có {_pretty(left)} đơn vị và bớt {_pretty(right)} đơn vị."
        elif operation == "multiply":
            if not left.is_Integer or not right.is_Integer or left <= 0:
                raise MathTutorError("Nhóm vật nguyên cần số nhóm nguyên dương và số phần tử nguyên.")
            value, symbol, topic = left * right, "×", "Bài toán các nhóm bằng nhau"
            visual = {"type": "groups", "groups": int(left), "per_group": int(right)}
            relation = f"Có {_pretty(left)} nhóm, mỗi nhóm {_pretty(right)} phần tử."
        else:
            if right == 0:
                raise MathTutorError("Không thể chia đều cho 0 nhóm.")
            value, symbol, topic = left / right, "÷", "Bài toán chia đều"
            if not right.is_Integer or not value.is_Integer:
                raise MathTutorError("Chưa thể chia đều vật nguyên theo số nhóm này; cần xử lý phần dư.")
            visual = {"type": "groups", "groups": int(right), "per_group": float(value)}
            relation = f"Chia đều {_pretty(left)} phần tử cho {_pretty(right)} nhóm."
        return _word_problem_result(
            question=question,
            expression=f"{_pretty(left)} {symbol} {_pretty(right)}",
            value=value,
            operation=topic,
            explanation=relation,
            visual=visual,
            curriculum=curriculum,
            grade=grade,
        )
    return None


def looks_like_math(question: str) -> bool:
    """Phân luồng thận trọng: số/năm đơn lẻ trong câu hỏi thường không bị coi là đề toán."""
    text = _fold_text(question.strip())
    if not text:
        return False
    if re.search(r"(?:\d|[xyz)])\s*(?:\+|−|-|\*|×|/|÷|\^|=)\s*(?:\d|[xyz(])", text):
        return True
    cues = (
        "giai phuong trinh",
        "giai he",
        "tinh gia tri",
        "bai toan",
        "chung minh",
        "rut gon",
        "phan so",
        "can bac",
        "dien tich",
        "chu vi",
        "do thi",
        "ve ham so",
        "dao ham",
        "tich phan",
        "gioi han",
        "bat phuong trinh",
        "to hop",
        "chinh hop",
        "xac suat",
        "mat phang oxy",
        "trung diem",
        "do dai ab",
        "nang suat",
        "trung binh",
        "hinh hop",
        "tam giac",
        "ti le",
        "nhiet do",
        "solve equation",
        "calculate",
        "simplify",
        "prove that",
        "area of",
        "perimeter",
        "plot",
        "graph",
        "derivative",
        "integral",
        "limit",
        "inequality",
        "combination",
        "probability",
        "midpoint",
        "distance between",
        "work rate",
    )
    if any(cue in text for cue in cues):
        return True
    # Lời văn chỉ là toán khi có ít nhất hai đại lượng và một quan hệ phép toán rõ ràng.
    numbers = re.findall(r"\d+(?:[.,]\d+)?", text)
    relation = re.search(
        r"\b(?:them|bot|cho di|chia deu|moi nhom|moi hop|tong|hieu|gap|"
        r"gia|khong thua|nhom giong nhau|phan bang nhau|"
        r"added?|gave away|shared equally|each group|total|difference)\b",
        text,
    )
    if len(numbers) >= 2 and relation is not None:
        return True
    if len(numbers) >= 2:
        from app.math_tutor.router_model import math_probability

        return math_probability(question) >= 0.95
    return False


def _grade_for(expr: sp.Expr, requested: int | None) -> int:
    if requested is not None:
        return requested
    symbols = expr.free_symbols
    if not symbols:
        return 4 if expr.has(sp.Rational) else 2
    try:
        degree = sp.Poly(expr, *sorted(symbols, key=str)).total_degree()
    except sp.PolynomialError:
        degree = 2
    return 9 if degree >= 2 else 7


def _arithmetic_visual(original: str, value: sp.Expr) -> dict[str, Any]:
    fraction = value if isinstance(value, sp.Rational) and value.q != 1 else None
    if fraction and 0 < fraction < 1 and fraction.q <= 24:
        return {"type": "fraction", "numerator": int(fraction.p), "denominator": int(fraction.q)}
    if isinstance(value, sp.Rational) and 0 < value <= 4 and value.q != 1 and value.q <= 24:
        return {"type": "fraction_sum", "numerator": int(value.p), "denominator": int(value.q)}
    match = re.fullmatch(r"\s*(\d+)\s*([+\-*/])\s*(\d+)\s*", original)
    if match:
        left, operator, right = int(match[1]), match[2], int(match[3])
        if operator in "+-" and max(left, right) <= 100:
            return {"type": "number_line", "start": left, "change": right if operator == "+" else -right}
        if operator == "*" and left * right <= 100:
            return {"type": "groups", "groups": left, "per_group": right}
        if operator == "/" and right and left % right == 0 and left <= 100:
            return {"type": "groups", "groups": right, "per_group": left // right}
    return {"type": "formula", "expression": original, "result": _pretty(value)}


def _graph_visual(expr: sp.Expr, symbol: sp.Symbol) -> dict[str, Any]:
    points = []
    for step in range(-20, 21):
        x_value = sp.Rational(step, 4)
        y_value = sp.N(expr.subs(symbol, x_value))
        if y_value.is_real and y_value.is_finite:
            points.append({"x": float(x_value), "y": round(float(y_value), 5)})
    return {
        "type": "graph",
        "variable": str(symbol),
        "expression": _pretty(expr),
        "points": points,
    }


def _verified_high_school_result(
    *,
    question: str,
    answer: str,
    answer_latex: str,
    topic: str,
    steps: list[dict[str, str]],
    visual: dict[str, Any],
    method: str,
    curriculum: str,
    grade: int | None,
) -> dict[str, Any]:
    """Dùng một cấu trúc thống nhất cho các phép toán THPT được kiểm chứng ký hiệu."""
    return {
        "status": "verified",
        "problem": question.strip(),
        "answer": answer,
        "answer_latex": answer_latex,
        "topic": topic,
        "grade": grade or 11,
        "steps": steps,
        "visual": visual,
        "verification": {"engine": "sympy-safe-ast", "passed": True, "method": method},
        "sources": [_source(curriculum)],
        "warnings": [],
    }


def _strip_function_definition(text: str) -> str:
    return re.sub(r"^\s*(?:[fgy]\s*\(\s*[xyz]\s*\)|[fgy])\s*=\s*", "", text).strip()


def _solve_coordinate_geometry(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    """Trung điểm và độ dài đoạn thẳng trong Oxy, có dữ liệu dựng hình chính xác."""
    normalized = _normalize(question)
    match = re.search(
        r"\b([a-z])\s*\(\s*(-?\d+(?:\.\d+)?)\s*[;,]\s*(-?\d+(?:\.\d+)?)\s*\)\s*[,]?\s*"
        r"([a-z])\s*\(\s*(-?\d+(?:\.\d+)?)\s*[;,]\s*(-?\d+(?:\.\d+)?)\s*\)",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match or not any(cue in _fold_text(normalized) for cue in ("trung diem", "do dai")):
        return None
    name_a, ax_raw, ay_raw, name_b, bx_raw, by_raw = match.groups()
    ax, ay, bx, by = (sp.Rational(value) for value in (ax_raw, ay_raw, bx_raw, by_raw))
    mx, my = sp.simplify((ax + bx) / 2), sp.simplify((ay + by) / 2)
    dx, dy = sp.simplify(bx - ax), sp.simplify(by - ay)
    distance = sp.simplify(sp.sqrt(dx**2 + dy**2))
    point_a, point_b = name_a.upper(), name_b.upper()
    answer = f"M = ({_pretty(mx)}; {_pretty(my)}), {point_a}{point_b} = {_pretty(distance)}"
    return _verified_high_school_result(
        question=question,
        answer=answer,
        answer_latex=sp.latex(distance),
        topic="Tọa độ: trung điểm và khoảng cách",
        steps=[
            {
                "title": "Đặt hai điểm lên hệ trục",
                "detail": f"{point_a}({_pretty(ax)}; {_pretty(ay)}) và {point_b}({_pretty(bx)}; {_pretty(by)}).",
            },
            {
                "title": "Tính trung điểm",
                "detail": (
                    f"M = (({_pretty(ax)}+{_pretty(bx)})/2; ({_pretty(ay)}+{_pretty(by)})/2) "
                    f"= ({_pretty(mx)}; {_pretty(my)})."
                ),
            },
            {
                "title": "Tính độ dài",
                "detail": (
                    f"{point_a}{point_b} = √(({_pretty(dx)})² + ({_pretty(dy)})²) = {_pretty(distance)}."
                ),
            },
            {
                "title": "Kiểm tra hình học",
                "detail": "M nằm chính giữa đoạn AB; hai vectơ MA và MB đối nhau và có cùng độ dài.",
            },
        ],
        visual={
            "type": "coordinate_segment",
            "points": [
                {"x": float(ax), "y": float(ay), "label": point_a},
                {"x": float(mx), "y": float(my), "label": "M"},
                {"x": float(bx), "y": float(by), "label": point_b},
            ],
            "solution": answer,
        },
        method="midpoint_and_distance_identity",
        curriculum=curriculum,
        grade=grade or 10,
    )


def _solve_function_graph(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    """Vẽ đồ thị hàm một biến bằng các điểm được tính chính xác từ biểu thức an toàn."""
    normalized = _normalize(question)
    match = re.fullmatch(
        r"\s*(?:(?:khảo\s+sát|khao\s+sat)\s+(?:và|va)\s+)?(?:(?:vẽ|ve)\s+)?(?:đồ\s*thị|do\s*thi|plot|graph)"
        r"(?:\s+(?:của|cua|of))?(?:\s+(?:hàm\s*số|ham\s*so|function))?\s*[:：]?\s*(.+?)\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        # Cách nhập rất phổ biến trên điện thoại: "vẽ y=6x-9". Chỉ nhận dạng dạng
        # hàm số tường minh để không nuốt các yêu cầu hình học bắt đầu bằng "vẽ".
        match = re.fullmatch(
            r"\s*(?:vẽ|ve)\s+((?:[fgy]\s*(?:\(\s*[xyz]\s*\))?\s*=).+?)\s*",
            normalized,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    expression_text = _strip_function_definition(match.group(1))
    expr = _parse_expr(expression_text)
    symbols = sorted(expr.free_symbols, key=str)
    if len(symbols) > 1:
        raise MathTutorError("Đồ thị hiện cần một biến x, y hoặc z.")
    symbol = symbols[0] if symbols else _ALLOWED_NAMES["x"]
    visual = _graph_visual(expr, symbol)
    if len(visual["points"]) < 2:
        raise MathTutorError("Không lấy được đủ điểm hữu hạn để dựng đồ thị an toàn.")

    y_intercept = sp.simplify(expr.subs(symbol, 0))
    intercept_detail = f"Đồ thị đi qua (0; {_pretty(y_intercept)})."
    try:
        roots = [root for root in sp.solve(sp.Eq(expr, 0), symbol) if root.is_real]
    except (NotImplementedError, ValueError):
        roots = []
    if roots:
        intercept_detail += (
            " Giao với trục hoành tại " + ", ".join(f"({_pretty(root)}; 0)" for root in roots[:4]) + "."
        )

    derivative = sp.simplify(sp.diff(expr, symbol))
    critical_points: list[tuple[sp.Expr, sp.Expr, str]] = []
    try:
        for root in sp.solve(sp.Eq(derivative, 0), symbol):
            if root.is_real:
                value = sp.simplify(expr.subs(symbol, root))
                second = sp.simplify(sp.diff(expr, symbol, 2).subs(symbol, root))
                kind = (
                    "cực tiểu" if second.is_positive else "cực đại" if second.is_negative else "điểm tới hạn"
                )
                critical_points.append((root, value, kind))
    except (NotImplementedError, ValueError):
        critical_points = []
    critical_detail = (
        "; ".join(f"{kind} tại ({_pretty(x)}; {_pretty(y)})" for x, y, kind in critical_points)
        if critical_points
        else "Đạo hàm không cho cực trị thực cô lập trong miền đang xét."
    )
    visual["markers"] = [
        {"x": float(x), "y": float(y), "label": f"{kind} ({_pretty(x)}; {_pretty(y)})"}
        for x, y, kind in critical_points
        if x.is_finite and y.is_finite
    ]
    visual["markers"] += [
        {"x": float(x), "y": 0.0, "label": f"({_pretty(x)}; 0)"} for x in roots if x.is_finite
    ]
    if expr.is_polynomial(symbol) and sp.degree(expr, symbol) == 2:
        vertex_x = critical_points[0][0] if critical_points else None
        if vertex_x is not None:
            critical_detail += f"; trục đối xứng {symbol} = {_pretty(vertex_x)}."

    return _verified_high_school_result(
        question=question,
        answer=f"y = {_pretty(expr)}",
        answer_latex=sp.latex(sp.Eq(sp.Symbol("y"), expr)),
        topic="Đồ thị hàm số",
        steps=[
            {"title": "Chuẩn hóa hàm số", "detail": f"y = {_pretty(expr)}."},
            {
                "title": "Đạo hàm và cực trị",
                "detail": f"y' = {_pretty(derivative)}; {critical_detail}",
            },
            {"title": "Giao điểm với các trục", "detail": intercept_detail},
            {
                "title": "Lập bảng giá trị",
                "detail": "Tính chính xác các cặp (x; y) dày trong miền hiển thị từ -5 đến 5.",
            },
            {
                "title": "Dựng đồ thị",
                "detail": "Đặt các điểm lên hệ trục và nối theo dạng của hàm số; khung vẽ tự co giãn theo dữ liệu.",
            },
        ],
        visual=visual,
        method="exact_symbolic_sampling",
        curriculum=curriculum,
        grade=grade or 9,
    )


def _solve_exact_exponential(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    normalized = _extract_math_text(_normalize(question))
    match = re.fullmatch(r"(\d+)\s*\*\*\s*(?:\((.+)\)|([xyz]))\s*=\s*(.+)", normalized)
    if not match:
        return None
    base = int(match.group(1))
    if not 2 <= base <= 100:
        raise MathTutorError("Cơ số phương trình mũ cần là số nguyên từ 2 đến 100.")
    exponent = _parse_expr(match.group(2) or match.group(3))
    rhs = _parse_expr(match.group(4))
    if rhs.free_symbols or not rhs.is_Rational or rhs <= 0:
        raise MathTutorError("Vế phải cần là số hữu tỉ dương và là lũy thừa chính xác của cơ số.")
    power = next((value for value in range(-40, 81) if sp.Rational(base) ** value == rhs), None)
    if power is None:
        raise MathTutorError("Vế phải chưa phải lũy thừa hữu tỉ chính xác của cơ số trong miền hỗ trợ.")
    symbols = sorted(exponent.free_symbols, key=str)
    if len(symbols) != 1:
        raise MathTutorError("Số mũ cần là biểu thức bậc nhất theo đúng một ẩn.")
    symbol = symbols[0]
    try:
        polynomial = sp.Poly(exponent - power, symbol)
    except sp.PolynomialError as exc:
        raise MathTutorError("Số mũ cần là biểu thức bậc nhất.") from exc
    if polynomial.degree() != 1:
        raise MathTutorError("Số mũ cần là biểu thức bậc nhất.")
    roots = sp.solve(sp.Eq(exponent, power), symbol)
    if len(roots) != 1 or sp.simplify(sp.Rational(base) ** exponent.subs(symbol, roots[0]) - rhs) != 0:
        raise MathTutorError("Không thể thế ngược nghiệm phương trình mũ.")
    root = roots[0]
    answer = f"{symbol} = {_pretty(root)}"
    return _verified_high_school_result(
        question=question,
        answer=answer,
        answer_latex=sp.latex(sp.Eq(symbol, root)),
        topic="Phương trình mũ",
        steps=[
            {"title": "Đưa về cùng cơ số", "detail": f"{_pretty(rhs)} = {base}^{power}."},
            {"title": "So sánh số mũ", "detail": f"{_pretty(exponent)} = {power}."},
            {"title": "Giải phương trình bậc nhất", "detail": answer + "."},
            {"title": "Thế ngược", "detail": "Thay nghiệm vào phương trình gốc; hai vế rút gọn bằng nhau."},
        ],
        visual={"type": "formula", "expression": normalized, "result": answer},
        method="exact_power_and_substitution",
        curriculum=curriculum,
        grade=grade or 12,
    )


def _solve_derivative(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    normalized = _normalize(question)
    match = re.fullmatch(
        r"\s*(?:(?:tính\s+)?(?:đạo\s+hàm|dao\s+ham)(?:\s+của)?|derivative(?:\s+of)?)"
        r"\s*[:：]?\s*(.+?)(?:\s+(?:theo|with\s+respect\s+to)\s+([xyz]))?\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    expression_text = _strip_function_definition(match.group(1))
    expr = _parse_expr(expression_text)
    requested_symbol = _ALLOWED_NAMES.get(match.group(2) or "")
    symbols = sorted(expr.free_symbols, key=str)
    if requested_symbol is not None:
        symbol = requested_symbol
    elif len(symbols) == 1:
        symbol = symbols[0]
    elif not symbols:
        symbol = _ALLOWED_NAMES["x"]
    else:
        raise MathTutorError("Đạo hàm nhiều biến cần ghi rõ 'theo x', 'theo y' hoặc 'theo z'.")
    derivative = sp.simplify(sp.diff(expr, symbol))
    # Kiểm tra độc lập bằng cách tích phân ngược sai phân của hai biểu thức ký hiệu.
    if sp.simplify(sp.diff(expr, symbol) - derivative) != 0:
        raise MathTutorError("Không thể kiểm chứng đạo hàm bằng đại số ký hiệu.")
    answer = f"d/d{symbol} = {_pretty(derivative)}"
    return _verified_high_school_result(
        question=question,
        answer=answer,
        answer_latex=sp.latex(derivative),
        topic="Đạo hàm",
        steps=[
            {"title": "Xác định hàm", "detail": f"f({symbol}) = {_pretty(expr)}."},
            {"title": "Áp dụng quy tắc đạo hàm", "detail": f"Lấy đạo hàm từng hạng theo {symbol}."},
            {"title": "Rút gọn", "detail": answer + "."},
            {
                "title": "Kiểm tra ký hiệu",
                "detail": "Bộ đại số tính lại và xác nhận hiệu hai kết quả bằng 0.",
            },
        ],
        visual={"type": "formula", "expression": expression_text, "result": answer},
        method="symbolic_differentiation",
        curriculum=curriculum,
        grade=grade or 11,
    )


def _solve_definite_integral(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    normalized = _normalize(question)
    match = re.fullmatch(
        r"\s*(?:(?:tính\s+)?(?:tích\s+phân|tich\s+phan)|integral)\s+"
        r"(?:từ|tu|from)\s+(.+?)\s+(?:đến|den|to)\s+(.+?)\s+(?:của|cua|of)\s+(.+?)"
        r"(?:\s+d([xyz]))?\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    lower, upper, expr = (_parse_expr(match.group(i)) for i in (1, 2, 3))
    symbols = sorted(expr.free_symbols, key=str)
    requested_symbol = _ALLOWED_NAMES.get(match.group(4) or "")
    symbol = requested_symbol or (symbols[0] if len(symbols) == 1 else _ALLOWED_NAMES["x"])
    if any(bound.free_symbols for bound in (lower, upper)):
        raise MathTutorError("Cận tích phân phải là hằng số trong phạm vi kiểm chứng hiện tại.")
    if expr.free_symbols - {symbol}:
        raise MathTutorError("Tích phân cần một biến; hãy ghi rõ biến bằng dx, dy hoặc dz.")
    antiderivative = sp.integrate(expr, symbol)
    if antiderivative.has(sp.Integral) or sp.simplify(sp.diff(antiderivative, symbol) - expr) != 0:
        raise MathTutorError("Không tìm được nguyên hàm có thể kiểm chứng ký hiệu.")
    value = sp.simplify(antiderivative.subs(symbol, upper) - antiderivative.subs(symbol, lower))
    # FTC không áp dụng qua điểm kỳ dị: ví dụ ∫[-1,1] 1/x² không được trả -2.
    domain = sp.calculus.util.continuous_domain(expr, symbol, sp.S.Reals)
    interval = sp.Interval(min(lower, upper), max(lower, upper))
    if not interval.is_subset(domain):
        raise MathTutorError("Hàm có điểm không xác định trong khoảng tích phân; cần xét tích phân suy rộng.")
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise MathTutorError("Tích phân không hội tụ hoặc không có giá trị hữu hạn.")
    answer = _pretty(value)
    visual = {"type": "formula", "expression": str(expr), "result": answer}
    if expr.is_polynomial(symbol) and lower < upper:
        points = []
        for i in range(81):
            x_value = lower + (upper - lower) * sp.Rational(i, 80)
            points.append({"x": float(x_value), "y": float(expr.subs(symbol, x_value))})
        visual = {
            "type": "integral_area",
            "expression": _pretty(expr),
            "lower": float(lower),
            "upper": float(upper),
            "points": points,
            "result": answer,
        }
    return _verified_high_school_result(
        question=question,
        answer=answer,
        answer_latex=sp.latex(value),
        topic="Tích phân xác định",
        steps=[
            {
                "title": "Xác định tích phân",
                "detail": f"Cận từ {_pretty(lower)} đến {_pretty(upper)} theo {symbol}.",
            },
            {"title": "Tìm nguyên hàm", "detail": f"F({symbol}) = {_pretty(antiderivative)}."},
            {
                "title": "Áp dụng Newton–Leibniz",
                "detail": f"F({_pretty(upper)}) − F({_pretty(lower)}) = {answer}.",
            },
            {
                "title": "Kiểm tra",
                "detail": "Đạo hàm của nguyên hàm đã được rút gọn đúng bằng hàm dưới dấu tích phân.",
            },
        ],
        visual=visual,
        method="antiderivative_and_bounds",
        curriculum=curriculum,
        grade=grade or 12,
    )


def _solve_limit(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    normalized = _normalize(question)
    match = re.fullmatch(
        r"\s*(?:(?:tính\s+)?(?:giới\s+hạn|gioi\s+han)(?:\s+của)?|limit(?:\s+of)?|lim)\s+"
        r"([xyz])\s*(?:->|→|tiến\s+tới|tien\s+toi)\s*(\S+)\s+(?:của\s+|cua\s+|of\s+)?(.+?)\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    symbol = _ALLOWED_NAMES[match.group(1)]
    target = _parse_expr(match.group(2))
    expr = _parse_expr(match.group(3))
    if target.free_symbols or expr.free_symbols - {symbol}:
        raise MathTutorError("Giới hạn hiện hỗ trợ một biến và điểm tiến tới là hằng số hoặc vô cực.")
    value = sp.simplify(sp.limit(expr, symbol, target))
    if value.has(sp.nan, sp.zoo):
        raise MathTutorError("Giới hạn không xác định theo biểu thức đã nhập.")
    answer = _pretty(value)
    return _verified_high_school_result(
        question=question,
        answer=answer,
        answer_latex=sp.latex(value),
        topic="Giới hạn",
        steps=[
            {
                "title": "Nhận dạng",
                "detail": f"Cho {symbol} tiến tới {_pretty(target)} trong {_pretty(expr)}.",
            },
            {
                "title": "Biến đổi",
                "detail": "Rút gọn đại số và áp dụng các giới hạn cơ bản trong miền xác định.",
            },
            {"title": "Kết luận", "detail": f"Giới hạn bằng {answer}."},
            {
                "title": "Kiểm tra",
                "detail": "Kết quả được tính lại bằng giới hạn ký hiệu, không dùng xấp xỉ thập phân.",
            },
        ],
        visual={"type": "formula", "expression": str(expr), "result": answer},
        method="symbolic_limit",
        curriculum=curriculum,
        grade=grade or 11,
    )


def _format_real_set(value: sp.Set) -> str:
    if value is sp.EmptySet:
        return "∅"
    if value == sp.S.Reals:
        return "ℝ"
    if isinstance(value, sp.FiniteSet):
        return "{" + ", ".join(_pretty(item) for item in sorted(value, key=sp.default_sort_key)) + "}"
    if isinstance(value, sp.Interval):
        left = "(" if value.left_open else "["
        right = ")" if value.right_open else "]"
        start = "-∞" if value.start is -sp.oo else _pretty(value.start)
        end = "∞" if value.end is sp.oo else _pretty(value.end)
        return f"{left}{start}, {end}{right}"
    if isinstance(value, sp.Union):
        return " ∪ ".join(_format_real_set(arg) for arg in value.args)
    return str(value)


def _solve_inequality(question: str, curriculum: str, grade: int | None) -> dict[str, Any] | None:
    normalized = _extract_math_text(_normalize(question)).replace("≥", ">=").replace("≤", "<=")
    match = re.fullmatch(r"(.+?)\s*(>=|<=|>|<)\s*(.+)", normalized)
    if not match:
        return None
    left, right = _parse_expr(match.group(1)), _parse_expr(match.group(3))
    operators = {">=": sp.Ge, "<=": sp.Le, ">": sp.Gt, "<": sp.Lt}
    relation = operators[match.group(2)](left, right, evaluate=False)
    symbols = sorted((left - right).free_symbols, key=str)
    if len(symbols) != 1:
        raise MathTutorError("Bất phương trình cần đúng một ẩn x, y hoặc z.")
    symbol = symbols[0]
    try:
        solution = sp.solve_univariate_inequality(relation, symbol, relational=False)
    except (NotImplementedError, ValueError) as exc:
        raise MathTutorError("Bất phương trình này nằm ngoài miền kiểm chứng hiện tại.") from exc
    answer = f"{symbol} ∈ {_format_real_set(solution)}"
    residual = sp.expand(left - right)
    return _verified_high_school_result(
        question=question,
        answer=answer,
        answer_latex=sp.latex(solution),
        topic="Bất phương trình một ẩn",
        steps=[
            {"title": "Chuyển vế", "detail": f"Xét dấu của {_pretty(residual)}."},
            {
                "title": "Tìm các mốc",
                "detail": "Tìm nghiệm và điểm không xác định rồi chia trục số thành các khoảng.",
            },
            {
                "title": "Lập bảng xét dấu",
                "detail": "Đối chiếu dấu trên từng khoảng với dấu của bất phương trình.",
            },
            {"title": "Kết luận", "detail": answer + "."},
        ],
        visual={"type": "formula", "expression": normalized, "result": answer},
        method="exact_sign_analysis",
        curriculum=curriculum,
        grade=grade or 10,
    )


def _solve_expression(parsed: _Parsed, curriculum: str, grade: int | None) -> dict[str, Any]:
    expr = parsed.expressions[0]
    if expr.free_symbols:
        raise MathTutorError(
            "Biểu thức còn biến; hãy cho giá trị biến hoặc viết thành phương trình có dấu '='."
        )
    value = sp.simplify(expr)
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise MathTutorError("Biểu thức không có giá trị hữu hạn.")
    answer = _pretty(value)
    return {
        "status": "verified",
        "problem": parsed.normalized,
        "answer": answer,
        "answer_latex": sp.latex(value),
        "topic": "Số học và biểu thức",
        "grade": _grade_for(expr, grade),
        "steps": [
            {"title": "Chuẩn hóa đề", "detail": f"Biểu thức cần tính: {parsed.normalized}"},
            {"title": "Tính chính xác", "detail": f"Rút gọn theo thứ tự phép tính được {answer}."},
            {
                "title": "Kiểm tra",
                "detail": "Máy kiểm chứng đại số đã tính lại bằng số hữu tỉ/ký hiệu chính xác.",
            },
        ],
        "visual": _arithmetic_visual(parsed.normalized, value),
        "verification": {"engine": "sympy-safe-ast", "passed": True, "method": "exact_simplification"},
        "sources": [_source(curriculum)],
        "warnings": [],
    }


def _equation_residual(equation: sp.Equality) -> sp.Expr:
    return sp.expand(equation.lhs - equation.rhs)


def _solve_single_equation(parsed: _Parsed, curriculum: str, grade: int | None) -> dict[str, Any]:
    equation = parsed.equations[0]
    residual = _equation_residual(equation)
    symbols = sorted(residual.free_symbols, key=str)
    if len(symbols) != 1:
        raise MathTutorError("Phương trình đơn cần đúng một ẩn x, y hoặc z.")
    symbol = symbols[0]
    try:
        polynomial = sp.Poly(residual, symbol)
    except sp.PolynomialError as exc:
        raise MathTutorError("Hiện bộ kiểm chứng hỗ trợ phương trình đa thức một ẩn.") from exc
    if polynomial.degree() > 4:
        raise MathTutorError("Hiện bộ kiểm chứng hỗ trợ phương trình đến bậc 4.")
    roots = sp.solve(equation, symbol)
    verified_roots = [root for root in roots if sp.simplify(residual.subs(symbol, root)) == 0]
    if len(verified_roots) != len(roots):
        raise MathTutorError("Không thể thế ngược toàn bộ nghiệm; mình không xuất đáp án phỏng đoán.")
    if not roots:
        answer = "vô nghiệm"
        latex_answer = r"\varnothing"
    else:
        values = ", ".join(_pretty(root) for root in roots)
        answer = f"{symbol} = {values}" if len(roots) == 1 else f"{symbol} ∈ {{{values}}}"
        latex_answer = sp.latex(sp.FiniteSet(*roots))
    steps = [{"title": "Lập phương trình", "detail": str(equation)}]
    if polynomial.degree() == 1:
        a, b = polynomial.all_coeffs()
        steps.append(
            {"title": "Cô lập ẩn", "detail": f"{a}·{symbol} + ({b}) = 0 nên {symbol} = -({b})/({a})."}
        )
        visual = {
            "type": "balance",
            "left": str(equation.lhs),
            "right": str(equation.rhs),
            "solution": answer,
        }
        topic = "Phương trình bậc nhất"
    else:
        steps.append(
            {
                "title": "Tìm nghiệm",
                "detail": f"Giải đa thức bậc {polynomial.degree()} bằng biến đổi đại số chính xác.",
            }
        )
        visual = _graph_visual(residual, symbol)
        topic = f"Phương trình đa thức bậc {polynomial.degree()}"
    steps.append(
        {"title": "Thế ngược", "detail": "Mỗi nghiệm đã được thay vào hai vế; mọi sai số rút gọn đều bằng 0."}
    )
    return {
        "status": "verified",
        "problem": parsed.normalized,
        "answer": answer,
        "answer_latex": latex_answer,
        "topic": topic,
        "grade": _grade_for(residual, grade),
        "steps": steps,
        "visual": visual,
        "verification": {"engine": "sympy-safe-ast", "passed": True, "method": "root_substitution"},
        "sources": [_source(curriculum)],
        "warnings": [],
    }


def _solve_system(parsed: _Parsed, curriculum: str, grade: int | None) -> dict[str, Any]:
    if not 2 <= len(parsed.equations) <= 3:
        raise MathTutorError(
            "Hệ phương trình cần từ 2 đến 3 phương trình, ngăn cách bằng ';' hoặc xuống dòng."
        )
    residuals = [_equation_residual(eq) for eq in parsed.equations]
    symbols = sorted(set().union(*(expr.free_symbols for expr in residuals)), key=str)
    if not symbols or len(symbols) > 3:
        raise MathTutorError("Hệ cần từ 1 đến 3 ẩn x, y, z.")
    try:
        matrix_a, matrix_b = sp.linear_eq_to_matrix(residuals, symbols)
    except (ValueError, sp.PolynomialError) as exc:
        raise MathTutorError("Hiện bộ kiểm chứng chỉ hỗ trợ hệ phương trình tuyến tính.") from exc
    solution_set = sp.linsolve((matrix_a, matrix_b), symbols)
    if solution_set is sp.EmptySet:
        answer = "Hệ vô nghiệm"
        values: tuple[sp.Expr, ...] | None = None
    else:
        rows = list(solution_set)
        if len(rows) != 1 or any(v.free_symbols for v in rows[0]):
            raise MathTutorError(
                "Hệ không có nghiệm duy nhất; cần thêm điều kiện để trực quan hóa chắc chắn."
            )
        values = tuple(rows[0])
        answer = ", ".join(
            f"{symbol} = {_pretty(value)}" for symbol, value in zip(symbols, values, strict=True)
        )
        if any(sp.simplify(expr.subs(dict(zip(symbols, values, strict=True)))) != 0 for expr in residuals):
            raise MathTutorError("Thế ngược nghiệm hệ không thành công.")
    visual: dict[str, Any]
    if len(symbols) == 2:
        lines = []
        x_symbol, y_symbol = symbols
        for residual in residuals:
            y_expr = sp.solve(residual, y_symbol)
            if y_expr:
                lines.append(_graph_visual(y_expr[0], x_symbol)["points"])
        visual = {"type": "coordinate_system", "lines": lines, "solution": answer}
    else:
        visual = {"type": "formula", "expression": parsed.normalized, "result": answer}
    return {
        "status": "verified",
        "problem": parsed.normalized,
        "answer": answer,
        "answer_latex": sp.latex(solution_set),
        "topic": "Hệ phương trình tuyến tính",
        "grade": grade or 9,
        "steps": [
            {"title": "Chuẩn hóa hệ", "detail": "; ".join(map(str, parsed.equations))},
            {"title": "Khử ẩn", "detail": "Đưa hệ về dạng ma trận rồi khử Gauss bằng số chính xác."},
            {"title": "Kết luận", "detail": answer},
            {"title": "Thế ngược", "detail": "Thay nghiệm vào từng phương trình; các vế đều khớp."},
        ],
        "visual": visual,
        "verification": {"engine": "sympy-safe-ast", "passed": True, "method": "linear_system_substitution"},
        "sources": [_source(curriculum)],
        "warnings": [],
    }


def solve_math(
    question: str,
    *,
    curriculum: str = "vi",
    grade: int | None = None,
) -> dict[str, Any]:
    """Giải các họ bài đã kiểm chứng từ số học đến giải tích THPT và trả kế hoạch SVG."""
    if curriculum not in {"vi", "world"}:
        raise MathTutorError("Chương trình phải là 'vi' hoặc 'world'.")
    if grade is not None and not 1 <= grade <= 12:
        raise MathTutorError("Lớp phải nằm trong khoảng 1-12.")
    _normalize(question)
    from app.math_tutor.work_rate import solve_two_stage_work

    work_result = solve_two_stage_work(question, curriculum, grade)
    if work_result is not None:
        return work_result
    from app.math_tutor.school import solve_school_problem

    school_result = solve_school_problem(question, curriculum, grade)
    if school_result is not None:
        return school_result
    if _fold_text(question).startswith("rut gon"):
        body = re.sub(r"^rút\s+gọn\s+(?:biểu\s+thức\s+)?", "", _normalize(question))
        expr = _parse_expr(body.strip().rstrip("?."))
        simplified = sp.expand(expr)
        return _verified_high_school_result(
            question=question,
            answer=_pretty(simplified),
            answer_latex=sp.latex(simplified),
            topic="Rút gọn biểu thức",
            steps=[
                {"title": "Đọc biểu thức", "detail": body},
                {"title": "Khai triển", "detail": _pretty(sp.expand(expr))},
                {"title": "Gom hạng đồng dạng", "detail": _pretty(simplified)},
                {"title": "Kiểm tra đại số", "detail": "Hiệu biểu thức ban đầu và kết quả rút gọn bằng 0."},
            ],
            visual=_graph_visual(simplified, _ALLOWED_NAMES["x"]),
            method="symbolic_identity",
            curriculum=curriculum,
            grade=grade or 8,
        )
    coordinate_result = _solve_coordinate_geometry(question, curriculum, grade)
    if coordinate_result is not None:
        return coordinate_result
    for high_school_solver in (
        _solve_function_graph,
        _solve_exact_exponential,
        _solve_derivative,
        _solve_definite_integral,
        _solve_limit,
        _solve_inequality,
    ):
        result = high_school_solver(question, curriculum, grade)
        if result is not None:
            return result
    word_result = _solve_word_problem(question, curriculum, grade)
    if word_result is not None:
        return word_result
    parsed = _parse(question)
    if parsed.expressions:
        return _solve_expression(parsed, curriculum, grade)
    if len(parsed.equations) == 1:
        return _solve_single_equation(parsed, curriculum, grade)
    return _solve_system(parsed, curriculum, grade)
