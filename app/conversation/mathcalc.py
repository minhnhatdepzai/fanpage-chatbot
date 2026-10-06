"""Tự nhận biết câu hỏi toán và ưu tiên bộ giải xác định trước LLM.

Phép tính cũ giữ định dạng Việt Nam; phương trình và lời văn đi qua MathScope. Câu hỏi bình thường hoặc đề toán
ngoài miền kiểm chứng trả ``None`` để tiếp tục qua chatbot/RAG thay vì bị chặn.
"""

from __future__ import annotations

import ast
import operator
import re
from decimal import Decimal, localcontext
from fractions import Fraction

from app.math_tutor import MathTutorError, looks_like_math, solve_math
from app.observability.redaction import strip_diacritics

_LEAD = re.compile(
    r"^(?:(?:cho )?(?:minh|toi|em|to|tui) hoi |hoi )?(?:tinh(?: giup| dum| ho)?(?: minh| toi| em)? )?"
)
_TAIL = re.compile(
    r"\s*(?:=|bang may|bang bao nhieu|la bao nhieu|bang gi|ra bao nhieu|la may|ra may)?\s*[?.!]*\s*$"
)
_CUE = re.compile(
    r"^(?:.* )?tinh |(?:=|bang may|bang bao nhieu|la bao nhieu|bang gi|ra bao nhieu|la may|ra may)\s*[?.!]*\s*$"
)
_WORD_OPS = [
    (re.compile(r"\bnhan(?: voi)?\b|(?<=\d)\s*[x×]\s*(?=[\d(])|(?<=\))\s*[x×]\s*(?=[\d(])"), "*"),
    (re.compile(r"\bchia(?: cho)?\b|÷|(?<=[\d)])\s*:\s*(?=[\d(])"), "/"),
    (re.compile(r"\bcong(?: voi)?\b"), "+"),
    (re.compile(r"\btru(?: di)?\b|−"), "-"),
    (re.compile(r"\bmu\b|\^"), "**"),
]
_EXPR = re.compile(r"^[\d\s.,+\-*/()]+$")
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_SHOW = {"*": "×", "/": ":", "-": "−", "**": "^"}
MAX_ABS = 10**15


class _Unsupported(ValueError):
    pass


def _number(tok: str) -> str:
    """Kiểu Việt Nam: "1.000.000" = một triệu, "3,5" = 3.5."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok):
        return tok.replace(".", "")
    if tok.count(",") == 1 and "." not in tok:
        return tok.replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", tok):
        return tok
    raise _Unsupported(tok)


def _eval(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 64 or abs(a) > 10**6):
            raise _Unsupported("pow")
        if isinstance(node.op, ast.Pow) and b.denominator != 1:
            raise _Unsupported("fractional_power")
        r = _OPS[type(node.op)](a, b)
        if isinstance(r, complex) or abs(r) > MAX_ABS:
            raise _Unsupported("range")
        return r
    raise _Unsupported(type(node).__name__)


def _fmt(v: Fraction) -> str:
    if v.denominator == 1:
        return f"{int(v):,}".replace(",", ".")
    with localcontext() as ctx:
        ctx.prec = max(28, len(str(abs(v.numerator))) + len(str(v.denominator)) + 8)
        s = f"{Decimal(v.numerator) / Decimal(v.denominator):,.6f}".rstrip("0").rstrip(".")
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def answer_math_question(text: str | None) -> str | None:
    if not text or len(text) > 1200:
        return None
    legacy = _answer_simple_arithmetic(text)
    if legacy is not None:
        return legacy
    if not looks_like_math(text):
        return None
    try:
        result = solve_math(text)
    except MathTutorError:
        return None
    steps = "\n".join(
        f"{index}. {step['title']}: {step['detail']}" for index, step in enumerate(result["steps"], 1)
    )
    return (
        f"Đáp án: {result['answer']}\n\n{steps}\n\n"
        f"Kiểm chứng: đạt ({result['verification']['method']}). "
        "Bạn có thể mở /web/math để xem hình trực quan tương tác."
    )


def _answer_simple_arithmetic(text: str) -> str | None:
    if len(text) > 120:
        return None
    t = strip_diacritics(text.lower()).strip()
    if not _CUE.search(t):
        return None
    body = _TAIL.sub("", _LEAD.sub("", t, count=1))
    for pat, op in _WORD_OPS:
        body = pat.sub(f" {op} ", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body or not _EXPR.match(body) or not re.search(r"\d\s*[-+*/]", body.replace("**", "*")):
        return None
    try:
        expr = re.sub(r"\d[\d.,]*", lambda m: _number(m.group(0)), body)
        value = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Phép tính có chia cho 0 nên không có kết quả."
    except (_Unsupported, SyntaxError, OverflowError):
        return None
    shown = re.sub(r"\*\*|[*/-]", lambda m: _SHOW[m.group(0)], body)
    shown = re.sub(r"\s*([×:+−^])\s*", r" \1 ", shown).replace("( ", "(").replace(" )", ")").strip()
    rendered = _fmt(value)
    reconstructed = Fraction(rendered.replace(".", "").replace(",", "."))
    exact = "" if reconstructed == value else " (làm tròn 6 chữ số thập phân)"
    return f"{shown} = {_fmt(value)}{exact}."
