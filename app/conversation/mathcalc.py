"""Tính phép tính số học đơn giản bằng code (không để model 4B tự tính, không gắn ghi chú "chưa có nguồn").

Chỉ nhận câu mà TOÀN BỘ nội dung là một phép tính: "50 nhân 50 bằng mấy", "tính 12,5 + 7", "2^10 = ?",
"(3 + 4) x 5 là bao nhiêu". Có chữ khác (phương trình "x bằng mấy", đơn vị, lời văn) -> None để luồng thường xử lý.
"""

from __future__ import annotations

import ast
import operator
import re

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


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(b) > 64 or abs(a) > 10**6):
            raise _Unsupported("pow")
        r = _OPS[type(node.op)](a, b)
        if isinstance(r, complex) or abs(r) > MAX_ABS:
            raise _Unsupported("range")
        return r
    raise _Unsupported(type(node).__name__)


def _fmt(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v):,}".replace(",", ".")
    s = f"{v:,.6f}".rstrip("0").rstrip(".")
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def answer_math_question(text: str | None) -> str | None:
    if not text or len(text) > 120:
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
    exact = (
        ""
        if float(value).is_integer() or len(_fmt(value).split(",")[-1]) < 6
        else " (làm tròn 6 chữ số thập phân)"
    )
    return f"{shown} = {_fmt(value)}{exact}."
