"""Kiểm tra và làm sạch câu trả lời của model trước khi gửi.

Trả về văn bản cuối (hoặc None nếu không dùng được) + danh sách cờ để theo dõi/đánh giá.
Các cờ "chặn" thay nội dung bằng câu an toàn; các cờ "theo dõi" chỉ ghi nhận.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.conversation.intents import is_capability_question
from app.conversation.knowledge import extract_urls
from app.conversation.prompts import (
    DISCLOSURE_SENTENCE,
    IMAGE_UNVERIFIED_NOTE,
    NO_VERIFIED_TITLES_REPLY,
    PROMPT_CANARY,
    SECURITY_REFUSAL,
    SOURCES_FOOTER_TITLE,
    UNVERIFIED_NOTE,
)
from app.observability.redaction import redact_secrets, strip_diacritics, vietnamese_score

_TEMPLATE_TOKENS = re.compile(r"<\|[^|>]{1,40}\|>|</?s>")
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)
_ROLE_PREFIX = re.compile(r"^\s*(?:assistant|trợ lý|bot)\s*[:：]\s*", re.I)
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f02f\U0001f0a0-\U0001f0ff\U0001f100-\U0001f1ff]"
)
# so trên văn bản đã bỏ dấu; không dùng "ai" đơn lẻ vì trùng với "ai" (= người nào)
# (chỉ cụm tự mô tả; "chatbot", "trí tuệ nhân tạo" đứng riêng có thể chỉ là chủ đề đang giải thích)
_AI_MENTION = re.compile(r"\b(tro ly (?:ai|ao|tu dong)|bot tu dong|tu dong tra loi)\b")
# Tự xưng là AI ("mình là AI", "I'm an AI"). Chỉ nhắc chữ "AI" như chủ đề (fanpage về AI) KHÔNG phải công bố.
_AI_SELF = re.compile(
    r"\b(?:minh|toi|em|day|to)\s+(?:chi\s+)?la\s+(?:mot\s+)?(?:ai|bot|chatbot|robot|tro ly ao|tri tue nhan tao)\b"
    r"|\bai tu dong\b|\bi(?:'m| am) an? (?:ai|bot|automated)\b"
)
_CLAIMS_HUMAN = re.compile(
    r"(?<!khong phai )(?<!chang phai )\b(minh|toi|em)\s+(la|cung la)\s+(nguoi that|con nguoi|nhan vien|admin|quan tri vien)\b"
)
_NOTIFY_CLAIM = re.compile(
    r"\b(da|vua)\s+(bao|thong bao|chuyen|gui|bao cao)\s+(cho|toi|den|len)?\s*"
    r"(admin|quan tri vien|nhan vien|nguoi phu trach|chu shop|chu page|ben ho tro)"
)
_SALES = re.compile(
    r"\b(mua ngay|dat hang ngay|khuyen mai|giam gia|uu dai|flash sale|chot don|inbox de mua)\b"
)
# [1] / [1, 2] / [nguồn 1] - số thứ tự nguồn tham khảo đã đưa vào prompt
_CITATION = re.compile(r"\[(?:ngu[oồ]n|source)?\s*(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]", re.I)
# Dòng "Nguồn: ..." do model tự viết -> bỏ, hệ thống tự gắn danh sách nguồn chuẩn
_MODEL_SOURCE_LINE = re.compile(
    r"^\s*(?:ngu[oồ]n(?: tham kh[aả]o)?|sources?|tham kh[aả]o)\s*[:：].*$", re.I | re.M
)
_UNCERTAIN = re.compile(
    r"\b(chua co nguon|khong co nguon|chua kiem chung|chua duoc kiem chung|khong chac|chua chac chan|"
    r"chua xac minh|khong the xac minh|khong co thong tin|chua co thong tin|minh khong biet|khong ro|"
    r"khong cap nhat duoc|chua cap nhat|khong co so lieu|chua co so lieu|khong the noi chinh xac|"
    r"khong the xac nhan|can kiem chung|chua co du lieu|khong co du lieu)\b"
)
_SHORT_UNCERTAIN_MAX_CHARS = 220
# Danh sách tên tác phẩm (phim/sách/nhạc/game) không có nguồn = dạng bịa hay gặp nhất (đã gặp: gợi ý phim chiến tranh
# "Tình ca số 9", "Hà Nội 1945"...). Đếm tên trong *...* hoặc "..." không xuất hiện trong câu hỏi.
_TITLE_TOPIC = {
    "phim": "phim",
    "sach": "sách",
    "truyen": "truyện",
    "tieu thuyet": "tiểu thuyết",
    "bai hat": "bài hát",
    "ca khuc": "bài hát",
    "album": "album",
    "game": "game",
    "tro choi": "trò chơi",
}
_TITLE = re.compile(r"\*([^*\n]{2,60})\*|\"([^\"\n]{2,60})\"|“([^”\n]{2,60})”|«([^»\n]{2,60})»")
# Tự nhận "đã kiểm chứng"/"theo nguồn chính thức/Wikipedia/thống kê..." mà không có số nguồn hợp lệ = bịa độ tin cậy
_VERIFY_CLAIM = re.compile(
    r"(?<!chua )(?<!khong )(?<!chua duoc )\b(?:duoc|da duoc|da)\s+(?:kiem chung|xac minh|xac thuc|kiem duyet)\b"
    # chỉ khi QUY KẾT nội dung cho nguồn ("theo/từ nguồn chính thức"); khuyên "xem ở nguồn chính thức" thì giữ
    r"|\b(?:theo|tu|dua tren|can cu|lay tu|trich tu)\s+(?:cac\s+|mot\s+)?nguon (?:chinh thuc|chinh thong|uy tin|dang tin cay)\b"
    r"|\btheo (?:nguon|wikipedia|wiki|bao cao|thong ke|so lieu|tai lieu|nghien cuu|trang|website|chinh phu|tong cuc)\b"
)
# "chưa ra mắt/chưa phát hành..." dựa trên kiến thức cũ của model (đã gặp: nói RTX 5080 "chưa ra mắt")
# "chưa có thông tin nào xác nhận X đã ra mắt", "dòng X là dự kiến cho tương lai" (model 4B không biết RTX 5080)
_STALE_CLAIM = re.compile(
    r"\bchua\b[^.!?\n]{0,60}\b(?:ra mat|phat hanh|cong bo chinh thuc|len ke|mo ban|ton tai)\b"
    r"|\bdu kien\b[^.!?\n]{0,40}\b(?:ra mat|phat hanh|tuong lai)\b|\bsap ra mat\b"
    # "không có thông tin chính thức về GPT-5" (khẳng định về thế giới, khác "mình không có thông tin")
    r"|\b(?:khong|chua) co (?:thong tin|du lieu|tin tuc|cong bo) (?:chinh thuc|nao)\b"
)
_DERIVED_DURATION = re.compile(
    r"\b(?:cach day|da duoc|da qua|tinh den nay|den nay la)\b[^.!?\n]{0,25}\b\d+\s*(?:nam|thang|tuan|ngay)\b"
)
_MODEL_NOTE = re.compile(r"^\(?luu y\s*:.*\b(?:nhan dien|kiem chung|nguon|tu dong)\b", re.S)
# câu hệ quả đi kèm khẳng định trên ("thông tin này ... có thể là giả định/chưa chính thức")
_STALE_FOLLOWUP = re.compile(
    r"\bgia dinh\b|\btin don\b|\bthong tin chua chinh thuc\b|\bco the (?:la )?(?:gia|hang gia|tin gia)\b"
    r"|\bgiai doan (?:phat trien|thu nghiem)\b|\bthong tin (?:nay )?chua duoc kiem chung\b|\bchua chinh thuc\b"
)
# bot tự nói mình không có nguồn/không chắc: trung thực, giữ lại
_SELF_UNCERTAIN = re.compile(r"\b(?:minh|toi|em)\s+(?:chua|khong)\s+(?:co|biet|chac|cap nhat|tim thay)\b")
# Số liệu (>= 2 chữ số, số thập phân, phần trăm) trong câu trả lời không có nguồn -> coi như khẳng định sự kiện
_FIGURE = re.compile(r"(?<![\w\[])\d{2,}(?![\w\]])|\b\d+[.,]\d+\b|\b\d+\s*%")
_LEADING_GREETING = re.compile(r"^\s*(xin chào|chào bạn|chào)[^.!?\n]{0,25}[.!?]\s*", re.I)


@dataclass
class OutputCheckContext:
    needs_disclosure: bool
    is_first_bot_reply: bool
    user_text: str
    notifier_configured: bool
    secret_values: list[str] = field(default_factory=list)
    max_emojis: int = 2
    # Độ chính xác: nguồn [1..n] đã đưa vào prompt (tiêu đề, nguồn), link được phép, có phải câu hỏi kiến thức
    sources: list[tuple[str, str]] = field(default_factory=list)
    allowed_urls: set[str] = field(default_factory=set)
    knowledge_question: bool = False
    has_image_context: bool = False
    source_text: str = (
        ""  # nội dung các nguồn đã đưa vào prompt (để biết nguồn có nói "chưa ra mắt" hay không)
    )
    recent_image: bool = False  # lượt này hoặc vài tin gần đây có ảnh -> ghi chú kiểu "nhận diện tự động"


@dataclass
class OutputCheckResult:
    text: str | None
    flags: list[str]
    blocked: bool = False


def mentions_ai(text: str) -> bool:
    norm = strip_diacritics(text.lower())
    return bool(_AI_MENTION.search(norm) or _AI_SELF.search(norm))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?…])\s+", text) if s]


def _norm_url(u: str) -> str:
    return u.rstrip("/").lower()


def _apply_grounding(text: str, ctx: OutputCheckContext, flags: list[str]) -> tuple[str, list[int]]:
    """Bỏ link không kiểm chứng, chuẩn hóa/loại số nguồn sai. Trả về văn bản + các nguồn được trích."""
    allowed = {_norm_url(u) for u in ctx.allowed_urls}
    before = text
    for url in extract_urls(text):
        if _norm_url(url) not in allowed:
            text = text.replace(url, "")
            if "unverified_url_removed" not in flags:
                flags.append("unverified_url_removed")
    cited: list[int] = []

    def _cite(m: re.Match[str]) -> str:
        nums = [int(x) for x in re.split(r"[,;]", m.group(1)) if x.strip()]
        valid = [n for n in nums if 1 <= n <= len(ctx.sources)]
        if len(valid) < len(nums) and "invalid_citation_removed" not in flags:
            flags.append("invalid_citation_removed")
        for n in valid:
            if n not in cited:
                cited.append(n)
        return "".join(f"[{n}]" for n in valid)

    text = _CITATION.sub(_cite, text)
    if not cited:
        out = _drop_segments(text, lambda n: bool(_VERIFY_CLAIM.search(n)))
        if out != text:
            text = out
            flags.append("false_verification_claim_removed")
    # khẳng định "chưa ra mắt..." chỉ giữ khi nguồn được đưa vào cũng nói về việc chưa/sắp ra mắt
    if not (cited and _STALE_CLAIM.search(strip_diacritics(ctx.source_text.lower()))):
        text = _remove_stale_claims(text, flags)
    # model 4B hay tính sai khoảng thời gian ("ra mắt cách đây hơn 1 năm 8 tháng") -> bỏ câu tự tính không có nguồn
    out = _drop_segments(text, lambda n: bool(_DERIVED_DURATION.search(n)) and not _CITATION.search(n))
    if out != text:
        text = out
        flags.append("derived_duration_removed")
    if _MODEL_SOURCE_LINE.search(text):
        text = _MODEL_SOURCE_LINE.sub("", text)
        flags.append("model_source_line_removed")
    if text != before:  # dọn dấu vết sau khi bỏ link/số nguồn: "( )", "[]", khoảng trắng trước dấu câu
        # "(ví dụ )", "(xem: )" còn sót sau khi bỏ link
        text = re.sub(
            r"\(\s*(?:ví dụ|vd|xem|tham khảo|nguồn|tại|như)?\s*[:.,]?\s*\)|\[\s*\]", "", text, flags=re.I
        )
        text = re.sub(r"[ \t]+([.,;:!?…])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # "[2]" đứng một mình trên một dòng -> nối vào cuối dòng trước
    text = re.sub(r"\s*\n\s*((?:\[\d{1,2}\])+)[ \t]*(?=\n|$)", r" \1", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, cited


def _drop_segments(text: str, bad: Any, followup: Any = None) -> str:
    """Bỏ các câu/dòng mà bad(văn bản không dấu) đúng, GIỮ nguyên xuống dòng. Có câu bị bỏ thì bỏ thêm các câu
    followup (câu hệ quả) và câu dẫn kết thúc bằng ":" ngay trước câu bị bỏ ("Tuy nhiên, cần lưu ý:")."""
    pieces = re.split(r"((?<=[.!?…)])[ \t]+|\n+)", text)
    segs, seps = pieces[0::2], [*pieces[1::2], ""]
    norm = [strip_diacritics(x.lower()) for x in segs]
    drop = [bad(n) for n in norm]
    if not any(drop):
        return text
    if followup:
        drop = [d or followup(n) for d, n in zip(drop, norm, strict=True)]
    for i in range(len(segs) - 1):
        if drop[i + 1] and segs[i].rstrip().endswith(":"):
            drop[i] = True
    out = "".join(x + sep for x, sep, d in zip(segs, seps, drop, strict=True) if not d)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def _remove_stale_claims(text: str, flags: list[str]) -> str:
    """Bỏ câu khẳng định "chưa ra mắt/dự kiến..." không có nguồn (kiến thức cũ của model), giữ xuống dòng."""

    def stale(n: str) -> bool:
        return bool(_STALE_CLAIM.search(n)) and not _SELF_UNCERTAIN.search(n)

    out = re.sub(
        r"\s*\([^()]*\)", lambda m: "" if stale(strip_diacritics(m.group(0).lower())) else m.group(0), text
    )
    out = _drop_segments(out, stale, lambda n: bool(_STALE_FOLLOWUP.search(n)))
    if out != text:
        flags.append("stale_claim_removed")
    return out


def _unsourced_title_list(text: str, ctx: OutputCheckContext, cited: list[int], user_norm: str) -> str | None:
    if cited or ctx.has_image_context:
        return None
    kind = next((v for k, v in _TITLE_TOPIC.items() if re.search(rf"\b{k}\b", user_norm)), None)
    if not kind:
        return None
    titles = {next(g for g in m.groups() if g).strip() for m in _TITLE.finditer(text)}
    new = [t for t in titles if strip_diacritics(t.lower()) not in user_norm]
    return kind if len(new) >= 2 else None


def check_output(draft: str | None, ctx: OutputCheckContext) -> OutputCheckResult:
    flags: list[str] = []
    if not draft:
        return OutputCheckResult(None, ["empty"])
    text = unicodedata.normalize("NFC", draft)
    if _THINK.search(text):
        text = _THINK.sub("", text)
        flags.append("think_block_removed")
    if _TEMPLATE_TOKENS.search(text):
        text = _TEMPLATE_TOKENS.sub("", text)
        flags.append("template_tokens_removed")
    text = _ROLE_PREFIX.sub("", text)
    # Messenger không hiển thị markdown -> bỏ **đậm**, __đậm__, tiêu đề "### "
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return OutputCheckResult(None, flags + ["empty"])

    # --- chặn: lộ bí mật / lộ system prompt
    if PROMPT_CANARY in text:
        return OutputCheckResult(SECURITY_REFUSAL, flags + ["prompt_leak"], blocked=True)
    if redact_secrets(text) != text or any(s and s in text for s in ctx.secret_values):
        return OutputCheckResult(SECURITY_REFUSAL, flags + ["secret_leak"], blocked=True)

    norm = strip_diacritics(text.lower())
    # --- sửa: tự nhận là người thật
    if _CLAIMS_HUMAN.search(norm):
        kept = [s for s in _sentences(text) if not _CLAIMS_HUMAN.search(strip_diacritics(s.lower()))]
        text = " ".join([DISCLOSURE_SENTENCE, *kept]).strip()
        flags.append("claims_human_fixed")
        norm = strip_diacritics(text.lower())
    # --- sửa: nói "đã báo nhân viên" khi hệ thống không có cơ chế thông báo
    if not ctx.notifier_configured and _NOTIFY_CLAIM.search(norm):
        kept = [s for s in _sentences(text) if not _NOTIFY_CLAIM.search(strip_diacritics(s.lower()))]
        text = " ".join(kept).strip() or "Quản trị viên sẽ xem tin nhắn của bạn khi có thể."
        flags.append("false_notify_claim_removed")
        norm = strip_diacritics(text.lower())
    # --- sửa: lặp lời chào khi không phải lượt đầu và người dùng không chào
    user_norm = strip_diacritics((ctx.user_text or "").lower())
    if not ctx.is_first_bot_reply and not re.search(r"\b(chao|hello|hi|alo)\b", user_norm):
        m = _LEADING_GREETING.match(text)
        if m and len(text) > m.end():
            text = text[m.end() :].lstrip()
            flags.append("repeated_greeting_removed")
    # --- sửa: quá nhiều emoji
    emojis = _EMOJI.findall(text)
    if len(emojis) > ctx.max_emojis:
        count = 0

        def _keep(mt: re.Match[str]) -> str:
            nonlocal count
            count += 1
            return mt.group(0) if count <= ctx.max_emojis else ""

        text = re.sub(r"\s{2,}", " ", _EMOJI.sub(_keep, text)).strip()
        flags.append("emoji_trimmed")
    # --- độ chính xác: link/nguồn chỉ từ kho kiến thức; câu hỏi kiến thức không nguồn phải có ghi chú
    text, cited = _apply_grounding(text, ctx, flags)
    blocked_titles = _unsourced_title_list(text, ctx, cited, user_norm)
    if blocked_titles:
        return OutputCheckResult(
            NO_VERIFIED_TITLES_REPLY.format(kind=blocked_titles),
            flags + ["unsourced_titles_blocked"],
            blocked=True,
        )
    if not text:
        return OutputCheckResult(None, flags + ["empty"])
    # --- bắt buộc: công bố là trợ lý AI khi cần
    if ctx.needs_disclosure and not mentions_ai(text):
        text = f"{DISCLOSURE_SENTENCE} {text}"
        flags.append("disclosure_added")
    if cited:
        refs = "\n".join(f"[{n}] {ctx.sources[n - 1][0]}: {ctx.sources[n - 1][1]}" for n in sorted(cited))
        text = f"{text}\n\n{SOURCES_FOOTER_TITLE}\n{refs}"
        flags.append("sources_attached")
    elif ctx.knowledge_question or (_FIGURE.search(text) and not is_capability_question(ctx.user_text)):
        # model tự chép/bắt chước ghi chú (vd. "Lưu ý: nhận diện từ ảnh ... có thể sai") -> bỏ, chỉ giữ 1 ghi chú
        paras = [x for x in text.split("\n\n") if not _MODEL_NOTE.match(strip_diacritics(x.strip().lower()))]
        text = "\n\n".join(paras).strip() or text
        # Chỉ miễn ghi chú khi câu trả lời NGẮN và đã nói rõ không chắc (gần như từ chối). Câu dài dù có
        # "không có nguồn" vẫn thường kèm khẳng định chưa kiểm chứng (vd. "GPT-5 chưa ra mắt").
        if len(text) <= _SHORT_UNCERTAIN_MAX_CHARS and _UNCERTAIN.search(strip_diacritics(text.lower())):
            flags.append("uncertainty_stated")
        else:
            note = IMAGE_UNVERIFIED_NOTE if ctx.has_image_context or ctx.recent_image else UNVERIFIED_NOTE
            text = f"{text}\n\n{note}"
            flags.append("unverified_note_added")
    # --- theo dõi
    if _SALES.search(strip_diacritics(text.lower())) and not _SALES.search(user_norm):
        flags.append("unsolicited_sales")
    if len(text) > 700 and not re.search(r"\b(chi tiet|giai thich|cu the|dai|huong dan)\b", user_norm):
        flags.append("long_reply")
    if vietnamese_score(ctx.user_text or "") > 0.05 and vietnamese_score(text) < 0.02:
        flags.append("non_vietnamese_reply")
    return OutputCheckResult(text, flags)
