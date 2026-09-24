"""Chữ ký webhook, phân tích sự kiện, chia tin nhắn, phân loại lỗi Send API."""

from __future__ import annotations

import json

from app.messenger.client import SendOutcome, classify_graph_error
from app.messenger.events import Echo, Ignored, Postback, UserMessage, WindowOpener, parse_webhook_payload
from app.messenger.signature import compute_signature, verify_signature
from app.messenger.text import split_message, utf16_len
from tests.helpers import SECRET, echo_event, msg_event, payload


# ------------------------------------------------------------------ chữ ký
def test_signature_valid_raw_body():
    body = json.dumps(payload(("PAGE1", [msg_event("u1", "m1", "chào")])), ensure_ascii=False).encode()
    assert verify_signature(SECRET, body, compute_signature(SECRET, body))


def test_signature_valid_when_meta_signs_escaped_unicode():
    """Meta ký trên bản escape unicode (\\u00e0...); raw body có thể chứa UTF-8 thô."""
    obj = payload(("PAGE1", [msg_event("u1", "m1", "Xin chào, mình tên Lan 😊")]))
    escaped = json.dumps(obj, ensure_ascii=True).encode()  # bản Meta dùng để ký
    raw_utf8 = json.dumps(obj, ensure_ascii=False).encode()  # bản nhận được (giả định xấu nhất)
    sig = compute_signature(SECRET, escaped)
    assert verify_signature(SECRET, escaped, sig)
    assert verify_signature(SECRET, raw_utf8, sig)


def test_signature_rejects_wrong_missing_malformed():
    body = b'{"object":"page","entry":[]}'
    good = compute_signature(SECRET, body)
    assert not verify_signature(SECRET, body, None)
    assert not verify_signature(SECRET, body, "")
    assert not verify_signature(SECRET, body, compute_signature("other-secret", body))
    assert not verify_signature(SECRET, body + b" ", good)
    assert not verify_signature(SECRET, body, good.replace("sha256=", "sha1="))
    assert not verify_signature(SECRET, body, "sha256=abc")
    assert not verify_signature("", body, good)


# ------------------------------------------------------------------ sự kiện
def test_parse_multiple_entries_and_events():
    obj = payload(
        ("PAGE1", [msg_event("u1", "m1", "a"), msg_event("u2", "m2", "b")]),
        ("PAGE1", [{"sender": {"id": "u1"}, "recipient": {"id": "PAGE1"}, "timestamp": 1, "delivery": {"mids": ["x"]}}]),
    )
    events = parse_webhook_payload(obj, {"PAGE1"})
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["UserMessage", "UserMessage", "Ignored"]
    assert events[0].psid == "u1" and events[1].psid == "u2"


def test_parse_other_page_is_ignored():
    events = parse_webhook_payload(payload(("OTHER", [msg_event("u1", "m1", "hi", page="OTHER")])), {"PAGE1"})
    assert len(events) == 1 and isinstance(events[0], Ignored) and events[0].reason == "unexpected_page"


def test_parse_echo_postback_reaction_attachments_sticker():
    obj = payload(
        (
            "PAGE1",
            [
                echo_event("u1", "e1", "bot reply", app_id="APP123", metadata="t:0"),
                {"sender": {"id": "u1"}, "recipient": {"id": "PAGE1"}, "timestamp": 1,
                 "postback": {"title": "Bắt đầu", "payload": "GET_STARTED", "mid": "pb1"}},
                {"sender": {"id": "u1"}, "recipient": {"id": "PAGE1"}, "timestamp": 1,
                 "reaction": {"reaction": "love", "action": "react", "mid": "m0"}},
                msg_event("u1", "m3", None, attachments=[{"type": "image", "payload": {"url": "https://x"}}]),
                msg_event("u1", "m4", None, sticker_id=369239263222822,
                          attachments=[{"type": "image", "payload": {"sticker_id": 369239263222822}}]),
            ],
        )
    )
    e = parse_webhook_payload(obj, {"PAGE1"})
    assert isinstance(e[0], Echo) and e[0].psid == "u1" and e[0].metadata == "t:0" and e[0].app_id == "APP123"
    assert isinstance(e[1], Postback) and e[1].payload == "GET_STARTED"
    assert isinstance(e[2], WindowOpener) and e[2].kind == "reaction"
    assert isinstance(e[3], UserMessage) and e[3].text is None and e[3].attachment_types == ["image"]
    assert isinstance(e[4], UserMessage) and e[4].is_like_sticker


def test_text_is_nfc_normalized():
    decomposed = "Tiếng Việt"  # dạng tổ hợp (NFD)
    e = parse_webhook_payload(payload(("PAGE1", [msg_event("u1", "m1", decomposed)])), {"PAGE1"})
    assert e[0].text == "Tiếng Việt"


def test_non_page_object_ignored():
    assert isinstance(parse_webhook_payload({"object": "instagram", "entry": []}, {"PAGE1"})[0], Ignored)


# ------------------------------------------------------------------ chia tin nhắn
def test_split_short_text_single_part():
    assert split_message("Chào bạn!", 1900, 3) == ["Chào bạn!"]


def test_split_long_text_respects_limit_and_max_parts():
    sentence = "Đây là một câu khá dài để kiểm tra việc chia tin nhắn theo giới hạn ký tự. "
    parts = split_message(sentence * 100, 300, 3)
    assert len(parts) == 3
    assert all(utf16_len(p) <= 300 for p in parts)
    assert parts[-1].endswith("…")


def test_split_counts_emoji_as_two_utf16_units():
    text = "😊" * 10
    assert utf16_len(text) == 20
    assert all(utf16_len(p) <= 8 for p in split_message(text, 8, 10))


# ------------------------------------------------------------------ phân loại lỗi Send API
def test_classify_graph_errors():
    assert classify_graph_error(400, {"error": {"code": 10, "error_subcode": 2018278}}).outcome == SendOutcome.window_closed
    assert classify_graph_error(400, {"error": {"code": 190}}).outcome == SendOutcome.auth_error
    assert classify_graph_error(400, {"error": {"code": 613}}).outcome == SendOutcome.rate_limited
    assert classify_graph_error(500, {"error": {"code": 1200}}).outcome == SendOutcome.retryable
    assert classify_graph_error(400, {"error": {"code": 100, "error_subcode": 2018001}}).outcome == SendOutcome.permanent
    assert classify_graph_error(503, None).outcome == SendOutcome.uncertain
    assert classify_graph_error(429, None).outcome == SendOutcome.rate_limited
