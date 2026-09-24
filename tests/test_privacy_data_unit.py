"""Redaction (log/trace), chia dataset không rò rỉ, giải nén an toàn, registry adapter."""

from __future__ import annotations

import io
import json
import logging
import stat
import zipfile
from pathlib import Path

import pytest

from app.observability.logging_setup import JsonFormatter
from app.observability.redaction import clear_registered_secrets, redact_pii, redact_secrets, register_secrets
from app.observability.tracing import Observation, _mask
from serving import registry
from training.data.safe_extract import ExtractLimits, UnsafeArchiveError, safe_extract_zip
from training.data.split import group_split, leakage_report


# ------------------------------------------------------------------ redaction
def test_redact_known_secret_and_token_patterns():
    clear_registered_secrets()
    register_secrets(["my-very-secret-value-123"])
    s = ("token EAAB1234567890123456789012345 và KGAT_0123456789abcdef0123 và my-very-secret-value-123 "
         "url?access_token=abc123&x=1 Authorization: Bearer abcdefghijklmnop "
         "postgresql+psycopg://chatbot:pw123456@host/db sha256=" + "a" * 64)
    out = redact_secrets(s)
    for leaked in ("EAAB123", "KGAT_0123", "my-very-secret", "abc123", "abcdefghijklmnop", "pw123456", "a" * 64):
        assert leaked not in out
    clear_registered_secrets()


def test_redact_pii_vietnamese_formats():
    t = "Gọi mình 0912 345 678 hoặc +84912345678, email lan.nguyen@gmail.com, CCCD 012345678901"
    out = redact_pii(t)
    assert "0912" not in out and "84912" not in out and "@gmail" not in out and "012345678901" not in out
    assert "Gọi mình" in out  # nội dung thường giữ nguyên


def test_json_log_formatter_redacts_secrets_in_message_and_extra():
    clear_registered_secrets()
    register_secrets(["page-token-secret-xyz"])
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, "call with page-token-secret-xyz", None, None)
    rec.url = "https://graph.facebook.com/v26.0/me?access_token=EAAB1234567890123456789012345"
    line = JsonFormatter().format(rec)
    assert "page-token-secret-xyz" not in line and "EAAB1234567890" not in line
    assert json.loads(line)["msg"].startswith("call with")
    clear_registered_secrets()


def test_trace_mask_and_metadata_only_mode():
    masked = _mask(data={"q": "sđt 0912345678", "nested": ["mail a@b.com"]})
    assert "0912345678" not in json.dumps(masked, ensure_ascii=False)
    assert "a@b.com" not in json.dumps(masked, ensure_ascii=False)
    # capture_content=False -> nội dung không được gửi
    assert Observation(None, capture_content=False).content("nội dung riêng tư") is None


# ------------------------------------------------------------------ dataset split
def _conv(cid: str, group: str, user: str, bot: str = "trả lời") -> dict:
    return {"conversation_id": cid, "group_id": group, "source": "t", "license": "x", "approved_for_training": True,
            "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": bot}]}


def test_group_split_keeps_groups_together_and_no_leakage():
    rows = []
    for g in range(30):
        rows.append(_conv(f"c{g}a", f"g{g}", f"câu hỏi số {g} về chủ đề riêng biệt {g * 7}"))
        rows.append(_conv(f"c{g}b", f"g{g}", f"câu tiếp theo của nhóm {g} hoàn toàn khác {g * 13}"))
    # gần trùng giữa 2 nhóm khác nhau -> phải bị gộp chung một tập
    rows.append(_conv("dupA", "gx", "Làm thế nào để viết văn hay hơn vậy bạn ơi?"))
    rows.append(_conv("dupB", "gy", "Làm thế nào để viết văn hay hơn vậy bạn ơi!"))
    splits, stats = group_split(rows, (0.7, 0.15, 0.15), seed=1, near_dup_threshold=0.8)
    where = {r["conversation_id"]: name for name, rs in splits.items() for r in rs}
    for g in range(30):
        assert where[f"c{g}a"] == where[f"c{g}b"]
    assert where["dupA"] == where["dupB"]
    assert all(splits[n] for n in splits)
    leaks = leakage_report(splits, 0.8)
    assert leaks["overlapping_conversation_ids"] == 0 and not leaks["near_duplicate_pairs_across_splits"]


# ------------------------------------------------------------------ giải nén an toàn
def _zip(entries: list[tuple[str, bytes, int | None]]) -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            if mode is not None:
                info.external_attr = mode << 16
            zf.writestr(info, data)
    p = Path(__import__("tempfile").mkstemp(suffix=".zip")[1])
    p.write_bytes(buf.getvalue())
    return p


def test_safe_extract_ok(tmp_path):
    files = safe_extract_zip(_zip([("data/train.csv", b"a,b\n1,2\n", None)]), tmp_path / "out")
    assert files[0]["path"] == "data/train.csv" and (tmp_path / "out/data/train.csv").exists()


@pytest.mark.parametrize(
    "entries",
    [
        [("../evil.csv", b"x", None)],
        [("/abs/evil.csv", b"x", None)],
        [("run.py", b"print(1)", None)],
        [("link.csv", b"/etc/passwd", stat.S_IFLNK | 0o777)],
        [("bomb.csv", b"0" * 5_000_000, None)],
    ],
)
def test_safe_extract_rejects_unsafe(tmp_path, entries):
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(_zip(entries), tmp_path / "out", ExtractLimits(max_ratio=100))


# ------------------------------------------------------------------ registry adapter
def _adapter(tmp: Path, aid: str) -> Path:
    d = tmp / aid
    d.mkdir()
    (d / "training_meta.json").write_text(json.dumps({"adapter_id": aid, "method": "lora", "base_model": "b",
                                                        "base_revision": "r", "dataset_version": "v1"}))
    return d


def test_registry_requires_eval_gate_and_supports_rollback(tmp_path):
    reg = tmp_path / "registry.json"
    a1 = registry.register(reg, _adapter(tmp_path, "lora-1"))
    with pytest.raises(registry.RegistryError):
        registry.promote(reg, a1, by="t", reason="thử")
    failing = tmp_path / "eval_fail.json"
    failing.write_text(json.dumps({"gate": {"passed": False}}))
    registry.attach_eval(reg, a1, failing)
    with pytest.raises(registry.RegistryError):
        registry.promote(reg, a1, by="t", reason="thử")
    passing = tmp_path / "eval_pass.json"
    passing.write_text(json.dumps({"gate": {"passed": True}}))
    registry.attach_eval(reg, a1, passing)
    registry.promote(reg, a1, by="t", reason="qua cổng")
    a2 = registry.register(reg, _adapter(tmp_path, "lora-2"))
    registry.attach_eval(reg, a2, passing)
    registry.promote(reg, a2, by="t", reason="tốt hơn")
    assert registry.active_adapter(reg)[0] == "lora-2"
    registry.rollback(reg, by="t", reason="lỗi thực tế")
    assert registry.active_adapter(reg)[0] == "lora-1"
    registry.rollback(reg, by="t", reason="về base")
    assert registry.active_adapter(reg)[0] is None
