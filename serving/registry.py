"""Registry adapter (tệp JSON, ghi nguyên tử + khóa tệp): đăng ký, promote có cổng đánh giá, rollback.

Không có adapter nào tự động được triển khai: promote phải được gọi thủ công và mặc định yêu cầu
báo cáo đánh giá đã PASS so với baseline.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RegistryError(RuntimeError):
    pass


def _empty() -> dict[str, Any]:
    return {"active": None, "adapters": {}, "history": []}


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else _empty()
        yield data
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".registry.", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else _empty()


def register(path: Path, adapter_dir: Path) -> str:
    meta_file = adapter_dir / "training_meta.json"
    if not meta_file.exists():
        raise RegistryError(f"Không thấy {meta_file}")
    meta = json.loads(meta_file.read_text())
    adapter_id = meta["adapter_id"]
    with _locked(path) as reg:
        reg["adapters"][adapter_id] = {
            "path": str(adapter_dir),
            "method": meta.get("method"),
            "base_model": meta.get("base_model"),
            "base_revision": meta.get("base_revision"),
            "dataset_version": meta.get("dataset_version"),
            "created_at": meta.get("finished_at"),
            "status": "candidate",
            "eval_report": None,
        }
    return adapter_id


def attach_eval(path: Path, adapter_id: str, report_path: Path) -> None:
    with _locked(path) as reg:
        if adapter_id not in reg["adapters"]:
            raise RegistryError(f"adapter {adapter_id} chưa đăng ký")
        reg["adapters"][adapter_id]["eval_report"] = str(report_path)


def promote(path: Path, adapter_id: str, *, by: str, reason: str, force: bool = False) -> dict[str, Any]:
    with _locked(path) as reg:
        ad = reg["adapters"].get(adapter_id)
        if ad is None:
            raise RegistryError(f"adapter {adapter_id} chưa đăng ký")
        gate: dict[str, Any] = {}
        if ad.get("eval_report") and Path(ad["eval_report"]).exists():
            gate = json.loads(Path(ad["eval_report"]).read_text()).get("gate", {})
        if not gate.get("passed") and not force:
            raise RegistryError(
                "Adapter chưa qua cổng đánh giá (gate.passed != true). Chạy đánh giá so với baseline trước; "
                "chỉ dùng --force khi có lý do rõ ràng."
            )
        previous = reg.get("active")
        if previous and previous in reg["adapters"]:
            reg["adapters"][previous]["status"] = "retired"
        ad["status"] = "active"
        reg["active"] = adapter_id
        entry = {
            "event": "promote",
            "adapter_id": adapter_id,
            "previous": previous,
            "at": datetime.now(UTC).isoformat(),
            "by": by,
            "reason": reason,
            "forced": force,
            "gate": gate,
        }
        reg["history"].append(entry)
        return entry


def rollback(path: Path, *, by: str, reason: str) -> dict[str, Any]:
    """Quay về adapter active trước đó (hoặc base model nếu không có)."""
    with _locked(path) as reg:
        current = reg.get("active")
        previous = None
        for h in reversed(reg["history"]):
            if h["event"] == "promote" and h["adapter_id"] == current:
                previous = h.get("previous")
                break
        if current and current in reg["adapters"]:
            reg["adapters"][current]["status"] = "rolled_back"
        if previous and previous in reg["adapters"]:
            reg["adapters"][previous]["status"] = "active"
        reg["active"] = previous
        entry = {
            "event": "rollback",
            "from": current,
            "to": previous,
            "at": datetime.now(UTC).isoformat(),
            "by": by,
            "reason": reason,
        }
        reg["history"].append(entry)
        return entry


def active_adapter(path: Path) -> tuple[str | None, str | None]:
    reg = load(path)
    aid = reg.get("active")
    if not aid:
        return None, None
    return aid, reg["adapters"][aid]["path"]
