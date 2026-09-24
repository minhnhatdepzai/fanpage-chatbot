"""Thông tin fanpage do quản trị viên cung cấp (nguồn DUY NHẤT để bot nói về fanpage)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FanpageProfile:
    name: str = ""
    description: str = ""
    facts: list[str] = field(default_factory=list)
    contact_hint: str = ""

    def as_prompt_block(self) -> str:
        lines: list[str] = []
        if self.name:
            lines.append(f"- Tên fanpage: {self.name}")
        if self.description:
            lines.append(f"- Giới thiệu: {self.description}")
        lines.extend(f"- {fact}" for fact in self.facts)
        if self.contact_hint:
            lines.append(f"- Liên hệ: {self.contact_hint}")
        return "\n".join(lines)

    @property
    def is_empty(self) -> bool:
        return not (self.name or self.description or self.facts or self.contact_hint)


def load_profile(path: Path) -> FanpageProfile:
    if not path.is_file():
        log.info("fanpage_profile_missing", extra={"path": str(path)})
        return FanpageProfile()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    facts = data.get("facts") or []
    return FanpageProfile(
        name=str(data.get("name") or "").strip(),
        description=str(data.get("description") or "").strip(),
        facts=[str(f).strip() for f in facts if str(f).strip()],
        contact_hint=str(data.get("contact_hint") or "").strip(),
    )
