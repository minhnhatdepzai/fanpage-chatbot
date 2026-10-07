"""Kiểm kê nguồn và đánh giá offline cho Math Tutor.

Mô-đun chỉ lưu metadata URL/tiêu đề khi kiểm kê VietJack. Nội dung câu hỏi chỉ
được đưa vào tập đánh giá sau khi có người duyệt đáp án tham chiếu.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import yaml

from app.math_tutor import MathTutorError, solve_math

ALLOWED_SOURCE_HOSTS = {"vietjack.com", "www.vietjack.com", "khoahoc.vietjack.com"}
USER_AGENT = "StudyScopeMathAudit/1.0 (metadata inventory; educational quality assurance)"


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower()).replace("đ", "d")
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _canonical_url(base_url: str, href: str) -> str | None:
    parsed = urlparse(urljoin(base_url, href.strip()))
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
        return None
    path = re.sub(r"/{2,}", "/", parsed.path)
    return urlunparse(("https", parsed.netloc.lower(), path, "", parsed.query, ""))


def _matches_grade(url: str, title: str, grade: int) -> bool:
    value = _fold(f"{url} {title}")
    return bool(re.search(rf"(?:lop|toan)[\s_/-]*{grade}(?!\d)", value))


def _source_kind(url: str, title: str) -> str:
    folded = _fold(f"{url} {title}")
    if urlparse(url).path.lower().endswith(".pdf"):
        return "school_exam_pdf"
    if "de thi" in folded or "luyen thi" in folded or "kiem tra" in folded:
        return "exam"
    if "bai " in folded or "trang " in folded or "giai-toan" in folded:
        return "lesson"
    return "catalog"


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._href: str | None = None
        self._parts: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a" or self._href is not None:
            return
        href = dict(attrs).get("href")
        if href:
            self._href = href
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            title = " ".join(" ".join(self._parts).split())
            self.links.append((self._href, title))
            self._href = None
            self._parts = []


@dataclass(frozen=True, slots=True)
class SourceSpec:
    id: str
    grade: int
    url: str
    scope: str


@dataclass(frozen=True, slots=True)
class SourceLink:
    url: str
    title: str
    grade: int
    kind: str
    parent_url: str


def load_source_specs(path: Path) -> list[SourceSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs = [SourceSpec(**item) for item in raw.get("sources", [])]
    if not specs or any(not 1 <= spec.grade <= 12 for spec in specs):
        raise ValueError("Danh mục nguồn phải có ít nhất một URL và lớp nằm trong khoảng 1-12.")
    return specs


def discover_links(html: str, base_url: str, grade: int) -> list[SourceLink]:
    parser = _LinkParser()
    parser.feed(html)
    unique: dict[str, SourceLink] = {}
    for href, title in parser.links:
        url = _canonical_url(base_url, href)
        if url is None or "toan" not in _fold(f"{url} {title}") or not _matches_grade(url, title, grade):
            continue
        unique[url] = SourceLink(
            url=url,
            title=title or Path(urlparse(url).path).stem,
            grade=grade,
            kind=_source_kind(url, title),
            parent_url=base_url,
        )
    return sorted(unique.values(), key=lambda item: item.url)


def inventory_sources(
    specs: list[SourceSpec],
    *,
    timeout_seconds: float = 20,
    delay_seconds: float = 0.25,
) -> dict[str, Any]:
    """Tải các trang mục lục được khai báo và lưu metadata liên kết Toán cùng lớp."""
    links: dict[str, SourceLink] = {}
    fetches: list[dict[str, Any]] = []
    headers = {"user-agent": USER_AGENT, "accept": "text/html,application/pdf;q=0.8"}
    with httpx.Client(headers=headers, timeout=timeout_seconds, follow_redirects=True) as client:
        for index, spec in enumerate(specs):
            if index and delay_seconds:
                time.sleep(delay_seconds)
            try:
                response = client.get(spec.url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                fetches.append(
                    {"id": spec.id, "grade": spec.grade, "url": spec.url, "ok": False, "error": str(exc)}
                )
                continue
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            fetches.append(
                {
                    "id": spec.id,
                    "grade": spec.grade,
                    "url": str(response.url),
                    "ok": True,
                    "content_type": content_type,
                    "bytes": len(response.content),
                }
            )
            if content_type != "text/html":
                url = _canonical_url(spec.url, str(response.url))
                if url:
                    links[url] = SourceLink(url, spec.id, spec.grade, _source_kind(url, spec.id), spec.url)
                continue
            for link in discover_links(response.text, str(response.url), spec.grade):
                links[link.url] = link
    counts = Counter(link.kind for link in links.values())
    per_grade = Counter(link.grade for link in links.values())
    return {
        "schema_version": 1,
        "policy": {
            "stored_content": "metadata_only",
            "allowed_hosts": sorted(ALLOWED_SOURCE_HOSTS),
            "note": "Không xem URL/đáp án VietJack là ground truth nếu chưa được người có chuyên môn duyệt.",
        },
        "summary": {
            "seed_sources": len(specs),
            "fetched": sum(bool(item["ok"]) for item in fetches),
            "failed": sum(not item["ok"] for item in fetches),
            "discovered_links": len(links),
            "by_kind": dict(sorted(counts.items())),
            "by_grade": {str(grade): per_grade[grade] for grade in range(1, 13)},
        },
        "fetches": fetches,
        "links": [asdict(link) for link in sorted(links.values(), key=lambda item: (item.grade, item.url))],
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(raw.get("cases", []))


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        checks = {"verified": False, "answer_match": False, "visual_match": False, "steps": False}
        try:
            solved = solve_math(case["question"], grade=int(case["grade"]))
            checks = {
                "verified": solved.get("status") == "verified"
                and solved.get("verification", {}).get("passed") is True,
                "answer_match": solved.get("answer") == str(case["expected_answer"]),
                "visual_match": solved.get("visual", {}).get("type") == case["expected_visual_type"],
                "steps": len(solved.get("steps", [])) >= int(case.get("minimum_steps", 2)),
            }
            actual = {"answer": solved.get("answer"), "visual_type": solved.get("visual", {}).get("type")}
            error = None
        except (MathTutorError, KeyError, TypeError, ValueError) as exc:
            actual = None
            error = str(exc)
        results.append(
            {
                "id": case.get("id"),
                "grade": case.get("grade"),
                "source_url": case.get("source_url"),
                "passed": all(checks.values()),
                "checks": checks,
                "actual": actual,
                "error": error,
            }
        )
    grade_totals = Counter(int(item["grade"]) for item in results)
    grade_passed = Counter(int(item["grade"]) for item in results if item["passed"])
    return {
        "schema_version": 1,
        "summary": {
            "total": len(results),
            "passed": sum(item["passed"] for item in results),
            "failed": sum(not item["passed"] for item in results),
            "by_grade": {
                str(grade): {"passed": grade_passed[grade], "total": grade_totals[grade]}
                for grade in sorted(grade_totals)
            },
        },
        "results": results,
    }


def dump_report(report: dict[str, Any], output: Path | None) -> str:
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return rendered
