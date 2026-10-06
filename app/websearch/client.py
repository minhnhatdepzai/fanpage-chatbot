"""Brave LLM Context client: lấy đoạn văn đã trích xuất và URL để grounding câu trả lời.

Không tải trực tiếp URL kết quả. Query được che PII trước khi gửi; nội dung web luôn được coi là dữ liệu không tin cậy
trong prompt và không thể tự thay đổi cấu hình hay gọi công cụ khác.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from app.config import Settings
from app.conversation.knowledge import KnowledgeEntry
from app.observability.redaction import redact_pii, strip_diacritics

log = logging.getLogger(__name__)
BRAVE_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"
WIKIPEDIA_API_URLS = {
    "vi": "https://vi.wikipedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}
MAX_ENTRY_CHARS = 2400
_EXCERPT_STOPWORDS = {
    "phan",
    "tich",
    "truyen",
    "tac",
    "pham",
    "van",
    "hoc",
    "tai",
    "sao",
    "nhu",
    "vay",
    "cua",
    "trong",
    "cho",
    "voi",
    "mot",
    "nhung",
    "the",
    "nao",
}
_FALLBACK_STOPWORDS = _EXCERPT_STOPWORDS | {
    "viet",
    "bai",
    "nghi",
    "luan",
    "tong",
    "hop",
    "nhieu",
    "nguon",
    "nhan",
    "dinh",
    "giau",
    "hinh",
    "anh",
}


class WebSearchError(RuntimeError):
    pass


class WebSearchProvider(Protocol):
    provider: str

    async def search(self, query: str) -> list[KnowledgeEntry]: ...


def _public_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return value if parsed.scheme in {"http", "https"} and parsed.hostname else None


def _relevant_excerpt(value: str, query: str) -> str:
    """Giữ mở đầu và các đoạn khớp góc hỏi thay vì luôn cắt 2.400 ký tự đầu bài dài."""
    clean = re.sub(r"[ \t]+", " ", value).strip()
    if len(clean) <= MAX_ENTRY_CHARS:
        return clean
    topical_query = re.sub(r'["“”][^"“”]{2,100}["“”]', " ", query)
    terms = {
        word
        for word in re.findall(r"\w+", strip_diacritics(topical_query.lower()))
        if len(word) >= 3 and word not in _EXCERPT_STOPWORDS
    }
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", clean) if part.strip()]
    if not paragraphs:
        return clean[:MAX_ENTRY_CHARS]
    scored: list[tuple[int, int, str]] = []
    for index, paragraph in enumerate(paragraphs):
        normalized = strip_diacritics(paragraph.lower())
        score = sum(min(3, normalized.count(term)) for term in terms)
        scored.append((score, index, paragraph))
    selected = [paragraphs[0]]
    for score, _, paragraph in sorted(scored[1:], key=lambda item: (-item[0], item[1])):
        if score <= 0 or paragraph in selected:
            continue
        selected.append(paragraph)
        if sum(len(part) for part in selected) >= MAX_ENTRY_CHARS:
            break
    return "\n\n".join(selected)[:MAX_ENTRY_CHARS]


class BraveWebSearch:
    provider = "brave_llm_context"

    def __init__(self, settings: Settings) -> None:
        self._key = settings.brave_search_api_key.get_secret_value()
        self._timeout = settings.web_search_timeout_seconds
        self._max_results = settings.web_search_max_results
        self._max_tokens = settings.web_search_max_context_tokens
        self._country = settings.web_search_country
        self._language = settings.web_search_language

    async def search(self, query: str) -> list[KnowledgeEntry]:
        safe_query = redact_pii(query).strip()[:600]
        if not safe_query:
            return []
        payload = {
            "q": safe_query,
            "country": self._country,
            "search_lang": self._language,
            "count": max(5, self._max_results),
            "maximum_number_of_urls": self._max_results,
            "maximum_number_of_tokens": min(32768, max(1024, self._max_tokens)),
            "maximum_number_of_tokens_per_url": 1024,
            "context_threshold_mode": "balanced",
            "safesearch": "moderate",
            "enable_source_metadata": True,
        }
        headers = {"X-Subscription-Token": self._key, "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(BRAVE_CONTEXT_URL, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise WebSearchError(type(exc).__name__) from exc
        if response.status_code != 200:
            raise WebSearchError(f"HTTP {response.status_code}")
        try:
            body: dict[str, Any] = response.json()
        except ValueError as exc:
            raise WebSearchError("bad_json") from exc

        items = (body.get("grounding") or {}).get("generic") or []
        entries: list[KnowledgeEntry] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            url = _public_url(item.get("url"))
            if not url or url in seen:
                continue
            snippets = [redact_pii(str(s)).strip() for s in (item.get("snippets") or []) if str(s).strip()]
            content = "\n".join(snippets)[:MAX_ENTRY_CHARS]
            if not content:
                continue
            title = str(item.get("title") or (body.get("sources") or {}).get(url, {}).get("title") or url)
            digest = hashlib.sha256(url.encode()).hexdigest()[:16]
            entries.append(KnowledgeEntry(f"web:{digest}", title[:240], url, content))
            seen.add(url)
            if len(entries) >= self._max_results:
                break
        return entries


class WikipediaWebSearch:
    """Fallback không cần key cho kiến thức phổ thông; không được quảng bá là nguồn tin mới nhất."""

    provider = "wikipedia"

    def __init__(self, settings: Settings) -> None:
        self._timeout = settings.web_search_timeout_seconds
        self._max_results = settings.web_search_max_results
        self._language = (
            settings.web_search_language if settings.web_search_language in {"vi", "en"} else "vi"
        )

    async def search(self, query: str) -> list[KnowledgeEntry]:
        safe_query = redact_pii(query).strip()[:600]
        if not safe_query:
            return []
        quoted = re.search(r'["“”]([^"“”]{2,100})["“”]', safe_query)
        search_query = quoted.group(1).strip() if quoted else safe_query
        params: dict[str, object] = {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": "1",
            "inprop": "url",
            "format": "json",
            "formatversion": "2",
            "utf8": "1",
        }
        if quoted:
            # Tên tác phẩm đã được router rút ra và đặt trong ngoặc kép: lấy đúng trang theo tiêu đề, tránh fuzzy
            # search trả một người/tác phẩm gần chữ nhưng hoàn toàn lạc đề.
            params.update({"titles": search_query, "redirects": "1"})
        else:
            params.update(
                {
                    "generator": "search",
                    "gsrsearch": search_query,
                    "gsrlimit": self._max_results,
                }
            )
        headers = {"User-Agent": "fanpage-chatbot/0.1 (citable knowledge retrieval)"}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(
                    WIKIPEDIA_API_URLS[self._language], params=params, headers=headers
                )
        except httpx.HTTPError as exc:
            raise WebSearchError(type(exc).__name__) from exc
        if response.status_code != 200:
            raise WebSearchError(f"HTTP {response.status_code}")
        try:
            pages = (response.json().get("query") or {}).get("pages") or []
        except (ValueError, AttributeError) as exc:
            raise WebSearchError("bad_json") from exc
        entries = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            url = _public_url(page.get("fullurl"))
            extract = _relevant_excerpt(redact_pii(str(page.get("extract") or "")), safe_query)
            if not url or not extract:
                continue
            page_id = str(page.get("pageid") or hashlib.sha256(url.encode()).hexdigest()[:16])
            entries.append(
                KnowledgeEntry(
                    f"web:wikipedia:{page_id}",
                    str(page.get("title") or url)[:240],
                    url,
                    (
                        "Nguồn bách khoa Wikipedia; kiểm tra nguồn chính thức nếu câu hỏi cần dữ liệu mới. "
                        + extract
                    )[:MAX_ENTRY_CHARS],
                )
            )
        return entries[: self._max_results]


class DDGSWebSearch:
    """Metasearch web rộng không cần key; chạy thư viện đồng bộ trong thread riêng."""

    provider = "ddgs_metasearch"

    def __init__(self, settings: Settings) -> None:
        self._timeout = settings.web_search_timeout_seconds
        self._max_results = settings.web_search_max_results
        self._region = "vn-vi" if settings.web_search_language == "vi" else "us-en"

    def _search_sync(self, query: str) -> list[dict[str, str]]:
        from ddgs import DDGS

        return list(
            DDGS(timeout=max(1, int(self._timeout))).text(
                query,
                region=self._region,
                safesearch="moderate",
                max_results=self._max_results,
                backend="auto",
            )
        )

    async def search(self, query: str) -> list[KnowledgeEntry]:
        safe_query = redact_pii(query).strip()[:600]
        if not safe_query:
            return []
        try:
            items = await asyncio.wait_for(
                asyncio.to_thread(self._search_sync, safe_query),
                timeout=self._timeout + 2,
            )
        except Exception as exc:  # thư viện gom nhiều backend và có thể ném nhiều loại lỗi mạng
            raise WebSearchError(type(exc).__name__) from exc
        entries: list[KnowledgeEntry] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            url = _public_url(item.get("href") or item.get("url"))
            if not url or url in seen:
                continue
            content = re.sub(
                r"\s+", " ", redact_pii(str(item.get("body") or item.get("snippet") or ""))
            ).strip()
            if not content:
                continue
            digest = hashlib.sha256(url.encode()).hexdigest()[:16]
            entries.append(
                KnowledgeEntry(
                    f"web:ddgs:{digest}",
                    str(item.get("title") or url)[:240],
                    url,
                    content[:MAX_ENTRY_CHARS],
                )
            )
            seen.add(url)
            if len(entries) >= self._max_results:
                break
        return entries


class FallbackWebSearch:
    """Kết hợp một kết quả bách khoa sát tiêu đề với metasearch; Wiki vẫn là đường lui khi web rộng lỗi."""

    provider = "fallback_chain"

    def __init__(
        self,
        primary: WebSearchProvider,
        fallback: WebSearchProvider | None = None,
        *,
        max_results: int = 4,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._max_results = max_results

    async def search(self, query: str) -> list[KnowledgeEntry]:
        try:
            primary_entries = await self._primary.search(query)
        except WebSearchError as exc:
            log.warning(
                "primary_web_search_failed", extra={"provider": self._primary.provider, "error": str(exc)}
            )
            primary_entries = []
        if self._fallback is None:
            return primary_entries
        try:
            fallback_entries = await self._fallback.search(query)
        except WebSearchError as exc:
            log.warning(
                "fallback_web_search_failed", extra={"provider": self._fallback.provider, "error": str(exc)}
            )
            fallback_entries = []

        # Truy vấn văn học được đặt tên trong dấu ngoặc kép. Ưu tiên trang Wiki có đúng tiêu đề, nhưng giữ các
        # nguồn web khác để model không phụ thuộc vào một trang duy nhất.
        quoted = re.search(r'["“”]([^"“”]{2,100})["“”]', query)
        focus = strip_diacritics(re.sub(r"\s+", " ", quoted.group(1).lower()).strip()) if quoted else ""
        query_terms = {
            term
            for term in re.findall(r"\w+", strip_diacritics(query.lower()))
            if len(term) >= 4 and term not in _FALLBACK_STOPWORDS
        }
        best_fallback: list[KnowledgeEntry] = []
        if fallback_entries:
            if focus:
                best = next(
                    (entry for entry in fallback_entries if focus in strip_diacritics(entry.title.lower())),
                    None,
                )
            else:
                best = next(
                    (
                        entry
                        for entry in fallback_entries
                        if len(query_terms & set(re.findall(r"\w+", strip_diacritics(entry.title.lower()))))
                        >= 2
                    ),
                    None,
                )
            # Fuzzy Wikipedia đôi khi trả một nhân vật hoàn toàn lạc đề. Không chèn chỉ để đủ số lượng.
            best_fallback = [best] if best is not None else []
        combined = [*best_fallback, *primary_entries]
        seen: set[str] = set()
        unique = []
        for entry in combined:
            if entry.source in seen:
                continue
            unique.append(entry)
            seen.add(entry.source)
            if len(unique) >= self._max_results:
                break
        return unique


def build_web_search(settings: Settings) -> WebSearchProvider | None:
    if not settings.web_search_enabled:
        return None
    if settings.brave_search_api_key.get_secret_value():
        return BraveWebSearch(settings)
    wikipedia = WikipediaWebSearch(settings) if settings.web_search_wikipedia_fallback else None
    if settings.web_search_ddgs_fallback:
        log.warning("brave_key_missing_using_ddgs_fallback")
        return FallbackWebSearch(
            DDGSWebSearch(settings),
            wikipedia,
            max_results=settings.web_search_max_results,
        )
    if settings.web_search_wikipedia_fallback:
        log.warning("brave_key_missing_using_wikipedia_fallback")
        return wikipedia
    log.warning("web_search_enabled_but_missing_brave_key")
    return None
