"""Tra cứu web có nguồn: contract Brave + gắn vào graph chỉ khi chính sách cho phép."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import SecretStr

from app.config import WebSearchMode, get_settings
from app.conversation.graph import GraphContext, build_graph, run_turn
from app.conversation.knowledge import KnowledgeBase, KnowledgeEntry
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.profile import FanpageProfile
from app.providers.llm import FakeProvider
from app.websearch.client import (
    BRAVE_CONTEXT_URL,
    WIKIPEDIA_API_URLS,
    BraveWebSearch,
    DDGSWebSearch,
    FallbackWebSearch,
    WikipediaWebSearch,
    build_web_search,
)


class StaticSearch:
    provider = "static"

    def __init__(self, entries):  # type: ignore[no-untyped-def]
        self.entries = entries

    async def search(self, query: str):  # type: ignore[no-untyped-def]
        return self.entries


async def test_fallback_drops_irrelevant_fuzzy_wikipedia_result():
    primary = StaticSearch(
        [KnowledgeEntry("web:1", "Phân tích Chí Phèo", "https://example.org/chi-pheo", "Nội dung")]
    )
    wikipedia = StaticSearch(
        [KnowledgeEntry("web:wiki", "Bùi Công Nam", "https://vi.wikipedia.org/wiki/Bui_Cong_Nam", "Lạc đề")]
    )
    provider = FallbackWebSearch(primary, wikipedia, max_results=8)
    entries = await provider.search('"Chí Phèo" phân tích bi kịch')
    assert [entry.id for entry in entries] == ["web:1"]


async def test_fallback_keeps_relevant_wikipedia_result():
    primary = StaticSearch([])
    wikipedia = StaticSearch(
        [KnowledgeEntry("web:wiki", "Chí Phèo", "https://vi.wikipedia.org/wiki/Chi_Pheo", "Nội dung")]
    )
    provider = FallbackWebSearch(primary, wikipedia, max_results=8)
    entries = await provider.search('"Chí Phèo" phân tích bi kịch')
    assert [entry.id for entry in entries] == ["web:wiki"]


async def test_brave_context_response_becomes_citable_entries(http_mock):
    settings = get_settings().model_copy(
        update={
            "brave_search_api_key": SecretStr("test-brave-key"),
            "web_search_max_results": 2,
        }
    )
    route = http_mock.post(BRAVE_CONTEXT_URL).respond(
        200,
        json={
            "grounding": {
                "generic": [
                    {
                        "url": "https://example.org/news",
                        "title": "Tin có nguồn",
                        "snippets": ["Thông tin cập nhật từ nguồn."],
                    }
                ]
            },
            "sources": {},
        },
    )
    entries = await BraveWebSearch(settings).search("tin mới, liên hệ 0912345678")
    assert entries[0].id.startswith("web:") and entries[0].source == "https://example.org/news"
    sent = route.calls[0].request.content.decode()
    assert "0912345678" not in sent


async def test_wikipedia_fallback_returns_citable_entries_without_key(http_mock):
    settings = get_settings().model_copy(
        update={
            "web_search_enabled": True,
            "brave_search_api_key": SecretStr(""),
            "web_search_ddgs_fallback": False,
            "web_search_wikipedia_fallback": True,
            "web_search_language": "vi",
            "web_search_max_results": 2,
        }
    )
    route = http_mock.get(WIKIPEDIA_API_URLS["vi"]).respond(
        200,
        json={
            "query": {
                "pages": [
                    {
                        "pageid": 123,
                        "title": "Phương trình bậc hai",
                        "fullurl": "https://vi.wikipedia.org/wiki/Phuong_trinh_bac_hai",
                        "extract": "Phương trình bậc hai là phương trình đa thức có bậc hai.",
                    }
                ]
            }
        },
    )
    provider = build_web_search(settings)
    assert isinstance(provider, WikipediaWebSearch)
    entries = await provider.search("phương trình bậc hai, gọi 0912345678")
    assert entries[0].id == "web:wikipedia:123"
    assert entries[0].source.startswith("https://vi.wikipedia.org/")
    assert "0912345678" not in route.calls[0].request.url.query.decode()


async def test_wikipedia_uses_exact_title_for_quoted_literary_work(http_mock):
    settings = get_settings().model_copy(update={"web_search_max_results": 8})
    route = http_mock.get(WIKIPEDIA_API_URLS["vi"]).respond(
        200,
        json={
            "query": {
                "pages": [
                    {
                        "pageid": 456,
                        "title": "Chí Phèo",
                        "fullurl": "https://vi.wikipedia.org/wiki/Chi_Pheo",
                        "extract": "Chí Phèo là truyện ngắn của Nam Cao.",
                    }
                ]
            }
        },
    )
    entries = await WikipediaWebSearch(settings).search('"Chí Phèo" phân tích bi kịch')
    query = route.calls[0].request.url.query.decode()
    assert "titles=Ch%C3%AD+Ph%C3%A8o" in query
    assert "generator=search" not in query
    assert entries[0].title == "Chí Phèo"


async def test_ddgs_results_become_citable_entries_and_query_is_redacted(monkeypatch):
    settings = get_settings().model_copy(update={"web_search_max_results": 2})
    provider = DDGSWebSearch(settings)
    seen: list[str] = []

    def fake_search(query: str) -> list[dict[str, str]]:
        seen.append(query)
        return [
            {
                "title": "Chí Phèo — Nam Cao",
                "href": "https://example.edu.vn/chi-pheo",
                "body": "Tác phẩm khắc họa bi kịch bị cự tuyệt quyền làm người.",
            }
        ]

    monkeypatch.setattr(provider, "_search_sync", fake_search)
    entries = await provider.search("Chí Phèo, số điện thoại 0912345678")
    assert entries[0].id.startswith("web:ddgs:")
    assert entries[0].source == "https://example.edu.vn/chi-pheo"
    assert "0912345678" not in seen[0]


class FakeWeb:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str) -> list[KnowledgeEntry]:
        self.queries.append(query)
        return [KnowledgeEntry("web:1", "Nguồn web", "https://example.org/current", "Giá hiện tại là 10.")]


async def test_graph_searches_web_and_allows_only_returned_url():
    settings = get_settings().model_copy(
        update={
            "web_search_enabled": True,
            "web_search_mode": WebSearchMode.always,
            "answer_verification_enabled": False,
        }
    )
    store = InMemoryConversationStore(now=lambda: datetime(2026, 10, 5, tzinfo=UTC))
    provider = FakeProvider(lambda m: "Giá hiện tại là 10 [1]. https://evil.example/x")
    web = FakeWeb()
    ctx = GraphContext(
        settings=settings,
        store=store,
        llm=provider,
        profile=FanpageProfile(name="AI Test"),
        knowledge=KnowledgeBase(),
        web_search=web,
        now=lambda: datetime(2026, 10, 5, tzinfo=UTC),
    )
    cid = store.ensure_conversation("PAGE1", "u1")
    store.add_user_message(cid, "giá hiện tại bao nhiêu?", ts=datetime(2026, 10, 5, tzinfo=UTC))
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)
    answer = "\n".join(result["reply_parts"])
    assert web.queries and "web_searched" in result["check_flags"]
    assert "https://example.org/current" in answer and "evil.example" not in answer
    assert "tra cứu web" in str(provider.calls[-1][0].content)


async def test_named_literary_work_forces_focused_web_search_even_with_local_hit():
    settings = get_settings().model_copy(
        update={
            "web_search_enabled": True,
            "web_search_mode": WebSearchMode.auto,
            "answer_verification_enabled": False,
        }
    )
    store = InMemoryConversationStore(now=lambda: datetime(2026, 10, 5, tzinfo=UTC))
    provider = FakeProvider(
        lambda _: (
            "Dân làng xa lánh Chí vì sợ bạo lực và vì định kiến đã biến anh thành người ngoài cộng đồng [2]."
        )
    )
    web = FakeWeb()
    local = KnowledgeBase(
        [
            KnowledgeEntry(
                "generic-literature",
                "Kiến thức Ngữ văn chung",
                "https://example.edu.vn/ngu-van",
                "Cách xây dựng luận điểm nghị luận văn học.",
                ("truyện",),
            )
        ]
    )
    ctx = GraphContext(
        settings=settings,
        store=store,
        llm=provider,
        profile=FanpageProfile(name="AI Test"),
        knowledge=local,
        web_search=web,
        now=lambda: datetime(2026, 10, 5, tzinfo=UTC),
    )
    cid = store.ensure_conversation("PAGE1", "literature-student")
    store.add_user_message(
        cid,
        "Phân tích góc nhìn của dân làng trong truyện Chí Phèo, tại sao họ thờ ơ và kì thị Chí?",
        ts=datetime(2026, 10, 5, tzinfo=UTC),
    )
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "writing"
    assert '"Chí Phèo"' in web.queries[0]
    assert "web_searched" in result["check_flags"]
    assert "https://example.org/current" in "\n".join(result["reply_parts"])


async def test_social_argument_forces_web_search_and_exposes_all_sources():
    settings = get_settings().model_copy(
        update={
            "web_search_enabled": True,
            "web_search_mode": WebSearchMode.auto,
            "answer_verification_enabled": False,
        }
    )
    store = InMemoryConversationStore(now=lambda: datetime(2026, 10, 5, tzinfo=UTC))
    provider = FakeProvider(lambda _: "Mạng xã hội vừa mở rộng kết nối vừa tạo áp lực so sánh [1].")
    web = FakeWeb()
    ctx = GraphContext(
        settings=settings,
        store=store,
        llm=provider,
        profile=FanpageProfile(name="AI Test"),
        knowledge=KnowledgeBase(),
        web_search=web,
        now=lambda: datetime(2026, 10, 5, tzinfo=UTC),
    )
    cid = store.ensure_conversation("PAGE1", "social-student")
    store.add_user_message(
        cid,
        "Viết bài nghị luận xã hội về tác động của mạng xã hội với học sinh",
        ts=datetime(2026, 10, 5, tzinfo=UTC),
    )
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "writing" and web.queries
    assert "web_searched" in result["check_flags"]
    assert "all_sources_attached" in result["check_flags"]
    assert "https://example.org/current" in "\n".join(result["reply_parts"])
