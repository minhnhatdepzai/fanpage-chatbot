"""Nhận diện Ngữ văn phải chính xác và không làm hỏng định tuyến chat/toán."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.config import get_settings
from app.conversation.graph import GraphContext, build_graph, run_turn
from app.conversation.knowledge import KnowledgeEntry
from app.conversation.memory_store import InMemoryConversationStore
from app.conversation.output_check import OutputCheckContext, check_output
from app.conversation.profile import FanpageProfile
from app.providers.llm import FakeProvider
from app.writing_tutor import (
    build_writing_instruction,
    continue_writing_task,
    detect_writing_task,
    literary_search_query,
    requested_word_count,
)
from app.writing_tutor.validation import (
    parse_constrained_lines,
    repair_contract,
    requests_output_only,
    validate_creative_output,
)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Lập dàn ý nghị luận xã hội lớp 9 về lòng biết ơn", "social_argument"),
        ("Phân tích nhân vật trong truyện này giúp em", "literary_argument"),
        ("viet bai van ta cay phuong lop 4", "creative"),
        ("Tạo một bài thơ haiku phong cách Nhật Bản, tối giản và buồn", "creative"),
        ("Create a sonnet in English", "creative"),
        ("Sửa bài văn này và giữ giọng viết của em", "revision"),
        ("Làm phần đọc hiểu văn bản sau", "reading_comprehension"),
        ("Viết giúp phần phản biện cho bài nghị luận về mạng xã hội", "component"),
        ("Write an essay about friendship", "essay"),
    ],
)
def test_detects_writing_tasks(text: str, kind: str):
    task = detect_writing_task(text)
    assert task.is_writing and task.kind == kind


def test_comparing_user_written_lines_is_a_literature_task():
    task = detect_writing_task("So sánh hai câu tự viết: A 'Gió nâng cánh diều'; B 'Gió khép hiên nhà'.")
    assert task.is_writing and task.kind == "literary_argument"


def test_analyzing_image_in_user_written_sentence_is_literature():
    task = detect_writing_task("Phân tích hình ảnh trong câu tự viết: ‘Bà đặt đôi dép quay mũi ra cửa.’")
    assert task.is_writing and task.kind == "literary_argument"


def test_literary_and_social_arguments_require_live_sources():
    literary = detect_writing_task("Phân tích nhân vật Chí Phèo")
    social = detect_writing_task("Nghị luận xã hội về tác động của mạng xã hội")
    assert literary.requires_sources is True
    assert social.requires_sources is True


def test_named_character_tragedy_without_work_keyword_is_still_literature():
    question = "Phân tích vì sao bi kịch của Chí Phèo không chỉ do cá nhân mà còn do cả làng Vũ Đại"
    task = detect_writing_task(question)
    assert task.kind == "literary_argument" and task.requires_sources
    assert literary_search_query(question).startswith('"Chí Phèo"')


def test_short_length_followup_inherits_literature_and_target_words():
    previous = "Phân tích bài thơ Lượm của Tố Hữu"
    current = "phân tích sâu và chi tiết hơn, dài 2000 chữ đi"
    task = continue_writing_task(previous, current)
    assert task is not None and task.kind == "literary_argument"
    assert task.target_words == 2000 and requested_word_count(current) == 2000


def test_counter_reading_of_legend_is_detected_as_literary_analysis():
    question = (
        "Nhưng tôi nghe một câu chuyện với góc nhìn khác cho rằng Thủy Tinh liên tục đánh Sơn Tinh vì bị "
        "chèn ép; voi chín ngà, ngựa chín hồng mao đều là sản vật trên cạn. Hãy phân tích theo hướng này "
        "và đi vào tâm lý con người."
    )
    task = detect_writing_task(question)
    assert task.is_writing and task.kind == "literary_argument"
    assert task.requires_sources and task.target_words == 2700


def test_generic_2500_word_analysis_and_write_it_followup_keep_literature_mode():
    counter_reading = (
        "Trong truyền thuyết Sơn Tinh - Thủy Tinh, hãy phân tích theo góc nhìn Thủy Tinh bị đối xử bất công "
        "và liên hệ tâm lý con người."
    )
    request = "phân tích nghị luận 2500 chữ đi"
    requested = continue_writing_task(counter_reading, request)
    assert requested is not None and requested.kind == "literary_argument"
    assert requested.target_words == 2500

    standalone = detect_writing_task(request)
    assert standalone.kind == "literary_argument" and standalone.target_words == 2500

    write_now = continue_writing_task(request, "viết luôn cho tôi 1 bài phân tích")
    assert write_now is not None and write_now.kind == "literary_argument"
    assert write_now.target_words == 2500


@pytest.mark.parametrize(
    "text",
    [
        "viết về tình mẫu tử đi",
        "làm cho tôi một bài về lòng biết ơn",
        "viết cho em bài nghị luận về trách nhiệm",
    ],
)
def test_generic_vietnamese_writing_requests_do_not_fall_back_to_normal_chat(text: str):
    task = detect_writing_task(text)
    assert task.is_writing and task.output == "essay"


def test_generic_writing_detection_does_not_capture_programming_request():
    assert not detect_writing_task("viết code Python đọc tệp CSV").is_writing


def test_long_writing_button_command_has_explicit_2800_word_target():
    task = detect_writing_task(
        "Hãy viết ngay một bài văn hoàn chỉnh khoảng 2800 chữ, tối thiểu 2500 chữ. "
        "Không trả lời bằng khung ý. Đề: tình mẫu tử."
    )
    assert task.is_writing and task.output == "essay"
    assert task.target_words == 2800


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Phân tích góc nhìn dân làng trong truyện Chí Phèo, tại sao họ kì thị Chí?",
            '"Chí Phèo" Phân tích góc nhìn dân làng trong truyện Chí Phèo, tại sao họ kì thị Chí?',
        ),
        (
            "Cảm nhận bài thơ Tây Tiến của Quang Dũng",
            '"Tây Tiến" Cảm nhận bài thơ Tây Tiến của Quang Dũng',
        ),
    ],
)
def test_literary_search_query_focuses_on_named_work(question: str, expected: str):
    assert literary_search_query(question) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Bạn có khỏe không?",
        "Phân tích dữ liệu bán hàng bằng Python",
        "Bài viết trên web bị lỗi CSS",
        "Giải phương trình 3x + 5 = 20",
        "AI là gì?",
    ],
)
def test_normal_chat_and_math_are_not_writing(text: str):
    assert not detect_writing_task(text).is_writing


def test_grade_and_copyright_rules_are_in_dynamic_prompt():
    task = detect_writing_task("Phân tích bài thơ này cho lớp 8")
    instruction = build_writing_instruction(task.to_state())
    assert "lớp 8" in instruction
    assert "không cung cấp toàn văn" in instruction
    assert "không trích nguyên văn" in instruction
    assert "bài nghe 3-6 phút" in instruction and "bài giảng/podcast" in instruction
    assert "không từ chối chỉ vì tác phẩm chưa có trong kho nội bộ" in instruction
    assert "Được viết bay bổng, giàu hình ảnh" in instruction
    assert "kỹ thuật tự sự" in instruction and "Độ dài tăng phải đi cùng luận điểm mới" in instruction
    assert "Không từ chối hoặc tự hạ" in instruction
    assert "giả thuyết diễn giải" in instruction and "không dùng cảm giác bị tổn thương" in instruction


def test_explicit_writing_length_is_added_to_prompt():
    task = detect_writing_task("Phân tích bài thơ Lượm dài 2000 chữ")
    instruction = build_writing_instruction(task.to_state())
    assert task.target_words == 2000
    assert "khoảng 2000 chữ" in instruction and "±10%" in instruction


def test_full_literary_analysis_defaults_to_2700_words_and_supports_3000():
    default = detect_writing_task("Phân tích nhân vật Chí Phèo")
    explicit = detect_writing_task("Phân tích nhân vật Chí Phèo khoảng 3000 chữ")
    assert default.target_words == 2700
    assert explicit.target_words == 3000 and requested_word_count("dài 3500 chữ") == 3200
    instruction = build_writing_instruction(default.to_state())
    assert "khoảng 2700 chữ" in instruction
    assert "Ma trận góc nhìn" in instruction and "liên hệ đời sống cụ thể" in instruction
    assert "mono no aware" in instruction and "không phải bản chất cố định" in instruction


def test_short_literary_request_is_not_forced_to_long_form():
    task = detect_writing_task("Phân tích ngắn gọn nhân vật Chí Phèo")
    assert task.target_words == 500


def test_social_argument_prompt_requires_multi_source_weighting():
    task = detect_writing_task("Nghị luận xã hội về tác động của mạng xã hội")
    instruction = build_writing_instruction(task.to_state())
    assert "tổng hợp đa nguồn" in instruction
    assert "hạ trọng số nguồn yếu" in instruction


def test_creative_poetry_has_form_culture_and_user_preferences():
    task = detect_writing_task("Tạo một bài thơ haiku phong cách Nhật Bản, bằng tiếng Việt, tối giản và buồn")
    assert task.creative_requested and task.output == "poem"
    assert task.form == "haiku" and task.tradition == "Nhật Bản"
    assert task.language == "tiếng Việt" and task.tones == ("buồn, trầm", "tối giản")
    instruction = build_writing_instruction(task.to_state())
    assert "được phép và phải tự sáng tạo câu thơ" in instruction
    assert "5-7-5" in instruction and "không hoàn toàn tương đương" in instruction
    assert "định kiến dân tộc" in instruction


def test_analysis_of_existing_poem_is_not_creative_permission():
    task = detect_writing_task("Phân tích bài thơ haiku này giúp mình")
    assert task.kind == "literary_argument" and not task.creative_requested


def test_outline_numbers_do_not_trigger_source_note_but_new_claim_number_does():
    context = OutputCheckContext(
        needs_disclosure=False,
        is_first_bot_reply=False,
        user_text="Lập dàn ý lớp 9 khoảng 180 chữ",
        notifier_configured=False,
        writing_mode=True,
    )
    outline = check_output("1. Mở bài\n2. Thân bài\nBài khoảng 180 chữ cho lớp 9.", context)
    assert "unverified_note_added" not in outline.flags
    claim = check_output("Bạn nên đọc sách 15 phút mỗi ngày.", context)
    assert "unverified_note_added" in claim.flags


def test_original_fiction_is_not_blocked_as_fake_titles_or_unsourced_figures():
    context = OutputCheckContext(
        needs_disclosure=False,
        is_first_bot_reply=False,
        user_text="Viết truyện ngắn mới về một bộ phim chưa ra mắt",
        notifier_configured=False,
        writing_mode=True,
        creative_writing=True,
    )
    draft = (
        "Truyện “Thành phố sau mưa” kể về bộ phim “Ngày thứ 100” chưa ra mắt trong thế giới hư cấu. "
        "Nhân vật đã chờ nó cách đây 20 năm."
    )
    result = check_output(draft, context)
    assert not result.blocked and result.text == draft
    assert "unverified_note_added" not in result.flags
    assert "stale_claim_removed" not in result.flags


def test_grounded_literary_analysis_is_not_mistaken_for_fake_title_list():
    result = check_output(
        'Trong *Chí Phèo*, cách gọi "con quỷ dữ" cho thấy Chí bị xem như "người khác" [1].',
        OutputCheckContext(
            needs_disclosure=False,
            is_first_bot_reply=False,
            user_text="Phân tích truyện Chí Phèo",
            notifier_configured=False,
            sources=[("Chí Phèo", "https://vi.wikipedia.org/wiki/Chi_Pheo")],
            allowed_urls={"https://vi.wikipedia.org/wiki/Chi_Pheo"},
            knowledge_question=True,
            writing_mode=True,
        ),
    )
    assert "unsourced_titles_blocked" not in result.flags
    assert result.text and "https://vi.wikipedia.org/wiki/Chi_Pheo" in result.text


def test_user_supplied_literary_quotes_are_not_mistaken_for_fake_titles():
    result = check_output(
        'Chi tiết "đồng hồ không kim" đối thoại với "tiếng tích tắc" và mở ra một cách đọc xã hội.',
        OutputCheckContext(
            needs_disclosure=False,
            is_first_bot_reply=False,
            user_text='Phân tích đoạn truyện tự viết có "đồng hồ không kim".',
            notifier_configured=False,
            writing_mode=True,
        ),
    )
    assert not result.blocked and "unsourced_titles_blocked" not in result.flags


def test_deterministic_creative_form_validation():
    valid = validate_creative_output(
        "haiku", "mưa rơi qua mái cũ\nlá non run dưới hiên chiều vắng\nđêm dài nghe đất thở"
    )
    assert valid.valid and valid.line_counts == (5, 7, 5)
    invalid = validate_creative_output("haiku", "mưa đầu hạ rơi\nlá rụng trên đường\ngió không ai gọi")
    assert not invalid.valid and invalid.line_counts == (4, 4, 4)
    assert requests_output_only("chi dua bai tho khong giai thich")
    assert "dòng 1 đúng 5" in repair_contract("haiku")
    structured = (
        '{"lines": [["mưa","rơi","qua","mái","cũ"],'
        '["lá","non","run","dưới","hiên","chiều","vắng"],'
        '["đêm","dài","nghe","đất","thở"]]}'
    )
    assert parse_constrained_lines(structured, "haiku") == (
        "mưa rơi qua mái cũ\nlá non run dưới hiên chiều vắng\nđêm dài nghe đất thở"
    )
    slotted = "D1=mưa|rơi|qua|mái|cũ|dư\nD2=lá|non|run|dưới|hiên|chiều|vắng\nD3=đêm|dài|nghe|đất|thở|dư"
    assert parse_constrained_lines(slotted, "haiku") == (
        "mưa rơi qua mái cũ\nlá non run dưới hiên chiều vắng\nđêm dài nghe đất thở"
    )
    assert parse_constrained_lines('{"lines": [["quá ít"]]}', "haiku") is None


def test_output_only_validation_detects_bracketed_commentary():
    result = validate_creative_output(
        None,
        "Mây qua đỉnh núi\nTrăng xuống hiên nhà\n[Chú thích: thơ nguyên bản]",
        strict_output_only=True,
    )
    assert not result.valid and "phần ghi chú" in result.issues[0]


async def test_graph_routes_writing_and_injects_specialized_prompt():
    now = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: "Dàn ý gồm mở bài, thân bài và kết bài.")
    settings = get_settings().model_copy(update={"writing_max_output_tokens": 1337})
    ctx = GraphContext(
        settings=settings,
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "student")
    store.add_user_message(cid, "Lập dàn ý nghị luận xã hội lớp 9 về lòng biết ơn", ts=now)
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "writing"
    assert result["writing_task"]["kind"] == "social_argument"
    system = str(provider.calls[0][0].content)
    assert "CHẾ ĐỘ GIA SƯ NGỮ VĂN" in system and "lớp 9" in system


async def test_graph_inherits_literature_mode_for_longer_followup():
    class LiteratureWeb:
        async def search(self, query: str) -> list[KnowledgeEntry]:
            return [KnowledgeEntry("web:luom", "Lượm", "https://example.org/luom", "Lượm của Tố Hữu.")]

    now = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: "Bài phân tích có luận đề và dẫn chứng [1].")
    settings = get_settings().model_copy(update={"answer_verification_enabled": False})
    ctx = GraphContext(
        settings=settings,
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        web_search=LiteratureWeb(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "followup-student")
    store.add_user_message(cid, "Phân tích bài thơ Lượm của Tố Hữu", ts=now)
    first_tid = store.begin_turn(cid)
    first = await run_turn(build_graph(), turn_id=first_tid, conversation_id=cid, context=ctx)
    store.apply_result(cid, first_tid, first)

    store.add_user_message(cid, "phân tích sâu và chi tiết hơn, dài 2000 chữ đi", ts=now)
    second_tid = store.begin_turn(cid)
    second = await run_turn(build_graph(), turn_id=second_tid, conversation_id=cid, context=ctx)

    assert second["mode"] == "writing"
    assert second["writing_task"]["kind"] == "literary_argument"
    assert second["writing_task"]["target_words"] == 2000
    assert "writing_followup_inherited" in second["check_flags"]
    assert "khoảng 2000 chữ" in str(provider.calls[-1][0].content)


async def test_graph_keeps_2500_word_counter_reading_across_write_now_followup():
    class LegendWeb:
        async def search(self, query: str) -> list[KnowledgeEntry]:
            return [
                KnowledgeEntry(
                    "web:legend",
                    "Sơn Tinh - Thủy Tinh",
                    "https://example.org/legend",
                    "Vua Hùng ra điều kiện sính lễ; Thủy Tinh đến sau và giao chiến với Sơn Tinh.",
                )
            ]

    now = datetime(2026, 10, 6, 2, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: "Bài phân tích phân biệt dữ kiện văn bản với giả thuyết diễn giải [1].")
    settings = get_settings().model_copy(update={"answer_verification_enabled": False})
    ctx = GraphContext(
        settings=settings,
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        web_search=LegendWeb(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "counter-reading-student")
    store.add_user_message(
        cid,
        "Trong truyền thuyết Sơn Tinh - Thủy Tinh, hãy phân tích theo góc nhìn Thủy Tinh bị chèn ép "
        "và liên hệ tâm lý con người.",
        ts=now,
    )
    first_tid = store.begin_turn(cid)
    first = await run_turn(build_graph(), turn_id=first_tid, conversation_id=cid, context=ctx)
    store.apply_result(cid, first_tid, first)

    store.add_user_message(cid, "phân tích nghị luận 2500 chữ đi", ts=now)
    second_tid = store.begin_turn(cid)
    second = await run_turn(build_graph(), turn_id=second_tid, conversation_id=cid, context=ctx)
    store.apply_result(cid, second_tid, second)

    store.add_user_message(cid, "viết luôn cho tôi 1 bài phân tích", ts=now)
    third_tid = store.begin_turn(cid)
    third = await run_turn(build_graph(), turn_id=third_tid, conversation_id=cid, context=ctx)

    assert second["mode"] == third["mode"] == "writing"
    assert second["writing_task"]["target_words"] == third["writing_task"]["target_words"] == 2500
    assert "writing_followup_inherited" in second["check_flags"]
    assert "writing_followup_inherited" in third["check_flags"]
    third_system = str(provider.calls[-1][0].content)
    assert "khoảng 2500 chữ" in third_system and "giả thuyết diễn giải" in third_system


@pytest.mark.parametrize(
    "question",
    [
        "Phân tích sâu đoạn truyện tự viết: “Người thợ đặt chiếc búa cũ bên cửa sổ rồi lặng im.”",
        "Sáng tác một bài thơ nguyên bản, không bắt chước tác giả cụ thể.",
        "Hướng dẫn lớp 11 phân tích truyện theo nhiều trường phái, nêu giới hạn từng góc nhìn.",
        "Viết phần phân tích nhân vật học sinh quay cóp vì sợ mất học bổng; liên hệ đời sống.",
    ],
)
async def test_supplied_creative_and_method_writing_do_not_require_external_sources(question: str):
    now = datetime(2026, 10, 6, 1, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: "Câu trả lời bám đúng yêu cầu và ngữ liệu người dùng cung cấp.")
    settings = get_settings().model_copy(update={"answer_verification_enabled": False})
    ctx = GraphContext(
        settings=settings, store=store, llm=provider, profile=FanpageProfile(), now=lambda: now
    )
    cid = store.ensure_conversation("PAGE1", f"routing-{len(question)}")
    store.add_user_message(cid, question, ts=now)
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "writing"
    assert result["knowledge_question"] is False
    assert result["math_mode"] is False
    assert "chưa có nguồn đã kiểm chứng" not in " ".join(result["reply_parts"])


async def test_supplied_literary_text_expands_in_multiple_rounds_to_target():
    now = datetime(2026, 10, 6, 1, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)

    def responder(messages):  # type: ignore[no-untyped-def]
        if "PHẦN BỔ SUNG" in str(messages[0].content):
            return " ".join(["luận điểm mới"] * 250)
        return " ".join(["phân tích gốc"] * 250)

    provider = FakeProvider(responder)
    settings = get_settings().model_copy(update={"answer_verification_enabled": False})
    ctx = GraphContext(
        settings=settings, store=store, llm=provider, profile=FanpageProfile(), now=lambda: now
    )
    cid = store.ensure_conversation("PAGE1", "long-supplied-literature")
    store.add_user_message(
        cid,
        "Phân tích đoạn truyện tự viết sau khoảng 2500 chữ: “Người thợ đặt chiếc búa cũ bên cửa sổ rồi lặng im.”",
        ts=now,
    )
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["verification_meta"]["rounds"] == 2
    assert result["verification_meta"]["words_after"] >= 2125
    assert "supplied_literary_expanded" in result["check_flags"]


async def test_long_writing_button_expands_generic_essay_past_2500_words():
    now = datetime(2026, 10, 6, 1, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)

    def responder(messages):  # type: ignore[no-untyped-def]
        if "PHẦN BỔ SUNG" in str(messages[0].content):
            return " ".join(["luận điểm mới"] * 250)
        return " ".join(["phân tích gốc"] * 250)

    provider = FakeProvider(responder)
    settings = get_settings().model_copy(update={"answer_verification_enabled": False})
    ctx = GraphContext(
        settings=settings, store=store, llm=provider, profile=FanpageProfile(), now=lambda: now
    )
    cid = store.ensure_conversation("PAGE1", "long-writing-button")
    store.add_user_message(
        cid,
        "Hãy viết ngay một bài văn hoàn chỉnh khoảng 2800 chữ, tối thiểu 2500 chữ. "
        "Không trả lời bằng khung ý. Đề: tình mẫu tử.",
        ts=now,
    )
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "writing"
    assert result["verification_meta"]["words_after"] >= 2500
    assert "long_writing_expanded" in result["check_flags"]


async def test_graph_traces_and_prompts_creative_style_profile():
    now = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: "mưa đầu hạ\nchạm khẽ mái hiên xưa\nmùi đất thức")
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "poet")
    store.add_user_message(
        cid,
        "Tạo một bài thơ haiku phong cách Nhật Bản, tối giản và buồn",
        ts=now,
    )
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "writing" and result["writing_task"]["creative_requested"]
    assert result["writing_task"]["form"] == "haiku"
    system = str(provider.calls[0][0].content)
    assert "SÁNG TÁC NGUYÊN BẢN" in system and "Hình thức haiku" in system


async def test_graph_regenerates_invalid_haiku_once():
    now = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    answers = iter(
        [
            "mưa đầu hạ rơi\nlá rụng trên đường\ngió không ai gọi",
            "mưa rơi qua mái cũ\nlá non run dưới hiên chiều vắng\nđêm dài nghe đất thở",
        ]
    )
    provider = FakeProvider(lambda _: next(answers))
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "poet")
    store.add_user_message(cid, "Tạo một bài thơ haiku bằng tiếng Việt. Chỉ đưa bài thơ.", ts=now)
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert len(provider.calls) == 2
    assert "creative_form_regenerated" in result["check_flags"]
    assert "mưa rơi qua mái cũ" in " ".join(result["reply_parts"])


async def test_graph_keeps_non_writing_question_in_normal_chat():
    now = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: "Bạn có thể dùng bảng tổng hợp.")
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "student")
    store.add_user_message(cid, "Phân tích dữ liệu bán hàng bằng cách nào?", ts=now)
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    assert result["mode"] == "chat"
    assert not result["writing_mode"]
    assert "CHẾ ĐỘ GIA SƯ NGỮ VĂN" not in str(provider.calls[0][0].content)


async def test_missing_literary_text_gets_deterministic_reply_without_model():
    now = datetime(2026, 10, 5, 8, 0, tzinfo=UTC)
    store = InMemoryConversationStore(now=lambda: now)
    provider = FakeProvider(lambda _: 'Bịa câu thơ: "một câu không tồn tại"')
    ctx = GraphContext(
        settings=get_settings(),
        store=store,
        llm=provider,
        profile=FanpageProfile(),
        now=lambda: now,
    )
    cid = store.ensure_conversation("PAGE1", "student")
    store.add_user_message(
        cid,
        "Phân tích đoạn thơ mình vừa nói ở trên và trích đúng những câu hay nhất",
        ts=now,
    )
    tid = store.begin_turn(cid)
    result = await run_turn(build_graph(), turn_id=tid, conversation_id=cid, context=ctx)

    reply = " ".join(result["reply_parts"])
    assert result["mode"] == "writing_missing_text"
    assert not provider.calls
    assert "gửi lại" in reply and "không tự dựng câu thơ" in reply
