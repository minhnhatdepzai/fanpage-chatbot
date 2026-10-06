"""Regression tests for multi-file SFT manifests and rich-answer evaluation rules."""

from pathlib import Path

from app.conversation.knowledge import KnowledgeBase
from app.conversation.profile import FanpageProfile
from evaluation.scripts.compare import score
from training.sft_data import DatasetError, build_splits


def _dataset(path: Path, example_id: str, answer: str = "Câu trả lời.") -> Path:
    path.write_text(
        "examples:\n"
        f"  - id: {example_id}\n"
        "    category: test\n"
        "    sources: []\n"
        "    messages:\n"
        "      - {role: user, content: 'Câu hỏi'}\n"
        f"      - {{role: assistant, content: '{answer}'}}\n",
        encoding="utf-8",
    )
    return path


def test_build_splits_combines_files_and_records_each_checksum(tmp_path: Path):
    first = _dataset(tmp_path / "first.yaml", "one")
    second = _dataset(tmp_path / "second.yaml", "two")

    splits, manifest = build_splits([first, second], KnowledgeBase(), FanpageProfile())

    assert len(splits["train"]) == 2
    assert len(manifest["dataset_files"]) == 2
    assert manifest["counts"] == {"train": 2, "val": 0}
    assert manifest["dataset_version"].startswith("first+second-")


def test_build_splits_rejects_duplicate_ids_across_files(tmp_path: Path):
    first = _dataset(tmp_path / "first.yaml", "same")
    second = _dataset(tmp_path / "second.yaml", "same")

    try:
        build_splits([first, second], KnowledgeBase(), FanpageProfile())
    except DatasetError as exc:
        assert "trùng id giữa các dataset" in str(exc)
    else:
        raise AssertionError("duplicate ids must be rejected")


def test_rich_score_checks_depth_and_explicit_length_control():
    reply = "Một câu đủ dài. Hai câu tiếp theo giải thích kỹ. Câu cuối đưa ví dụ thực tế."
    result = score(
        {"include_all": ["giải thích", "ví dụ"], "min_chars": 60, "min_sentences": 3, "max_chars": 200},
        reply,
        [],
    )

    assert result["passed"] is True
    assert result["sentences"] == 3


def test_rich_score_checks_word_range():
    result = score({"min_words": 4, "max_words": 6}, "một hai ba bốn năm", [])

    assert result["passed"] is True
    assert result["words"] == 5
