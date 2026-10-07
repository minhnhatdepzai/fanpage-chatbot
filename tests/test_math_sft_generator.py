from __future__ import annotations

from collections import Counter

from training.scripts.build_math_sft import build_examples


def test_math_sft_generator_covers_every_grade_with_separate_validation():
    examples = build_examples(per_grade=20, seed=20261006)
    assert len(examples) == 240
    assert len({example["id"] for example in examples}) == len(examples)
    assert len({example["messages"][0]["content"] for example in examples}) == len(examples)
    grades = Counter(int(example["category"].removeprefix("math_grade_")) for example in examples)
    validation = Counter(
        int(example["category"].removeprefix("math_grade_"))
        for example in examples
        if example["split"] == "val"
    )
    assert grades == {grade: 20 for grade in range(1, 13)}
    assert validation == {grade: 2 for grade in range(1, 13)}
    assert all(example["provenance"] == "synthetic_deterministic" for example in examples)
    assert all("Kiểm tra độc lập" in example["messages"][-1]["content"] for example in examples)
    answers = "\n".join(example["messages"][-1]["content"] for example in examples)
    assert "Trực quan hóa phù hợp: mixture" in answers
    assert "Trực quan hóa phù hợp: motion_comparison" in answers
