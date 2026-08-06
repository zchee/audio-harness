"""Tests for the interview-question generator.

No network, no randomness: the English set is a fixed hand-authored list, so
these tests pin its shape and the MACHINE-DRAFT-PENDING policy for every
other locale rather than its exact wording.
"""

from __future__ import annotations

from pathlib import Path

from audio_harness.entities import parse_annotated
from audio_harness.interview_prompts import (
    INTERVIEW_QUESTIONS_EN,
    build_prompts,
    generate,
)
from audio_harness.prompt_suite import LOCALES, read_jsonl


class TestBuildPrompts:
    """Prompt construction is a pure, order-preserving transform."""

    def test_one_prompt_per_question_in_order(self) -> None:
        prompts = build_prompts("en-US", ("First?", "Second?"))

        assert [p.text for p in prompts] == ["First?", "Second?"]
        assert [p.prompt_id for p in prompts] == ["interview-0000", "interview-0001"]

    def test_every_prompt_carries_the_own_ip_license(self) -> None:
        prompts = build_prompts("en-US", ("Only question?",))

        assert prompts[0].category == "interview"
        assert prompts[0].license == "own-IP"

    def test_english_set_has_around_30_varied_questions(self) -> None:
        assert 25 <= len(INTERVIEW_QUESTIONS_EN) <= 35
        assert len(set(INTERVIEW_QUESTIONS_EN)) == len(INTERVIEW_QUESTIONS_EN), "no duplicate questions"
        assert all(q.strip().endswith(("?", ".")) for q in INTERVIEW_QUESTIONS_EN)

    def test_no_stray_entity_tags_leak_into_hand_authored_text(self) -> None:
        # These are plain speakable questions, not annotated references —
        # parse_annotated must see them as a single untagged segment.
        for question in INTERVIEW_QUESTIONS_EN:
            assert parse_annotated(question) == [(question, None)]


class TestGenerate:
    """The generator writes real content for en and a marker for everyone else."""

    def test_writes_the_english_jsonl_and_flat_txt(self, tmp_path: Path) -> None:
        jsonl_path, _pending = generate(tmp_path)

        prompts = read_jsonl(jsonl_path)
        assert len(prompts) == len(INTERVIEW_QUESTIONS_EN)
        assert all(p.language == "en-US" for p in prompts)

        flat = (tmp_path / "prompts-en" / "interview.txt").read_text(encoding="utf-8")
        assert len(flat.splitlines()) == len(INTERVIEW_QUESTIONS_EN)

    def test_every_non_english_locale_gets_a_pending_marker(self, tmp_path: Path) -> None:
        _jsonl_path, pending = generate(tmp_path)

        assert len(pending) == len(LOCALES) - 1
        for marker in pending:
            assert marker.name == "interview.PENDING.md"
            assert "MACHINE-DRAFT-PENDING" in marker.read_text(encoding="utf-8")

    def test_no_jsonl_or_txt_written_for_pending_locales(self, tmp_path: Path) -> None:
        generate(tmp_path)

        assert not (tmp_path / "prompts-ko" / "interview.jsonl").exists()
        assert not (tmp_path / "prompts-ko" / "interview.txt").exists()
        assert (tmp_path / "prompts-ko" / "interview.PENDING.md").exists()
