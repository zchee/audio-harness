"""Tests for the shared prompt-suite file format and loader glue.

The JSONL form is the source of truth (metadata, entity tags); the flattened
``.txt`` form is what ``dataset.py:load_prompts`` already reads with zero
changes to that module. Both round-trips are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_harness.prompt_suite import (
    CATEGORIES,
    LOCALES,
    SuitePrompt,
    flatten_to_prompts_txt,
    read_jsonl,
    write_jsonl,
)


def _prompt(
    prompt_id: str,
    text: str,
    *,
    language: str = "en-US",
    category: str = "interview",
    license: str = "own-IP",
    source: str = "hand-authored",
    annotated: str | None = None,
    entity_class: str | None = None,
) -> SuitePrompt:
    return SuitePrompt(
        prompt_id=prompt_id,
        text=text,
        language=language,
        category=category,
        license=license,
        source=source,
        annotated=annotated,
        entity_class=entity_class,
    )


class TestLocales:
    """The 14-locale map is the harness' authoritative BCP-47 tag list."""

    def test_covers_14_locales(self) -> None:
        assert len(LOCALES) == 14

    def test_every_tag_is_region_qualified(self) -> None:
        for tag in LOCALES.values():
            assert "-" in tag, f"{tag} should be a full BCP-47 tag"

    def test_categories_match_the_plans_three_layers(self) -> None:
        assert CATEGORIES == ("interview", "general", "entities")


class TestJsonlRoundTrip:
    """The JSONL form must survive a write/read cycle byte-for-byte in meaning."""

    def test_round_trips_every_field(self, tmp_path: Path) -> None:
        prompts = [
            _prompt("p0000", "Tell me about yourself."),
            _prompt(
                "p0001",
                "Please transfer $420.",
                category="entities",
                license="generated",
                source="tools/gen_entity_prompts.py",
                annotated="Please transfer <currency>four hundred and twenty "
                "dollars</currency>.",
                entity_class="currency",
            ),
        ]

        path = write_jsonl(tmp_path / "interview.jsonl", prompts)
        back = read_jsonl(path)

        assert back == prompts

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_jsonl(tmp_path / "nope.jsonl")

    def test_invalid_json_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            read_jsonl(path)

    def test_write_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dir" / "interview.jsonl"
        write_jsonl(path, [_prompt("p0000", "hi")])
        assert path.is_file()


class TestFlattenToPromptsTxt:
    """The flattened form must match exactly what ``load_prompts`` expects."""

    def test_one_line_per_prompt_in_order(self, tmp_path: Path) -> None:
        prompts = [_prompt("p0000", "First."), _prompt("p0001", "Second.")]

        path = flatten_to_prompts_txt(prompts, tmp_path / "interview.txt")

        assert path.read_text(encoding="utf-8").splitlines() == ["First.", "Second."]

    def test_blank_text_is_dropped(self, tmp_path: Path) -> None:
        prompts = [_prompt("p0000", "Real text."), _prompt("p0001", "   ")]

        path = flatten_to_prompts_txt(prompts, tmp_path / "interview.txt")

        assert path.read_text(encoding="utf-8").splitlines() == ["Real text."]

    def test_empty_input_writes_an_empty_file(self, tmp_path: Path) -> None:
        path = flatten_to_prompts_txt([], tmp_path / "interview.txt")

        assert path.read_text(encoding="utf-8") == ""

    def test_flattened_file_loads_through_dataset_load_prompts(
        self, tmp_path: Path
    ) -> None:
        # The whole point of the flat form: it must work with the existing,
        # untouched loader — no dataset.py changes required.
        from audio_harness.config import DatasetConfig
        from audio_harness.dataset import load_prompts

        prompts = [_prompt("ignored", "Tell me about yourself.")]
        path = flatten_to_prompts_txt(prompts, tmp_path / "interview.txt")

        loaded = load_prompts(DatasetConfig(prompts=str(path), language="en-US"))

        assert [p.text for p in loaded] == ["Tell me about yourself."]
        assert loaded[0].language == "en-US"
