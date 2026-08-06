"""Tests for the Common Voice sentence-selection logic.

No network here: fetching lives in ``tools/gen_common_voice_prompts.py`` and
is untested by convention (matching ``tools/fetch_musan.py``,
``tools/curate_yodas.py``); only the pure filter/sample/build functions are
pinned, deterministically.
"""

from __future__ import annotations

from audio_harness.common_voice_prompts import (
    CATEGORY,
    LICENSE,
    build_prompts,
    filter_sentences,
    sample_sentences,
)


class TestFilterSentences:
    """The length band biases toward general/long-form text, per script."""

    def test_keeps_sentences_in_the_word_count_band(self) -> None:
        lines = [
            "Too short.",
            "This sentence has a perfectly reasonable number of words in it.",
            "One two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
            "twenty twenty-one twenty-two twenty-three twenty-four twenty-five "
            "twenty-six twenty-seven twenty-eight twenty-nine thirty thirty-one "
            "thirty-two thirty-three thirty-four thirty-five thirty-six "
            "thirty-seven thirty-eight thirty-nine forty forty-one.",
        ]

        kept = filter_sentences(lines, language="en-US")

        assert kept == [
            "This sentence has a perfectly reasonable number of words in it."
        ]

    def test_deduplicates_repeated_lines(self) -> None:
        lines = ["A repeated general-length sentence goes here."] * 3

        kept = filter_sentences(lines, language="en-US")

        assert kept == ["A repeated general-length sentence goes here."]

    def test_blank_lines_are_dropped(self) -> None:
        lines = ["", "   ", "A perfectly ordinary sentence of decent length."]

        kept = filter_sentences(lines, language="en-US")

        assert kept == ["A perfectly ordinary sentence of decent length."]

    def test_japanese_uses_a_character_band_not_word_count(self) -> None:
        # No spaces, so a word-count filter would reject every real sentence.
        short = "短い。"
        long_enough = "これは十分な長さを持つ一般的な日本語の文章の例です。"

        kept = filter_sentences([short, long_enough], language="ja-JP")

        assert kept == [long_enough]


class TestSampleSentences:
    """Sampling is deterministic and never returns more than is available."""

    def test_same_seed_reproduces_the_same_sample(self) -> None:
        candidates = [f"Sentence number {i}." for i in range(50)]

        first = sample_sentences(candidates, count=10, seed=1)
        second = sample_sentences(candidates, count=10, seed=1)

        assert first == second

    def test_different_seeds_can_diverge(self) -> None:
        candidates = [f"Sentence number {i}." for i in range(50)]

        first = sample_sentences(candidates, count=10, seed=1)
        second = sample_sentences(candidates, count=10, seed=2)

        assert first != second

    def test_never_exceeds_count(self) -> None:
        candidates = [f"Sentence number {i}." for i in range(5)]

        assert len(sample_sentences(candidates, count=100, seed=1)) == 5

    def test_does_not_mutate_the_input_list(self) -> None:
        candidates = ["a", "b", "c", "d", "e"]
        original = list(candidates)

        sample_sentences(candidates, count=3, seed=1)

        assert candidates == original


class TestBuildPrompts:
    """Prompts carry the CC0 license and per-locale provenance."""

    def test_every_prompt_is_cc0_and_general(self) -> None:
        prompts = build_prompts("en", ["Hello there."], ["contributor.txt"])

        assert prompts[0].license == LICENSE
        assert prompts[0].category == CATEGORY == "general"
        assert prompts[0].language == "en-US"

    def test_source_records_every_file_used(self) -> None:
        prompts = build_prompts("ja", ["text"], ["b.txt", "a.txt"])

        assert "a.txt" in prompts[0].source
        assert "b.txt" in prompts[0].source

    def test_ids_are_stable_and_ordered(self) -> None:
        prompts = build_prompts("en", ["First.", "Second."], ["f.txt"])

        assert [p.prompt_id for p in prompts] == ["general-0000", "general-0001"]
