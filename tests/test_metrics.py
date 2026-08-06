"""Tests for normalization, error rates, percentiles and churn."""

from __future__ import annotations

import math
from typing import TypedDict

import pytest

from audio_harness.metrics import (
    ErrorCounts,
    partial_instability,
    percentile,
    score_pair,
)
from audio_harness.normalize import (
    normalize_english,
    normalize_japanese,
    normalizer_for,
    uses_character_metric,
)
from audio_harness.types import Partial


class TestNormalization:
    """Normalization must erase formatting policy, not content."""

    def test_english_cases(self) -> None:
        tests = {
            "lowercases and strips punctuation": {
                "raw": "Hello, World!",
                "expected": "hello world",
            },
            "expands contractions": {
                "raw": "I can't won't don't",
                "expected": "i cannot will not do not",
            },
            "collapses whitespace runs": {
                "raw": "  a   b \t c \n",
                "expected": "a b c",
            },
            "folds full-width latin to ascii": {
                "raw": "ＨＥＬＬＯ",  # ruff: ignore[ambiguous-unicode-character-string] - full-width input is the fixture
                "expected": "hello",
            },
            "keeps digits intact": {
                "raw": "Room 101.",
                "expected": "room 101",
            },
        }
        for name, case in tests.items():
            assert normalize_english(case["raw"]) == case["expected"], name

    def test_japanese_cases(self) -> None:
        tests = {
            "removes japanese punctuation": {
                "raw": "こんにちは、世界。",
                "expected": "こんにちは世界",
            },
            "removes all whitespace": {
                "raw": "今日 は いい 天気",
                "expected": "今日はいい天気",
            },
            "folds half-width kana via nfkc": {
                "raw": "ｱｲｳ",
                "expected": "アイウ",
            },
            "folds full-width digits": {
                "raw": "１２３",  # ruff: ignore[ambiguous-unicode-character-string] - full-width input is the fixture
                "expected": "123",
            },
        }
        for name, case in tests.items():
            assert normalize_japanese(case["raw"]) == case["expected"], name

    def test_number_formatting_does_not_count_as_error(self) -> None:
        """Corpora spell numbers out; recognizers emit digits.

        Without this folding the accuracy column measures each vendor's
        inverse text normalization instead of what it heard.
        """
        tests = {
            "compound cardinal": {
                "spoken": "four hundred and twenty",
                "written": "420",
            },
            "currency symbol": {
                "spoken": "twenty five dollars",
                "written": "$25",
            },
            "dictated digit sequence": {
                "spoken": "nine one seven three",
                "written": "9173",
            },
            "thousands": {
                "spoken": "one thousand two hundred thirty four",
                "written": "1,234",
            },
            "millions": {
                "spoken": "one million two hundred thousand",
                "written": "1200000",
            },
            "word ordinal": {"spoken": "February third", "written": "February 3rd"},
            "tens plus unit": {"spoken": "twenty five", "written": "25"},
            "bare unit": {
                "spoken": "roughly two kilometres",
                "written": "roughly 2 kilometres",
            },
        }
        for name, case in tests.items():
            assert normalize_english(case["spoken"]) == normalize_english(case["written"]), name

    def test_digit_sequences_are_not_summed(self) -> None:
        """A dictated account number must not collapse into an arithmetic sum."""
        assert normalize_english("nine one seven three") == "9173"
        assert normalize_english("one two three") == "123"

    def test_genuine_compositions_still_combine(self) -> None:
        assert normalize_english("twenty five") == "25"
        assert normalize_english("four hundred and twenty") == "420"
        assert normalize_english("nineteen eighty four") == "19 84", (
            "two independent numbers stay separate rather than summing to 103"
        )

    def test_conjunction_outside_a_number_survives(self) -> None:
        assert normalize_english("cats and dogs") == "cats and dogs"
        assert normalize_english("black and white") == "black and white"

    def test_normalizer_selection_ignores_region_subtag(self) -> None:
        assert normalizer_for("ja-JP") is normalize_japanese
        assert normalizer_for("en-GB") is normalize_english

    def test_unknown_language_falls_back_without_raising(self) -> None:
        assert normalizer_for("xx-YY")("Hola, Mundo!") == "hola mundo"

    def test_character_metric_languages(self) -> None:
        assert uses_character_metric("ja-JP")
        assert uses_character_metric("zh")
        assert not uses_character_metric("en-US")


class _EditCase(TypedDict):
    """One edit-operation counting case."""

    ref: str
    hyp: str
    expected: ErrorCounts


class TestScorePair:
    """Edit counts drive every accuracy number in the report."""

    def test_identical_after_normalization_scores_zero(self) -> None:
        counts = score_pair("Hello, world!", "hello world", "en-US")
        assert counts.errors == 0
        assert counts.rate == 0.0

    def test_counts_each_edit_operation(self) -> None:
        tests: dict[str, _EditCase] = {
            "substitution": {
                "ref": "the cat sat",
                "hyp": "the dog sat",
                "expected": ErrorCounts(1, 0, 0, 3),
            },
            "deletion": {
                "ref": "the cat sat",
                "hyp": "the sat",
                "expected": ErrorCounts(0, 1, 0, 3),
            },
            "insertion": {
                "ref": "the cat sat",
                "hyp": "the big cat sat",
                "expected": ErrorCounts(0, 0, 1, 3),
            },
        }
        for name, case in tests.items():
            counts = score_pair(case["ref"], case["hyp"], "en-US")
            assert counts == case["expected"], name

    def test_japanese_is_scored_per_character(self) -> None:
        counts = score_pair("東京都", "東京と", "ja-JP")
        assert counts.reference_length == 3, "three characters, not one word"
        assert counts.errors == 1

    def test_japanese_spacing_differences_are_free(self) -> None:
        counts = score_pair("今日はいい天気", "今日 は いい 天気", "ja-JP")
        assert counts.errors == 0

    def test_empty_reference_counts_hypothesis_as_insertions(self) -> None:
        counts = score_pair("", "spurious words here", "en-US")
        assert counts.reference_length == 0
        assert counts.insertions == 3
        assert counts.rate == math.inf

    def test_empty_hypothesis_deletes_everything(self) -> None:
        counts = score_pair("alpha bravo charlie", "", "en-US")
        assert counts.deletions == 3
        assert counts.rate == 1.0

    def test_counts_accumulate_for_corpus_level_rate(self) -> None:
        short = score_pair("a", "b", "en-US")
        long = score_pair(" ".join(["word"] * 99), " ".join(["word"] * 99), "en-US")
        total = short + long

        assert total.reference_length == 100
        assert total.rate == pytest.approx(0.01), (
            "one bad word in a hundred must not become a 50% corpus rate; "
            "that is what averaging per-utterance rates would produce"
        )


class TestPercentile:
    """Latency is reported as percentiles, so the maths must be right."""

    def test_empty_input_returns_none(self) -> None:
        assert percentile([], 50) is None

    def test_single_sample_returns_itself(self) -> None:
        assert percentile([0.42], 95) == 0.42

    def test_interpolates_between_samples(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert percentile(values, 50) == pytest.approx(2.0)
        assert percentile(values, 0) == pytest.approx(0.0)
        assert percentile(values, 100) == pytest.approx(4.0)
        assert percentile(values, 25) == pytest.approx(1.0)

    def test_unsorted_input_is_ordered_first(self) -> None:
        assert percentile([3.0, 1.0, 2.0], 50) == pytest.approx(2.0)

    def test_p95_ignores_a_one_percent_outlier(self) -> None:
        values = [0.1] * 99 + [10.0]

        assert percentile(values, 50) == pytest.approx(0.1)
        assert percentile(values, 95) == pytest.approx(0.1), (
            "a single slow request in a hundred is below the 95th percentile; "
            "a mean would have reported 0.2 and hidden that the typical "
            "request is fast"
        )
        assert percentile(values, 100) == pytest.approx(10.0)

    def test_p95_moves_once_the_tail_is_thick_enough(self) -> None:
        values = [0.1] * 90 + [10.0] * 10

        assert percentile(values, 50) == pytest.approx(0.1)
        assert percentile(values, 95) == pytest.approx(10.0), "when a tenth of requests are slow, p95 must surface it"


class TestPartialInstability:
    """Churn measures how often a provider retracts text it already showed."""

    def _partials(self, texts: list[str]) -> list[Partial]:
        return [Partial(t_s=float(i), text=text, is_final=False) for i, text in enumerate(texts)]

    def test_monotonic_growth_is_perfectly_stable(self) -> None:
        partials = self._partials(["he", "hell", "hello", "hello wo"])
        assert partial_instability(partials) == 0.0

    def test_every_rewrite_is_counted(self) -> None:
        partials = self._partials(["hello", "goodbye", "farewell"])
        assert partial_instability(partials) == 1.0

    def test_mixed_stability_is_a_fraction(self) -> None:
        partials = self._partials(["a", "ab", "xy", "xyz"])
        assert partial_instability(partials) == pytest.approx(1 / 3)

    def test_too_few_interim_events_yields_none(self) -> None:
        assert partial_instability([]) is None
        assert partial_instability(self._partials(["only one"])) is None

    def test_final_events_are_excluded(self) -> None:
        partials = [
            Partial(t_s=0.0, text="a", is_final=False),
            Partial(t_s=1.0, text="totally different", is_final=True),
            Partial(t_s=2.0, text="ab", is_final=False),
        ]
        assert partial_instability(partials) == 0.0, "a final between two growing partials is not a rewrite"
