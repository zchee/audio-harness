"""Tests for entity-level scoring.

A digit swap inside an account number and a dropped filler word are the same
edit to corpus WER; these tests pin the machinery that tells them apart —
tag parsing, alignment-based per-class scoring, the shared-ITN equivalences,
and the summarize/report wiring.
"""

from __future__ import annotations

import pytest

from audio_harness.entities import (
    EntityClassScore,
    parse_annotated,
    score_entities,
    strip_tags,
)
from audio_harness.metrics import ErrorCounts, summarize
from audio_harness.report import render_stt_markdown, stt_summary_frame
from audio_harness.types import Mode, SttResult


class TestAnnotationParsing:
    """Inline tags mark entity spans without altering the transcript."""

    def test_segments_cover_the_whole_string(self) -> None:
        segments = parse_annotated(
            "pay <currency>$20</currency> by <date>May 1st</date>"
        )
        assert segments == [
            ("pay ", None),
            ("$20", "currency"),
            (" by ", None),
            ("May 1st", "date"),
        ]

    def test_strip_tags_restores_the_plain_transcript(self) -> None:
        annotated = "call <name>Anna</name> at <number>nine one seven</number>"
        assert strip_tags(annotated) == "call Anna at nine one seven"

    def test_unknown_tags_stay_visible_in_the_text(self) -> None:
        """An annotation typo must surface in the diff, not delete a span."""
        assert strip_tags("a <bogus>x</bogus> b") == "a <bogus>x</bogus> b"

    def test_untagged_reference_yields_no_scores(self) -> None:
        assert score_entities("no tags here", "no tags here", "en-US") == {}


class TestPerClassScoring:
    """Each entity class scores through the shared ITN."""

    def test_number_folds_before_judging(self) -> None:
        scores = score_entities(
            "<number>four hundred and twenty</number> dollars", "420 dollars", "en-US"
        )
        assert scores["number"].counts.errors == 0
        assert scores["number"].exact_match_rate == 1.0

    def test_date_ordinals_fold_before_judging(self) -> None:
        scores = score_entities(
            "meet on <date>February third</date>", "meet on february 3rd", "en-US"
        )
        assert scores["date"].exact_match_rate == 1.0

    def test_currency_symbol_and_words_agree(self) -> None:
        scores = score_entities(
            "it costs <currency>twenty five dollars</currency>", "it costs $25", "en-US"
        )
        assert scores["currency"].exact_match_rate == 1.0

    def test_id_digit_dictation_merges_before_judging(self) -> None:
        scores = score_entities(
            "code <id>nine one seven three</id>", "code 9173", "en-US"
        )
        assert scores["id"].exact_match_rate == 1.0

    def test_id_single_digit_error_is_caught(self) -> None:
        scores = score_entities(
            "code <id>nine one seven three</id>", "code 9273", "en-US"
        )
        assert scores["id"].counts == ErrorCounts(1, 0, 0, 1), (
            "the merged id is one token, and it is wrong"
        )
        assert scores["id"].exact_match_rate == 0.0

    def test_name_substitution_scores_per_token(self) -> None:
        scores = score_entities(
            "this is <name>Samantha Lee</name>", "this is Samantha Leigh", "en-US"
        )
        assert scores["name"].counts == ErrorCounts(1, 0, 0, 2)
        assert scores["name"].error_rate == pytest.approx(0.5)
        assert scores["name"].exact_match_rate == 0.0


class TestAlignmentAttribution:
    """Errors land on the entity only when alignment puts them there."""

    def test_deleted_entity_counts_every_token(self) -> None:
        scores = score_entities(
            "my name is <name>Anna Maria</name>", "my name is", "en-US"
        )
        assert scores["name"].counts == ErrorCounts(0, 2, 0, 2)
        assert scores["name"].error_rate == pytest.approx(1.0)

    def test_insertion_inside_an_entity_counts_against_it(self) -> None:
        scores = score_entities(
            "I am <name>Anna Maria</name>", "I am Anna von Maria", "en-US"
        )
        assert scores["name"].counts.insertions == 1
        assert scores["name"].exact_match_rate == 0.0

    def test_errors_outside_the_entity_do_not_leak_in(self) -> None:
        scores = score_entities(
            "the total was <number>ninety</number> exactly",
            "a total is 90 exactly",
            "en-US",
        )
        assert scores["number"].counts.errors == 0, (
            "context words changed; the entity itself was perfect"
        )
        assert scores["number"].exact_match_rate == 1.0

    def test_repeated_value_cannot_vouch_for_a_misheard_occurrence(self) -> None:
        scores = score_entities(
            "<number>five</number> plus <number>five</number>",
            "five plus nine",
            "en-US",
        )
        assert scores["number"].occurrences == 2
        assert scores["number"].exact_matches == 1, (
            "alignment judges each occurrence in place; substring search "
            "would have credited both"
        )

    def test_character_metric_language_scores_per_character(self) -> None:
        scores = score_entities("<name>東京</name>に行く", "東京に行く", "ja-JP")
        assert scores["name"].counts.reference_length == 2
        assert scores["name"].exact_match_rate == 1.0

    def test_empty_hypothesis_deletes_every_entity(self) -> None:
        scores = score_entities("<number>five</number>", "", "en-US")
        assert scores["number"].error_rate == pytest.approx(1.0)


class TestScoreAccumulation:
    """Scores add so corpus rates come from sums, not averages."""

    def test_addition_accumulates_all_fields(self) -> None:
        first = EntityClassScore(ErrorCounts(1, 0, 0, 2), occurrences=1)
        second = EntityClassScore(ErrorCounts(0, 0, 0, 3), 1, 1)

        total = first + second

        assert total.counts.reference_length == 5
        assert total.occurrences == 2
        assert total.exact_matches == 1
        assert total.error_rate == pytest.approx(0.2)
        assert total.exact_match_rate == pytest.approx(0.5)

    def test_empty_score_has_no_rates(self) -> None:
        assert EntityClassScore().error_rate is None
        assert EntityClassScore().exact_match_rate is None


def _result(text: str, annotated: str, reference: str) -> SttResult:
    result = SttResult(provider="p1", clip_id="c1", mode=Mode.STREAM, text=text)
    result.audio_s = 2.0
    result.raw["reference"] = reference
    result.raw["reference_annotated"] = annotated
    result.raw["language"] = "en-US"
    return result


class TestSummarizeWiring:
    """Entity scores ride the existing provider/mode/language keying."""

    def test_annotated_references_populate_entity_scores(self) -> None:
        results = [
            _result("code 9173", "code <id>nine one seven three</id>", "code 9173"),
            _result("code 9273", "code <id>nine one seven three</id>", "code 9173"),
        ]

        summaries = summarize(results, "en-US")

        assert len(summaries) == 1
        score = summaries[0].entities["id"]
        assert score.occurrences == 2
        assert score.exact_matches == 1
        assert score.error_rate == pytest.approx(0.5)

    def test_unannotated_results_leave_entities_empty(self) -> None:
        result = SttResult(provider="p1", clip_id="c1", mode=Mode.STREAM, text="hi")
        result.raw["reference"] = "hi"
        assert summarize([result], "en-US")[0].entities == {}

    def test_entity_columns_render_in_the_report(self) -> None:
        results = [
            _result("code 9173", "code <id>nine one seven three</id>", "code 9173")
        ]

        markdown = render_stt_markdown(stt_summary_frame(results, "en-US"))

        assert "Ent id" in markdown
        assert "EM 100.00%" in markdown
