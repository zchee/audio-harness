"""Tests for the semantic-fidelity STT judge (plan P4.E1).

The judge lane's promises are: no vote is ever re-billed, the majority and
gate arithmetic are exact, the deterministic entity cross-check catches a
lenient judge, and the SeMaScore fallback is a pure function of its inputs.
Every test here runs against a stubbed judge and stubbed embeddings; the one
live smoke is env-gated per the repo's minimal-real-API policy.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from audio_harness.judge.semantic import (
    EXPERIMENTAL_BANNER,
    KAPPA_GATE,
    RUBRIC,
    VOTE_SEEDS,
    GateStatus,
    GeminiJudge,
    ItemJudgement,
    JudgeBudgetError,
    JudgeItem,
    JudgeRunStats,
    JudgeVote,
    VoteCache,
    bootstrap_kappa_ci,
    cache_key,
    cohens_kappa,
    entity_check,
    evaluate_gates,
    judgeable_items,
    load_anchor,
    majority_label,
    render_semantic_markdown,
    run_judge,
    semascore,
    summarize_semantic,
    write_semantic_results,
)
from audio_harness.types import Mode, SttResult


def make_result(
    provider: str = "deepgram-nova3",
    clip_id: str = "clip-1",
    mode: Mode = Mode.STREAM,
    text: str = "hello world",
    reference: str = "hello world",
    language: str = "en-US",
    error: str | None = None,
    gold_status: str | None = None,
    annotated: str | None = None,
) -> SttResult:
    """Build a saved-result record the way read_stt_results reconstructs one."""
    result = SttResult(
        provider=provider, clip_id=clip_id, mode=mode, text=text, error=error
    )
    result.raw["reference"] = reference
    result.raw["language"] = language
    if gold_status:
        result.raw["gold_status"] = gold_status
    if annotated:
        result.raw["reference_annotated"] = annotated
    return result


def make_item(
    provider: str = "deepgram-nova3",
    clip_id: str = "clip-1",
    mode: str = "stream",
    reference: str = "hello world",
    hypothesis: str = "hello world",
    language: str = "en-US",
    annotated: str | None = None,
) -> JudgeItem:
    """Build one judgeable pair."""
    return JudgeItem(
        provider=provider,
        mode=mode,
        clip_id=clip_id,
        language=language,
        reference=reference,
        hypothesis=hypothesis,
        reference_annotated=annotated,
    )


class ScriptedJudge:
    """Stub judge that returns a fixed label and counts its invocations."""

    def __init__(self, label: str = "harmless") -> None:
        self.label = label
        self.calls = 0

    def __call__(self, item: JudgeItem, seed: int) -> JudgeVote:
        self.calls += 1
        return JudgeVote(label=self.label, input_tokens=100, output_tokens=10)


class ExplodingJudge:
    """Stub judge that must never be reached (cache-hit assertions)."""

    def __call__(self, item: JudgeItem, seed: int) -> JudgeVote:
        raise AssertionError("judge was called although every vote was cached")


class TestJudgeableItems:
    """Only scoreable transcripts with trusted references are judged."""

    def test_selects_ok_results_with_references(self) -> None:
        items = judgeable_items([make_result()], "en-US")
        assert len(items) == 1
        assert items[0].provider == "deepgram-nova3"
        assert items[0].reference == "hello world"

    def test_skips_failures_empty_references_and_unverified(self) -> None:
        results = [
            make_result(clip_id="failed", error="boom"),
            make_result(clip_id="no-ref", reference=""),
            make_result(clip_id="subtitle", gold_status="unverified"),
        ]
        assert judgeable_items(results, "en-US") == []

    def test_empty_hypothesis_is_still_judgeable(self) -> None:
        """Dropping the whole utterance is an error to judge, not to skip."""
        items = judgeable_items([make_result(text="")], "en-US")
        assert len(items) == 1
        assert items[0].hypothesis == ""

    def test_duplicate_lane_rows_keep_the_last_occurrence(self) -> None:
        results = [
            make_result(text="first pass"),
            make_result(text="second pass"),
        ]
        items = judgeable_items(results, "en-US")
        assert len(items) == 1
        assert items[0].hypothesis == "second pass"

    def test_language_falls_back_when_unrecorded(self) -> None:
        result = make_result(language="")
        items = judgeable_items([result], "ja-JP")
        assert items[0].language == "ja-JP"


class TestMajority:
    """Three votes resolve to one deterministic verdict."""

    def test_unanimous(self) -> None:
        assert majority_label(("harmless",) * 3) == "harmless"

    def test_two_one_split_resolves_by_count(self) -> None:
        assert majority_label(("entity", "harmless", "harmless")) == "harmless"

    def test_three_way_split_falls_back_to_severity(self) -> None:
        """An uncertain judge must never launder an item into harmless."""
        votes = ("harmless", "meaning-changing", "entity")
        assert majority_label(votes) == "entity"

    def test_split_without_entity_prefers_meaning_changing(self) -> None:
        votes = ("harmless", "meaning-changing", "harmless")
        assert majority_label(votes) == "harmless"
        assert majority_label(("meaning-changing", "harmless")) == "meaning-changing"


class TestVoteCache:
    """The cache is what makes re-runs free."""

    def test_roundtrip_and_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        cache = VoteCache(path)
        assert cache.get("k1") is None
        cache.put("k1", "entity")
        assert cache.get("k1") == "entity"

        reloaded = VoteCache(path)
        assert reloaded.get("k1") == "entity"

    def test_cache_key_covers_content_not_identity(self) -> None:
        """Identical pairs share votes; different content never collides."""
        base = make_item()
        same_content = make_item(provider="soniox-rt-v5", clip_id="other")
        different = make_item(hypothesis="hello word")
        assert cache_key(base, 1) == cache_key(same_content, 1)
        assert cache_key(base, 1) != cache_key(different, 1)
        assert cache_key(base, 1) != cache_key(base, 2)


class TestRunJudge:
    """Vote orchestration: billing, caching and the budget cap."""

    def test_bills_three_votes_per_item(self, tmp_path: Path) -> None:
        judge = ScriptedJudge()
        items = [make_item(), make_item(clip_id="clip-2", hypothesis="hi world")]
        judgements, stats = run_judge(items, judge, VoteCache(tmp_path / "cache.jsonl"))
        assert judge.calls == len(items) * len(VOTE_SEEDS)
        assert stats.live_calls == 6
        assert stats.cached_votes == 0
        assert stats.input_tokens == 600
        assert stats.output_tokens == 60
        assert [j.majority for j in judgements] == ["harmless", "harmless"]

    def test_second_run_is_fully_cached(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        items = [make_item()]
        run_judge(items, ScriptedJudge(), VoteCache(path))

        judgements, stats = run_judge(items, ExplodingJudge(), VoteCache(path))
        assert stats.live_calls == 0
        assert stats.cached_votes == len(VOTE_SEEDS)
        assert stats.estimated_usd == 0.0
        assert judgements[0].majority == "harmless"

    def test_budget_cap_fails_before_any_call(self, tmp_path: Path) -> None:
        judge = ScriptedJudge()
        items = [make_item(), make_item(clip_id="clip-2", hypothesis="x")]
        with pytest.raises(JudgeBudgetError):
            run_judge(items, judge, VoteCache(tmp_path / "c.jsonl"), max_calls=5)
        assert judge.calls == 0

    def test_unknown_label_fails_loudly(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="unknown label"):
            run_judge(
                [make_item()],
                ScriptedJudge(label="mostly-fine"),
                VoteCache(tmp_path / "c.jsonl"),
            )

    def test_wer_and_semascore_ride_each_judgement(self, tmp_path: Path) -> None:
        items = [make_item(reference="hello world", hypothesis="hello word")]
        judgements, _ = run_judge(
            items,
            ScriptedJudge(),
            VoteCache(tmp_path / "c.jsonl"),
            semascore_fn=lambda item: 0.5,
        )
        assert judgements[0].wer == pytest.approx(0.5)
        assert judgements[0].semascore == 0.5


class TestEntityCrossCheck:
    """The deterministic scorer keeps the judge honest."""

    def test_damaged_entity_detected(self) -> None:
        item = make_item(
            reference="my code is 4 2 1",
            hypothesis="my code is 4 3 1",
            annotated="my code is <number>4 2 1</number>",
        )
        assert entity_check(item) is True

    def test_intact_entity_passes(self) -> None:
        item = make_item(
            reference="my code is 4 2 1",
            hypothesis="my code is 4 2 1",
            annotated="my code is <number>4 2 1</number>",
        )
        assert entity_check(item) is False

    def test_untaggable_reference_yields_none(self) -> None:
        assert entity_check(make_item(reference="well hello there")) is None

    def test_autotag_fallback_without_annotation(self) -> None:
        """Plain references still get the rule tagger's protection."""
        item = make_item(
            reference="the meeting is on March 3 2026",
            hypothesis="the meeting is on March 5 2026",
        )
        assert entity_check(item) is True

    def test_judge_missed_entity_flag(self) -> None:
        judgement = ItemJudgement(
            item=make_item(),
            votes=("harmless",) * 3,
            majority="harmless",
            entity_errors=True,
            semascore=None,
            wer=0.1,
        )
        assert judgement.judge_missed_entity is True
        assert judgement.unanimous is True


class TestKappa:
    """Gate arithmetic against hand-computed fixtures."""

    def test_hand_computed_two_label_fixture(self) -> None:
        """po=0.7, pe=0.5 by hand, so kappa is exactly 0.4."""
        pairs = (
            [("harmless", "harmless")] * 20
            + [("harmless", "entity")] * 5
            + [("entity", "harmless")] * 10
            + [("entity", "entity")] * 15
        )
        assert cohens_kappa(pairs) == pytest.approx(0.4)

    def test_perfect_agreement(self) -> None:
        pairs = [(label, label) for label in RUBRIC for _ in range(5)]
        assert cohens_kappa(pairs) == pytest.approx(1.0)

    def test_degenerate_single_label_marginals(self) -> None:
        assert cohens_kappa([("harmless", "harmless")] * 10) == 1.0
        assert cohens_kappa([]) is None

    def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate(self) -> None:
        pairs = (
            [("harmless", "harmless")] * 30
            + [("entity", "entity")] * 30
            + [("entity", "harmless")] * 10
            + [("meaning-changing", "meaning-changing")] * 20
            + [("meaning-changing", "harmless")] * 10
        )
        first = bootstrap_kappa_ci(pairs)
        second = bootstrap_kappa_ci(pairs)
        assert first == second
        assert first is not None
        kappa = cohens_kappa(pairs)
        assert kappa is not None
        assert first[0] <= kappa <= first[1]
        assert bootstrap_kappa_ci([]) is None


class TestSemascore:
    """The fallback metric is a pure function of the transcripts."""

    @staticmethod
    def embed_table(vectors: dict[str, list[float]]):
        def embed(text: str) -> np.ndarray:
            return np.asarray(vectors[text], dtype=np.float64)

        return embed

    def test_identical_transcripts_score_one_without_embedding(self) -> None:
        def explode(text: str) -> np.ndarray:
            raise AssertionError("equal transcripts must not be embedded")

        assert semascore("Hello, world", "hello world", "en-US", explode) == 1.0

    def test_orthogonal_substitution_scores_by_span_weight(self) -> None:
        """'the cat sat' vs 'the dog sat': 2 of 3 tokens match, sub scores 0."""
        embed = self.embed_table({"cat": [1.0, 0.0], "dog": [0.0, 1.0]})
        score = semascore("the cat sat", "the dog sat", "en-US", embed)
        assert score == pytest.approx(2.0 / 3.0)

    def test_semantically_close_substitution_outscores_a_flip(self) -> None:
        close = self.embed_table({"cat": [1.0, 0.1], "kitten": [1.0, 0.2]})
        flipped = self.embed_table({"cat": [1.0, 0.0], "not": [-1.0, 0.0]})
        near = semascore("the cat sat", "the kitten sat", "en-US", close)
        far = semascore("the cat sat", "the not sat", "en-US", flipped)
        assert near > far

    def test_deletion_and_insertion_score_zero(self) -> None:
        def unused(text: str) -> np.ndarray:
            raise AssertionError("pure deletions embed nothing")

        assert semascore("hello world", "", "en-US", unused) == 0.0
        assert semascore("", "", "en-US", unused) == 1.0

    def test_deterministic_across_calls(self) -> None:
        embed = self.embed_table({"cat": [0.3, 0.7], "dog": [0.6, 0.4]})
        runs = {
            semascore("the cat sat", "the dog sat", "en-US", embed) for _ in range(3)
        }
        assert len(runs) == 1


class TestAnchorFile:
    """The anchor CSV is part of the gate protocol."""

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.csv"
        path.write_text(
            "clip_id,provider,human_label\n"
            "clip-1,deepgram-nova3,harmless\n"
            "clip-2,soniox-rt-v5,entity\n",
            encoding="utf-8",
        )
        anchor = load_anchor(path)
        assert anchor == {
            ("clip-1", "deepgram-nova3"): "harmless",
            ("clip-2", "soniox-rt-v5"): "entity",
        }

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.csv"
        path.write_text("clip_id,human_label\nclip-1,harmless\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing columns: provider"):
            load_anchor(path)

    def test_unknown_label_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.csv"
        path.write_text(
            "clip_id,provider,human_label\nclip-1,deepgram-nova3,fine\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown label"):
            load_anchor(path)

    def test_duplicate_row_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "anchor.csv"
        path.write_text(
            "clip_id,provider,human_label\n"
            "clip-1,deepgram-nova3,harmless\n"
            "clip-1,deepgram-nova3,entity\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_anchor(path)


def make_judgement(
    clip_id: str,
    majority: str,
    *,
    provider: str = "deepgram-nova3",
    mode: str = "stream",
    language: str = "en-US",
    votes: tuple[str, ...] | None = None,
) -> ItemJudgement:
    """Build a judged item without going through a judge."""
    return ItemJudgement(
        item=make_item(
            provider=provider, clip_id=clip_id, mode=mode, language=language
        ),
        votes=votes or (majority,) * 3,
        majority=majority,
        entity_errors=None,
        semascore=None,
        wer=0.0,
    )


class TestGates:
    """Per-language gate evaluation and the experimental default."""

    def test_no_anchor_renders_experimental(self) -> None:
        gates = evaluate_gates([make_judgement("clip-1", "harmless")], None)
        assert len(gates) == 1
        gate = gates[0]
        assert gate.kappa is None
        assert not gate.passed
        assert gate.status == EXPERIMENTAL_BANNER

    def test_anchor_pairs_prefer_the_stream_judgement(self) -> None:
        """One anchor row must never pair with both modes of a lane."""
        judgements = [
            make_judgement("clip-1", "harmless", mode="batch"),
            make_judgement("clip-1", "entity", mode="stream"),
        ]
        anchor = {("clip-1", "deepgram-nova3"): "entity"}
        gate = evaluate_gates(judgements, anchor)[0]
        assert gate.anchored == 1
        assert gate.kappa == 1.0

    def test_unanimity_diagnostic(self) -> None:
        judgements = [
            make_judgement("clip-1", "harmless"),
            make_judgement(
                "clip-2",
                "entity",
                votes=("entity", "entity", "harmless"),
            ),
        ]
        gate = evaluate_gates(judgements, None)[0]
        assert gate.unanimity == pytest.approx(0.5)

    def test_languages_gate_independently(self) -> None:
        judgements = [
            make_judgement("clip-1", "harmless", language="en-US"),
            make_judgement("clip-2", "harmless", language="ja-JP"),
        ]
        anchor = {("clip-1", "deepgram-nova3"): "harmless"}
        gates = evaluate_gates(judgements, anchor)
        by_language = {gate.language: gate for gate in gates}
        assert by_language["en-US"].anchored == 1
        assert by_language["ja-JP"].anchored == 0
        assert by_language["ja-JP"].status == EXPERIMENTAL_BANNER

    def test_gate_threshold_decides_on_the_point_estimate(self) -> None:
        gate = GateStatus(
            language="en-US",
            judged=100,
            anchored=100,
            kappa=KAPPA_GATE,
            ci=(0.65, 0.85),
            unanimity=0.9,
        )
        assert gate.passed
        assert gate.near_threshold
        assert "near threshold" in gate.status


class TestSummariesAndReport:
    """Aggregation and the experimental rendering rule."""

    def test_summarize_counts_labels_and_wer(self) -> None:
        judgements = [
            make_judgement("clip-1", "harmless"),
            make_judgement("clip-2", "entity"),
        ]
        summaries = summarize_semantic(judgements)
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.items == 2
        assert summary.label_rate("entity") == pytest.approx(0.5)
        assert summary.label_rate("harmless") == pytest.approx(0.5)
        assert summary.unanimity_rate == 1.0
        assert summary.error_rate == 0.0

    def test_render_marks_unanchored_languages_experimental(self) -> None:
        judgements = [make_judgement("clip-1", "harmless")]
        markdown = render_semantic_markdown(
            summarize_semantic(judgements), evaluate_gates(judgements, None)
        )
        assert EXPERIMENTAL_BANNER in markdown
        assert "Cohen's kappa" in markdown

    def test_render_shows_gate_pass(self) -> None:
        judgements = [make_judgement(f"clip-{i}", "harmless") for i in range(10)] + [
            make_judgement(f"clip-e{i}", "entity") for i in range(10)
        ]
        anchor = {
            (judgement.item.clip_id, "deepgram-nova3"): judgement.majority
            for judgement in judgements
        }
        markdown = render_semantic_markdown(
            summarize_semantic(judgements), evaluate_gates(judgements, anchor)
        )
        assert "gate passed" in markdown

    def test_write_semantic_results_jsonl(self, tmp_path: Path) -> None:
        import orjson

        judgements = [make_judgement("clip-1", "harmless")]
        gates = evaluate_gates(judgements, None)
        path = write_semantic_results(
            judgements, gates, JudgeRunStats(live_calls=3), tmp_path / "out.jsonl"
        )
        records = [
            orjson.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert records[0]["majority"] == "harmless"
        assert records[0]["model"]
        assert records[1]["gate_language"] == "en-US"
        assert records[1]["passed"] is False
        assert records[2]["run_stats"]["live_calls"] == 3
        assert math.isfinite(records[2]["run_stats"]["estimated_usd"])


LIVE_FLAG = "AUDIO_HARNESS_TEST_SEMANTIC_LIVE"


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG) or not os.environ.get("GEMINI_API_KEY"),
    reason=f"live smoke needs {LIVE_FLAG}=1 and GEMINI_API_KEY (one billed Flash call)",
)
class TestLiveJudge:
    """One-vote live smoke; a few cents at most."""

    def test_single_vote_returns_a_rubric_label(self) -> None:
        item = make_item(
            reference="the total is fifty dollars",
            hypothesis="the total is fifteen dollars",
        )
        vote = GeminiJudge()(item, seed=1)
        assert vote.label in RUBRIC
        assert vote.input_tokens > 0
