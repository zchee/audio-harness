"""Tests for cross-model transcript agreement reporting."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from audio_harness.agreement import (
    compute_agreement,
    load_agreement_runs,
    render_agreement_markdown,
    write_agreement_report,
)
from audio_harness.runner import write_stt_results
from audio_harness.types import Mode, SttResult


Lane = tuple[str, str]


def _result(
    provider: str,
    clip_id: str,
    text: str,
    language: str,
    *,
    error: str | None = None,
) -> SttResult:
    result = SttResult(
        provider=provider,
        clip_id=clip_id,
        mode=Mode.STREAM,
        text=text,
        error=error,
    )
    result.raw["language"] = language
    return result


def _write_run(tmp_path: Path, name: str, results: list[SttResult]) -> Path:
    return write_stt_results(results, tmp_path / name).parent


def _fixture_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_a = _write_run(
        tmp_path,
        "run-a",
        [
            _result("alpha", "c1", "the cat sat", "en-US"),
            _result("alpha", "c2", "hello world", "en-US"),
            _result("alpha", "c3", "東京都", "ja-JP"),
            _result("alpha", "c4", "ignored error", "en-US", error="timeout"),
            _result("alpha", "c5", "", "en-US"),
        ],
    )
    run_b = _write_run(
        tmp_path,
        "run-b",
        [
            _result("bravo", "c1", "the dog sat", "en-US"),
            _result("bravo", "c2", "hello brave world", "en-US"),
            _result("bravo", "c3", "東京と", "ja-JP"),
            _result("bravo", "c4", "must be skipped", "en-US"),
            _result("bravo", "c5", "must also be skipped", "en-US"),
        ],
    )
    run_c = _write_run(
        tmp_path,
        "run-c",
        [_result("charlie", "c1", "the cat sat", "en-US")],
    )
    return run_a, run_b, run_c


class TestAgreementComputation:
    """Pairwise counts are hand-checkable and corpus-weighted."""

    def test_pairwise_matrix_and_language_breakdown(self, tmp_path: Path) -> None:
        runs = load_agreement_runs(_fixture_runs(tmp_path))
        report = compute_agreement(runs)
        alpha: Lane = ("alpha", "stream")
        bravo: Lane = ("bravo", "stream")
        charlie: Lane = ("charlie", "stream")

        alpha_bravo = report.overall.cell(alpha, bravo)
        alpha_charlie = report.overall.cell(alpha, charlie)
        bravo_charlie = report.overall.cell(bravo, charlie)

        assert alpha_bravo is not None
        assert alpha_bravo.clips == 3
        assert alpha_bravo.counts.errors == 6
        assert alpha_bravo.counts.reference_length == 17
        assert alpha_bravo.disagreement_rate == 6 / 17
        assert alpha_charlie is not None
        assert alpha_charlie.clips == 1
        assert alpha_charlie.disagreement_rate == 0.0
        assert bravo_charlie is not None
        assert bravo_charlie.clips == 1
        assert bravo_charlie.disagreement_rate == 1 / 3

        english = report.per_language["en-US"].cell(alpha, bravo)
        japanese = report.per_language["ja-JP"].cell(alpha, bravo)
        assert english is not None
        assert english.clips == 2
        assert english.counts.errors == 4
        assert english.counts.reference_length == 11
        assert english.disagreement_rate == 4 / 11
        assert japanese is not None
        assert japanese.clips == 1
        assert japanese.counts.errors == 2
        assert japanese.counts.reference_length == 6
        assert japanese.disagreement_rate == 1 / 3

    def test_errored_and_empty_results_are_skipped(self, tmp_path: Path) -> None:
        runs = load_agreement_runs(_fixture_runs(tmp_path)[:2])
        report = compute_agreement(runs)
        cell = report.overall.cell(("alpha", "stream"), ("bravo", "stream"))

        assert cell is not None
        assert cell.clips == 3, "the errored c4 and empty c5 transcripts must not be scored"

    def test_non_overlapping_lanes_produce_an_empty_cell(self) -> None:
        runs = {
            ("alpha", "stream"): [_result("alpha", "a", "hello", "en-US")],
            ("bravo", "stream"): [_result("bravo", "b", "hello", "en-US")],
        }

        report = compute_agreement(runs)
        cell = report.overall.cell(("alpha", "stream"), ("bravo", "stream"))

        assert cell is not None
        assert cell.clips == 0
        assert cell.counts.reference_length == 0
        assert cell.disagreement_rate is None

    def test_one_lane_cannot_be_compared(self) -> None:
        runs = {("alpha", "stream"): [_result("alpha", "c1", "hello", "en-US")]}

        with pytest.raises(ValueError, match="at least two lanes"):
            compute_agreement(runs)


class TestAgreementLoading:
    """Run loading accepts directories and direct result paths."""

    def test_single_run_directory_is_rejected(self, tmp_path: Path) -> None:
        run = _write_run(tmp_path, "one", [_result("alpha", "c1", "hello", "en-US")])

        with pytest.raises(ValueError, match="at least two run"):
            load_agreement_runs([run])

    def test_missing_results_file_is_clear(self, tmp_path: Path) -> None:
        first = tmp_path / "missing-a"
        second = tmp_path / "missing-b"

        with pytest.raises(FileNotFoundError, match=r"missing-a/stt-results\.jsonl"):
            load_agreement_runs([first, second])

    def test_groups_results_by_provider_and_mode(self, tmp_path: Path) -> None:
        run_a, run_b, _ = _fixture_runs(tmp_path)

        runs = load_agreement_runs([run_a / "stt-results.jsonl", run_b])

        assert set(runs) == {("alpha", "stream"), ("bravo", "stream")}
        assert [result.clip_id for result in runs["alpha", "stream"]][:3] == ["c1", "c2", "c3"]


class TestAgreementOutput:
    """Human and machine-readable reports expose the same matrices."""

    def test_markdown_has_overall_and_per_language_matrices(self, tmp_path: Path) -> None:
        report = compute_agreement(load_agreement_runs(_fixture_runs(tmp_path)))

        markdown = render_agreement_markdown(report)

        assert markdown.startswith("# Cross-model agreement")
        assert "## Overall" in markdown
        assert "## Language: en-US" in markdown
        assert "## Language: ja-JP" in markdown
        assert "alpha (stream)" in markdown
        assert "35.29% (n=3)" in markdown

    def test_json_round_trip_preserves_matrix_shape(self, tmp_path: Path) -> None:
        report = compute_agreement(load_agreement_runs(_fixture_runs(tmp_path)))

        markdown_path, json_path = write_agreement_report(report, tmp_path / "agreement")
        payload = orjson.loads(json_path.read_bytes())

        assert markdown_path.name == "agreement.md"
        assert json_path.name == "agreement.json"
        assert markdown_path.is_file()
        assert payload["lanes"] == [
            {"provider": "alpha", "mode": "stream"},
            {"provider": "bravo", "mode": "stream"},
            {"provider": "charlie", "mode": "stream"},
        ]
        assert len(payload["overall"]["rates"]) == 3
        assert all(len(row) == 3 for row in payload["overall"]["rates"])
        assert payload["overall"]["clip_counts"][0][1] == 3
        assert payload["per_language"]["ja-JP"]["clip_counts"][0][1] == 1
