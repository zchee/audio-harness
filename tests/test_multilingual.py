"""Tests for benchmarking several languages in one run."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import wave

import polars as pl
import pytest

from audio_harness.config import BenchmarkConfig, ConfigError, DatasetConfig
from audio_harness.dataset import load_clips
from audio_harness.metrics import summarize
from audio_harness.report import stt_summary_frame
from audio_harness.types import Mode, SttResult


def _wav(seconds: float = 0.4, rate: int = 16000) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def _corpus(path: Path, prefix: str, texts: list[str]) -> Path:
    pl.DataFrame({
        "id": [f"{prefix}-{i}" for i in range(len(texts))],
        "audio": [{"bytes": _wav(), "path": None} for _ in texts],
        "utt": texts,
    }).write_parquet(path)
    return path


def _result(provider: str, language: str, reference: str, text: str) -> SttResult:
    result = SttResult(provider=provider, clip_id="c", mode=Mode.STREAM, text=text)
    result.audio_s = 1.0
    result.raw["reference"] = reference
    result.raw["language"] = language
    return result


class TestMultiSourceConfig:
    """A multilingual benchmark is several sources in one config."""

    def test_sources_inherit_shared_defaults(self) -> None:
        config = BenchmarkConfig.from_dict({
            "dataset": {
                "id_column": "id",
                "audio_column": "audio",
                "text_column": "utt",
                "limit": 5,
                "sample_seed": 42,
                "sources": [
                    {"parquet": "a.parquet", "language": "fr-FR"},
                    {"parquet": "b.parquet", "language": "de-DE"},
                ],
            }
        })
        sources = config.dataset.resolved_sources()

        assert [s.language for s in sources] == ["fr-FR", "de-DE"]
        assert all(s.text_column == "utt" for s in sources), (
            "shared column names should be stated once, not per language"
        )
        assert all(s.limit == 5 and s.sample_seed == 42 for s in sources)

    def test_per_source_override_wins(self) -> None:
        config = BenchmarkConfig.from_dict({
            "dataset": {
                "limit": 5,
                "sources": [
                    {"parquet": "a.parquet", "language": "fr-FR"},
                    {"parquet": "b.parquet", "language": "ko-KR", "limit": 2},
                ],
            }
        })
        assert [s.limit for s in config.dataset.resolved_sources()] == [5, 2]

    def test_single_source_config_still_works(self) -> None:
        config = BenchmarkConfig.from_dict({"dataset": {"parquet": "a.parquet", "language": "en-US"}})
        sources = config.dataset.resolved_sources()

        assert len(sources) == 1
        assert sources[0].parquet == "a.parquet"
        assert sources[0].language == "en-US"

    def test_source_without_a_corpus_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="parquet or manifest"):
            BenchmarkConfig.from_dict({"dataset": {"sources": [{"language": "fr-FR"}]}})

    def test_unknown_source_key_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="unknown key"):
            BenchmarkConfig.from_dict({"dataset": {"sources": [{"parquet": "a.parquet", "langauge": "fr-FR"}]}})


class TestMultiSourceLoading:
    """Every clip carries the language of the source it came from."""

    def test_clips_from_all_sources_are_tagged(self, tmp_path: Path) -> None:
        fr = _corpus(tmp_path / "fr.parquet", "fr", ["bonjour le monde", "merci"])
        de = _corpus(tmp_path / "de.parquet", "de", ["guten tag"])
        config = BenchmarkConfig.from_dict({
            "dataset": {
                "id_column": "id",
                "text_column": "utt",
                "sources": [
                    {"parquet": str(fr), "language": "fr-FR"},
                    {"parquet": str(de), "language": "de-DE"},
                ],
            }
        }).dataset

        clips = load_clips(config)

        assert len(clips) == 3
        assert [c.language for c in clips] == ["fr-FR", "fr-FR", "de-DE"]
        assert clips[2].reference == "guten tag"

    def test_per_source_limit_applies_independently(self, tmp_path: Path) -> None:
        fr = _corpus(tmp_path / "fr.parquet", "fr", [f"phrase {i}" for i in range(10)])
        de = _corpus(tmp_path / "de.parquet", "de", [f"satz {i}" for i in range(10)])
        config = DatasetConfig(
            id_column="id",
            text_column="utt",
            sources=list(
                BenchmarkConfig.from_dict({
                    "dataset": {
                        "id_column": "id",
                        "text_column": "utt",
                        "limit": 3,
                        "sources": [
                            {"parquet": str(fr), "language": "fr-FR"},
                            {"parquet": str(de), "language": "de-DE"},
                        ],
                    }
                }).dataset.resolved_sources()
            ),
        )
        clips = load_clips(config)

        assert len(clips) == 6, "the limit is per language, not for the whole run"
        assert sum(1 for c in clips if c.language == "fr-FR") == 3


class TestPerLanguageScoring:
    """Error rates from different languages must never be pooled."""

    def test_languages_are_summarized_separately(self) -> None:
        results = [
            _result("p", "fr-FR", "bonjour le monde", "bonjour le monde"),
            _result("p", "de-DE", "guten tag mein freund", "guten abend mein feind"),
        ]
        summaries = {s.language: s for s in summarize(results, "en-US")}

        assert set(summaries) == {"fr-FR", "de-DE"}
        assert summaries["fr-FR"].error_rate == pytest.approx(0.0)
        assert summaries["de-DE"].error_rate == pytest.approx(0.5)

    def test_pooling_would_have_hidden_a_bad_language(self) -> None:
        """The whole point: one weak language must stay visible."""
        results = [
            *[_result("p", "fr-FR", "un deux trois", "un deux trois") for _ in range(9)],
            _result("p", "vi-VN", "xin chao ban", "hoan toan sai"),
        ]
        summaries = {s.language: s for s in summarize(results, "en-US")}

        assert summaries["fr-FR"].error_rate == pytest.approx(0.0)
        assert summaries["vi-VN"].error_rate == pytest.approx(1.0), (
            "averaged across the run this would read as 10% and look healthy"
        )

    def test_metric_choice_follows_each_language(self) -> None:
        results = [
            _result("p", "en-US", "hello world", "hello world"),
            _result("p", "ja-JP", "東京都", "東京と"),
        ]
        summaries = {s.language: s for s in summarize(results, "en-US")}

        assert summaries["en-US"].metric_name == "WER"
        assert summaries["ja-JP"].metric_name == "CER", "a mixed run scores each language by its own appropriate metric"

    def test_overview_chart_needs_two_languages(self, tmp_path: Path) -> None:
        from audio_harness.plot import plot_overview

        single = stt_summary_frame([_result("p", "fr-FR", "bonjour", "bonjour")], "en-US")
        assert plot_overview(single, tmp_path / "single.png") is None, (
            "one language has its own per-language charts; an overview of it would just be a worse table"
        )

    def test_overview_chart_renders_a_mixed_metric_run(self, tmp_path: Path) -> None:
        """The overview must survive WER and CER side by side plus gaps."""
        from audio_harness.plot import plot_overview

        results = []
        for provider, err in [("p", ""), ("q", "x ")]:
            for language, ref in [
                ("fr-FR", "bonjour le monde entier"),
                ("ja-JP", "今日は良い天気です"),
                ("de-DE", "guten tag mein freund"),
            ]:
                if provider == "q" and language == "de-DE":
                    continue  # missing lane must render as a gap, not a crash
                result = _result(provider, language, ref, err + ref)
                result.finalize_s = 0.4 if provider == "p" else 0.9
                results.append(result)

        frame = stt_summary_frame(results, "en-US")
        output = plot_overview(frame, tmp_path / "overview.png")

        assert output is not None
        assert output.is_file(), "plot_overview must write the file it returns"
        assert output.stat().st_size > 10_000, "the chart should be a real render, not an empty canvas"

    def test_report_has_one_row_per_language(self) -> None:
        results = [
            _result("p", "fr-FR", "bonjour", "bonjour"),
            _result("p", "de-DE", "guten tag", "guten tag"),
            _result("q", "fr-FR", "bonjour", "bonsoir"),
        ]
        frame = stt_summary_frame(results, "en-US")

        assert frame.height == 3
        assert set(frame["language"]) == {"fr-FR", "de-DE"}
        assert frame["language"].to_list() == sorted(frame["language"].to_list()), (
            "rows group by language so two languages are not read as a ranking"
        )
