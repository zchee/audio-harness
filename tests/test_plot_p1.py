"""Tests for the P1 chart set: TTS latency, hallucination, entity overview.

Charts are the last mile of a metric — a number that never renders is a
number nobody acts on. These tests pin that each new P1 metric has a chart,
that missing data degrades to no chart rather than a misleading one, and
that the PNGs actually get written.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from audio_harness import plot
from audio_harness.metrics import HallucinationSummary


def _tts_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "provider": "p",
        "mode": "stream",
        "ttfb_p50_s": None,
        "ttfa_p50_s": None,
        "ttfb_cold_s": None,
        "gap_p99_s": None,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


class TestTtsLatencyChart:
    """TTFB→TTFA rows with cold markers and the stutter panel."""

    def test_renders_stream_lanes(self, tmp_path: Path) -> None:
        frame = _tts_frame([
            {
                "provider": "fast",
                "ttfb_p50_s": 0.10,
                "ttfa_p50_s": 0.30,
                "ttfb_cold_s": 0.90,
                "gap_p99_s": 0.02,
            },
            {
                "provider": "slow",
                "ttfb_p50_s": 0.25,
                "ttfa_p50_s": 0.28,
                "gap_p99_s": 0.15,
            },
            {"provider": "loaded", "mode": "stream x2", "ttfb_p50_s": 0.40},
        ])

        output = plot.plot_tts_latency(frame, tmp_path / "tts_latency.png")

        assert output is not None
        assert output.is_file()
        assert output.stat().st_size > 0

    def test_batch_only_frame_renders_nothing(self, tmp_path: Path) -> None:
        frame = _tts_frame([{"provider": "p", "mode": "batch", "ttfb_p50_s": 0.1}])

        assert plot.plot_tts_latency(frame, tmp_path / "none.png") is None

    def test_no_ttfb_renders_nothing(self, tmp_path: Path) -> None:
        frame = _tts_frame([{"provider": "p"}])

        assert plot.plot_tts_latency(frame, tmp_path / "none.png") is None

    def test_empty_frame_renders_nothing(self, tmp_path: Path) -> None:
        assert plot.plot_tts_latency(pl.DataFrame(), tmp_path / "none.png") is None


def _summary(
    provider: str,
    condition: str,
    *,
    fabricated: int = 0,
    phantom: int = 0,
    clips: int = 10,
) -> HallucinationSummary:
    return HallucinationSummary(
        provider=provider,
        mode="stream",
        language="en-US",
        condition=condition,
        clips=clips,
        fabricated_clips=fabricated,
        phantom_final_clips=phantom,
        inserted_words=fabricated * 4,
        audio_s=clips * 5.0,
    )


class TestHallucinationChart:
    """Fabrication and phantom-final rates per provider and condition."""

    def test_renders_both_panels(self, tmp_path: Path) -> None:
        summaries = [
            _summary("a", "silence", fabricated=3, phantom=2),
            _summary("a", "trailing_silence", fabricated=1),
            _summary("b", "silence", fabricated=0, phantom=0),
            _summary("b", "noise", fabricated=5, phantom=4),
        ]

        output = plot.plot_hallucination(summaries, tmp_path / "hallucination.png")

        assert output is not None
        assert output.is_file()
        assert output.stat().st_size > 0

    def test_speech_only_conditions_render_nothing(self, tmp_path: Path) -> None:
        """The lane exists for the synthetic conditions; plain corpus runs
        must not produce a hallucination chart out of nothing."""
        summaries = [_summary("a", "speech", fabricated=1)]

        assert plot.plot_hallucination(summaries, tmp_path / "none.png") is None

    def test_wrong_mode_renders_nothing(self, tmp_path: Path) -> None:
        summaries = [_summary("a", "silence", fabricated=1)]

        assert plot.plot_hallucination(summaries, tmp_path / "none.png", mode="batch") is None


def _stt_row(
    provider: str,
    language: str,
    error: float,
    *,
    entity: float | None = None,
) -> dict[str, object]:
    return {
        "provider": provider,
        "mode": "stream",
        "language": language,
        "metric": "WER",
        "failures": 0,
        "error_rate": error,
        "finalize_p50_s": 0.4,
        "ent_err[number]": entity,
    }


class TestOverviewEntityPanel:
    """The overview grows an entity panel only when annotations exist."""

    def test_annotated_frame_renders_three_panels(self, tmp_path: Path) -> None:
        frame = pl.DataFrame([
            _stt_row("a", "de-DE", 0.10, entity=0.05),
            _stt_row("a", "fr-FR", 0.12, entity=0.20),
            _stt_row("b", "de-DE", 0.20, entity=None),
            _stt_row("b", "fr-FR", 0.25, entity=0.30),
        ])

        output = plot.plot_overview(frame, tmp_path / "overview.png")

        assert output is not None
        assert output.is_file()
        assert output.stat().st_size > 0

    def test_unannotated_frame_still_renders_two_panels(self, tmp_path: Path) -> None:
        rows = [
            {k: v for k, v in _stt_row("a", "de-DE", 0.10).items() if "ent" not in k},
            {k: v for k, v in _stt_row("a", "fr-FR", 0.12).items() if "ent" not in k},
            {k: v for k, v in _stt_row("b", "de-DE", 0.20).items() if "ent" not in k},
        ]
        frame = pl.DataFrame(rows)

        output = plot.plot_overview(frame, tmp_path / "overview.png")

        assert output is not None
        assert output.is_file()

    def test_single_language_renders_nothing(self, tmp_path: Path) -> None:
        frame = pl.DataFrame([_stt_row("a", "de-DE", 0.1, entity=0.1)])

        assert plot.plot_overview(frame, tmp_path / "none.png") is None
