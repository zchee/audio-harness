"""Tests for the batch-versus-stream degradation delta view.

Streaming synthesis buys latency with a smaller lookahead; whatever that
costs never shows up in a latency table. These tests pin the comparison —
same provider, identical prompts, both transports — and the pause profiling
underneath it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from audio_harness import report, runner
from audio_harness.audio import PauseStats, measure_pauses, pcm16_to_float
from audio_harness.types import Mode, TtsResult


RATE = 24000
PROVIDER = "delta-fake-tts"
"""Deliberately unregistered: it forms its own family, so the registered
deepgram judge is cross-family and the ranked round-trip score applies."""


def _pcm(*segments: tuple[str, float]) -> bytes:
    """Mono 16-bit PCM built from ("tone" | "silence", seconds) segments."""
    parts: list[np.ndarray] = []
    for kind, seconds in segments:
        count = int(RATE * seconds)
        if kind == "tone":
            t = np.linspace(0, seconds, count, endpoint=False)
            parts.append(0.5 * np.sin(2 * np.pi * 220 * t))
        else:
            parts.append(np.zeros(count))
    samples = np.concatenate(parts).astype(np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class TestMeasurePauses:
    """Only silence *inside* the speech counts as a pause."""

    def _measure(self, *segments: tuple[str, float]) -> PauseStats:
        return measure_pauses(pcm16_to_float(_pcm(*segments)), RATE)

    def test_one_internal_pause_is_measured(self) -> None:
        stats = self._measure(("tone", 0.3), ("silence", 0.3), ("tone", 0.3))

        assert stats.count == 1
        assert stats.total_s == pytest.approx(0.3, abs=0.05)
        assert stats.longest_s == pytest.approx(0.3, abs=0.05)

    def test_continuous_speech_has_no_pauses(self) -> None:
        assert self._measure(("tone", 0.5)) == PauseStats(0.0, 0.0, 0)

    def test_leading_and_trailing_silence_are_not_pauses(self) -> None:
        stats = self._measure(("silence", 0.3), ("tone", 0.3), ("silence", 0.4))

        assert stats == PauseStats(0.0, 0.0, 0), (
            "edge silence belongs to latency and endpointing metrics; counting it here would double-charge the vendor"
        )

    def test_multiple_pauses_accumulate(self) -> None:
        stats = self._measure(
            ("tone", 0.2),
            ("silence", 0.2),
            ("tone", 0.2),
            ("silence", 0.4),
            ("tone", 0.2),
        )

        assert stats.count == 2
        assert stats.total_s == pytest.approx(0.6, abs=0.08)
        assert stats.longest_s == pytest.approx(0.4, abs=0.05)

    def test_articulation_gaps_do_not_count(self) -> None:
        """Stop consonants produce ~50 ms of silence in natural speech."""
        stats = self._measure(("tone", 0.2), ("silence", 0.06), ("tone", 0.2))

        assert stats.count == 0

    def test_all_silence_profiles_to_zero(self) -> None:
        assert self._measure(("silence", 0.5)) == PauseStats(0.0, 0.0, 0)


def _run(
    prompt_id: str,
    mode: Mode,
    *,
    audio: bytes,
    rt_text: str,
    provider: str = PROVIDER,
) -> TtsResult:
    result = TtsResult(
        provider=provider,
        prompt_id=prompt_id,
        mode=mode,
        audio=audio,
        sample_rate=RATE,
        audio_s=len(audio) / (RATE * 2),
        chars=11,
        total_s=0.5,
        raw={"text": "hello world"},
    )
    result.raw["roundtrip"] = [{"provider": "deepgram-nova3", "text": rt_text, "error": None}]
    return result


def _paired_results() -> list[TtsResult]:
    """Two prompts through both modes: stream is worse on every axis."""
    batch_audio = _pcm(("tone", 1.0))
    stream_audio = _pcm(("tone", 0.5), ("silence", 0.3), ("tone", 0.5))
    return [
        _run("p0", Mode.BATCH, audio=batch_audio, rt_text="hello world"),
        _run("p1", Mode.BATCH, audio=batch_audio, rt_text="hello world"),
        _run("p0", Mode.STREAM, audio=stream_audio, rt_text="hello word"),
        _run("p1", Mode.STREAM, audio=stream_audio, rt_text="hello word"),
    ]


class TestModeDeltaFrame:
    """One row per provider that ran identical prompts through both modes."""

    def test_stream_degradation_shows_in_every_delta(self) -> None:
        row = report.tts_mode_delta_frame(_paired_results(), "en-US").to_dicts()[0]

        assert row["provider"] == PROVIDER
        assert row["prompts"] == 2
        assert row["rt_batch"] == pytest.approx(0.0)
        assert row["rt_stream"] == pytest.approx(0.5)
        assert row["rt_delta"] == pytest.approx(0.5)
        assert row["dur_delta_s"] == pytest.approx(0.3, abs=0.02)
        assert row["pause_delta_s"] == pytest.approx(0.3, abs=0.05)
        assert row["longest_pause_delta_s"] == pytest.approx(0.3, abs=0.05)

    def test_single_mode_provider_yields_no_row(self) -> None:
        single = [run for run in _paired_results() if run.mode is Mode.BATCH]

        assert report.tts_mode_delta_frame(single, "en-US").is_empty()

    def test_only_shared_prompts_are_compared(self) -> None:
        """A prompt run in one mode only must not skew either side."""
        results = _paired_results()
        results.append(
            _run(
                "p2",
                Mode.STREAM,
                audio=_pcm(("tone", 5.0)),
                rt_text="completely wrong everything",
            )
        )

        row = report.tts_mode_delta_frame(results, "en-US").to_dicts()[0]

        assert row["prompts"] == 2
        assert row["rt_stream"] == pytest.approx(0.5), (
            "p2 exists only in the stream lane; its garbage transcript must not contaminate the paired comparison"
        )

    def test_cold_and_load_runs_are_excluded(self) -> None:
        results = _paired_results()
        cold = _run("p0", Mode.BATCH, audio=_pcm(("tone", 9.0)), rt_text="junk")
        cold.cold = True
        loaded = _run("p0", Mode.STREAM, audio=_pcm(("tone", 9.0)), rt_text="junk")
        loaded.raw["load"] = 2
        results += [cold, loaded]

        row = report.tts_mode_delta_frame(results, "en-US").to_dicts()[0]

        assert row["rt_batch"] == pytest.approx(0.0)
        assert row["rt_stream"] == pytest.approx(0.5)
        assert row["dur_delta_s"] == pytest.approx(0.3, abs=0.02)

    def test_failed_runs_are_excluded(self) -> None:
        results = _paired_results()
        results.append(TtsResult(provider=PROVIDER, prompt_id="p0", mode=Mode.STREAM, error="boom"))

        row = report.tts_mode_delta_frame(results, "en-US").to_dicts()[0]

        assert row["rt_stream"] == pytest.approx(0.5)

    def test_saved_results_recover_pauses_from_the_wav(self, tmp_path: Path) -> None:
        """Reloaded JSONL drops audio bytes; the saved WAV must fill in."""
        path = runner.write_tts_results(_paired_results(), tmp_path, save_audio=True)
        loaded = runner.read_tts_results(path)
        assert all(not run.audio for run in loaded), "precondition: bytes dropped"

        row = report.tts_mode_delta_frame(loaded, "en-US").to_dicts()[0]

        assert row["pause_delta_s"] == pytest.approx(0.3, abs=0.05)
        assert row["rt_delta"] == pytest.approx(0.5)

    def test_missing_audio_degrades_to_no_pause_stats(self, tmp_path: Path) -> None:
        path = runner.write_tts_results(_paired_results(), tmp_path, save_audio=False)
        loaded = runner.read_tts_results(path)

        row = report.tts_mode_delta_frame(loaded, "en-US").to_dicts()[0]

        assert row["pause_delta_s"] is None
        assert row["rt_delta"] == pytest.approx(0.5), (
            "round-trip and duration survive without audio; only the pause profile needs the waveform"
        )


class TestModeDeltaMarkdown:
    """The rendered table carries the delta columns."""

    def test_renders_provider_row_and_delta_columns(self) -> None:
        markdown = report.render_tts_mode_delta_markdown(report.tts_mode_delta_frame(_paired_results(), "en-US"))

        assert "Δ RT" in markdown
        assert "Δ Pause" in markdown
        assert "Δ Longest pause" in markdown
        assert PROVIDER in markdown
        assert "+50.00pt" in markdown

    def test_empty_frame_renders_the_placeholder(self) -> None:
        empty = report.tts_mode_delta_frame([], "en-US")

        assert report.render_tts_mode_delta_markdown(empty) == "_No results._"
