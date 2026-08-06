"""Tests for the Distill-MOS perceptual regression guardrail.

Distill-MOS inference is heavy (it pulls in torch) and, per the module's own
warning, its scores are only meaningful as a within-provider drift signal —
never as a cross-provider ranking. Every test below except the one gated on
``AUDIO_HARNESS_TEST_DISTILLMOS`` injects a stub scorer, so the aggregation,
baseline and alerting logic is verified without paying the model-load cost or
depending on what a specific checkpoint happens to predict.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from audio_harness.runner import write_tts_results
from audio_harness.tts_quality import (
    REGRESSION_THRESHOLD,
    ClipScore,
    ProviderMosSummary,
    apply_baseline,
    load_baseline,
    run_guardrail,
    save_baseline,
    score_directory,
    score_file,
    score_results_file,
    score_waveform,
    summarize,
)
from audio_harness.types import Mode, TtsResult


def _wav(path: Path, *, seconds: float = 0.2, rate: int = 16000) -> Path:
    """Write a minimal valid WAV so ``score_file`` has something to decode."""
    import soundfile as sf

    samples = np.zeros(int(rate * seconds), dtype="float32")
    sf.write(path, samples, rate, subtype="PCM_16")
    return path


class TestProviderFromFilename:
    """Provider keys can themselves contain hyphens, so a naive split is wrong."""

    def test_recovers_provider_around_the_mode_marker(self, tmp_path: Path) -> None:
        _wav(tmp_path / "cartesia-sonic35-batch-p1.wav")

        scores = score_directory(tmp_path, score_fn=lambda _p: 3.0)

        assert scores == [
            ClipScore(
                path=str(tmp_path / "cartesia-sonic35-batch-p1.wav"),
                provider="cartesia-sonic35",
                mos=3.0,
            )
        ]

    def test_stream_marker_is_also_recognized(self, tmp_path: Path) -> None:
        _wav(tmp_path / "deepgram-aura2-stream-p1.wav")

        scores = score_directory(tmp_path, score_fn=lambda _p: 3.0)

        assert scores[0].provider == "deepgram-aura2"

    def test_unrecognized_name_scores_with_empty_provider(self, tmp_path: Path) -> None:
        _wav(tmp_path / "not-the-expected-shape.wav")

        scores = score_directory(tmp_path, score_fn=lambda _p: 3.0)

        assert scores[0].provider == ""


class TestScoreDirectory:
    """Directory scoring groups by provider and stays in filename order."""

    def test_scores_every_wav_file_in_order(self, tmp_path: Path) -> None:
        _wav(tmp_path / "vendorA-batch-p1.wav")
        _wav(tmp_path / "vendorA-batch-p2.wav")
        _wav(tmp_path / "vendorB-stream-p1.wav")
        given = {
            "vendorA-batch-p1.wav": 4.0,
            "vendorA-batch-p2.wav": 3.0,
            "vendorB-stream-p1.wav": 2.0,
        }

        scores = score_directory(tmp_path, score_fn=lambda p: given[p.name])

        assert [s.mos for s in scores] == [4.0, 3.0, 2.0]

    def test_ignores_non_wav_files(self, tmp_path: Path) -> None:
        _wav(tmp_path / "vendorA-batch-p1.wav")
        (tmp_path / "notes.txt").write_text("irrelevant")

        scores = score_directory(tmp_path, score_fn=lambda _p: 3.0)

        assert len(scores) == 1

    def test_empty_directory_yields_nothing(self, tmp_path: Path) -> None:
        assert score_directory(tmp_path, score_fn=lambda _p: 3.0) == []


class TestScoreResultsFile:
    """The results-file route reads saved audio through the runner's own paths."""

    def _result(
        self, provider: str, mode: Mode, prompt_id: str, *, error: str | None = None
    ) -> TtsResult:
        # audio_s must be nonzero: read_tts_results() drops the raw audio
        # bytes and TtsResult.ok falls back to audio_s as evidence a run
        # actually produced audio (see runner.py:read_tts_results).
        return TtsResult(
            provider=provider,
            prompt_id=prompt_id,
            mode=mode,
            audio=b"\x00\x00" * 8000,
            sample_rate=16000,
            audio_s=0.25,
            error=error,
        )

    def test_scores_only_successful_saved_audio(self, tmp_path: Path) -> None:
        results = [
            self._result("vendorA", Mode.BATCH, "p1"),
            self._result("vendorA", Mode.BATCH, "p2", error="boom"),
            self._result("vendorB", Mode.STREAM, "p1"),
        ]
        path = write_tts_results(results, tmp_path, save_audio=True)

        scores = score_results_file(path, score_fn=lambda _p: 4.2)

        assert {s.provider for s in scores} == {"vendorA", "vendorB"}, (
            "the errored result must not contribute a score"
        )
        assert len(scores) == 2

    def test_run_without_saved_audio_yields_nothing(self, tmp_path: Path) -> None:
        results = [self._result("vendorA", Mode.BATCH, "p1")]
        path = write_tts_results(results, tmp_path, save_audio=False)

        assert score_results_file(path, score_fn=lambda _p: 4.2) == []

    def test_moved_or_deleted_audio_is_skipped_not_errored(
        self, tmp_path: Path
    ) -> None:
        results = [self._result("vendorA", Mode.BATCH, "p1")]
        path = write_tts_results(results, tmp_path, save_audio=True)
        for wav_file in (path.parent / "audio").glob("*.wav"):
            wav_file.unlink()

        assert score_results_file(path, score_fn=lambda _p: 4.2) == []


class TestSummarize:
    """The corpus mean is a plain average of every clip's score."""

    def test_averages_by_provider(self) -> None:
        scores = [
            ClipScore(path="a1", provider="vendorA", mos=4.0),
            ClipScore(path="a2", provider="vendorA", mos=2.0),
            ClipScore(path="b1", provider="vendorB", mos=5.0),
        ]

        summaries = {s.provider: s for s in summarize(scores)}

        assert summaries["vendorA"].mean_mos == pytest.approx(3.0)
        assert summaries["vendorA"].clips == 2
        assert summaries["vendorB"].mean_mos == pytest.approx(5.0)

    def test_no_scores_yields_no_summaries(self) -> None:
        assert summarize([]) == []


class TestBaselinePersistence:
    """The baseline file round-trips through a plain provider -> mean map."""

    def test_missing_baseline_file_is_empty(self, tmp_path: Path) -> None:
        assert load_baseline(tmp_path / "nope.json") == {}

    def test_save_then_load_round_trips(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "baseline.json"
        summaries = [
            ProviderMosSummary(provider="vendorA", clips=3, mean_mos=3.7),
            ProviderMosSummary(provider="vendorB", clips=1, mean_mos=4.1),
        ]

        save_baseline(baseline_path, summaries)

        assert load_baseline(baseline_path) == {"vendorA": 3.7, "vendorB": 4.1}

    def test_save_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "nested" / "baseline.json"

        save_baseline(baseline_path, [ProviderMosSummary("vendorA", 1, 3.0)])

        assert baseline_path.is_file()


class TestApplyBaseline:
    """Baseline lookup is per provider and leaves unknown providers unset."""

    def test_attaches_matching_baseline(self) -> None:
        summaries = [ProviderMosSummary(provider="vendorA", clips=2, mean_mos=3.0)]

        result = apply_baseline(summaries, {"vendorA": 3.5})

        assert result[0].baseline_mos == 3.5

    def test_provider_without_a_baseline_stays_none(self) -> None:
        summaries = [ProviderMosSummary(provider="vendorA", clips=2, mean_mos=3.0)]

        result = apply_baseline(summaries, {"vendorB": 3.5})

        assert result[0].baseline_mos is None


class TestAlertThreshold:
    """The alert is a tripwire against a provider's own history, not a rank."""

    def test_drop_past_threshold_alerts(self) -> None:
        summary = ProviderMosSummary(
            provider="vendorA", clips=1, mean_mos=3.0, baseline_mos=3.31
        )
        assert summary.alert

    def test_drop_exactly_at_threshold_does_not_alert(self) -> None:
        summary = ProviderMosSummary(
            provider="vendorA",
            clips=1,
            mean_mos=3.0,
            baseline_mos=3.0 + REGRESSION_THRESHOLD,
        )
        assert not summary.alert

    def test_improvement_never_alerts(self) -> None:
        summary = ProviderMosSummary(
            provider="vendorA", clips=1, mean_mos=4.5, baseline_mos=3.0
        )
        assert not summary.alert

    def test_no_baseline_never_alerts(self) -> None:
        summary = ProviderMosSummary(provider="vendorA", clips=1, mean_mos=1.0)
        assert not summary.alert
        assert summary.delta is None


class TestRunGuardrail:
    """The orchestration entry point routes by source type and can refresh state."""

    def test_directory_source_scores_and_compares(self, tmp_path: Path) -> None:
        _wav(tmp_path / "vendorA-batch-p1.wav")
        baseline_path = tmp_path / "baseline.json"
        save_baseline(baseline_path, [ProviderMosSummary("vendorA", 5, 4.0)])

        summaries = run_guardrail(
            tmp_path, baseline_path=baseline_path, score_fn=lambda _p: 3.0
        )

        assert len(summaries) == 1
        assert summaries[0].alert, "3.0 vs a 4.0 baseline is a 1.0-point drop"

    def test_results_file_source_is_detected_by_is_file(self, tmp_path: Path) -> None:
        results = [
            TtsResult(
                provider="vendorA",
                prompt_id="p1",
                mode=Mode.BATCH,
                audio=b"\x00\x00" * 8000,
                sample_rate=16000,
                audio_s=0.25,
            )
        ]
        path = write_tts_results(results, tmp_path, save_audio=True)
        baseline_path = tmp_path / "baseline.json"

        summaries = run_guardrail(
            path, baseline_path=baseline_path, score_fn=lambda _p: 4.0
        )

        assert summaries[0].provider == "vendorA"
        assert summaries[0].baseline_mos is None

    def test_update_baseline_persists_this_runs_means(self, tmp_path: Path) -> None:
        _wav(tmp_path / "vendorA-batch-p1.wav")
        baseline_path = tmp_path / "baseline.json"

        run_guardrail(
            tmp_path,
            baseline_path=baseline_path,
            update_baseline=True,
            score_fn=lambda _p: 3.8,
        )

        assert load_baseline(baseline_path) == {"vendorA": 3.8}

    def test_without_update_baseline_the_file_is_left_untouched(
        self, tmp_path: Path
    ) -> None:
        _wav(tmp_path / "vendorA-batch-p1.wav")
        baseline_path = tmp_path / "baseline.json"
        save_baseline(baseline_path, [ProviderMosSummary("vendorA", 1, 4.0)])

        run_guardrail(tmp_path, baseline_path=baseline_path, score_fn=lambda _p: 1.0)

        assert load_baseline(baseline_path) == {"vendorA": 4.0}


@pytest.mark.skipif(
    not os.environ.get("AUDIO_HARNESS_TEST_DISTILLMOS"),
    reason="set AUDIO_HARNESS_TEST_DISTILLMOS=1 to run live Distill-MOS "
    "inference (requires the guardrail-mos optional dependency)",
)
class TestDistillMosLive:
    """Live inference through the pinned Distill-MOS checkpoint.

    A pure sine tone is not treated as high quality by this predictor — it is
    trained on speech and, per the plan's own risk register, single-utterance
    MOS predictors collapse out-of-domain (arXiv:2506.19441). This test only
    asserts the model runs and actually distinguishes two different signals,
    not that "clean" scores higher than "noisy"; the guardrail only ever
    compares a provider's audio against its own history, never one signal
    class against another.
    """

    def test_scores_are_in_range_and_distinguish_signals(self, tmp_path: Path) -> None:
        rate = 16000
        t = np.linspace(0, 2.0, rate * 2, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        rng = np.random.default_rng(0)
        noise = np.clip(0.5 * rng.standard_normal(len(t)), -1.0, 1.0).astype(np.float32)

        tone_mos = score_waveform(tone, rate)
        noise_mos = score_waveform(noise, rate)

        assert 1.0 <= tone_mos <= 5.0
        assert 1.0 <= noise_mos <= 5.0
        assert tone_mos != pytest.approx(noise_mos, abs=1e-3)

    def test_score_file_decodes_and_resamples(self, tmp_path: Path) -> None:
        path = _wav(tmp_path / "tone.wav", seconds=1.0, rate=48000)

        mos = score_file(path)

        assert 1.0 <= mos <= 5.0
