"""Tests for the SNR robustness matrix.

The matrix's promise is comparability: five mixtures of a clip that differ
only in noise gain, ids that survive the results JSONL, and an AUC whose
arithmetic can be checked by hand. Each test pins one leg of that promise —
no network, no MUSAN, all material generated in-place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pytest
import soundfile as sf

from audio_harness.config import SourceConfig
from audio_harness.dataset import load_source
from audio_harness.snr import (
    SNR_LEVELS,
    is_telephony,
    render_snr_markdown,
    snr_level_of,
    summarize_snr,
    synthesize_snr_source,
)
from audio_harness.types import Mode, SttResult


RATE = 16000


@pytest.fixture
def base_manifest(tmp_path: Path) -> Path:
    """Two quiet tone utterances behind a legacy audio-path manifest."""
    records = []
    for index in range(2):
        t = np.linspace(0, 2.0, RATE * 2, endpoint=False)
        tone = (0.1 * np.sin(2 * np.pi * (200 + 60 * index) * t)).astype("float32")
        path = tmp_path / f"b{index}.wav"
        sf.write(path, tone, RATE)
        records.append({"audio": path.name, "text": "ein test satz", "id": f"b{index}"})
    manifest = tmp_path / "bases.jsonl"
    manifest.write_bytes(b"\n".join(orjson.dumps(r) for r in records) + b"\n")
    return manifest


@pytest.fixture
def noise_dir(tmp_path: Path) -> Path:
    """One deterministic white-noise recording."""
    rng = np.random.default_rng(7)
    noise = (0.05 * rng.standard_normal(RATE * 5)).astype("float32")
    directory = tmp_path / "noise"
    directory.mkdir()
    sf.write(directory / "white.wav", noise, RATE)
    return directory


def _snr_source(base_manifest: Path, noise_dir: Path) -> SourceConfig:
    return SourceConfig(
        synthetic="snr",
        manifest=str(base_manifest),
        language="de-DE",
        limit=2,
        sample_seed=7,
        noise_dir=str(noise_dir),
    )


class TestMatrixGeneration:
    """Five mixtures per base, differing only in noise gain."""

    def test_every_base_appears_at_every_level(self, base_manifest: Path, noise_dir: Path) -> None:
        clips = synthesize_snr_source(_snr_source(base_manifest, noise_dir))

        assert len(clips) == 2 * len(SNR_LEVELS)
        assert [c.clip_id for c in clips[:5]] == [
            "snr+20-b0",
            "snr+10-b0",
            "snr+05-b0",
            "snr+00-b0",
            "snr-05-b0",
        ]
        assert all(c.reference == "ein test satz" for c in clips)
        assert all(c.language == "de-DE" for c in clips)

    def test_noise_grows_as_snr_falls(self, base_manifest: Path, noise_dir: Path) -> None:
        clips = synthesize_snr_source(_snr_source(base_manifest, noise_dir))
        speech = np.frombuffer(
            load_source(SourceConfig(manifest=str(base_manifest), language="de-DE", limit=1))[0].pcm,
            dtype="<i2",
        ).astype(np.float32)

        residuals = []
        for clip in clips[:5]:
            mixed = np.frombuffer(clip.pcm, dtype="<i2").astype(np.float32)
            residuals.append(float(np.sqrt(np.mean((mixed - speech) ** 2))))

        assert residuals == sorted(residuals), "levels are ordered +20 to -5; noise energy must rise monotonically"

    def test_one_noise_draw_is_shared_across_levels(self, base_manifest: Path, noise_dir: Path) -> None:
        """The five residuals must be the same noise at different gains, so a
        level-to-level WER delta can only be the SNR's doing."""
        clips = synthesize_snr_source(_snr_source(base_manifest, noise_dir))
        speech = np.frombuffer(
            load_source(SourceConfig(manifest=str(base_manifest), language="de-DE", limit=1))[0].pcm,
            dtype="<i2",
        ).astype(np.float32)

        loud = np.frombuffer(clips[4].pcm, dtype="<i2").astype(np.float32) - speech
        quiet = np.frombuffer(clips[0].pcm, dtype="<i2").astype(np.float32) - speech

        cosine = float(np.dot(loud, quiet) / (np.linalg.norm(loud) * np.linalg.norm(quiet)))
        assert cosine > 0.99

    def test_dispatches_through_the_dataset_layer(self, base_manifest: Path, noise_dir: Path) -> None:
        clips = load_source(_snr_source(base_manifest, noise_dir))
        assert len(clips) == 10


class TestTelephony:
    """The 8 kHz round-trip removes exactly the narrowband-lost content."""

    def _tone_manifest(self, tmp_path: Path, frequency: float) -> Path:
        t = np.linspace(0, 1.0, RATE, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * frequency * t)).astype("float32")
        path = tmp_path / "tone.wav"
        sf.write(path, tone, RATE)
        manifest = tmp_path / "tone.jsonl"
        manifest.write_bytes(orjson.dumps({"audio": path.name, "text": "ton", "id": "t0"}) + b"\n")
        return manifest

    def test_high_band_content_is_removed(self, tmp_path: Path) -> None:
        manifest = self._tone_manifest(tmp_path, 6000.0)
        clips = synthesize_snr_source(SourceConfig(synthetic="telephony", manifest=str(manifest), limit=1))

        samples = np.frombuffer(clips[0].pcm, dtype="<i2").astype(np.float32) / 32767.0
        assert clips[0].clip_id == "tel8k-t0"
        assert float(np.sqrt(np.mean(samples**2))) < 0.01, "a 6 kHz tone cannot survive an 8 kHz sample-rate round-trip"

    def test_voice_band_content_survives(self, tmp_path: Path) -> None:
        manifest = self._tone_manifest(tmp_path, 1000.0)
        clips = synthesize_snr_source(SourceConfig(synthetic="telephony", manifest=str(manifest), limit=1))

        samples = np.frombuffer(clips[0].pcm, dtype="<i2").astype(np.float32) / 32767.0
        assert float(np.sqrt(np.mean(samples**2))) > 0.15


class TestConditionParsing:
    """Conditions ride the clip id and survive the results JSONL."""

    def test_levels_round_trip_through_ids(self) -> None:
        tests = {
            "snr+20-clip": 20.0,
            "snr+05-clip": 5.0,
            "snr+00-clip": 0.0,
            "snr-05-clip": -5.0,
        }
        for clip_id, expected in tests.items():
            assert snr_level_of(clip_id) == expected, clip_id

    def test_non_matrix_ids_are_ignored(self) -> None:
        assert snr_level_of("lowsnr-clip") is None
        assert snr_level_of("plain-clip") is None
        assert is_telephony("tel8k-clip")
        assert not is_telephony("snr+20-clip")


def _result(
    clip_id: str,
    text: str,
    *,
    finalize_s: float | None = None,
    error: str | None = None,
) -> SttResult:
    result = SttResult(provider="p1", clip_id=clip_id, mode=Mode.STREAM, text=text, error=error)
    result.audio_s = 2.0
    result.finalize_s = finalize_s
    result.raw["reference"] = "a b c d"
    result.raw["language"] = "de-DE"
    return result


class TestSummarize:
    """Per-level rates, hand-checkable AUC, and the endpointing hook."""

    def test_auc_matches_hand_computation(self) -> None:
        results = [
            _result("snr+20-b0", "a b c d"),
            _result("snr+05-b0", "a b c x"),
            _result("snr-05-b0", "a b x y"),
        ]

        summary = summarize_snr(results, "de-DE")[0]

        assert summary.rate(20.0) == 0.0
        assert summary.rate(5.0) == pytest.approx(0.25)
        assert summary.rate(-5.0) == pytest.approx(0.5)
        # Trapezoids over [-5, 5] and [5, 20], normalized by the 25 dB span.
        assert summary.wer_auc == pytest.approx(((0.5 + 0.25) / 2 * 10 + (0.25 + 0.0) / 2 * 15) / 25)

    def test_failures_count_per_level(self) -> None:
        results = [
            _result("snr-05-b0", "", error="timeout after 300s"),
            _result("snr-05-b1", "a b c d"),
        ]

        summary = summarize_snr(results, "de-DE")[0]

        assert summary.failures == {-5.0: 1}
        assert summary.rate(-5.0) == 0.0, "the surviving clip still scores"

    def test_finalize_degradation_from_clean_to_noisy(self) -> None:
        results = [
            _result("snr+20-b0", "a b c d", finalize_s=0.4),
            _result("snr-05-b0", "a b c d", finalize_s=0.9),
        ]

        summary = summarize_snr(results, "de-DE")[0]

        assert summary.finalize_degradation_s == pytest.approx(0.5), "noise cost half a second of turn-taking latency"

    def test_telephony_accumulates_separately(self) -> None:
        results = [
            _result("tel8k-b0", "a b c x"),
            _result("snr+20-b0", "a b c d"),
        ]

        summary = summarize_snr(results, "de-DE")[0]

        assert summary.telephony is not None
        assert summary.telephony.rate == pytest.approx(0.25)
        assert summary.rate(20.0) == 0.0

    def test_non_snr_results_are_invisible(self) -> None:
        assert summarize_snr([_result("plain-clip", "a b c d")], "de-DE") == []

    def test_markdown_renders_levels_auc_and_telephony(self) -> None:
        results = [
            _result("snr+20-b0", "a b c d"),
            _result("snr-05-b0", "a b x y"),
            _result("tel8k-b0", "a b c x"),
        ]

        markdown = render_snr_markdown(summarize_snr(results, "de-DE"))

        assert "WER +20 dB" in markdown
        assert "WER -5 dB" in markdown
        assert "WER AUC" in markdown
        assert "8 kHz WER" in markdown
        assert "25.00%" in markdown
