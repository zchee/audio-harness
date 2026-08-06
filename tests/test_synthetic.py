"""Tests for deterministic synthetic clip generation.

The hallucination lane's conclusions are only trustworthy if the audio behind
them is exactly what the config promised: byte-identical across runs, silent
where it claims silence, and mixed at the SNR it claims to mix at.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl
import pytest
import soundfile as sf

from audio_harness.config import BenchmarkConfig, ConfigError, SourceConfig
from audio_harness.dataset import DatasetError, load_clips, load_source
from audio_harness.synthetic import (
    NOISE_LEVEL_DBFS,
    condition_of,
    mix_at_snr,
    silence_clip,
    synthesize_source,
)


RATE = 16000


def _tone(seconds: float, *, amplitude: float = 0.2, hz: float = 440.0) -> np.ndarray:
    """A sine burst followed by trailing silence, standing in for speech."""
    t = np.arange(int(RATE * seconds)) / RATE
    voiced = (amplitude * np.sin(2 * math.pi * hz * t)).astype(np.float32)
    return np.concatenate([voiced, np.zeros(int(RATE * 0.2), dtype=np.float32)])


def _tone_wav(seconds: float) -> bytes:
    """Encode the tone as WAV bytes for a parquet corpus cell."""
    import io

    buffer = io.BytesIO()
    sf.write(buffer, _tone(seconds), RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _speech_corpus(path: Path, rows: int = 4) -> Path:
    """Write a tiny parquet corpus of tone 'utterances' with references."""
    pl.DataFrame({
        "sample_id": [f"clip-{i:03d}" for i in range(rows)],
        "audio": [{"bytes": _tone_wav(0.4 + i * 0.1), "path": None} for i in range(rows)],
        "transcription": [f"utterance number {i}" for i in range(rows)],
    }).write_parquet(path)
    return path


def _noise_dir(path: Path, files: int = 2, seconds: float = 3.0) -> Path:
    """Write seeded uniform noise files, standing in for MUSAN."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(99)
    for index in range(files):
        samples = rng.uniform(-0.3, 0.3, int(RATE * seconds)).astype(np.float32)
        sf.write(path / f"noise-{index}.wav", samples, RATE)
    return path


class TestSilence:
    """Pure silence is the cheapest hallucination bait there is."""

    def test_generates_the_requested_condition_set(self) -> None:
        source = SourceConfig(synthetic="silence", limit=5, duration_s=1.0)
        clips = synthesize_source(source, sample_rate=RATE)

        assert [c.clip_id for c in clips] == [f"silence-{i:03d}" for i in range(5)]
        assert all(c.duration_s == pytest.approx(1.0) for c in clips)
        assert all(set(c.pcm) == {0} for c in clips), "silence must be digital zero"

    def test_reference_is_empty_not_missing(self) -> None:
        clip = silence_clip(0, duration_s=0.5, sample_rate=RATE, language="en-US")
        assert clip.reference == "", (
            "the correct transcript for silence is the empty string; None "
            "would read as 'no reference collected' and skip scoring"
        )


class _ErrorCase(TypedDict):
    """One synthetic-source misconfiguration case."""

    source: SourceConfig
    match: str


class TestNoise:
    """Noise-only clips must be reproducible cuts of the noise corpus."""

    def test_same_seed_produces_identical_audio(self, tmp_path: Path) -> None:
        source = SourceConfig(
            synthetic="noise",
            limit=4,
            duration_s=1.0,
            noise_dir=str(_noise_dir(tmp_path / "noise")),
            sample_seed=42,
        )
        first = [c.pcm for c in synthesize_source(source, sample_rate=RATE)]
        second = [c.pcm for c in synthesize_source(source, sample_rate=RATE)]

        assert first == second, "a pinned seed must give byte-identical clips"
        assert len({bytes(pcm) for pcm in first}) > 1, "different indices must cut different segments"

    def test_clips_are_normalized_to_the_documented_level(self, tmp_path: Path) -> None:
        source = SourceConfig(
            synthetic="noise",
            limit=2,
            duration_s=1.0,
            noise_dir=str(_noise_dir(tmp_path / "noise")),
            sample_seed=7,
        )
        for clip in synthesize_source(source, sample_rate=RATE):
            samples = np.frombuffer(clip.pcm, dtype="<i2").astype(np.float32) / 32767.0
            dbfs = 20 * math.log10(float(np.sqrt(np.mean(samples**2))))
            assert dbfs == pytest.approx(NOISE_LEVEL_DBFS, abs=0.5)
            assert clip.reference == ""

    def test_error_cases(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        tests: dict[str, _ErrorCase] = {
            "missing noise_dir names the fetch tool": {
                "source": SourceConfig(synthetic="noise", limit=2),
                "match": "fetch_musan",
            },
            "nonexistent directory": {
                "source": SourceConfig(synthetic="noise", limit=2, noise_dir=str(tmp_path / "gone")),
                "match": "noise_dir not found",
            },
            "directory without audio": {
                "source": SourceConfig(synthetic="noise", limit=2, noise_dir=str(empty)),
                "match": "no audio files",
            },
            "missing limit": {
                "source": SourceConfig(synthetic="noise", noise_dir=str(tmp_path / "noise")),
                "match": "positive limit",
            },
            "unknown kind": {
                "source": SourceConfig(synthetic="reverb", limit=2),
                "match": "unknown synthetic source kind",
            },
        }
        for case in tests.values():
            with pytest.raises(DatasetError, match=case["match"]):
                synthesize_source(case["source"], sample_rate=RATE)


class TestTrailingSilence:
    """Extended trailing silence baits post-speech fabrication."""

    def test_extends_base_utterances(self, tmp_path: Path) -> None:
        source = SourceConfig(
            synthetic="trailing_silence",
            parquet=str(_speech_corpus(tmp_path / "c.parquet", rows=3)),
            limit=3,
            trailing_silence_s=2.0,
        )
        bases = load_source(SourceConfig(parquet=source.parquet, limit=3), sample_rate=RATE)
        clips = synthesize_source(source, sample_rate=RATE)

        assert [c.clip_id for c in clips] == [f"trailsil-{b.clip_id}" for b in bases]
        for base, clip in zip(bases, clips, strict=True):
            assert clip.duration_s == pytest.approx(base.duration_s + 2.0, abs=0.01)
            assert clip.reference == base.reference, "any text beyond the base reference must count as insertion"
            tail = clip.pcm[-int(RATE * 1.5) * 2 :]
            assert set(tail) == {0}, "the appended tail must be digital silence"
            assert clip.speech_end_s <= base.duration_s + 0.05, (
                "appending silence must not move the detected end of speech"
            )

    def test_requires_a_base_corpus(self) -> None:
        with pytest.raises(DatasetError, match="base utterances"):
            synthesize_source(SourceConfig(synthetic="trailing_silence", limit=3), sample_rate=RATE)


class TestLowSnr:
    """Speech below the noise floor must sit at exactly the promised SNR."""

    def test_mix_hits_the_target_snr(self) -> None:
        speech = _tone(1.0, amplitude=0.05)
        rng = np.random.default_rng(3)
        noise = rng.uniform(-0.02, 0.02, len(speech)).astype(np.float32)

        mixed = mix_at_snr(speech, noise, snr_db=-10.0, sample_rate=RATE)
        noise_part = mixed - speech
        active_rms = 0.05 / math.sqrt(2)
        achieved = 20 * math.log10(active_rms / float(np.sqrt(np.mean(noise_part**2))))
        assert achieved == pytest.approx(-10.0, abs=0.3), (
            "SNR must be computed against active speech, not whole-clip RMS"
        )

    def test_clipping_normalization_preserves_the_mix(self) -> None:
        speech = _tone(0.5, amplitude=0.9)
        noise = np.ones(len(speech), dtype=np.float32) * 0.5

        mixed = mix_at_snr(speech, noise, snr_db=-10.0, sample_rate=RATE)
        assert float(np.max(np.abs(mixed))) <= 1.0

    def test_end_to_end_is_deterministic_and_keeps_references(self, tmp_path: Path) -> None:
        source = SourceConfig(
            synthetic="low_snr",
            parquet=str(_speech_corpus(tmp_path / "c.parquet", rows=2)),
            limit=2,
            snr_db=-10.0,
            noise_dir=str(_noise_dir(tmp_path / "noise")),
            sample_seed=11,
        )
        first = synthesize_source(source, sample_rate=RATE)
        second = synthesize_source(source, sample_rate=RATE)

        assert [c.pcm for c in first] == [c.pcm for c in second]
        assert [c.clip_id for c in first] == ["lowsnr-clip-000", "lowsnr-clip-001"]
        assert first[0].reference == "utterance number 0"
        assert first[0].pcm != load_source(SourceConfig(parquet=source.parquet, limit=1), sample_rate=RATE)[0].pcm, (
            "the mix must actually change the audio"
        )


class TestConditionOf:
    """Conditions must survive the trip through clip ids in results JSONL."""

    def test_round_trips_every_prefix(self) -> None:
        tests = {
            "silence-013": "silence",
            "noise-002": "noise",
            "trailsil-clip-001": "trailing_silence",
            "lowsnr-clip-040": "low_snr",
            "clip-001": None,
            "silencer": None,
        }
        for clip_id, expected in tests.items():
            assert condition_of(clip_id) == expected, clip_id


class TestConfigWiring:
    """Synthetic sources ride the existing dataset/config machinery."""

    def test_load_clips_mixes_synthetic_and_corpus_sources(self, tmp_path: Path) -> None:
        corpus = _speech_corpus(tmp_path / "c.parquet", rows=2)
        config = BenchmarkConfig.from_dict({
            "dataset": {
                "language": "en-US",
                "sources": [
                    {"synthetic": "silence", "limit": 2, "duration_s": 0.5},
                    {
                        "synthetic": "trailing_silence",
                        "limit": 2,
                        "trailing_silence_s": 1.0,
                        "parquet": str(corpus),
                    },
                ],
            }
        })
        clips = load_clips(config.dataset)
        assert [condition_of(c.clip_id) for c in clips] == [
            "silence",
            "silence",
            "trailing_silence",
            "trailing_silence",
        ]
        assert all(c.language == "en-US" for c in clips)

    def test_source_without_corpus_or_synthetic_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="synthetic"):
            BenchmarkConfig.from_dict({"dataset": {"sources": [{"language": "en-US"}]}})
