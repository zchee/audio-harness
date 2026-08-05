"""Tests for decoding, chunking and real-time pacing."""

from __future__ import annotations

import math
import time
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from audio_harness.audio import (
    BYTES_PER_SAMPLE,
    chunk_pcm,
    decode_audio_duration,
    load_clip,
    pace_chunks,
    pcm_duration_s,
    wrap_wav,
)
from audio_harness.types import AudioClip


def _tone(path: Path, *, seconds: float, rate: int, channels: int = 1) -> Path:
    """Write a sine tone so tests exercise real decoding, not a stub."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    wave_data = 0.5 * np.sin(2 * np.pi * 440 * t)
    if channels > 1:
        wave_data = np.column_stack([wave_data] * channels)
    sf.write(path, wave_data.astype("float32"), rate, subtype="PCM_16")
    return path


class TestLoadClip:
    """Decoding must produce canonical mono 16 kHz PCM regardless of input."""

    def test_loads_and_reports_duration(self, tmp_path: Path) -> None:
        path = _tone(tmp_path / "tone.wav", seconds=1.0, rate=16000)
        clip = load_clip(path, clip_id="c1", reference="hi", language="en-US")

        assert clip.sample_rate == 16000
        assert clip.duration_s == pytest.approx(1.0, abs=0.01)
        assert len(clip.pcm) == 16000 * BYTES_PER_SAMPLE
        assert clip.reference == "hi"
        assert clip.clip_id == "c1"

    def test_resamples_to_target_rate(self, tmp_path: Path) -> None:
        path = _tone(tmp_path / "tone48.wav", seconds=1.0, rate=48000)
        clip = load_clip(path, clip_id="c", reference=None, language="en-US")

        assert clip.sample_rate == 16000
        assert clip.duration_s == pytest.approx(1.0, abs=0.02), (
            "resampling must preserve wall-clock duration"
        )

    def test_downmixes_stereo_to_mono(self, tmp_path: Path) -> None:
        path = _tone(tmp_path / "stereo.wav", seconds=0.5, rate=16000, channels=2)
        clip = load_clip(path, clip_id="c", reference=None, language="en-US")

        assert len(clip.pcm) == int(16000 * 0.5) * BYTES_PER_SAMPLE

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_clip(tmp_path / "nope.wav", clip_id="c", reference=None, language="en")


class TestChunking:
    """Chunk boundaries must be exact so sequence numbers stay meaningful."""

    def _clip(self, seconds: float, rate: int = 16000) -> AudioClip:
        samples = int(rate * seconds)
        return AudioClip(
            clip_id="c",
            pcm=b"\x00\x00" * samples,
            sample_rate=rate,
            duration_s=seconds,
            reference=None,
            language="en-US",
            source_path="<memory>",
        )

    def test_chunks_are_exact_and_cover_everything(self) -> None:
        clip = self._clip(1.0)
        chunks = list(chunk_pcm(clip, 20))

        assert len(chunks) == 50, "1s of audio in 20ms frames"
        assert all(len(c) == 640 for c in chunks), "20ms at 16kHz mono s16le"
        assert b"".join(chunks) == clip.pcm

    def test_trailing_partial_chunk_is_preserved(self) -> None:
        clip = self._clip(0.05)
        chunks = list(chunk_pcm(clip, 20))

        assert len(chunks) == 3
        assert len(chunks[-1]) == 320, "final 10ms is short, not padded or dropped"
        assert b"".join(chunks) == clip.pcm


class TestPacing:
    """Streaming latency is meaningless unless audio arrives at 1x speed."""

    async def test_realtime_pacing_takes_about_the_clip_duration(self) -> None:
        clip = TestChunking()._clip(0.4)
        started = time.perf_counter()
        chunks = [chunk async for chunk in pace_chunks(clip, 20, realtime=True)]
        elapsed = time.perf_counter() - started

        assert len(chunks) == 20
        assert elapsed == pytest.approx(0.4, abs=0.15), (
            "paced streaming must track wall clock; feeding faster would "
            "measure throughput and report fictitiously low latency"
        )

    async def test_disabling_pacing_runs_far_faster(self) -> None:
        clip = TestChunking()._clip(2.0)
        started = time.perf_counter()
        chunks = [chunk async for chunk in pace_chunks(clip, 20, realtime=False)]
        elapsed = time.perf_counter() - started

        assert len(chunks) == 100
        assert elapsed < 0.5, "unpaced mode must not sleep"

    async def test_deadlines_do_not_accumulate_drift(self) -> None:
        clip = TestChunking()._clip(0.6)
        started = time.perf_counter()
        stamps = []
        async for _ in pace_chunks(clip, 20, realtime=True):
            stamps.append(time.perf_counter() - started)

        expected_last = (len(stamps) - 1) * 0.02
        assert stamps[-1] == pytest.approx(expected_last, abs=0.08), (
            "per-chunk sleeps would compound scheduler jitter into drift"
        )


class TestDurations:
    """Duration maths feeds real-time factor and cost estimates."""

    def test_pcm_duration(self) -> None:
        assert pcm_duration_s(b"\x00\x00" * 16000, 16000) == pytest.approx(1.0)
        assert pcm_duration_s(b"", 16000) == 0.0

    def test_wrap_wav_round_trips(self) -> None:
        pcm = b"\x01\x02" * 8000
        blob = wrap_wav(pcm, 16000)

        assert blob[:4] == b"RIFF"
        import io

        with wave.open(io.BytesIO(blob)) as handle:
            assert handle.getnchannels() == 1
            assert handle.getframerate() == 16000
            assert handle.getsampwidth() == BYTES_PER_SAMPLE
            assert handle.readframes(handle.getnframes()) == pcm

    def test_decode_duration_for_raw_pcm(self) -> None:
        duration = decode_audio_duration(
            b"\x00\x00" * 24000, encoding="pcm_s16le", sample_rate=24000
        )
        assert duration == pytest.approx(1.0)

    def test_decode_duration_reads_container_header(self, tmp_path: Path) -> None:
        pcm = b"\x00\x00" * 24000
        duration = decode_audio_duration(
            wrap_wav(pcm, 24000), encoding="wav", sample_rate=8000
        )
        assert duration == pytest.approx(1.0), (
            "the container's own rate must win over the requested rate"
        )

    def test_unparseable_payload_reports_zero_not_a_crash(self) -> None:
        duration = decode_audio_duration(
            b"not audio", encoding="mp3", sample_rate=24000
        )
        assert duration == 0.0

    def test_empty_payload_reports_zero(self) -> None:
        assert (
            decode_audio_duration(b"", encoding="pcm_s16le", sample_rate=16000) == 0.0
        )
        assert not math.isnan(
            decode_audio_duration(b"", encoding="mp3", sample_rate=16000)
        )
