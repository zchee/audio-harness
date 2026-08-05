"""Tests for turn-latency accounting and transient-failure handling."""

from __future__ import annotations

import numpy as np
import pytest

from audio_harness.audio import detect_speech_end_s
from audio_harness.runner import _is_transient, _rebase_finalize
from audio_harness.stt.ws import StreamProtocolError
from audio_harness.types import AudioClip, Mode, Partial, SttResult


def _clip(duration_s: float, speech_end_s: float) -> AudioClip:
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00",
        sample_rate=16000,
        duration_s=duration_s,
        reference="hello",
        language="en-US",
        source_path="<memory>",
        speech_end_s=speech_end_s,
    )


def _result(final_at_s: float | None, eof_finalize: float | None = None) -> SttResult:
    partials = [Partial(t_s=0.5, text="hel", is_final=False)]
    if final_at_s is not None:
        partials.append(Partial(t_s=final_at_s, text="hello", is_final=True))
    return SttResult(
        provider="p",
        clip_id="c1",
        mode=Mode.STREAM,
        text="hello",
        partials=partials,
        finalize_s=eof_finalize,
    )


class TestSpeechEndDetection:
    """Turn latency is measured from the last voiced frame."""

    def _tone_then_silence(self, tone_s: float, silence_s: float) -> np.ndarray:
        rate = 16000
        t = np.linspace(0, tone_s, int(rate * tone_s), endpoint=False)
        return np.concatenate(
            [0.5 * np.sin(2 * np.pi * 220 * t), np.zeros(int(rate * silence_s))]
        ).astype("float32")

    def test_finds_the_end_of_speech_before_trailing_silence(self) -> None:
        samples = self._tone_then_silence(1.0, 0.8)
        assert detect_speech_end_s(samples, 16000) == pytest.approx(1.0, abs=0.05)

    def test_speech_running_to_the_end_returns_the_full_duration(self) -> None:
        samples = self._tone_then_silence(1.0, 0.0)
        assert detect_speech_end_s(samples, 16000) == pytest.approx(1.0, abs=0.05)

    def test_all_silence_falls_back_to_full_duration(self) -> None:
        samples = np.zeros(16000, dtype="float32")
        assert detect_speech_end_s(samples, 16000) == pytest.approx(1.0, abs=0.05)

    def test_quiet_recording_is_not_treated_as_silence(self) -> None:
        """The threshold is relative to the clip's own peak, not absolute."""
        samples = self._tone_then_silence(1.0, 0.5) * 0.01
        assert detect_speech_end_s(samples, 16000) == pytest.approx(1.0, abs=0.05)


class TestRebaseFinalize:
    """Trailing silence must not be scored as free turn latency."""

    def test_measures_from_end_of_speech_not_end_of_file(self) -> None:
        clip = _clip(duration_s=4.0, speech_end_s=3.2)
        result = _result(final_at_s=3.6, eof_finalize=0.0)

        _rebase_finalize(result, clip, realtime=True)

        assert result.finalize_s == pytest.approx(0.4, abs=1e-6), (
            "the provider finalized 0.4s after the speaker stopped; measuring "
            "from the file end would have reported 0.0s and made aggressive "
            "endpointing look free"
        )

    def test_preserves_the_end_of_file_figure_for_comparison(self) -> None:
        clip = _clip(duration_s=4.0, speech_end_s=3.2)
        result = _result(final_at_s=3.6, eof_finalize=0.0)

        _rebase_finalize(result, clip, realtime=True)

        assert result.raw["finalize_from_eof_s"] == 0.0
        assert result.raw["speech_end_s"] == 3.2

    def test_finalizing_before_speech_ends_clamps_to_zero(self) -> None:
        clip = _clip(duration_s=4.0, speech_end_s=3.2)
        result = _result(final_at_s=3.0)

        _rebase_finalize(result, clip, realtime=True)

        assert result.finalize_s == 0.0, "latency is never negative"

    def test_no_final_transcript_yields_no_latency(self) -> None:
        clip = _clip(duration_s=4.0, speech_end_s=3.2)
        result = _result(final_at_s=None)

        _rebase_finalize(result, clip, realtime=True)

        assert result.finalize_s is None

    def test_unpaced_runs_keep_the_original_figure(self) -> None:
        """Without real-time pacing, elapsed time is not playback position."""
        clip = _clip(duration_s=4.0, speech_end_s=3.2)
        result = _result(final_at_s=3.6, eof_finalize=0.9)

        _rebase_finalize(result, clip, realtime=False)

        assert result.finalize_s == 0.9

    def test_missing_speech_end_keeps_the_original_figure(self) -> None:
        clip = _clip(duration_s=4.0, speech_end_s=0.0)
        result = _result(final_at_s=3.6, eof_finalize=0.7)

        _rebase_finalize(result, clip, realtime=True)

        assert result.finalize_s == 0.7


class TestTransientClassification:
    """Capacity refusals are the harness' fault, not the provider's."""

    def test_capacity_refusals_are_transient(self) -> None:
        tests = {
            "assemblyai": "Unauthorized Connection: Too many concurrent sessions",
            "speechmatics": "quota_exceeded: Concurrent Quota Exceeded",
            "rate limited": "You have reached your rate limit",
            "http 429": "HTTP 429: slow down",
            "http 503": "HTTP 503: temporarily unavailable",
        }
        for name, message in tests.items():
            assert _is_transient(StreamProtocolError(message)), name

    def test_real_errors_are_not_retried(self) -> None:
        tests = {
            "auth": "HTTP 401: invalid api key",
            "billing": "402: Organization balance exhausted",
            "bad request": "HTTP 400: language_code is not supported",
            "bad audio": "audio decoded to zero samples",
            "frame size": "Input Duration Violation: 20.0 ms",
        }
        for name, message in tests.items():
            assert not _is_transient(StreamProtocolError(message)), name

    def test_billing_exhaustion_is_not_mistaken_for_capacity(self) -> None:
        """An empty balance never resolves by waiting, so retrying wastes time."""
        assert not _is_transient(
            StreamProtocolError("soniox: 402: Organization balance exhausted")
        )
