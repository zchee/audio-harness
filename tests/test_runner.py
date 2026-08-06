"""Tests for turn-latency accounting, transient handling and lane persistence."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

import numpy as np
import orjson
import pytest

from audio_harness import report, runner, stt, tts
from audio_harness.audio import detect_speech_end_s
from audio_harness.config import BenchmarkConfig, ProviderConfig, RunConfig
from audio_harness.runner import _is_transient, _rebase_finalize
from audio_harness.stt.ws import StreamProtocolError
from audio_harness.types import (
    AudioClip,
    Mode,
    Partial,
    SttResult,
    TtsPrompt,
    TtsResult,
)


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
        return np.concatenate([0.5 * np.sin(2 * np.pi * 220 * t), np.zeros(int(rate * silence_s))]).astype("float32")

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
        assert not _is_transient(StreamProtocolError("soniox: 402: Organization balance exhausted"))


class _SimulatedCrash(BaseException):
    """Escapes the runner's per-run Exception handling like a real abort.

    ``KeyboardInterrupt`` is what an operator's abort raises, but asyncio
    re-raises it into the event loop itself, which under pytest would abort
    the whole session; a private ``BaseException`` subclass gets the same
    "nothing catches this" treatment without that side effect.
    """


@stt.register
class _InstantStt(stt.SttProvider):
    """Returns a canned transcript immediately, for persistence tests."""

    key = "fake-persist-stt"
    vendor = "fake-persist-stt"
    supports_batch = True

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Return a deterministic transcript without touching a network."""
        result = self._result(clip, Mode.BATCH)
        result.text = f"transcript of {clip.clip_id}"
        result.total_s = 0.25
        return result


@stt.register
class _SlowStt(_InstantStt):
    """Finishes after the instant lane, so canonical ordering is exercised."""

    key = "fake-persist-slow-stt"
    vendor = "fake-persist-slow-stt"

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Delay long enough that the instant lane always flushes first."""
        await asyncio.sleep(0.05)
        return await super().transcribe_batch(clip)


class _CrashingStt(stt.SttProvider):
    """Blocks until the test releases it, then aborts the whole run.

    Registered by call rather than decorator so the name keeps its subclass
    type and the ``gate`` attribute stays visible to type checkers.
    """

    key = "fake-crash-stt"
    vendor = "fake-crash-stt"
    supports_batch = True
    gate: ClassVar[asyncio.Event | None] = None

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Wait for the gate, then die the way a killed process does."""
        gate = type(self).gate
        assert gate is not None, "the test must arm the gate before running"
        await gate.wait()
        raise _SimulatedCrash


stt.register(_CrashingStt)


@tts.register
class _InstantTts(tts.TtsProvider):
    """Synthesizes a canned clip immediately, for persistence tests."""

    key = "fake-persist-tts"
    vendor = "fake-persist-tts"
    supports_batch = True

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Return deterministic audio without touching a network."""
        result = self._result(prompt, Mode.BATCH)
        result.audio = b"\x00\x01" * 160
        result.audio_s = 0.02
        result.total_s = 0.2
        return result


@tts.register
class _SlowTts(_InstantTts):
    """Finishes after the instant lane, so canonical ordering is exercised."""

    key = "fake-persist-slow-tts"
    vendor = "fake-persist-slow-tts"

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Delay long enough that the instant lane always flushes first."""
        await asyncio.sleep(0.05)
        return await super().synthesize(prompt)


class _CrashingTts(tts.TtsProvider):
    """Blocks until the test releases it, then aborts the whole run.

    Registered by call rather than decorator so the name keeps its subclass
    type and the ``gate`` attribute stays visible to type checkers.
    """

    key = "fake-crash-tts"
    vendor = "fake-crash-tts"
    supports_batch = True
    gate: ClassVar[asyncio.Event | None] = None

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Wait for the gate, then die the way a killed process does."""
        gate = type(self).gate
        assert gate is not None, "the test must arm the gate before running"
        await gate.wait()
        raise _SimulatedCrash


tts.register(_CrashingTts)


def _bench_clip(clip_id: str) -> AudioClip:
    return AudioClip(
        clip_id=clip_id,
        pcm=b"\x00\x00" * 160,
        sample_rate=16000,
        duration_s=1.0,
        reference=f"reference for {clip_id}",
        language="en-US",
        source_path="<memory>",
    )


def _stt_config(providers: list[str], output_dir: Path, repeats: int = 1) -> BenchmarkConfig:
    return BenchmarkConfig(
        stt=[ProviderConfig(name=name, modes=["batch"]) for name in providers],
        run=RunConfig(repeats=repeats, warmup=0, settle_ms=0, output_dir=str(output_dir)),
    )


def _tts_config(providers: list[str], output_dir: Path, warmup: int = 0) -> BenchmarkConfig:
    return BenchmarkConfig(
        tts=[ProviderConfig(name=name, modes=["batch"]) for name in providers],
        run=RunConfig(repeats=1, warmup=warmup, settle_ms=0, output_dir=str(output_dir)),
    )


async def _wait_for_snapshot(root: Path, name: str) -> Path:
    """Poll until a run under ``root`` has flushed its first lane to disk."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10.0
    while loop.time() < deadline:
        found = [p for p in root.glob(f"*/{name}") if p.stat().st_size > 0]  # ruff: ignore[blocking-path-method-in-async-function] -- tmp-file poll at 10 ms cadence; blocking stat is negligible
        if found:
            return found[0]
        await asyncio.sleep(0.01)
    raise AssertionError(f"no {name} snapshot appeared under {root}")


def _legacy_stt_file(results: list[SttResult]) -> bytes:
    """The results file exactly as the pre-persistence writer produced it.

    A frozen copy of the old ``write_stt_results`` serialization, kept so any
    format or ordering drift in the shared record path fails loudly here.
    """
    lines: list[bytes] = []
    for result in results:
        lines.extend((
            orjson.dumps({
                "provider": result.provider,
                "clip_id": result.clip_id,
                "mode": str(result.mode),
                "text": result.text,
                "reference": result.raw.get("reference", ""),
                "reference_annotated": result.raw.get("reference_annotated"),
                "license": result.raw.get("license"),
                "gold_status": result.raw.get("gold_status"),
                "language": result.raw.get("language", ""),
                "audio_s": result.audio_s,
                "total_s": result.total_s,
                "ttft_s": result.ttft_s,
                "finalize_s": result.finalize_s,
                "rtf": result.rtf,
                "chunk_ms": result.raw.get("chunk_ms"),
                "speech_end_s": result.raw.get("speech_end_s"),
                "pauses": result.raw.get("pauses"),
                "ws_rtt_s": result.raw.get("ws_rtt_s"),
                "eou_source": result.raw.get("eou_source"),
                "endpoint_config": result.raw.get("endpoint_config"),
                "error": result.error,
                "partials": [
                    {"t_s": p.t_s, "text": p.text, "is_final": p.is_final, "kind": p.kind} for p in result.partials
                ],
            }),
            b"\n",
        ))
    return b"".join(lines)


def _legacy_tts_file(results: list[TtsResult]) -> bytes:
    """The results file exactly as the pre-persistence writer produced it.

    A frozen copy of the old ``write_tts_results`` serialization with
    ``save_audio=False``, so format drift in the shared path fails loudly.
    """
    lines: list[bytes] = []
    for result in results:
        lines.extend((
            orjson.dumps({
                "provider": result.provider,
                "prompt_id": result.prompt_id,
                "mode": str(result.mode),
                "chars": result.chars,
                "audio_s": result.audio_s,
                "ttfb_s": result.ttfb_s,
                "ttfa_s": result.ttfa_s,
                "gap_p99_s": result.gap_p99_s,
                "cold": result.cold,
                "chunk_t_s": result.chunk_t_s,
                "total_s": result.total_s,
                "rtf": result.rtf,
                "load": result.raw.get("load"),
                "input_streaming": result.raw.get("input_streaming"),
                "error": result.error,
                "text": result.raw.get("text", ""),
                "roundtrip": result.raw.get("roundtrip"),
                "audio_path": None,
            }),
            b"\n",
        ))
    return b"".join(lines)


class TestSttLanePersistence:
    """A killed STT run must keep every completed lane on disk."""

    async def test_completed_lane_survives_an_aborted_run(self, tmp_path: Path) -> None:
        _CrashingStt.gate = asyncio.Event()
        clips = [_bench_clip("c1"), _bench_clip("c2")]
        config = _stt_config(["fake-persist-stt", "fake-crash-stt"], tmp_path)

        task = asyncio.create_task(runner.run_stt(config, clips))
        snapshot = await _wait_for_snapshot(tmp_path, "stt-results.jsonl")
        _CrashingStt.gate.set()
        with pytest.raises(_SimulatedCrash):
            await task

        loaded = runner.read_stt_results(snapshot)
        assert [r.provider for r in loaded] == ["fake-persist-stt"] * 2, (
            "the finished lane's records must be on disk; the crashed lane's must not"
        )
        assert {r.clip_id for r in loaded} == {"c1", "c2"}
        assert loaded[0].text == "transcript of c1"
        assert loaded[0].raw["reference"] == "reference for c1"

    async def test_an_interrupted_file_renders_through_the_report(self, tmp_path: Path) -> None:
        _CrashingStt.gate = asyncio.Event()
        config = _stt_config(["fake-persist-stt", "fake-crash-stt"], tmp_path)

        task = asyncio.create_task(runner.run_stt(config, [_bench_clip("c1")]))
        snapshot = await _wait_for_snapshot(tmp_path, "stt-results.jsonl")
        _CrashingStt.gate.set()
        with pytest.raises(_SimulatedCrash):
            await task

        frame = report.stt_summary_frame(runner.read_stt_results(snapshot), "en-US")
        assert not frame.is_empty(), (
            "salvaging an interrupted run is the point; its file must flow "
            "through the same report path as a completed one"
        )

    async def test_completed_run_matches_the_legacy_file_exactly(self, tmp_path: Path) -> None:
        """Byte-for-byte compatibility with the pre-persistence writer."""
        clips = [_bench_clip("c1"), _bench_clip("c2")]
        config = _stt_config(["fake-persist-slow-stt", "fake-persist-stt"], tmp_path, repeats=2)

        results = await runner.run_stt(config, clips)
        snapshot = next(tmp_path.glob("*/stt-results.jsonl"))  # ruff: ignore[blocking-path-method-in-async-function] -- assertion-phase tmp_path I/O; the run under test has finished
        snapshot_lines = sorted(snapshot.read_bytes().splitlines())

        path = runner.write_stt_results(results, str(tmp_path))

        assert path == snapshot, "the canonical file replaces the lane snapshots"
        assert len(list(tmp_path.iterdir())) == 1, (  # ruff: ignore[blocking-path-method-in-async-function] -- assertion-phase tmp_path I/O; the run under test has finished
            "the end-of-run write must reuse the run's directory, not open a second timestamped one"
        )
        assert results[0].provider == "fake-persist-slow-stt", (
            "results stay in config order even when that lane finished last"
        )
        assert path.read_bytes() == _legacy_stt_file(results)
        assert sorted(path.read_bytes().splitlines()) == snapshot_lines, (
            "snapshots hold the same records; only lane order may differ"
        )
        assert not list(tmp_path.glob("*/*.tmp")), "no flush leftovers"  # ruff: ignore[blocking-path-method-in-async-function] -- assertion-phase tmp_path I/O; the run under test has finished


class TestTtsLanePersistence:
    """A killed TTS run must keep every completed lane on disk."""

    async def test_completed_lane_survives_an_aborted_run(self, tmp_path: Path) -> None:
        _CrashingTts.gate = asyncio.Event()
        prompts = [TtsPrompt(prompt_id="p1", text="hello there", language="en-US")]
        config = _tts_config(["fake-persist-tts", "fake-crash-tts"], tmp_path)

        task = asyncio.create_task(runner.run_tts(config, prompts))
        snapshot = await _wait_for_snapshot(tmp_path, "tts-results.jsonl")
        _CrashingTts.gate.set()
        with pytest.raises(_SimulatedCrash):
            await task

        loaded = runner.read_tts_results(snapshot)
        assert [r.provider for r in loaded] == ["fake-persist-tts"], (
            "the finished lane's records must be on disk; the crashed lane's must not"
        )
        assert loaded[0].raw["text"] == "hello there"
        frame = report.tts_summary_frame(loaded, "en-US")
        assert not frame.is_empty()

    async def test_completed_run_matches_the_legacy_file_exactly(self, tmp_path: Path) -> None:
        """Byte-for-byte compatibility with the pre-persistence writer."""
        prompts = [
            TtsPrompt(prompt_id="p1", text="hello there", language="en-US"),
            TtsPrompt(prompt_id="p2", text="general kenobi", language="en-US"),
        ]
        config = _tts_config(["fake-persist-slow-tts", "fake-persist-tts"], tmp_path, warmup=1)

        results = await runner.run_tts(config, prompts)
        snapshot = next(tmp_path.glob("*/tts-results.jsonl"))  # ruff: ignore[blocking-path-method-in-async-function] -- assertion-phase tmp_path I/O; the run under test has finished
        snapshot_lines = sorted(snapshot.read_bytes().splitlines())

        path = runner.write_tts_results(results, str(tmp_path), save_audio=False)

        assert path == snapshot, "the canonical file replaces the lane snapshots"
        assert len(list(tmp_path.iterdir())) == 1, (  # ruff: ignore[blocking-path-method-in-async-function] -- assertion-phase tmp_path I/O; the run under test has finished
            "the end-of-run write must reuse the run's directory, not open a second timestamped one"
        )
        assert any(r.cold for r in results), (
            "warmup runs are recorded, so the snapshot must carry the cold flag through unchanged"
        )
        assert path.read_bytes() == _legacy_tts_file(results)
        assert sorted(path.read_bytes().splitlines()) == snapshot_lines, (
            "snapshots hold the same records; only lane order may differ"
        )
        assert not list(tmp_path.glob("*/*.tmp")), "no flush leftovers"  # ruff: ignore[blocking-path-method-in-async-function] -- assertion-phase tmp_path I/O; the run under test has finished
