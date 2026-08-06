"""Tests for the hallucination/silence lane metrics (plan AC4).

Fabrication, phantom finals and decoder loops are scored from saved result
streams, so every test here builds results the same shape the runner persists
— including one that drives a real WebSocket round trip with a fabricating
server, because the metrics must read genuine event streams, not idealized
fixtures.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
import os
from pathlib import Path
from typing import TypedDict

import numpy as np
import orjson
import pytest
import soundfile as sf
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.server import ServerConnection, serve

from audio_harness.config import BenchmarkConfig, ProviderConfig, RunConfig
from audio_harness.metrics import (
    FABRICATION_MIN_RUN,
    HallucinationSummary,
    has_ngram_loop,
    insertion_run_lengths,
    phantom_final_count,
    summarize_hallucination,
)
from audio_harness.runner import read_stt_results, write_stt_results
from audio_harness.stt.base import StreamTimeline
from audio_harness.stt.ws import run_stream
from audio_harness.synthetic import low_snr_clip, noise_clip, silence_clip
from audio_harness.types import Mode, Partial, SttResult


FABRICATED = "thank you for watching please subscribe"


def _result(
    clip_id: str,
    text: str,
    *,
    reference: str = "",
    finals: tuple[str, ...] = (),
    audio_s: float = 8.0,
    error: str | None = None,
    language: str = "en-US",
) -> SttResult:
    """Build a result shaped like the runner records it."""
    result = SttResult(
        provider="mock",
        clip_id=clip_id,
        mode=Mode.STREAM,
        text=text,
        audio_s=audio_s,
        error=error,
    )
    result.raw = {"reference": reference, "language": language}
    result.partials = [Partial(t_s=float(i), text=final, is_final=True) for i, final in enumerate(finals)]
    return result


class _RunsCase(TypedDict):
    """One insertion-run detection case."""

    ref: str
    hyp: str
    language: str
    expected: list[int]


class TestInsertionRuns:
    """Run lengths separate scattered noise from invented phrases."""

    def test_run_detection(self) -> None:
        tests: dict[str, _RunsCase] = {
            "no speech makes the whole hypothesis one run": {
                "ref": "",
                "hyp": "thank you for watching",
                "language": "en-US",
                "expected": [4],
            },
            "empty hypothesis has no runs": {
                "ref": "",
                "hyp": "",
                "language": "en-US",
                "expected": [],
            },
            "perfect transcript has no runs": {
                "ref": "the cat sat",
                "hyp": "the cat sat",
                "language": "en-US",
                "expected": [],
            },
            "adjacent insertions merge into one run": {
                "ref": "the cat sat",
                "hyp": "the big red cat sat",
                "language": "en-US",
                "expected": [2],
            },
            "separated insertions stay separate runs": {
                "ref": "the cat sat on the mat",
                "hyp": "the big cat sat on the old mat",
                "language": "en-US",
                "expected": [1, 1],
            },
            "substitutions are not insertions": {
                "ref": "the cat sat",
                "hyp": "the dog sat",
                "language": "en-US",
                "expected": [],
            },
            "character languages count characters": {
                "ref": "",
                "hyp": "ありがとう",
                "language": "ja-JP",
                "expected": [5],
            },
        }
        for name, case in tests.items():
            runs = insertion_run_lengths(case["ref"], case["hyp"], case["language"])
            assert runs == case["expected"], name

    def test_normalization_matches_the_accuracy_metric(self) -> None:
        runs = insertion_run_lengths("room 101", "Room 101.", "en-US")
        assert runs == [], "formatting differences the WER forgives must not count as fabricated words either"


class _LoopCase(TypedDict):
    """One n-gram loop detection case."""

    text: str
    language: str
    expected: bool


class TestNgramLoop:
    """Loop detection targets runaway decoding, not natural repetition."""

    def test_loop_detection(self) -> None:
        tests: dict[str, _LoopCase] = {
            "classic whisper loop": {
                "text": "thank you thank you thank you",
                "language": "en-US",
                "expected": True,
            },
            "long single-word run": {
                "text": "so so so so so so",
                "language": "en-US",
                "expected": True,
            },
            "mundane doubled word is not a loop": {
                "text": "very very very",
                "language": "en-US",
                "expected": False,
            },
            "normal sentence": {
                "text": "the cat sat on the mat",
                "language": "en-US",
                "expected": False,
            },
            "empty text": {"text": "", "language": "en-US", "expected": False},
            "japanese phrase loop spans many characters": {
                "text": "ありがとうございました" * 3,
                "language": "ja-JP",
                "expected": True,
            },
            "japanese sentence without loops": {
                "text": "今日はいい天気ですね",
                "language": "ja-JP",
                "expected": False,
            },
            "loop embedded in surrounding text": {
                "text": "well i think thank you thank you thank you is all",
                "language": "en-US",
                "expected": True,
            },
        }
        for name, case in tests.items():
            assert has_ngram_loop(case["text"], case["language"]) is case["expected"], name


class _PhantomCase(TypedDict):
    """One phantom-final counting case."""

    result: SttResult
    expected: int


class TestPhantomFinals:
    """A final on silence is a commitment to text that never happened."""

    def test_counts_only_no_speech_clips(self) -> None:
        tests: dict[str, _PhantomCase] = {
            "finals with text on silence count": {
                "result": _result("silence-000", FABRICATED, finals=(FABRICATED,)),
                "expected": 1,
            },
            "empty finals are keepalives, not phantoms": {
                "result": _result("silence-001", "", finals=("", "  ")),
                "expected": 0,
            },
            "clips with a reference never count": {
                "result": _result("clip-000", "hello", reference="hello", finals=("hello",)),
                "expected": 0,
            },
            "multiple phantom finals all count": {
                "result": _result("noise-000", "uh", finals=("uh", "uh huh")),
                "expected": 2,
            },
        }
        for name, case in tests.items():
            assert phantom_final_count(case["result"]) == case["expected"], name


class TestSummarizeHallucination:
    """AC4: fabricating providers score above zero, silent ones exactly zero."""

    def test_fabricating_provider_scores_above_zero(self) -> None:
        results = [_result(f"silence-{i:03d}", FABRICATED, finals=(FABRICATED,)) for i in range(3)]
        (summary,) = summarize_hallucination(results, "en-US")

        assert summary.condition == "silence"
        assert summary.fabrication_rate is not None
        assert summary.fabrication_rate > 0
        assert summary.inserted_words == 3 * len(FABRICATED.split())
        assert summary.phantom_finals == 3
        assert summary.inserted_words_per_min == pytest.approx(len(FABRICATED.split()) / (8.0 / 60.0))

    def test_silent_provider_scores_exactly_zero(self) -> None:
        results = [_result(f"silence-{i:03d}", "") for i in range(3)]
        (summary,) = summarize_hallucination(results, "en-US")

        assert summary.fabrication_rate == 0.0
        assert summary.phantom_final_rate == 0.0
        assert summary.inserted_words == 0
        assert summary.loop_rate == 0.0

    def test_short_insertions_do_not_count_as_fabrication(self) -> None:
        below = " ".join(["word"] * (FABRICATION_MIN_RUN - 1))
        (summary,) = summarize_hallucination([_result("silence-000", below)], "en-US")
        assert summary.fabrication_rate == 0.0, "a run below the threshold is insertion noise, not fabrication"
        assert summary.inserted_words == FABRICATION_MIN_RUN - 1, "the words still count toward inserted-words/min"

    def test_conditions_are_separate_rows(self) -> None:
        results = [
            _result("silence-000", FABRICATED),
            _result("noise-000", ""),
            _result(
                "trailsil-clip-000",
                "hello world extra words here",
                reference="hello world",
            ),
            _result("lowsnr-clip-000", "hello world", reference="hello world"),
        ]
        summaries = {s.condition: s for s in summarize_hallucination(results, "en-US")}

        assert set(summaries) == {"silence", "noise", "trailing_silence", "low_snr"}
        assert summaries["silence"].fabrication_rate == 1.0
        assert summaries["noise"].fabrication_rate == 0.0
        assert summaries["trailing_silence"].fabrication_rate == 1.0
        assert summaries["low_snr"].fabrication_rate == 0.0

    def test_failures_are_counted_but_not_scored(self) -> None:
        results = [
            _result("silence-000", FABRICATED),
            _result("silence-001", "", error="timeout after 180s"),
        ]
        (summary,) = summarize_hallucination(results, "en-US")

        assert summary.clips == 2
        assert summary.failures == 1
        assert summary.scored == 1
        assert summary.fabrication_rate == 1.0, "a failed clip must not dilute the rate's denominator"

    def test_all_failures_yield_none_not_zero(self) -> None:
        (summary,) = summarize_hallucination([_result("silence-000", "", error="boom")], "en-US")
        assert summary.fabrication_rate is None

    def test_loop_rate_flags_looping_transcripts(self) -> None:
        results = [
            _result("noise-000", "thank you thank you thank you"),
            _result("noise-001", ""),
        ]
        (summary,) = summarize_hallucination(results, "en-US")
        assert summary.loop_rate == pytest.approx(0.5)

    def test_survives_the_results_jsonl_round_trip(self, tmp_path: Path) -> None:
        results = [
            _result("silence-000", FABRICATED, finals=(FABRICATED,)),
            _result(
                "trailsil-clip-000",
                "hello world and then some more words",
                reference="hello world",
            ),
            _result("noise-000", "", error="timeout after 180s"),
        ]
        path = write_stt_results(results, tmp_path)
        reloaded = summarize_hallucination(read_stt_results(path), "en-US")
        direct = summarize_hallucination(results, "en-US")

        def key(s: HallucinationSummary) -> tuple[str, str, str, str]:
            return (s.provider, s.mode, s.language, s.condition)

        assert sorted(map(key, reloaded)) == sorted(map(key, direct))
        for before, after in zip(sorted(direct, key=key), sorted(reloaded, key=key), strict=True):
            assert before == after, "re-scoring a saved run must not need the audio or the runner"


class TestConditionCounts:
    """AC4: the lane config pins the pre-registered condition counts."""

    def test_config_declares_20_20_50_50(self) -> None:
        config = BenchmarkConfig.from_yaml("configs/stt-hallucination.yaml")
        counts = {source.synthetic: source.limit for source in config.dataset.resolved_sources()}
        assert counts == {
            "silence": 20,
            "noise": 20,
            "trailing_silence": 50,
            "low_snr": 50,
        }

    def test_streaming_only_and_realtime(self) -> None:
        config = BenchmarkConfig.from_yaml("configs/stt-hallucination.yaml")
        assert all(entry.modes == ["stream"] for entry in config.stt), "fabrication is a streaming-endpoint behaviour"
        assert config.run.realtime, "endpointing-driven fabrication only exists under real-time pacing"


class FabricatingServer:
    """A WebSocket 'vendor' that invents a transcript for silent audio."""

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    async def __call__(self, socket: ServerConnection) -> None:
        try:
            async for frame in socket:
                if isinstance(frame, bytes):
                    continue
                with contextlib.suppress(orjson.JSONDecodeError):
                    if orjson.loads(frame).get("type") == "eos":
                        break
        except websockets.ConnectionClosed:
            return
        with contextlib.suppress(websockets.ConnectionClosed):
            await socket.send(orjson.dumps({"type": "transcript", "text": self.transcript, "final": True}).decode())
            await socket.send(orjson.dumps({"type": "done"}).decode())


@pytest.fixture
async def fabricating_server() -> AsyncIterator[str]:
    """Serve the fabricating vendor on an ephemeral port."""
    handler = FabricatingServer(FABRICATED)
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


class TestStreamPath:
    """The metrics must read an event stream a real socket produced."""

    async def test_fabrication_scores_from_a_live_stream(self, fabricating_server: str) -> None:
        clip = silence_clip(0, duration_s=0.4, sample_rate=16000, language="en-US")
        timeline = StreamTimeline()

        def handle_message(payload: object, timeline: StreamTimeline) -> bool:
            if not isinstance(payload, dict):
                return False
            if payload.get("type") == "done":
                return True
            if payload.get("type") == "transcript":
                timeline.record(str(payload["text"]), is_final=bool(payload.get("final")))
            return False

        async def eos(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "eos"}).decode())

        await run_stream(
            url=fabricating_server,
            headers={},
            clip=clip,
            chunk_ms=20,
            realtime=False,
            timeline=timeline,
            handle_message=handle_message,
            on_input_done=eos,
            finalize_timeout_s=5.0,
        )

        result = SttResult(
            provider="fake-ws",
            clip_id=clip.clip_id,
            mode=Mode.STREAM,
            text=timeline.last_final(),
            audio_s=clip.duration_s,
        )
        result.partials = timeline.partials
        result.raw = {"reference": clip.reference or "", "language": clip.language}

        (summary,) = summarize_hallucination([result], "en-US")
        assert summary.condition == "silence"
        assert summary.fabrication_rate == 1.0
        assert summary.phantom_finals >= 1
        assert summary.inserted_words == len(FABRICATED.split())


LIVE_FLAG = "AUDIO_HARNESS_TEST_HALLUCINATION_LIVE"


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG) or not os.environ.get("DEEPGRAM_API_KEY"),
    reason=f"live smoke needs {LIVE_FLAG}=1 and DEEPGRAM_API_KEY (3 clips, ~20 s of audio, fractions of a cent)",
)
class TestLiveSmoke:
    """Minimal real-vendor pass: do the metrics read genuine event streams?

    Deliberately tiny — one cheap vendor, three short clips — per the team's
    testing policy: mocks breed bugs, but a full lane run needs explicit
    approval.
    """

    async def test_three_clips_against_deepgram(self, tmp_path: Path) -> None:
        from audio_harness.runner import run_stt

        rate = 16000
        rng = np.random.default_rng(20260806)
        noise_dir = tmp_path / "noise"
        noise_dir.mkdir()
        sf.write(
            noise_dir / "n0.wav",
            rng.uniform(-0.3, 0.3, rate * 10).astype(np.float32),
            rate,
        )
        noise_files = [noise_dir / "n0.wav"]

        t = np.arange(rate) / rate
        tone = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        sf.write(tmp_path / "tone.wav", tone, rate)
        from audio_harness.audio import load_clip

        base = load_clip(
            tmp_path / "tone.wav",
            clip_id="clip-smoke",
            reference="",
            language="en-US",
        )

        clips = [
            silence_clip(0, duration_s=4.0, sample_rate=rate, language="en-US"),
            noise_clip(
                0,
                noise_files,
                duration_s=4.0,
                sample_rate=rate,
                seed=1,
                language="en-US",
            ),
            low_snr_clip(base, 0, noise_files, snr_db=-10.0, seed=1),
        ]
        config = BenchmarkConfig(
            stt=[ProviderConfig(name="deepgram-nova3", modes=["stream"])],
            run=RunConfig(
                repeats=1,
                warmup=0,
                timeout_s=60.0,
                settle_ms=0,
                output_dir=str(tmp_path),
            ),
        )

        results = await asyncio.wait_for(run_stt(config, clips), timeout=120.0)
        assert len(results) == 3
        assert all(r.ok for r in results), [r.error for r in results]

        path = write_stt_results(results, tmp_path)
        summaries = summarize_hallucination(read_stt_results(path), "en-US")
        conditions = {s.condition for s in summaries}
        assert conditions <= {"silence", "noise", "low_snr", "no_speech"}
        for summary in summaries:
            assert summary.fabrication_rate is not None
            assert 0.0 <= summary.fabrication_rate <= 1.0
            assert summary.inserted_words >= 0
