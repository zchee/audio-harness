"""Tests for the TTS latency lane: TTFA, chunk timing, warm/cold, feeding modes.

Time to first byte rewards a vendor for emitting a container header or leading
silence before any audible speech. These tests pin the honest metric — time to
first *audible* audio — and the stutter, warm/cold-split, input-streaming and
load-pass machinery around it.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
import os
from pathlib import Path

import httpx
import numpy as np
import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import report, runner, tts
from audio_harness.audio import detect_speech_onset_s, wav_data_offset, wrap_wav
from audio_harness.config import BenchmarkConfig
from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import cartesia
from audio_harness.tts.base import (
    ChunkTimeline,
    stamp_stream_timing,
    token_pieces,
)
from audio_harness.types import Mode, TtsPrompt, TtsResult


RATE = 24000


@pytest.fixture(autouse=True)
def _isolated_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep per-lane run snapshots out of the repository's results directory.

    ``runner.run_tts`` persists completed lanes under the configured
    ``output_dir`` — the relative ``results`` by default — the moment they
    finish. These tests drive the runner with fake providers, so the working
    directory moves into the test's own tmp dir to keep fake lanes from
    mixing into real benchmark artifacts.
    """
    monkeypatch.chdir(tmp_path)


def make_pcm(silence_s: float, tone_s: float, rate: int = RATE) -> bytes:
    """Mono 16-bit PCM: leading silence followed by a 220 Hz tone."""
    t = np.linspace(0, tone_s, int(rate * tone_s), endpoint=False)
    samples = np.concatenate([np.zeros(int(rate * silence_s)), 0.5 * np.sin(2 * np.pi * 220 * t)]).astype(np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def timeline_of(chunks: list[bytes], t_s: list[float]) -> ChunkTimeline:
    """Build a timeline with fabricated arrival times for deterministic tests."""
    timeline = ChunkTimeline()
    timeline.chunks = list(chunks)
    timeline.t_s = list(t_s)
    return timeline


def stream_result(provider: str = "fake-latency-tts") -> TtsResult:
    return TtsResult(
        provider=provider,
        prompt_id="p1",
        mode=Mode.STREAM,
        sample_rate=RATE,
        raw={"text": "hello world"},
    )


def stamped(chunks: list[bytes], t_s: list[float]) -> TtsResult:
    """Run fabricated chunks through the real stamping path."""
    result = stream_result()
    timeline = timeline_of(chunks, t_s)
    result.audio = timeline.audio
    result.audio_s = len(result.audio) / (RATE * 2)
    stamp_stream_timing(result, timeline)
    return result


class TestOnsetDetection:
    """The onset locates the first audible frame, not the first byte."""

    def test_silence_has_no_onset(self) -> None:
        assert detect_speech_onset_s(np.zeros(RATE, dtype=np.float32), RATE) is None

    def test_leading_silence_is_skipped(self) -> None:
        samples = np.frombuffer(make_pcm(0.2, 0.3), dtype="<i2").astype(np.float32)
        assert detect_speech_onset_s(samples / 32768.0, RATE) == pytest.approx(0.2, abs=0.05)

    def test_immediate_speech_starts_at_zero(self) -> None:
        samples = np.frombuffer(make_pcm(0.0, 0.3), dtype="<i2").astype(np.float32)
        assert detect_speech_onset_s(samples / 32768.0, RATE) == pytest.approx(0.0, abs=0.05)

    def test_quiet_audio_is_not_treated_as_silence(self) -> None:
        """The threshold is relative to the clip's own peak, not absolute."""
        samples = np.frombuffer(make_pcm(0.2, 0.3), dtype="<i2").astype(np.float32)
        quiet = samples / 32768.0 * 0.01
        assert detect_speech_onset_s(quiet, RATE) == pytest.approx(0.2, abs=0.05)


class TestWavHeaderOffset:
    """A WAV header must not read as an audible click at t=0."""

    def test_headerless_pcm_offsets_zero(self) -> None:
        assert wav_data_offset(make_pcm(0.0, 0.1)) == 0

    def test_wav_container_data_is_located(self) -> None:
        pcm = make_pcm(0.0, 0.1)
        wav = wrap_wav(pcm, RATE)
        offset = wav_data_offset(wav)
        assert offset > 0
        assert wav[offset:] == pcm

    def test_short_garbage_offsets_zero(self) -> None:
        assert wav_data_offset(b"RIFFxx") == 0


class TestStampStreamTiming:
    """TTFA, gap-P99 and chunk arrivals derived from one chunk record."""

    def test_leading_silence_separates_ttfa_from_ttfb(self) -> None:
        """AC3: 200 ms of leading silence yields TTFA - TTFB >= 180 ms."""
        result = stamped([make_pcm(0.2, 0.3)], [0.05])

        assert result.ttfb_s is not None
        assert result.ttfa_s is not None
        assert result.ttfb_s == pytest.approx(0.05)
        assert result.ttfa_s - result.ttfb_s >= 0.18, (
            "the first byte was silence; the audible onset is 200 ms of "
            "playback later, and TTFB alone would hide that padding"
        )
        assert result.ttfa_s == pytest.approx(0.25, abs=0.03)

    def test_late_audible_chunk_is_arrival_bound(self) -> None:
        """When the voiced chunk arrives late, its arrival sets TTFA."""
        result = stamped([make_pcm(0.2, 0.0), make_pcm(0.0, 0.1)], [0.05, 1.0])

        assert result.ttfa_s == pytest.approx(1.0, abs=0.03)

    def test_fast_delivery_is_playback_bound(self) -> None:
        """When everything arrives at once, playback of the silence gates TTFA."""
        result = stamped([make_pcm(0.2, 0.0), make_pcm(0.0, 0.1)], [0.05, 0.06])

        assert result.ttfa_s == pytest.approx(0.25, abs=0.03)

    def test_all_silence_has_no_ttfa(self) -> None:
        result = stamped([make_pcm(0.3, 0.0)], [0.05])

        assert result.ttfa_s is None
        assert result.ttfb_s == pytest.approx(0.05)

    def test_wav_wrapped_payload_still_finds_the_onset(self) -> None:
        """A vendor wrapping "raw" PCM in WAV must not score TTFA = TTFB."""
        result = stamped([wrap_wav(make_pcm(0.2, 0.3), RATE)], [0.05])

        assert result.ttfa_s == pytest.approx(0.25, abs=0.03)

    def test_no_chunks_leaves_latency_unset(self) -> None:
        result = stamped([], [])

        assert result.ttfb_s is None
        assert result.ttfa_s is None
        assert result.gap_p99_s is None
        assert result.chunk_t_s == []

    def test_gap_p99_interpolates_between_gaps(self) -> None:
        chunk = make_pcm(0.0, 0.05)
        result = stamped([chunk, chunk, chunk], [0.0, 0.1, 0.3])

        assert result.gap_p99_s == pytest.approx(0.199, abs=1e-6), (
            "gaps are 0.1 and 0.2; their 99th percentile interpolates to 0.199"
        )

    def test_chunk_arrivals_are_recorded(self) -> None:
        chunk = make_pcm(0.0, 0.05)
        result = stamped([chunk, chunk], [0.1, 0.2])

        assert result.chunk_t_s == [0.1, 0.2]

    def test_single_chunk_has_no_gap(self) -> None:
        result = stamped([make_pcm(0.0, 0.1)], [0.1])

        assert result.gap_p99_s is None


class TestTokenPieces:
    """Piece splitting must reassemble to the exact prompt text."""

    def test_pieces_concatenate_to_the_original(self) -> None:
        tests = {
            "plain words": "hello world again",
            "extra whitespace": "hello  world\nagain ",
            "single word": "hello",
            "japanese": "こんにちは 世界",
        }
        for name, text in tests.items():
            assert "".join(token_pieces(text)) == text, name

    def test_empty_text_yields_one_piece(self) -> None:
        assert token_pieces("") == [""]


@tts.register
class _FakeLatencyTts(tts.TtsProvider):
    """Streams a canned padded clip through the real timing helpers."""

    key = "fake-latency-tts"
    vendor = "fake-latency"
    supports_batch = True
    supports_stream = True

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        result = self._result(prompt, Mode.BATCH)
        result.audio = make_pcm(0.0, 0.1)
        result.audio_s = 0.1
        return result

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        result = self._result(prompt, Mode.STREAM)
        timeline = ChunkTimeline()
        timeline.add(make_pcm(0.2, 0.0))
        timeline.add(make_pcm(0.0, 0.1))
        result.audio = timeline.audio
        result.audio_s = 0.3
        stamp_stream_timing(result, timeline)
        return result


@tts.register
class _FakeIncrementalTts(_FakeLatencyTts):
    """Fake adapter whose protocol accepts streamed input text."""

    key = "fake-incremental-tts"
    vendor = "fake-incremental"
    supports_input_streaming = True

    async def synthesize_incremental(self, prompt: TtsPrompt, *, token_rate: float) -> TtsResult:
        result = await self.synthesize_stream(prompt)
        result.raw["input_streaming"] = True
        result.raw["token_rate"] = token_rate
        return result


def _prompts(count: int = 2) -> list[TtsPrompt]:
    return [TtsPrompt(prompt_id=f"p{index}", text=f"prompt number {index}", language="en-US") for index in range(count)]


def _config(name: str, **run_overrides: object) -> BenchmarkConfig:
    run = {"repeats": 1, "warmup": 1, "timeout_s": 10.0, **run_overrides}
    return BenchmarkConfig.from_dict({"tts": [{"name": name, "modes": ["stream"]}], "run": run})


class TestWarmColdSplit:
    """The warmup pass is recorded and flagged, never discarded."""

    async def test_warmup_runs_are_recorded_cold(self) -> None:
        config = _config("fake-latency-tts", repeats=2)
        prompts = _prompts(2)

        results = await runner.run_tts(config, prompts)

        cold = [r for r in results if r.cold]
        warm = [r for r in results if not r.cold]
        assert len(cold) == 1, "one warmup run, recorded rather than thrown away"
        assert cold[0].prompt_id == "p0"
        assert len(warm) == 4, "2 repeats by 2 prompts of measured runs"

    async def test_no_warmup_means_no_cold_runs(self) -> None:
        config = _config("fake-latency-tts", warmup=0)

        results = await runner.run_tts(config, _prompts(1))

        assert len(results) == 1
        assert not results[0].cold

    async def test_progress_counts_exclude_cold_runs(self) -> None:
        """The progress total promises repeats times prompts; cold runs are extra."""
        seen: list[bool] = []
        progress = runner.Progress(on_result=lambda p, m, ok: seen.append(ok))
        config = _config("fake-latency-tts", repeats=2)

        await runner.run_tts(config, _prompts(2), progress)

        assert len(seen) == 4

    async def test_streamed_results_carry_chunk_arrivals(self) -> None:
        """AC3: chunk_t_s is populated for every streamed result."""
        config = _config("fake-latency-tts", repeats=2)

        results = await runner.run_tts(config, _prompts(2))

        assert all(r.chunk_t_s for r in results)


class TestIncrementalRouting:
    """Input streaming is used where the protocol has it, and never faked."""

    async def test_supporting_adapter_gets_the_incremental_path(self) -> None:
        config = _config("fake-incremental-tts", warmup=0, tts_incremental_text=True)

        results = await runner.run_tts(config, _prompts(1))

        assert results[0].raw["input_streaming"] is True
        assert results[0].raw["token_rate"] == 40.0

    async def test_unsupporting_adapter_falls_back_and_says_so(self) -> None:
        config = _config("fake-latency-tts", warmup=0, tts_incremental_text=True)

        results = await runner.run_tts(config, _prompts(1))

        assert results[0].raw["input_streaming"] is False, (
            "a whole-prompt run in an incremental benchmark must be marked, "
            "or the report would compare feeding modes as if they were one"
        )

    async def test_disabled_incremental_leaves_no_marker(self) -> None:
        config = _config("fake-incremental-tts", warmup=0)

        results = await runner.run_tts(config, _prompts(1))

        assert "input_streaming" not in results[0].raw


class TestLoadPass:
    """The optional load repeat runs concurrently and is tagged apart."""

    async def test_load_runs_are_tagged_with_the_factor(self) -> None:
        config = _config("fake-latency-tts", warmup=0, tts_load_concurrency=2)

        results = await runner.run_tts(config, _prompts(1))

        sequential = [r for r in results if "load" not in r.raw]
        loaded = [r for r in results if r.raw.get("load") == 2]
        assert len(sequential) == 1
        assert len(loaded) == 2

    async def test_batch_mode_never_gets_a_load_pass(self) -> None:
        config = BenchmarkConfig.from_dict({
            "tts": [{"name": "fake-latency-tts", "modes": ["batch"]}],
            "run": {"repeats": 1, "warmup": 0, "tts_load_concurrency": 2},
        })

        results = await runner.run_tts(config, _prompts(1))

        assert len(results) == 1


class TestPersistence:
    """New fields survive the JSONL round trip; legacy files still load."""

    def test_write_then_read_preserves_latency_fields(self, tmp_path: Path) -> None:
        result = stamped([make_pcm(0.2, 0.3)], [0.05])
        result.cold = True
        result.raw["load"] = 2
        result.raw["input_streaming"] = True

        path = runner.write_tts_results([result], tmp_path, save_audio=False)
        loaded = runner.read_tts_results(path)[0]

        assert loaded.ttfa_s == pytest.approx(result.ttfa_s)
        assert loaded.gap_p99_s is None
        assert loaded.cold is True
        assert loaded.chunk_t_s == [0.05]
        assert loaded.raw["load"] == 2
        assert loaded.raw["input_streaming"] is True

    def test_legacy_record_loads_with_defaults(self, tmp_path: Path) -> None:
        record = {
            "provider": "deepgram-aura2",
            "prompt_id": "p1",
            "mode": "stream",
            "chars": 11,
            "audio_s": 0.1,
            "ttfb_s": 0.1,
            "total_s": 0.2,
            "rtf": 2.0,
            "error": None,
            "text": "hello world",
            "audio_path": None,
        }
        file = tmp_path / "tts-results.jsonl"
        file.write_bytes(orjson.dumps(record) + b"\n")

        loaded = runner.read_tts_results(file)[0]

        assert loaded.ttfa_s is None
        assert loaded.gap_p99_s is None
        assert loaded.cold is False
        assert loaded.chunk_t_s == []
        assert "load" not in loaded.raw


class TestReportView:
    """Warm/cold and load lanes render as promised in AC3."""

    def _warm(self, ttfb: float, ttfa: float, gap: float) -> TtsResult:
        result = stream_result()
        result.audio_s = 0.3
        result.ttfb_s = ttfb
        result.ttfa_s = ttfa
        result.gap_p99_s = gap
        result.total_s = 0.5
        return result

    def _results(self) -> list[TtsResult]:
        cold = self._warm(0.5, 0.7, 0.05)
        cold.cold = True
        loaded = self._warm(0.4, 0.6, 0.04)
        loaded.raw["load"] = 2
        return [self._warm(0.1, 0.3, 0.01), self._warm(0.2, 0.4, 0.03), cold, loaded]

    def test_warm_percentiles_exclude_cold_and_load_runs(self) -> None:
        frame = report.tts_summary_frame(self._results(), "en-US")
        rows = {row["mode"]: row for row in frame.to_dicts()}

        row = rows["stream"]
        assert row["ttfb_p50_s"] == pytest.approx(0.15)
        assert row["ttfa_p50_s"] == pytest.approx(0.35)
        assert row["gap_p99_s"] == pytest.approx(0.02)
        assert row["ttfb_cold_s"] == pytest.approx(0.5)

    def test_load_runs_form_their_own_lane(self) -> None:
        frame = report.tts_summary_frame(self._results(), "en-US")
        rows = {row["mode"]: row for row in frame.to_dicts()}

        assert rows["stream x2"]["ttfb_p50_s"] == pytest.approx(0.4)
        assert rows["stream x2"]["prompts"] == 1

    def test_markdown_renders_the_new_columns(self) -> None:
        markdown = report.render_tts_markdown(report.tts_summary_frame(self._results(), "en-US"))

        assert "TTFA p50" in markdown
        assert "Gap p99" in markdown
        assert "TTFB cold" in markdown
        assert "stream x2" in markdown


class FakeCartesiaServer:
    """Speaks the Cartesia WebSocket protocol shape the adapter expects."""

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.pcm = make_pcm(0.2, 0.1)

    async def __call__(self, socket: ServerConnection) -> None:
        async for frame in socket:
            message = orjson.loads(frame)
            self.messages.append(message)
            done = "continue" not in message or message.get("continue") is False
            if done:
                half = len(self.pcm) // 2 // 2 * 2
                for piece in (self.pcm[:half], self.pcm[half:]):
                    await socket.send(
                        orjson.dumps({
                            "type": "chunk",
                            "data": base64.b64encode(piece).decode(),
                        }).decode()
                    )
                await socket.send(orjson.dumps({"type": "done"}).decode())
                return


@pytest.fixture
async def cartesia_ws(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[FakeCartesiaServer]:
    """Run a fake Cartesia endpoint and point the adapter at it."""
    handler = FakeCartesiaServer()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(cartesia, "WS_URL", f"ws://127.0.0.1:{port}")
        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        monkeypatch.setenv("CARTESIA_VOICE_ID", "test-voice")
        yield handler


class TestCartesiaProtocol:
    """The adapter's wire behavior against a real local WebSocket."""

    PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")

    async def test_plain_stream_populates_chunk_timing(self, cartesia_ws: FakeCartesiaServer) -> None:
        adapter = tts.create("cartesia-sonic35")

        result = await adapter.synthesize_stream(self.PROMPT)

        assert result.ok, result.error
        assert len(result.chunk_t_s) == 2
        assert result.ttfb_s is not None
        assert result.ttfa_s is not None
        assert result.ttfa_s - result.ttfb_s >= 0.18, "the canned clip leads with 200 ms of silence; TTFA must see it"

    async def test_incremental_feeds_one_context_and_closes_it(self, cartesia_ws: FakeCartesiaServer) -> None:
        adapter = tts.create("cartesia-sonic35")

        result = await adapter.synthesize_incremental(self.PROMPT, token_rate=400.0)

        assert result.ok, result.error
        assert result.raw["input_streaming"] is True

        messages = cartesia_ws.messages
        pieces = token_pieces(self.PROMPT.text)
        assert len(messages) == len(pieces) + 1, "every piece plus the closing frame"
        assert {str(m["context_id"]) for m in messages} == {"cartesia-sonic35-p1-incremental"}, (
            "continuations only work inside one context"
        )
        assert [m["continue"] for m in messages] == [True] * len(pieces) + [False]
        assert messages[-1]["transcript"] == ""
        assert "".join(str(m["transcript"]) for m in messages[:-1]) == self.PROMPT.text
        assert result.audio_s > 0


class _UnreadByteStream(httpx.AsyncByteStream):
    """An async byte stream that has *not* been buffered at construction.

    ``httpx.Response(content=...)`` eagerly reads a ``ByteStream`` body the
    moment it is built, which would make an unread-response bug untestable —
    ``.text`` would never raise. A real network response streamed through
    ``client.stream()`` is unread until something calls ``aread()`` or
    iterates it, which is what this fake reproduces.
    """

    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body


class TestCartesiaBatchErrorHandling:
    """A real HTTP error must surface its body, not crash on an unread stream.

    ``synthesize`` reads the batch endpoint through ``http.stream`` (to time
    TTFB), so a vendor error response is still unread when ``raise_for_status``
    inspects it; without buffering the body first that raises
    ``httpx.ResponseNotRead`` instead of the vendor's actual error.
    """

    async def test_http_error_is_raised_with_the_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARTESIA_API_KEY", "test-key")
        monkeypatch.setenv("CARTESIA_VOICE_ID", "test-voice")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, stream=_UnreadByteStream(b'{"error": "invalid voice"}'))

        adapter = tts.create("cartesia-sonic35")
        adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with pytest.raises(ProviderHttpError, match="invalid voice"):
            await adapter.synthesize(TtsPrompt(prompt_id="p1", text="hello world", language="en-US"))


class TestDeepgramProtocol:
    """The ``/speak`` streaming endpoint: chunk assembly and error handling."""

    async def test_chunks_are_assembled_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        audio = make_pcm(0.0, 0.1)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=audio)

        adapter = tts.create("deepgram-aura2")
        adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        result = await adapter.synthesize_stream(TtsPrompt(prompt_id="p1", text="hello world", language="en-US"))

        assert result.ok, result.error
        assert result.audio == audio

    async def test_http_error_is_raised_with_the_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A real error must surface as ``ProviderHttpError``, not ``ResponseNotRead``."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, stream=_UnreadByteStream(b'{"err_msg": "invalid api key"}'))

        adapter = tts.create("deepgram-aura2")
        adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with pytest.raises(ProviderHttpError, match="invalid api key"):
            await adapter.synthesize_stream(TtsPrompt(prompt_id="p1", text="hello world", language="en-US"))


LIVE_FLAG = "AUDIO_HARNESS_TEST_TTS_LIVE"


def _needs(*names: str) -> pytest.MarkDecorator:
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(names)} not set")


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run one short live synthesis per vendor (fractions of a cent each)",
)
class TestLiveStreamSmoke:
    """One short prompt per adapter against the real vendor endpoint.

    Mocks confirm whatever timing the adapter claims; these verify the chunk
    timeline against real streams. Deliberately tiny — full runs need
    explicit approval.
    """

    PROMPT = TtsPrompt(
        prompt_id="smoke",
        text="The quick brown fox jumps over the lazy dog.",
        language="en-US",
    )

    async def _assert_stream(self, key: str) -> TtsResult:
        adapter = tts.create(key)
        try:
            result = await adapter.synthesize_stream(self.PROMPT)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.chunk_t_s, "a streamed result must carry chunk arrivals"
        assert result.ttfb_s is not None
        assert result.ttfb_s > 0
        if result.ttfa_s is not None:
            assert result.ttfa_s >= result.ttfb_s
        if len(result.chunk_t_s) >= 2:
            assert result.gap_p99_s is not None
        return result

    @_needs("CARTESIA_API_KEY", "CARTESIA_VOICE_ID")
    async def test_cartesia_stream(self) -> None:
        await self._assert_stream("cartesia-sonic35")

    @_needs("CARTESIA_API_KEY", "CARTESIA_VOICE_ID")
    async def test_cartesia_incremental(self) -> None:
        adapter = tts.create("cartesia-sonic35")
        result = await adapter.synthesize_incremental(self.PROMPT, token_rate=40.0)

        assert result.ok, result.error
        assert result.raw["input_streaming"] is True
        assert result.chunk_t_s
        assert result.audio_s > 0

    @_needs("DEEPGRAM_API_KEY")
    async def test_deepgram_stream(self) -> None:
        await self._assert_stream("deepgram-aura2")

    @_needs("GEMINI_API_KEY")
    async def test_gemini_stream(self) -> None:
        await self._assert_stream("gemini-tts")
