"""Protocol tests for the Gladia Solaria realtime speech-to-text adapter.

Two transports are pinned: the ``POST /v2/live`` session-init request (mocked
via ``httpx.MockTransport``, matching the ElevenLabs protocol-test pattern)
and the WebSocket it hands back a one-time URL for (a real local server,
matching the fake-server pattern in ``test_stream_driver.py``), since neither
shape is observable by driving the generic streaming driver alone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import orjson
import polars as pl
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import stt
from audio_harness.audio import load_clip_bytes
from audio_harness.stt import gladia
from audio_harness.stt.base import ProviderHttpError
from audio_harness.types import AudioClip, EventKind


def make_clip(seconds: float = 0.1, rate: int = 16000, language: str = "en-US") -> AudioClip:
    """Build a silent clip of a known duration."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="hello world",
        language=language,
        source_path="<memory>",
    )


LIVE_AUDIO = Path(__file__).parents[1] / "data/hf/stt-benchmark-data/data/train-00000-of-00001.parquet"


def make_live_clip() -> AudioClip:
    """Load one short real English utterance from the local benchmark corpus."""
    if not LIVE_AUDIO.is_file():
        raise FileNotFoundError(f"live smoke audio corpus not found: {LIVE_AUDIO}")
    row = pl.read_parquet(LIVE_AUDIO, n_rows=1).select("sample_id", "transcription", "audio").to_dicts()[0]
    audio = row["audio"]
    if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
        raise TypeError(f"live smoke row has no embedded audio bytes: {LIVE_AUDIO}")
    clip = load_clip_bytes(
        audio["bytes"],
        clip_id=str(row["sample_id"]),
        reference=str(row["transcription"]),
        language="en-US",
        source_path=f"{LIVE_AUDIO}#{row['sample_id']}",
    )
    if not 0.5 <= clip.duration_s <= 6.0:
        raise ValueError(f"live smoke clip must be 0.5-6.0 seconds, got {clip.duration_s:.3f}s")
    return clip


class FakeGladiaWs:
    """Speaks the ``v2/live`` WebSocket protocol shape the adapter expects.

    Mirrors the real service's documented flow: it waits for
    ``stop_recording``, then emits a partial, a bare ``speech_end`` marker, a
    final transcript, and ``post_final_transcript`` before the handler
    returns and the library closes the socket with code 1000.
    """

    def __init__(self) -> None:
        self.received_audio = bytearray()
        self.received_json: list[dict[str, Any]] = []

    async def __call__(self, socket: ServerConnection) -> None:
        async for frame in socket:
            if isinstance(frame, bytes):
                self.received_audio.extend(frame)
                continue
            message = orjson.loads(frame)
            self.received_json.append(message)
            if message.get("type") == "stop_recording":
                break

        for event in (
            {
                "session_id": "s1",
                "created_at": "2026-08-07T00:00:00Z",
                "type": "transcript",
                "data": {"id": "u1", "is_final": False, "utterance": {"text": "hello", "language": "en"}},
            },
            {
                "session_id": "s1",
                "created_at": "2026-08-07T00:00:00Z",
                "type": "speech_end",
                "data": {"time": 0.09, "channel": 0},
            },
            {
                "session_id": "s1",
                "created_at": "2026-08-07T00:00:00Z",
                "type": "transcript",
                "data": {"id": "u1", "is_final": True, "utterance": {"text": "hello world", "language": "en"}},
            },
            {
                "session_id": "s1",
                "created_at": "2026-08-07T00:00:00Z",
                "type": "post_final_transcript",
                "data": {},
            },
        ):
            await socket.send(orjson.dumps(event).decode())


@pytest.fixture
async def gladia_ws(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[gladia.GladiaSolaria1, FakeGladiaWs, list[httpx.Request]]]:
    """Run a fake Gladia WebSocket and point the adapter's init POST at it."""
    monkeypatch.setenv("GLADIA_API_KEY", "test-key")
    handler = FakeGladiaWs()
    captured: list[httpx.Request] = []

    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]

        def respond(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(201, json={"id": "session-1", "url": f"ws://127.0.0.1:{port}"})

        adapter = stt.create("gladia-solaria1")
        assert isinstance(adapter, gladia.GladiaSolaria1)
        adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(respond))

        yield adapter, handler, captured


class TestInitRequest:
    """The ``POST /v2/live`` request shape."""

    async def test_default_body_and_auth_header(
        self, gladia_ws: tuple[gladia.GladiaSolaria1, FakeGladiaWs, list[httpx.Request]]
    ) -> None:
        adapter, _handler, captured = gladia_ws
        clip = make_clip()

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert result.ok, result.error
        [request] = captured
        assert request.method == "POST"
        assert request.url.host == "api.gladia.io"
        assert request.url.path == "/v2/live"
        assert request.headers["x-gladia-key"] == "test-key"

        body = orjson.loads(request.content)
        assert body["encoding"] == "wav/pcm"
        assert body["bit_depth"] == 16
        assert body["sample_rate"] == clip.sample_rate
        assert body["channels"] == 1
        assert body["model"] == "solaria-1"
        assert body["language_config"] == {"languages": ["en"], "code_switching": False}
        assert body["endpointing"] == pytest.approx(0.05)
        assert body["maximum_duration_without_endpointing"] == pytest.approx(5.0)
        assert body["messages_config"] == {
            "receive_partial_transcripts": True,
            "receive_final_transcripts": True,
            "receive_speech_events": True,
        }

    async def test_empty_language_option_requests_auto_detect(
        self, gladia_ws: tuple[gladia.GladiaSolaria1, FakeGladiaWs, list[httpx.Request]]
    ) -> None:
        adapter, _handler, captured = gladia_ws
        adapter.options["language"] = ""

        await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

        body = orjson.loads(captured[0].content)
        assert body["language_config"]["languages"] == []

    async def test_region_becomes_a_query_param(
        self, gladia_ws: tuple[gladia.GladiaSolaria1, FakeGladiaWs, list[httpx.Request]]
    ) -> None:
        adapter, _handler, captured = gladia_ws
        adapter.options["region"] = "eu-west"

        await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

        assert captured[0].url.params["region"] == "eu-west"


class TestWebSocketProtocol:
    """Audio delivery, ``stop_recording`` signaling, and event capture."""

    async def test_audio_and_stop_recording_are_sent(
        self, gladia_ws: tuple[gladia.GladiaSolaria1, FakeGladiaWs, list[httpx.Request]]
    ) -> None:
        adapter, handler, _captured = gladia_ws
        clip = make_clip()

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert bytes(handler.received_audio) == clip.pcm
        assert handler.received_json == [{"type": "stop_recording"}]

    async def test_transcript_text_and_speech_end_eou_are_recorded(
        self, gladia_ws: tuple[gladia.GladiaSolaria1, FakeGladiaWs, list[httpx.Request]]
    ) -> None:
        adapter, _handler, _captured = gladia_ws

        result = await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

        assert result.text == "hello world"

        eous = [p for p in result.partials if p.kind == EventKind.EOU]
        assert len(eous) == 1, "speech_end is the only event that should carry EOU"
        assert eous[0].text == ""

        finals = [p for p in result.partials if p.is_final]
        assert len(finals) == 1
        assert finals[0].text == "hello world"
        assert finals[0].kind == EventKind.SEGMENT_FINAL, "is_final is a decode boundary, not the EOU decision"

    async def test_eou_source_and_endpoint_config_are_recorded(
        self, gladia_ws: tuple[gladia.GladiaSolaria1, FakeGladiaWs, list[httpx.Request]]
    ) -> None:
        adapter, _handler, _captured = gladia_ws
        adapter.options["endpointing"] = 0.2
        adapter.options["maximum_duration_without_endpointing"] = 10.0

        result = await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

        assert result.raw["eou_source"] == "speech_end"
        assert result.raw["endpoint_config"] == {
            "endpointing": 0.2,
            "maximum_duration_without_endpointing": 10.0,
        }

    async def test_disabling_speech_events_leaves_eou_source_unset(
        self, gladia_ws: tuple[gladia.GladiaSolaria1, FakeGladiaWs, list[httpx.Request]]
    ) -> None:
        adapter, _handler, captured = gladia_ws
        adapter.options["receive_speech_events"] = False

        result = await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)

        assert "eou_source" not in result.raw
        body = orjson.loads(captured[0].content)
        assert body["messages_config"]["receive_speech_events"] is False


class TestInitHttpError:
    """A rejected session-init request surfaces the vendor's error body."""

    async def test_http_error_is_raised_with_the_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GLADIA_API_KEY", "test-key")

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="invalid api key")

        adapter = stt.create("gladia-solaria1")
        adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(respond))

        with pytest.raises(ProviderHttpError, match="401"):
            await adapter.transcribe_stream(make_clip(), chunk_ms=20, realtime=False)


LIVE_FLAG = "AUDIO_HARNESS_TEST_GLADIA_LIVE"


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a couple of short live transcriptions (fractions of a cent total)",
)
class TestLiveSmoke:
    """A handful of short clips against the real vendor, en + ja.

    Minimal-API-testing policy: two short clips only, no bulk runs here.
    """

    async def test_stream_en(self) -> None:
        adapter = stt.create("gladia-solaria1")
        try:
            result = await adapter.transcribe_stream(make_clip(0.6), chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0

    async def test_stream_ja(self) -> None:
        clip = make_clip(0.6, language="ja-JP")
        adapter = stt.create("gladia-solaria1")
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0


class Solaria3Transport:
    """Mock Gladia's upload, init, and result-polling HTTP surfaces."""

    def __init__(
        self,
        *,
        terminal: dict[str, Any] | None = None,
        failure_path: str | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self.poll_count = 0
        self.terminal = terminal or {
            "status": "done",
            "result": {"transcription": {"full_transcript": "hello from solaria three"}},
        }
        self.failure_path = failure_path

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == self.failure_path:
            return httpx.Response(503, text="vendor temporarily unavailable")
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"audio_url": "https://files.gladia.io/audio.wav"})
        if request.url.path == "/v2/pre-recorded":
            return httpx.Response(
                201,
                json={
                    "id": "transcription-1",
                    "result_url": "https://api.gladia.io/v2/pre-recorded/transcription-1",
                },
            )
        if request.url.path == "/v2/pre-recorded/transcription-1":
            self.poll_count += 1
            if self.poll_count == 1:
                return httpx.Response(200, json={"status": "queued"})
            return httpx.Response(200, json=self.terminal)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")


def make_solaria3_adapter(
    monkeypatch: pytest.MonkeyPatch,
    transport: Solaria3Transport,
) -> gladia.GladiaSolaria3:
    """Build a Solaria-3 adapter with a zero-delay mocked polling loop."""
    monkeypatch.setenv("GLADIA_API_KEY", "test-key")
    monkeypatch.setattr(gladia, "POLL_INTERVAL_S", 0.0)
    adapter = gladia.GladiaSolaria3()
    adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return adapter


class TestSolaria3HappyPath:
    """Upload, init, polling, transcript extraction, and timing metadata."""

    async def test_upload_init_queued_then_done(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = Solaria3Transport()
        adapter = make_solaria3_adapter(monkeypatch, transport)
        try:
            result = await adapter.transcribe_batch(make_clip(language="en-US"))
        finally:
            await adapter.aclose()

        assert result.text == "hello from solaria three"
        assert result.error is None
        assert result.total_s > 0
        upload_s = result.raw["upload_s"]
        queue_poll_s = result.raw["queue_poll_s"]
        assert isinstance(upload_s, float)
        assert isinstance(queue_poll_s, float)
        assert upload_s >= 0
        assert queue_poll_s >= 0
        assert result.raw["transcription_id"] == "transcription-1"
        assert transport.poll_count == 2

        upload, created, first_poll, second_poll = transport.requests
        assert upload.method == "POST"
        assert upload.url == httpx.URL("https://api.gladia.io/v2/upload")
        assert upload.headers["x-gladia-key"] == "test-key"
        assert b'name="audio"' in upload.content
        assert b'filename="audio.wav"' in upload.content
        assert b"RIFF" in upload.content

        assert created.method == "POST"
        assert created.url == httpx.URL("https://api.gladia.io/v2/pre-recorded")
        assert created.headers["x-gladia-key"] == "test-key"
        body = orjson.loads(created.content)
        assert body["audio_url"] == "https://files.gladia.io/audio.wav"
        assert body["model"] == "solaria-3"
        assert body["language_config"] == {"languages": ["en"], "code_switching": False}

        assert first_poll.headers["x-gladia-key"] == "test-key"
        assert second_poll.headers["x-gladia-key"] == "test-key"


class TestSolaria3TerminalError:
    """A terminal vendor error is returned as a failed transcription result."""

    async def test_error_status_sets_result_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = Solaria3Transport(terminal={"status": "error", "error": "audio could not be decoded"})
        adapter = make_solaria3_adapter(monkeypatch, transport)
        try:
            result = await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()

        assert result.text == ""
        assert result.error == "audio could not be decoded"
        assert result.total_s > 0
        assert result.raw["response"] == {
            "status": "error",
            "error": "audio could not be decoded",
        }


class TestSolaria3HttpFailures:
    """Every failed HTTP phase uses the shared body-preserving exception."""

    @pytest.mark.parametrize(
        "failure_path",
        ["/v2/upload", "/v2/pre-recorded", "/v2/pre-recorded/transcription-1"],
    )
    async def test_http_error_includes_vendor_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure_path: str,
    ) -> None:
        transport = Solaria3Transport(failure_path=failure_path)
        adapter = make_solaria3_adapter(monkeypatch, transport)
        try:
            with pytest.raises(ProviderHttpError, match="vendor temporarily unavailable"):
                await adapter.transcribe_batch(make_clip())
        finally:
            await adapter.aclose()


class TestSolaria3LanguageGuard:
    """Unsupported languages fail before credentials or HTTP are touched."""

    async def test_unsupported_language_makes_no_http_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        transport = Solaria3Transport()
        adapter = make_solaria3_adapter(monkeypatch, transport)
        try:
            with pytest.raises(ValueError, match=r"pt-BR.*de, en, es, fr, it"):
                await adapter.transcribe_batch(make_clip(language="pt-BR"))
        finally:
            await adapter.aclose()

        assert transport.requests == []


@pytest.mark.skipif(
    not os.environ.get("GLADIA_API_KEY") or not os.environ.get(LIVE_FLAG),
    reason=f"live smoke needs GLADIA_API_KEY and {LIVE_FLAG}=1",
)
class TestSolaria3LiveSmoke:
    """One short real English clip against Solaria-3 batch."""

    async def test_batch_en(self) -> None:
        clip = make_live_clip()
        adapter = gladia.GladiaSolaria3()
        try:
            result = await adapter.transcribe_batch(clip)
        finally:
            await adapter.aclose()

        logging.getLogger(__name__).info(
            "Solaria-3 live result: %s",
            orjson.dumps({
                "transcript": result.text,
                "ttft_s": result.ttft_s,
                "finalize_s": result.finalize_s,
                "total_s": result.total_s,
                "upload_s": result.raw.get("upload_s"),
                "queue_poll_s": result.raw.get("queue_poll_s"),
            }).decode(),
        )
        assert result.error is None, result.error
        assert result.text
        assert result.total_s > 0
