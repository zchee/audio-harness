"""Protocol tests for the Gladia Solaria realtime speech-to-text adapter.

Two transports are pinned: the ``POST /v2/live`` session-init request (mocked
via ``httpx.MockTransport``, matching the ElevenLabs protocol-test pattern)
and the WebSocket it hands back a one-time URL for (a real local server,
matching the fake-server pattern in ``test_stream_driver.py``), since neither
shape is observable by driving the generic streaming driver alone.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import os
from typing import Any

import httpx
import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import stt
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
