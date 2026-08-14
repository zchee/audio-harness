"""Tests for the Mistral Voxtral realtime STT adapter's wire protocol.

The fake server speaks the shape verified against Mistral's own open-source
``client-python`` SDK (``src/mistralai/extra/realtime``): a ``session.created``
handshake, incremental ``transcription.text.delta`` fragments that must be
concatenated, a ``transcription.segment`` finalizing one span, and a single
terminal ``transcription.done`` carrying the authoritative transcript. There
is no end-of-turn/VAD event in the protocol, so the descriptive-only
endpointing claim is asserted directly (no ``eou_source`` in ``raw``).
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
import os
from urllib.parse import parse_qsl, urlsplit

import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import stt
from audio_harness.stt import voxtral
from audio_harness.stt.ws import StreamProtocolError
from audio_harness.types import AudioClip


def make_clip(seconds: float = 0.3, rate: int = 16000) -> AudioClip:
    """Build a silent clip of a known duration."""
    return AudioClip(
        clip_id="c1",
        pcm=b"\x00\x00" * int(rate * seconds),
        sample_rate=rate,
        duration_s=seconds,
        reference="hello wonderful world",
        language="en-US",
        source_path="<memory>",
    )


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials for the mocked protocol tests.

    Not autouse: :class:`TestLiveSmoke` needs the *real* environment
    variable, and this fixture would otherwise clobber it.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


class FakeMistralServer:
    """Speaks the Voxtral realtime protocol shape the adapter expects."""

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.received_audio = bytearray()
        self.request_path: str | None = None
        self.auth_header: str | None = None
        self.text_deltas = ["hello", " wonderful", " world"]
        self.segment_text = "hello wonderful world"
        self.done_text = "hello wonderful world"
        self.language: str | None = "en"
        self.fail_with: str | None = None

    async def __call__(self, socket: ServerConnection) -> None:
        await socket.send(
            orjson.dumps({
                "type": "session.created",
                "session": {
                    "request_id": "req-1",
                    "model": "voxtral-mini-transcribe-realtime-2602",
                    "audio_format": {"encoding": "pcm_mulaw", "sample_rate": 8000},
                },
            }).decode()
        )

        async for frame in socket:
            message = orjson.loads(frame)
            self.messages.append(message)
            kind = message.get("type")
            if kind == "input_audio.append":
                self.received_audio.extend(base64.b64decode(message["audio"]))
            elif kind == "input_audio.end":
                break

        self.request_path = socket.request.path if socket.request is not None else None
        self.auth_header = socket.request.headers.get("Authorization") if socket.request is not None else None

        if self.fail_with is not None:
            await socket.send(
                orjson.dumps({"type": "error", "error": {"message": self.fail_with, "code": 42}}).decode()
            )
            return

        if self.language is not None:
            await socket.send(
                orjson.dumps({"type": "transcription.language", "audio_language": self.language}).decode()
            )
        for text in self.text_deltas:
            await socket.send(orjson.dumps({"type": "transcription.text.delta", "text": text}).decode())
        await socket.send(
            orjson.dumps({
                "type": "transcription.segment",
                "text": self.segment_text,
                "start": 0.0,
                "end": 0.3,
            }).decode()
        )
        await socket.send(
            orjson.dumps({
                "type": "transcription.done",
                "model": "voxtral-mini-transcribe-realtime-2602",
                "text": self.done_text,
                "usage": {"prompt_audio_seconds": 1},
                "language": self.language,
            }).decode()
        )


@pytest.fixture
async def mistral_ws(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeMistralServer]:
    """Run a fake Voxtral realtime endpoint and point the adapter at it."""
    handler = FakeMistralServer()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(voxtral, "STREAM_URL", f"ws://127.0.0.1:{port}/v1/audio/transcriptions/realtime")
        yield handler


class TestRealtimeProtocol:
    """The adapter's wire behavior against a real local WebSocket."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_session_update_precedes_audio_and_pins_the_format(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.2)
        adapter = stt.create("mistral-voxtral-realtime")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert result.error is None, result.error
        first = mistral_ws.messages[0]
        assert first == {
            "type": "session.update",
            "session": {"audio_format": {"encoding": "pcm_s16le", "sample_rate": clip.sample_rate}},
        }, "audio format must be configured before any audio is sent"

    async def test_all_audio_arrives_intact(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.2)
        adapter = stt.create("mistral-voxtral-realtime")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert bytes(mistral_ws.received_audio) == clip.pcm

    async def test_end_of_input_flushes_then_ends(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("mistral-voxtral-realtime")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        trailer = [m["type"] for m in mistral_ws.messages if m.get("type") in {"input_audio.flush", "input_audio.end"}]
        assert trailer == ["input_audio.flush", "input_audio.end"]

    async def test_default_model_is_the_dated_release(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("mistral-voxtral-realtime")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert mistral_ws.request_path is not None
        query = dict(parse_qsl(urlsplit(mistral_ws.request_path).query))
        assert query["model"] == "voxtral-mini-transcribe-realtime-2602"

    async def test_model_option_overrides_the_default(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("mistral-voxtral-realtime", {"model": "voxtral-mini-transcribe-realtime-latest"})

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        query = dict(parse_qsl(urlsplit(mistral_ws.request_path or "").query))
        assert query["model"] == "voxtral-mini-transcribe-realtime-latest"

    async def test_bearer_auth_header_is_sent(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("mistral-voxtral-realtime")

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert mistral_ws.auth_header == "Bearer test-key"

    async def test_target_streaming_delay_ms_rides_the_session_update(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("mistral-voxtral-realtime", {"target_streaming_delay_ms": 500})

        await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        session = mistral_ws.messages[0]["session"]
        assert isinstance(session, dict)
        assert session["target_streaming_delay_ms"] == 500

    async def test_deltas_are_concatenated_into_the_growing_hypothesis(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.2)
        adapter = stt.create("mistral-voxtral-realtime")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        interim_texts = [p.text for p in result.partials if not p.is_final]
        assert interim_texts == ["hello", "hello wonderful", "hello wonderful world"], (
            "text.delta fragments are appended, not restated, so each interim "
            "hypothesis must accumulate the prior fragments"
        )

    async def test_done_event_is_authoritative_and_terminates_the_session(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.2)
        adapter = stt.create("mistral-voxtral-realtime")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert result.text == "hello wonderful world"
        response = result.raw["response"]
        assert isinstance(response, dict)
        assert response["type"] == "transcription.done"
        assert result.ttft_s is not None
        assert result.finalize_s is not None

    async def test_language_event_is_captured(self, mistral_ws: FakeMistralServer) -> None:
        clip = make_clip(0.1)
        adapter = stt.create("mistral-voxtral-realtime")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert result.raw["detected_language"] == "en"

    async def test_no_eou_signal_exists_on_the_wire(self, mistral_ws: FakeMistralServer) -> None:
        """Voxtral realtime has no end-of-turn/VAD event, so this lane must
        never claim one: the endpointing bench only ranks lanes that set
        ``eou_source``.
        """
        clip = make_clip(0.1)
        adapter = stt.create("mistral-voxtral-realtime")

        result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)

        assert "eou_source" not in result.raw
        assert all(p.kind != "eou" for p in result.partials)

    async def test_vendor_error_propagates(self, mistral_ws: FakeMistralServer) -> None:
        mistral_ws.fail_with = "invalid audio format"
        clip = make_clip(0.1)
        adapter = stt.create("mistral-voxtral-realtime")

        with pytest.raises(StreamProtocolError, match="invalid audio format"):
            await adapter.transcribe_stream(clip, chunk_ms=20, realtime=False)


LIVE_FLAG = "AUDIO_HARNESS_TEST_MISTRAL_LIVE"


def _needs(*names: str) -> pytest.MarkDecorator:
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(names)} not set")


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a couple of short live transcriptions (fractions of a cent total)",
)
class TestLiveSmoke:
    """A handful of short clips against the real vendor, en + ja.

    Minimal-API-testing policy: two short clips only, no bulk runs here.
    """

    @_needs("MISTRAL_API_KEY")
    async def test_stream_en(self) -> None:
        clip = make_clip(0.6)
        adapter = stt.create("mistral-voxtral-realtime")
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0

    @_needs("MISTRAL_API_KEY")
    async def test_stream_ja(self) -> None:
        clip = AudioClip(
            clip_id="live-ja",
            pcm=b"\x00\x00" * int(16000 * 0.6),
            sample_rate=16000,
            duration_s=0.6,
            reference=None,
            language="ja-JP",
            source_path="<memory>",
        )
        adapter = stt.create("mistral-voxtral-realtime")
        try:
            result = await adapter.transcribe_stream(clip, chunk_ms=20, realtime=True)
        finally:
            await adapter.aclose()

        assert result.error is None, result.error
        assert result.total_s > 0


class TestBatchLanes:
    """Protocol tests for the direct La Plateforme batch lanes."""

    @pytest.fixture(autouse=True)
    def _credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

    async def test_mini_uses_the_transcription_endpoint(self) -> None:
        import httpx

        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"model": "voxtral-mini-latest", "text": "hello world", "segments": []})

        adapter = stt.create("mistral-voxtral-mini")
        adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(respond))

        result = await adapter.transcribe_batch(make_clip())

        assert result.ok, result.error
        assert result.text == "hello world"
        assert result.raw["model"] == "voxtral-mini-latest"
        assert "segments" not in result.raw["response"]
        [request] = requests
        assert str(request.url) == voxtral.TRANSCRIPTIONS_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        assert b"voxtral-mini-2507" in request.content

    async def test_small_uses_prompted_chat_transcription(self) -> None:
        import httpx

        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200,
                json={
                    "model": "voxtral-small-2507",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello world"}}],
                    "usage": {"prompt_tokens": 17},
                },
            )

        adapter = stt.create("mistral-voxtral-small")
        adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(respond))

        result = await adapter.transcribe_batch(make_clip())

        assert result.ok, result.error
        assert result.text == "hello world"
        assert result.raw["prompted_transcription"] is True
        assert result.raw["model"] == "voxtral-small-2507"
        [request] = requests
        assert str(request.url) == voxtral.CHAT_URL
        body = orjson.loads(request.content)
        assert body["model"] == "voxtral-small-2507"
        assert body["temperature"] == 0
        content = body["messages"][0]["content"]
        assert content[0]["type"] == "input_audio"
        assert base64.b64decode(content[0]["input_audio"])[:4] == b"RIFF"
        assert content[1] == {"type": "text", "text": voxtral.TRANSCRIBE_PROMPT}

    def test_registry_metadata(self) -> None:
        tests = {
            "success: mini is batch-only direct mistral": "mistral-voxtral-mini",
            "success: small is batch-only direct mistral": "mistral-voxtral-small",
        }
        for name, key in tests.items():
            adapter = stt.create(key)
            assert adapter.vendor == "mistral", name
            assert adapter.supports_batch is True, name
            assert adapter.supports_stream is False, name
