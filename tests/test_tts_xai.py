"""Protocol tests for the xAI Grok text-to-speech adapter."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
import os
import re
from typing import Any, cast
from urllib.parse import parse_qsl, urlsplit

import httpx
import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import xai
from audio_harness.tts.base import ChunkTimeline, available, token_pieces
from audio_harness.types import TtsPrompt, TtsResult


PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")


def test_registration_and_capabilities() -> None:
    assert "xai-grok-tts" in available()
    assert xai.XaiGrokTts.vendor == "xai"
    assert xai.XaiGrokTts.family == "xai"
    assert xai.XaiGrokTts.supports_batch is True
    assert xai.XaiGrokTts.supports_stream is True
    assert xai.XaiGrokTts.supports_input_streaming is True


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "test-key")


Respond = Callable[[httpx.Request], httpx.Response]


class _RecordingHttp:
    """Record requests made through an ``httpx.MockTransport`` client."""

    def __init__(self, respond: Respond) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handler))


def _mocked_adapter(respond: Respond, options: dict[str, Any] | None = None) -> tuple[xai.XaiGrokTts, _RecordingHttp]:
    adapter = xai.XaiGrokTts(options)
    recorder = _RecordingHttp(respond)
    adapter._http = recorder.client()
    return adapter, recorder


class TestBatchProtocol:
    """Pin the REST request and raw PCM response contract."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_request_shape_auth_and_raw_pcm(self) -> None:
        pcm = b"\x00\x01" * 100

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=pcm, headers={"Content-Type": "audio/pcm"})

        adapter, recorder = _mocked_adapter(respond)
        result = await adapter.synthesize(PROMPT)

        assert result.audio == pcm
        assert result.encoding == "pcm_s16le"
        assert result.sample_rate == 24000
        assert result.raw["model_unspecified"] is True
        assert result.ttfb_s is None

        [request] = recorder.requests
        assert request.method == "POST"
        assert str(request.url) == xai.BATCH_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        assert orjson.loads(request.content) == {
            "text": PROMPT.text,
            "voice_id": "eve",
            "language": "en",
            "output_format": {"codec": "pcm", "sample_rate": 24000},
        }

    async def test_options_override_voice_and_sample_rate(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        adapter, recorder = _mocked_adapter(respond, {"voice_id": "ara", "sample_rate": 16000})
        await adapter.synthesize(PROMPT)
        body = orjson.loads(recorder.requests[0].content)
        assert body["voice_id"] == "ara"
        assert body["output_format"]["sample_rate"] == 16000

    async def test_unsupported_sample_rate_raises(self) -> None:
        adapter, _ = _mocked_adapter(lambda request: httpx.Response(200), {"sample_rate": 12345})
        with pytest.raises(ValueError, match=r"expected one of \[8000, 16000, 22050, 24000, 44100, 48000\]"):
            await adapter.synthesize(PROMPT)

    async def test_http_error_preserves_vendor_body(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "invalid api key"})

        adapter, _ = _mocked_adapter(respond)
        with pytest.raises(ProviderHttpError, match="invalid api key"):
            await adapter.synthesize(PROMPT)


class FakeXaiTtsWs:
    """Capture client text frames, then emit two PCM deltas and completion."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.path = ""
        self.authorization = ""
        self.chunks = [b"\x00\x01" * 50, b"\x02\x03" * 50]

    async def __call__(self, socket: ServerConnection) -> None:
        assert socket.request is not None
        self.path = socket.request.path
        self.authorization = socket.request.headers.get("Authorization", "")
        async for frame in socket:
            message = orjson.loads(frame)
            self.messages.append(message)
            if message.get("type") == "text.done":
                break
        for chunk in self.chunks:
            await socket.send(orjson.dumps({"type": "audio.delta", "delta": base64.b64encode(chunk).decode()}).decode())
        await socket.send(orjson.dumps({"type": "audio.done"}).decode())


class FakeErrorWs:
    """Return an error frame once the client completes its text input."""

    async def __call__(self, socket: ServerConnection) -> None:
        async for frame in socket:
            if orjson.loads(frame).get("type") == "text.done":
                await socket.send(orjson.dumps({"type": "error", "error": {"message": "voice unavailable"}}).decode())
                return


class InProcessSocket:
    """Minimal socket double for terminal-state races without a TCP bind."""

    def __init__(
        self,
        frames: list[dict[str, Any]],
        *,
        send_error: Exception | None = None,
        block_after_frames: bool = False,
    ) -> None:
        self.frames = frames
        self.send_error = send_error
        self.block_after_frames = block_after_frames

    def __aiter__(self) -> Any:
        return self._frames()

    async def _frames(self) -> Any:
        # Give the feeder task one turn so a send failure can race the first
        # server frame exactly as it can on a closing real socket.
        await asyncio.sleep(0)
        for frame in self.frames:
            yield orjson.dumps(frame).decode()
        if self.block_after_frames:
            await asyncio.Event().wait()

    async def send(self, message: str) -> None:
        if self.send_error is not None:
            raise self.send_error


class InProcessConnect:
    """Async context manager returned by a monkeypatched ``connect`` call."""

    def __init__(self, socket: InProcessSocket) -> None:
        self.socket = socket

    async def __aenter__(self) -> InProcessSocket:
        return self.socket

    async def __aexit__(self, *exc: object) -> None:
        return None


async def _run_ws(
    monkeypatch: pytest.MonkeyPatch,
    *,
    incremental: bool,
    options: dict[str, Any] | None = None,
) -> tuple[TtsResult, FakeXaiTtsWs]:
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    handler = FakeXaiTtsWs()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(xai, "WS_URL", f"ws://127.0.0.1:{port}/v1/tts")
        adapter = xai.XaiGrokTts(options)
        if incremental:
            result = await adapter.synthesize_incremental(PROMPT, token_rate=400.0)
        else:
            result = await adapter.synthesize_stream(PROMPT)
    return result, handler


class TestWebSocketProtocol:
    """Pin query configuration, client frames and audio-event parsing."""

    async def test_whole_prompt_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, handler = await _run_ws(monkeypatch, incremental=False)

        params = dict(parse_qsl(urlsplit(handler.path).query))
        assert params == {"voice": "eve", "language": "en", "codec": "pcm", "sample_rate": "24000"}
        assert "voice_id" not in params
        assert handler.authorization == "Bearer test-key"
        assert handler.messages == [
            {"type": "text.delta", "delta": PROMPT.text},
            {"type": "text.done"},
        ]
        assert result.audio == b"".join(handler.chunks)
        assert result.encoding == "pcm_s16le"
        assert result.raw["input_streaming"] is False
        assert result.raw["model_unspecified"] is True
        assert len(result.chunk_t_s) == 2
        assert result.ttfb_s is not None

    async def test_incremental_text_pieces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, handler = await _run_ws(monkeypatch, incremental=True)
        pieces = token_pieces(PROMPT.text)

        assert handler.messages == [
            *({"type": "text.delta", "delta": piece} for piece in pieces),
            {"type": "text.done"},
        ]
        assert result.audio == b"".join(handler.chunks)
        assert result.raw["input_streaming"] is True
        assert result.raw["text_pieces"] == len(pieces)

    async def test_query_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await _run_ws(monkeypatch, incremental=False, options={"voice_id": "ara", "sample_rate": 48000})
        params = dict(parse_qsl(urlsplit(handler.path).query))
        assert params["voice"] == "ara"
        assert params["sample_rate"] == "48000"

    async def test_error_frame_raises_full_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        async with serve(FakeErrorWs(), "127.0.0.1", 0) as running:
            port = running.sockets[0].getsockname()[1]
            monkeypatch.setattr(xai, "WS_URL", f"ws://127.0.0.1:{port}/v1/tts")
            adapter = xai.XaiGrokTts()
            with pytest.raises(xai.XaiTtsProtocolError, match=re.escape("voice unavailable")):
                await adapter.synthesize_stream(PROMPT)


class TestWebSocketTerminalState:
    """Protocol termination and feeder races are fail-closed."""

    async def test_eof_before_audio_done_raises(self) -> None:
        socket = InProcessSocket([{"type": "audio.delta", "delta": base64.b64encode(b"\x00\x01").decode()}])
        with pytest.raises(xai.XaiTtsProtocolError, match=re.escape("ended before audio.done")):
            await xai.XaiGrokTts()._consume(cast(Any, socket), ChunkTimeline())

    async def test_protocol_error_is_not_masked_by_feeder_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        socket = InProcessSocket(
            [{"type": "error", "error": {"message": "primary protocol failure"}}],
            send_error=RuntimeError("secondary feeder failure"),
        )
        monkeypatch.setattr(xai, "connect", lambda *args, **kwargs: InProcessConnect(socket))

        with pytest.raises(xai.XaiTtsProtocolError, match="primary protocol failure"):
            await xai.XaiGrokTts().synthesize_stream(PROMPT)

    async def test_feeder_failure_cancels_a_silent_consumer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XAI_API_KEY", "test-key")
        socket = InProcessSocket([], send_error=RuntimeError("send failed"), block_after_frames=True)
        monkeypatch.setattr(xai, "connect", lambda *args, **kwargs: InProcessConnect(socket))

        with pytest.raises(RuntimeError, match="send failed"):
            await asyncio.wait_for(xai.XaiGrokTts().synthesize_stream(PROMPT), timeout=0.1)


@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or not os.environ.get("AUDIO_HARNESS_LIVE"),
    reason="live smoke needs XAI_API_KEY and AUDIO_HARNESS_LIVE=1",
)
class TestLiveSmoke:
    """One short REST call and one short WebSocket call; total cost is negligible."""

    LIVE_PROMPT = TtsPrompt(prompt_id="xai-live", text="Hello from Grok.", language="en-US")

    async def test_batch_en(self) -> None:
        adapter = xai.XaiGrokTts()
        try:
            result = await adapter.synthesize(self.LIVE_PROMPT)
        finally:
            await adapter.aclose()
        assert result.ok, result.error
        assert result.audio_s > 0

    async def test_stream_en(self) -> None:
        adapter = xai.XaiGrokTts()
        try:
            result = await adapter.synthesize_stream(self.LIVE_PROMPT)
        finally:
            await adapter.aclose()
        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.ttfb_s is not None
