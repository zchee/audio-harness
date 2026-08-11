"""Protocol tests for the Soniox tts-rt-v2 text-to-speech adapter."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable
import os
import re
from typing import Any, cast

import httpx
import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import soniox
from audio_harness.tts.base import ChunkTimeline, available, token_pieces
from audio_harness.types import TtsPrompt, TtsResult


PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")
JP_PROMPT = TtsPrompt(prompt_id="p-ja", text="こんにちは、すばらしい世界。", language="ja-JP")


def test_registration_and_capabilities() -> None:
    assert "soniox-tts-rt-v2" in available()
    assert soniox.SonioxTts.vendor == "soniox"
    assert soniox.SonioxTts.family == "soniox"
    assert soniox.SonioxTts.supports_batch is True
    assert soniox.SonioxTts.supports_stream is True
    assert soniox.SonioxTts.supports_input_streaming is True


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SONIOX_API_KEY", "test-key")


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


def _mocked_adapter(respond: Respond, options: dict[str, Any] | None = None) -> tuple[soniox.SonioxTts, _RecordingHttp]:
    adapter = soniox.SonioxTts(options)
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
        assert result.raw["model"] == "tts-rt-v2"
        assert result.ttfb_s is None

        [request] = recorder.requests
        assert request.method == "POST"
        assert str(request.url) == soniox.BATCH_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["Content-Type"] == "application/json"
        assert orjson.loads(request.content) == {
            "model": "tts-rt-v2",
            "language": "en",
            "voice": "Adrian",
            "audio_format": "pcm_s16le",
            "sample_rate": 24000,
            "text": PROMPT.text,
        }

    async def test_options_override_voice_and_sample_rate(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        adapter, recorder = _mocked_adapter(respond, {"voice": "Ava", "sample_rate": 16000})
        await adapter.synthesize(PROMPT)
        body = orjson.loads(recorder.requests[0].content)
        assert body["voice"] == "Ava"
        assert body["sample_rate"] == 16000

    async def test_unsupported_sample_rate_raises_before_request(self) -> None:
        adapter, recorder = _mocked_adapter(lambda request: httpx.Response(200), {"sample_rate": 22050})
        with pytest.raises(ValueError, match=r"expected one of \[8000, 16000, 24000, 44100, 48000\]"):
            await adapter.synthesize(PROMPT)
        assert recorder.requests == []

    async def test_http_error_preserves_vendor_body(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error_code": "unauthorized", "error_message": "invalid api key"},
            )

        adapter, _ = _mocked_adapter(respond)
        with pytest.raises(ProviderHttpError, match="invalid api key"):
            await adapter.synthesize(PROMPT)

    async def test_japanese_language(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        adapter, recorder = _mocked_adapter(respond)
        await adapter.synthesize(JP_PROMPT)
        assert orjson.loads(recorder.requests[0].content)["language"] == "ja"


class FakeSonioxTtsWs:
    """Assert client frames, then emit two PCM chunks and termination."""

    def __init__(
        self,
        expected_texts: list[str],
        *,
        language: str,
        voice: str = "Adrian",
        sample_rate: int = 24000,
    ) -> None:
        self.expected_texts = expected_texts
        self.language = language
        self.voice = voice
        self.sample_rate = sample_rate
        self.messages: list[dict[str, Any]] = []
        self.authorization = ""
        self.chunks = [b"\x00\x01" * 50, b"\x02\x03" * 50]

    async def __call__(self, socket: ServerConnection) -> None:
        assert socket.request is not None
        self.authorization = socket.request.headers.get("Authorization", "")

        config = orjson.loads(await socket.recv())
        self.messages.append(config)
        assert config == {
            "api_key": "test-key",
            "stream_id": "stream-1",
            "model": "tts-rt-v2",
            "language": self.language,
            "voice": self.voice,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
        }

        for expected_text in self.expected_texts:
            message = orjson.loads(await socket.recv())
            self.messages.append(message)
            assert message == {"text": expected_text, "text_end": False, "stream_id": "stream-1"}

        terminal = orjson.loads(await socket.recv())
        self.messages.append(terminal)
        assert terminal == {"text": "", "text_end": True, "stream_id": "stream-1"}

        for chunk in self.chunks:
            await socket.send(orjson.dumps({"audio": base64.b64encode(chunk).decode()}).decode())
        await socket.send(orjson.dumps({"terminated": True}).decode())


class FakeErrorWs:
    """Return an error frame once the client completes its text input."""

    async def __call__(self, socket: ServerConnection) -> None:
        async for frame in socket:
            if orjson.loads(frame).get("text_end") is True:
                await socket.send(
                    orjson.dumps({"error_code": "some_code", "error_message": "voice unavailable"}).decode()
                )
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
    prompt: TtsPrompt = PROMPT,
    options: dict[str, Any] | None = None,
) -> tuple[TtsResult, FakeSonioxTtsWs]:
    monkeypatch.setenv("SONIOX_API_KEY", "test-key")
    pieces = token_pieces(prompt.text) if incremental else [prompt.text]
    configured = options or {}
    handler = FakeSonioxTtsWs(
        pieces,
        language=prompt.language.split("-")[0],
        voice=str(configured.get("voice", "Adrian")),
        sample_rate=int(configured.get("sample_rate", 24000)),
    )
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(soniox, "WS_URL", f"ws://127.0.0.1:{port}/tts-websocket")
        adapter = soniox.SonioxTts(options)
        if incremental:
            result = await adapter.synthesize_incremental(prompt, token_rate=400.0)
        else:
            result = await adapter.synthesize_stream(prompt)
    return result, handler


class TestWebSocketProtocol:
    """Pin first-frame configuration, text frames and audio parsing."""

    async def test_whole_prompt_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, handler = await _run_ws(monkeypatch, incremental=False)

        assert handler.authorization == ""
        assert handler.messages == [
            {
                "api_key": "test-key",
                "stream_id": "stream-1",
                "model": "tts-rt-v2",
                "language": "en",
                "voice": "Adrian",
                "audio_format": "pcm_s16le",
                "sample_rate": 24000,
            },
            {"text": PROMPT.text, "text_end": False, "stream_id": "stream-1"},
            {"text": "", "text_end": True, "stream_id": "stream-1"},
        ]
        assert result.audio == b"".join(handler.chunks)
        assert result.encoding == "pcm_s16le"
        assert result.raw["model"] == "tts-rt-v2"
        assert result.raw["input_streaming"] is False
        assert len(result.chunk_t_s) == 2
        assert result.ttfb_s is not None

    async def test_incremental_text_pieces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result, handler = await _run_ws(monkeypatch, incremental=True)
        pieces = token_pieces(PROMPT.text)

        assert handler.messages == [
            {
                "api_key": "test-key",
                "stream_id": "stream-1",
                "model": "tts-rt-v2",
                "language": "en",
                "voice": "Adrian",
                "audio_format": "pcm_s16le",
                "sample_rate": 24000,
            },
            *({"text": piece, "text_end": False, "stream_id": "stream-1"} for piece in pieces),
            {"text": "", "text_end": True, "stream_id": "stream-1"},
        ]
        assert result.audio == b"".join(handler.chunks)
        assert result.raw["model"] == "tts-rt-v2"
        assert result.raw["input_streaming"] is True
        assert result.raw["text_pieces"] == len(pieces)

    async def test_error_frame_raises_full_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SONIOX_API_KEY", "test-key")
        async with serve(FakeErrorWs(), "127.0.0.1", 0) as running:
            port = running.sockets[0].getsockname()[1]
            monkeypatch.setattr(soniox, "WS_URL", f"ws://127.0.0.1:{port}/tts-websocket")
            adapter = soniox.SonioxTts()
            with pytest.raises(soniox.SonioxTtsProtocolError, match=re.escape("voice unavailable")):
                await adapter.synthesize_stream(PROMPT)

    async def test_japanese_language(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, handler = await _run_ws(monkeypatch, incremental=False, prompt=JP_PROMPT)
        assert handler.messages[0]["language"] == "ja"


class TestWebSocketTerminalState:
    """Protocol termination and feeder races are fail-closed."""

    async def test_eof_before_terminated_raises(self) -> None:
        socket = InProcessSocket([{"audio": base64.b64encode(b"\x00\x01").decode()}])
        with pytest.raises(soniox.SonioxTtsProtocolError, match=re.escape("ended before terminated")):
            await soniox.SonioxTts()._consume(cast(Any, socket), ChunkTimeline())

    async def test_protocol_error_is_not_masked_by_feeder_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SONIOX_API_KEY", "test-key")
        socket = InProcessSocket(
            [{"error_code": "protocol_error", "error_message": "primary protocol failure"}],
            send_error=RuntimeError("secondary feeder failure"),
        )
        monkeypatch.setattr(soniox, "connect", lambda *args, **kwargs: InProcessConnect(socket))

        with pytest.raises(soniox.SonioxTtsProtocolError, match="primary protocol failure"):
            await soniox.SonioxTts().synthesize_stream(PROMPT)

    async def test_feeder_failure_cancels_a_silent_consumer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SONIOX_API_KEY", "test-key")
        socket = InProcessSocket([], send_error=RuntimeError("send failed"), block_after_frames=True)
        monkeypatch.setattr(soniox, "connect", lambda *args, **kwargs: InProcessConnect(socket))

        with pytest.raises(RuntimeError, match="send failed"):
            await asyncio.wait_for(soniox.SonioxTts().synthesize_stream(PROMPT), timeout=0.1)


@pytest.mark.skipif(
    not os.environ.get("SONIOX_API_KEY") or not os.environ.get("AUDIO_HARNESS_LIVE"),
    reason="live smoke needs SONIOX_API_KEY and AUDIO_HARNESS_LIVE=1",
)
class TestLiveSmoke:
    """One short REST call and one short WebSocket call; total cost is negligible."""

    LIVE_PROMPT = TtsPrompt(prompt_id="soniox-live", text="Hello from Soniox.", language="en-US")

    async def test_batch_en(self) -> None:
        adapter = soniox.SonioxTts()
        try:
            result = await adapter.synthesize(self.LIVE_PROMPT)
        finally:
            await adapter.aclose()
        assert result.ok, result.error
        assert result.audio_s > 0

    async def test_stream_en(self) -> None:
        adapter = soniox.SonioxTts()
        try:
            result = await adapter.synthesize_stream(self.LIVE_PROMPT)
        finally:
            await adapter.aclose()
        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.ttfb_s is not None
