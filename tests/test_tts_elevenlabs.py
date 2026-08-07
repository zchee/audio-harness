"""Tests for the ElevenLabs TTS adapters' wire protocols.

Three transports are exercised: the plain and ``/stream`` HTTP endpoints
(mocked with ``httpx.MockTransport`` so the request shape and response
parsing are pinned without a network call) and the ``stream-input`` WebSocket
that backs :meth:`synthesize_incremental` (a real local server, matching the
Cartesia protocol-test pattern). A cross-family round-trip config is also
pinned: ElevenLabs and Deepgram share no lineage, so a judge set of
``{deepgram-nova3, whisper-local}`` must cover this candidate.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Callable
import os

import httpx
import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import tts
from audio_harness.cli import validate_roundtrip_judges
from audio_harness.config import BenchmarkConfig
from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import elevenlabs
from audio_harness.tts.base import token_pieces
from audio_harness.types import TtsPrompt, TtsResult


PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials for the mocked protocol tests.

    Not autouse: :class:`TestLiveSmoke` needs the *real* environment
    variables, and this fixture would otherwise clobber them.
    """
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "test-voice")


class _Chunks(httpx.AsyncByteStream):
    """An async byte stream that yields fixed chunks, for streamed responses."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


Respond = Callable[[httpx.Request], httpx.Response]


class _RecordingHttp:
    """Records every request made through a mocked adapter HTTP client."""

    def __init__(self, respond: Respond) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self._handler))


def _mocked_adapter(respond: Respond) -> tuple[elevenlabs.ElevenLabsFlash25, _RecordingHttp]:
    adapter = tts.create("elevenlabs-flash25")
    assert isinstance(adapter, elevenlabs.ElevenLabsFlash25)
    recorder = _RecordingHttp(respond)
    adapter._http = recorder.client()
    return adapter, recorder


def _mocked_v3_adapter(respond: Respond) -> tuple[elevenlabs.ElevenLabsV3, _RecordingHttp]:
    adapter = tts.create("elevenlabs-v3")
    assert isinstance(adapter, elevenlabs.ElevenLabsV3)
    recorder = _RecordingHttp(respond)
    adapter._http = recorder.client()
    return adapter, recorder


class TestBatchProtocol:
    """The plain endpoint: URL, headers, body and response parsing."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_request_shape_and_response_parsing(self) -> None:
        pcm = b"\x00\x01" * 100

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=pcm)

        adapter, recorder = _mocked_adapter(respond)

        result = await adapter.synthesize(PROMPT)

        assert result.ok, result.error
        assert result.audio == pcm
        assert result.ttfb_s is None
        assert result.total_s > 0

        [request] = recorder.requests
        assert request.method == "POST"
        assert request.url.path == "/v1/text-to-speech/test-voice"
        assert request.url.params["output_format"] == "pcm_24000"
        assert request.headers["xi-api-key"] == "test-key"
        assert orjson.loads(request.content) == {
            "text": PROMPT.text,
            "model_id": "eleven_flash_v2_5",
            "language_code": "en",
        }

    async def test_http_error_is_raised_with_the_body(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="invalid api key")

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="401"):
            await adapter.synthesize(PROMPT)

    async def test_model_id_option_overrides_the_default(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        adapter, recorder = _mocked_adapter(respond)
        adapter.options["model_id"] = "eleven_flash_v2"

        await adapter.synthesize(PROMPT)

        assert orjson.loads(recorder.requests[0].content)["model_id"] == "eleven_flash_v2"


class TestStreamProtocol:
    """The ``/stream`` endpoint: chunked delivery is stamped and reassembled."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_chunks_are_stamped_and_assembled(self) -> None:
        chunks = [b"\x00\x01" * 50, b"\x02\x03" * 50]

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_Chunks(chunks))

        adapter, recorder = _mocked_adapter(respond)

        result = await adapter.synthesize_stream(PROMPT)

        assert result.ok, result.error
        assert result.audio == b"".join(chunks)
        assert result.ttfb_s is not None
        assert len(result.chunk_t_s) == 2

        [request] = recorder.requests
        assert request.url.path == "/v1/text-to-speech/test-voice/stream"
        assert request.url.params["output_format"] == "pcm_24000"

    async def test_sample_rate_option_selects_the_output_format(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_Chunks([b"\x00\x00"]))

        adapter = tts.create("elevenlabs-flash25", {"sample_rate": 16000})
        assert isinstance(adapter, elevenlabs.ElevenLabsFlash25)
        recorder = _RecordingHttp(respond)
        adapter._http = recorder.client()

        await adapter.synthesize_stream(PROMPT)

        assert recorder.requests[0].url.params["output_format"] == "pcm_16000"

    async def test_http_error_is_raised_with_the_body(self) -> None:
        """A real error must surface as ``ProviderHttpError``, not ``ResponseNotRead``.

        The response is unread when ``raise_for_status`` first sees it (this
        is what ``http.stream()`` returns before any chunk is consumed), so
        the error body must be constructed via ``stream=`` — an
        ``httpx.Response(content=...)`` shortcut is eagerly buffered at
        construction and would never reproduce the bug.
        """

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, stream=_Chunks([b'{"detail": "invalid voice id"}']))

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="invalid voice id"):
            await adapter.synthesize_stream(PROMPT)


class TestOutputFormatValidation:
    def test_unsupported_sample_rate_raises(self) -> None:
        adapter = tts.create("elevenlabs-flash25", {"sample_rate": 12345})
        assert isinstance(adapter, elevenlabs.ElevenLabsFlash25)

        with pytest.raises(ValueError, match="unsupported sample rate"):
            adapter._output_format()  # exercising the validation directly


class TestV3Protocol:
    """V3 reuses HTTP transports but never exposes stream-input."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_batch_and_stream_use_eleven_v3(self) -> None:
        pcm = b"\x00\x01" * 10

        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/stream"):
                return httpx.Response(200, stream=_Chunks([pcm]))
            return httpx.Response(200, content=pcm)

        adapter, recorder = _mocked_v3_adapter(respond)

        batch = await adapter.synthesize(PROMPT)
        stream = await adapter.synthesize_stream(PROMPT)

        assert batch.audio == pcm
        assert stream.audio == pcm
        assert [orjson.loads(request.content)["model_id"] for request in recorder.requests] == [
            "eleven_v3",
            "eleven_v3",
        ]
        assert [request.url.path for request in recorder.requests] == [
            "/v1/text-to-speech/test-voice",
            "/v1/text-to-speech/test-voice/stream",
        ]

    async def test_input_streaming_is_unsupported_by_the_base_contract(self) -> None:
        adapter, _ = _mocked_v3_adapter(lambda request: httpx.Response(200, content=b""))

        assert adapter.supports_input_streaming is False
        assert "synthesize_incremental" not in elevenlabs.ElevenLabsV3.__dict__
        with pytest.raises(NotImplementedError, match="does not accept streamed input text"):
            await adapter.synthesize_incremental(PROMPT, token_rate=40.0)


class FakeElevenLabsServer:
    """Speaks the ``stream-input`` protocol shape the adapter expects."""

    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.pcm = b"\x00\x01" * 200

    async def __call__(self, socket: ServerConnection) -> None:
        async for frame in socket:
            message = orjson.loads(frame)
            self.messages.append(message)
            if message.get("text") == "":
                half = len(self.pcm) // 2 // 2 * 2
                for piece in (self.pcm[:half], self.pcm[half:]):
                    await socket.send(orjson.dumps({"audio": base64.b64encode(piece).decode()}).decode())
                await socket.send(orjson.dumps({"isFinal": True}).decode())
                return


class FakeErrorServer:
    """Ends the generation with an error frame instead of audio."""

    async def __call__(self, socket: ServerConnection) -> None:
        async for frame in socket:
            message = orjson.loads(frame)
            if message.get("text") == "":
                await socket.send(orjson.dumps({"error": "synthesis failed"}).decode())
                return


def _ws_url_for(port: int) -> str:
    return f"ws://127.0.0.1:{port}/v1/text-to-speech/{{voice_id}}/stream-input"


@pytest.fixture
async def elevenlabs_ws(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeElevenLabsServer]:
    """Run a fake stream-input endpoint and point the adapter at it."""
    handler = FakeElevenLabsServer()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(elevenlabs, "WS_URL", _ws_url_for(port))
        yield handler


@pytest.fixture
async def elevenlabs_error_ws(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeErrorServer]:
    handler = FakeErrorServer()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(elevenlabs, "WS_URL", _ws_url_for(port))
        yield handler


class TestIncrementalProtocol:
    """The adapter's wire behavior against a real local WebSocket."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_feeds_pieces_then_closes_the_generation(self, elevenlabs_ws: FakeElevenLabsServer) -> None:
        adapter = tts.create("elevenlabs-flash25")

        result = await adapter.synthesize_incremental(PROMPT, token_rate=400.0)

        assert result.ok, result.error
        assert result.raw["input_streaming"] is True
        assert result.raw["text_pieces"] == len(token_pieces(PROMPT.text))

        messages = elevenlabs_ws.messages
        pieces = token_pieces(PROMPT.text)
        assert messages[0] == {"text": " "}, "the session opens with the documented single-space handshake"
        assert [m["text"] for m in messages[1:-1]] == pieces
        assert messages[-1] == {"text": ""}, "the generation closes with an empty-text message"
        assert "".join(str(m["text"]) for m in messages[1:-1]) == PROMPT.text
        assert result.audio_s > 0
        assert len(result.chunk_t_s) == 2

    async def test_server_error_is_recorded(self, elevenlabs_error_ws: FakeErrorServer) -> None:
        adapter = tts.create("elevenlabs-flash25")

        result = await adapter.synthesize_incremental(PROMPT, token_rate=400.0)

        assert result.error == "synthesis failed"


class TestCrossFamilyValidation:
    """The candidate must stay ranked once judged by a cross-family set.

    ElevenLabs shares no lineage with Deepgram or the OpenAI-lineage local
    Whisper judge, so both are valid cross-family judges for this candidate.
    """

    def test_family_is_the_vendor(self) -> None:
        assert tts.family_of("elevenlabs-flash25") == "elevenlabs"

    def test_deepgram_and_whisper_local_judges_cover_the_candidate(self) -> None:
        config = BenchmarkConfig.from_dict({
            "tts": ["elevenlabs-flash25"],
            "roundtrip_stt": [{"name": "deepgram-nova3"}, "whisper-local"],
        })

        validate_roundtrip_judges(config)


LIVE_FLAG = "AUDIO_HARNESS_TEST_ELEVENLABS_TTS_LIVE"


def _needs(*names: str) -> pytest.MarkDecorator:
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(names)} not set")


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a few short live syntheses (fractions of a cent each)",
)
class TestLiveSmoke:
    """A handful of short prompts against the real vendor, en + ja.

    Mocks confirm the wire protocol each mode claims to speak; this confirms
    the real endpoint honors it end to end. One call per transport, per the
    project's minimal-live-API policy.
    """

    BATCH_EN = TtsPrompt(
        prompt_id="smoke-batch-en", text="The quick brown fox jumps over the lazy dog.", language="en-US"
    )
    STREAM_JA = TtsPrompt(
        prompt_id="smoke-stream-ja", text="こんにちは、本日はよろしくお願いいたします。", language="ja-JP"
    )
    INCREMENTAL_EN = TtsPrompt(
        prompt_id="smoke-incremental-en", text="Please confirm your appointment for three o'clock.", language="en-US"
    )

    @_needs("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")
    async def test_batch_en(self) -> None:
        adapter = tts.create("elevenlabs-flash25")
        try:
            result = await adapter.synthesize(self.BATCH_EN)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0

    @_needs("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")
    async def test_stream_ja(self) -> None:
        adapter = tts.create("elevenlabs-flash25")
        try:
            result: TtsResult = await adapter.synthesize_stream(self.STREAM_JA)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.chunk_t_s
        assert result.ttfb_s is not None

    @_needs("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")
    async def test_incremental_en(self) -> None:
        adapter = tts.create("elevenlabs-flash25")
        try:
            result = await adapter.synthesize_incremental(self.INCREMENTAL_EN, token_rate=40.0)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.raw["input_streaming"] is True
        assert result.audio_s > 0

    @_needs("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")
    async def test_v3_batch_en(self) -> None:
        adapter = tts.create("elevenlabs-v3")
        try:
            result = await adapter.synthesize(
                TtsPrompt(prompt_id="smoke-v3-en", text="A quiet evening begins.", language="en-US")
            )
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0

    @_needs("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID")
    async def test_v3_stream_ja(self) -> None:
        adapter = tts.create("elevenlabs-v3")
        try:
            result = await adapter.synthesize_stream(
                TtsPrompt(prompt_id="smoke-v3-ja", text="静かな夜が始まります。", language="ja-JP")
            )
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.ttfa_s is not None
