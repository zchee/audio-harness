"""Protocol and live-smoke tests for the MiniMax Speech-2.8 Turbo adapter.

The request shape (``GroupId`` query parameter, Bearer header, hex-encoded
audio, ``base_resp`` as the true status channel) and the SSE streaming frame
(``data:`` lines, skipping any aggregated-summary chunk that still carries
``extra_info``) are pinned against ``httpx.MockTransport`` rather than a real
server, since MiniMax's streaming transport is plain HTTP, not a WebSocket.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
import os

import httpx
import orjson
import pytest

from audio_harness import tts
from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import minimax
from audio_harness.types import TtsPrompt


PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials for the mocked protocol tests.

    Not autouse: :class:`TestLiveSmoke` needs the *real* environment
    variables, and this fixture would otherwise clobber them.
    """
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_GROUP_ID", "test-group")


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


def _mocked_adapter(respond: Respond) -> tuple[minimax.MiniMaxSpeech28Turbo, _RecordingHttp]:
    adapter = tts.create("minimax-speech28turbo")
    assert isinstance(adapter, minimax.MiniMaxSpeech28Turbo)
    recorder = _RecordingHttp(respond)
    adapter._http = recorder.client()
    return adapter, recorder


def _sse_chunk(*, audio: bytes, status: int, extra_info: dict[str, object] | None = None) -> bytes:
    """Build one ``data:`` SSE frame carrying a hex-encoded audio fragment."""
    payload: dict[str, object] = {
        "data": {"audio": audio.hex(), "status": status},
        "trace_id": "trace-1",
        "base_resp": {"status_code": 0, "status_msg": ""},
    }
    if extra_info is not None:
        payload["extra_info"] = extra_info
    return b"data: " + orjson.dumps(payload) + b"\n\n"


class TestBatchProtocol:
    """The ``t2a_v2`` endpoint with ``stream: false``: request and response shape."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_request_shape_and_response_parsing(self) -> None:
        pcm = b"\x00\x01" * 100

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": {"audio": pcm.hex(), "status": 2},
                    "extra_info": {"audio_length": 100, "audio_sample_rate": 24000},
                    "trace_id": "trace-1",
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                },
            )

        adapter, recorder = _mocked_adapter(respond)

        result = await adapter.synthesize(PROMPT)

        assert result.ok, result.error
        assert result.audio == pcm
        assert result.ttfb_s is None
        assert result.total_s > 0

        [request] = recorder.requests
        assert request.method == "POST"
        assert request.url.path == "/v1/t2a_v2"
        assert request.url.params["GroupId"] == "test-group"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = orjson.loads(request.content)
        assert body["model"] == "speech-2.8-turbo"
        assert body["text"] == PROMPT.text
        assert body["stream"] is False
        assert body["voice_setting"] == {"voice_id": minimax.DEFAULT_VOICE_ID}
        assert body["audio_setting"] == {"format": "pcm", "sample_rate": 24000, "channel": 1}
        assert body["language_boost"] == "English"
        assert "stream_options" not in body

    async def test_logical_error_raises_despite_http_200(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": {"audio": "", "status": 0},
                    "base_resp": {"status_code": 1004, "status_msg": "invalid params"},
                },
            )

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="1004"):
            await adapter.synthesize(PROMPT)

    async def test_http_error_is_raised_with_the_body(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="invalid api key")

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="401"):
            await adapter.synthesize(PROMPT)

    async def test_model_id_option_overrides_the_default(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": {"audio": ""}, "base_resp": {"status_code": 0, "status_msg": ""}},
            )

        adapter, recorder = _mocked_adapter(respond)
        adapter.options["model_id"] = "speech-2.8-hd"

        await adapter.synthesize(PROMPT)

        assert orjson.loads(recorder.requests[0].content)["model"] == "speech-2.8-hd"

    async def test_voice_id_option_overrides_the_default(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": {"audio": ""}, "base_resp": {"status_code": 0, "status_msg": ""}},
            )

        adapter, recorder = _mocked_adapter(respond)
        adapter.options["voice_id"] = "Japanese_CalmLady"

        await adapter.synthesize(PROMPT)

        assert orjson.loads(recorder.requests[0].content)["voice_setting"]["voice_id"] == "Japanese_CalmLady"

    @pytest.mark.parametrize(
        ("language", "expected"),
        [
            ("ja-JP", "Japanese"),
            ("hu-HU", "Hungarian"),
            ("vi-VN", "Vietnamese"),
            ("pl-PL", "Polish"),
            ("tr-TR", "Turkish"),
            ("xx-XX", "auto"),
        ],
    )
    async def test_language_boost_mapping(self, language: str, expected: str) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"data": {"audio": ""}, "base_resp": {"status_code": 0, "status_msg": ""}},
            )

        adapter, recorder = _mocked_adapter(respond)
        prompt = TtsPrompt(prompt_id="p2", text="hi", language=language)

        await adapter.synthesize(prompt)

        assert orjson.loads(recorder.requests[0].content)["language_boost"] == expected


class TestStreamProtocol:
    """The same endpoint with ``stream: true``: SSE framing and chunk timing."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_chunks_are_stamped_and_assembled(self) -> None:
        chunks = [b"\x00\x01" * 50, b"\x02\x03" * 50]
        body = b"".join(_sse_chunk(audio=c, status=1) for c in chunks)

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_Chunks([body]))

        adapter, recorder = _mocked_adapter(respond)

        result = await adapter.synthesize_stream(PROMPT)

        assert result.ok, result.error
        assert result.audio == b"".join(chunks)
        assert result.ttfb_s is not None
        assert len(result.chunk_t_s) == 2

        [request] = recorder.requests
        assert request.url.path == "/v1/t2a_v2"
        assert request.url.params["GroupId"] == "test-group"
        req_body = orjson.loads(request.content)
        assert req_body["stream"] is True
        assert req_body["stream_options"] == {"exclude_aggregated_audio": True}

    async def test_skips_aggregated_summary_chunk(self) -> None:
        """A final chunk still carrying ``extra_info`` must not duplicate audio."""
        real_chunks = [b"\x00\x01" * 50, b"\x02\x03" * 50]
        body = b"".join(_sse_chunk(audio=c, status=1) for c in real_chunks)
        body += _sse_chunk(
            audio=b"".join(real_chunks), status=2, extra_info={"audio_length": 100, "audio_sample_rate": 24000}
        )

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_Chunks([body]))

        adapter, _ = _mocked_adapter(respond)

        result = await adapter.synthesize_stream(PROMPT)

        assert result.ok, result.error
        assert result.audio == b"".join(real_chunks)
        assert len(result.chunk_t_s) == 2

    async def test_chunks_split_across_transport_boundaries(self) -> None:
        """SSE lines must reassemble correctly when split mid-frame on the wire."""
        chunks = [b"\x00\x01" * 50, b"\x02\x03" * 50]
        body = b"".join(_sse_chunk(audio=c, status=1) for c in chunks)
        midpoint = len(body) // 2

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_Chunks([body[:midpoint], body[midpoint:]]))

        adapter, _ = _mocked_adapter(respond)

        result = await adapter.synthesize_stream(PROMPT)

        assert result.ok, result.error
        assert result.audio == b"".join(chunks)

    async def test_logical_error_mid_stream_is_raised(self) -> None:
        body = _sse_chunk(audio=b"\x00\x01", status=1)
        body += (
            b"data: "
            + orjson.dumps({
                "data": {"audio": "", "status": 0},
                "base_resp": {"status_code": 1013, "status_msg": "internal error"},
            })
            + b"\n\n"
        )

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_Chunks([body]))

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="1013"):
            await adapter.synthesize_stream(PROMPT)

    async def test_outright_rejection_is_raised_without_sse_framing(self) -> None:
        """A request refused before streaming starts arrives as bare JSON, no ``data:`` lines.

        Confirmed live against the real vendor: an out-of-balance account
        answers a ``stream: true`` request with a single plain JSON object
        (``content-type: application/json``, not ``text/event-stream``) and
        no SSE framing at all. Treating every non-``data:`` line as noise
        would silently drop this and return an empty-but-``ok`` result.
        """
        body = orjson.dumps({"base_resp": {"status_code": 1008, "status_msg": "insufficient balance"}})

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=_Chunks([body]))

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="insufficient balance"):
            await adapter.synthesize_stream(PROMPT)

    async def test_http_error_is_raised_with_the_body(self) -> None:
        """A real error must surface as ``ProviderHttpError``, not ``ResponseNotRead``.

        The response is unread when ``raise_for_status`` first sees it (this
        is what ``http.stream()`` returns before any chunk is consumed), so
        the error body must be constructed via ``stream=`` — an
        ``httpx.Response(content=...)`` shortcut is eagerly buffered at
        construction and would never reproduce the bug.
        """

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, stream=_Chunks([b'{"base_resp": {"status_msg": "bad request"}}']))

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="bad request"):
            await adapter.synthesize_stream(PROMPT)


class TestCrossFamilyValidation:
    def test_family_is_the_vendor(self) -> None:
        assert tts.family_of("minimax-speech28turbo") == "minimax"


LIVE_FLAG = "AUDIO_HARNESS_TEST_MINIMAX_LIVE"


def _needs(*names: str) -> pytest.MarkDecorator:
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(names)} not set")


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a few short live syntheses (fractions of a cent total)",
)
class TestLiveSmoke:
    """A handful of short prompts against the real vendor, en + ja.

    Minimal-API-testing policy: batch and streaming, one English and one
    Japanese prompt only — no bulk runs here.
    """

    BATCH_EN = TtsPrompt(prompt_id="live-batch-en", text="Hello there.", language="en-US")
    STREAM_JA = TtsPrompt(prompt_id="live-stream-ja", text="こんにちは、今日はいい天気ですね。", language="ja-JP")

    @_needs("MINIMAX_API_KEY", "MINIMAX_GROUP_ID")
    async def test_batch_en(self) -> None:
        adapter = tts.create("minimax-speech28turbo")
        try:
            result = await adapter.synthesize(self.BATCH_EN)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0

    @_needs("MINIMAX_API_KEY", "MINIMAX_GROUP_ID")
    async def test_stream_ja(self) -> None:
        adapter = tts.create("minimax-speech28turbo", {"voice_id": "Japanese_CalmLady"})
        try:
            result = await adapter.synthesize_stream(self.STREAM_JA)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.chunk_t_s
        assert result.ttfb_s is not None
