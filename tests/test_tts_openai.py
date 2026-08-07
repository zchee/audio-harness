"""Tests for the OpenAI gpt-4o-mini-tts adapter's wire protocol.

Both transports share one endpoint (``POST /v1/audio/speech``) and are pinned
with ``httpx.MockTransport`` — no network call — the same pattern as MiniMax's
plain-HTTP adapter. The streaming lane uses ``stream_format=audio`` (raw
chunked bytes), not the ``sse`` variant, so chunk assembly needs no
event-parsing: ``response.aiter_bytes()`` already yields exactly what
``ChunkTimeline`` wants.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
import os

import httpx
import orjson
import pytest

from audio_harness import tts
from audio_harness.cli import validate_roundtrip_judges
from audio_harness.config import BenchmarkConfig, ConfigError
from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import openai
from audio_harness.types import TtsPrompt


PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials for the mocked protocol tests.

    Not autouse: :class:`TestLiveSmoke` needs the *real* environment
    variable, and this fixture would otherwise clobber it.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


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


def _mocked_adapter(respond: Respond) -> tuple[openai.OpenAiGpt4oMiniTts, _RecordingHttp]:
    adapter = tts.create("openai-gpt4o-mini-tts")
    assert isinstance(adapter, openai.OpenAiGpt4oMiniTts)
    recorder = _RecordingHttp(respond)
    adapter._http = recorder.client()
    return adapter, recorder


class TestBatchProtocol:
    """The plain (non-streaming) request: URL, headers, body and parsing."""

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
        assert request.url.path == "/v1/audio/speech"
        assert request.headers["Authorization"] == "Bearer test-key"
        body = orjson.loads(request.content)
        assert body == {
            "model": "gpt-4o-mini-tts-2025-12-15",
            "input": PROMPT.text,
            "voice": "alloy",
            "response_format": "pcm",
        }

    async def test_http_error_is_raised_with_the_body(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, text="invalid api key")

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="401"):
            await adapter.synthesize(PROMPT)

    async def test_voice_id_option_overrides_the_default(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        adapter, recorder = _mocked_adapter(respond)
        adapter.options["voice_id"] = "coral"

        await adapter.synthesize(PROMPT)

        assert orjson.loads(recorder.requests[0].content)["voice"] == "coral"

    async def test_model_id_option_overrides_the_default(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        adapter, recorder = _mocked_adapter(respond)
        adapter.options["model_id"] = "gpt-4o-mini-tts"

        await adapter.synthesize(PROMPT)

        assert orjson.loads(recorder.requests[0].content)["model"] == "gpt-4o-mini-tts"

    async def test_instructions_and_speed_options_are_included_when_set(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        adapter, recorder = _mocked_adapter(respond)
        adapter.options["instructions"] = "speak calmly"
        adapter.options["speed"] = 1.25

        await adapter.synthesize(PROMPT)

        body = orjson.loads(recorder.requests[0].content)
        assert body["instructions"] == "speak calmly"
        assert body["speed"] == 1.25

    async def test_instructions_and_speed_are_omitted_by_default(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        adapter, recorder = _mocked_adapter(respond)

        await adapter.synthesize(PROMPT)

        body = orjson.loads(recorder.requests[0].content)
        assert "instructions" not in body
        assert "speed" not in body


class TestStreamProtocol:
    """The same endpoint with ``stream_format=audio``: chunked delivery."""

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
        assert request.url.path == "/v1/audio/speech"
        body = orjson.loads(request.content)
        assert body["stream_format"] == "audio"
        assert body["response_format"] == "pcm"

    async def test_http_error_is_raised_with_the_body(self) -> None:
        """A real error must surface as ``ProviderHttpError``, not ``ResponseNotRead``.

        The response is unread when ``raise_for_status`` first sees it (this
        is what ``http.stream()`` returns before any chunk is consumed), so
        the error body must be constructed via ``stream=`` — an
        ``httpx.Response(content=...)`` shortcut is eagerly buffered at
        construction and would never reproduce the bug.
        """

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, stream=_Chunks([b'{"error": {"message": "invalid voice"}}']))

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(ProviderHttpError, match="invalid voice"):
            await adapter.synthesize_stream(PROMPT)


class TestCrossFamilyValidation:
    """family=\"openai\" collides with whisper-local by design (P0's rule);
    the judge set for this candidate must include a non-OpenAI-lineage judge.
    """

    def test_family_is_openai(self) -> None:
        assert tts.family_of("openai-gpt4o-mini-tts") == "openai"

    def test_whisper_local_alone_cannot_judge_this_lane(self) -> None:
        config = BenchmarkConfig.from_dict({
            "tts": ["openai-gpt4o-mini-tts"],
            "roundtrip_stt": ["whisper-local"],
        })

        with pytest.raises(ConfigError, match="openai"):
            validate_roundtrip_judges(config)

    def test_a_non_openai_judge_is_accepted(self) -> None:
        config = BenchmarkConfig.from_dict({
            "tts": ["openai-gpt4o-mini-tts"],
            "roundtrip_stt": [{"name": "deepgram-nova3"}],
        })

        validate_roundtrip_judges(config)

    def test_deepgram_and_whisper_local_judges_cover_the_candidate(self) -> None:
        config = BenchmarkConfig.from_dict({
            "tts": ["openai-gpt4o-mini-tts"],
            "roundtrip_stt": [{"name": "deepgram-nova3"}, "whisper-local"],
        })

        validate_roundtrip_judges(config)


LIVE_FLAG = "AUDIO_HARNESS_TEST_OPENAI_TTS_LIVE"


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

    @_needs("OPENAI_API_KEY")
    async def test_batch_en(self) -> None:
        adapter = tts.create("openai-gpt4o-mini-tts")
        try:
            result = await adapter.synthesize(self.BATCH_EN)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0

    @_needs("OPENAI_API_KEY")
    async def test_stream_ja(self) -> None:
        adapter = tts.create("openai-gpt4o-mini-tts")
        try:
            result = await adapter.synthesize_stream(self.STREAM_JA)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.chunk_t_s
        assert result.ttfb_s is not None
