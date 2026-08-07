"""Tests for the Mistral Voxtral Mini TTS HTTP and SSE protocols."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Callable
import os

import httpx
import numpy as np
import orjson
import pytest

from audio_harness import tts
from audio_harness.audio import pcm_f32le_to_s16le
from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import mistral
from audio_harness.types import TtsPrompt


PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake key only for mocked protocol tests."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")


class _Chunks(httpx.AsyncByteStream):
    """An async response body split at caller-selected byte boundaries."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


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


def _mocked_adapter(
    respond: Respond, options: dict[str, object] | None = None
) -> tuple[mistral.MistralVoxtralTts, _RecordingHttp]:
    adapter = tts.create("mistral-voxtral-tts", options)
    assert isinstance(adapter, mistral.MistralVoxtralTts)
    recorder = _RecordingHttp(respond)
    adapter._http = recorder.client()
    return adapter, recorder


def _json_audio(samples: list[float]) -> bytes:
    f32le = np.asarray(samples, dtype="<f4").tobytes()
    return orjson.dumps({"audio_data": base64.b64encode(f32le).decode("ascii")})


def test_registration_and_fixed_capabilities() -> None:
    adapter = tts.create("mistral-voxtral-tts", {"sample_rate": 48000})

    assert isinstance(adapter, mistral.MistralVoxtralTts)
    assert adapter.vendor == "mistral"
    assert adapter.family == "mistral"
    assert adapter.supports_batch is True
    assert adapter.supports_stream is True
    assert adapter.supports_input_streaming is False
    assert adapter.sample_rate == 24000


class TestBatchProtocol:
    """The JSON-wrapped float32 batch response and its request contract."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_request_shape_auth_and_conversion(self) -> None:
        samples = [-0.5, 0.0, 0.25, 0.5]

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_json_audio(samples))

        adapter, recorder = _mocked_adapter(respond)
        result = await adapter.synthesize(PROMPT)

        assert result.ok, result.error
        assert result.encoding == "pcm_s16le"
        assert result.sample_rate == 24000
        assert result.raw["wire_format"] == "f32le_json"
        assert np.frombuffer(result.audio, dtype="<i2").tolist() == [-16383, 0, 8191, 16383]
        assert result.audio_s == pytest.approx(4 / 24000)
        assert result.ttfb_s is None

        [request] = recorder.requests
        assert request.method == "POST"
        assert request.url.path == "/v1/audio/speech"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert orjson.loads(request.content) == {
            "model": "voxtral-mini-tts-2603",
            "input": PROMPT.text,
            "voice_id": "en_paul_neutral",
            "response_format": "pcm",
        }

    async def test_model_and_voice_options_override_defaults(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=_json_audio([]))

        adapter, recorder = _mocked_adapter(
            respond, {"model": "voxtral-mini-tts-latest", "voice_id": "en_alice_neutral"}
        )
        await adapter.synthesize(PROMPT)

        body = orjson.loads(recorder.requests[0].content)
        assert body["model"] == "voxtral-mini-tts-latest"
        assert body["voice_id"] == "en_alice_neutral"

    async def test_invalid_voice_http_error_preserves_vendor_body(self) -> None:
        error = {"object": "error", "type": "invalid_voice", "code": "1902"}

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, content=orjson.dumps(error))

        adapter, _ = _mocked_adapter(respond, {"voice_id": "vivian"})

        with pytest.raises(ProviderHttpError, match="invalid_voice"):
            await adapter.synthesize(PROMPT)


class TestLanguageGuard:
    @pytest.mark.parametrize("language", ["ja", "ja-JP"])
    @pytest.mark.parametrize("method", ["synthesize", "synthesize_stream"])
    async def test_japanese_is_rejected_before_io(self, language: str, method: str) -> None:
        prompt = TtsPrompt(prompt_id="ja", text="こんにちは", language=language)
        adapter = mistral.MistralVoxtralTts()

        with pytest.raises(ValueError, match="unsupported language"):
            await getattr(adapter, method)(prompt)


class TestStreamProtocol:
    """SSE deltas are independently converted and assembled in order."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_two_audio_deltas_then_done(self) -> None:
        first = np.asarray([0.0, 0.25], dtype="<f4").tobytes()
        second = np.asarray([-0.25, 1.0], dtype="<f4").tobytes()
        event_one = (
            b"event: speech.audio.delta\n"
            + b"data: "
            + orjson.dumps({"audio_data": base64.b64encode(first).decode("ascii")})
            + b"\n\n"
        )
        event_two = (
            b"event: speech.audio.delta\r\n"
            + b"data: "
            + orjson.dumps({"audio_data": base64.b64encode(second).decode("ascii")})
            + b"\r\n\r\n"
        )
        done = b"event: speech.audio.done\n\n"

        def respond(request: httpx.Request) -> httpx.Response:
            # Split inside an SSE field to ensure parsing does not depend on
            # transport chunks aligning with event boundaries.
            chunks = [event_one[:17], event_one[17:] + event_two[:11], event_two[11:], done]
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=_Chunks(chunks))

        adapter, recorder = _mocked_adapter(respond)
        result = await adapter.synthesize_stream(PROMPT)

        assert result.ok, result.error
        assert result.audio == pcm_f32le_to_s16le(first) + pcm_f32le_to_s16le(second)
        assert np.frombuffer(result.audio, dtype="<i2").tolist() == [0, 8191, -8191, 32767]
        assert result.audio_s == pytest.approx(4 / 24000)
        assert len(result.chunk_t_s) == 2
        assert result.ttfb_s is not None
        assert result.raw["wire_format"] == "f32le_json"

        [request] = recorder.requests
        assert orjson.loads(request.content) == {
            "model": "voxtral-mini-tts-2603",
            "input": PROMPT.text,
            "voice_id": "en_paul_neutral",
            "response_format": "pcm",
            "stream": True,
        }

    async def test_eof_before_done_raises_instead_of_finalizing_partial_audio(self) -> None:
        partial = np.asarray([0.0, 0.25], dtype="<f4").tobytes()
        event = (
            b"event: speech.audio.delta\n"
            + b"data: "
            + orjson.dumps({"audio_data": base64.b64encode(partial).decode("ascii")})
            + b"\n\n"
        )

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=_Chunks([event]))

        adapter, _ = _mocked_adapter(respond)

        with pytest.raises(RuntimeError, match=r"SSE stream ended before speech\.audio\.done"):
            await adapter.synthesize_stream(PROMPT)

    async def test_leading_utf8_bom_does_not_hide_the_first_delta(self) -> None:
        sample = np.asarray([0.5], dtype="<f4").tobytes()
        stream = (
            b"\xef\xbb\xbfevent: speech.audio.delta\n"
            + b"data: "
            + orjson.dumps({"audio_data": base64.b64encode(sample).decode("ascii")})
            + b"\n\nevent: speech.audio.done\n\n"
        )

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=_Chunks([stream]))

        adapter, _ = _mocked_adapter(respond)
        result = await adapter.synthesize_stream(PROMPT)

        assert np.frombuffer(result.audio, dtype="<i2").tolist() == [16383]
        assert len(result.chunk_t_s) == 1

    async def test_streamed_http_error_preserves_vendor_body(self) -> None:
        error = orjson.dumps({"object": "error", "type": "invalid_voice", "code": "1902"})

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, stream=_Chunks([error]))

        adapter, _ = _mocked_adapter(respond, {"voice_id": "vivian"})

        with pytest.raises(ProviderHttpError, match="invalid_voice"):
            await adapter.synthesize_stream(PROMPT)


class TestFloat32Conversion:
    def test_clamps_to_unit_range_before_scaling(self) -> None:
        payload = np.asarray([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype="<f4").tobytes()

        converted = np.frombuffer(pcm_f32le_to_s16le(payload), dtype="<i2")

        assert converted.tolist() == [-32767, -32767, -16383, 0, 16383, 32767, 32767]

    def test_empty_input_returns_empty_bytes(self) -> None:
        assert pcm_f32le_to_s16le(b"") == b""

    @pytest.mark.parametrize("trailing", [b"\xff", b"\xff\xfe", b"\xff\xfe\xfd"])
    def test_trailing_partial_sample_is_truncated(self, trailing: bytes) -> None:
        complete = np.asarray([0.25, -0.25], dtype="<f4").tobytes()

        assert pcm_f32le_to_s16le(complete + trailing) == pcm_f32le_to_s16le(complete)


LIVE_FLAG = "AUDIO_HARNESS_TEST_MISTRAL_LIVE"


def _needs(*names: str) -> pytest.MarkDecorator:
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(names)} not set")


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run one short live synthesis (a fraction of a cent)",
)
class TestLiveSmoke:
    """One short English batch synthesis against the real endpoint."""

    @_needs("MISTRAL_API_KEY")
    async def test_batch_en(self) -> None:
        prompt = TtsPrompt(prompt_id="smoke-batch-en", text="Hello from Voxtral.", language="en-US")
        adapter = mistral.MistralVoxtralTts()
        try:
            result = await adapter.synthesize(prompt)
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.total_s > 0
