"""Protocol and gated live-smoke tests for OpenRouter audio adapters."""

from __future__ import annotations

from collections.abc import Callable
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
import os
import sys
import wave

import httpx
import pytest

from audio_harness import stt, tts
from audio_harness.stt import openrouter
from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import openrouter as tts_openrouter
from audio_harness.types import AudioClip, TtsPrompt


def make_clip(seconds: float = 0.2, rate: int = 16000) -> AudioClip:
    """Build a silent clip with known PCM and duration."""
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
    """Install a fake credential for protocol tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


Respond = Callable[[httpx.Request], httpx.Response]


def _mocked_adapter(respond: Respond) -> tuple[openrouter.OpenRouterParakeet, list[httpx.Request]]:
    """Create a Parakeet adapter whose HTTP client records requests."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    adapter = stt.create("or-parakeet")
    assert isinstance(adapter, openrouter.OpenRouterParakeet)
    adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return adapter, requests


def _multipart_parts(request: httpx.Request) -> dict[str, list[tuple[str | None, bytes]]]:
    """Parse an HTTPX multipart request into named filename/payload parts."""
    content_type = request.headers["Content-Type"]
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + request.content
    )
    assert message.is_multipart()
    parts: dict[str, list[tuple[str | None, bytes]]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        payload = part.get_payload(decode=True)
        assert isinstance(name, str)
        assert isinstance(payload, bytes)
        parts.setdefault(name, []).append((part.get_filename(), payload))
    return parts


class TestBatchProtocol:
    """OpenRouter's OpenAI-compatible multipart transcription contract."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_request_shape_transcript_and_wav_wrapping(self) -> None:
        clip = make_clip()
        adapter, requests = _mocked_adapter(
            lambda request: httpx.Response(200, json={"text": "hello world", "usage": {"seconds": 0.2}})
        )

        result = await adapter.transcribe_batch(clip)

        assert result.ok, result.error
        assert result.text == "hello world"
        assert result.audio_s == clip.duration_s
        assert result.raw["hosted_proxy"] is True
        assert result.raw["response"] == {"text": "hello world", "usage": {"seconds": 0.2}}

        [request] = requests
        assert request.method == "POST"
        assert request.url == openrouter.TRANSCRIPTIONS_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        parts = _multipart_parts(request)
        assert set(parts) == {"file", "model"}
        assert parts["model"] == [(None, b"nvidia/parakeet-tdt-0.6b-v3")]

        [(filename, wav_payload)] = parts["file"]
        assert filename == "audio.wav"
        with wave.open(BytesIO(wav_payload), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == clip.sample_rate
            assert handle.readframes(handle.getnframes()) == clip.pcm

    async def test_optional_fields_use_the_exact_supported_wire_names(self) -> None:
        adapter, requests = _mocked_adapter(lambda request: httpx.Response(200, json={"text": "bonjour"}))
        adapter.options.update({
            "language": "fr",
            "response_format": "json",
            "temperature": 0,
            "timestamp_granularities": ["segment", "word"],
        })

        await adapter.transcribe_batch(make_clip())

        parts = _multipart_parts(requests[0])
        assert set(parts) == {
            "file",
            "model",
            "language",
            "response_format",
            "temperature",
            "timestamp_granularities[]",
        }
        assert parts["language"] == [(None, b"fr")]
        assert parts["response_format"] == [(None, b"json")]
        assert parts["temperature"] == [(None, b"0")]
        assert parts["timestamp_granularities[]"] == [(None, b"segment"), (None, b"word")]

    async def test_http_error_is_raised_with_the_body(self) -> None:
        adapter, _ = _mocked_adapter(lambda request: httpx.Response(429, text="rate limited"))

        with pytest.raises(ProviderHttpError, match="rate limited"):
            await adapter.transcribe_batch(make_clip())


class TestRegistration:
    """Registry metadata for the OpenRouter Parakeet lane."""

    def test_batch_only_openrouter_family(self) -> None:
        adapter = stt.create("or-parakeet")

        assert isinstance(adapter, openrouter.OpenRouterParakeet)
        assert adapter.model == "nvidia/parakeet-tdt-0.6b-v3"
        assert adapter.vendor == "openrouter"
        assert adapter.family == "openrouter"
        assert adapter.supports_batch is True
        assert adapter.supports_stream is False
        assert stt.family_of("or-parakeet") == "openrouter"

    def test_mai_transcribe_lane(self) -> None:
        adapter = stt.create("or-mai-transcribe")

        assert isinstance(adapter, openrouter.OpenRouterMaiTranscribe)
        assert adapter.model == "microsoft/mai-transcribe-1.5"
        assert adapter.supports_batch is True
        assert adapter.supports_stream is False
        assert stt.family_of("or-mai-transcribe") == "openrouter"

    def test_fish_transcribe_lane(self) -> None:
        adapter = stt.create("or-fish-transcribe")

        assert isinstance(adapter, openrouter.OpenRouterFishTranscribe)
        assert adapter.model == "fish-audio/transcribe-1"
        assert adapter.supports_batch is True
        assert adapter.supports_stream is False
        assert stt.family_of("or-fish-transcribe") == "openrouter"

    def test_2026_08_14_asr_lanes(self) -> None:
        tests = {
            "success: qwen3 asr 0.6b": ("or-qwen3-asr-06b", "qwen/qwen3-asr-0.6b"),
            "success: qwen3 asr 1.7b": ("or-qwen3-asr-17b", "qwen/qwen3-asr-1.7b"),
            "success: voxtral mini 3b": ("or-voxtral-mini", "mistralai/voxtral-mini-3b-2507"),
            "success: voxtral small 24b": ("or-voxtral-small", "mistralai/voxtral-small-24b-2507-stt"),
            "success: nemotron 3.5 asr": (
                "or-nemotron-asr",
                "nvidia/nemotron-3.5-asr-streaming-multilingual-0.6b",
            ),
        }
        for name, (key, model) in tests.items():
            adapter = stt.create(key)
            assert isinstance(adapter, openrouter.OpenRouterStt), name
            assert adapter.model == model, name
            assert adapter.supports_batch is True, name
            assert adapter.supports_stream is False, name
            assert stt.family_of(key) == "openrouter", name


LIVE_FLAG = "AUDIO_HARNESS_TEST_OPENROUTER_LIVE"


def _needs(*names: str) -> pytest.MarkDecorator:
    """Skip a live test when one of its required credentials is absent."""
    missing = [name for name in names if not os.environ.get(name)]
    return pytest.mark.skipif(bool(missing), reason=f"{', '.join(names)} not set")


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run three short OpenRouter audio requests (under $0.05 total)",
)
class TestLiveSmoke:
    """One short Kokoro TTS, MiniMax TTS, and Parakeet STT round trip."""

    @_needs("OPENROUTER_API_KEY")
    async def test_tts_and_stt_lanes(self) -> None:
        kokoro_prompt = TtsPrompt(prompt_id="live-kokoro", text="Hello from OpenRouter.", language="en-US")
        minimax_prompt = TtsPrompt(prompt_id="live-minimax", text="Hello there.", language="en-US")
        kokoro = tts.create("or-kokoro")
        minimax = tts.create("or-minimax-turbo")
        parakeet = stt.create("or-parakeet")
        assert isinstance(kokoro, tts_openrouter.OpenRouterKokoro)
        assert isinstance(minimax, tts_openrouter.OpenRouterMiniMaxTurbo)
        assert isinstance(parakeet, openrouter.OpenRouterParakeet)

        try:
            kokoro_result = await kokoro.synthesize(kokoro_prompt)
            minimax_result = await minimax.synthesize(minimax_prompt)
            clip = AudioClip(
                clip_id="live-kokoro-roundtrip",
                pcm=kokoro_result.audio,
                sample_rate=kokoro_result.sample_rate,
                duration_s=kokoro_result.audio_s,
                reference=kokoro_prompt.text,
                language=kokoro_prompt.language,
                source_path="<openrouter-kokoro>",
            )
            parakeet_result = await parakeet.transcribe_batch(clip)
        finally:
            await kokoro.aclose()
            await minimax.aclose()
            await parakeet.aclose()

        assert kokoro_result.audio_s > 0
        assert minimax_result.audio_s > 0
        assert parakeet_result.text.strip()
        sys.stdout.write(
            "OpenRouter live smoke: "
            f"kokoro_audio_s={kokoro_result.audio_s:.3f}, "
            f"minimax_audio_s={minimax_result.audio_s:.3f}, "
            f"parakeet_transcript={parakeet_result.text!r}\n"
        )
