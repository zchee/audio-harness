"""Protocol and live-independent tests for OpenRouter TTS adapters."""

from __future__ import annotations

from collections.abc import Callable
from io import BytesIO

import httpx
import numpy as np
import orjson
import pytest
import soundfile as sf

from audio_harness import tts
from audio_harness.stt.base import ProviderHttpError
from audio_harness.tts import openrouter
from audio_harness.types import TtsPrompt


PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")

LANES = [
    ("or-kokoro", "hexgrad/kokoro-82m", "af_heart"),
    ("or-orpheus", "canopylabs/orpheus-3b-0.1-ft", "tara"),
    ("or-csm", "sesame/csm-1b", "conversational_a"),
    ("or-zonos", "zyphra/zonos-v0.1-hybrid", "american_female"),
    ("or-minimax-turbo", "minimax/speech-2.8-turbo", "English_expressive_narrator"),
    ("or-minimax-hd", "minimax/speech-2.8-hd", "English_expressive_narrator"),
    ("or-qwen-tts-flash", "qwen/qwen-audio-3.0-tts-flash", "loongjohn"),
    ("or-qwen-tts-plus", "qwen/qwen-audio-3.0-tts-plus", "longanlingxin"),
    ("or-fish-s1", "fish-audio/s1", ""),
    ("or-fish-s2-pro", "fish-audio/s2-pro", ""),
    ("or-fish-s21-pro", "fish-audio/s2.1-pro", ""),
    ("or-flux-tts", "deepgram/flux-tts:free", "flux-hannah-en"),
    ("or-mai-voice-2", "microsoft/mai-voice-2", "en-US-AvaNeural"),
    ("or-mai-voice-2-flash", "microsoft/mai-voice-2-flash", "en-US-AvaNeural"),
]


@pytest.fixture
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake credential for protocol tests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.fixture(scope="module")
def mp3_payload() -> bytes:
    """Encode a tiny real MP3 at 24 kHz for native-rate decode tests."""
    sample_rate = 24000
    times = np.arange(sample_rate // 10, dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
    buffer = BytesIO()
    sf.write(buffer, samples, sample_rate, format="MP3")
    return buffer.getvalue()


Respond = Callable[[httpx.Request], httpx.Response]


def _mocked_adapter(key: str, respond: Respond) -> tuple[openrouter.OpenRouterTts, list[httpx.Request]]:
    """Create an adapter whose HTTP client records requests."""
    requests: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    adapter = tts.create(key)
    assert isinstance(adapter, openrouter.OpenRouterTts)
    adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return adapter, requests


class TestBatchProtocol:
    """OpenRouter's raw-MP3 batch response and request contract."""

    pytestmark = pytest.mark.usefixtures("_credentials")

    async def test_request_shape_and_native_rate_mp3_decode(self, mp3_payload: bytes) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=mp3_payload, headers={"X-Generation-Id": "gen-1"})

        adapter, requests = _mocked_adapter("or-kokoro", respond)

        result = await adapter.synthesize(PROMPT)

        assert result.ok, result.error
        assert result.encoding == "pcm_s16le"
        assert result.sample_rate == 24000
        assert result.audio
        assert result.audio_s > 0
        assert result.ttfb_s is None
        assert result.raw["wire_format"] == "mp3"
        assert result.raw["hosted_proxy"] is True
        assert result.raw["generation_id"] == "gen-1"

        [request] = requests
        assert request.method == "POST"
        assert request.url == openrouter.SPEECH_URL
        assert request.headers["Authorization"] == "Bearer test-key"
        assert orjson.loads(request.content) == {
            "model": "hexgrad/kokoro-82m",
            "input": PROMPT.text,
            "voice": "af_heart",
            "response_format": "mp3",
        }

    @pytest.mark.parametrize(("key", "model", "voice"), LANES)
    async def test_each_lane_sends_its_model_and_default_voice(
        self,
        key: str,
        model: str,
        voice: str,
        mp3_payload: bytes,
    ) -> None:
        adapter, requests = _mocked_adapter(key, lambda request: httpx.Response(200, content=mp3_payload))

        await adapter.synthesize(PROMPT)

        body = orjson.loads(requests[0].content)
        assert body["model"] == model
        if voice:
            assert body["voice"] == voice
        else:
            assert "voice" not in body
        assert body["response_format"] == "mp3"

    @pytest.mark.parametrize("key", [lane[0] for lane in LANES])
    async def test_explicit_voice_option_wins(self, key: str, mp3_payload: bytes) -> None:
        adapter, requests = _mocked_adapter(key, lambda request: httpx.Response(200, content=mp3_payload))
        adapter.options["voice"] = "explicit-voice"

        await adapter.synthesize(PROMPT)

        assert orjson.loads(requests[0].content)["voice"] == "explicit-voice"

    async def test_http_error_is_raised_with_the_body(self) -> None:
        adapter, _ = _mocked_adapter("or-kokoro", lambda request: httpx.Response(401, text="invalid api key"))

        with pytest.raises(ProviderHttpError, match="invalid api key"):
            await adapter.synthesize(PROMPT)


class TestRegistration:
    """Registry metadata shared by all OpenRouter TTS lanes."""

    @pytest.mark.parametrize("key", [lane[0] for lane in LANES])
    def test_batch_only_openrouter_family(self, key: str) -> None:
        adapter = tts.create(key)

        assert isinstance(adapter, openrouter.OpenRouterTts)
        assert adapter.vendor == "openrouter"
        assert adapter.family == "openrouter"
        assert adapter.supports_batch is True
        assert adapter.supports_stream is False
        assert tts.family_of(key) == "openrouter"
