"""Protocol and live-smoke tests for the Inworld TTS-2 Realtime adapter.

The WebSocket message shapes, header-based auth and HTTP batch response shape
pin the wire format inferred from the vendor's own reference client
(``inworld-api-examples/tts/python/example_websocket.py`` and
``example_tts.py``), since the auto-generated API reference disagreed with
that working example on how WebSocket auth is transmitted.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
import os
from typing import Any

import httpx
import numpy as np
import orjson
import pytest
from websockets.asyncio.server import ServerConnection, serve

from audio_harness import tts
from audio_harness.tts import inworld
from audio_harness.types import TtsPrompt


RATE = 24000


def make_pcm(seconds: float, rate: int = RATE) -> bytes:
    """Mono 16-bit PCM tone, headerless."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    samples = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class FakeInworldWs:
    """Speaks the ``voice:streamBidirectional`` protocol shape the adapter expects."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.auth_header: str | None = None
        self.pcm = make_pcm(0.2)

    async def __call__(self, socket: ServerConnection) -> None:
        self.auth_header = socket.request.headers.get("Authorization") if socket.request else None
        context_id: str | None = None
        async for frame in socket:
            message = orjson.loads(frame)
            self.messages.append(message)
            context_id = str(message.get("context_id", context_id))
            if "create" in message:
                await socket.send(
                    orjson.dumps({
                        "result": {"context_id": context_id, "contextCreated": message["create"]},
                    }).decode()
                )
            elif "send_text" in message:
                half = len(self.pcm) // 2 // 2 * 2
                for piece in (self.pcm[:half], self.pcm[half:]):
                    await socket.send(
                        orjson.dumps({
                            "result": {
                                "context_id": context_id,
                                "audioChunk": {"audioContent": base64.b64encode(piece).decode()},
                            },
                        }).decode()
                    )
            elif "close_context" in message:
                await socket.send(orjson.dumps({"result": {"context_id": context_id, "contextClosed": {}}}).decode())
                return


@pytest.fixture
async def inworld_ws(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeInworldWs]:
    """Run a fake Inworld WebSocket endpoint and point the adapter at it."""
    handler = FakeInworldWs()
    async with serve(handler, "127.0.0.1", 0) as running:
        port = running.sockets[0].getsockname()[1]
        monkeypatch.setattr(inworld, "WS_URL", f"ws://127.0.0.1:{port}")
        monkeypatch.setenv("INWORLD_API_KEY", "test-key")
        yield handler


class TestInworldWebSocketProtocol:
    """The adapter's wire behavior against a real local WebSocket."""

    PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")

    async def test_stream_populates_chunk_timing(self, inworld_ws: FakeInworldWs) -> None:
        adapter = tts.create("inworld-tts2")

        result = await adapter.synthesize_stream(self.PROMPT)

        assert result.ok, result.error
        assert len(result.chunk_t_s) == 2
        assert result.ttfb_s is not None
        assert result.audio_s > 0

    async def test_sends_basic_auth_header_not_query_param(self, inworld_ws: FakeInworldWs) -> None:
        adapter = tts.create("inworld-tts2")

        await adapter.synthesize_stream(self.PROMPT)

        assert inworld_ws.auth_header == "Basic test-key"

    async def test_context_flow_is_create_send_close(self, inworld_ws: FakeInworldWs) -> None:
        adapter = tts.create("inworld-tts2")

        await adapter.synthesize_stream(self.PROMPT)

        kinds = [next(k for k in m if k != "context_id") for m in inworld_ws.messages]
        assert kinds == ["create", "send_text", "close_context"]

        create_msg = inworld_ws.messages[0]["create"]
        assert create_msg["voice_id"] == inworld.DEFAULT_VOICE_ID
        assert create_msg["model_id"] == "inworld-tts-2"
        assert create_msg["audio_config"]["audio_encoding"] == "PCM"

        context_ids = {str(m["context_id"]) for m in inworld_ws.messages}
        assert context_ids == {"inworld-tts2-p1"}, "one context per synthesis call"

        send_text_msg = inworld_ws.messages[1]["send_text"]
        assert send_text_msg["text"] == self.PROMPT.text
        assert "flush_context" in send_text_msg

    async def test_voice_and_model_options_are_honored(self, inworld_ws: FakeInworldWs) -> None:
        adapter = tts.create("inworld-tts2", {"voice_id": "Hana", "model_id": "inworld-tts-1.5-max"})

        await adapter.synthesize_stream(self.PROMPT)

        create_msg = inworld_ws.messages[0]["create"]
        assert create_msg["voice_id"] == "Hana"
        assert create_msg["model_id"] == "inworld-tts-1.5-max"

    async def test_server_error_is_recorded(self, inworld_ws: FakeInworldWs, monkeypatch: pytest.MonkeyPatch) -> None:
        async def error_handler(socket: ServerConnection) -> None:
            async for _frame in socket:
                await socket.send(orjson.dumps({"error": {"message": "boom"}}).decode())
                return

        async with serve(error_handler, "127.0.0.1", 0) as running:
            port = running.sockets[0].getsockname()[1]
            monkeypatch.setattr(inworld, "WS_URL", f"ws://127.0.0.1:{port}")
            adapter = tts.create("inworld-tts2")

            result = await adapter.synthesize_stream(self.PROMPT)

        assert not result.ok
        assert "boom" in str(result.error)


class TestInworldBatchProtocol:
    """The adapter's batch HTTP request/response shape, mocked via httpx transport."""

    PROMPT = TtsPrompt(prompt_id="p1", text="hello wonderful world", language="en-US")

    async def test_batch_decodes_base64_audio_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INWORLD_API_KEY", "test-key")
        pcm = make_pcm(0.1)
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json={"audioContent": base64.b64encode(pcm).decode()})

        adapter = tts.create("inworld-tts2")
        adapter._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        result = await adapter.synthesize(self.PROMPT)

        assert result.ok, result.error
        assert result.audio == pcm
        assert result.audio_s == pytest.approx(0.1, abs=0.01)

        assert len(captured) == 1
        request = captured[0]
        assert request.url == inworld.HTTP_URL
        assert request.headers["Authorization"] == "Basic test-key"
        body = orjson.loads(request.content)
        assert body["text"] == self.PROMPT.text
        assert body["voice_id"] == inworld.DEFAULT_VOICE_ID
        assert body["model_id"] == "inworld-tts-2"
        assert body["language"] == "en-US"
        assert body["audio_config"] == {"audio_encoding": "PCM", "sample_rate_hertz": 24000}


LIVE_FLAG = "AUDIO_HARNESS_TEST_INWORLD_LIVE"


@pytest.mark.skipif(
    not os.environ.get(LIVE_FLAG),
    reason=f"set {LIVE_FLAG}=1 to run a few short live syntheses (fractions of a cent total)",
)
class TestLiveSmoke:
    """A handful of short prompts against the real Inworld endpoints.

    Minimal-API-testing policy: batch, stream and one Japanese stream prompt
    only — no bulk runs here.
    """

    async def test_batch_en(self) -> None:
        adapter = tts.create("inworld-tts2")
        try:
            result = await adapter.synthesize(TtsPrompt(prompt_id="live-batch", text="Hello there.", language="en-US"))
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0

    async def test_stream_en(self) -> None:
        adapter = tts.create("inworld-tts2")
        try:
            result = await adapter.synthesize_stream(
                TtsPrompt(prompt_id="live-stream-en", text="What a lovely day for a test.", language="en-US")
            )
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
        assert result.chunk_t_s
        assert result.ttfb_s is not None

    async def test_stream_ja(self) -> None:
        adapter = tts.create("inworld-tts2")
        try:
            result = await adapter.synthesize_stream(
                TtsPrompt(prompt_id="live-stream-ja", text="こんにちは、今日はいい天気ですね。", language="ja-JP")
            )
        finally:
            await adapter.aclose()

        assert result.ok, result.error
        assert result.audio_s > 0
