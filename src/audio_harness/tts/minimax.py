"""MiniMax Speech-2.8 Turbo text-to-speech adapter."""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urlencode

import orjson

from audio_harness.audio import decode_audio_duration
from audio_harness.config import require_env
from audio_harness.stt.base import ProviderHttpError, raise_for_status
from audio_harness.types import Mode, TtsPrompt, TtsResult

from .base import ChunkTimeline, TtsProvider, register, stamp_stream_timing


T2A_URL = "https://api.minimax.io/v1/t2a_v2"
DEFAULT_VOICE_ID = "English_expressive_narrator"
"""Confirmed system voice id (docs: platform.minimax.io/docs/faq/system-voice-id)."""

AUDIO_FORMAT = "pcm"
"""Headerless PCM — same reasoning as the other adapters' ``container=none``
choice: a WAV/MP3 container would count its own bytes as audio."""

_LANGUAGE_BOOST = {
    "zh": "Chinese",
    "yue": "Chinese,Yue",
    "en": "English",
    "ar": "Arabic",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
    "de": "German",
    "tr": "Turkish",
    "nl": "Dutch",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ja": "Japanese",
    "it": "Italian",
    "ko": "Korean",
    "th": "Thai",
    "pl": "Polish",
    "ro": "Romanian",
    "el": "Greek",
    "cs": "Czech",
    "fi": "Finnish",
    "hi": "Hindi",
    "bg": "Bulgarian",
    "da": "Danish",
    "he": "Hebrew",
    "iw": "Hebrew",
    "ms": "Malay",
    "fa": "Persian",
    "sk": "Slovak",
    "sv": "Swedish",
    "hr": "Croatian",
    "fil": "Filipino",
    "tl": "Filipino",
    "hu": "Hungarian",
    "no": "Norwegian",
    "nb": "Norwegian",
    "sl": "Slovenian",
    "ca": "Catalan",
    "nn": "Nynorsk",
    "ta": "Tamil",
    "af": "Afrikaans",
}
"""BCP-47 primary subtag to the vendor's ``language_boost`` enum value
(docs: platform.minimax.io/docs/api-reference/speech-t2a-http). A prompt
language outside this map falls back to ``"auto"``, the vendor's own
detect-and-enhance value."""


def _language_boost(language: str) -> str:
    """Map a prompt's BCP-47 tag to MiniMax's ``language_boost`` enum."""
    primary = language.split("-")[0].lower()
    return _LANGUAGE_BOOST.get(primary, "auto")


@register
class MiniMaxSpeech28Turbo(TtsProvider):
    """MiniMax Speech-2.8 Turbo, batch and HTTP-streaming (SSE) modes.

    One voice is pinned across every language in the benchmark — the same
    convention ElevenLabs, Inworld and Cartesia use here, comparing engines on
    a fixed voice rather than cross-shopping voices per language. MiniMax's
    327+ system voices are catalogued per language (docs confirm a dedicated
    Japanese set, e.g. ``Japanese_CalmLady``), but the model is documented to
    speak all 40 supported languages from any voice, with ``language_boost``
    as a pronunciation hint — so the default English voice carries across
    languages, and ``language_boost`` is derived per prompt instead. Hungarian,
    Vietnamese, Polish and Turkish are confirmed in the vendor's
    ``language_boost`` enum (docs) but were not individually voice-checked
    beyond that; Japanese is the one language explicitly cross-checked against
    both the voice catalog and the enum.

    There is no dedicated non-streaming endpoint distinct from the streaming
    one: both request shapes hit ``t2a_v2``, differing only in the ``stream``
    flag. Streaming responses arrive as Server-Sent Events, one JSON object
    per ``data:`` line, each carrying a hex-encoded audio fragment;
    ``stream_options.exclude_aggregated_audio`` asks the vendor to omit the
    final chunk that otherwise repeats the entire utterance, and any chunk
    that still arrives with an ``extra_info`` key (that duplicate) is skipped
    defensively rather than trusted to never appear.

    A request can be HTTP 200 while the vendor's own ``base_resp.status_code``
    reports failure — that JSON field, not the HTTP status line, is the
    authoritative success signal on this API.

    Options:
        model_id: Model identifier; defaults to ``speech-2.8-turbo``.
        voice_id: System, cloned or AI-generated voice id. Falls back to the
            ``MINIMAX_VOICE_ID`` environment variable, then a built-in voice.
        sample_rate: Output rate; defaults to 24 kHz (vendor accepts 8000,
            16000, 22050, 24000, 32000 or 44100 Hz).
    """

    key = "minimax-speech28turbo"
    vendor = "minimax"
    supports_batch = True
    supports_stream = True

    def _model(self) -> str:
        return str(self.options.get("model_id", "speech-2.8-turbo"))

    def _voice_id(self) -> str:
        voice = self.options.get("voice_id")
        if voice:
            return str(voice)
        return os.environ.get("MINIMAX_VOICE_ID", DEFAULT_VOICE_ID)

    def _api_key(self) -> str:
        return require_env("MINIMAX_API_KEY", self.key)

    def _group_id(self) -> str:
        return require_env("MINIMAX_GROUP_ID", self.key)

    def _url(self) -> str:
        return f"{T2A_URL}?{urlencode({'GroupId': self._group_id()})}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

    def _audio_setting(self) -> dict[str, Any]:
        return {
            "format": AUDIO_FORMAT,
            "sample_rate": self.sample_rate,
            "channel": 1,
        }

    def _body(self, prompt: TtsPrompt, *, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model(),
            "text": prompt.text,
            "stream": stream,
            "voice_setting": {"voice_id": self._voice_id()},
            "audio_setting": self._audio_setting(),
            "language_boost": _language_boost(prompt.language),
        }
        if stream:
            body["stream_options"] = {"exclude_aggregated_audio": True}
        return body

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize over HTTP; the response is one JSON blob with hex audio."""
        result = self._result(prompt, Mode.BATCH)
        started = time.perf_counter()
        response = await self.http.post(self._url(), headers=self._headers(), json=self._body(prompt, stream=False))
        raise_for_status(response, self.key)
        payload = response.json()
        _raise_for_base_resp(payload, self.key)
        result.ttfb_s = None
        result.total_s = time.perf_counter() - started
        return _finish(result, bytes.fromhex(payload["data"]["audio"]))

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize over the SSE HTTP stream, stamping every audio chunk."""
        result = self._result(prompt, Mode.STREAM)
        timeline = ChunkTimeline()

        async with self.http.stream(
            "POST", self._url(), headers=self._headers(), json=self._body(prompt, stream=True)
        ) as response:
            if response.status_code >= 400:
                # raise_for_status reads response.text; on a streamed response
                # that raises ResponseNotRead unless the body is buffered
                # first, which would replace the vendor's error with a
                # confusing one about our own HTTP client.
                await response.aread()
            raise_for_status(response, self.key)
            async for line in response.aiter_lines():
                payload = _parse_stream_line(line)
                if payload is None:
                    continue
                _raise_for_base_resp(payload, self.key)
                if "extra_info" in payload:
                    continue
                chunk = payload.get("data", {}).get("audio")
                if chunk:
                    timeline.add(bytes.fromhex(chunk))

        return _finish_stream(result, timeline)


def _parse_stream_line(line: str) -> dict[str, Any] | None:
    """Parse one line of a ``stream: true`` response body, or ``None``.

    Ordinary chunks arrive as SSE ``data: {...}`` lines. A request the vendor
    rejects outright before streaming starts — an out-of-balance account, for
    one confirmed case — instead answers with a single plain JSON object and
    no SSE framing at all, so a bare JSON line is parsed too rather than
    silently dropped; a request that fails silently is worse than one that
    raises with the vendor's own message.
    """
    line = line.strip()
    if not line:
        return None
    data = line.removeprefix("data:").strip() if line.startswith("data:") else line
    try:
        payload = orjson.loads(data)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _raise_for_base_resp(payload: dict[str, Any], provider: str) -> None:
    """Raise if the vendor's own status channel reports failure.

    ``base_resp.status_code`` can be nonzero on an HTTP 200 response — the
    JSON body, not the HTTP status line, is this API's authoritative outcome.
    """
    base_resp = payload.get("base_resp") or {}
    code = base_resp.get("status_code", 0)
    if code:
        raise ProviderHttpError(f"{provider}: MiniMax error {code}: {base_resp.get('status_msg', 'unknown error')}")


def _finish(result: TtsResult, audio: bytes) -> TtsResult:
    """Attach audio to a result and derive its duration."""
    result.audio = audio
    result.encoding = "pcm_s16le"
    result.audio_s = decode_audio_duration(audio, encoding=result.encoding, sample_rate=result.sample_rate)
    return result


def _finish_stream(result: TtsResult, timeline: ChunkTimeline) -> TtsResult:
    """Attach streamed audio and stamp the chunk-timing metrics."""
    _finish(result, timeline.audio)
    stamp_stream_timing(result, timeline)
    return result
