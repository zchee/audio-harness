"""Deepgram Nova-3 and Flux speech-to-text adapters."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

import orjson
from websockets.asyncio.client import ClientConnection

from audio_harness.audio import wrap_wav
from audio_harness.config import require_env
from audio_harness.types import AudioClip, EventKind, Mode, SttResult

from .base import StreamTimeline, SttProvider, raise_for_status, register
from .ws import StreamProtocolError, run_stream


BATCH_URL = "https://api.deepgram.com/v1/listen"
STREAM_URL = "wss://api.deepgram.com/v1/listen"
FLUX_STREAM_URL = "wss://api.deepgram.com/v2/listen"


@register
class DeepgramNova3(SttProvider):
    """Deepgram Nova-3 over the REST and WebSocket listen endpoints.

    Options:
        model: Model name; ``nova-3`` (monolingual) or ``nova-3-multilingual``.
        language: Language code sent to Deepgram. Use ``multi`` with the
            multilingual model to enable code-switching.
        smart_format: Whether Deepgram applies punctuation and formatting.
        endpointing: Streaming silence threshold in milliseconds before a
            ``speech_final`` result, or ``false`` to disable. Deepgram's
            server default is 10 ms when unset.
        utterance_end_ms: When set, Deepgram also sends ``UtteranceEnd``
            messages after this word-gap (>= 1000 recommended; requires
            interim results, which this adapter always enables).
    """

    key = "deepgram-nova3"
    vendor = "deepgram"
    supports_batch = True
    supports_stream = True

    def _model(self) -> str:
        return str(self.options.get("model", "nova-3"))

    def _language(self, clip: AudioClip) -> str:
        return str(self.options.get("language", clip.language.split("-")[0]))

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Token {require_env('DEEPGRAM_API_KEY', self.key)}"}

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Post the clip as a WAV file to the pre-recorded endpoint."""
        result = self._result(clip, Mode.BATCH)
        params = {
            "model": self._model(),
            "language": self._language(clip),
            "smart_format": str(self.options.get("smart_format", True)).lower(),
        }
        response = await self.http.post(
            f"{BATCH_URL}?{urlencode(params)}",
            headers={**self._auth(), "Content-Type": "audio/wav"},
            content=wrap_wav(clip.pcm, clip.sample_rate),
        )
        raise_for_status(response, self.key)
        payload = response.json()
        result.total_s = response.elapsed.total_seconds()
        result.text = _batch_transcript(payload)
        result.raw["response"] = payload
        return result

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream PCM over the listen WebSocket with interim results enabled."""
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        params = {
            "model": self._model(),
            "language": self._language(clip),
            "encoding": "linear16",
            "sample_rate": str(clip.sample_rate),
            "channels": "1",
            "interim_results": "true",
            "smart_format": str(self.options.get("smart_format", True)).lower(),
        }
        endpointing = self.options.get("endpointing")
        utterance_end_ms = self.options.get("utterance_end_ms")
        if endpointing is not None:
            params["endpointing"] = str(endpointing).lower()
        if utterance_end_ms:
            params["utterance_end_ms"] = str(utterance_end_ms)

        async def close_stream(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "CloseStream"}).decode())

        await run_stream(
            url=f"{STREAM_URL}?{urlencode(params)}",
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=_handle_message,
            on_input_done=close_stream,
        )

        result.text = timeline.concat_finals()
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        # Deepgram's endpointing is on server-side by default (10 ms), so
        # speech_final events exist on every stream — the lane is always
        # EOU-capable. The effective knob values are recorded because a
        # ranking is only comparable when each lane's configuration is known.
        result.raw["eou_source"] = "speech_final+utterance_end" if utterance_end_ms else "speech_final"
        result.raw["endpoint_config"] = {
            "endpointing": endpointing if endpointing is not None else 10,
            "utterance_end_ms": utterance_end_ms,
        }
        return result


@register
class DeepgramFlux(SttProvider):
    """Deepgram Flux over the model-native turn-based streaming endpoint.

    Options:
        model: ``flux-general-en`` (default) or ``flux-general-multi``.
        language_hint: Language-code list sent as repeatable query parameters
            for ``flux-general-multi`` only.
        eot_threshold: End-of-turn confidence threshold from 0.5 to 0.9;
            the server default is 0.7 when omitted.
        eager_eot_threshold: Eager end-of-turn threshold from 0.3 to 0.9.
            It is unset by default because the server emits eager events only
            when explicitly enabled, and it cannot exceed ``eot_threshold``
            when both options are set.
        eot_timeout_ms: Forced end-of-turn timeout from 500 to 60000 ms; the
            server default is 5000 ms when omitted.

    Flux claims approximately 260 ms end-of-turn latency and recommends 80 ms
    audio chunks. This lane is the flagship candidate for this project's
    endpointing benchmark. It uses the v2 listen endpoint, whose documented
    end-of-input signal remains the text JSON ``{"type":"CloseStream"}``.
    Observed live 2026-08-08: the server closes promptly on ``CloseStream``
    WITHOUT draining pending turn decisions — an ``EndOfTurn`` fires only
    while audio is still flowing, so speech running to the end of input
    yields interim hypotheses but no final (see the fallback in
    :meth:`transcribe_stream`).
    """

    key = "deepgram-flux"
    vendor = "deepgram"
    supports_stream = True

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Token {require_env('DEEPGRAM_API_KEY', self.key)}"}

    def _stream_options(self) -> tuple[str, list[str], dict[str, float | int]]:
        """Validate and normalize Flux query options."""
        model = str(self.options.get("model", "flux-general-en"))
        if model not in {"flux-general-en", "flux-general-multi"}:
            raise ValueError(f"{self.key}: model must be 'flux-general-en' or 'flux-general-multi', got {model!r}")

        language_hint_value = self.options.get("language_hint", [])
        if not isinstance(language_hint_value, list) or not all(
            isinstance(value, str) for value in language_hint_value
        ):
            raise ValueError(f"{self.key}: language_hint must be a list of language codes")
        language_hints = language_hint_value

        endpoint_config: dict[str, float | int] = {}
        if "eot_threshold" in self.options:
            endpoint_config["eot_threshold"] = _flux_float_option(
                "eot_threshold", self.options["eot_threshold"], 0.5, 0.9
            )
        if "eager_eot_threshold" in self.options:
            endpoint_config["eager_eot_threshold"] = _flux_float_option(
                "eager_eot_threshold", self.options["eager_eot_threshold"], 0.3, 0.9
            )
        if "eot_timeout_ms" in self.options:
            endpoint_config["eot_timeout_ms"] = _flux_int_option(
                "eot_timeout_ms", self.options["eot_timeout_ms"], 500, 60000
            )

        eot = endpoint_config.get("eot_threshold")
        eager = endpoint_config.get("eager_eot_threshold")
        if eot is not None and eager is not None and eager > eot:
            raise ValueError(f"{self.key}: eager_eot_threshold ({eager}) must be <= eot_threshold ({eot})")
        return model, language_hints, endpoint_config

    async def transcribe_stream(self, clip: AudioClip, *, chunk_ms: int, realtime: bool) -> SttResult:
        """Stream PCM to Flux and collect its model-native turn events."""
        model, language_hints, endpoint_config = self._stream_options()
        result = self._result(clip, Mode.STREAM)
        timeline = StreamTimeline()
        params: dict[str, str | list[str]] = {
            "model": model,
            "encoding": "linear16",
            "sample_rate": str(clip.sample_rate),
        }
        if model == "flux-general-multi" and language_hints:
            params["language_hint"] = language_hints
        params.update({name: str(value) for name, value in endpoint_config.items()})

        async def close_stream(socket: ClientConnection) -> None:
            await socket.send(orjson.dumps({"type": "CloseStream"}).decode())

        handler = _FluxTurnHandler()
        await run_stream(
            url=f"{FLUX_STREAM_URL}?{urlencode(params, doseq=True)}",
            headers=self._auth(),
            clip=clip,
            chunk_ms=chunk_ms,
            realtime=realtime,
            timeline=timeline,
            handle_message=handler,
            on_input_done=close_stream,
        )

        result.text = timeline.concat_finals()
        if not result.text and timeline.partials:
            # Speech that runs to the end of input never crosses the model's
            # end-of-turn confidence threshold, and after CloseStream the
            # live server closes without restating a final (observed
            # 2026-08-08). The last cumulative hypothesis is the transcript;
            # the missing turn decision is recorded, not invented.
            result.text = timeline.partials[-1].text
            result.raw["turn_completed"] = False
        result.partials = timeline.partials
        result.ttft_s = timeline.ttft_s
        result.finalize_s = timeline.finalize_s
        result.total_s = timeline.total_s
        result.raw["ws_rtt_s"] = timeline.ws_rtt_s
        result.raw["eou_source"] = "end_of_turn"
        result.raw["eager_events"] = handler.eager_events
        result.raw["endpoint_config"] = endpoint_config
        return result


def _flux_float_option(name: str, value: Any, minimum: float, maximum: float) -> float:
    """Return a bounded floating-point Flux option."""
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"deepgram-flux: {name} must be a number from {minimum} to {maximum}") from exc
    if not minimum <= normalized <= maximum:
        raise ValueError(f"deepgram-flux: {name} must be from {minimum} to {maximum}, got {value!r}")
    return normalized


def _flux_int_option(name: str, value: Any, minimum: int, maximum: int) -> int:
    """Return a bounded integer Flux option."""
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"deepgram-flux: {name} must be an integer from {minimum} to {maximum}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"deepgram-flux: {name} must be an integer, got {value!r}")
    if not minimum <= normalized <= maximum:
        raise ValueError(f"deepgram-flux: {name} must be from {minimum} to {maximum}, got {value!r}")
    return normalized


class _FluxTurnHandler:
    """Map Flux turn events and deduplicate definitive turn decisions."""

    __slots__ = ("_eou_turns", "eager_events")

    def __init__(self) -> None:
        self._eou_turns: set[int] = set()
        self.eager_events: list[dict[str, object]] = []

    def __call__(self, payload: Any, timeline: StreamTimeline) -> bool:
        if not isinstance(payload, dict):
            return False
        message_type = payload.get("type")
        if message_type in {"Error", "error"} or payload.get("error"):
            raise StreamProtocolError(f"deepgram-flux: {payload}")
        if message_type != "TurnInfo":
            return False

        event = payload.get("event")
        turn_index = int(payload.get("turn_index") or 0)
        if event in {"EagerEndOfTurn", "TurnResumed"}:
            self.eager_events.append({
                "event": event,
                "t_s": timeline.elapsed(),
                "turn_index": turn_index,
                "end_of_turn_confidence": payload.get("end_of_turn_confidence"),
            })
            return False
        if event == "EndOfTurn":
            if turn_index in self._eou_turns:
                return False
            self._eou_turns.add(turn_index)
            timeline.record(
                str(payload.get("transcript", "")),
                is_final=True,
                kind=EventKind.EOU,
            )
            return False
        if event in {"StartOfTurn", "Update"}:
            timeline.record(str(payload.get("transcript", "")), is_final=False)
        return False


def _batch_transcript(payload: dict[str, Any]) -> str:
    """Extract the transcript from a pre-recorded response."""
    channels = payload.get("results", {}).get("channels", [])
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives", [])
    return str(alternatives[0].get("transcript", "")) if alternatives else ""


def _handle_message(payload: Any, timeline: StreamTimeline) -> bool:
    """Record one listen-socket event.

    Deepgram emits one ``Results`` frame per interim update and one with
    ``is_final`` per finished segment, so the full transcript is the
    concatenation of the final frames. Two independent signals mark end of
    utterance: ``speech_final`` on a Results frame (VAD endpointing) and the
    bare ``UtteranceEnd`` message (word-gap analysis, when enabled). Both are
    recorded as EOU events; a segment final without ``speech_final`` is not
    an endpointing decision and stays ``segment_final``.
    """
    if not isinstance(payload, dict):
        return False
    kind = payload.get("type")
    if kind == "Metadata":
        return True
    if kind == "UtteranceEnd":
        # last_word_end of -1 means the result was already finalized before
        # the gap condition was met; Deepgram documents it as a stale
        # notification to ignore.
        if payload.get("last_word_end") != -1:
            timeline.record("", is_final=False, kind=EventKind.EOU)
        return False
    if kind != "Results":
        return False

    alternatives = payload.get("channel", {}).get("alternatives", [])
    if not alternatives:
        return False
    timeline.record(
        str(alternatives[0].get("transcript", "")),
        is_final=bool(payload.get("is_final")),
        kind=EventKind.EOU if payload.get("speech_final") else "",
    )
    return False
