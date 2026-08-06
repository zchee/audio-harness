"""Shared contract for text-to-speech adapters.

The metric that matters for a voice agent is time to first audio byte: it sets
how long a user waits after the model decides what to say. Adapters therefore
consume responses incrementally and stamp the first byte as it arrives, rather
than reading a complete response and reporting total time.
"""

from __future__ import annotations

import abc
import asyncio
import re
import time
from collections.abc import AsyncIterator
from itertools import pairwise
from typing import Any, ClassVar

import httpx

from ..audio import (
    BYTES_PER_SAMPLE,
    detect_speech_onset_s,
    pcm16_to_float,
    wav_data_offset,
)
from ..metrics import percentile
from ..types import Mode, TtsPrompt, TtsResult


class TtsProvider(abc.ABC):
    """Base class for a text-to-speech adapter.

    Attributes:
        key: Registry key used in configuration files.
        vendor: Account the adapter bills against. Adapters sharing a vendor
            share concurrency limits.
        family: Model lineage used by judge/candidate coupling rules. Defaults
            to ``vendor``; kept separate because lineage and billing diverge
            (an OpenAI-lineage voice served by a reseller must still never be
            scored by an OpenAI-lineage recognizer).
        supports_batch: Whether :meth:`synthesize` is implemented.
        supports_stream: Whether :meth:`synthesize_stream` is implemented.
        supports_input_streaming: Whether :meth:`synthesize_incremental` is
            implemented — true only when the vendor's wire protocol accepts
            text appended to an open synthesis, never simulated client-side.
        default_sample_rate: Output rate requested when config says nothing.
    """

    key: ClassVar[str]
    vendor: ClassVar[str] = ""
    family: ClassVar[str] = ""
    supports_batch: ClassVar[bool] = False
    supports_stream: ClassVar[bool] = False
    supports_input_streaming: ClassVar[bool] = False
    default_sample_rate: ClassVar[int] = 24000

    @property
    def billing_group(self) -> str:
        """Key used to serialize runs that share one vendor account."""
        return self.vendor or self.key

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        """Store adapter options and lazily prepare an HTTP client.

        Args:
            options: Adapter-specific overrides from the benchmark config.
        """
        self.options: dict[str, Any] = dict(options or {})
        self._http: httpx.AsyncClient | None = None

    @property
    def http(self) -> httpx.AsyncClient:
        """A connection-pooled HTTP client shared across this adapter's runs."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=15.0))
        return self._http

    async def aclose(self) -> None:
        """Release the HTTP connection pool."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    @property
    def sample_rate(self) -> int:
        """Output sample rate requested from the vendor."""
        return int(self.options.get("sample_rate", self.default_sample_rate))

    async def synthesize(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize a prompt and return the complete audio.

        Args:
            prompt: Text to speak.

        Returns:
            The synthesis result.

        Raises:
            NotImplementedError: If the adapter has no batch endpoint.
        """
        raise NotImplementedError(f"{self.key} has no batch endpoint")

    async def synthesize_stream(self, prompt: TtsPrompt) -> TtsResult:
        """Synthesize a prompt incrementally, measuring time to first byte.

        Args:
            prompt: Text to speak.

        Returns:
            The synthesis result.

        Raises:
            NotImplementedError: If the adapter has no streaming endpoint.
        """
        raise NotImplementedError(f"{self.key} has no streaming endpoint")

    async def synthesize_incremental(
        self, prompt: TtsPrompt, *, token_rate: float
    ) -> TtsResult:
        """Synthesize while the text itself arrives at LLM-token cadence.

        A voice agent never holds a finished sentence: text trickles out of a
        language model while synthesis is already running. Adapters whose wire
        protocol accepts appended text implement this; the rest keep this
        default and the runner falls back to whole-prompt streaming, recording
        that it did.

        Args:
            prompt: Text to speak.
            token_rate: Simulated LLM decode speed in tokens per second; a
                token is approximated as :data:`TOKEN_CHARS` characters.

        Returns:
            The synthesis result.

        Raises:
            NotImplementedError: If the protocol cannot accept streamed text.
        """
        raise NotImplementedError(f"{self.key} does not accept streamed input text")

    def _result(self, prompt: TtsPrompt, mode: Mode) -> TtsResult:
        """Build a result pre-populated with the fields every run shares."""
        return TtsResult(
            provider=self.key,
            prompt_id=prompt.prompt_id,
            mode=mode,
            chars=prompt.chars,
            sample_rate=self.sample_rate,
            raw={"text": prompt.text},
        )


class ChunkTimeline:
    """Audio chunks and their arrival times for one streaming synthesis.

    Owns the stopwatch every streaming adapter needs: construction starts the
    clock (so it must happen before the connection is opened — the handshake
    is part of the latency), :meth:`add` stamps each received chunk, and
    :func:`stamp_stream_timing` turns the record into the derived latency
    facts. Centralizing this keeps the adapters identical in how they measure.

    Attributes:
        chunks: Received audio chunks, in arrival order.
        t_s: Arrival timestamp of each chunk, seconds since construction.
    """

    __slots__ = ("_started", "chunks", "t_s")

    def __init__(self) -> None:
        """Start the clock."""
        self._started = time.perf_counter()
        self.chunks: list[bytes] = []
        self.t_s: list[float] = []

    def add(self, chunk: bytes) -> None:
        """Record one received audio chunk, stamping its arrival time."""
        if not chunk:
            return
        self.t_s.append(time.perf_counter() - self._started)
        self.chunks.append(chunk)

    @property
    def audio(self) -> bytes:
        """All received audio, concatenated in arrival order."""
        return b"".join(self.chunks)

    def elapsed_s(self) -> float:
        """Seconds since the clock started."""
        return time.perf_counter() - self._started


def stamp_stream_timing(result: TtsResult, timeline: ChunkTimeline) -> None:
    """Write TTFB, chunk arrivals, stutter and audible-onset latency.

    Call after the adapter has attached the decoded audio to ``result``, so
    the RMS onset can be located in the final waveform.

    Args:
        result: Streamed result to stamp, mutated in place.
        timeline: Chunk record for the run.
    """
    result.total_s = timeline.elapsed_s()
    result.chunk_t_s = list(timeline.t_s)
    if not timeline.t_s:
        return
    result.ttfb_s = timeline.t_s[0]
    gaps = [later - earlier for earlier, later in pairwise(timeline.t_s)]
    result.gap_p99_s = percentile(gaps, 99)
    result.ttfa_s = _audible_onset_latency(result, timeline)


def _audible_onset_latency(result: TtsResult, timeline: ChunkTimeline) -> float | None:
    """Locate the audible onset in the decoded PCM and translate it to wall time.

    Models a client that begins playback the moment the first byte arrives and
    never pauses voluntarily: audio at a given offset cannot play before its
    chunk has arrived, nor before the audio preceding it has played out. The
    first audible instant is therefore the maximum, over every chunk up to the
    onset, of that chunk's arrival time plus the remaining playback distance
    to the onset. For a payload whose leading silence and first voiced sample
    arrive together, that reduces to TTFB plus the silence duration.

    Returns:
        The onset latency in seconds, or ``None`` when the payload is not PCM
        (a compressed container cannot be mapped byte-to-time) or when nothing
        in it is audible.
    """
    if not result.encoding.startswith("pcm") or result.sample_rate <= 0:
        return None
    payload = result.audio
    if not payload:
        return None

    header = wav_data_offset(payload)
    onset_s = detect_speech_onset_s(
        pcm16_to_float(payload[header:]), result.sample_rate
    )
    if onset_s is None:
        return None

    bytes_per_s = result.sample_rate * BYTES_PER_SAMPLE
    onset_byte = header + onset_s * bytes_per_s
    first_audible = 0.0
    start = 0
    for arrival, chunk in zip(timeline.t_s, timeline.chunks, strict=True):
        if start > onset_byte:
            break
        first_audible = max(first_audible, arrival + (onset_byte - start) / bytes_per_s)
        start += len(chunk)
    return first_audible


TOKEN_CHARS = 4
"""Approximate characters per LLM token — the usual BPE average."""


def token_pieces(text: str) -> list[str]:
    """Split text into word-sized pieces whose concatenation is the text.

    Language models emit sub-word tokens, but TTS vendors document sending
    word or phrase fragments, so words are the simulation unit; the cadence
    (not the piece size) carries the token-rate realism.
    """
    pieces = [match.group(0) for match in re.finditer(r"\S+\s*", text)]
    return pieces or [text]


async def pace_tokens(pieces: list[str], token_rate: float) -> AsyncIterator[str]:
    """Yield text pieces at a simulated LLM decode cadence.

    The first piece is yielded immediately — the upstream model's own latency
    is not the TTS vendor's to answer for. Deadlines are computed from one
    start instant, mirroring :func:`audio_harness.audio.pace_chunks`, so
    scheduler jitter cannot drift over a long prompt.

    Args:
        pieces: Text fragments, usually from :func:`token_pieces`.
        token_rate: Simulated decode speed in tokens per second; each piece
            costs ``len(piece) / TOKEN_CHARS`` tokens of budget.

    Yields:
        The pieces, in order, at cadence.
    """
    start = time.perf_counter()
    budget = 0.0
    for piece in pieces:
        delay = start + budget - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        yield piece
        budget += len(piece) / (TOKEN_CHARS * max(token_rate, 0.1))


_REGISTRY: dict[str, type[TtsProvider]] = {}


def register(cls: type[TtsProvider]) -> type[TtsProvider]:
    """Register a TTS adapter under its ``key`` class attribute.

    Args:
        cls: Adapter class to register.

    Returns:
        The same class, so this can be used as a decorator.

    Raises:
        ValueError: If the key is already taken.
    """
    if cls.key in _REGISTRY:
        raise ValueError(f"duplicate TTS provider key: {cls.key}")
    _REGISTRY[cls.key] = cls
    return cls


def create(key: str, options: dict[str, Any] | None = None) -> TtsProvider:
    """Instantiate a registered TTS adapter.

    Args:
        key: Registry key from the benchmark configuration.
        options: Adapter-specific overrides.

    Returns:
        The adapter instance.

    Raises:
        KeyError: If no adapter is registered under ``key``.
    """
    if key not in _REGISTRY:
        available_keys = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown TTS provider {key!r}; available: {available_keys}")
    return _REGISTRY[key](options)


def available() -> list[str]:
    """Return every registered TTS provider key, sorted."""
    return sorted(_REGISTRY)


def family_of(key: str) -> str:
    """Return the model family for a TTS provider key.

    See :func:`audio_harness.stt.base.family_of`; the same coupling rule reads
    both sides. An unregistered key forms its own family so historical results
    keep rendering.
    """
    cls = _REGISTRY.get(key)
    if cls is None:
        return key
    return cls.family or cls.vendor or cls.key
