"""Shared contract and timing machinery for speech-to-text adapters.

Every adapter talks to its vendor over raw HTTP or WebSocket rather than
through the vendor SDK. SDKs buffer, batch and retry on their own schedule,
which is exactly the behaviour a latency benchmark must not measure.
"""

from __future__ import annotations

import abc
import time
from typing import Any, ClassVar

import httpx

from ..types import AudioClip, Mode, Partial, SttResult


class ProviderHttpError(RuntimeError):
    """An HTTP error carrying the vendor's response body.

    ``httpx.HTTPStatusError`` reports only the status line, which turns a
    precise vendor complaint like "language_code is not supported for this
    model" into an unactionable "400 Bad Request".
    """


def raise_for_status(response: httpx.Response, provider: str) -> None:
    """Raise :class:`ProviderHttpError` with the body when a request fails.

    Args:
        response: Response to inspect.
        provider: Registry key, so the message names the failing adapter.

    Raises:
        ProviderHttpError: If the status code indicates failure.
    """
    if response.status_code < 400:
        return
    body = response.text.strip().replace("\n", " ")
    raise ProviderHttpError(
        f"{provider}: HTTP {response.status_code} from {response.request.url}: "
        f"{body[:500] or '<empty body>'}"
    )


class StreamTimeline:
    """Records streaming events on a clock anchored to the first audio byte.

    All timestamps are relative to :meth:`start`, so they are directly
    comparable across providers regardless of connection setup cost.
    """

    __slots__ = ("_audio_end_s", "_start", "partials")

    def __init__(self) -> None:
        self._start: float | None = None
        self._audio_end_s: float | None = None
        self.partials: list[Partial] = []

    def start(self) -> None:
        """Anchor the clock. Call immediately before writing the first chunk."""
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        """Seconds since :meth:`start`, or ``0.0`` if the clock is unset."""
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start

    def audio_complete(self) -> None:
        """Mark the instant the last audio byte was written to the socket."""
        self._audio_end_s = self.elapsed()

    @property
    def audio_end_s(self) -> float | None:
        """Seconds at which input finished, or ``None`` while still sending."""
        return self._audio_end_s

    def record(self, text: str, *, is_final: bool) -> None:
        """Append a hypothesis event at the current instant.

        Empty hypotheses are dropped: several vendors emit blank keepalive
        results that would otherwise register as a spuriously fast first token.
        """
        if not text:
            return
        self.partials.append(Partial(t_s=self.elapsed(), text=text, is_final=is_final))

    @property
    def ttft_s(self) -> float | None:
        """Seconds to the first non-empty hypothesis of any kind."""
        return self.partials[0].t_s if self.partials else None

    @property
    def finalize_s(self) -> float | None:
        """Seconds from the last audio byte to the last final hypothesis.

        Returns ``None`` when audio never completed or no final arrived. The
        value is clamped at zero because a provider that finalizes mid-stream
        has zero perceived tail latency, not negative latency.
        """
        finals = [p for p in self.partials if p.is_final]
        if self._audio_end_s is None or not finals:
            return None
        return max(0.0, finals[-1].t_s - self._audio_end_s)

    @property
    def total_s(self) -> float:
        """Seconds from the first audio byte to the last recorded event."""
        return self.partials[-1].t_s if self.partials else self.elapsed()

    def concat_finals(self) -> str:
        """Join every final segment in arrival order.

        Correct for vendors that emit one final per utterance segment, where
        the full transcript is the concatenation of those segments.
        """
        return " ".join(p.text.strip() for p in self.partials if p.is_final).strip()

    def last_final(self) -> str:
        """Return the most recent final hypothesis.

        Correct for vendors whose finals are cumulative — each one restates the
        whole transcript so far, so concatenating them would duplicate text.
        """
        finals = [p for p in self.partials if p.is_final]
        return finals[-1].text.strip() if finals else ""


class SttProvider(abc.ABC):
    """Base class for a speech-to-text adapter.

    Subclasses declare which transport modes they implement and provide the
    corresponding coroutine. The runner never calls an unsupported mode.

    Attributes:
        key: Registry key used in configuration files.
        vendor: Account the adapter bills against. Adapters sharing a vendor
            share concurrency limits, so the runner never lets two of them
            hold sessions at once against a plan that allows only one.
        supports_batch: Whether :meth:`transcribe_batch` is implemented.
        supports_stream: Whether :meth:`transcribe_stream` is implemented.
        min_chunk_ms: Smallest audio frame the vendor accepts.
        max_chunk_ms: Largest audio frame the vendor accepts, or ``None``.
        settle_ms: Vendor-specific pause between streaming sessions, overriding
            the run default. Raise it for vendors that keep counting a session
            against the concurrency limit after the socket has closed.
    """

    key: ClassVar[str]
    vendor: ClassVar[str] = ""
    supports_batch: ClassVar[bool] = False
    supports_stream: ClassVar[bool] = False
    min_chunk_ms: ClassVar[int] = 0
    max_chunk_ms: ClassVar[int | None] = None
    settle_ms: ClassVar[int] = 0

    def effective_chunk_ms(self, requested: int) -> int:
        """Clamp a requested frame size into the vendor's accepted range.

        Vendors disagree about framing — 20 ms is the telephony norm but some
        reject anything under 50 ms. Clamping keeps a run from failing outright,
        and the value actually used is recorded on the result, because a larger
        frame adds its own latency and would otherwise silently handicap the
        provider in the comparison.

        Args:
            requested: Frame size from the run configuration, in milliseconds.

        Returns:
            The frame size this provider will actually receive.
        """
        chunk = max(requested, self.min_chunk_ms)
        if self.max_chunk_ms is not None:
            chunk = min(chunk, self.max_chunk_ms)
        return chunk

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

    async def transcribe_batch(self, clip: AudioClip) -> SttResult:
        """Transcribe a whole clip through the pre-recorded endpoint.

        Args:
            clip: Audio to transcribe.

        Returns:
            The transcription result.

        Raises:
            NotImplementedError: If the adapter has no batch endpoint.
        """
        raise NotImplementedError(f"{self.key} has no batch endpoint")

    async def transcribe_stream(
        self, clip: AudioClip, *, chunk_ms: int, realtime: bool
    ) -> SttResult:
        """Transcribe a clip through the streaming endpoint.

        Args:
            clip: Audio to transcribe.
            chunk_ms: Chunk size written to the socket, in milliseconds.
            realtime: Whether to pace chunks at playback speed. Latency
                figures are only meaningful when this is ``True``.

        Returns:
            The transcription result.

        Raises:
            NotImplementedError: If the adapter has no streaming endpoint.
        """
        raise NotImplementedError(f"{self.key} has no streaming endpoint")

    def _result(self, clip: AudioClip, mode: Mode) -> SttResult:
        """Build a result pre-populated with the fields every run shares.

        The reference transcript is carried in ``raw`` so the metrics layer can
        score a result without holding on to the corpus.
        """
        return SttResult(
            provider=self.key,
            clip_id=clip.clip_id,
            mode=mode,
            audio_s=clip.duration_s,
            raw={"reference": clip.reference or "", "language": clip.language},
        )


_REGISTRY: dict[str, type[SttProvider]] = {}


def register(cls: type[SttProvider]) -> type[SttProvider]:
    """Register an STT adapter under its ``key`` class attribute.

    Args:
        cls: Adapter class to register.

    Returns:
        The same class, so this can be used as a decorator.

    Raises:
        ValueError: If the key is already taken.
    """
    if cls.key in _REGISTRY:
        raise ValueError(f"duplicate STT provider key: {cls.key}")
    _REGISTRY[cls.key] = cls
    return cls


def create(key: str, options: dict[str, Any] | None = None) -> SttProvider:
    """Instantiate a registered STT adapter.

    Args:
        key: Registry key from the benchmark configuration.
        options: Adapter-specific overrides.

    Returns:
        The adapter instance.

    Raises:
        KeyError: If no adapter is registered under ``key``.
    """
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown STT provider {key!r}; available: {available}")
    return _REGISTRY[key](options)


def available() -> list[str]:
    """Return every registered STT provider key, sorted."""
    return sorted(_REGISTRY)
