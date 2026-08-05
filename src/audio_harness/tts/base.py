"""Shared contract for text-to-speech adapters.

The metric that matters for a voice agent is time to first audio byte: it sets
how long a user waits after the model decides what to say. Adapters therefore
consume responses incrementally and stamp the first byte as it arrives, rather
than reading a complete response and reporting total time.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar

import httpx

from ..types import Mode, TtsPrompt, TtsResult


class TtsProvider(abc.ABC):
    """Base class for a text-to-speech adapter.

    Attributes:
        key: Registry key used in configuration files.
        vendor: Account the adapter bills against. Adapters sharing a vendor
            share concurrency limits.
        supports_batch: Whether :meth:`synthesize` is implemented.
        supports_stream: Whether :meth:`synthesize_stream` is implemented.
        default_sample_rate: Output rate requested when config says nothing.
    """

    key: ClassVar[str]
    vendor: ClassVar[str] = ""
    supports_batch: ClassVar[bool] = False
    supports_stream: ClassVar[bool] = False
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
