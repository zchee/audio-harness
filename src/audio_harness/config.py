"""Benchmark configuration and vendor pricing.

Pricing is data, not code, because it changes without notice. Every rate below
carries the date it was checked; treat a stale ``checked`` field as a signal to
re-verify against the vendor's pricing page before quoting a cost figure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 20
"""20 ms matches the frame size used by WebRTC and most telephony stacks."""

PRICING_CHECKED = "2026-08-05"
"""Date the rates in this module were last verified against vendor pages."""


@dataclass(slots=True, frozen=True)
class SttPricing:
    """Published rates for a speech-to-text model, in USD per audio hour.

    Attributes:
        batch_per_hour: Rate for pre-recorded transcription.
        stream_per_hour: Rate for real-time streaming transcription.
        note: Caveats such as tiering or token-based billing approximations.
    """

    batch_per_hour: float | None = None
    stream_per_hour: float | None = None
    note: str = ""


@dataclass(slots=True, frozen=True)
class TtsPricing:
    """Published rates for a text-to-speech model.

    Attributes:
        per_million_chars: USD per one million input characters, when the
            vendor bills by character.
        per_audio_minute: USD per minute of generated audio, when the vendor
            bills by output duration or audio tokens.
        note: Caveats such as tiering or token-based billing approximations.
    """

    per_million_chars: float | None = None
    per_audio_minute: float | None = None
    note: str = ""


STT_PRICING: dict[str, SttPricing] = {
    "deepgram-nova3": SttPricing(
        batch_per_hour=0.26,
        stream_per_hour=0.46,
        note="pay-as-you-go; multilingual streaming is cheaper at ~$0.35/hr",
    ),
    "elevenlabs-scribe2": SttPricing(
        batch_per_hour=0.36,
        stream_per_hour=0.36,
        note="credit-based; ~$0.28/hr on annual Business commitments",
    ),
    "google-chirp3": SttPricing(
        batch_per_hour=0.96,
        stream_per_hour=0.96,
        note="v2 API standard rate; volume tiers and committed use reduce this",
    ),
    "xai-grok-stt": SttPricing(batch_per_hour=0.10, stream_per_hour=0.20),
    "assemblyai-universal35pro": SttPricing(batch_per_hour=0.21, stream_per_hour=0.45),
    "speechmatics-standard": SttPricing(
        batch_per_hour=0.24, stream_per_hour=0.24, note="20 free hours/month"
    ),
    "speechmatics-enhanced": SttPricing(
        batch_per_hour=0.43, stream_per_hour=0.43, note="20 free hours/month"
    ),
    "soniox-rt-v5": SttPricing(
        batch_per_hour=0.12,
        stream_per_hour=0.12,
        note="token-billed ($2/M audio tokens); hourly figure is an estimate",
    ),
}

TTS_PRICING: dict[str, TtsPricing] = {
    "cartesia-sonic3": TtsPricing(per_million_chars=None, note="plan-tiered"),
    "cartesia-sonic35": TtsPricing(per_million_chars=None, note="plan-tiered"),
    "deepgram-aura2": TtsPricing(per_million_chars=30.0),
    "gemini-tts": TtsPricing(
        per_audio_minute=0.015,
        note="$10/M audio tokens at 25 tokens/s, plus $0.50/M input text tokens",
    ),
}


class ConfigError(RuntimeError):
    """Raised when a benchmark configuration cannot be used as written."""


@dataclass(slots=True)
class ProviderConfig:
    """One provider entry from the benchmark configuration.

    Attributes:
        name: Registry key selecting the adapter implementation.
        modes: Transport modes to exercise (``batch``, ``stream``).
        options: Adapter-specific overrides such as voice id or region.
    """

    name: str
    modes: list[str] = field(default_factory=lambda: ["batch"])
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    """Where the harness reads evaluation material from.

    Attributes:
        manifest: JSONL file with one record per clip. Each record needs an
            ``audio`` path and, for accuracy scoring, a ``text`` reference.
        language: BCP-47 tag applied to every clip lacking its own.
        limit: Maximum clips to evaluate, or ``None`` for the whole manifest.
        prompts: Text file with one TTS prompt per line.
    """

    manifest: str | None = None
    language: str = "en-US"
    limit: int | None = None
    prompts: str | None = None


@dataclass(slots=True)
class RunConfig:
    """Execution parameters that shape how measurements are taken.

    Attributes:
        repeats: Times each clip is run, so latency percentiles have samples.
        warmup: Discarded runs per provider that absorb TLS and DNS setup.
        chunk_ms: Streaming chunk size in milliseconds.
        realtime: Whether streaming audio is paced to real time. Disabling this
            invalidates every latency figure; it exists only for smoke tests.
        provider_concurrency: Providers exercised in parallel. Clips within a
            provider always run sequentially so latency is not self-inflicted.
        vendor_concurrency: Lanes allowed to run at once against a single
            vendor account. Free plans often cap concurrent sessions at one or
            two, and entries like Speechmatics Standard and Enhanced share one
            account, so raising this trades quota errors for wall-clock time.
        timeout_s: Per-clip ceiling before a run is recorded as a failure.
        output_dir: Directory receiving result JSONL and reports.
    """

    repeats: int = 3
    warmup: int = 1
    chunk_ms: int = DEFAULT_CHUNK_MS
    realtime: bool = True
    provider_concurrency: int = 4
    vendor_concurrency: int = 1
    timeout_s: float = 300.0
    output_dir: str = "results"


@dataclass(slots=True)
class BenchmarkConfig:
    """A complete benchmark definition.

    Attributes:
        stt: STT providers to exercise.
        tts: TTS providers to exercise.
        dataset: Evaluation material.
        run: Execution parameters.
        roundtrip_stt: STT provider used to score TTS intelligibility, or
            ``None`` to skip round-trip scoring. Accepts a bare provider key
            or a mapping with ``options``.

            Turn the recognizer's text formatting **off**. With it on, a
            spoken "four hundred and twenty dollars" comes back as "$420" and
            every TTS engine scores identically badly — the metric then
            measures the recognizer's number formatting, not the voice.
    """

    stt: list[ProviderConfig] = field(default_factory=list)
    tts: list[ProviderConfig] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    run: RunConfig = field(default_factory=RunConfig)
    roundtrip_stt: ProviderConfig | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkConfig:
        """Load a benchmark configuration from a YAML file.

        Args:
            path: Path to the YAML document.

        Returns:
            The parsed configuration.

        Raises:
            ConfigError: If the file is missing, empty, or not a mapping.
        """
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raise ConfigError(f"config file is empty: {path}")
        if not isinstance(raw, dict):
            raise ConfigError(f"config root must be a mapping: {path}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> BenchmarkConfig:
        """Build a configuration from an already-parsed mapping.

        Args:
            raw: Mapping with optional ``stt``, ``tts``, ``dataset``, ``run``
                and ``roundtrip_stt`` keys.

        Returns:
            The parsed configuration.

        Raises:
            ConfigError: If a provider entry is malformed.
        """
        roundtrip = raw.get("roundtrip_stt")
        return cls(
            stt=[_provider(entry) for entry in raw.get("stt", [])],
            tts=[_provider(entry) for entry in raw.get("tts", [])],
            dataset=DatasetConfig(**(raw.get("dataset") or {})),
            run=RunConfig(**(raw.get("run") or {})),
            roundtrip_stt=None if roundtrip is None else _provider(roundtrip),
        )


def _provider(entry: Any) -> ProviderConfig:
    """Coerce a provider entry, which may be a bare name or a mapping."""
    if isinstance(entry, str):
        return ProviderConfig(name=entry)
    if not isinstance(entry, dict) or "name" not in entry:
        raise ConfigError(f"provider entry must be a name or mapping: {entry!r}")
    known = {"name", "modes", "options"}
    options = dict(entry.get("options") or {})
    options.update({k: v for k, v in entry.items() if k not in known})
    return ProviderConfig(
        name=entry["name"],
        modes=list(entry.get("modes") or ["batch"]),
        options=options,
    )


def require_env(name: str, provider: str) -> str:
    """Read a required credential from the environment.

    Args:
        name: Environment variable holding the credential.
        provider: Provider key, used to make the error actionable.

    Returns:
        The credential value.

    Raises:
        ConfigError: If the variable is unset or empty.
    """
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"{provider}: environment variable {name} is not set. "
            f"Add it to .env or export it before running the benchmark."
        )
    return value
