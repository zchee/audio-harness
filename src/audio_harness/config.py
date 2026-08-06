"""Benchmark configuration and vendor pricing.

Pricing is data, not code, because it changes without notice. Every rate below
carries the date it was checked; treat a stale ``checked`` field as a signal to
re-verify against the vendor's pricing page before quoting a cost figure.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field, fields
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
class SourceConfig:
    """One corpus in one language.

    A multilingual benchmark is several of these. Language lives here rather
    than on the run because it decides normalization and whether accuracy is
    scored per word or per character, which differ within a single run.

    Attributes:
        parquet: Parquet file with embedded audio.
        manifest: JSONL manifest, as an alternative to ``parquet``.
        language: BCP-47 tag for every clip from this source.
        id_column: Parquet column holding the clip identifier.
        audio_column: Parquet column holding the audio.
        text_column: Parquet column holding the reference transcript.
        limit: Maximum clips from this source. For synthetic sources this is
            the clip count of the condition set.
        sample_seed: Seed for a reproducible random subset. Synthetic sources
            reuse it to pin noise selection, offsets and pairing.
        synthetic: Generate clips instead of reading a corpus: ``silence``,
            ``noise``, ``trailing_silence`` or ``low_snr`` (see
            ``synthetic.py``). The derived kinds still read base utterances
            from ``parquet``/``manifest``; the generated kinds need neither.
        noise_dir: Directory of noise recordings (MUSAN, CC BY 4.0) for the
            ``noise`` and ``low_snr`` kinds.
        duration_s: Length of generated ``silence``/``noise`` clips.
        trailing_silence_s: Silence appended by ``trailing_silence``.
        snr_db: Active-speech SNR targeted by ``low_snr``; negative values
            put the speech below the noise.
        silence_spans_column: Parquet column of labeled silence spans
            (``{start, end}`` structs, eot-bench schema). The final span is
            the true end of the turn; earlier spans become the clip's
            mid-turn ``pauses`` and ``speech_end_s`` comes from the label
            instead of energy detection.
        words_column: Parquet column of word timings; joined into a
            reference transcript when the corpus has no plain-text column.
    """

    parquet: str | None = None
    manifest: str | None = None
    language: str = "en-US"
    id_column: str = "sample_id"
    audio_column: str = "audio"
    text_column: str = "transcription"
    limit: int | None = None
    sample_seed: int | None = None
    synthetic: str | None = None
    noise_dir: str | None = None
    duration_s: float | None = None
    trailing_silence_s: float | None = None
    snr_db: float | None = None
    silence_spans_column: str | None = None
    words_column: str | None = None


@dataclass(slots=True)
class DatasetConfig:
    """Where the harness reads evaluation material from.

    Either name a single corpus with the fields below, or list several under
    ``sources`` to benchmark more than one language in one run.

    Exactly one of ``manifest`` or ``parquet`` supplies the STT clips.

    Attributes:
        manifest: JSONL file with one record per clip. Each record needs an
            ``audio`` path and, for accuracy scoring, a ``text`` reference.
        parquet: Parquet file with embedded audio, as distributed by Hugging
            Face audio datasets. Column names are configurable below.
        id_column: Parquet column holding the clip identifier.
        audio_column: Parquet column holding the audio. Either raw encoded
            bytes or a ``{bytes, path}`` struct, which is how the Hugging Face
            ``Audio`` feature serializes.
        text_column: Parquet column holding the reference transcript.
        language: BCP-47 tag applied to every clip lacking its own.
        limit: Maximum clips to evaluate, or ``None`` for the whole corpus.
        sample_seed: When set alongside ``limit``, take a random sample with
            this seed instead of the first N rows. Corpora are often ordered by
            length or source, so the head is not a representative subset.
        prompts: Text file with one TTS prompt per line.
        silence_spans_column: Parquet column of labeled silence spans; acts
            as the shared default for every source (see ``SourceConfig``).
        words_column: Parquet column of word timings; shared default for
            every source.
    """

    manifest: str | None = None
    parquet: str | None = None
    id_column: str = "sample_id"
    audio_column: str = "audio"
    text_column: str = "transcription"
    language: str = "en-US"
    limit: int | None = None
    sample_seed: int | None = None
    prompts: str | None = None
    silence_spans_column: str | None = None
    words_column: str | None = None
    sources: list[SourceConfig] = field(default_factory=list)

    def resolved_sources(self) -> list[SourceConfig]:
        """Return the corpora to load, however the config expressed them.

        An explicit ``sources`` list wins; otherwise the top-level fields are
        treated as a single source, so single-language configs stay unchanged.
        """
        if self.sources:
            return self.sources
        return [
            SourceConfig(
                parquet=self.parquet,
                manifest=self.manifest,
                language=self.language,
                id_column=self.id_column,
                audio_column=self.audio_column,
                text_column=self.text_column,
                limit=self.limit,
                sample_seed=self.sample_seed,
                silence_spans_column=self.silence_spans_column,
                words_column=self.words_column,
            )
        ]


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
        transient_retries: Extra attempts for capacity refusals — concurrency
            caps and rate limits. These say nothing about a provider's quality,
            so retrying them keeps the harness' own scheduling out of the
            failure column. Genuine errors are never retried.
        retry_backoff_s: Base delay before a retry; doubles per attempt.
        settle_ms: Pause between streaming clips in one lane, giving the vendor
            time to release the finished session before the next one opens.
        tts_incremental_text: Feed TTS prompts word-by-word at LLM cadence on
            streaming lanes whose wire protocol accepts appended text. Lanes
            without protocol support fall back to whole-prompt streaming and
            record that they did, so the lanes stay distinguishable.
        tts_token_rate: Simulated LLM decode speed for the incremental lane,
            in tokens per second; a token is approximated as four characters.
        tts_load_concurrency: When above one, repeat the streaming TTS prompt
            set with this many syntheses in flight at once against each
            adapter. Those runs are tagged with the load factor and reported
            as their own lane, so queueing under load is visible without
            polluting the sequential percentiles.
        output_dir: Directory receiving result JSONL and reports.
    """

    repeats: int = 3
    warmup: int = 1
    chunk_ms: int = DEFAULT_CHUNK_MS
    realtime: bool = True
    provider_concurrency: int = 4
    vendor_concurrency: int = 1
    timeout_s: float = 300.0
    transient_retries: int = 3
    retry_backoff_s: float = 3.0
    settle_ms: int = 250
    tts_incremental_text: bool = False
    tts_token_rate: float = 40.0
    tts_load_concurrency: int = 1
    output_dir: str = "results"


@dataclass(slots=True)
class BenchmarkConfig:
    """A complete benchmark definition.

    Attributes:
        stt: STT providers to exercise.
        tts: TTS providers to exercise.
        dataset: Evaluation material.
        run: Execution parameters.
        roundtrip_stt: STT judges that score TTS intelligibility, or an empty
            list to skip round-trip scoring. The YAML accepts a list of
            judges, or — deprecated — a single bare key or mapping, which is
            parsed as a one-judge list. Judges should span vendor families:
            a lane is ranked only by judges outside its own family, so a
            same-family-only judge set leaves that lane unranked.

            Turn each recognizer's text formatting **off**. With it on, a
            spoken "four hundred and twenty dollars" comes back as "$420" and
            every TTS engine scores identically badly — the metric then
            measures the recognizer's number formatting, not the voice.
    """

    stt: list[ProviderConfig] = field(default_factory=list)
    tts: list[ProviderConfig] = field(default_factory=list)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    run: RunConfig = field(default_factory=RunConfig)
    roundtrip_stt: list[ProviderConfig] = field(default_factory=list)

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
        return cls(
            stt=[_provider(entry) for entry in raw.get("stt", [])],
            tts=[_provider(entry) for entry in raw.get("tts", [])],
            dataset=_dataset(raw.get("dataset") or {}),
            run=RunConfig(**(raw.get("run") or {})),
            roundtrip_stt=_roundtrip(raw.get("roundtrip_stt")),
        )


def _dataset(raw: dict[str, Any]) -> DatasetConfig:
    """Build a dataset config, expanding a ``sources`` list if present.

    Fields set alongside ``sources`` act as defaults for every entry, so a
    twelve-language config states the shared column names and limit once
    instead of repeating them per language.

    Raises:
        ConfigError: If a source entry is not a mapping or names no corpus.
    """
    entries = raw.get("sources")
    if not entries:
        return DatasetConfig(**raw)

    shared = {k: v for k, v in raw.items() if k != "sources"}
    known = {f.name for f in fields(SourceConfig)}
    defaults = {k: v for k, v in shared.items() if k in known}

    sources: list[SourceConfig] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ConfigError(f"dataset.sources entry must be a mapping: {entry!r}")
        merged = {**defaults, **entry}
        unknown = sorted(set(merged) - known)
        if unknown:
            raise ConfigError(
                f"dataset.sources entry has unknown key(s): {', '.join(unknown)}"
            )
        if (
            not merged.get("parquet")
            and not merged.get("manifest")
            and not merged.get("synthetic")
        ):
            raise ConfigError(
                f"dataset.sources entry needs a parquet or manifest corpus, "
                f"or a synthetic kind: {entry!r}"
            )
        sources.append(SourceConfig(**merged))

    return DatasetConfig(**shared, sources=sources)


def _roundtrip(raw: Any) -> list[ProviderConfig]:
    """Parse the round-trip judge list, still accepting the legacy scalar.

    Existing configs name a single judge; they keep loading as a one-judge
    list so saved benchmarks stay runnable, but a single judge cannot satisfy
    the cross-family rule for its own family's lanes — hence the deprecation.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [_provider(entry) for entry in raw]
    warnings.warn(
        "roundtrip_stt: the single-judge scalar form is deprecated; list "
        "judges spanning vendor families so every TTS lane gets a ranked "
        "cross-family score",
        DeprecationWarning,
        stacklevel=2,
    )
    return [_provider(raw)]


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
