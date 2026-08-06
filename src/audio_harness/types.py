"""Core result records shared by every STT and TTS provider adapter.

Every adapter returns one of these dataclasses so the runner, the metrics layer
and the reporter never need to know which vendor produced a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Mode(StrEnum):
    """Transport mode a provider was exercised in."""

    BATCH = "batch"
    STREAM = "stream"


class EventKind(StrEnum):
    """What a streaming transcript event *means*, beyond its text.

    Vendor final events are semantically heterogeneous: AssemblyAI's
    ``end_of_turn`` is a genuine end-of-utterance decision, while Deepgram's
    ``is_final`` merely marks a transcript segment as immutable. Collapsing
    both into one boolean makes endpointing metrics wrong by design, so the
    kind is recorded explicitly and endpointing scores only ``EOU`` events.
    """

    INTERIM = "interim"
    """A provisional hypothesis that may still change."""

    SEGMENT_FINAL = "segment_final"
    """An immutable transcript segment — a decoding decision, not a
    turn-taking one."""

    EOU = "eou"
    """The vendor's end-of-utterance decision: it believes the speaker is
    done. May carry no text (Deepgram ``UtteranceEnd``, Speechmatics
    ``EndOfUtterance``, Soniox ``<end>`` arrive as bare markers)."""


@dataclass(slots=True, frozen=True)
class AudioClip:
    """A single benchmark utterance.

    Attributes:
        clip_id: Stable identifier used to join results across providers.
        pcm: Mono 16-bit little-endian PCM samples at ``sample_rate``.
        sample_rate: Sample rate of ``pcm`` in Hz.
        duration_s: Wall-clock duration of the audio in seconds.
        reference: Ground-truth transcript, or ``None`` for latency-only clips.
        language: BCP-47 language tag (e.g. ``en-US``, ``ja-JP``).
        source_path: Original file the clip was decoded from.
        speech_end_s: Offset of the last voiced audio. Recorded clips carry
            trailing silence — a median of 0.78 s in the Pipecat corpus — and a
            provider that endpoints during it finalizes before the file ends.
            Turn latency measured from end-of-file would score that as zero, so
            this is the reference point the metric actually uses.
        pauses: Labeled mid-turn silence spans as ``(start_s, end_s)`` pairs.
            Endpointing corpora annotate hesitations the speaker talks
            through; a vendor end-of-utterance decision inside one is a false
            cutoff. Empty for corpora without pause labels.
        license: License of the clip's source material, when the corpus
            records one. Rides into results so reports keep per-source
            attribution after lanes are merged.
        gold_status: Reference-transcript trust level for curated corpora:
            ``unverified`` subtitles must not feed ranked accuracy, though
            latency needs no transcript truth and still counts. Empty for
            corpora with trusted human transcripts.
    """

    clip_id: str
    pcm: bytes
    sample_rate: int
    duration_s: float
    reference: str | None
    language: str
    source_path: str
    speech_end_s: float = 0.0
    pauses: tuple[tuple[float, float], ...] = ()
    license: str = ""
    gold_status: str = ""


@dataclass(slots=True, frozen=True)
class TtsPrompt:
    """A single text prompt to synthesize.

    Attributes:
        prompt_id: Stable identifier used to join results across providers.
        text: Text to speak.
        language: BCP-47 language tag for the prompt.
    """

    prompt_id: str
    text: str
    language: str

    @property
    def chars(self) -> int:
        """Character count, the billing unit for most TTS vendors."""
        return len(self.text)


@dataclass(slots=True)
class Partial:
    """One transcript event observed on a streaming connection.

    Attributes:
        t_s: Seconds since the first audio byte was written to the socket.
        text: Full hypothesis for the utterance at that instant. Empty only
            for bare end-of-utterance markers.
        is_final: Whether the provider marked this hypothesis as immutable.
        kind: Event semantics as an :class:`EventKind` value. Defaults from
            ``is_final`` so untouched adapters and historical results keep
            their existing behaviour: finals map to ``segment_final``,
            everything else to ``interim``.
    """

    t_s: float
    text: str
    is_final: bool
    kind: str = ""

    def __post_init__(self) -> None:
        """Derive the default kind from the finality flag."""
        if not self.kind:
            self.kind = EventKind.SEGMENT_FINAL if self.is_final else EventKind.INTERIM


@dataclass(slots=True)
class SttResult:
    """Outcome of transcribing one clip with one provider.

    Latency fields are ``None`` when the mode cannot produce them: batch runs
    have no interim hypotheses, so ``ttft_s`` and ``finalize_s`` stay unset.

    Attributes:
        provider: Registry key of the adapter (e.g. ``deepgram-nova3``).
        clip_id: Identifier of the clip that was transcribed.
        mode: Whether the clip went through the batch or streaming endpoint.
        text: Final transcript as returned by the provider, unnormalized.
        audio_s: Duration of the submitted audio.
        total_s: Wall-clock seconds from request start to final transcript.
        ttft_s: Seconds from first audio byte to first interim hypothesis.
        finalize_s: Seconds from last audio byte to the last final hypothesis.
            This is the number that governs perceived turn-taking latency.
        partials: Every interim event, ordered by arrival, for churn analysis.
        error: Failure description, or ``None`` when the run succeeded.
        raw: Provider-specific payload retained for debugging.
    """

    provider: str
    clip_id: str
    mode: Mode
    text: str = ""
    audio_s: float = 0.0
    total_s: float = 0.0
    ttft_s: float | None = None
    finalize_s: float | None = None
    partials: list[Partial] = field(default_factory=list)
    error: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the run produced a usable transcript."""
        return self.error is None

    @property
    def rtf(self) -> float | None:
        """Real-time factor: processing seconds per audio second.

        Values below 1.0 mean the provider transcribes faster than real time.
        """
        if self.audio_s <= 0:
            return None
        return self.total_s / self.audio_s


@dataclass(slots=True)
class TtsResult:
    """Outcome of synthesizing one text prompt with one provider.

    Attributes:
        provider: Registry key of the adapter (e.g. ``cartesia-sonic35``).
        prompt_id: Identifier of the prompt that was synthesized.
        mode: Whether the prompt went through the batch or streaming endpoint.
        audio: Raw synthesized audio bytes in ``encoding``.
        encoding: Container/codec of ``audio`` (``pcm_s16le``, ``mp3``, ...).
        sample_rate: Sample rate of ``audio`` in Hz.
        audio_s: Duration of the synthesized audio in seconds.
        chars: Character count of the input text, for per-character cost.
        ttfb_s: Seconds from request start to the first audio byte received.
            This is the number that governs perceived response latency.
        ttfa_s: Seconds from request start until a client that starts playback
            on the first byte would hear the first *audible* sound. Derived
            from the RMS onset of the decoded waveform and per-chunk arrival
            times, because the first byte can be a container header or leading
            silence — TTFB credits a vendor for both, TTFA for neither.
        gap_p99_s: 99th percentile of gaps between successive chunk arrivals
            within this run. Stutter: a real-time client stalls when a gap
            outruns its buffer.
        cold: Whether the run executed while the adapter's connection stack
            was cold. The warmup pass records these instead of discarding
            them; the report splits them out of the warm percentiles.
        chunk_t_s: Arrival timestamp of every audio chunk, seconds since
            request start. The raw data behind ``gap_p99_s`` and ``ttfa_s``,
            kept so both can be recomputed from saved results.
        total_s: Wall-clock seconds from request start to the last audio byte.
        error: Failure description, or ``None`` when the run succeeded.
        raw: Provider-specific payload retained for debugging.
    """

    provider: str
    prompt_id: str
    mode: Mode
    audio: bytes = b""
    encoding: str = "pcm_s16le"
    sample_rate: int = 24000
    audio_s: float = 0.0
    chars: int = 0
    ttfb_s: float | None = None
    ttfa_s: float | None = None
    gap_p99_s: float | None = None
    cold: bool = False
    chunk_t_s: list[float] = field(default_factory=list)
    total_s: float = 0.0
    error: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the run produced usable audio.

        Results loaded back from JSONL drop the audio bytes but keep the
        measured duration, so either counts as evidence that audio existed.
        """
        return self.error is None and (len(self.audio) > 0 or self.audio_s > 0)

    @property
    def rtf(self) -> float | None:
        """Real-time factor: generation seconds per audio second."""
        if self.audio_s <= 0:
            return None
        return self.total_s / self.audio_s
