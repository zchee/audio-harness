"""Accuracy, latency and stability metrics.

Accuracy is aggregated at corpus level — total edits divided by total reference
length — rather than by averaging per-utterance rates. Averaging per-utterance
rates lets a three-word clip outweigh a thirty-second one and is the most common
way published STT comparisons go wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise

import jiwer

from .normalize import normalizer_for, uses_character_metric
from .types import Partial, SttResult


@dataclass(slots=True, frozen=True)
class ErrorCounts:
    """Edit operations between one reference and one hypothesis.

    Attributes:
        substitutions: Tokens replaced by a different token.
        deletions: Reference tokens absent from the hypothesis.
        insertions: Hypothesis tokens absent from the reference.
        reference_length: Token count of the reference.
    """

    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        """Total edit operations."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Error rate for this pair, or ``inf`` for an empty reference."""
        if self.reference_length == 0:
            return math.inf if self.errors else 0.0
        return self.errors / self.reference_length

    def __add__(self, other: ErrorCounts) -> ErrorCounts:
        """Accumulate counts so a corpus rate can be computed from the sum."""
        return ErrorCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_length=self.reference_length + other.reference_length,
        )


ZERO_COUNTS = ErrorCounts(0, 0, 0, 0)


def score_pair(reference: str, hypothesis: str, language: str) -> ErrorCounts:
    """Count edit operations between a reference and a hypothesis.

    Both strings are normalized for ``language`` first. Languages written
    without spaces are scored per character; every other language is scored
    per word.

    Args:
        reference: Ground-truth transcript.
        hypothesis: Provider transcript.
        language: BCP-47 language tag driving normalization and tokenization.

    Returns:
        Edit counts for the pair.
    """
    normalize = normalizer_for(language)
    ref, hyp = normalize(reference), normalize(hypothesis)

    if uses_character_metric(language):
        ref_tokens = " ".join(ref)
        hyp_tokens = " ".join(hyp)
    else:
        ref_tokens, hyp_tokens = ref, hyp

    if not ref_tokens.strip():
        return ErrorCounts(0, 0, len(hyp_tokens.split()), 0)

    output = jiwer.process_words(ref_tokens, hyp_tokens)
    return ErrorCounts(
        substitutions=output.substitutions,
        deletions=output.deletions,
        insertions=output.insertions,
        reference_length=output.substitutions + output.deletions + output.hits,
    )


def percentile(values: list[float], q: float) -> float | None:
    """Return the ``q``-th percentile using linear interpolation.

    Args:
        values: Samples; may be unsorted. Empty input yields ``None``.
        q: Percentile in [0, 100].

    Returns:
        The interpolated percentile, or ``None`` for empty input.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def partial_instability(partials: list[Partial]) -> float | None:
    """Fraction of interim hypotheses that rewrote already-displayed text.

    A partial is *stable* when it merely extends its predecessor. A partial that
    contradicts the prefix forces a UI to erase text the user already read, and
    in a voice agent it can retract a phrase the turn-taking logic already
    acted on. Lower is better; ``0.0`` means the hypothesis only ever grew.

    Args:
        partials: Interim events in arrival order.

    Returns:
        Instability in [0, 1], or ``None`` when there are fewer than two
        interim events to compare.
    """
    interim = [p for p in partials if not p.is_final]
    if len(interim) < 2:
        return None
    rewrites = sum(
        1
        for previous, current in pairwise(interim)
        if not current.text.startswith(previous.text)
    )
    return rewrites / (len(interim) - 1)


@dataclass(slots=True)
class ProviderSummary:
    """Aggregated benchmark results for one provider in one mode.

    Attributes:
        provider: Registry key of the adapter.
        mode: Transport mode the runs used.
        language: BCP-47 tag these runs were scored under.
        clips: Number of clips attempted.
        failures: Number of clips that errored.
        counts: Summed edit operations across successful clips.
        character_metric: Whether ``error_rate`` is a CER rather than a WER.
        ttft_s: Interim-hypothesis latencies across successful clips.
        finalize_s: End-of-audio to final-transcript latencies.
        rtf: Real-time factors.
        instability: Per-clip partial churn values.
        interim_rate: Interim hypotheses emitted per second of audio. Churn is
            a ratio, so it is only interpretable next to this: three rewrites
            out of seven updates and zero out of forty-four are very different
            behaviours that a percentage alone flattens.
        audio_s: Total audio submitted, for cost estimation.
    """

    provider: str
    mode: str
    language: str = ""
    clips: int = 0
    failures: int = 0
    counts: ErrorCounts = ZERO_COUNTS
    character_metric: bool = False
    ttft_s: list[float] = field(default_factory=list)
    finalize_s: list[float] = field(default_factory=list)
    rtf: list[float] = field(default_factory=list)
    instability: list[float] = field(default_factory=list)
    interim_rate: list[float] = field(default_factory=list)
    audio_s: float = 0.0
    chunk_ms: int | None = None

    @property
    def error_rate(self) -> float | None:
        """Corpus-level error rate, or ``None`` when no reference was scored."""
        if self.counts.reference_length == 0:
            return None
        return self.counts.rate

    @property
    def metric_name(self) -> str:
        """Human-readable name of the accuracy metric in use."""
        return "CER" if self.character_metric else "WER"


def summarize(results: list[SttResult], language: str) -> list[ProviderSummary]:
    """Aggregate per-clip STT results into one summary per provider, mode and
    language.

    Language is part of the key, never averaged over. A 2% error rate on French
    and 8% on Vietnamese say something specific about a provider; pooling them
    into 5% describes no system anyone can deploy, and the mix would shift
    whenever the corpus mix did. Each result is scored with its own language's
    normalizer, so a run may legitimately mix per-word and per-character rates.

    Args:
        results: Every run to aggregate; failures are counted, not scored.
        language: Fallback BCP-47 tag for results that recorded none.

    Returns:
        Summaries keyed by provider, mode and language.
    """
    summaries: dict[tuple[str, str, str], ProviderSummary] = {}

    for result in results:
        recorded = result.raw.get("language")
        clip_language = recorded if isinstance(recorded, str) and recorded else language
        key = (result.provider, str(result.mode), clip_language)
        summary = summaries.setdefault(
            key,
            ProviderSummary(
                provider=result.provider,
                mode=str(result.mode),
                language=clip_language,
                character_metric=uses_character_metric(clip_language),
            ),
        )
        summary.clips += 1
        if not result.ok:
            summary.failures += 1
            continue

        summary.audio_s += result.audio_s
        chunk_ms = result.raw.get("chunk_ms")
        if isinstance(chunk_ms, int):
            summary.chunk_ms = chunk_ms
        if result.ttft_s is not None:
            summary.ttft_s.append(result.ttft_s)
        if result.finalize_s is not None:
            summary.finalize_s.append(result.finalize_s)
        rtf = result.rtf
        if rtf is not None:
            summary.rtf.append(rtf)
        churn = partial_instability(result.partials)
        if churn is not None:
            summary.instability.append(churn)
        if result.audio_s > 0 and result.partials:
            interim = sum(1 for p in result.partials if not p.is_final)
            summary.interim_rate.append(interim / result.audio_s)

        reference = result.raw.get("reference")
        if isinstance(reference, str) and reference:
            summary.counts = summary.counts + score_pair(
                reference, result.text, clip_language
            )

    return list(summaries.values())
