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
from typing import TYPE_CHECKING

import jiwer

from .normalize import comparison_fold_for, normalizer_for, uses_character_metric
from .types import Partial, SttResult

if TYPE_CHECKING:
    from .entities import EntityClassScore


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

    Both strings are normalized for ``language`` first, then pushed through
    the language's comparison fold — an equivalence applied only for scoring,
    such as Japanese kana folding, where script choice is vendor formatting
    policy rather than recognition.

    Languages written without spaces are scored per character; every other
    language is scored per word.

    Args:
        reference: Ground-truth transcript.
        hypothesis: Provider transcript.
        language: BCP-47 language tag driving normalization and tokenization.

    Returns:
        Edit counts for the pair.
    """
    normalize = normalizer_for(language)
    fold = comparison_fold_for(language)
    ref, hyp = fold(normalize(reference)), fold(normalize(hypothesis))

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
    # Bare end-of-utterance markers carry no text; comparing prefixes against
    # them would count every marker as a rewrite, so churn skips them.
    interim = [p for p in partials if not p.is_final and p.text]
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
        entities: Per-class entity scores, populated only for results whose
            reference carries inline entity tags. Kept separate from
            ``counts`` because a digit swap inside an account number and a
            dropped filler word are the same edit but not the same failure.
        unverified: Clips whose reference is an unverified subtitle
            (``gold_status: unverified``) and therefore contributed latency
            and churn but no accuracy — ranking against caption quality
            would measure the captioner, not the recognizer.
        licenses: Source licenses seen across this lane's clips, so merged
            reports keep per-source attribution.
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
    entities: dict[str, EntityClassScore] = field(default_factory=dict)
    unverified: int = 0
    licenses: set[str] = field(default_factory=set)

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
            interim = sum(1 for p in result.partials if not p.is_final and p.text)
            summary.interim_rate.append(interim / result.audio_s)

        license_tag = result.raw.get("license")
        if isinstance(license_tag, str) and license_tag:
            summary.licenses.add(license_tag)

        # An unverified reference is a subtitle, not ground truth: scoring it
        # would rank vendors against caption quality. Latency, churn and the
        # counters above need no transcript truth, so the clip still counts
        # there — only accuracy is withheld.
        if result.raw.get("gold_status") == "unverified":
            summary.unverified += 1
            continue

        reference = result.raw.get("reference")
        if isinstance(reference, str) and reference:
            summary.counts = summary.counts + score_pair(
                reference, result.text, clip_language
            )

        annotated = result.raw.get("reference_annotated")
        if isinstance(annotated, str) and annotated:
            # Imported here, not at module top: entities.py reuses this
            # module's ErrorCounts, so a top-level import would be circular.
            from .entities import score_entities

            for label, score in score_entities(
                annotated, result.text, clip_language
            ).items():
                existing = summary.entities.get(label)
                summary.entities[label] = (
                    score if existing is None else existing + score
                )

    return list(summaries.values())


FABRICATION_MIN_RUN = 3
"""Consecutive inserted words that count as a fabricated phrase.

One or two stray insertions are ordinary recognition noise; three consecutive
words the audio never contained are invented content. The threshold follows
the hallucination lane's pre-registered definition (research-beyond-wer)."""

LOOP_MIN_REPEATS = 3
"""Times an n-gram must repeat back-to-back to count as a decoder loop."""

LOOP_MIN_TOKENS = 6
"""Minimum tokens the repeated span must cover. Keeps mundane doubled words
("very very very") out of a metric meant for runaway decoding."""

LOOP_MAX_NGRAM_WORDS = 5
"""Longest repeating unit searched in word-tokenized languages."""

LOOP_MAX_NGRAM_CHARS = 20
"""Longest repeating unit searched in character-scored languages, where one
looped phrase spans many tokens (a repeated ありがとうございました is eleven)."""


def _comparison_tokens(text: str, language: str) -> list[str]:
    """Tokenize text exactly as :func:`score_pair` would score it.

    Hallucination counters must agree with the accuracy metric about what a
    "word" is, or the same transcript would produce different insertion totals
    in different report columns.
    """
    normalize = normalizer_for(language)
    fold = comparison_fold_for(language)
    folded = fold(normalize(text))
    if uses_character_metric(language):
        return list(folded.replace(" ", ""))
    return folded.split()


def insertion_run_lengths(reference: str, hypothesis: str, language: str) -> list[int]:
    """Lengths of maximal runs of consecutive inserted tokens.

    Insertions are what hallucination is made of: text present in the
    hypothesis with no corresponding audio evidence. Run lengths — not just
    the total — matter because scattered single insertions and one four-word
    invented phrase are very different failures.

    Args:
        reference: Ground-truth transcript; empty means the clip has no
            speech, making the entire hypothesis one inserted run.
        hypothesis: Provider transcript.
        language: BCP-47 tag driving normalization and tokenization.

    Returns:
        One length per maximal insertion run, in transcript order.
    """
    ref_tokens = _comparison_tokens(reference, language)
    hyp_tokens = _comparison_tokens(hypothesis, language)
    if not hyp_tokens:
        return []
    if not ref_tokens:
        return [len(hyp_tokens)]

    output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    # jiwer merges adjacent same-type operations into a single chunk, so each
    # insert chunk is exactly one maximal run.
    return [
        chunk.hyp_end_idx - chunk.hyp_start_idx
        for chunk in output.alignments[0]
        if chunk.type == "insert"
    ]


def has_ngram_loop(
    text: str,
    language: str,
    *,
    min_repeats: int = LOOP_MIN_REPEATS,
    min_tokens: int = LOOP_MIN_TOKENS,
) -> bool:
    """Whether a transcript contains a runaway n-gram repetition.

    Autoregressive decoders stuck in a loop emit the same unit over and over
    ("thank you thank you thank you ..."). A span counts as a loop when some
    n-gram repeats at least ``min_repeats`` times back-to-back and the
    repeated span covers at least ``min_tokens`` tokens.

    Args:
        text: Provider transcript.
        language: BCP-47 tag driving normalization and tokenization.
        min_repeats: Consecutive repetitions required.
        min_tokens: Minimum tokens the whole repeated span must cover.

    Returns:
        ``True`` when any qualifying loop exists.
    """
    tokens = _comparison_tokens(text, language)
    max_n = (
        LOOP_MAX_NGRAM_CHARS
        if uses_character_metric(language)
        else LOOP_MAX_NGRAM_WORDS
    )
    for n in range(1, max_n + 1):
        if n * min_repeats > len(tokens):
            break
        for start in range(len(tokens) - n * min_repeats + 1):
            unit = tokens[start : start + n]
            repeats = 1
            while tokens[start + repeats * n : start + (repeats + 1) * n] == unit:
                repeats += 1
            if repeats >= min_repeats and repeats * n >= min_tokens:
                return True
    return False


def phantom_final_count(result: SttResult) -> int:
    """Final events carrying text on a clip that contains no speech.

    An interim hypothesis on noise is recoverable — the provider may retract
    it. A *final* is a commitment: downstream turn-taking logic acts on it.
    Only meaningful for clips whose reference is empty by construction, i.e.
    the silence and noise-only condition sets.

    Args:
        result: A streamed result from a no-speech clip.

    Returns:
        Number of final events with non-empty text, or ``0`` when the clip
        has a reference and the notion does not apply.
    """
    reference = result.raw.get("reference")
    if isinstance(reference, str) and reference.strip():
        return 0
    return sum(1 for p in result.partials if p.is_final and p.text.strip())


@dataclass(slots=True)
class HallucinationSummary:
    """Hallucination behaviour of one provider on one condition set.

    Attributes:
        provider: Registry key of the adapter.
        mode: Transport mode the runs used.
        language: BCP-47 tag these runs were scored under.
        condition: Synthetic condition (``silence``, ``noise``,
            ``trailing_silence``, ``low_snr``), or ``speech``/``no_speech``
            for clips outside the synthetic lane.
        clips: Number of clips attempted.
        failures: Number of clips that errored.
        fabricated_clips: Clips with an insertion run of at least
            :data:`FABRICATION_MIN_RUN` words.
        inserted_words: Total inserted tokens across scored clips.
        phantom_finals: Final events with text across no-speech clips.
        phantom_final_clips: No-speech clips with at least one such final.
        looped_clips: Clips whose transcript contains an n-gram loop.
        audio_s: Total audio submitted, the denominator for per-minute rates.
    """

    provider: str
    mode: str
    language: str
    condition: str
    clips: int = 0
    failures: int = 0
    fabricated_clips: int = 0
    inserted_words: int = 0
    phantom_finals: int = 0
    phantom_final_clips: int = 0
    looped_clips: int = 0
    audio_s: float = 0.0

    @property
    def scored(self) -> int:
        """Clips that produced a scoreable transcript."""
        return self.clips - self.failures

    @property
    def fabrication_rate(self) -> float | None:
        """Fraction of scored clips with a fabricated phrase."""
        if self.scored == 0:
            return None
        return self.fabricated_clips / self.scored

    @property
    def inserted_words_per_min(self) -> float | None:
        """Inserted tokens per minute of submitted audio."""
        if self.audio_s <= 0:
            return None
        return self.inserted_words / (self.audio_s / 60.0)

    @property
    def phantom_final_rate(self) -> float | None:
        """Fraction of scored no-speech clips that produced a final."""
        if self.scored == 0:
            return None
        return self.phantom_final_clips / self.scored

    @property
    def loop_rate(self) -> float | None:
        """Fraction of scored clips whose transcript loops."""
        if self.scored == 0:
            return None
        return self.looped_clips / self.scored


def summarize_hallucination(
    results: list[SttResult], language: str
) -> list[HallucinationSummary]:
    """Aggregate hallucination counters per provider, mode, language and
    condition.

    Designed to run over saved results JSONL: every input it reads —
    reference, transcript, partials, audio duration, clip id — survives
    :func:`runner.write_stt_results`, so a normalization or threshold change
    never requires paying for the audio again.

    Args:
        results: Per-clip results, typically from the hallucination lane.
        language: Fallback BCP-47 tag for results that recorded none.

    Returns:
        Summaries keyed by provider, mode, language and condition.
    """
    # Imported here, not at module top: synthetic.py sits atop the dataset
    # loaders, and pulling that stack into every metrics import is needless.
    from .synthetic import condition_of

    summaries: dict[tuple[str, str, str, str], HallucinationSummary] = {}

    for result in results:
        recorded = result.raw.get("language")
        clip_language = recorded if isinstance(recorded, str) and recorded else language
        raw_reference = result.raw.get("reference")
        reference = raw_reference if isinstance(raw_reference, str) else ""
        condition = condition_of(result.clip_id) or (
            "speech" if reference.strip() else "no_speech"
        )

        key = (result.provider, str(result.mode), clip_language, condition)
        summary = summaries.setdefault(
            key,
            HallucinationSummary(
                provider=result.provider,
                mode=str(result.mode),
                language=clip_language,
                condition=condition,
            ),
        )
        summary.clips += 1
        if not result.ok:
            summary.failures += 1
            continue

        summary.audio_s += result.audio_s
        runs = insertion_run_lengths(reference, result.text, clip_language)
        summary.inserted_words += sum(runs)
        if any(run >= FABRICATION_MIN_RUN for run in runs):
            summary.fabricated_clips += 1
        if has_ngram_loop(result.text, clip_language):
            summary.looped_clips += 1
        if not reference.strip():
            phantom = phantom_final_count(result)
            summary.phantom_finals += phantom
            if phantom:
                summary.phantom_final_clips += 1

    return list(summaries.values())
