"""Entity-level accuracy scoring.

A transcript can be 95% correct and still useless to an interview agent that
recorded the wrong account number. Corpus WER treats every token alike, so a
digit swap inside an ID costs the same as a dropped filler word; this module
scores the tokens that carry the interview's facts — numbers, dates, currency
amounts, alphanumeric IDs and proper names — separately, per class.

References are annotated with inline tags in the dataset metadata::

    my flight is <id>AB 1234</id> on <date>March third</date>

Both sides pass through the shared per-language normalization first, so a
reference "<number>four hundred and twenty</number>" and a hypothesis "420"
agree before any entity is judged. Scoring then aligns the full transcripts
and reads errors off the entity token ranges — an entity is judged in its
sentence context, not by substring search, so a number appearing twice cannot
vouch for the occurrence that was misheard.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import jiwer

from .metrics import ZERO_COUNTS, ErrorCounts
from .normalize import comparison_fold_for, normalizer_for, uses_character_metric

ENTITY_CLASSES = ("number", "date", "currency", "id", "name")
"""Tag vocabulary: cardinal numbers, dates, currency amounts, alphanumeric
IDs, and proper names — the classes whose corruption breaks an interview."""

_TAG_PATTERN = re.compile(rf"<({'|'.join(ENTITY_CLASSES)})>(.*?)</\1>")


@dataclass(slots=True, frozen=True)
class EntityClassScore:
    """Accumulated scores for one entity class.

    Attributes:
        counts: Edit operations over this class's reference tokens, so a
            corpus-level entity-WER can be computed from the sum.
        occurrences: Tagged spans seen.
        exact_matches: Spans reproduced with zero edits.
    """

    counts: ErrorCounts = ZERO_COUNTS
    occurrences: int = 0
    exact_matches: int = 0

    @property
    def error_rate(self) -> float | None:
        """Corpus-level entity-WER, or ``None`` without reference tokens."""
        if self.counts.reference_length == 0:
            return None
        return self.counts.rate

    @property
    def exact_match_rate(self) -> float | None:
        """Fraction of spans reproduced verbatim after normalization.

        Reported alongside entity-WER because they answer different
        questions: a one-digit error in every phone number is a low WER and a
        0% exact match, and the agent dials none of them.
        """
        if self.occurrences == 0:
            return None
        return self.exact_matches / self.occurrences

    def __add__(self, other: EntityClassScore) -> EntityClassScore:
        """Accumulate scores so corpus rates can be computed from the sum."""
        return EntityClassScore(
            counts=self.counts + other.counts,
            occurrences=self.occurrences + other.occurrences,
            exact_matches=self.exact_matches + other.exact_matches,
        )


def parse_annotated(reference: str) -> list[tuple[str, str | None]]:
    """Split an annotated reference into (text, entity class) segments.

    Untagged stretches carry ``None``. Unknown tags are left in the text
    verbatim rather than silently swallowed, so an annotation typo surfaces
    in the transcript diff instead of deleting a span from scoring.

    Args:
        reference: Reference transcript with inline entity tags.

    Returns:
        Ordered segments covering the whole string.
    """
    segments: list[tuple[str, str | None]] = []
    position = 0
    for match in _TAG_PATTERN.finditer(reference):
        if match.start() > position:
            segments.append((reference[position : match.start()], None))
        segments.append((match.group(2), match.group(1)))
        position = match.end()
    if position < len(reference):
        segments.append((reference[position:], None))
    return segments


def strip_tags(reference: str) -> str:
    """Return the plain transcript with entity tags removed.

    The plain form is what overall WER scores and what reports display; the
    tags exist only for this module.
    """
    return "".join(text for text, _ in parse_annotated(reference))


def _tokenize(text: str, language: str) -> list[str]:
    """Normalize and tokenize the way the overall metric for ``language`` does.

    Word languages split on spaces; scriptio-continua languages score per
    character, so their tokens are characters. Reusing the shared
    normalization (including the comparison fold) is the point: entity scores
    must agree with the headline metric about what counts as a difference.
    """
    normalized = comparison_fold_for(language)(normalizer_for(language)(text))
    if uses_character_metric(language):
        return list(normalized.replace(" ", ""))
    return normalized.split()


def score_entities(
    annotated_reference: str, hypothesis: str, language: str
) -> dict[str, EntityClassScore]:
    """Score every tagged entity in one reference against a hypothesis.

    The full transcripts are aligned and each entity's errors are read off
    its reference token range. Insertions strictly inside a span count
    against it; insertions at a span boundary belong to the surrounding
    context, where alignment cannot say which side they disturbed.

    Segments are normalized independently so entity token ranges survive
    normalization — tags should therefore cover complete spoken phrases
    ("<number>twenty one</number>", not "twenty <number>one</number>").

    Args:
        annotated_reference: Reference with inline entity tags.
        hypothesis: Provider transcript, untagged.
        language: BCP-47 tag driving normalization and tokenization.

    Returns:
        Scores keyed by entity class; empty when nothing is tagged.
    """
    tokens: list[str] = []
    spans: list[tuple[str, int, int]] = []
    for text, label in parse_annotated(annotated_reference):
        segment = _tokenize(text, language)
        if label is not None and segment:
            spans.append((label, len(tokens), len(tokens) + len(segment)))
        tokens.extend(segment)

    if not spans or not tokens:
        return {}

    hyp_tokens = _tokenize(hypothesis, language)
    output = jiwer.process_words(" ".join(tokens), " ".join(hyp_tokens))
    alignment = output.alignments[0]

    scores: dict[str, EntityClassScore] = {}
    for label, start, end in spans:
        substitutions = deletions = insertions = 0
        for chunk in alignment:
            if chunk.type == "insert":
                if start < chunk.ref_start_idx < end:
                    insertions += chunk.hyp_end_idx - chunk.hyp_start_idx
                continue
            overlap = min(chunk.ref_end_idx, end) - max(chunk.ref_start_idx, start)
            if overlap <= 0:
                continue
            if chunk.type == "substitute":
                substitutions += overlap
            elif chunk.type == "delete":
                deletions += overlap
        counts = ErrorCounts(
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            reference_length=end - start,
        )
        score = EntityClassScore(
            counts=counts,
            occurrences=1,
            exact_matches=int(counts.errors == 0),
        )
        existing = scores.get(label)
        scores[label] = score if existing is None else existing + score
    return scores
