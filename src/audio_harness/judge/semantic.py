"""Semantic-fidelity STT judge: what a transcription error costs an interview.

WER prices every edit alike, but a dropped "uh" and a dropped "not" are very
different failures for an agent that acts on what it heard. This experimental
lane (plan P4.E1) has a pinned Flash-class text LLM classify each saved
transcript against its reference under a three-label rubric —
``meaning-changing`` / ``entity`` / ``harmless`` — with three votes per item
and a majority label. Two deterministic guards keep the judge honest:

* The entity scorer (``entities.py``) cross-checks every verdict: an item the
  deterministic scorer proves entity-damaged but the judge calls harmless is
  a judge miss, counted and reported.
* SeMaScore-style segment similarity (roberta-base, pinned revision, greedy)
  is computed alongside as the no-API fallback metric, so the lane degrades
  to a deterministic number when judge spend is off the table.

Validity gate (pre-registered, per language): unweighted Cohen's kappa
between the judge's majority label and a human anchor (CSV columns
``clip_id, provider, human_label``; 100 items per language is the target)
must reach 0.75 on the point estimate. A bootstrap CI is reported alongside
— the CI informs, the point estimate decides, and near-threshold results are
called out as such. Three-vote unanimity of at least 85% is a diagnostic,
never a gate. Until a language's anchor exists and its gate passes, every
number renders as "experimental — not ranked (anchor pending)" and must not
feed a vendor recommendation.

Judging reads saved ``stt-results.jsonl`` only — no new audio is submitted —
and every vote is cached by content, so re-running never re-bills an
already-scored item.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import jiwer
import numpy as np
import orjson

from audio_harness.autotag import autotag_reference
from audio_harness.config import require_env
from audio_harness.entities import score_entities
from audio_harness.metrics import ZERO_COUNTS, ErrorCounts, percentile, score_pair
from audio_harness.normalize import comparison_fold_for, normalizer_for, uses_character_metric
from audio_harness.types import SttResult


if TYPE_CHECKING:
    from google.genai import Client

JUDGE_MODEL = "gemini-3.6-flash"
"""Pinned judge model. An alias like ``gemini-flash-latest`` would silently
re-judge history with a different model, so the exact versioned id is fixed
here and rides into the vote cache key."""

JUDGE_USD_PER_1M_INPUT = 1.50
JUDGE_USD_PER_1M_OUTPUT = 7.50
JUDGE_PRICING_CHECKED = "2026-08-06"
"""Standard-tier pricing for :data:`JUDGE_MODEL`, last verified on the date
above. Output pricing includes thinking tokens, which is why the usage
accounting sums them into the output total."""

PROMPT_VERSION = 1
"""Bumped whenever the rubric prompt changes, invalidating cached votes."""

RUBRIC = ("meaning-changing", "entity", "harmless")
"""The three verdicts, in the plan's order."""

_SEVERITY = ("entity", "meaning-changing", "harmless")
"""Tie-break order for a 1-1-1 vote split: the most damaging verdict present
wins. Entity damage outranks general meaning change because a corrupted fact
is the failure an interview agent can least recover from."""

VOTES_PER_ITEM = 3
VOTE_SEEDS = (1, 2, 3)
"""One fixed seed per vote. Votes need diversity for the unanimity
diagnostic to mean anything, so the temperature is non-zero — but each vote
is individually reproducible given its seed."""

JUDGE_TEMPERATURE = 0.3

MAX_JUDGE_CALLS = 4800
"""Hard budget cap on live judge calls per run (plan P4.16 cost envelope)."""

KAPPA_GATE = 0.75
UNANIMITY_DIAGNOSTIC = 0.85
ANCHOR_TARGET_ITEMS = 100
"""Anchor size the gate protocol assumes; smaller anchors still compute but
their CI widens accordingly and the report shows the actual n."""

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260806
"""Fixed bootstrap seed: the CI is part of the published gate record, so two
renders of the same data must print the same interval."""

SEMASCORE_MODEL = "FacebookAI/roberta-base"
SEMASCORE_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
"""Pinned encoder for the deterministic fallback metric. Inference is greedy
— eval mode, no dropout, no sampling — so a score is a pure function of the
two transcripts."""

ANCHOR_COLUMNS = ("clip_id", "provider", "human_label")

EXPERIMENTAL_BANNER = "experimental — not ranked (anchor pending)"


@dataclass(slots=True, frozen=True)
class JudgeItem:
    """One (transcript, reference) pair to judge.

    Attributes:
        provider: Registry key of the adapter that produced the transcript.
        mode: Transport mode of the run.
        clip_id: Identifier of the transcribed clip.
        language: BCP-47 tag the pair is judged under.
        reference: Ground-truth transcript, never empty.
        hypothesis: Provider transcript; may be empty (dropping everything is
            itself a meaning-changing error, not a reason to skip).
        reference_annotated: Entity-tagged reference when the results carry
            one; the deterministic cross-check falls back to the rule tagger
            otherwise.
    """

    provider: str
    mode: str
    clip_id: str
    language: str
    reference: str
    hypothesis: str
    reference_annotated: str | None = None


@dataclass(slots=True, frozen=True)
class JudgeVote:
    """One judge verdict with its billing evidence.

    Attributes:
        label: A :data:`RUBRIC` member.
        input_tokens: Prompt tokens billed for this vote; 0 for cache hits.
        output_tokens: Response plus thinking tokens billed; 0 for cache hits.
    """

    label: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class ItemJudgement:
    """Judge outcome for one item, with the deterministic cross-checks.

    Attributes:
        item: The judged pair.
        votes: The three vote labels, in seed order.
        majority: Majority label; a full three-way split falls back to the
            most severe label present (see :data:`_SEVERITY`).
        entity_errors: Whether the deterministic entity scorer found at least
            one damaged entity, or ``None`` when nothing was taggable.
        semascore: Deterministic fallback similarity in [0, 1], or ``None``
            when the optional encoder was unavailable.
        wer: Per-item error rate under the shared normalization, for
            semantic-vs-WER divergence analysis.
    """

    item: JudgeItem
    votes: tuple[str, ...]
    majority: str
    entity_errors: bool | None
    semascore: float | None
    wer: float | None

    @property
    def unanimous(self) -> bool:
        """Whether all votes agree."""
        return len(set(self.votes)) == 1

    @property
    def judge_missed_entity(self) -> bool | None:
        """Judge called it harmless while the entity scorer proved damage.

        ``None`` when the deterministic check had nothing to verify.
        """
        if self.entity_errors is None:
            return None
        return self.entity_errors and self.majority == "harmless"


class JudgeBudgetError(RuntimeError):
    """Raised before any call is made when a run would exceed the cap."""


def judgeable_items(results: list[SttResult], language: str) -> list[JudgeItem]:
    """Select the saved results the judge can score.

    Failed runs have no transcript to judge; reference-free clips (latency
    and hallucination material) have no truth to judge against; unverified
    subtitle references would grade the judge against caption quality. All
    are skipped. Duplicate (provider, mode, clip) rows keep the last
    occurrence, matching the supersede-merge convention.

    Args:
        results: Saved runs, typically a supersede merge of several files.
        language: Fallback BCP-47 tag for results that recorded none.

    Returns:
        One item per judgeable (provider, mode, clip), in stable order.
    """
    items: dict[tuple[str, str, str], JudgeItem] = {}
    for result in results:
        if not result.ok:
            continue
        reference = result.raw.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            continue
        if result.raw.get("gold_status") == "unverified":
            continue
        recorded = result.raw.get("language")
        clip_language = recorded if isinstance(recorded, str) and recorded else language
        annotated = result.raw.get("reference_annotated")
        key = (result.provider, str(result.mode), result.clip_id)
        items[key] = JudgeItem(
            provider=result.provider,
            mode=str(result.mode),
            clip_id=result.clip_id,
            language=clip_language,
            reference=reference,
            hypothesis=result.text,
            reference_annotated=annotated if isinstance(annotated, str) else None,
        )
    return list(items.values())


_PROMPT = """\
You grade speech-to-text output for a spoken interview agent.
Compare the transcript against the reference and classify the worst \
transcription error with exactly one label:

- "entity": a name, number, date, time, currency amount, address or \
alphanumeric ID is wrong, missing or invented. Entity damage outranks \
everything else: if a fact-carrying token is corrupted, answer "entity" \
even when other meaning changed too.
- "meaning-changing": no entity is damaged, but the transcript asserts, \
negates, omits or invents something that changes what the speaker said.
- "harmless": only fillers, punctuation, casing, spelling variants or \
wording differences that leave the meaning intact.

Language: {language}
Reference: {reference}
Transcript: {hypothesis}
"""


def build_prompt(item: JudgeItem) -> str:
    """Render the rubric prompt for one item."""
    return _PROMPT.format(
        language=item.language,
        reference=item.reference,
        hypothesis=item.hypothesis,
    )


def cache_key(item: JudgeItem, seed: int) -> str:
    """Content hash identifying one vote, stable across runs.

    The key covers everything that determines the verdict — model, prompt
    version, language, both transcripts and the vote seed — and nothing
    that does not (provider and clip id stay out, so identical pairs share
    votes instead of re-billing).
    """
    payload = orjson.dumps([
        JUDGE_MODEL,
        PROMPT_VERSION,
        JUDGE_TEMPERATURE,
        item.language,
        item.reference,
        item.hypothesis,
        seed,
    ])
    return hashlib.sha256(payload).hexdigest()


class VoteCache:
    """Append-only JSONL store making judge runs idempotent.

    Every billed vote is recorded under its content key the moment it
    arrives, so a crash mid-run loses nothing and a re-run re-bills nothing.

    Args:
        path: Cache file; created on first write, loaded when present.
    """

    def __init__(self, path: str | Path) -> None:
        """Load any existing cache records from ``path``."""
        self.path = Path(path)
        self._votes: dict[str, str] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = orjson.loads(line)
                self._votes[record["key"]] = record["label"]

    def get(self, key: str) -> str | None:
        """Return the cached label for ``key``, or ``None``."""
        return self._votes.get(key)

    def put(self, key: str, label: str) -> None:
        """Record a vote, appending it to the backing file immediately."""
        self._votes[key] = label
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(orjson.dumps({"key": key, "label": label}))
            handle.write(b"\n")


Judge = Callable[[JudgeItem, int], JudgeVote]
"""A judge maps (item, seed) to one vote. Injectable so tests never bill."""


class GeminiJudge:
    """Pinned Gemini Flash judge with structured single-label output.

    Votes use a non-zero temperature with a fixed per-vote seed: zero
    temperature would make three votes one vote and the unanimity diagnostic
    meaningless, while an unseeded sample would make no vote reproducible.
    Thinking is pinned to the lowest level — this is a classification, and
    output tokens bill at five times the input rate.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Store the key override and defer client construction."""
        self._api_key = api_key
        self._client: Client | None = None

    def _genai_client(self) -> Client:
        """Return a lazily constructed genai client."""
        if self._client is None:
            from google import genai

            key = self._api_key or require_env("GEMINI_API_KEY", "judge-semantic")
            self._client = genai.Client(api_key=key)
        return self._client

    def __call__(self, item: JudgeItem, seed: int) -> JudgeVote:
        """Request one vote for ``item`` under ``seed``."""
        from google.genai import types as genai_types

        client = self._genai_client()
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=build_prompt(item),
            config=genai_types.GenerateContentConfig(
                temperature=JUDGE_TEMPERATURE,
                seed=seed,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "label": {"type": "STRING", "enum": list(RUBRIC)},
                    },
                    "required": ["label"],
                },
                thinking_config=genai_types.ThinkingConfig(thinking_level=genai_types.ThinkingLevel.LOW),
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return JudgeVote(
            label=_parse_label(response.text or ""),
            input_tokens=getattr(usage, "prompt_token_count", None) or 0,
            output_tokens=(getattr(usage, "candidates_token_count", None) or 0)
            + (getattr(usage, "thoughts_token_count", None) or 0),
        )


def _parse_label(text: str) -> str:
    """Extract the rubric label from a judge response.

    The structured-output schema makes valid JSON the overwhelmingly common
    case; a lenient substring scan covers a model that wraps the JSON in
    prose. Anything else is an error — an unparseable vote must fail loudly,
    not be guessed at and cached forever.
    """
    try:
        record = orjson.loads(text)
    except orjson.JSONDecodeError:
        record = None
    if isinstance(record, dict):
        label = record.get("label")
        if label in RUBRIC:
            return str(label)
    lowered = text.lower()
    # Substring order matters: "meaning-changing" does not contain the other
    # labels, but a prose answer may mention several; severity order keeps
    # the scan deterministic and conservative.
    for label in _SEVERITY:
        if label in lowered:
            return label
    raise RuntimeError(f"judge returned no rubric label: {text!r}")


def majority_label(votes: Sequence[str]) -> str:
    """Majority vote with a severity tie-break.

    A 2-1 split resolves by count. A full three-way split has no majority,
    so the most severe label present wins — an uncertain judge must never
    launder an item into "harmless".
    """
    counts = Counter(votes)
    best = max(counts.values())
    leaders = {label for label, count in counts.items() if count == best}
    for label in _SEVERITY:
        if label in leaders:
            return label
    raise ValueError(f"votes contain no rubric label: {votes!r}")


def entity_check(item: JudgeItem) -> bool | None:
    """Deterministically verify the item's entities.

    Uses the results file's annotation when present, else the rule tagger.

    Returns:
        ``True``/``False`` for damaged/intact entities, or ``None`` when the
        reference contains nothing taggable, in which case there is nothing
        for the judge to be checked against.
    """
    annotated = item.reference_annotated or autotag_reference(item.reference, item.language)
    scores = score_entities(annotated, item.hypothesis, item.language)
    if not scores:
        return None
    return any(score.counts.errors > 0 for score in scores.values())


@dataclass(slots=True)
class JudgeRunStats:
    """Billing evidence for one judge run.

    Attributes:
        live_calls: Votes that hit the API this run.
        cached_votes: Votes served from the cache, billed on a prior run.
        input_tokens: Prompt tokens billed this run.
        output_tokens: Response plus thinking tokens billed this run.
    """

    live_calls: int = 0
    cached_votes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def estimated_usd(self) -> float:
        """Spend estimate from the pinned per-token pricing."""
        return self.input_tokens / 1e6 * JUDGE_USD_PER_1M_INPUT + self.output_tokens / 1e6 * JUDGE_USD_PER_1M_OUTPUT


def run_judge(
    items: Sequence[JudgeItem],
    judge: Judge,
    cache: VoteCache,
    *,
    max_calls: int = MAX_JUDGE_CALLS,
    semascore_fn: Callable[[JudgeItem], float | None] | None = None,
) -> tuple[list[ItemJudgement], JudgeRunStats]:
    """Judge every item, spending API calls only on uncached votes.

    The budget is checked before the first call: a run that would exceed
    ``max_calls`` raises immediately rather than stopping half-billed.

    Args:
        items: Pairs to judge.
        judge: Vote source; the real API judge or a test stub.
        cache: Idempotency store consulted per vote.
        max_calls: Hard cap on live calls for this run.
        semascore_fn: Optional deterministic fallback metric, computed per
            item alongside the votes.

    Returns:
        Judgements in input order, and the run's billing stats.

    Raises:
        JudgeBudgetError: If the uncached vote count exceeds ``max_calls``.
    """
    needed = sum(1 for item in items for seed in VOTE_SEEDS if cache.get(cache_key(item, seed)) is None)
    if needed > max_calls:
        raise JudgeBudgetError(
            f"run needs {needed} live judge calls, over the {max_calls}-call "
            f"cap; raise --max-calls only with an approved budget"
        )

    stats = JudgeRunStats()
    judgements: list[ItemJudgement] = []
    for item in items:
        votes: list[str] = []
        for seed in VOTE_SEEDS:
            key = cache_key(item, seed)
            cached = cache.get(key)
            if cached is not None:
                stats.cached_votes += 1
                votes.append(cached)
                continue
            vote = judge(item, seed)
            if vote.label not in RUBRIC:
                raise RuntimeError(f"judge returned unknown label {vote.label!r} for {item.provider}/{item.clip_id}")
            cache.put(key, vote.label)
            stats.live_calls += 1
            stats.input_tokens += vote.input_tokens
            stats.output_tokens += vote.output_tokens
            votes.append(vote.label)

        counts = score_pair(item.reference, item.hypothesis, item.language)
        judgements.append(
            ItemJudgement(
                item=item,
                votes=tuple(votes),
                majority=majority_label(votes),
                entity_errors=entity_check(item),
                semascore=semascore_fn(item) if semascore_fn is not None else None,
                wer=counts.rate if counts.reference_length else None,
            )
        )
    return judgements, stats


def _tokens(text: str, language: str) -> list[str]:
    """Tokenize the way the language's headline metric does."""
    folded = comparison_fold_for(language)(normalizer_for(language)(text))
    if uses_character_metric(language):
        return list(folded.replace(" ", ""))
    return folded.split()


def semascore(
    reference: str,
    hypothesis: str,
    language: str,
    embed: Callable[[str], np.ndarray],
) -> float:
    """Segment-aligned semantic similarity in [0, 1] (SeMaScore construction).

    Follows the construction of SeMaScore (arXiv:2401.07506): align the
    normalized transcripts, then score each alignment segment — exact
    matches score 1, substituted spans score the embedding cosine of the two
    span texts, deleted and inserted spans score 0 — and combine the segment
    scores weighted by span length. Identical transcripts score exactly 1.0;
    a semantically close paraphrase outscores a meaning flip of equal edit
    distance, which is the whole point of the fallback.

    Deterministic by design: greedy alignment, greedy encoding, no sampling.

    Args:
        reference: Ground-truth transcript.
        hypothesis: Provider transcript.
        language: BCP-47 tag driving normalization and tokenization.
        embed: Text-to-vector encoder; :func:`roberta_embedder` in
            production, a stub in tests.

    Returns:
        Similarity in [0, 1].
    """
    ref_tokens = _tokens(reference, language)
    hyp_tokens = _tokens(hypothesis, language)
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    output = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens))
    weighted = 0.0
    weight_total = 0.0
    for chunk in output.alignments[0]:
        ref_span = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
        hyp_span = hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx]
        weight = float(max(len(ref_span), len(hyp_span)))
        if chunk.type == "equal":
            score = 1.0
        elif chunk.type == "substitute":
            score = _cosine(embed(" ".join(ref_span)), embed(" ".join(hyp_span)))
        else:  # delete / insert: content lost or invented has no counterpart
            score = 0.0
        weighted += weight * score
        weight_total += weight
    return weighted / weight_total if weight_total else 1.0


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity clipped to [0, 1]; zero vectors score 0."""
    norms = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if norms <= 0.0:
        return 0.0
    return float(np.clip(np.dot(left, right) / norms, 0.0, 1.0))


def roberta_embedder() -> Callable[[str], np.ndarray]:
    """Build the pinned deterministic sentence encoder.

    Mean-pools the masked last hidden state of :data:`SEMASCORE_MODEL` at
    :data:`SEMASCORE_REVISION`, in eval mode under ``no_grad`` — no dropout,
    no sampling, so equal inputs always embed equally. Embeddings are cached
    per text because judged corpora repeat references across providers.

    Returns:
        A text-to-vector callable.

    Raises:
        RuntimeError: If the optional dependencies are missing; install with
            ``uv sync --extra judge-semantic``.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "SeMaScore needs the optional judge-semantic dependency group: uv sync --extra judge-semantic"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(SEMASCORE_MODEL, revision=SEMASCORE_REVISION)
    if tokenizer is None:
        raise RuntimeError(
            f"AutoTokenizer.from_pretrained returned no tokenizer for {SEMASCORE_MODEL}@{SEMASCORE_REVISION}"
        )
    model = AutoModel.from_pretrained(SEMASCORE_MODEL, revision=SEMASCORE_REVISION)
    model.eval()
    cache: dict[str, np.ndarray] = {}

    def embed(text: str) -> np.ndarray:
        cached = cache.get(text)
        if cached is not None:
            return cached
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state[0]
        mask = encoded["attention_mask"][0].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=0) / mask.sum()
        vector = np.asarray(pooled.numpy(), dtype=np.float64)
        cache[text] = vector
        return vector

    return embed


def load_anchor(path: str | Path) -> dict[tuple[str, str], str]:
    """Load a human anchor file.

    The anchor format is part of the gate protocol: a CSV with columns
    ``clip_id, provider, human_label``, one row per anchored item, labels
    drawn from the rubric.

    Args:
        path: Anchor CSV.

    Returns:
        ``(clip_id, provider)`` to human label.

    Raises:
        ValueError: On a missing column, an unknown label, or a duplicate
            row — silent anchor corruption would corrupt the gate.
    """
    anchor: dict[tuple[str, str], str] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(ANCHOR_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"anchor file {path} is missing columns: {', '.join(sorted(missing))}")
        for number, row in enumerate(reader, 2):
            label = (row["human_label"] or "").strip()
            if label not in RUBRIC:
                raise ValueError(f"{path}:{number}: unknown label {label!r}; expected one of {', '.join(RUBRIC)}")
            key = (row["clip_id"].strip(), row["provider"].strip())
            if key in anchor:
                raise ValueError(f"{path}:{number}: duplicate anchor row {key}")
            anchor[key] = label
    return anchor


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    """Unweighted Cohen's kappa between two label sequences.

    Degenerate marginals — both raters using a single identical label — make
    expected agreement 1; kappa is then 1 for perfect observed agreement and
    0 otherwise, the conventional convention.

    Args:
        pairs: ``(human, judge)`` label pairs.

    Returns:
        Kappa, or ``None`` for empty input.
    """
    if not pairs:
        return None
    n = len(pairs)
    matches = sum(1 for human, judge in pairs if human == judge)
    observed = matches / n
    human_marginal = Counter(human for human, _ in pairs)
    judge_marginal = Counter(judge for _, judge in pairs)
    # Degeneracy is decided in exact integer arithmetic: expected agreement
    # is 1 iff the marginal product mass equals n^2, observed iff all match.
    expected_mass = sum(human_marginal[label] * judge_marginal[label] for label in RUBRIC)
    if expected_mass == n * n:
        return 1.0 if matches == n else 0.0
    expected = expected_mass / (n * n)
    return (observed - expected) / (1.0 - expected)


def bootstrap_kappa_ci(
    pairs: Sequence[tuple[str, str]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """95% percentile-bootstrap CI for :func:`cohens_kappa`.

    Reported next to the gate's point estimate — at the target anchor size
    of 100 items the half-width is roughly 0.1, which is why the protocol
    says the CI informs and the point estimate decides.

    Args:
        pairs: ``(human, judge)`` label pairs.
        resamples: Bootstrap iterations.
        seed: RNG seed, fixed so a re-render prints the same interval.

    Returns:
        ``(low, high)``, or ``None`` for empty input.
    """
    if not pairs:
        return None
    rng = np.random.default_rng(seed)
    kappas: list[float] = []
    for _ in range(resamples):
        indices = rng.integers(0, len(pairs), size=len(pairs))
        resampled = [pairs[int(index)] for index in indices]
        kappa = cohens_kappa(resampled)
        if kappa is not None:
            kappas.append(kappa)
    low = percentile(kappas, 2.5)
    high = percentile(kappas, 97.5)
    if low is None or high is None:
        return None
    return (low, high)


@dataclass(slots=True, frozen=True)
class GateStatus:
    """Gate verdict for one language.

    Attributes:
        language: BCP-47 tag the verdict covers.
        judged: Items judged in this language.
        anchored: Anchor rows that matched a judged item.
        kappa: Point estimate against the anchor, or ``None`` unanchored.
        ci: Bootstrap 95% CI, or ``None`` unanchored.
        unanimity: Fraction of judged items with three agreeing votes —
            a diagnostic, never a gate.
    """

    language: str
    judged: int
    anchored: int
    kappa: float | None
    ci: tuple[float, float] | None
    unanimity: float | None

    @property
    def passed(self) -> bool:
        """Whether the point estimate clears the pre-registered gate."""
        return self.kappa is not None and self.kappa >= KAPPA_GATE

    @property
    def near_threshold(self) -> bool:
        """Whether the CI straddles the gate, so the verdict is fragile."""
        return self.ci is not None and self.ci[0] < KAPPA_GATE <= self.ci[1]

    @property
    def status(self) -> str:
        """Rendered status string for report tables."""
        if self.kappa is None:
            return EXPERIMENTAL_BANNER
        verdict = (
            f"gate passed (kappa {self.kappa:.2f} >= {KAPPA_GATE})"
            if self.passed
            else f"experimental — not ranked (kappa {self.kappa:.2f} < {KAPPA_GATE})"
        )
        if self.near_threshold:
            verdict += " — near threshold, CI straddles the gate"
        return verdict


def evaluate_gates(
    judgements: Sequence[ItemJudgement],
    anchor: Mapping[tuple[str, str], str] | None,
) -> list[GateStatus]:
    """Compute the per-language gate record.

    Anchor rows key on ``(clip_id, provider)`` without a mode; when both
    modes of a lane were judged, the stream judgement is the one compared —
    streaming is the product path — and the batch one never silently double
    counts an anchor row.

    Args:
        judgements: Every judged item.
        anchor: Human anchor labels, or ``None`` when no anchor exists yet.

    Returns:
        One status per judged language, sorted by language.
    """
    by_language: dict[str, list[ItemJudgement]] = {}
    for judgement in judgements:
        by_language.setdefault(judgement.item.language, []).append(judgement)

    statuses: list[GateStatus] = []
    for language, group in sorted(by_language.items()):
        chosen: dict[tuple[str, str], ItemJudgement] = {}
        for judgement in group:
            key = (judgement.item.clip_id, judgement.item.provider)
            existing = chosen.get(key)
            if existing is None or (existing.item.mode != "stream" and judgement.item.mode == "stream"):
                chosen[key] = judgement

        pairs: list[tuple[str, str]] = []
        if anchor:
            for key, judgement in chosen.items():
                human = anchor.get(key)
                if human is not None:
                    pairs.append((human, judgement.majority))

        unanimity = sum(1 for judgement in group if judgement.unanimous) / len(group) if group else None
        statuses.append(
            GateStatus(
                language=language,
                judged=len(group),
                anchored=len(pairs),
                kappa=cohens_kappa(pairs),
                ci=bootstrap_kappa_ci(pairs),
                unanimity=unanimity,
            )
        )
    return statuses


@dataclass(slots=True)
class SemanticSummary:
    """Judge outcomes for one provider, mode and language.

    Attributes:
        provider: Registry key of the adapter.
        mode: Transport mode of the judged runs.
        language: BCP-47 tag the items were judged under.
        items: Judged items.
        labels: Majority-label counts.
        unanimous: Items with three agreeing votes.
        entity_checked: Items the deterministic entity check could verify.
        judge_missed_entities: Verified items the judge called harmless
            despite proven entity damage.
        semascores: Fallback similarities for items that had one.
        counts: Summed edit operations, for the WER-vs-semantics divergence.
    """

    provider: str
    mode: str
    language: str
    items: int = 0
    labels: dict[str, int] = field(default_factory=dict)
    unanimous: int = 0
    entity_checked: int = 0
    judge_missed_entities: int = 0
    semascores: list[float] = field(default_factory=list)
    counts: ErrorCounts = ZERO_COUNTS

    def label_rate(self, label: str) -> float | None:
        """Fraction of items whose majority verdict is ``label``."""
        if self.items == 0:
            return None
        return self.labels.get(label, 0) / self.items

    @property
    def unanimity_rate(self) -> float | None:
        """Fraction of items with unanimous votes."""
        if self.items == 0:
            return None
        return self.unanimous / self.items

    @property
    def mean_semascore(self) -> float | None:
        """Mean fallback similarity, or ``None`` when never computed."""
        if not self.semascores:
            return None
        return sum(self.semascores) / len(self.semascores)

    @property
    def error_rate(self) -> float | None:
        """Corpus WER over the judged items, for divergence analysis."""
        if self.counts.reference_length == 0:
            return None
        return self.counts.rate


def summarize_semantic(
    judgements: Sequence[ItemJudgement],
) -> list[SemanticSummary]:
    """Aggregate judgements per provider, mode and language.

    Args:
        judgements: Every judged item.

    Returns:
        Summaries sorted by language, provider and mode.
    """
    summaries: dict[tuple[str, str, str], SemanticSummary] = {}
    for judgement in judgements:
        item = judgement.item
        key = (item.provider, item.mode, item.language)
        summary = summaries.setdefault(
            key,
            SemanticSummary(provider=item.provider, mode=item.mode, language=item.language),
        )
        summary.items += 1
        summary.labels[judgement.majority] = summary.labels.get(judgement.majority, 0) + 1
        if judgement.unanimous:
            summary.unanimous += 1
        if judgement.entity_errors is not None:
            summary.entity_checked += 1
            if judgement.judge_missed_entity:
                summary.judge_missed_entities += 1
        if judgement.semascore is not None:
            summary.semascores.append(judgement.semascore)
        summary.counts = summary.counts + score_pair(item.reference, item.hypothesis, item.language)
    return sorted(summaries.values(), key=lambda s: (s.language, s.provider, s.mode))


def render_semantic_markdown(summaries: Sequence[SemanticSummary], gates: Sequence[GateStatus]) -> str:
    """Render the judge tables as GitHub-flavoured markdown.

    Every language's rows carry the gate status inline; an unanchored or
    failed language renders as experimental and must not be read as a
    ranking.
    """
    if not summaries:
        return "_No judgeable results._"

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value * 100:.1f}%"

    gate_by_language = {gate.language: gate for gate in gates}

    lines = ["### Gate status (per language)", ""]
    lines += [
        "| Lang | Judged | Anchored | Cohen's kappa | 95% CI | Unanimity | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for gate in gates:
        ci = "—" if gate.ci is None else f"[{gate.ci[0]:.2f}, {gate.ci[1]:.2f}]"
        kappa = "—" if gate.kappa is None else f"{gate.kappa:.2f}"
        lines.append(
            f"| {gate.language} | {gate.judged} | {gate.anchored} | {kappa} "
            f"| {ci} | {pct(gate.unanimity)} | {gate.status} |"
        )
    lines += [
        "",
        f"_Gate: point-estimate kappa >= {KAPPA_GATE} on a "
        f"{ANCHOR_TARGET_ITEMS}-item human anchor decides; the CI is "
        f"reported, not gating. Unanimity >= {UNANIMITY_DIAGNOSTIC:.0%} is "
        "a diagnostic only._",
        "",
        "### Judge verdicts",
        "",
        "| Provider | Mode | Lang | Items | WER | Meaning-chg | Entity "
        "| Harmless | Unanimity | Judge missed entity | SeMaScore | Status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        language_gate = gate_by_language.get(summary.language)
        status = language_gate.status if language_gate is not None else EXPERIMENTAL_BANNER
        missed = f"{summary.judge_missed_entities}/{summary.entity_checked}" if summary.entity_checked else "—"
        mean_semascore = summary.mean_semascore
        lines.append(
            f"| {summary.provider} | {summary.mode} | {summary.language} "
            f"| {summary.items} | {pct(summary.error_rate)} "
            f"| {pct(summary.label_rate('meaning-changing'))} "
            f"| {pct(summary.label_rate('entity'))} "
            f"| {pct(summary.label_rate('harmless'))} "
            f"| {pct(summary.unanimity_rate)} | {missed} "
            f"| {'—' if mean_semascore is None else f'{mean_semascore:.3f}'} "
            f"| {status} |"
        )
    lines += [
        "",
        "_Meaning-chg/Entity/Harmless: fraction of judged items whose "
        "majority-of-3 verdict is that label. Judge missed entity: items "
        "the deterministic entity scorer proved damaged but the judge "
        "called harmless. SeMaScore: deterministic no-API fallback "
        "similarity (1.0 = semantically identical)._",
    ]
    return "\n".join(lines)


def write_semantic_results(
    judgements: Sequence[ItemJudgement],
    gates: Sequence[GateStatus],
    stats: JudgeRunStats,
    path: str | Path,
) -> Path:
    """Persist judgements, per-language gate metrics and billing as JSONL.

    The gate record rides in the results file (AC8): a reader must never
    have to guess whether a number was gated.

    Args:
        judgements: Every judged item.
        gates: Per-language gate statuses.
        stats: The run's billing evidence.
        path: Output file.

    Returns:
        The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        for judgement in judgements:
            item = judgement.item
            handle.write(
                orjson.dumps({
                    "provider": item.provider,
                    "clip_id": item.clip_id,
                    "mode": item.mode,
                    "language": item.language,
                    "reference": item.reference,
                    "hypothesis": item.hypothesis,
                    "votes": list(judgement.votes),
                    "majority": judgement.majority,
                    "unanimous": judgement.unanimous,
                    "entity_errors": judgement.entity_errors,
                    "judge_missed_entity": judgement.judge_missed_entity,
                    "semascore": judgement.semascore,
                    "wer": judgement.wer,
                    "model": JUDGE_MODEL,
                    "prompt_version": PROMPT_VERSION,
                })
            )
            handle.write(b"\n")
        for gate in gates:
            handle.write(
                orjson.dumps({
                    "gate_language": gate.language,
                    "judged": gate.judged,
                    "anchored": gate.anchored,
                    "kappa": gate.kappa,
                    "ci": list(gate.ci) if gate.ci else None,
                    "unanimity": gate.unanimity,
                    "passed": gate.passed,
                    "near_threshold": gate.near_threshold,
                })
            )
            handle.write(b"\n")
        handle.write(
            orjson.dumps({
                "run_stats": {
                    "model": JUDGE_MODEL,
                    "live_calls": stats.live_calls,
                    "cached_votes": stats.cached_votes,
                    "input_tokens": stats.input_tokens,
                    "output_tokens": stats.output_tokens,
                    "estimated_usd": stats.estimated_usd,
                    "pricing_checked": JUDGE_PRICING_CHECKED,
                }
            })
        )
        handle.write(b"\n")
    return target
