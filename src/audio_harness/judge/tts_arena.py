"""Audio-LLM pairwise TTS arena (AudioJudge protocol, arXiv:2507.12705).

Pairwise comparison is the only reliable way to use an audio-LLM as a TTS
judge: absolute example-level scores are near chance, while system-level
Bradley-Terry rankings over order-counterbalanced pairs track human panels
(AudioJudge, arXiv:2507.12705). This module therefore never scores a clip on
its own. Every judged item is one concatenated audio file holding two
renditions of the same text, judged twice (once per presentation order) by
three pinned aspect judges, and only the aggregated system-level ranking is
reported.

Validity gate (plan P4 step 17; all three criteria required, with the number
of systems stated at gate time):

(i)   Human-panel agreement. Decision rule: no pair of systems whose
      Bradley-Terry bootstrap CIs are separated may be ranked discordantly
      versus the human-panel Bradley-Terry ranking (panel of at least
      :data:`PANEL_MIN_RATERS` raters rating at least
      :data:`PANEL_MIN_PAIRS_PER_RATER` pairs each; inter-rater agreement
      reported; Spearman rho and exact permutation p reported descriptively,
      never as the decision). No panel has been collected yet, so the lane
      renders "experimental (panel pending)".
(ii)  Cross-family judge agreement rho >= :data:`CROSS_FAMILY_RHO_GATE`
      between judges from different model families. Only a Gemini audio-in
      judge is available today (no OpenAI key), so this criterion is
      UNCOMPUTABLE; it is reported as such and never faked.
(iii) Order-flip rate <= :data:`ORDER_FLIP_GATE`: computable now and
      computed on every run.

Because criterion (ii) cannot currently be evaluated, the gate can never
pass with a single judge family: the arena stays "experimental - not ranked"
by construction, which is the honest reading of the evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import random
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import orjson
import soxr
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from ..audio import _float_to_pcm16, read_audio_samples, wrap_wav
from ..config import require_env
from ..types import TtsPrompt

JUDGE_MODEL = "gemini-2.5-flash"
"""Pinned audio-in judge model. Changing it invalidates every cached verdict."""

JUDGE_FAMILY = "gemini"
"""Model family of the pinned judge, for the cross-family gate criterion."""

ARENA_SAMPLE_RATE = 16000
"""Rate every clip is resampled to before concatenation; Gemini's audio floor."""

PAIR_GAP_S = 0.8
"""Silence inserted between the two clips of a concatenated pair."""

ASPECTS = ("naturalness", "prosody", "artifacts")
"""Pinned aspect judges; each aspect is a separate call over the same pair."""

_ASPECT_FOCUS: dict[str, str] = {
    "naturalness": (
        "Judge which clip sounds more natural: closer to a fluent human "
        "speaker and less obviously synthetic."
    ),
    "prosody": (
        "Judge which clip has better prosody: rhythm, stress, intonation "
        "and phrasing appropriate to the text."
    ),
    "artifacts": (
        "Judge which clip has fewer audio artifacts: glitches, clicks, "
        "buzzing, distortion, dropouts or robotic timbre."
    ),
}

ORDER_FLIP_GATE = 0.15
"""Maximum tolerated fraction of pairs whose verdict follows position."""

CROSS_FAMILY_RHO_GATE = 0.9
"""Minimum Spearman rho required between judges from different families."""

PANEL_MIN_RATERS = 2
PANEL_MIN_PAIRS_PER_RATER = 100
"""Pre-registered minimum human-panel size for criterion (i)."""

BOOTSTRAP_RESAMPLES = 1000
"""Cluster-bootstrap resamples (over prompts) behind every reported CI."""

STRENGTH_FLOOR = 1e-12
"""Floor on a Bradley-Terry strength so an all-loss system stays finite."""

JUDGE_PRICING_CHECKED = "2026-08-06"
JUDGE_USD_PER_M_INPUT_TOKENS = 1.00
"""Gemini 2.5 Flash audio-input rate; text share of a call is negligible, so
every prompt token is priced at the audio rate (a slight over-estimate)."""

JUDGE_USD_PER_M_OUTPUT_TOKENS = 2.50


class ArenaError(RuntimeError):
    """Raised when the arena cannot run as configured."""


JudgeCall = Callable[[bytes, str], Awaitable["JudgeReply"]]
"""An async judge: (pair WAV bytes, instruction) -> raw model reply."""


@dataclass(slots=True, frozen=True)
class JudgeReply:
    """Raw reply from one judge call.

    Attributes:
        text: Verbatim model output, parsed later by :func:`parse_verdict`.
        prompt_tokens: Input tokens billed for the call, when reported.
        output_tokens: Output tokens billed for the call, when reported.
    """

    text: str
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass(slots=True)
class Verdict:
    """One judged (aspect, prompt, ordered pair) comparison.

    Attributes:
        aspect: Which aspect judge produced this verdict.
        prompt_id: Prompt whose two renditions were compared.
        first: System presented as clip one.
        second: System presented as clip two.
        winner: ``first``, ``second``, ``tie``, or ``error`` when the call
            failed or the reply did not parse.
        model: Judge model that produced the verdict.
        cached: Whether the verdict was served from the on-disk cache.
        prompt_tokens: Input tokens billed, zero for cached verdicts' reruns.
        output_tokens: Output tokens billed.
        raw_text: Verbatim model reply, kept for auditing parse decisions.
        error: Failure description when ``winner`` is ``error``.
    """

    aspect: str
    prompt_id: str
    first: str
    second: str
    winner: str
    model: str = JUDGE_MODEL
    cached: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0
    raw_text: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the verdict carries a usable judgement."""
        return self.winner in ("first", "second", "tie")

    @property
    def winner_system(self) -> str | None:
        """The winning system's key, or ``None`` for a tie or an error."""
        if self.winner == "first":
            return self.first
        if self.winner == "second":
            return self.second
        return None


@dataclass(slots=True, frozen=True)
class Comparison:
    """One Bradley-Terry observation.

    Attributes:
        first: One system of the pair.
        second: The other system.
        first_wins: Win credit for ``first``: 1.0, 0.5 (tie) or 0.0.
        prompt_id: Prompt the comparison came from; the cluster-bootstrap
            resamples whole prompts because verdicts within one prompt are
            correlated.
    """

    first: str
    second: str
    first_wins: float
    prompt_id: str = ""


@dataclass(slots=True, frozen=True)
class BtScore:
    """Bradley-Terry strength of one system, with its bootstrap CI.

    Attributes:
        system: Registry key of the TTS adapter.
        log_strength: Log of the Bradley-Terry strength, geometric-mean
            centred so scores are comparable across runs.
        ci_low: Lower 95% cluster-bootstrap bound on ``log_strength``.
        ci_high: Upper 95% cluster-bootstrap bound.
        wins: Win credit accumulated across every comparison (ties count
            half).
        games: Number of comparisons the system took part in.
    """

    system: str
    log_strength: float
    ci_low: float
    ci_high: float
    wins: float
    games: float


@dataclass(slots=True, frozen=True)
class OrderFlipStats:
    """Order-consistency bookkeeping for criterion (iii).

    Every (aspect, prompt, pair) is judged twice with the presentation order
    swapped. A judge without position bias names the same *system* (or a tie)
    both times; naming different systems means the verdict followed the
    position, not the audio.

    Attributes:
        judged: Swap-groups where both orders produced a usable verdict.
        flips: Swap-groups whose two verdicts name different systems.
        dropped: Swap-groups excluded because at least one verdict errored.
    """

    judged: int
    flips: int
    dropped: int

    @property
    def rate(self) -> float | None:
        """Flip fraction, or ``None`` when nothing was judged."""
        if self.judged == 0:
            return None
        return self.flips / self.judged


@dataclass(slots=True, frozen=True)
class PanelVote:
    """One human rater's judgement of one presented pair.

    The panel file is JSONL, one object per vote, with exactly these fields;
    ``winner`` uses the same presentation-order convention as :class:`Verdict`
    (``first`` / ``second`` / ``tie``).
    """

    rater: str
    prompt_id: str
    first: str
    second: str
    winner: str

    @property
    def winner_system(self) -> str | None:
        """The winning system's key, or ``None`` for a tie."""
        if self.winner == "first":
            return self.first
        if self.winner == "second":
            return self.second
        return None


@dataclass(slots=True, frozen=True)
class Criterion:
    """One gate criterion with its honest status.

    Attributes:
        name: Short criterion label.
        status: ``pass``, ``fail``, ``pending`` (evidence not yet collected)
            or ``uncomputable`` (evidence cannot currently be collected).
        detail: Everything a reader needs to audit the status.
    """

    name: str
    status: str
    detail: str


@dataclass(slots=True, frozen=True)
class GateReport:
    """The three-criterion validity gate, evaluated for one language.

    Attributes:
        criteria: The three criteria, in plan order.
        n_systems: Number of systems ranked, stated at gate time as the plan
            requires (CI separation behaves differently at n=4 than n=8).
    """

    criteria: tuple[Criterion, ...]
    n_systems: int

    @property
    def passed(self) -> bool:
        """Whether every criterion passed — required before ranking."""
        return all(criterion.status == "pass" for criterion in self.criteria)

    @property
    def label(self) -> str:
        """Report label rendering the gate outcome honestly."""
        if self.passed:
            return f"gate passed - ranked (n={self.n_systems} systems)"
        failed = [c.name for c in self.criteria if c.status == "fail"]
        if failed:
            return (
                f"experimental - not ranked (n={self.n_systems} systems; "
                f"failed: {', '.join(failed)})"
            )
        return f"experimental (panel pending; n={self.n_systems} systems)"


@dataclass(slots=True)
class ArenaRun:
    """Everything one arena execution produced and spent.

    Attributes:
        verdicts: Every judged comparison, including cached and errored ones.
        systems: Systems that took part, sorted.
        language: BCP-47 tag of the prompts.
        model: Judge model used.
        live_calls: API calls actually made by this run.
        cached_calls: Verdicts served from the on-disk cache.
        error_calls: Calls that failed after retries or did not parse.
        live_prompt_tokens: Input tokens billed by this run's live calls.
        live_output_tokens: Output tokens billed by this run's live calls.
        missing_audio: ``system:prompt_id`` labels whose WAV was absent, so
            every pair involving them was skipped rather than silently
            shrunk.
    """

    verdicts: list[Verdict] = field(default_factory=list)
    systems: tuple[str, ...] = ()
    language: str = ""
    model: str = JUDGE_MODEL
    live_calls: int = 0
    cached_calls: int = 0
    error_calls: int = 0
    live_prompt_tokens: int = 0
    live_output_tokens: int = 0
    missing_audio: list[str] = field(default_factory=list)

    @property
    def est_usd(self) -> float:
        """Estimated spend of this run's live calls (cached calls are free).

        Prompt tokens are priced at the audio-input rate — the text share of
        a call is a few dozen tokens against hundreds of audio tokens — so
        the figure is a slight over-estimate, never an under-estimate.
        """
        return (
            self.live_prompt_tokens * JUDGE_USD_PER_M_INPUT_TOKENS
            + self.live_output_tokens * JUDGE_USD_PER_M_OUTPUT_TOKENS
        ) / 1e6


def load_arena_prompts(
    specs: Sequence[str], *, language: str, seed: int
) -> list[TtsPrompt]:
    """Build the arena prompt set from ``path[:count]`` file specs.

    Prompt ids are ``{file stem}-{line number:03d}``, so the same line always
    maps to the same id: synthesized WAVs and cached judge verdicts survive
    re-runs and prompt-mix changes. Sampling within a file is seeded and the
    selection is re-sorted by line number, so a given (file, count, seed)
    triple is fully reproducible.

    Args:
        specs: Prompt files, each optionally suffixed ``:count`` to take a
            seeded sample of that size instead of every line.
        language: BCP-47 tag stamped on every prompt.
        seed: Sampling seed shared by every file in the mix.

    Returns:
        The mixed prompt list, in spec order then line order.

    Raises:
        ArenaError: If a spec is malformed, a file is missing or empty, or
            two files would collide on prompt ids.
    """
    prompts: list[TtsPrompt] = []
    seen_stems: set[str] = set()
    for spec in specs:
        path_part, _, count_part = spec.rpartition(":")
        if path_part and count_part.isdigit():
            path, count = Path(path_part), int(count_part)
        else:
            path, count = Path(spec), None
        if not path.is_file():
            raise ArenaError(f"prompt file not found: {path}")
        stem = path.stem
        if stem in seen_stems:
            raise ArenaError(f"duplicate prompt file stem {stem!r}: ids would collide")
        seen_stems.add(stem)

        lines = [
            (number, line.strip())
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            )
            if line.strip()
        ]
        if not lines:
            raise ArenaError(f"prompt file yielded no prompts: {path}")
        if count is not None and count < len(lines):
            rng = random.Random(f"{seed}:{stem}")
            lines = sorted(rng.sample(lines, count))
        prompts.extend(
            TtsPrompt(prompt_id=f"{stem}-{number:03d}", text=text, language=language)
            for number, text in lines
        )
    if not prompts:
        raise ArenaError("no prompt files given")
    return prompts


def load_system_audio(
    audio_dir: str | Path, system: str, prompt_id: str
) -> np.ndarray | None:
    """Load one saved synthesis as mono float at :data:`ARENA_SAMPLE_RATE`.

    Reads the ``{system}-batch-{prompt_id}.wav`` file written by
    :func:`audio_harness.runner.write_tts_results`; the WAV header carries
    the true sample rate, which loaded JSONL results do not.

    Returns:
        The samples, or ``None`` when the file is absent or unreadable —
        the caller records the gap instead of judging a partial pair.
    """
    decoded = read_audio_samples(Path(audio_dir) / f"{system}-batch-{prompt_id}.wav")
    if decoded is None:
        return None
    samples, rate = decoded
    if rate != ARENA_SAMPLE_RATE:
        samples = soxr.resample(samples, rate, ARENA_SAMPLE_RATE, quality="HQ")
    return samples.astype(np.float32)


def pair_wav_bytes(
    first: np.ndarray, second: np.ndarray, *, gap_s: float = PAIR_GAP_S
) -> bytes:
    """Concatenate two clips into the single WAV a judge call hears.

    The AudioJudge protocol judges one audio file holding both renditions —
    clip one, a pause, clip two — rather than two attachments, which is the
    presentation the paper validated.

    Args:
        first: Clip one, mono float at :data:`ARENA_SAMPLE_RATE`.
        second: Clip two, same format.
        gap_s: Silence between the clips.

    Returns:
        WAV bytes at :data:`ARENA_SAMPLE_RATE`.
    """
    gap = np.zeros(int(ARENA_SAMPLE_RATE * gap_s), dtype=np.float32)
    combined = np.concatenate([first, gap, second])
    return wrap_wav(_float_to_pcm16(combined), ARENA_SAMPLE_RATE)


def aspect_instruction(aspect: str) -> str:
    """Build the pinned judge instruction for one aspect.

    Raises:
        KeyError: If ``aspect`` is not one of :data:`ASPECTS`.
    """
    return (
        "You will hear one audio file containing two text-to-speech clips "
        "speaking the same text, separated by a short pause: clip one, then "
        f"clip two. {_ASPECT_FOCUS[aspect]} "
        "Answer with exactly one word: FIRST if clip one is better, SECOND "
        "if clip two is better, or TIE if you cannot hear a difference."
    )


_VERDICT_TOKEN = re.compile(r"\b(first|second|tie)\b")


def parse_verdict(text: str) -> str | None:
    """Extract the judged winner from a raw model reply.

    The first occurrence of ``first``/``second``/``tie`` (case-insensitive)
    decides, so a chatty reply like "The second clip sounds cleaner" still
    parses. A reply naming none of them returns ``None`` and is recorded as
    an error rather than guessed at.
    """
    match = _VERDICT_TOKEN.search(text.lower())
    return match.group(1) if match else None


def _gemini_judge(model: str) -> JudgeCall:
    """Build the live judge callable over the pinned Gemini audio-in model.

    Temperature 0 and a zero thinking budget keep the call as deterministic
    and as cheap as the API allows; verdict quality comes from the pairwise
    protocol, not from reasoning tokens.
    """
    client = genai.Client(api_key=require_env("GEMINI_API_KEY", "tts-arena"))
    config = genai_types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )

    async def judge(wav: bytes, instruction: str) -> JudgeReply:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                instruction,
            ],
            config=config,
        )
        usage = getattr(response, "usage_metadata", None)
        return JudgeReply(
            text=response.text or "",
            prompt_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        )

    return judge


def _cache_key(
    model: str, aspect: str, prompt_id: str, first: str, second: str, wav: bytes
) -> str:
    """Idempotency key for one judge call.

    Hashing the pair WAV ties the key to the actual audio judged: a
    re-synthesized clip, a different order, or a different judge model each
    produce a fresh key instead of a stale hit.
    """
    audio_sha = hashlib.sha256(wav).hexdigest()
    material = f"{model}|{aspect}|{prompt_id}|{first}|{second}|{audio_sha}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, Verdict]:
    """Load previously judged verdicts keyed by their cache key."""
    if not path.is_file():
        return {}
    cache: dict[str, Verdict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = orjson.loads(line)
        cache[record["key"]] = Verdict(
            aspect=record["aspect"],
            prompt_id=record["prompt_id"],
            first=record["first"],
            second=record["second"],
            winner=record["winner"],
            model=record["model"],
            cached=True,
            prompt_tokens=int(record.get("prompt_tokens", 0)),
            output_tokens=int(record.get("output_tokens", 0)),
            raw_text=record.get("raw_text", ""),
        )
    return cache


_RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 504})
_JUDGE_ATTEMPTS = 4
_RETRY_BASE_S = 2.0

JUDGE_TIMEOUT_S = 120.0
"""Deadline per judge attempt. Without one, a hung API call permanently
occupies a concurrency slot; eight hung calls deadlock the whole run — a
timed-out attempt instead retries and, failing that, records an error."""


async def run_arena(
    *,
    audio_dir: str | Path,
    systems: Sequence[str],
    prompts: Sequence[TtsPrompt],
    cache_path: str | Path,
    model: str = JUDGE_MODEL,
    judge: JudgeCall | None = None,
    concurrency: int = 8,
    call_timeout_s: float = JUDGE_TIMEOUT_S,
    on_progress: Callable[[int, int], None] | None = None,
) -> ArenaRun:
    """Judge every pair of systems over every prompt, order-counterbalanced.

    For each prompt and each unordered pair of systems, both presentation
    orders of the concatenated pair audio are judged once per aspect:
    ``C(n, 2) * len(prompts) * len(ASPECTS) * 2`` calls in total. Calls
    already answered in the cache are not repeated, so an interrupted run
    resumes for free.

    Args:
        audio_dir: Directory of ``{system}-batch-{prompt_id}.wav`` files.
        systems: TTS registry keys taking part.
        prompts: Arena prompt set.
        cache_path: JSONL judge-call cache, created if absent. Errors are
            never cached, so a transient failure is retried on the next run.
        model: Judge model; part of every cache key.
        judge: Injected judge callable for tests; ``None`` uses the live
            pinned Gemini judge.
        concurrency: Concurrent judge calls in flight.
        call_timeout_s: Deadline per judge attempt; a timed-out attempt is
            retried like any transient failure.
        on_progress: Optional ``(done, total)`` callback.

    Returns:
        The run record, including spend accounting.

    Raises:
        ArenaError: If fewer than two systems are given.
    """
    ordered = tuple(sorted(systems))
    if len(ordered) < 2:
        raise ArenaError(f"arena needs at least two systems, got {list(systems)}")

    run = ArenaRun(systems=ordered, model=model)
    if prompts:
        run.language = prompts[0].language

    audio: dict[tuple[str, str], np.ndarray] = {}
    for system in ordered:
        for prompt in prompts:
            samples = load_system_audio(audio_dir, system, prompt.prompt_id)
            if samples is None:
                run.missing_audio.append(f"{system}:{prompt.prompt_id}")
            else:
                audio[(system, prompt.prompt_id)] = samples

    jobs: list[tuple[str, str, str, str, bytes]] = []
    for prompt in prompts:
        for one, two in itertools.combinations(ordered, 2):
            if (one, prompt.prompt_id) not in audio:
                continue
            if (two, prompt.prompt_id) not in audio:
                continue
            forward = pair_wav_bytes(
                audio[(one, prompt.prompt_id)], audio[(two, prompt.prompt_id)]
            )
            backward = pair_wav_bytes(
                audio[(two, prompt.prompt_id)], audio[(one, prompt.prompt_id)]
            )
            for aspect in ASPECTS:
                jobs.append((aspect, prompt.prompt_id, one, two, forward))
                jobs.append((aspect, prompt.prompt_id, two, one, backward))

    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_file)
    judge_call = judge if judge is not None else _gemini_judge(model)
    limiter = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    done = 0

    async def one_job(
        aspect: str, prompt_id: str, first: str, second: str, wav: bytes
    ) -> Verdict:
        nonlocal done
        key = _cache_key(model, aspect, prompt_id, first, second, wav)
        hit = cache.get(key)
        if hit is not None:
            run.cached_calls += 1
            done += 1
            if on_progress:
                on_progress(done, len(jobs))
            return hit

        verdict = Verdict(
            aspect=aspect,
            prompt_id=prompt_id,
            first=first,
            second=second,
            winner="error",
            model=model,
        )
        instruction = aspect_instruction(aspect)
        async with limiter:
            for attempt in range(_JUDGE_ATTEMPTS):
                try:
                    reply = await asyncio.wait_for(
                        judge_call(wav, instruction), timeout=call_timeout_s
                    )
                except genai_errors.APIError as exc:
                    verdict.error = f"judge call failed: {exc}"
                    code = getattr(exc, "code", None)
                    if code not in _RETRYABLE_CODES:
                        break
                except (TimeoutError, OSError) as exc:
                    verdict.error = f"judge call failed: {exc!r}"
                else:
                    run.live_calls += 1
                    run.live_prompt_tokens += reply.prompt_tokens
                    run.live_output_tokens += reply.output_tokens
                    verdict.prompt_tokens = reply.prompt_tokens
                    verdict.output_tokens = reply.output_tokens
                    verdict.raw_text = reply.text
                    parsed = parse_verdict(reply.text)
                    if parsed is None:
                        verdict.error = f"unparseable reply: {reply.text!r}"
                    else:
                        verdict.winner = parsed
                        verdict.error = None
                    break
                if attempt < _JUDGE_ATTEMPTS - 1:
                    await asyncio.sleep(_RETRY_BASE_S * 2**attempt)

        if verdict.ok:
            async with write_lock:
                with cache_file.open("ab") as handle:
                    handle.write(
                        orjson.dumps(
                            {
                                "key": key,
                                "aspect": verdict.aspect,
                                "prompt_id": verdict.prompt_id,
                                "first": verdict.first,
                                "second": verdict.second,
                                "winner": verdict.winner,
                                "model": verdict.model,
                                "prompt_tokens": verdict.prompt_tokens,
                                "output_tokens": verdict.output_tokens,
                                "raw_text": verdict.raw_text,
                            }
                        )
                    )
                    handle.write(b"\n")
        else:
            run.error_calls += 1
        done += 1
        if on_progress:
            on_progress(done, len(jobs))
        return verdict

    run.verdicts = list(await asyncio.gather(*(one_job(*job) for job in jobs)))
    return run


def comparisons_from_verdicts(verdicts: Iterable[Verdict]) -> list[Comparison]:
    """Turn usable verdicts into Bradley-Terry observations.

    Both presentation orders feed the model — counterbalancing is what
    cancels position bias in aggregate — and errored verdicts are dropped
    rather than imputed.
    """
    weight = {"first": 1.0, "tie": 0.5, "second": 0.0}
    return [
        Comparison(
            first=v.first,
            second=v.second,
            first_wins=weight[v.winner],
            prompt_id=v.prompt_id,
        )
        for v in verdicts
        if v.ok
    ]


_BT_MAX_ITER = 1000
_BT_TOLERANCE = 1e-10


def bt_scores(
    comparisons: Iterable[Comparison], systems: Sequence[str]
) -> dict[str, float]:
    """Fit Bradley-Terry log-strengths by the Hunter (2004) MM algorithm.

    Ties contribute half a win to each side. Strengths are normalized to a
    geometric mean of one, so log-strengths are centred at zero and directly
    comparable across runs. A system that never wins is floored at
    :data:`STRENGTH_FLOOR` instead of collapsing to zero, which keeps
    bootstrap percentiles finite.

    Args:
        comparisons: Observations; systems outside ``systems`` are an error.
        systems: Every system to score, whether or not it appears.

    Returns:
        Mapping of system to log-strength.
    """
    index = {system: i for i, system in enumerate(systems)}
    n = len(systems)
    wins = np.zeros((n, n))
    for comparison in comparisons:
        i, j = index[comparison.first], index[comparison.second]
        wins[i, j] += comparison.first_wins
        wins[j, i] += 1.0 - comparison.first_wins

    games = wins + wins.T
    strengths = np.ones(n)
    for _ in range(_BT_MAX_ITER):
        totals = wins.sum(axis=1)
        denominators = np.zeros(n)
        for i in range(n):
            played = games[i] > 0
            if played.any():
                denominators[i] = (
                    games[i][played] / (strengths[i] + strengths[played])
                ).sum()
        updated = np.where(
            denominators > 0, totals / np.maximum(denominators, STRENGTH_FLOOR), 1.0
        )
        updated = np.maximum(updated, STRENGTH_FLOOR)
        updated /= np.exp(np.mean(np.log(updated)))
        if np.max(np.abs(np.log(updated) - np.log(strengths))) < _BT_TOLERANCE:
            strengths = updated
            break
        strengths = updated
    return {system: float(np.log(strengths[index[system]])) for system in systems}


def bootstrap_bt(
    verdicts: Sequence[Verdict],
    systems: Sequence[str],
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """95% cluster-bootstrap CIs on Bradley-Terry log-strengths.

    Resampling is over prompts, not individual verdicts: the six aspect/order
    verdicts of one prompt share the same audio and are correlated, so a
    per-verdict bootstrap would understate the interval.

    Args:
        verdicts: Arena verdicts; errors are excluded.
        systems: Every system to score.
        n_boot: Bootstrap resamples.
        seed: RNG seed, making the CIs reproducible.

    Returns:
        Mapping of system to ``(low, high)`` log-strength bounds.
    """
    by_prompt: dict[str, list[Comparison]] = {}
    for comparison in comparisons_from_verdicts(verdicts):
        by_prompt.setdefault(comparison.prompt_id, []).append(comparison)
    prompt_ids = sorted(by_prompt)
    if not prompt_ids:
        return {system: (0.0, 0.0) for system in systems}

    rng = random.Random(seed)
    samples: dict[str, list[float]] = {system: [] for system in systems}
    for _ in range(n_boot):
        chosen = rng.choices(prompt_ids, k=len(prompt_ids))
        resample = [c for prompt_id in chosen for c in by_prompt[prompt_id]]
        for system, score in bt_scores(resample, systems).items():
            samples[system].append(score)
    return {
        system: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        for system, values in samples.items()
    }


def bt_table(
    verdicts: Sequence[Verdict],
    systems: Sequence[str],
    *,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> list[BtScore]:
    """Point estimates plus bootstrap CIs, sorted strongest first."""
    comparisons = comparisons_from_verdicts(verdicts)
    points = bt_scores(comparisons, systems)
    intervals = bootstrap_bt(verdicts, systems, n_boot=n_boot, seed=seed)

    wins = dict.fromkeys(systems, 0.0)
    games = dict.fromkeys(systems, 0.0)
    for comparison in comparisons:
        wins[comparison.first] += comparison.first_wins
        wins[comparison.second] += 1.0 - comparison.first_wins
        games[comparison.first] += 1.0
        games[comparison.second] += 1.0

    table = [
        BtScore(
            system=system,
            log_strength=points[system],
            ci_low=intervals[system][0],
            ci_high=intervals[system][1],
            wins=wins[system],
            games=games[system],
        )
        for system in systems
    ]
    return sorted(table, key=lambda score: score.log_strength, reverse=True)


def ci_separated_pairs(scores: Sequence[BtScore]) -> list[tuple[str, str]]:
    """Pairs whose bootstrap CIs do not overlap, as ``(better, worse)``.

    These are the only pairs the human-panel decision rule tests: at n=4-6
    systems Spearman over full rankings quantizes, so separation — not rank
    correlation — is the pre-registered agreement test.
    """
    return [
        (a.system, b.system)
        for a, b in itertools.combinations(scores, 2)
        if a.ci_low > b.ci_high
    ] + [
        (b.system, a.system)
        for a, b in itertools.combinations(scores, 2)
        if b.ci_low > a.ci_high
    ]


def order_flip_stats(verdicts: Iterable[Verdict]) -> OrderFlipStats:
    """Compute criterion (iii) bookkeeping over every swap-group."""
    groups: dict[tuple[str, str, frozenset[str]], list[Verdict]] = {}
    for verdict in verdicts:
        key = (
            verdict.aspect,
            verdict.prompt_id,
            frozenset((verdict.first, verdict.second)),
        )
        groups.setdefault(key, []).append(verdict)

    judged = flips = dropped = 0
    for members in groups.values():
        usable = [v for v in members if v.ok]
        if len(usable) < 2:
            dropped += 1
            continue
        judged += 1
        winners = {v.winner_system for v in usable}
        if len(winners) > 1:
            flips += 1
    return OrderFlipStats(judged=judged, flips=flips, dropped=dropped)


def load_panel(path: str | Path) -> list[PanelVote]:
    """Load human-panel votes from their JSONL file.

    Args:
        path: Panel file; see :class:`PanelVote` for the record format.

    Returns:
        Every vote, in file order.

    Raises:
        ArenaError: If the file is missing or a record is malformed.
    """
    file = Path(path)
    if not file.is_file():
        raise ArenaError(f"panel file not found: {file}")
    votes: list[PanelVote] = []
    for number, line in enumerate(
        file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = orjson.loads(line)
        try:
            vote = PanelVote(
                rater=str(record["rater"]),
                prompt_id=str(record["prompt_id"]),
                first=str(record["first"]),
                second=str(record["second"]),
                winner=str(record["winner"]),
            )
        except KeyError as exc:
            raise ArenaError(f"{file}:{number}: missing panel field {exc}") from exc
        if vote.winner not in ("first", "second", "tie"):
            raise ArenaError(
                f"{file}:{number}: winner must be first/second/tie, got {vote.winner!r}"
            )
        votes.append(vote)
    return votes


def panel_size_ok(votes: Sequence[PanelVote]) -> bool:
    """Whether the panel meets the pre-registered minimum size."""
    per_rater: dict[str, int] = {}
    for vote in votes:
        per_rater[vote.rater] = per_rater.get(vote.rater, 0) + 1
    return len(per_rater) >= PANEL_MIN_RATERS and all(
        count >= PANEL_MIN_PAIRS_PER_RATER for count in per_rater.values()
    )


def _panel_item(vote: PanelVote) -> tuple[str, tuple[str, str]]:
    """Orientation-free identity of the pair a vote judged."""
    one, two = sorted((vote.first, vote.second))
    return vote.prompt_id, (one, two)


def _panel_category(vote: PanelVote) -> str:
    """Orientation-free outcome label: the winning system's key, or ``tie``."""
    return vote.winner_system or "tie"


def inter_rater_agreement(
    votes: Sequence[PanelVote],
) -> tuple[float | None, float | None]:
    """Mean pairwise percent agreement and Cohen's kappa across raters.

    Votes are compared on items (prompt, unordered pair) that two raters
    both judged, after normalizing away presentation order. Kappa's expected
    agreement uses each rater's own outcome distribution over the shared
    items.

    Returns:
        ``(percent agreement, mean pairwise kappa)``; either is ``None``
        when no rater pair shares an item (or kappa is undefined).
    """
    by_rater: dict[str, dict[tuple[str, tuple[str, str]], str]] = {}
    for vote in votes:
        by_rater.setdefault(vote.rater, {})[_panel_item(vote)] = _panel_category(vote)

    agreements: list[float] = []
    kappas: list[float] = []
    for rater_a, rater_b in itertools.combinations(sorted(by_rater), 2):
        shared = sorted(set(by_rater[rater_a]) & set(by_rater[rater_b]))
        if not shared:
            continue
        labels_a = [by_rater[rater_a][item] for item in shared]
        labels_b = [by_rater[rater_b][item] for item in shared]
        observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / len(
            shared
        )
        agreements.append(observed)

        categories = sorted(set(labels_a) | set(labels_b))
        expected = sum(
            (labels_a.count(c) / len(shared)) * (labels_b.count(c) / len(shared))
            for c in categories
        )
        if expected < 1.0:
            kappas.append((observed - expected) / (1.0 - expected))

    percent = sum(agreements) / len(agreements) if agreements else None
    kappa = sum(kappas) / len(kappas) if kappas else None
    return percent, kappa


def human_bt(votes: Sequence[PanelVote], systems: Sequence[str]) -> dict[str, float]:
    """Bradley-Terry log-strengths from the pooled human-panel votes."""
    weight = {"first": 1.0, "tie": 0.5, "second": 0.0}
    comparisons = [
        Comparison(
            first=vote.first,
            second=vote.second,
            first_wins=weight[vote.winner],
            prompt_id=vote.prompt_id,
        )
        for vote in votes
    ]
    return bt_scores(comparisons, systems)


def _rank(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman_rho(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman rank correlation, or ``None`` when it is undefined."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ranks_a = np.array(_rank(a))
    ranks_b = np.array(_rank(b))
    if ranks_a.std() == 0 or ranks_b.std() == 0:
        return None
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


_EXACT_PERMUTATION_MAX_N = 8


def exact_permutation_p(
    a: Sequence[float], b: Sequence[float], *, seed: int = 0
) -> float | None:
    """One-sided permutation p for the observed Spearman rho.

    Exhaustive over every permutation up to n=:data:`_EXACT_PERMUTATION_MAX_N`
    systems (8! = 40320); beyond that a seeded 10000-permutation Monte Carlo
    approximation is used and callers should label the p accordingly.
    """
    observed = spearman_rho(a, b)
    if observed is None:
        return None
    if len(a) <= _EXACT_PERMUTATION_MAX_N:
        permutations: Iterable[tuple[float, ...]] = itertools.permutations(b)
        total = 0
        at_least = 0
        for permuted in permutations:
            rho = spearman_rho(a, permuted)
            if rho is None:
                continue
            total += 1
            if rho >= observed - 1e-12:
                at_least += 1
        return at_least / total if total else None
    rng = random.Random(seed)
    shuffled = list(b)
    at_least = 0
    trials = 10000
    for _ in range(trials):
        rng.shuffle(shuffled)
        rho = spearman_rho(a, shuffled)
        if rho is not None and rho >= observed - 1e-12:
            at_least += 1
    return at_least / trials


def evaluate_gate(
    scores: Sequence[BtScore],
    flips: OrderFlipStats,
    panel: Sequence[PanelVote] | None,
    *,
    second_family_scores: dict[str, float] | None = None,
) -> GateReport:
    """Evaluate the three-criterion validity gate, recording all, faking none.

    Args:
        scores: Judge Bradley-Terry table with bootstrap CIs.
        flips: Order-flip bookkeeping from :func:`order_flip_stats`.
        panel: Human-panel votes, or ``None`` while no panel exists.
        second_family_scores: Bradley-Terry log-strengths from a judge of a
            *different* model family, or ``None`` while none is available —
            in which case criterion (ii) is reported uncomputable.

    Returns:
        The gate report; ranked status requires every criterion to pass.
    """
    systems = [score.system for score in scores]
    separated = ci_separated_pairs(scores)

    if panel is None:
        human_criterion = Criterion(
            name="human-panel agreement",
            status="pending",
            detail=(
                "no human panel recorded. Decision rule once collected "
                f"(>= {PANEL_MIN_RATERS} raters x "
                f"{PANEL_MIN_PAIRS_PER_RATER} pairs): no CI-separated system "
                "pair may be ranked discordantly vs the human Bradley-Terry "
                f"ranking; currently {len(separated)} CI-separated pair(s) "
                "await that check."
            ),
        )
    else:
        human_scores = human_bt(panel, systems)
        discordant = [
            (better, worse)
            for better, worse in separated
            if human_scores[better] < human_scores[worse]
        ]
        percent, kappa = inter_rater_agreement(panel)
        judge_vector = [score.log_strength for score in scores]
        human_vector = [human_scores[score.system] for score in scores]
        rho = spearman_rho(judge_vector, human_vector)
        p_value = exact_permutation_p(judge_vector, human_vector)
        raters = len({vote.rater for vote in panel})
        descriptive = (
            f"{raters} rater(s), {len(panel)} votes; inter-rater agreement "
            f"{_fmt(percent)} (kappa {_fmt(kappa)}); descriptive Spearman "
            f"rho {_fmt(rho)} (exact permutation p {_fmt(p_value)}); "
            f"{len(separated)} CI-separated pair(s), "
            f"{len(discordant)} discordant vs human BT"
        )
        if not panel_size_ok(panel):
            status, prefix = (
                "pending",
                "panel below the pre-registered size, preview only: ",
            )
        elif discordant:
            pairs = ", ".join(f"{a} vs {b}" for a, b in discordant)
            status, prefix = "fail", f"discordant CI-separated pair(s) [{pairs}]: "
        else:
            status, prefix = "pass", ""
        human_criterion = Criterion(
            name="human-panel agreement", status=status, detail=prefix + descriptive
        )

    if second_family_scores is None:
        family_criterion = Criterion(
            name="cross-family judge agreement",
            status="uncomputable",
            detail=(
                f"only the {JUDGE_FAMILY} judge family is available (no "
                "OpenAI audio key), so agreement rho >= "
                f"{CROSS_FAMILY_RHO_GATE} between judge families cannot be "
                "computed. Reported as uncomputable rather than faked; the "
                "gate cannot pass until a second family is judged."
            ),
        )
    else:
        judge_vector = [score.log_strength for score in scores]
        other_vector = [second_family_scores[score.system] for score in scores]
        rho = spearman_rho(judge_vector, other_vector)
        ok = rho is not None and rho >= CROSS_FAMILY_RHO_GATE
        family_criterion = Criterion(
            name="cross-family judge agreement",
            status="pass" if ok else "fail",
            detail=(
                f"rho {_fmt(rho)} vs second judge family "
                f"(gate >= {CROSS_FAMILY_RHO_GATE})"
            ),
        )

    rate = flips.rate
    if rate is None:
        flip_criterion = Criterion(
            name="order-flip rate",
            status="pending",
            detail="no swap-groups judged yet",
        )
    else:
        flip_criterion = Criterion(
            name="order-flip rate",
            status="pass" if rate <= ORDER_FLIP_GATE else "fail",
            detail=(
                f"{flips.flips}/{flips.judged} swap-groups flipped = "
                f"{rate:.1%} (gate <= {ORDER_FLIP_GATE:.0%}; "
                f"{flips.dropped} dropped on errors)"
            ),
        )

    return GateReport(
        criteria=(human_criterion, family_criterion, flip_criterion),
        n_systems=len(systems),
    )


def _fmt(value: float | None) -> str:
    """Format an optional statistic without inventing precision."""
    return "n/a" if value is None else f"{value:.3f}"


def render_arena_markdown(
    run: ArenaRun,
    scores: Sequence[BtScore],
    gate: GateReport,
    *,
    notes: Sequence[str] = (),
) -> str:
    """Render the arena report section.

    System-level only, as the AudioJudge protocol requires: no per-clip
    numbers appear, and the gate label leads so the ranking is never read
    without its validity status.
    """
    lines = [
        f"## TTS arena ({run.language}) - {gate.label}",
        "",
        f"AudioJudge pairwise protocol (arXiv:2507.12705), judge `{run.model}`, "
        f"aspects {', '.join(ASPECTS)}, order-counterbalanced, system-level "
        "Bradley-Terry with 95% cluster-bootstrap CIs (example-level judging "
        "is near chance and is not reported).",
        "",
        "| System | BT log-strength | 95% CI | Wins | Games |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {score.system} | {score.log_strength:+.3f} "
        f"| [{score.ci_low:+.3f}, {score.ci_high:+.3f}] "
        f"| {score.wins:.1f} | {score.games:.0f} |"
        for score in scores
    )
    lines += [
        "",
        "| Gate criterion | Status | Detail |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {criterion.name} | {criterion.status} | {criterion.detail} |"
        for criterion in gate.criteria
    )
    lines += [
        "",
        f"Calls: {run.live_calls} live + {run.cached_calls} cached, "
        f"{run.error_calls} errored; est. judge spend this run "
        f"${run.est_usd:.2f} (rates checked {JUDGE_PRICING_CHECKED}).",
    ]
    if run.missing_audio:
        lines.append(
            f"Missing audio skipped {len(run.missing_audio)} system:prompt "
            f"slot(s): {', '.join(sorted(run.missing_audio))}."
        )
    lines.extend(f"Note: {note}" for note in notes)
    return "\n".join(lines)


def write_arena_outputs(
    output_dir: str | Path,
    run: ArenaRun,
    scores: Sequence[BtScore],
    flips: OrderFlipStats,
    gate: GateReport,
    *,
    notes: Sequence[str] = (),
) -> tuple[Path, Path, Path]:
    """Persist verdicts, gate metrics and the report section.

    The summary carries the gate metrics for the run's language (AC8), so a
    later report merge can render gate status next to every judged number
    without re-deriving it.

    Returns:
        Paths of ``arena-results.jsonl``, ``arena-summary.json`` and
        ``arena-report.md``.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    results_path = directory / "arena-results.jsonl"
    with results_path.open("wb") as handle:
        for verdict in run.verdicts:
            handle.write(
                orjson.dumps(
                    {
                        "aspect": verdict.aspect,
                        "prompt_id": verdict.prompt_id,
                        "first": verdict.first,
                        "second": verdict.second,
                        "winner": verdict.winner,
                        "model": verdict.model,
                        "cached": verdict.cached,
                        "prompt_tokens": verdict.prompt_tokens,
                        "output_tokens": verdict.output_tokens,
                        "raw_text": verdict.raw_text,
                        "error": verdict.error,
                    }
                )
            )
            handle.write(b"\n")

    summary_path = directory / "arena-summary.json"
    summary_path.write_bytes(
        orjson.dumps(
            {
                "language": run.language,
                "model": run.model,
                "systems": list(run.systems),
                "gate": {
                    "label": gate.label,
                    "n_systems": gate.n_systems,
                    "criteria": [
                        {
                            "name": criterion.name,
                            "status": criterion.status,
                            "detail": criterion.detail,
                        }
                        for criterion in gate.criteria
                    ],
                },
                "bt": [
                    {
                        "system": score.system,
                        "log_strength": score.log_strength,
                        "ci_low": score.ci_low,
                        "ci_high": score.ci_high,
                        "wins": score.wins,
                        "games": score.games,
                    }
                    for score in scores
                ],
                "order_flip": {
                    "judged": flips.judged,
                    "flips": flips.flips,
                    "dropped": flips.dropped,
                    "rate": flips.rate,
                },
                "calls": {
                    "live": run.live_calls,
                    "cached": run.cached_calls,
                    "errors": run.error_calls,
                },
                "tokens": {
                    "prompt": run.live_prompt_tokens,
                    "output": run.live_output_tokens,
                },
                "est_usd": run.est_usd,
                "missing_audio": sorted(run.missing_audio),
                "notes": list(notes),
            },
            option=orjson.OPT_INDENT_2,
        )
    )

    report_path = directory / "arena-report.md"
    report_path.write_text(
        render_arena_markdown(run, scores, gate, notes=notes) + "\n",
        encoding="utf-8",
    )
    return results_path, summary_path, report_path
