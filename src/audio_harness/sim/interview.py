"""Simulated-interview end-to-end lane (plan P4 step 18, lane E3).

The pipeline: an interviewer LLM conducts a scripted interview scenario, a
persona LLM answers spontaneously with program-sampled facts woven in, the
answers are voiced by a pinned out-of-suite OSS voice (Kokoro), degraded
through the pinned telephony condition, streamed to every candidate STT
vendor, and scored by deterministic field extraction against the sampled
ground truth. Personas are sampled fresh every run, so no vendor can have
memorized the material — contamination resistance by construction.

Three design points keep the score honest:

* **Ground truth is sampled, not generated.** Slot values (names, phone
  numbers, dates, amounts, codes) come from a seeded sampler, and a turn only
  becomes scorable after the persona's *text* passes the same deterministic
  extractor the transcripts are scored with. An LLM that garbles its own
  phone number produces an unscorable turn, never a vendor penalty.
* **Extraction is deterministic and schema-aware.** The scorer knows which
  entity class each turn carries and matches candidates over the shared
  per-language normalization (``normalize.py``), so "five five five one two
  three four five six seven", "(555) 123-4567" and "5551234567" all agree.
  No LLM touches the scoring path.
* **The lane ships gated.** Its vendor ranking is compared against the
  pre-registered real-corpus composite — mean within-language rank over
  {entity-WER, finalize p50} on the canonical merge — and rankings are only
  trusted at Spearman rho ≥ 0.8 (average-rank ties, exact permutation p). A
  failing gate triggers a written divergence analysis before any verdict.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import permutations
import math
from pathlib import Path
import secrets
from typing import Any

import numpy as np
import orjson
import soxr
import yaml

from audio_harness import runner, synthetic
from audio_harness.autotag import autotag_reference
from audio_harness.config import STT_PRICING, BenchmarkConfig, ConfigError, require_env
from audio_harness.entities import ENTITY_CLASSES
from audio_harness.metrics import percentile, summarize
from audio_harness.normalize import normalizer_for
from audio_harness.snr import TELEPHONY_RATE
from audio_harness.types import AudioClip, SttResult


GEMINI_MODEL = "gemini-2.5-flash"
"""Pinned interviewer/persona model. Overridable per config, but the gate is
only meaningful against a pinned generator."""

GEMINI_FLASH_USD_PER_M_INPUT = 0.30
GEMINI_FLASH_USD_PER_M_OUTPUT = 2.50
"""Gemini 2.5 Flash list prices (USD per million tokens), checked 2026-08-06.
Pricing is data that rots; re-verify before quoting a spend figure."""

KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
"""Pinned model revision (2025-04-10). Kokoro is deliberately out-of-suite:
no candidate STT vendor ships it, so no lane is scored on its own family's
voice (plan principle 2)."""

KOKORO_SAMPLE_RATE = 24000
KOKORO_VOICES = ("af_heart", "am_michael", "bf_emma", "bm_george")
"""Pinned voice roster, rotated per persona so the lane hears more than one
speaker. All four exist at :data:`KOKORO_REVISION`."""

STT_SAMPLE_RATE = 16000

DEGRADATION = "tel8k"
"""The pinned degradation condition: the P2 telephony round-trip
(16 k → 8 k → 16 k, ``snr.py``). Chosen over babble mixing because it is
deterministic and needs no MUSAN download; the mixer remains available for a
future babble condition once MUSAN is fetched."""

_EST_INTERVIEWER_TOKENS = (260, 40)
_EST_PERSONA_TOKENS = (300, 120)
_EST_RETRY_FACTOR = 1.3
"""Rough per-call token shapes for the spend estimate; the actual run records
real usage from the API's usage metadata."""

LlmFn = Callable[[str, str, float], Awaitable[str]]
"""``(system_instruction, prompt, temperature) -> text``."""

TtsFn = Callable[[str, str], np.ndarray]
"""``(text, voice) -> float32 mono samples at 24 kHz``."""


# --------------------------------------------------------------------------
# Scenario specification
# --------------------------------------------------------------------------

SLOT_KINDS = ("person", "company", "phone", "small_count", "amount", "date", "code")
"""Value samplers a scenario slot may name."""

_KIND_ENTITY = {
    "person": "name",
    "company": "name",
    "phone": "number",
    "small_count": "number",
    "amount": "currency",
    "date": "date",
    "code": "id",
}


@dataclass(slots=True, frozen=True)
class SlotSpec:
    """One information slot an interviewer question elicits.

    Attributes:
        name: Stable slot identifier within its scenario.
        kind: Which sampler produces the ground-truth value.
        entity: Entity class (``entities.py`` vocabulary) the value belongs
            to; derived from ``kind`` unless the scenario overrides it.
        question: Scripted question text the interviewer LLM renders.
    """

    name: str
    kind: str
    entity: str
    question: str


@dataclass(slots=True, frozen=True)
class Scenario:
    """One interview scenario: an ordered list of slot-bearing questions."""

    scenario_id: str
    language: str
    description: str
    slots: tuple[SlotSpec, ...]


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load and validate a scenario spec YAML.

    Args:
        path: YAML document with a top-level ``scenarios`` list.

    Returns:
        The parsed scenarios, in file order.

    Raises:
        ConfigError: If the file is missing or a scenario is malformed.
    """
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"scenario spec not found: {file}")
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    entries = (raw or {}).get("scenarios")
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"scenario spec has no scenarios list: {file}")

    scenarios: list[Scenario] = []
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("id"):
            raise ConfigError(f"scenario entry needs an id: {entry!r}")
        slots: list[SlotSpec] = []
        for slot in entry.get("slots") or []:
            kind = str(slot.get("kind", ""))
            if kind not in SLOT_KINDS:
                raise ConfigError(
                    f"scenario {entry['id']}: unknown slot kind {kind!r}; expected one of {', '.join(SLOT_KINDS)}"
                )
            entity = str(slot.get("entity") or _KIND_ENTITY[kind])
            if entity not in ENTITY_CLASSES:
                raise ConfigError(f"scenario {entry['id']}: unknown entity class {entity!r}")
            if not slot.get("name") or not slot.get("question"):
                raise ConfigError(f"scenario {entry['id']}: slot needs name and question: {slot!r}")
            slots.append(
                SlotSpec(
                    name=str(slot["name"]),
                    kind=kind,
                    entity=entity,
                    question=str(slot["question"]),
                )
            )
        if not slots:
            raise ConfigError(f"scenario {entry['id']} has no slots")
        scenarios.append(
            Scenario(
                scenario_id=str(entry["id"]),
                language=str(entry.get("language", "en-US")),
                description=str(entry.get("description", "")),
                slots=tuple(slots),
            )
        )
    return scenarios


# --------------------------------------------------------------------------
# Persona sampling (the ground truth)
# --------------------------------------------------------------------------

_FIRST_NAMES = (
    "Alice",
    "Brandon",
    "Carla",
    "Derek",
    "Elena",
    "Felix",
    "Grace",
    "Hector",
    "Irene",
    "Jonas",
    "Katrina",
    "Leon",
    "Miriam",
    "Nathan",
    "Olivia",
    "Patrick",
    "Renata",
    "Simon",
    "Teresa",
    "Victor",
)
_LAST_NAMES = (
    "Anderson",
    "Bennett",
    "Calloway",
    "Dawson",
    "Ellison",
    "Foster",
    "Griffin",
    "Hargrove",
    "Ibarra",
    "Jennings",
    "Kowalski",
    "Lambert",
    "Mercado",
    "Norwood",
    "Ortega",
    "Pearson",
    "Ramsey",
    "Sandoval",
    "Thornton",
    "Whitfield",
)
_COMPANY_HEADS = (
    "Brightwater",
    "Northgate",
    "Silvermont",
    "Crestline",
    "Harborview",
    "Stonefield",
    "Blueridge",
    "Fairhaven",
    "Westbrook",
    "Cedarline",
)
_COMPANY_TAILS = (
    "Systems",
    "Analytics",
    "Logistics",
    "Consulting",
    "Dynamics",
    "Industries",
    "Solutions",
    "Partners",
)
_CODE_LETTERS = "BCDFGHJKLMNPRSTVWXZ"
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_DIGIT_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)


@dataclass(slots=True, frozen=True)
class SlotValue:
    """One sampled ground-truth value in its three faces.

    Attributes:
        canonical: Normalized comparable form the scorer matches on
            (e.g. ``5551230142``, ``march 14``, ``95000 dollars``).
        written: Conventional written form shown to the LLM.
        spoken: Fully spelled-out spoken form the persona is asked to voice.
    """

    canonical: str
    written: str
    spoken: str


@dataclass(slots=True, frozen=True)
class Persona:
    """One sampled interviewee with a value for every scenario slot."""

    persona_id: str
    name: str
    values: dict[str, SlotValue]


def _spell_digits(digits: str, group: int = 0) -> str:
    """Spell a digit string as words, optionally comma-grouped."""
    words = [_DIGIT_WORDS[int(ch)] for ch in digits]
    if group <= 0:
        return " ".join(words)
    parts = [" ".join(words[i : i + group]) for i in range(0, len(words), group)]
    return ", ".join(parts)


def canonical_value(entity: str, written: str, language: str) -> str:
    """Fold a written value to the form the extractor compares on.

    The fold is the language's scoring normalizer plus a per-class squash, so
    the truth and every transcript candidate pass through identical machinery.

    Args:
        entity: Entity class of the value.
        written: Conventional written form.
        language: BCP-47 tag driving normalization.

    Returns:
        The canonical comparison form.
    """
    normalized = normalizer_for(language)(written)
    if entity == "number":
        return "".join(ch for ch in normalized if ch.isdigit())
    if entity == "id":
        return "".join(ch for ch in normalized if ch.isalnum())
    return " ".join(normalized.split())


def _sample_value(kind: str, rng: np.random.Generator, language: str) -> SlotValue:
    """Draw one ground-truth value for a slot kind."""
    if kind == "person":
        written = (
            f"{_FIRST_NAMES[int(rng.integers(len(_FIRST_NAMES)))]} {_LAST_NAMES[int(rng.integers(len(_LAST_NAMES)))]}"
        )
        return SlotValue(canonical_value("name", written, language), written, written)
    if kind == "company":
        written = (
            f"{_COMPANY_HEADS[int(rng.integers(len(_COMPANY_HEADS)))]} "
            f"{_COMPANY_TAILS[int(rng.integers(len(_COMPANY_TAILS)))]}"
        )
        return SlotValue(canonical_value("name", written, language), written, written)
    if kind == "phone":
        area = f"{rng.integers(2, 10)}{rng.integers(10, 100)}"
        line = f"01{rng.integers(10, 100)}"
        digits = f"{area}555{line}"
        written = f"{area}-555-{line}"
        return SlotValue(
            canonical_value("number", written, language),
            written,
            _spell_digits(digits, group=0),
        )
    if kind == "small_count":
        count = int(rng.integers(2, 31))
        return SlotValue(str(count), str(count), _number_words(count))
    if kind == "amount":
        amount = int(rng.integers(24, 190)) * 1000
        written = f"${amount:,}"
        return SlotValue(
            canonical_value("currency", written, language),
            written,
            f"{_number_words(amount)} dollars",
        )
    if kind == "date":
        # Days stay in 1..20: the en ITN tables fold single-word ordinals
        # ("fourteenth" → 14) but split compound ones ("twenty sixth" →
        # "20 6"), which would penalize a vendor for transcribing verbatim.
        # Follow-up: compound-ordinal folding belongs in normalize.py.
        month = _MONTHS[int(rng.integers(12))]
        day = int(rng.integers(1, 21))
        written = f"{month} {day}"
        return SlotValue(
            canonical_value("date", written, language),
            written,
            f"{month} {_ORDINALS[day]}",
        )
    if kind == "code":
        letters = [_CODE_LETTERS[int(rng.integers(len(_CODE_LETTERS)))] for _ in range(2)]
        digits = f"{rng.integers(1000, 10000)}"
        tail = _CODE_LETTERS[int(rng.integers(len(_CODE_LETTERS)))]
        written = f"{letters[0]}{letters[1]}{digits}{tail}"
        spoken = f"{letters[0]} {letters[1]} {_spell_digits(digits)} {tail}"
        return SlotValue(canonical_value("id", written, language), written, spoken)
    raise ConfigError(f"unknown slot kind {kind!r}")


_UNITS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_ORDINALS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
}
"""Single-word day ordinals only — see the day-range note in the sampler."""


def _number_words(value: int) -> str:
    """Spell a non-negative integer below one million as English words."""
    if value < 20:
        return _UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        return _TENS[tens] + (f" {_UNITS[unit]}" if unit else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        head = f"{_UNITS[hundreds]} hundred"
        return head + (f" {_number_words(rest)}" if rest else "")
    thousands, rest = divmod(value, 1000)
    head = f"{_number_words(thousands)} thousand"
    return head + (f" {_number_words(rest)}" if rest else "")


def sample_personas(scenario: Scenario, count: int, rng: np.random.Generator) -> list[Persona]:
    """Sample ``count`` fresh personas for one scenario.

    Every slot gets a value; slots of kind ``person`` reuse the persona's own
    sampled name so the identity stays coherent across the interview.

    Args:
        scenario: Scenario whose slots need values.
        count: Personas to sample.
        rng: Seeded generator; fresh runs pass a fresh seed.

    Returns:
        The sampled personas.
    """
    personas: list[Persona] = []
    for index in range(count):
        identity = _sample_value("person", rng, scenario.language)
        values: dict[str, SlotValue] = {}
        for slot in scenario.slots:
            if slot.kind == "person":
                values[slot.name] = identity
            else:
                values[slot.name] = _sample_value(slot.kind, rng, scenario.language)
        personas.append(
            Persona(
                persona_id=f"{scenario.scenario_id}-p{index:02d}",
                name=identity.written,
                values=values,
            )
        )
    return personas


# --------------------------------------------------------------------------
# Deterministic field extraction
# --------------------------------------------------------------------------

_CURRENCY_UNITS = {
    "dollars": "dollars",
    "dollar": "dollars",
    "euros": "euros",
    "euro": "euros",
    "pounds": "pounds",
    "pound": "pounds",
    "yen": "yen",
}
"""Currency words folded to the plural form the normalizer's symbol
expansion emits (``$95,000`` → ``95000 dollars``)."""

_MONTH_WORDS = tuple(month.lower() for month in _MONTHS)


def _digit_run_candidates(tokens: list[str]) -> set[str]:
    """All contiguous digit-token concatenations, e.g. ``555 1234`` → 5551234.

    Vendors split the same digits differently ("5551230142" vs "555 123
    0142"), so every contiguous sub-run of a maximal digit run is a
    candidate; concatenation happens at token granularity only, never inside
    a token, so truth ``12`` can never match a transcript ``125``.
    """
    candidates: set[str] = set()
    run: list[str] = []
    for token in [*tokens, ""]:
        if token.isdigit():
            run.append(token)
            continue
        for start in range(len(run)):
            candidates.update("".join(run[start:stop]) for stop in range(start + 1, len(run) + 1))
        run = []
    return candidates


def _alnum_run_candidates(tokens: list[str]) -> set[str]:
    """ID candidates: squashed sub-runs of alphanumeric-ish token runs.

    "the code is x k 4792 b" tokenizes into a run that includes the stray
    "is"; enumerating sub-runs means the true ``xk4792b`` is still among the
    candidates without any stop-word list. Only mixed letter+digit squashes
    of length ≥ 3 qualify, so ordinary words never become IDs.
    """
    candidates: set[str] = set()
    run: list[str] = []
    for token in [*tokens, "…"]:
        if token.isalnum() and len(token) <= 8:
            run.append(token)
            continue
        for start in range(len(run)):
            for stop in range(start + 1, min(start + 9, len(run) + 1)):
                squashed = "".join(run[start:stop])
                if len(squashed) >= 3 and any(ch.isdigit() for ch in squashed) and any(ch.isalpha() for ch in squashed):
                    candidates.add(squashed)
        run = []
    return candidates


def _currency_candidates(tokens: list[str]) -> set[str]:
    """Amount candidates: digit runs immediately before a currency unit."""
    candidates: set[str] = set()
    for index, token in enumerate(tokens):
        unit = _CURRENCY_UNITS.get(token)
        if unit is None:
            continue
        digits: list[str] = []
        cursor = index - 1
        while cursor >= 0 and tokens[cursor].isdigit():
            digits.insert(0, tokens[cursor])
            cursor -= 1
        candidates.update(f"{''.join(digits[start:])} {unit}" for start in range(len(digits)))
    return candidates


def _date_candidates(tokens: list[str]) -> set[str]:
    """Date candidates in canonical ``month day`` order, both spoken orders."""
    candidates: set[str] = set()
    for index, token in enumerate(tokens):
        if token not in _MONTH_WORDS:
            continue
        following = tokens[index + 1 : index + 3]
        for offset, nxt in enumerate(following):
            if nxt.isdigit() and len(nxt) <= 2:
                candidates.add(f"{token} {int(nxt)}")
                break
            if offset == 0 and nxt == "the":
                continue
            break
        for prev_index in (index - 1, index - 2):
            if prev_index < 0:
                continue
            prev = tokens[prev_index]
            if prev.isdigit() and len(prev) <= 2:
                candidates.add(f"{token} {int(prev)}")
                break
    return candidates


def extract_slot(transcript: str, truth: SlotValue, entity: str, language: str) -> str | None:
    """Extract one slot value from a transcript, deterministically.

    The extractor is schema-aware: it knows the entity class it is looking
    for, generates that class's candidate set from the normalized transcript,
    and returns the candidate matching the ground truth — the behaviour of a
    field-extraction stage that validates against a form schema. No LLM.

    Args:
        transcript: Provider transcript, unnormalized.
        truth: Ground-truth value being sought.
        entity: Entity class of the slot.
        language: BCP-47 tag driving normalization.

    Returns:
        The extracted canonical value on success, else ``None``.
    """
    normalized = normalizer_for(language)(transcript)
    tokens = normalized.split()
    if entity == "number":
        candidates = _digit_run_candidates(tokens)
    elif entity == "id":
        candidates = _alnum_run_candidates(tokens)
    elif entity == "currency":
        candidates = _currency_candidates(tokens)
    elif entity == "date":
        candidates = _date_candidates(tokens)
    elif entity == "name":
        width = len(truth.canonical.split())
        candidates = {" ".join(tokens[i : i + width]) for i in range(max(0, len(tokens) - width + 1))}
    else:
        raise ConfigError(f"unknown entity class {entity!r}")
    return truth.canonical if truth.canonical in candidates else None


# --------------------------------------------------------------------------
# Dialogue generation
# --------------------------------------------------------------------------

_INTERVIEWER_SYSTEM = (
    "You are a professional interviewer conducting a phone interview: "
    "{description}. Render the scripted question as one short, natural "
    "spoken sentence that fits the conversation so far, without changing "
    "what is being asked. Output only the words you speak."
)

_PERSONA_SYSTEM = (
    "You are {name}, answering a live phone interview: {description}. "
    "Answer in the first person, spontaneously, in one to three short "
    "sentences. Mild natural disfluencies (um, uh, brief false starts) are "
    "welcome. Write every number, date, amount and code in words, exactly "
    "as you would say it aloud — never as digits or symbols."
)

_FACT_TEMPLATES = {
    "person": "your full name: {spoken}",
    "company": "the company name: {spoken}",
    "phone": "your phone number, spoken digit by digit: {spoken}",
    "small_count": "the number: {spoken}",
    "amount": "the amount: {spoken}",
    "date": "the date: {spoken}",
    "code": "the code, spoken character by character: {spoken}",
}


@dataclass(slots=True)
class Turn:
    """One question/answer exchange and its scoring material.

    Attributes:
        scenario_id: Scenario the turn belongs to.
        persona_id: Persona answering.
        index: Zero-based position within the interview.
        slot: Slot the question elicits.
        truth: Sampled ground-truth value.
        question: Interviewer-LLM rendering of the scripted question.
        answer: Persona-LLM answer text (the TTS script).
        verified: Whether the answer text passed deterministic extraction —
            only verified turns are synthesized and scored.
        attempts: Persona generations it took to verify.
        voice: Kokoro voice the answer is synthesized with.
        clip_id: Identifier joining the turn to its STT results.
    """

    scenario_id: str
    persona_id: str
    index: int
    slot: SlotSpec
    truth: SlotValue
    question: str = ""
    answer: str = ""
    verified: bool = False
    attempts: int = 0
    voice: str = ""
    clip_id: str = ""


async def conduct_interview(
    scenario: Scenario,
    persona: Persona,
    llm: LlmFn,
    *,
    voice: str,
    max_attempts: int = 3,
) -> list[Turn]:
    """Run one full interview dialogue through the LLM pair.

    Args:
        scenario: Scenario being conducted.
        persona: Persona answering, with its sampled ground truth.
        llm: Text-generation callable (pinned Gemini in production).
        voice: Kokoro voice assigned to this persona.
        max_attempts: Persona generations before a turn is left unscorable.

    Returns:
        One turn per scenario slot, in order; unverified turns carry their
        last answer and ``verified=False``.
    """
    history: list[str] = []
    turns: list[Turn] = []
    interviewer_system = _INTERVIEWER_SYSTEM.format(description=scenario.description or scenario.scenario_id)
    persona_system = _PERSONA_SYSTEM.format(
        name=persona.name,
        description=scenario.description or scenario.scenario_id,
    )

    for index, slot in enumerate(scenario.slots):
        truth = persona.values[slot.name]
        turn = Turn(
            scenario_id=scenario.scenario_id,
            persona_id=persona.persona_id,
            index=index,
            slot=slot,
            truth=truth,
            voice=voice,
            clip_id=f"sim-{persona.persona_id}-t{index:02d}-{slot.name}",
        )

        context = "\n".join(history[-6:]) or "(the call has just started)"
        turn.question = (
            await llm(
                interviewer_system,
                f"Conversation so far:\n{context}\n\nScripted question to ask next: {slot.question}",
                0.3,
            )
        ).strip() or slot.question

        fact = _FACT_TEMPLATES[slot.kind].format(spoken=truth.spoken)
        prompt = f"Interviewer asks: {turn.question}\n\nYou must clearly state {fact}\nAnswer naturally."
        for attempt in range(1, max_attempts + 1):
            turn.attempts = attempt
            answer = (await llm(persona_system, prompt, 1.0)).strip()
            turn.answer = " ".join(answer.split())
            if turn.answer and extract_slot(turn.answer, truth, slot.entity, scenario.language) is not None:
                turn.verified = True
                break
            prompt = (
                f"Interviewer asks: {turn.question}\n\n"
                f"You must clearly state {fact}\n"
                f"Your previous answer did not state the value clearly "
                f"enough. Say it plainly this time, word for word: "
                f"{truth.spoken}"
            )

        history.extend((f"Interviewer: {turn.question}", f"You: {turn.answer}"))
        turns.append(turn)
    return turns


def gemini_llm(model: str, usage: LlmUsage | None = None) -> LlmFn:
    """Build the pinned-Gemini text callable, recording token usage.

    Args:
        model: Gemini model identifier (pin: :data:`GEMINI_MODEL`).
        usage: Mutable accumulator actual spend is computed from; created
            when omitted. It rides on the returned callable as ``usage``.

    Returns:
        An :data:`LlmFn` closing over one shared client.
    """
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=require_env("GEMINI_API_KEY", "sim-interview"))
    usage = usage if usage is not None else LlmUsage()

    # Gemini 2.x accepts thinking_budget=0 to disable thinking; 3.x models
    # reject that form. On 3.x the lane runs thinking ON (dynamic budget,
    # user directive 2026-08-12) with a raised output cap so thought tokens
    # cannot starve the answer text.
    gemini3 = model.startswith(("gemini-3", "models/gemini-3"))
    thinking = genai_types.ThinkingConfig(thinking_budget=-1 if gemini3 else 0)
    max_output_tokens = 1024 if gemini3 else 256

    async def call(system: str, prompt: str, temperature: float) -> str:
        response = None
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        thinking_config=thinking,
                    ),
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2.0 * (attempt + 1))
        assert response is not None
        meta = getattr(response, "usage_metadata", None)
        usage.add(
            int(getattr(meta, "prompt_token_count", 0) or 0),
            int(getattr(meta, "candidates_token_count", 0) or 0),
        )
        return response.text or ""

    # Function attributes are untypeable; the closure carries its usage tally
    # so callers can read token spend without a wrapper class.
    call.usage = usage  # ty: ignore[unresolved-attribute]
    return call


@dataclass(slots=True)
class LlmUsage:
    """Token usage accumulated across every LLM call of a run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        """Record one call."""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    @property
    def usd(self) -> float:
        """Spend at the pinned Flash list prices."""
        return (
            self.input_tokens / 1e6 * GEMINI_FLASH_USD_PER_M_INPUT
            + self.output_tokens / 1e6 * GEMINI_FLASH_USD_PER_M_OUTPUT
        )


# --------------------------------------------------------------------------
# Synthesis and degradation
# --------------------------------------------------------------------------


class KokoroSynth:
    """Kokoro TTS pinned to :data:`KOKORO_REVISION`.

    Every artifact — config, weights, voice packs — is fetched at the pinned
    revision via ``hf_hub_download``, because ``KPipeline`` alone always
    pulls repo head (same trap the whisper-local judge hit, see the P0
    notepad). Requires the ``sim-kokoro`` optional dependency group, which
    also pins the spaCy ``en_core_web_sm`` wheel: misaki otherwise tries a
    runtime ``pip install`` that terminates uv-managed interpreters with
    ``SystemExit`` (no pip in the environment).
    """

    def __init__(self, voices: tuple[str, ...] = KOKORO_VOICES) -> None:
        """Load the pinned model and voice packs, downloading if needed."""
        try:
            from huggingface_hub import hf_hub_download
            from kokoro import KModel, KPipeline
        except ImportError as exc:
            raise RuntimeError("kokoro is not installed; run: uv sync --extra sim-kokoro") from exc

        config = hf_hub_download(KOKORO_REPO, "config.json", revision=KOKORO_REVISION)
        weights = hf_hub_download(KOKORO_REPO, "kokoro-v1_0.pth", revision=KOKORO_REVISION)
        model = KModel(repo_id=KOKORO_REPO, config=config, model=weights).eval()
        self._pipeline = KPipeline(lang_code="a", repo_id=KOKORO_REPO, model=model)
        self._voice_paths = {
            voice: hf_hub_download(KOKORO_REPO, f"voices/{voice}.pt", revision=KOKORO_REVISION) for voice in voices
        }

    def __call__(self, text: str, voice: str) -> np.ndarray:
        """Synthesize one answer to float32 mono samples at 24 kHz."""
        chunks = [
            np.asarray(audio, dtype=np.float32)
            for _, _, audio in self._pipeline(text, voice=self._voice_paths[voice])
            if audio is not None
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


def degrade_tel8k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the pinned telephony round-trip (mirrors ``snr.py``)."""
    narrow = soxr.resample(samples, sample_rate, TELEPHONY_RATE, quality="HQ")
    restored = soxr.resample(narrow, TELEPHONY_RATE, sample_rate, quality="HQ")
    return restored.astype(np.float32)


def clips_from_turns(
    turns: list[Turn], tts: TtsFn, *, language: str, degradation: str = DEGRADATION
) -> list[AudioClip]:
    """Voice and degrade every verified turn.

    Unverified turns are skipped — they cost audio minutes and can score
    nothing. The clip reference is the persona's answer text, so the standard
    WER/entity machinery works over the same results file.

    Args:
        turns: Interview turns; only verified ones become clips.
        tts: Synthesizer (pinned Kokoro in production).
        language: BCP-47 tag stamped on every clip.
        degradation: Degradation condition; only :data:`DEGRADATION` exists.

    Returns:
        One 16 kHz clip per verified turn, in turn order.

    Raises:
        ConfigError: If an unknown degradation condition is requested.
    """
    if degradation != DEGRADATION:
        raise ConfigError(
            f"unknown degradation {degradation!r}; the pinned condition is "
            f"{DEGRADATION!r} (babble mixing needs the MUSAN fetch first)"
        )
    clips: list[AudioClip] = []
    for turn in turns:
        if not turn.verified:
            continue
        wav24 = tts(turn.answer, turn.voice)
        wav16 = soxr.resample(wav24, KOKORO_SAMPLE_RATE, STT_SAMPLE_RATE, quality="HQ").astype(np.float32)
        degraded = degrade_tel8k(wav16, STT_SAMPLE_RATE)
        clips.append(
            synthetic.to_clip(
                degraded,
                clip_id=turn.clip_id,
                reference=turn.answer,
                language=language,
                sample_rate=STT_SAMPLE_RATE,
                source_path=f"<sim:{DEGRADATION}:{turn.voice}>",
            )
        )
    return clips


# --------------------------------------------------------------------------
# Spend accounting
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class SpendEstimate:
    """Expected spend, computed and printed before anything paid runs.

    Attributes:
        turns: Planned slot-bearing turns.
        audio_hours: Expected billed audio per lane, warmup included.
        stt_usd: Expected streaming STT spend per provider.
        llm_calls: Expected LLM calls including a retry factor.
        llm_usd: Expected LLM spend at pinned Flash prices.
    """

    turns: int
    audio_hours: float
    stt_usd: dict[str, float]
    llm_calls: int
    llm_usd: float

    @property
    def total_usd(self) -> float:
        """Total expected spend. Kokoro synthesis is local and free."""
        return sum(self.stt_usd.values()) + self.llm_usd

    def render(self) -> str:
        """Human-readable breakdown for the pre-run printout."""
        lines = [
            f"turns: {self.turns} (~{self.audio_hours:.2f} h audio incl. warmup)",
        ]
        for provider, usd in sorted(self.stt_usd.items()):
            lines.append(f"  stt {provider}: ${usd:.2f}")
        lines.extend((
            f"  llm ~{self.llm_calls} calls ({GEMINI_MODEL}): ${self.llm_usd:.2f}",
            "  tts kokoro (local): $0.00",
            f"expected total: ${self.total_usd:.2f}",
        ))
        return "\n".join(lines)


def estimate_spend(
    bench: BenchmarkConfig,
    scenarios: list[Scenario],
    personas_per_scenario: int,
    *,
    est_answer_s: float,
) -> SpendEstimate:
    """Compute the expected spend of one full run.

    Args:
        bench: Benchmark definition naming the candidate STT providers.
        scenarios: Scenarios to be conducted.
        personas_per_scenario: Personas sampled per scenario.
        est_answer_s: Expected seconds of audio per answer.

    Returns:
        The estimate; providers without a pricing entry contribute $0 with
        their absence visible in the per-provider breakdown.
    """
    turns = sum(len(s.slots) for s in scenarios) * personas_per_scenario
    audio_s = turns * est_answer_s
    per_lane_s = audio_s + bench.run.warmup * est_answer_s
    hours = per_lane_s / 3600.0

    stt_usd: dict[str, float] = {}
    for entry in bench.stt:
        pricing = STT_PRICING.get(entry.name)
        rate = pricing.stream_per_hour if pricing else None
        stt_usd[entry.name] = (rate or 0.0) * hours * len(entry.modes)

    calls = int(turns * 2 * _EST_RETRY_FACTOR)
    interviewer_in, interviewer_out = _EST_INTERVIEWER_TOKENS
    persona_in, persona_out = _EST_PERSONA_TOKENS
    llm_usd = (interviewer_in + persona_in) / 2 * calls / 1e6 * GEMINI_FLASH_USD_PER_M_INPUT + (
        interviewer_out + persona_out
    ) / 2 * calls / 1e6 * GEMINI_FLASH_USD_PER_M_OUTPUT
    return SpendEstimate(
        turns=turns,
        audio_hours=hours,
        stt_usd=stt_usd,
        llm_calls=calls,
        llm_usd=llm_usd,
    )


def ensure_within_cap(estimate: SpendEstimate, cap_usd: float) -> None:
    """Enforce the hard spend cap before anything paid executes.

    Raises:
        ConfigError: If the expected total exceeds the cap.
    """
    if estimate.total_usd > cap_usd:
        raise ConfigError(
            f"expected spend ${estimate.total_usd:.2f} exceeds the hard cap "
            f"${cap_usd:.2f}; reduce scenarios/personas or raise the cap"
        )


def actual_spend(results: list[SttResult], usage: LlmUsage, *, warmup_s: float) -> dict[str, float]:
    """Compute the realized spend from measured audio and recorded tokens.

    Streamed audio is billed whether or not the transcript came back, so
    failed runs count too.

    Args:
        results: Every STT result of the run.
        usage: LLM token usage recorded by :func:`gemini_llm`.
        warmup_s: Estimated per-lane warmup audio the runner discards.

    Returns:
        Spend per provider plus ``llm`` and ``total`` entries.
    """
    spend: dict[str, float] = {}
    for result in results:
        pricing = STT_PRICING.get(result.provider)
        rate = pricing.stream_per_hour if pricing else None
        spend[result.provider] = spend.get(result.provider, 0.0) + (result.audio_s / 3600.0 * (rate or 0.0))
    for provider in list(spend):
        pricing = STT_PRICING.get(provider)
        rate = pricing.stream_per_hour if pricing else None
        spend[provider] += warmup_s / 3600.0 * (rate or 0.0)
    spend["llm"] = usage.usd
    spend["total"] = sum(spend.values())
    return spend


# --------------------------------------------------------------------------
# Ranking mathematics (average-rank ties, exact permutation p)
# --------------------------------------------------------------------------


def average_ranks(scores: dict[str, float | None], *, higher_is_better: bool = False) -> dict[str, float]:
    """Rank vendors with average-rank tie handling; missing ranks worst.

    A vendor with no value for a cell — its lane failed, or produced nothing
    scoreable — cannot outrank one that was measured, so missing vendors
    share the worst remaining ranks (tied, averaged). The composite's
    decisions list records when this fires.

    Args:
        scores: Score per vendor; ``None`` marks a missing measurement.
        higher_is_better: Sort direction; ranks always start at 1 = best.

    Returns:
        Average rank per vendor.
    """
    present = sorted(
        ((v, s) for v, s in scores.items() if s is not None),
        key=lambda item: -item[1] if higher_is_better else item[1],
    )
    ranks: dict[str, float] = {}
    position = 1
    index = 0
    while index < len(present):
        stop = index
        while stop < len(present) and present[stop][1] == present[index][1]:
            stop += 1
        tied = stop - index
        rank = position + (tied - 1) / 2.0
        for vendor, _ in present[index:stop]:
            ranks[vendor] = rank
        position += tied
        index = stop

    missing = [v for v, s in scores.items() if s is None]
    if missing:
        worst = (position + (len(scores))) / 2.0
        for vendor in missing:
            ranks[vendor] = worst
    return ranks


def spearman_rho(a: dict[str, float], b: dict[str, float]) -> float:
    """Spearman rho between two rank maps over their shared keys.

    Computed as the Pearson correlation of the rank vectors, which is the
    correct form under ties. ``nan`` when either vector has no variance.
    """
    keys = sorted(set(a) & set(b))
    xs = np.array([a[k] for k in keys], dtype=np.float64)
    ys = np.array([b[k] for k in keys], dtype=np.float64)
    if len(keys) < 2:
        return math.nan
    xd = xs - xs.mean()
    yd = ys - ys.mean()
    denom = math.sqrt(float((xd**2).sum()) * float((yd**2).sum()))
    if denom <= 0.0:
        return math.nan
    return float((xd * yd).sum()) / denom


def exact_permutation_p(a: dict[str, float], b: dict[str, float]) -> float:
    """One-sided exact permutation p for the observed Spearman rho.

    Every permutation of the vendor assignment of ``b`` is enumerated
    (n! — trivial at the lane's vendor counts) and the p-value is the
    fraction with rho at least the observed value. The identity permutation is
    included, so p ≥ 1/n! always.
    """
    keys = sorted(set(a) & set(b))
    observed = spearman_rho(a, b)
    if math.isnan(observed):
        return 1.0
    b_values = [b[k] for k in keys]
    at_least = 0
    total = 0
    for perm in permutations(b_values):
        permuted = dict(zip(keys, perm, strict=True))
        rho = spearman_rho(a, permuted)
        total += 1
        if not math.isnan(rho) and rho >= observed - 1e-12:
            at_least += 1
    return at_least / total


# --------------------------------------------------------------------------
# Real-corpus composite (the pre-registered comparison metric)
# --------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CompositeCell:
    """Vendor scores and ranks for one (language, metric) pair."""

    language: str
    metric: str
    scores: dict[str, float | None]
    ranks: dict[str, float]


@dataclass(slots=True)
class CompositeRanking:
    """The interview-fitness composite over the canonical merge.

    Pre-registered (plan step 18): for every language, vendors are ranked on
    entity-WER (pooled over entity classes) and on finalize p50; the
    composite score is each vendor's mean rank over all cells.
    """

    vendors: list[str]
    cells: list[CompositeCell] = field(default_factory=list)
    mean_rank: dict[str, float] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)


def load_canonical(paths: list[Path]) -> list[SttResult]:
    """Load canonical result files with supersede-order lane merging.

    Mirrors the report command: when the same ``(provider, mode)`` lane
    appears in several files, the later file wins wholesale.

    Args:
        paths: Result files, earliest first.

    Returns:
        The merged results.
    """
    lanes: dict[tuple[str, str], list[SttResult]] = {}
    for path in paths:
        loaded = runner.read_stt_results(path)
        fresh: dict[tuple[str, str], list[SttResult]] = {}
        for item in loaded:
            fresh.setdefault((item.provider, str(item.mode)), []).append(item)
        lanes.update(fresh)
    return [item for runs in lanes.values() for item in runs]


def composite_ranking(
    results: list[SttResult], vendors: list[str], *, fallback_language: str = "en-US"
) -> CompositeRanking:
    """Compute the pre-registered composite from merged canonical results.

    Entity-WER is only comparable when every vendor is scored over the same
    reference spans, but superseding re-run files may lack the annotations
    the original run carried. Annotations are therefore propagated across
    lanes by ``(language, clip_id)`` — clips are shared, so any lane's
    annotation is every lane's annotation — and only clips no lane annotated
    fall back to the deterministic rule tagger. Only streaming lanes
    participate: the E3 candidates are the streaming vendor set.

    Args:
        results: Canonical merged results.
        vendors: The vendor set the gate compares (registry keys).
        fallback_language: Tag for results that recorded none.

    Returns:
        The composite with per-cell ranks and per-vendor mean ranks.
    """
    shared: dict[tuple[str, str], tuple[str, str]] = {}
    for result in results:
        annotated = result.raw.get("reference_annotated")
        reference = result.raw.get("reference")
        language = str(result.raw.get("language") or fallback_language)
        if isinstance(annotated, str) and annotated and isinstance(reference, str):
            shared[language, result.clip_id] = (reference, annotated)

    for result in results:
        reference = result.raw.get("reference")
        language = str(result.raw.get("language") or fallback_language)
        if not isinstance(reference, str) or not reference or result.raw.get("reference_annotated"):
            continue
        known = shared.get((language, result.clip_id))
        if known is not None and " ".join(known[0].split()) == " ".join(reference.split()):
            result.raw["reference_annotated"] = known[1]
            continue
        annotated = autotag_reference(reference, language)
        if annotated != reference:
            result.raw["reference_annotated"] = annotated

    summaries = [
        summary
        for summary in summarize(results, fallback_language)
        if summary.provider in vendors and summary.mode == "stream"
    ]
    composite = CompositeRanking(vendors=sorted(vendors))
    composite.decisions.append(
        "streaming lanes only; annotations propagated across lanes by "
        "(language, clip_id) so every vendor is scored over the same entity "
        "spans; unannotated clips fall back to the deterministic rule tagger"
    )
    composite.decisions.append(
        "a vendor with no measurement in a cell shares the worst remaining "
        "ranks (tied, averaged): a lane that failed to produce the "
        "measurement cannot outrank one that did"
    )

    languages = sorted({s.language for s in summaries})
    for language in languages:
        by_vendor = {s.provider: s for s in summaries if s.language == language}
        entity_scores: dict[str, float | None] = {}
        finalize_scores: dict[str, float | None] = {}
        for vendor in composite.vendors:
            summary = by_vendor.get(vendor)
            if summary is None:
                entity_scores[vendor] = None
                finalize_scores[vendor] = None
                continue
            pooled = None
            for score in summary.entities.values():
                pooled = score.counts if pooled is None else pooled + score.counts
            entity_scores[vendor] = pooled.rate if pooled is not None and pooled.reference_length > 0 else None
            finalize_scores[vendor] = percentile(summary.finalize_s, 50)

        for metric, scores in (
            ("entity-wer", entity_scores),
            ("finalize-p50", finalize_scores),
        ):
            if all(value is None for value in scores.values()):
                composite.decisions.append(f"cell dropped (no vendor measured): {language}/{metric}")
                continue
            composite.cells.append(
                CompositeCell(
                    language=language,
                    metric=metric,
                    scores=scores,
                    ranks=average_ranks(scores),
                )
            )

    for vendor in composite.vendors:
        cell_ranks = [cell.ranks[vendor] for cell in composite.cells]
        composite.mean_rank[vendor] = sum(cell_ranks) / len(cell_ranks) if cell_ranks else math.nan
    return composite


# --------------------------------------------------------------------------
# Scoring and the gate
# --------------------------------------------------------------------------


@dataclass(slots=True)
class VendorScore:
    """Task-success tally for one candidate vendor."""

    provider: str
    scorable: int = 0
    correct: int = 0
    failures: int = 0
    finalize_s: list[float] = field(default_factory=list)
    by_class: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def success(self) -> float | None:
        """Fraction of scorable slots extracted correctly."""
        if self.scorable == 0:
            return None
        return self.correct / self.scorable


def score_vendors(
    turns: list[Turn], results: list[SttResult], *, language: str
) -> tuple[dict[str, VendorScore], list[dict[str, Any]]]:
    """Score every vendor's transcripts against the slot ground truth.

    A vendor run that errored counts as an incorrect slot — failing to
    transcribe the caller is a task failure, not missing data.

    Args:
        turns: All turns; only verified ones are scorable.
        results: STT results joined by ``clip_id``.
        language: BCP-47 tag driving extraction normalization.

    Returns:
        Per-vendor scores, and one JSONL-ready record per turn.
    """
    by_clip: dict[str, list[SttResult]] = {}
    for result in results:
        by_clip.setdefault(result.clip_id, []).append(result)

    scores: dict[str, VendorScore] = {}
    records: list[dict[str, Any]] = []
    for turn in turns:
        vendor_cells: dict[str, dict[str, Any]] = {}
        for result in by_clip.get(turn.clip_id, []):
            score = scores.setdefault(result.provider, VendorScore(provider=result.provider))
            extracted = extract_slot(result.text, turn.truth, turn.slot.entity, language) if result.ok else None
            correct = extracted is not None
            score.scorable += 1
            score.correct += int(correct)
            if not result.ok:
                score.failures += 1
            if result.finalize_s is not None:
                score.finalize_s.append(result.finalize_s)
            done, total = score.by_class.get(turn.slot.entity, (0, 0))
            score.by_class[turn.slot.entity] = (done + int(correct), total + 1)
            vendor_cells[result.provider] = {
                "text": result.text,
                "ok": result.ok,
                "error": result.error,
                "finalize_s": result.finalize_s,
                "extracted": extracted,
                "correct": correct,
            }
        records.append({
            "scenario": turn.scenario_id,
            "persona": turn.persona_id,
            "turn": turn.index,
            "slot": turn.slot.name,
            "entity": turn.slot.entity,
            "kind": turn.slot.kind,
            "question": turn.question,
            "answer": turn.answer,
            "verified": turn.verified,
            "attempts": turn.attempts,
            "voice": turn.voice,
            "clip_id": turn.clip_id,
            "truth_written": turn.truth.written,
            "truth_canonical": turn.truth.canonical,
            "degradation": DEGRADATION,
            "vendors": vendor_cells,
        })
    return scores, records


@dataclass(slots=True)
class GateVerdict:
    """The E3 validity gate outcome (plan step 18, pre-registered)."""

    rho: float
    p_exact: float
    threshold: float
    passed: bool
    vendors: list[str]
    e3_success: dict[str, float | None]
    e3_rank: dict[str, float]
    composite_mean_rank: dict[str, float]
    composite_rank: dict[str, float]
    divergence: list[dict[str, Any]]
    decisions: list[str]


def evaluate_gate(
    vendor_scores: dict[str, VendorScore],
    composite: CompositeRanking,
    *,
    threshold: float = 0.8,
) -> GateVerdict:
    """Compare the E3 ranking against the composite ranking.

    Args:
        vendor_scores: E3 task-success scores per vendor.
        composite: The pre-registered real-corpus composite.
        threshold: Gate threshold on Spearman rho.

    Returns:
        The verdict, including the per-vendor divergence table a failing
        gate's written analysis starts from.
    """
    vendors = sorted(set(composite.vendors) & set(vendor_scores))
    e3_success: dict[str, float | None] = {vendor: vendor_scores[vendor].success for vendor in vendors}
    e3_rank = average_ranks(e3_success, higher_is_better=True)
    composite_rank = average_ranks({vendor: composite.mean_rank.get(vendor) for vendor in vendors})
    rho = spearman_rho(e3_rank, composite_rank)
    p_exact = exact_permutation_p(e3_rank, composite_rank)
    divergence = [
        {
            "vendor": vendor,
            "e3_success": e3_success[vendor],
            "e3_rank": e3_rank[vendor],
            "composite_rank": composite_rank[vendor],
            "delta": e3_rank[vendor] - composite_rank[vendor],
        }
        for vendor in sorted(vendors, key=lambda v: -abs(e3_rank[v] - composite_rank[v]))
    ]
    return GateVerdict(
        rho=rho,
        p_exact=p_exact,
        threshold=threshold,
        passed=not math.isnan(rho) and rho >= threshold,
        vendors=vendors,
        e3_success=e3_success,
        e3_rank=e3_rank,
        composite_mean_rank={vendor: composite.mean_rank.get(vendor, math.nan) for vendor in vendors},
        composite_rank=composite_rank,
        divergence=divergence,
        decisions=list(composite.decisions),
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass(slots=True)
class SimConfig:
    """The ``sim`` section of a simulated-interview config."""

    scenarios_path: str
    personas_per_scenario: int = 8
    model: str = GEMINI_MODEL
    voices: tuple[str, ...] = KOKORO_VOICES
    degradation: str = DEGRADATION
    est_answer_s: float = 12.0
    hard_cap_usd: float = 50.0
    seed: int | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SimConfig:
        """Parse the ``sim`` mapping of a config file."""
        if not raw.get("scenarios"):
            raise ConfigError("sim config needs a scenarios path")
        known = {
            "scenarios",
            "personas_per_scenario",
            "model",
            "voices",
            "degradation",
            "est_answer_s",
            "hard_cap_usd",
            "seed",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"sim config has unknown key(s): {', '.join(unknown)}")
        return cls(
            scenarios_path=str(raw["scenarios"]),
            personas_per_scenario=int(raw.get("personas_per_scenario", 8)),
            model=str(raw.get("model", GEMINI_MODEL)),
            voices=tuple(raw.get("voices") or KOKORO_VOICES),
            degradation=str(raw.get("degradation", DEGRADATION)),
            est_answer_s=float(raw.get("est_answer_s", 12.0)),
            hard_cap_usd=float(raw.get("hard_cap_usd", 50.0)),
            seed=int(raw["seed"]) if raw.get("seed") is not None else None,
        )


@dataclass(slots=True)
class SimOutcome:
    """Everything one simulated-interview run produced."""

    seed: int
    turns: list[Turn]
    results: list[SttResult]
    vendor_scores: dict[str, VendorScore]
    records: list[dict[str, Any]]
    usage: LlmUsage
    spend: dict[str, float]
    language: str
    pins: dict[str, str]


async def run_sim(
    bench: BenchmarkConfig,
    sim: SimConfig,
    *,
    llm: LlmFn,
    tts: TtsFn,
    run_stt: Callable[..., Awaitable[list[SttResult]]] | None = None,
    progress: runner.Progress | None = None,
) -> SimOutcome:
    """Execute one simulated-interview run end to end.

    The caller is expected to have printed and cap-checked the spend
    estimate already (the CLI does); this function re-enforces the cap as a
    fail-fast guard.

    Args:
        bench: Benchmark definition naming the candidate STT lanes.
        sim: Simulation parameters.
        llm: Text generation callable; tests inject a local stub.
        tts: Synthesizer; tests inject a local stub.
        run_stt: STT executor; defaults to the shared runner.
        progress: Optional progress sink for lane updates.

    Returns:
        The complete outcome, scored and spend-accounted.
    """
    scenarios = load_scenarios(sim.scenarios_path)
    estimate = estimate_spend(
        bench,
        scenarios,
        sim.personas_per_scenario,
        est_answer_s=sim.est_answer_s,
    )
    ensure_within_cap(estimate, sim.hard_cap_usd)

    seed = sim.seed if sim.seed is not None else secrets.randbits(31)
    rng = np.random.default_rng(seed)
    language = scenarios[0].language

    all_turns: list[Turn] = []
    for scenario in scenarios:
        personas = sample_personas(scenario, sim.personas_per_scenario, rng)
        interviews = await asyncio.gather(
            *(
                conduct_interview(
                    scenario,
                    persona,
                    llm,
                    voice=sim.voices[index % len(sim.voices)],
                )
                for index, persona in enumerate(personas)
            )
        )
        for turns in interviews:
            all_turns.extend(turns)

    clips = clips_from_turns(all_turns, tts, language=language, degradation=sim.degradation)
    if not clips:
        raise ConfigError("no verified turns produced any audio; nothing to run")

    executor = run_stt if run_stt is not None else runner.run_stt
    results = await executor(bench, clips, progress)
    for result in results:
        reference = result.raw.get("reference")
        if isinstance(reference, str) and reference:
            annotated = autotag_reference(reference, language)
            if annotated != reference:
                result.raw["reference_annotated"] = annotated

    vendor_scores, records = score_vendors(all_turns, results, language=language)
    usage = _usage_of(llm)
    spend = actual_spend(
        results,
        usage,
        warmup_s=bench.run.warmup * (clips[0].duration_s if clips else 0.0),
    )
    return SimOutcome(
        seed=seed,
        turns=all_turns,
        results=results,
        vendor_scores=vendor_scores,
        records=records,
        usage=usage,
        spend=spend,
        language=language,
        pins={
            "model": sim.model,
            "kokoro_repo": KOKORO_REPO,
            "kokoro_revision": KOKORO_REVISION,
            "voices": ",".join(sim.voices),
            "degradation": sim.degradation,
        },
    )


def _usage_of(llm: LlmFn) -> LlmUsage:
    """Recover the usage accumulator a production LLM callable carries."""
    usage = getattr(llm, "usage", None)
    return usage if isinstance(usage, LlmUsage) else LlmUsage()


# --------------------------------------------------------------------------
# Persistence and reporting
# --------------------------------------------------------------------------


def write_sim_outputs(outcome: SimOutcome, gate: GateVerdict | None, results_path: Path) -> tuple[Path, Path, Path]:
    """Write the sim results, gate metrics and report next to the raw runs.

    Args:
        outcome: The run to persist.
        gate: Gate verdict, or ``None`` when no canonical files were given.
        results_path: The ``stt-results.jsonl`` the runner wrote; sim files
            land in the same directory.

    Returns:
        Paths of ``sim-results.jsonl``, ``sim-gate.json``, ``sim-report.md``.
    """
    directory = results_path.parent
    sim_path = directory / "sim-results.jsonl"
    with sim_path.open("wb") as handle:
        for record in outcome.records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")

    gate_path = directory / "sim-gate.json"
    payload: dict[str, Any] = {
        "seed": outcome.seed,
        "pins": outcome.pins,
        "spend_usd": outcome.spend,
        "llm_usage": {
            "calls": outcome.usage.calls,
            "input_tokens": outcome.usage.input_tokens,
            "output_tokens": outcome.usage.output_tokens,
        },
        "gate": None,
    }
    if gate is not None:
        payload["gate"] = {
            "rho": None if math.isnan(gate.rho) else gate.rho,
            "p_exact": gate.p_exact,
            "threshold": gate.threshold,
            "passed": gate.passed,
            "vendors": gate.vendors,
            "e3_success": gate.e3_success,
            "e3_rank": gate.e3_rank,
            "composite_mean_rank": {k: None if math.isnan(v) else v for k, v in gate.composite_mean_rank.items()},
            "composite_rank": gate.composite_rank,
            "divergence": gate.divergence,
            "decisions": gate.decisions,
        }
    gate_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    report_path = directory / "sim-report.md"
    report_path.write_text(render_sim_markdown(outcome, gate), encoding="utf-8")
    return sim_path, gate_path, report_path


def render_sim_markdown(outcome: SimOutcome, gate: GateVerdict | None) -> str:
    """Render the lane's report section as GitHub-flavoured markdown."""
    verified = sum(1 for t in outcome.turns if t.verified)
    lines = [
        "# Simulated-interview E2E (experimental lane E3)",
        "",
        f"Seed {outcome.seed}; {len(outcome.turns)} turns generated, "
        f"{verified} verified and voiced; degradation `{DEGRADATION}`; "
        f"persona voice Kokoro `{KOKORO_REVISION[:12]}` "
        f"({outcome.pins['voices']}); LLM `{outcome.pins['model']}`.",
        "",
        "| Provider | Scorable | Correct | Task success | Fail | Fin p50 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(
        outcome.vendor_scores.values(),
        key=lambda s: -(s.success if s.success is not None else -1.0),
    )
    for score in ordered:
        fin = percentile(score.finalize_s, 50)
        lines.append(
            f"| {score.provider} | {score.scorable} | {score.correct} | "
            f"{'—' if score.success is None else f'{score.success * 100:.1f}%'} | "
            f"{score.failures} | "
            f"{'—' if fin is None else f'{fin:.3f}s'} |"
        )

    classes = sorted({c for s in ordered for c in s.by_class})
    if classes:
        lines += [
            "",
            "| Provider | " + " | ".join(classes) + " |",
            "| --- | " + " | ".join("---" for _ in classes) + " |",
        ]
        for score in ordered:
            cells = []
            for entity in classes:
                correct, total = score.by_class.get(entity, (0, 0))
                cells.append("—" if total == 0 else f"{correct}/{total}")
            lines.append(f"| {score.provider} | " + " | ".join(cells) + " |")

    lines += ["", "## Validity gate", ""]
    if gate is None:
        lines.append("_No canonical results supplied; gate not evaluated._")
    else:
        rho = "nan" if math.isnan(gate.rho) else f"{gate.rho:.3f}"
        verdict = "PASSED" if gate.passed else "FAILED"
        lines += [
            f"Spearman rho = {rho} (threshold {gate.threshold}), exact "
            f"permutation p = {gate.p_exact:.4f} over "
            f"{len(gate.vendors)}! vendor orderings — **{verdict}**.",
            "",
            "| Vendor | E3 success | E3 rank | Composite rank | Δ |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in gate.divergence:
            success = row["e3_success"]
            lines.append(
                f"| {row['vendor']} | "
                f"{'—' if success is None else f'{success * 100:.1f}%'} | "
                f"{row['e3_rank']:.1f} | {row['composite_rank']:.1f} | "
                f"{row['delta']:+.1f} |"
            )
        lines += ["", "Decisions applied:"]
        lines += [f"- {decision}" for decision in gate.decisions]
        if not gate.passed:
            lines += [
                "",
                "_Gate failed: a written divergence analysis "
                "(sim-divergence.md) is required before any kill or demote "
                "verdict — see the table above for where the rankings part._",
            ]
    lines += [
        "",
        "_Experimental lane: rankings above are not publishable until the gate passes (plan AC8)._",
        "",
    ]
    return "\n".join(lines)
