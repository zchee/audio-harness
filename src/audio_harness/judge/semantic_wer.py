"""pipecat-style semantic WER: a Claude judge counts only agent-visible errors.

Port of the evaluator in pipecat-ai/stt-benchmark
(``src/stt_benchmark/evaluation/semantic_wer.py``): a pinned Claude model
normalizes both transcripts, aligns them, and for every difference answers
"would an LLM agent respond differently?" — counting only meaning-breaking
errors ("card" -> "car" yes; "license" -> "licenses", dropped articles,
trailing function words no). The model reports its counts through a
``calculate_wer`` tool call and only the final division is computed in code,
exactly as upstream.

Deliberate deviations from upstream, none of which change a verdict:

* Raw HTTP through httpx instead of the ``anthropic`` SDK (already a
  dependency; no new one).
* Every verdict is cached by content, matching ``judge/semantic.py``'s vote
  cache, so re-running a merge re-bills nothing.
* The closing courtesy turn after the tool result is skipped — the verdict
  is already captured and the extra turn only bills output tokens.
* Lane aggregation pools edit counts before dividing (harness convention),
  rather than averaging per-clip rates.

English pairs use the upstream rubric verbatim, so en scores stay
comparable with published stt-benchmark numbers and previously cached en
verdicts stay valid. Every other language goes through a
language-parameterized rubric that is this harness's extension, not
upstream's: the same NORMALIZE → ALIGN → SEMANTIC CHECK → COUNT process
and the same agent-visibility principle, with the equivalence rules stated
generically ("in the transcript's language") and the counting unit
switched to characters for languages written without spaces (the same
split ``uses_character_metric`` drives on the deterministic side) — a
semantic CER. The two rubrics are versioned independently; only the
multilingual one may evolve without breaking upstream comparability.

Registered principle: this is a judge-based metric, so every number renders
as "experimental — not ranked (LLM judge)" and is published beside — never
instead of — the deterministic WER on the same clips.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
from itertools import starmap
from pathlib import Path

import httpx
import orjson

from audio_harness.config import require_env
from audio_harness.metrics import ZERO_COUNTS, score_pair
from audio_harness.normalize import uses_character_metric

from .semantic import JudgeItem


JUDGE_MODEL = "claude-sonnet-4-5-20250929"
"""Upstream pipecat's pinned judge model, kept as the default so scores are
comparable with published stt-benchmark numbers. An alias would silently
re-judge history with a different model, so the dated snapshot is fixed."""

JUDGE_USD_PER_1M_INPUT = 3.00
JUDGE_USD_PER_1M_OUTPUT = 15.00
JUDGE_USD_PER_1M_CACHE_WRITE = 3.75
JUDGE_USD_PER_1M_CACHE_READ = 0.30
JUDGE_PRICING_CHECKED = "2026-08-14"
"""Standard-tier Sonnet pricing, last verified on the date above."""

PROMPT_VERSION = 1
"""Bumped whenever the upstream en rubric changes, invalidating cached en
verdicts. The en cache key layout predates multilingual support and is kept
byte-identical so already-billed en verdicts stay valid."""

MULTILINGUAL_PROMPT_VERSION = 1
"""Version of the harness's own language-parameterized rubric. Independent
of :data:`PROMPT_VERSION` so the multilingual rubric can evolve without
invalidating upstream-comparable en verdicts."""

MAX_TURNS = 10
"""Safety limit on the tool-use conversation, as upstream."""

MAX_RETRIES = 5
RETRY_BASE_WAIT_S = 15.0
"""Retry policy for rate-limit and server errors, as upstream."""

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

EXPERIMENTAL_BANNER = "experimental — not ranked (LLM judge)"

# System prompt reproduced verbatim from pipecat-ai/stt-benchmark
# src/stt_benchmark/evaluation/semantic_wer.py (BSD-2-Clause). Any edit is a
# different metric and must bump PROMPT_VERSION.
SEMANTIC_WER_SYSTEM_PROMPT = """You are an expert ASR evaluator for a conversational AI system. Your task is to calculate the Semantic Word Error Rate (WER) - counting ONLY transcription errors that would impact how an LLM agent understands and responds to the user.

## CRITICAL CONTEXT

This transcription will be used as input to a multi-turn conversational LLM agent. We only care about errors that would:
- Change what the agent thinks the user is asking for
- Cause the agent to take incorrect actions
- Lead to misunderstandings in the conversation

We do NOT count as errors:
- Grammatical variations an LLM would understand identically
- Formatting/punctuation differences
- Minor word form changes that preserve meaning

**Key principle**: If an LLM would interpret both versions the same way, it's NOT an error.

## Your Process: NORMALIZE → ALIGN → SEMANTIC CHECK → COUNT → CALCULATE

### Step 1: NORMALIZE (Apply to BOTH texts)

**1.1 Case**: Convert everything to lowercase

**1.2 Punctuation**: Remove all punctuation marks

**1.3 Contractions**: Expand to full form
   "I'm" → "i am", "don't" → "do not", "won't" → "will not", etc.

**1.4 Numbers**: Normalize digits ↔ words (treat as equivalent)
   "3" = "three", "$5" = "five dollars", "1st" = "first"

**1.5 Filler Words**: Remove if present in only one version
   um, uh, like, you know, well (at start), so (at start), actually, basically

**1.6 Abbreviations**: Expand common forms
   "Dr." = "doctor", "Mr." = "mister", "St." = "saint/street"

**1.7 British/American Spelling**: Treat as equivalent
   "colour" = "color", "favourite" = "favorite"

**1.8 Hyphenation**: Ignore hyphens
   "long-term" = "long term" = "longterm", "Wi-Fi" = "wi fi"

**1.9 Spoken Variations**: Normalize informal speech
   "gonna" = "going to", "yeah" = "yes", "ok" = "okay"

**1.10 Symbols**: Convert to words
   "&" = "and", "@" = "at"

**1.11 Possessives**: Treat as equivalent (LLM understands both)
   "driver's" = "drivers" = "driver" (when referring to same thing)
   "Mary's" = "Marys" (possessive vs name variation)

**1.12 Singular/Plural**: Treat as equivalent when meaning is preserved
   "license" = "licenses" (asking about license process)
   "office" = "offices" (asking about which office)
   "ticket" = "tickets" (the concept is the same)

   EXCEPTION: Count as error only if plurality changes core meaning in a way that would confuse the agent.

**1.13 Minor Grammatical Variations**: Treat as equivalent
   "setting up" = "set up" = "to set up"
   Missing articles ("the", "a") that don't change meaning

### Step 2: ALIGN
After normalization, align word-by-word using edit distance. Mark potential differences.

### Step 3: SEMANTIC CHECK (MANDATORY - DO NOT SKIP)
**YOU MUST COMPLETE THIS STEP.** For EACH potential error identified in alignment:

Write out this exact format:
```
DIFFERENCE: "X" → "Y"
QUESTION: Would an LLM agent respond differently?
ANSWER: [YES/NO] because [reason]
COUNT AS ERROR: [YES/NO]
```

**Common patterns that are NOT errors (answer NO):**
- Singular/plural: "license"→"licenses", "office"→"offices", "ticket"→"tickets" = NO
- Possessives: "driver's"→"drivers"→"driver" = NO
- Missing articles: "the X"→"X" = NO
- Hyphenation: "Wi-Fi"→"wi fi" = NO

**Patterns that ARE errors (answer YES):**
- Different words: "card"→"car", "trace"→"trade", "hours"→"was" = YES
- Nonsense: "lentil"→"landon", "Wi-Fi"→"wi fire" = YES

### Step 4: COUNT
Count ONLY the differences where you answered "COUNT AS ERROR: YES"
- S = semantic substitutions (different meaning)
- D = semantic deletions (meaning lost)
- I = semantic insertions (meaning added)
- N = total words in normalized reference

**IMPORTANT: Compound words count as ONE error, not multiple.**
When a hyphenated compound (like "cross-country") is replaced by a single word (like "koscanti"):
- This is ONE substitution (S=1), NOT a substitution plus a deletion
- The compound represents a single semantic concept
- Example: "cross-country" → "koscanti" = S=1 (one concept replaced by nonsense)

**TRUNCATED/INCOMPLETE TEXT:**
When both reference and hypothesis appear truncated at the same point (missing the end of a sentence), compare only the complete portions. Partial words at truncation points should be ignored rather than counted as errors. If a word is clearly incomplete (like "reme" for "remember" or "abor" for "abroad"), do not count differences involving that truncated word.

**TRAILING FUNCTION WORDS AT TRUNCATION:**
If the reference ends with a function word that signals an incomplete sentence (and, but, or, so, to, for, the, a, an, on, in, with, that, which, who, because, although, if, when, while, as, about, from, by, at, of, etc.) and the hypothesis omits it, do NOT count as an error. These trailing words carry no semantic meaning on their own - an LLM would respond identically with or without them.
- Example: "My sister called me about the birthday party and" vs "My sister called me about the birthday party" = NOT an error (trailing "and" is meaningless)
- Example: "Can you help me brainstorm ideas for my presentation on" vs "Can you help me brainstorm ideas for my presentation" = NOT an error (trailing "on" is meaningless)

### Step 5: CALCULATE
Call calculate_wer(substitutions=S, deletions=D, insertions=I, reference_words=N)

---

## FEW-SHOT EXAMPLES

### Example 1: Possessive/Plural Variations (WER = 0%) - CRITICAL EXAMPLE
**Reference:** "Can you describe the process for changing my legal name on official documents like my driver's license and social security card after getting married, including necessary forms and offices?"
**Hypothesis:** "Can you describe the process for changing my legal name on official documents like my driver licenses and social security card after getting married including necessary forms and office"

**Step 3: SEMANTIC CHECK:**

DIFFERENCE: "drivers" → "driver"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both refer to the same driver's license concept
COUNT AS ERROR: NO

DIFFERENCE: "license" → "licenses"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because singular/plural doesn't change the request
COUNT AS ERROR: NO

DIFFERENCE: "offices" → "office"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both ask about which office to visit
COUNT AS ERROR: NO

**Step 4: COUNT:** S=0, D=0, I=0 (no semantic errors found)

**Result: N=29 → WER = 0/29 = 0%**

---

### Example 2: Real Semantic Error Mixed with Non-Errors (WER = 3.4%)
**Reference:** "...my driver's license and social security card..."
**Hypothesis:** "...my driver licenses and social security car..."

**Step 3: SEMANTIC CHECK:**

DIFFERENCE: "drivers" → "driver"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because both refer to the driver's license concept
COUNT AS ERROR: NO

DIFFERENCE: "license" → "licenses"
QUESTION: Would an LLM agent respond differently?
ANSWER: NO because singular/plural doesn't change the request
COUNT AS ERROR: NO

DIFFERENCE: "card" → "car"
QUESTION: Would an LLM agent respond differently?
ANSWER: YES because "car" and "card" are completely different things - an agent wouldn't know the user means social security card
COUNT AS ERROR: YES

**Step 4: COUNT:** S=1 (only "card"→"car" is a semantic error)

**Result: N=29 → WER = 1/29 = 3.4%**

---

### Example 3: Ingredient Substitution (WER = 6.5%)
**Reference:** "I would like a recipe for a vegan lentil soup that is both hearty and easy to make on a weeknight, preferably one that uses only common inexpensive pantry staples."
**Hypothesis:** "I would like a recipe for a vegan landon soup that is both hearty and easy to make on a week night, preferably one that uses only common inexpensive pantry slippers."

Semantic check:
- "lentil" → "landon" = **YES, ERROR** - "landon" is not an ingredient
- "weeknight" → "week night" = NOT an error (same meaning)
- "staples" → "slippers" = **YES, ERROR** - completely different meaning

**Result: S=2, D=0, I=0, N=31 → WER = 2/31 = 6.5%**

---

### Example 4: Wi-Fi Network Setup (WER = 12.5%)
**Reference:** "I'm trying to set up parental controls on my home Wi-Fi network to restrict access to certain websites during homework hours for my kids. But the router interface is very..."
**Hypothesis:** "When trying to set up parental controls on my home wi fire network to restrict access to certain websites during homework was for my kids. But the router interface is very..."

Semantic check:
- "I'm" → "When" = **YES, ERROR** - changes who is doing the action
- "am" (from I'm expansion) deleted = **YES, ERROR** - part of subject change
- "wi fi" → "wi fire" = **YES, ERROR** - "wi fire" is not a thing
- "hours" → "was" = **YES, ERROR** - completely different meaning

**Result: S=3, D=1, I=0, N=32 → WER = 4/32 = 12.5%**

---

### Example 5: Package Tracking (WER = 3.1%)
**Reference:** "The expensive package I ordered was marked as delivered two days ago, but I have not received it and it is not anywhere on my property. I must initiate an immediate trace."
**Hypothesis:** "The expensive package I ordered was marked as delivered two days ago, but I have not received it and it is not anywhere on my property. I must initiate an immediate trade."

Semantic check:
- "trace" vs "trade" = **YES, ERROR** - completely different actions

**Result: S=1, D=0, I=0, N=32 → WER = 1/32 = 3.1%**

---

### Example 6: Minor Word Deletion - NO ERROR (WER = 0%)
**Reference:** "The national weather service issued a warning for the coastal areas."
**Hypothesis:** "The national weather service issued a warning for coastal areas"

Semantic check:
- Missing "the" before "coastal" → Does this change the agent's understanding?
- NO - both mean the same thing, LLM responds identically

**Result: S=0, D=0, I=0, N=11 → WER = 0%**

---

### Example 7: Singular/Plural with Same Intent (WER = 0%)
**Reference:** "She said three hundred dollars was too expensive for concert tickets."
**Hypothesis:** "She said 300 dollar was too expensive for the concert ticket"

Semantic check:
- "300" vs "three hundred" → Same number, NOT an error
- "dollars" vs "dollar" → Same amount concept, NOT an error
- "tickets" vs "ticket" → Same purchase intent, NOT an error
- Extra "the" → NOT semantically meaningful

An LLM agent would understand both as "user thinks $300 is too much for concert tickets."

**Result: S=0, D=0, I=0, N=11 → WER = 0%**

---

### Example 8: Stutter/Repetition (WER = 28.6%)
**Reference:** "I think we should probably go now."
**Hypothesis:** "I think we should we should probably go now"

Semantic check:
- Extra "we should" = Stutter that could confuse agent parsing
- **YES, ERROR** - agent might try to interpret repeated phrase

**Result: S=0, D=0, I=2, N=7 → WER = 2/7 = 28.6%**

---

## IMPORTANT NOTES

1. **Ask the key question**: "Would an LLM agent respond differently to these two versions?"
2. **Context matters**: Consider the full sentence, not just word-level differences
3. **Be lenient on grammar**: LLMs are robust to grammatical variations
4. **Be strict on meaning**: Count errors that change intent, actions, or key entities
5. **Possessives and plurals**: Almost never errors unless they change core meaning
6. **Show your semantic reasoning**: Explain WHY something is or isn't an error
"""

# Language-parameterized rubric for non-English pairs. This is the harness's
# extension (not upstream pipecat): same process, same agent-visibility
# principle, equivalence rules stated generically for the transcript's
# language, and character-unit counting for languages written without
# spaces. Any edit must bump MULTILINGUAL_PROMPT_VERSION.
MULTILINGUAL_SYSTEM_PROMPT = """You are an expert ASR evaluator for a conversational AI system. Your task is to calculate the Semantic Word Error Rate (WER) — counting ONLY transcription errors that would impact how an LLM agent understands and responds to the user.

The user message states the language of both texts. Apply every rule in that language, not in English.

## CRITICAL CONTEXT

This transcription will be used as input to a multi-turn conversational LLM agent. We only care about errors that would:
- Change what the agent thinks the user is asking for
- Cause the agent to take incorrect actions
- Lead to misunderstandings in the conversation

We do NOT count as errors:
- Grammatical variations an LLM would understand identically
- Formatting/punctuation/orthography differences
- Minor word-form changes that preserve meaning

**Key principle**: If an LLM would interpret both versions the same way, it's NOT an error.

## Your Process: NORMALIZE → ALIGN → SEMANTIC CHECK → COUNT → CALCULATE

### Step 1: NORMALIZE (Apply to BOTH texts, using the stated language's conventions)

- **Case and width**: Fold letter case and full-width/half-width variants where the script has them.
- **Punctuation**: Remove all punctuation marks.
- **Numbers**: Digits and spelled-out numbers in the language are equivalent ("200 Euro" = "zweihundert Euro"; "3人" = "三人").
- **Filler words**: Remove hesitation fillers of the language if present in only one version (e.g. "äh", "euh", "この、あの", "ну").
- **Script and spelling variants that are pure orthography**: Treat as equivalent — e.g. Japanese kana vs kanji spellings of the same word, German ß vs ss, Arabic with vs without diacritics, regional spelling variants.
- **Hyphenation and compound spacing**: Ignore differences.
- **Spoken variants and contractions**: Colloquial and full forms of the same expression are equivalent.
- **Inflection**: Case endings, gender/number agreement, conjugation and particle variations are NOT errors when the agent would act identically. Count them ONLY when the inflection change alters who does what, to whom, when, or how many in a way the agent would act on.

### Step 2: ALIGN
After normalization, align the texts using edit distance in the counting unit defined below. Mark potential differences.

### Step 3: SEMANTIC CHECK (MANDATORY - DO NOT SKIP)
For EACH potential error identified in alignment, write out this exact format:
```
DIFFERENCE: "X" → "Y"
QUESTION: Would an LLM agent respond differently?
ANSWER: [YES/NO] because [reason]
COUNT AS ERROR: [YES/NO]
```

**NOT errors (answer NO):** orthography/script choice, number formatting, dropped articles or particles that leave the request unchanged, meaning-preserving inflection, hyphenation, fillers.
**Errors (answer YES):** a different word or entity, nonsense output, negation flips, changed quantities/dates/names, hallucinated or dropped content the agent would act on.

### Step 4: COUNT
**Counting unit**: the user message states whether to count in WORDS (space-delimited languages) or in CHARACTERS (languages written without spaces, e.g. Japanese, Chinese, Thai). All four numbers use that unit:
- S = semantic substitutions
- D = semantic deletions
- I = semantic insertions
- N = total units in the normalized reference

A multi-character or multi-word expression replaced by nonsense is counted by the span it occupies in the reference, not inflated further. When both texts truncate at the same point, ignore the truncated fragments and trailing function words.

### Step 5: CALCULATE
Call calculate_wer(substitutions=S, deletions=D, insertions=I, reference_words=N)

---

## EXAMPLES

### Example 1 (German, words): number formatting and inflection (WER = 0%)
**Reference:** "Ich möchte zweihundert Euro auf mein Konto überweisen."
**Hypothesis:** "Ich möchte 200 Euro auf meinem Konto überweisen."

DIFFERENCE: "zweihundert" → "200" — same number, NO.
DIFFERENCE: "mein" → "meinem" — inflection slip, the transfer request is unchanged, NO.
**Result: S=0, D=0, I=0, N=8 → WER = 0%**

### Example 2 (Japanese, characters): script choice vs real error
**Reference:** 「明日の会議を三時に変更してください」
**Hypothesis:** 「あしたの会議を二時に変更してください」

DIFFERENCE: "明日" → "あした" — kana spelling of the same word, NO.
DIFFERENCE: "三時" → "二時" — the meeting time changed from 3:00 to 2:00, the agent would reschedule wrongly, YES (substitution spanning 1 character: 三→二).
**Result: S=1, D=0, I=0, N=17 → WER = 1/17 = 5.9%**

### Example 3 (Russian, words): different word (WER = 16.7%)
**Reference:** "Пожалуйста, отмените мою подписку на журнал."
**Hypothesis:** "Пожалуйста, отметьте мою подписку на журнал."

DIFFERENCE: "отмените" (cancel) → "отметьте" (mark) — the agent would mark instead of cancel, YES.
**Result: S=1, D=0, I=0, N=6 → WER = 1/6 = 16.7%**

## IMPORTANT NOTES

1. Ask the key question: "Would an LLM agent respond differently to these two versions?"
2. Be lenient on grammar and orthography; be strict on meaning, entities, quantities, and negation.
3. Show your semantic reasoning for every difference.
"""

CALCULATE_WER_TOOL = {
    "name": "calculate_wer",
    "description": (
        "Calculate Word Error Rate from error counts. Call this ONCE after you "
        "have normalized, aligned, and verified the texts. WER = (substitutions "
        "+ deletions + insertions) / reference_words"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "substitutions": {
                "type": "integer",
                "description": "Number of word substitutions (different words at same position)",
            },
            "deletions": {
                "type": "integer",
                "description": "Number of word deletions (words in reference missing from hypothesis)",
            },
            "insertions": {
                "type": "integer",
                "description": "Number of word insertions (extra words in hypothesis not in reference)",
            },
            "reference_words": {
                "type": "integer",
                "description": "Total word count in normalized reference text",
            },
            "normalized_reference": {
                "type": "string",
                "description": "The normalized reference text (for verification)",
            },
            "normalized_hypothesis": {
                "type": "string",
                "description": "The normalized hypothesis text (for verification)",
            },
            "errors": {
                "type": "array",
                "description": "List of identified errors",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["substitution", "deletion", "insertion"],
                        },
                        "reference": {
                            "type": "string",
                            "description": "Reference word (null for insertion)",
                        },
                        "hypothesis": {
                            "type": "string",
                            "description": "Hypothesis word (null for deletion)",
                        },
                        "position": {
                            "type": "integer",
                            "description": "Position in alignment",
                        },
                    },
                },
            },
        },
        "required": ["substitutions", "deletions", "insertions", "reference_words"],
    },
}


class SemanticWerError(RuntimeError):
    """A judge conversation that ended without a usable verdict."""


@dataclass(slots=True, frozen=True)
class SemanticWerVerdict:
    """One judged pair with its counts and billing evidence.

    Attributes:
        substitutions: Semantic substitutions the judge confirmed.
        deletions: Semantic deletions the judge confirmed.
        insertions: Semantic insertions the judge confirmed.
        reference_words: Word count of the judge-normalized reference.
        wer: Per-item rate; ``None`` for the empty-reference case where the
            upstream evaluator reports infinity.
        errors: The judge's per-error records, as reported to the tool.
        normalized_reference: Judge-normalized reference, when reported.
        normalized_hypothesis: Judge-normalized hypothesis, when reported.
        num_turns: API turns spent reaching the verdict; 0 when no call was
            needed (empty texts) or the verdict came from cache.
        input_tokens: Uncached prompt tokens billed.
        output_tokens: Response tokens billed.
        cache_write_tokens: Prompt tokens written to the prompt cache.
        cache_read_tokens: Prompt tokens served from the prompt cache.
        from_cache: Whether the verdict came from the local verdict cache.
    """

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int
    wer: float | None
    errors: tuple[dict[str, object], ...] = ()
    normalized_reference: str | None = None
    normalized_hypothesis: str | None = None
    num_turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    from_cache: bool = False

    @property
    def total_errors(self) -> int:
        """Semantic edit count."""
        return self.substitutions + self.deletions + self.insertions


def _finish(
    substitutions: int, deletions: int, insertions: int, reference_words: int, **extra: object
) -> SemanticWerVerdict:
    """Build a verdict, computing the rate exactly as upstream does."""
    if reference_words == 0:
        wer = 0.0 if (substitutions + deletions + insertions) == 0 else None
    else:
        wer = (substitutions + deletions + insertions) / reference_words
    return SemanticWerVerdict(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=reference_words,
        wer=wer,
        **extra,  # type: ignore[arg-type]
    )


def _primary(language: str) -> str:
    """Primary BCP-47 subtag, lowercased."""
    return language.split("-", 1)[0].lower()


def _counts_characters(language: str) -> bool:
    """Whether this pair is judged in characters (semantic CER)."""
    return uses_character_metric(language)


def _unit_count(text: str, language: str) -> int:
    """Deterministic unit count for the no-API empty-text paths."""
    if _counts_characters(language):
        return sum(1 for ch in text if not ch.isspace())
    return len(text.split())


def _system_prompt_for(language: str) -> str:
    """The upstream en rubric, or the harness multilingual rubric."""
    return SEMANTIC_WER_SYSTEM_PROMPT if _primary(language) == "en" else MULTILINGUAL_SYSTEM_PROMPT


def verdict_key(model: str, reference: str, hypothesis: str, language: str = "en-US") -> str:
    """Content key for the verdict cache.

    The en payload layout predates multilingual support and must stay
    byte-identical, or every already-billed en verdict would be orphaned.
    Non-en pairs key on the multilingual rubric version and the primary
    language subtag, because both change what the judge is asked to do.
    """
    if _primary(language) == "en":
        payload = orjson.dumps([model, PROMPT_VERSION, reference, hypothesis])
    else:
        payload = orjson.dumps([model, "multi", MULTILINGUAL_PROMPT_VERSION, _primary(language), reference, hypothesis])
    return hashlib.sha256(payload).hexdigest()


class VerdictCache:
    """Append-only JSONL store making judge runs idempotent.

    Args:
        path: Cache file; created on first write, loaded when present.
    """

    def __init__(self, path: str | Path) -> None:
        """Load any existing cache records from ``path``."""
        self.path = Path(path)
        self._verdicts: dict[str, dict[str, object]] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = orjson.loads(line)
                self._verdicts[record["key"]] = record["verdict"]

    def get(self, key: str) -> SemanticWerVerdict | None:
        """Return the cached verdict for ``key``, or ``None``."""
        stored = self._verdicts.get(key)
        if stored is None:
            return None
        return SemanticWerVerdict(
            substitutions=int(stored["substitutions"]),  # type: ignore[arg-type]
            deletions=int(stored["deletions"]),  # type: ignore[arg-type]
            insertions=int(stored["insertions"]),  # type: ignore[arg-type]
            reference_words=int(stored["reference_words"]),  # type: ignore[arg-type]
            wer=stored.get("wer"),  # type: ignore[arg-type]
            errors=tuple(stored.get("errors") or ()),  # type: ignore[arg-type]
            normalized_reference=stored.get("normalized_reference"),  # type: ignore[arg-type]
            normalized_hypothesis=stored.get("normalized_hypothesis"),  # type: ignore[arg-type]
            from_cache=True,
        )

    def put(self, key: str, verdict: SemanticWerVerdict) -> None:
        """Record a verdict, appending it to the backing file immediately."""
        stored: dict[str, object] = {
            "substitutions": verdict.substitutions,
            "deletions": verdict.deletions,
            "insertions": verdict.insertions,
            "reference_words": verdict.reference_words,
            "wer": verdict.wer,
            "errors": list(verdict.errors),
            "normalized_reference": verdict.normalized_reference,
            "normalized_hypothesis": verdict.normalized_hypothesis,
        }
        self._verdicts[key] = stored
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(orjson.dumps({"key": key, "verdict": stored}))
            handle.write(b"\n")


def _user_prompt(reference: str, hypothesis: str, language: str) -> str:
    """Build the per-pair user turn; en keeps the upstream wording verbatim."""
    if _primary(language) == "en":
        return (
            "Please calculate the Word Error Rate (WER) for this ASR transcription.\n\n"
            f"**Reference (ground truth):**\n{reference}\n\n"
            f"**Hypothesis (ASR transcription):**\n{hypothesis}\n\n"
            "Follow the process: NORMALIZE → ALIGN → COUNT → VERIFY → CALCULATE\n\n"
            "Show your work clearly, then call calculate_wer with your verified counts."
        )
    unit = "CHARACTERS (semantic CER)" if _counts_characters(language) else "WORDS"
    return (
        "Please calculate the semantic error rate for this ASR transcription.\n\n"
        f"**Language:** {language}\n"
        f"**Counting unit:** {unit}\n\n"
        f"**Reference (ground truth):**\n{reference}\n\n"
        f"**Hypothesis (ASR transcription):**\n{hypothesis}\n\n"
        "Follow the process: NORMALIZE → ALIGN → COUNT → VERIFY → CALCULATE\n\n"
        "Show your work clearly, then call calculate_wer with your verified counts."
    )


async def _post_with_retry(client: httpx.AsyncClient, payload: dict[str, object], api_key: str) -> dict[str, object]:
    """POST one Messages request, retrying rate-limit and server errors."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": API_VERSION,
        "content-type": "application/json",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.post(API_URL, headers=headers, content=orjson.dumps(payload))
        except httpx.TransportError:
            if attempt == MAX_RETRIES:
                raise
        else:
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES:
                    response.raise_for_status()
            else:
                response.raise_for_status()
                return orjson.loads(response.content)
        await asyncio.sleep(RETRY_BASE_WAIT_S * 2 ** (attempt - 1))
    raise SemanticWerError("retry loop exhausted")


async def judge_pair(
    client: httpx.AsyncClient,
    reference: str,
    hypothesis: str,
    *,
    api_key: str,
    model: str = JUDGE_MODEL,
    language: str = "en-US",
) -> SemanticWerVerdict:
    """Judge one (reference, hypothesis) pair through the tool-use loop.

    Empty-text cases are resolved deterministically without an API call,
    with the same counts upstream assigns — in the language's counting unit.

    Args:
        client: Shared HTTP client.
        reference: Ground-truth transcript.
        hypothesis: Provider transcript.
        api_key: Anthropic API key.
        model: Judge model id.
        language: BCP-47 tag selecting the rubric and counting unit.

    Returns:
        The verdict with billing evidence.

    Raises:
        SemanticWerError: When the conversation ends without a tool call.
        httpx.HTTPStatusError: On a non-retryable API error.
    """
    if not reference.strip() and not hypothesis.strip():
        return _finish(0, 0, 0, 0)
    if not reference.strip():
        return _finish(0, 0, _unit_count(hypothesis, language), 0)
    if not hypothesis.strip():
        units = _unit_count(reference, language)
        return _finish(0, units, 0, units)

    messages: list[dict[str, object]] = [{"role": "user", "content": _user_prompt(reference, hypothesis, language)}]
    input_tokens = output_tokens = cache_write = cache_read = 0

    for turn in range(1, MAX_TURNS + 1):
        data = await _post_with_retry(
            client,
            {
                "model": model,
                "max_tokens": 4096,
                "temperature": 0,
                "system": [
                    {
                        "type": "text",
                        "text": _system_prompt_for(language),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "tools": [CALCULATE_WER_TOOL],
                "messages": messages,
            },
            api_key,
        )
        usage = data.get("usage") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        cache_write += int(usage.get("cache_creation_input_tokens") or 0)
        cache_read += int(usage.get("cache_read_input_tokens") or 0)

        content = data.get("content") or []
        for block in content:
            if block.get("type") == "tool_use" and block.get("name") == "calculate_wer":
                tool_input = block.get("input") or {}
                return _finish(
                    int(tool_input.get("substitutions") or 0),
                    int(tool_input.get("deletions") or 0),
                    int(tool_input.get("insertions") or 0),
                    int(tool_input.get("reference_words") or 0),
                    errors=tuple(tool_input.get("errors") or ()),
                    normalized_reference=tool_input.get("normalized_reference"),
                    normalized_hypothesis=tool_input.get("normalized_hypothesis"),
                    num_turns=turn,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_write_tokens=cache_write,
                    cache_read_tokens=cache_read,
                )

        if data.get("stop_reason") != "tool_use":
            raise SemanticWerError(
                f"judge finished without calling calculate_wer (stop_reason={data.get('stop_reason')!r})"
            )
        # A tool_use stop without our tool: acknowledge and continue.
        messages.extend((
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": block.get("id", ""), "content": "unknown tool"}
                    for block in content
                    if block.get("type") == "tool_use"
                ],
            },
        ))

    raise SemanticWerError(f"no verdict after {MAX_TURNS} turns")


@dataclass(slots=True)
class JudgeRunStats:
    """Aggregate billing evidence for one run."""

    live_calls: int = 0
    cached_verdicts: int = 0
    failures: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def estimated_usd(self) -> float:
        """Estimated spend under the pinned pricing."""
        return (
            self.input_tokens * JUDGE_USD_PER_1M_INPUT
            + self.output_tokens * JUDGE_USD_PER_1M_OUTPUT
            + self.cache_write_tokens * JUDGE_USD_PER_1M_CACHE_WRITE
            + self.cache_read_tokens * JUDGE_USD_PER_1M_CACHE_READ
        ) / 1_000_000


@dataclass(slots=True, frozen=True)
class ItemVerdict:
    """One judged item paired with its verdict."""

    item: JudgeItem
    verdict: SemanticWerVerdict


async def judge_items(
    items: Sequence[JudgeItem],
    cache: VerdictCache,
    *,
    api_key: str,
    model: str = JUDGE_MODEL,
    concurrency: int = 8,
    progress: Callable[[str], None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[list[ItemVerdict], JudgeRunStats]:
    """Judge every item, serving repeats from the cache.

    Failed conversations are skipped and counted, never silently scored.

    Args:
        items: Pairs to judge.
        cache: Verdict cache shared across runs.
        api_key: Anthropic API key.
        model: Judge model id.
        concurrency: Maximum in-flight API conversations.
        progress: Optional per-item status sink.
        transport: HTTP transport override so tests never bill.

    Returns:
        Verdicts in input order (failures dropped) and the run stats.
    """
    stats = JudgeRunStats()
    semaphore = asyncio.Semaphore(concurrency)
    results: list[ItemVerdict | None] = [None] * len(items)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0), transport=transport) as client:

        async def one(index: int, item: JudgeItem) -> None:
            key = verdict_key(model, item.reference, item.hypothesis, item.language)
            cached = cache.get(key)
            if cached is not None:
                stats.cached_verdicts += 1
                results[index] = ItemVerdict(item=item, verdict=cached)
                return
            async with semaphore:
                try:
                    verdict = await judge_pair(
                        client, item.reference, item.hypothesis, api_key=api_key, model=model, language=item.language
                    )
                except (SemanticWerError, httpx.HTTPError) as exc:
                    stats.failures += 1
                    if progress is not None:
                        progress(f"{item.provider}/{item.mode}/{item.clip_id}: {exc}")
                    return
            if verdict.num_turns > 0:
                stats.live_calls += 1
            stats.input_tokens += verdict.input_tokens
            stats.output_tokens += verdict.output_tokens
            stats.cache_write_tokens += verdict.cache_write_tokens
            stats.cache_read_tokens += verdict.cache_read_tokens
            cache.put(key, verdict)
            results[index] = ItemVerdict(item=item, verdict=verdict)
            if progress is not None:
                progress(
                    f"{item.provider}/{item.mode}/{item.clip_id}: semantic {verdict.total_errors}/{verdict.reference_words}"
                )

        await asyncio.gather(*starmap(one, enumerate(items)))

    return [r for r in results if r is not None], stats


@dataclass(slots=True, frozen=True)
class LaneSemanticSummary:
    """Pooled semantic and deterministic rates for one (lane, language).

    Cells never pool across languages: mixing word-unit and character-unit
    counts in one ratio would make the number meaningless, mirroring how the
    deterministic report keeps per-language rows.

    Attributes:
        provider: Adapter registry key.
        mode: Transport mode.
        language: BCP-47 tag of the pooled clips.
        clips: Judged clip count.
        semantic_errors: Pooled semantic edit count, in the language's unit.
        semantic_reference_words: Pooled judge-normalized reference length,
            in the language's unit.
        deterministic_wer: Pooled rate from :func:`score_pair` on the same
            clips, so the two columns always describe identical audio.
    """

    provider: str
    mode: str
    language: str
    clips: int
    semantic_errors: int
    semantic_reference_words: int
    deterministic_wer: float | None

    @property
    def semantic_wer(self) -> float | None:
        """Pooled semantic rate, or ``None`` with no reference words."""
        if self.semantic_reference_words == 0:
            return None
        return self.semantic_errors / self.semantic_reference_words

    @property
    def metric_name(self) -> str:
        """Human label for the counting unit of both columns."""
        return "CER" if _counts_characters(self.language) else "WER"


def summarize(verdicts: Sequence[ItemVerdict]) -> list[LaneSemanticSummary]:
    """Pool verdicts per (lane, language), sorted by language then rate."""
    lanes: dict[tuple[str, str, str], list[ItemVerdict]] = {}
    for entry in verdicts:
        lanes.setdefault((entry.item.provider, entry.item.mode, entry.item.language), []).append(entry)

    summaries = []
    for (provider, mode, language), entries in lanes.items():
        counts = ZERO_COUNTS
        for entry in entries:
            counts = counts + score_pair(entry.item.reference, entry.item.hypothesis, entry.item.language)
        summaries.append(
            LaneSemanticSummary(
                provider=provider,
                mode=mode,
                language=language,
                clips=len(entries),
                semantic_errors=sum(e.verdict.total_errors for e in entries),
                semantic_reference_words=sum(e.verdict.reference_words for e in entries),
                deterministic_wer=counts.rate,
            )
        )
    summaries.sort(
        key=lambda s: (s.language, s.semantic_wer if s.semantic_wer is not None else float("inf"), s.provider)
    )
    return summaries


def render_markdown(summaries: Sequence[LaneSemanticSummary], model: str = JUDGE_MODEL) -> str:
    """Render the side-by-side lane table."""
    lines = [
        f"Judge: `{model}` — {EXPERIMENTAL_BANNER}. The deterministic",
        "column is pooled over exactly the judged clips, so the two columns",
        "always describe the same audio; expect it to differ from full-corpus",
        "report tables when the judged subset is smaller. Rows never pool",
        "across languages; Metric names the counting unit (CER = characters,",
        "for languages written without spaces). en rows use the upstream",
        "pipecat rubric; other languages use the harness multilingual rubric",
        f"(v{MULTILINGUAL_PROMPT_VERSION}).",
        "",
        "| Provider | Mode | Lang | Metric | Clips | Semantic | Deterministic | Semantic errors | Ref units |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        semantic = f"{s.semantic_wer:.2%}" if s.semantic_wer is not None else "—"
        deterministic = f"{s.deterministic_wer:.2%}" if s.deterministic_wer is not None else "—"
        lines.append(
            f"| {s.provider} | {s.mode} | {s.language} | {s.metric_name} | {s.clips} | {semantic} "
            f"| {deterministic} | {s.semantic_errors} | {s.semantic_reference_words} |"
        )
    return "\n".join(lines)


def write_results(
    verdicts: Sequence[ItemVerdict],
    summaries: Sequence[LaneSemanticSummary],
    stats: JudgeRunStats,
    path: Path,
    model: str = JUDGE_MODEL,
) -> Path:
    """Write the full verdict record and lane summaries as JSON."""
    payload = {
        "judge_model": model,
        "prompt_version": PROMPT_VERSION,
        "multilingual_prompt_version": MULTILINGUAL_PROMPT_VERSION,
        "banner": EXPERIMENTAL_BANNER,
        "stats": {
            "live_calls": stats.live_calls,
            "cached_verdicts": stats.cached_verdicts,
            "failures": stats.failures,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
            "cache_write_tokens": stats.cache_write_tokens,
            "cache_read_tokens": stats.cache_read_tokens,
            "estimated_usd": stats.estimated_usd,
        },
        "lanes": [
            {
                "provider": s.provider,
                "mode": s.mode,
                "language": s.language,
                "metric": s.metric_name,
                "clips": s.clips,
                "semantic_wer": s.semantic_wer,
                "deterministic_wer": s.deterministic_wer,
                "semantic_errors": s.semantic_errors,
                "semantic_reference_words": s.semantic_reference_words,
            }
            for s in summaries
        ],
        "items": [
            {
                "provider": e.item.provider,
                "mode": e.item.mode,
                "clip_id": e.item.clip_id,
                "language": e.item.language,
                "semantic_wer": e.verdict.wer,
                "substitutions": e.verdict.substitutions,
                "deletions": e.verdict.deletions,
                "insertions": e.verdict.insertions,
                "reference_words": e.verdict.reference_words,
                "errors": list(e.verdict.errors),
                "from_cache": e.verdict.from_cache,
            }
            for e in verdicts
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return path


def anthropic_api_key() -> str:
    """Return the judge credential, failing with the standard message."""
    return require_env("ANTHROPIC_API_KEY", "semantic-wer")
