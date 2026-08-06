"""The "general" layer of the TTS prompt suite (plan step P2.12b).

Source: ``github.com/common-voice/common-voice`` — ``server/data/<locale>/
*.txt``, the plain-text sentence bank Common Voice itself draws collection
prompts from, CC0-1.0 licensed (``server/data/LICENSE`` in that repo).
Verified reachable, ungated and license-tagged as of 2026-08-06, and
covering all 14 locales this harness targets.

This was not the first source tried. The canonical Common Voice dataset on
Hugging Face (``mozilla-foundation/common_voice_17_0``) moved to Mozilla
Data Collective in October 2025 and is no longer fetchable here; its ungated
legacy snapshot (``legacy-datasets/common_voice``) has its dataset viewer
disabled (no ``datasets-server`` access without the deprecated loading-
script path); and a maintained re-upload (``deepdml/common_voice_26_0``) is
manually gated. The GitHub sentence corpus sidesteps all three — it is
plain CC0 text, not audio, requires no account, and is what this generator
actually fetches.

Only the sentence SELECTION logic lives here — dedup, a length filter tuned
per script (word count normally, character count for scriptio-continua
languages, matching :func:`audio_harness.normalize.uses_character_metric`),
and deterministic sampling — importable and unit-testable without a network
call. ``tools/gen_common_voice_prompts.py`` is the network shell.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from .normalize import uses_character_metric
from .prompt_suite import LOCALES, SuitePrompt

CATEGORY = "general"
LICENSE = "CC0-1.0"
SOURCE_REPO = "github.com/common-voice/common-voice (server/data)"

MIN_WORDS = 4
MAX_WORDS = 40
"""Word-count band biasing selection toward "general/long-form" sentences,
away from single-word collection prompts and multi-sentence run-ons."""

MIN_CHARS = 8
MAX_CHARS = 100
"""Character-count band for scriptio-continua languages (ja), where a
whitespace word count is meaningless."""


def _length_ok(text: str, language: str) -> bool:
    """Whether ``text`` falls inside the "general/long-form" length band."""
    if uses_character_metric(language):
        return MIN_CHARS <= len(text.replace(" ", "")) <= MAX_CHARS
    return MIN_WORDS <= len(text.split()) <= MAX_WORDS


def filter_sentences(lines: Iterable[str], *, language: str) -> list[str]:
    """Deduplicate raw candidate lines and keep only ones in the length band.

    Args:
        lines: Raw lines from one or more source files, in any order.
        language: Full BCP-47 tag, deciding word- vs character-length rules.

    Returns:
        Distinct candidate sentences, in first-seen order.
    """
    seen: set[str] = set()
    candidates: list[str] = []
    for line in lines:
        text = line.strip()
        if not text or text in seen or not _length_ok(text, language):
            continue
        seen.add(text)
        candidates.append(text)
    return candidates


def sample_sentences(candidates: list[str], *, count: int, seed: int) -> list[str]:
    """Deterministically sample up to ``count`` sentences from ``candidates``.

    Args:
        candidates: Filtered, deduplicated sentences.
        count: Maximum sentences to return.
        seed: Seed for reproducible sampling.

    Returns:
        Up to ``count`` sentences; all of them if fewer are available.
    """
    rng = random.Random(seed)
    shuffled = candidates.copy()
    rng.shuffle(shuffled)
    return shuffled[:count]


def build_prompts(
    subtag: str, sentences: list[str], source_files: list[str]
) -> list[SuitePrompt]:
    """Turn selected sentences into ordered, ID'd, provenance-tagged prompts.

    Args:
        subtag: Bare primary-subtag key into
            :data:`audio_harness.prompt_suite.LOCALES`.
        sentences: Sentences in the order they should be written out.
        source_files: Names of the source ``.txt`` files sentences were
            drawn from, recorded in each prompt's provenance string.

    Returns:
        One :class:`SuitePrompt` per sentence.
    """
    language = LOCALES[subtag]
    source = f"{SOURCE_REPO}/{subtag} [{', '.join(sorted(source_files))}]"
    return [
        SuitePrompt(
            prompt_id=f"general-{index:04d}",
            text=sentence,
            language=language,
            category=CATEGORY,
            license=LICENSE,
            source=source,
        )
        for index, sentence in enumerate(sentences)
    ]
