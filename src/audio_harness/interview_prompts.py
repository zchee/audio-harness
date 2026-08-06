"""The "interview" layer of the TTS prompt suite (plan step P2.12a).

Interview-agent questions are the harness' own IP — no license or
attribution concerns, unlike the Common Voice or entity-template layers. Only
English is authored here: the plan explicitly treats non-English interview
questions as a follow-up task for a stronger multilingual lane, so every
other locale gets an ``interview.PENDING.md`` marker instead of a
low-confidence machine translation. ``tools/gen_interview_prompts.py`` is the
CLI shell around :func:`generate`.
"""

from __future__ import annotations

from pathlib import Path

from .prompt_suite import LOCALES, SuitePrompt, flatten_to_prompts_txt, write_jsonl

CATEGORY = "interview"
LICENSE = "own-IP"
SOURCE = "hand-authored 2026-08-06 (audio-harness interview-agent domain)"

INTERVIEW_QUESTIONS_EN: tuple[str, ...] = (
    "Hi there, thanks for making time today. How are you doing?",
    "Before we start, could you tell me your full name?",
    "Great, and what's the best phone number to reach you at?",
    "Could you give me a quick summary of your current role?",
    "How many years of experience do you have in this field?",
    "What made you decide to apply for this position?",
    "Walk me through a project you're especially proud of.",
    "What was your specific contribution to that project?",
    "Tell me about a time you disagreed with a teammate. How did you handle it?",
    "Describe a situation where you had to learn something new very quickly.",
    "What's the most difficult technical problem you've solved recently?",
    "How do you usually prioritize when you have several deadlines at once?",
    "Tell me about a time a project didn't go as planned. What happened?",
    "What did you learn from that experience?",
    "How would a former manager describe your working style?",
    "What kind of environment helps you do your best work?",
    "Are you currently working with a team, or mostly independently?",
    "How do you handle feedback that you initially disagree with?",
    "Tell me about a time you had to explain something technical to a non-expert.",
    "What tools or technologies have you been using most recently?",
    "Where do you see yourself professionally in about three years?",
    "What's something you're actively trying to improve right now?",
    "How many people were on the largest team you've worked with?",
    "When would you be available to start, if this moves forward?",
    "Do you have any scheduling constraints I should know about?",
    "What questions do you have for me so far?",
    "Is there anything important about your background we haven't covered?",
    "On a scale of one to ten, how confident do you feel about this role?",
    "Before we wrap up, is there anything you'd like to add?",
    "Thanks again for your time today. We'll follow up within a week.",
)
"""Hand-authored, ~30 questions spanning warm-up, background, behavioral,
situational and closing turns — deliberately varied in length and
punctuation so TTS engines see realistic prosody, and including a few
naturally number/date-bearing lines (phone number, years, scale, timeline)
useful once this layer starts feeding the entity/roundtrip lanes."""


def build_prompts(language: str, questions: tuple[str, ...]) -> list[SuitePrompt]:
    """Turn a flat question list into ordered, ID'd suite prompts.

    Args:
        language: Full BCP-47 tag for every prompt.
        questions: Questions in speaking order.

    Returns:
        One :class:`SuitePrompt` per question, IDs zero-padded and stable
        across regenerations as long as ``questions`` doesn't reorder.
    """
    return [
        SuitePrompt(
            prompt_id=f"interview-{index:04d}",
            text=question,
            language=language,
            category=CATEGORY,
            license=LICENSE,
            source=SOURCE,
        )
        for index, question in enumerate(questions)
    ]


PENDING_MARKER = """\
MACHINE-DRAFT-PENDING

No interview-question set for {language} yet. The plan (P2.12a) treats
non-English interview questions as a follow-up task requiring a stronger
multilingual lane, not a low-confidence machine translation of the English
set. Do not backfill this file with a translation; replace it with a
reviewed, locale-native question set once that follow-up lands.

See: .omc/plans/2026-08-06-benchmark-expansion-v2.md, plan step P2.12a.
"""


def write_pending_marker(path: Path, language: str) -> Path:
    """Write a MACHINE-DRAFT-PENDING marker for a locale with no real questions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PENDING_MARKER.format(language=language), encoding="utf-8")
    return path


def generate(out_dir: Path) -> tuple[Path, list[Path]]:
    """Write the English interview set and pending markers for every other locale.

    Args:
        out_dir: Root directory holding ``prompts-<lang>/`` subdirectories.

    Returns:
        The English JSONL path, and the list of pending-marker paths written.
    """
    en_prompts = build_prompts(LOCALES["en"], INTERVIEW_QUESTIONS_EN)
    en_dir = out_dir / "prompts-en"
    jsonl_path = write_jsonl(en_dir / f"{CATEGORY}.jsonl", en_prompts)
    flatten_to_prompts_txt(en_prompts, en_dir / f"{CATEGORY}.txt")

    pending: list[Path] = []
    for subtag, language in LOCALES.items():
        if subtag == "en":
            continue
        marker = write_pending_marker(
            out_dir / f"prompts-{subtag}" / f"{CATEGORY}.PENDING.md", language
        )
        pending.append(marker)
    return jsonl_path, pending
