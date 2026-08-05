"""Transcript normalization.

Vendors disagree about casing, punctuation, digit formatting and full-width
characters. Comparing raw transcripts therefore measures formatting policy
rather than recognition quality, so both reference and hypothesis are pushed
through the same normalizer before any edit distance is computed.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable

_PUNCT_CATEGORIES = frozenset({"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Sm", "Sk"})
_WHITESPACE_RUN = re.compile(r"\s+")

_ENGLISH_CONTRACTIONS: dict[str, str] = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
}

_CURRENCY_WORDS: dict[str, str] = {
    "$": "dollars",
    "¥": "yen",
    "€": "euros",
    "£": "pounds",
}
_CURRENCY_PATTERN = re.compile(r"([$¥€£])\s?([\d.,]+)")

_UNITS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES: dict[str, int] = {"thousand": 1000, "million": 1_000_000, "billion": 10**9}
_NUMBER_WORDS = frozenset(_UNITS) | frozenset(_TENS) | frozenset(_SCALES) | {"hundred"}
_DIGIT_RUN = re.compile(r"^\d$")

_ORDINALS: dict[str, str] = {
    "first": "1",
    "second": "2",
    "third": "3",
    "fourth": "4",
    "fifth": "5",
    "sixth": "6",
    "seventh": "7",
    "eighth": "8",
    "ninth": "9",
    "tenth": "10",
    "eleventh": "11",
    "twelfth": "12",
    "thirteenth": "13",
    "fourteenth": "14",
    "fifteenth": "15",
    "sixteenth": "16",
    "seventeenth": "17",
    "eighteenth": "18",
    "nineteenth": "19",
    "twentieth": "20",
    "thirtieth": "30",
    "fortieth": "40",
    "fiftieth": "50",
}
_DIGIT_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$")


def _spell_currency(text: str) -> str:
    """Rewrite ``$420`` as ``420 dollars`` so symbols and words align.

    Recognizers emit the symbol; corpora usually spell the unit. Moving both
    onto the spoken form keeps the comparison about the number.
    """
    return _CURRENCY_PATTERN.sub(
        lambda m: f"{m.group(2)} {_CURRENCY_WORDS[m.group(1)]}", text
    )


def _words_to_digits(tokens: list[str]) -> list[str]:
    """Fold spelled-out cardinals into digit strings.

    Speech corpora write "four hundred and twenty"; recognizers return "420".
    Both sides are folded to the digit form so the edit distance reflects what
    was heard rather than how each side chose to write it.

    Only genuine compositions combine. Two bare units in a row are a dictated
    digit sequence, not a sum: "nine one seven three" is an account number that
    must stay 9, 1, 7, 3 for :func:`_merge_digit_runs` to join into "9173" —
    summing them would silently produce "20".

    Args:
        tokens: Whitespace-separated tokens.

    Returns:
        Tokens with number phrases replaced by their numeric value.
    """
    output: list[str] = []
    total = 0
    current = 0
    active = False
    saw_unit = False
    saw_tens = False

    def flush() -> None:
        nonlocal total, current, active, saw_unit, saw_tens
        if active:
            output.append(str(total + current))
        total = current = 0
        active = saw_unit = saw_tens = False

    for index, token in enumerate(tokens):
        if token in _UNITS:
            if saw_unit:
                flush()
            current += _UNITS[token]
            active = True
            saw_unit = True
        elif token in _TENS:
            if saw_unit or saw_tens:
                flush()
            current += _TENS[token]
            active = True
            saw_tens = True
        elif token == "hundred" and active:
            current = (current or 1) * 100
            saw_unit = saw_tens = False
        elif token in _SCALES and active:
            total += (current or 1) * _SCALES[token]
            current = 0
            saw_unit = saw_tens = False
        elif token == "and" and active and _next_is_number(tokens, index):
            continue
        else:
            flush()
            output.append(token)
    flush()
    return output


def _normalize_ordinals(tokens: list[str]) -> list[str]:
    """Reduce ordinals to their cardinal digits.

    Dates are the common case: a corpus writes "February third" where a
    recognizer writes "February 3rd". Mapping both to "3" removes a difference
    that has nothing to do with what was said.
    """
    return [_ORDINALS.get(token, _DIGIT_ORDINAL.sub(r"\1", token)) for token in tokens]


def _next_is_number(tokens: list[str], index: int) -> bool:
    """Whether the token after ``index`` continues a number phrase."""
    return index + 1 < len(tokens) and tokens[index + 1] in _NUMBER_WORDS


def _merge_digit_runs(tokens: list[str]) -> list[str]:
    """Join runs of two or more single digits into one token.

    Account and phone numbers are dictated digit by digit — "nine one seven
    three" — and returned as "9173". Collapsing runs on both sides makes those
    agree instead of scoring as four errors.
    """
    output: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if len(run) >= 2:
            output.append("".join(run))
        else:
            output.extend(run)
        run.clear()

    for token in tokens:
        if _DIGIT_RUN.match(token):
            run.append(token)
        else:
            flush()
            output.append(token)
    flush()
    return output


def _strip_punctuation(text: str) -> str:
    """Drop punctuation and symbol characters, keeping letters and digits."""
    return "".join(
        char for char in text if unicodedata.category(char) not in _PUNCT_CATEGORIES
    )


def normalize_english(text: str) -> str:
    """Normalize an English transcript for word-level comparison.

    Folds case and width, expands contractions, strips punctuation, and puts
    numbers into a single canonical form. Without the number step the score is
    dominated by inverse text normalization — a corpus writing "four hundred
    and twenty dollars" against a recognizer writing "$420" scores five errors
    on a phrase it heard perfectly.

    Args:
        text: Raw transcript.

    Returns:
        Normalized transcript with single-space token separators.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    for contraction, expansion in _ENGLISH_CONTRACTIONS.items():
        text = text.replace(contraction, expansion)
    text = _spell_currency(text)
    text = text.replace(",", "")
    text = _strip_punctuation(text)

    tokens = _WHITESPACE_RUN.sub(" ", text).strip().split()
    tokens = _normalize_ordinals(tokens)
    return " ".join(_merge_digit_runs(_words_to_digits(tokens)))


def normalize_japanese(text: str) -> str:
    """Normalize a Japanese transcript for character-level comparison.

    NFKC folds full-width Latin and half-width kana onto canonical forms.
    All whitespace is removed because Japanese has no orthographic word
    boundaries and vendors insert spaces inconsistently.

    Args:
        text: Raw transcript.

    Returns:
        Normalized transcript with no whitespace.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _strip_punctuation(text)
    return _WHITESPACE_RUN.sub("", text).strip()


def normalize_generic(text: str) -> str:
    """Normalize a transcript in a language the harness has no rules for."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _strip_punctuation(text)
    return _WHITESPACE_RUN.sub(" ", text).strip()


_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "en": normalize_english,
    "ja": normalize_japanese,
}


def normalizer_for(language: str) -> Callable[[str], str]:
    """Return the normalizer for a BCP-47 language tag.

    Args:
        language: Tag such as ``en-US`` or ``ja-JP``; only the primary subtag
            is consulted.

    Returns:
        The language-specific normalizer, or the generic one when the language
        has no dedicated rules.
    """
    primary = language.split("-", 1)[0].lower()
    return _NORMALIZERS.get(primary, normalize_generic)


def uses_character_metric(language: str) -> bool:
    """Whether accuracy for a language should be scored per character.

    Word error rate needs reliable word boundaries. For languages written
    without spaces, tokenization would dominate the score, so character error
    rate is the honest primary metric.
    """
    return language.split("-", 1)[0].lower() in {"ja", "zh", "th", "lo", "my", "km"}
