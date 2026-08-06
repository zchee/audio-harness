"""Transcript normalization.

Vendors disagree about casing, punctuation, digit formatting and full-width
characters. Comparing raw transcripts therefore measures formatting policy
rather than recognition quality, so both reference and hypothesis are pushed
through the same normalizer before any edit distance is computed.

Rules live in per-language tables — number lexicons, currency words, counter
lists — driven by shared engines, with one module-level registry keyed by
BCP-47 primary subtag. A new language is a new table entry, not a new
algorithm.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

_PUNCT_CATEGORIES = frozenset({"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Sm", "Sk"})
_WHITESPACE_RUN = re.compile(r"\s+")
_DIGIT_RUN = re.compile(r"^\d$")


@dataclass(slots=True, frozen=True)
class NumberLexicon:
    """Number-word tables for one language's words-to-digits folding.

    Attributes:
        units: Additive words below twenty (plus lexicalized forms such as
            Spanish ``veintiuno``), mapped to their value.
        tens: Words for 20-90, including lexicalized irregulars such as
            French ``quatrevingtdix`` after phrase replacement.
        hundreds: Words carrying an absolute hundreds value (``doscientos``).
        hundred_words: Multiplier words (``hundred``, ``cent``, ``hundert``).
        scales: Words that multiply the accumulated value (thousand and up).
        connectors: Words joining number phrases (``and``, ``et``, ``y``,
            ``und``) that must vanish inside a number but survive outside one.
        unit_then_tens: Whether a connector composes unit-before-tens, the
            German ``einundzwanzig`` ordering. English keeps this off so a
            dictated "nineteen eighty" stays two numbers.
        hundred_standalone: Whether a bare hundred word is a number. French
            says ``cent vingt`` for 120; English "hundred" alone is prose.
        scales_standalone: Whether a bare scale word is a number. Spanish and
            French say ``mil``/``mille`` for 1000 with no leading one.
    """

    units: Mapping[str, int]
    tens: Mapping[str, int]
    hundreds: Mapping[str, int] = field(default_factory=dict)
    hundred_words: frozenset[str] = frozenset()
    scales: Mapping[str, int] = field(default_factory=dict)
    connectors: frozenset[str] = frozenset()
    unit_then_tens: bool = False
    hundred_standalone: bool = False
    scales_standalone: bool = False

    def is_number_word(self, token: str) -> bool:
        """Whether a token can continue a number phrase."""
        return (
            token in self.units
            or token in self.tens
            or token in self.hundreds
            or token in self.hundred_words
            or token in self.scales
        )


@dataclass(slots=True, frozen=True)
class WordRules:
    """Normalization tables for one space-delimited language.

    Attributes:
        lexicon: Number-word tables for digit folding.
        replacements: Ordered verbatim substring rewrites applied before
            tokenization — English contractions, where substring semantics
            ("n't" inside a word) are the point.
        currency_words: Currency symbol to spoken unit, so ``$420`` and
            "four hundred twenty dollars" meet in the middle.
        currency_lead: Compiled symbol-before-amount pattern.
        currency_trail: Compiled amount-before-symbol pattern (``420 €``).
        regex_replacements: Boundary-anchored rewrites applied after
            ``replacements`` — the French vigesimal joins live here because a
            plain substring replace would fire across an earlier join
            (``quatrevingtdix sept`` contains ``dix sept`` but is not 17).
        ordinals: Ordinal words reduced to cardinal digits.
        ordinal_suffix: Digit-ordinal suffix pattern (``3rd`` → ``3``).
        fold_hyphens: Whether hyphenated groups are folded before other
            rules. Golden-diff evidence (de/fr, 2026-08-06): vendors hyphenate
            where references space ("To-do-Liste" / "to do liste",
            "Pourrais-tu" / "pourrais tu"), so the default fold is a split —
            except that groups with a single-letter fragment ("E-Mail",
            "ge-e-mailed") or a bound affix are spelling variants of the
            joined word and must join. English keeps its historical behavior
            of dropping the hyphen outright at punctuation stripping.
        hyphen_join_fragments: Bound affixes that force a hyphen group to
            join — German "ge-" never stands alone, so "ge-emailt" is
            "geemailt", not "ge emailt".
        number_morphemes: Longest-first morphemes for decomposing agglutinated
            number tokens — German writes ``vierhundertzwanzig`` as one word.
    """

    lexicon: NumberLexicon
    replacements: tuple[tuple[str, str], ...] = ()
    regex_replacements: tuple[tuple[re.Pattern[str], str], ...] = ()
    currency_words: Mapping[str, str] = field(default_factory=dict)
    currency_lead: re.Pattern[str] | None = None
    currency_trail: re.Pattern[str] | None = None
    ordinals: Mapping[str, str] = field(default_factory=dict)
    ordinal_suffix: re.Pattern[str] | None = None
    fold_hyphens: bool = False
    hyphen_join_fragments: frozenset[str] = frozenset()
    number_morphemes: tuple[str, ...] = ()


_HYPHEN_GROUP = re.compile(r"\w+(?:-\w+)+")


def _fold_hyphens(text: str, join_fragments: frozenset[str]) -> str:
    """Fold hyphenated groups: split by default, join spelling variants.

    A hyphen usually stands where the reference put a space ("To-do-Liste"
    vs "to do liste"), so splitting is the default. A single-letter fragment
    ("E-Mail") or a bound affix ("ge-emailt") marks the group as a hyphenated
    spelling of one word, where only joining matches the unhyphenated side.
    """

    def fold(match: re.Match[str]) -> str:
        fragments = match.group(0).split("-")
        join = any(
            len(fragment) == 1 or fragment in join_fragments for fragment in fragments
        )
        return ("" if join else " ").join(fragments)

    return _HYPHEN_GROUP.sub(fold, text)


def _currency_patterns(
    words: Mapping[str, str],
) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Compile both amount/symbol orders for a currency word table."""
    symbols = "".join(map(re.escape, words))
    return (
        re.compile(rf"([{symbols}])\s?([\d.,]+)"),
        re.compile(rf"([\d.,]+)\s?([{symbols}])"),
    )


def _spell_currency(text: str, rules: WordRules) -> str:
    """Rewrite ``$420`` and ``420 €`` as ``420 dollars``/``420 euro``.

    Recognizers emit the symbol; corpora usually spell the unit. Moving both
    onto the spoken form keeps the comparison about the number.
    """
    words = rules.currency_words
    if rules.currency_lead is not None:
        text = rules.currency_lead.sub(
            lambda m: f"{m.group(2)} {words[m.group(1)]}", text
        )
    if rules.currency_trail is not None:
        text = rules.currency_trail.sub(
            lambda m: f"{m.group(1)} {words[m.group(2)]}", text
        )
    return text


def _words_to_digits(tokens: list[str], lex: NumberLexicon) -> list[str]:
    """Fold spelled-out cardinals into digit strings.

    Speech corpora write "four hundred and twenty"; recognizers return "420".
    Both sides are folded to the digit form so the edit distance reflects what
    was heard rather than how each side chose to write it.

    Only genuine compositions combine. Two bare units in a row are a dictated
    digit sequence, not a sum: "nine one seven three" is an account number that
    must stay 9, 1, 7, 3 for :func:`_merge_digit_runs` to join into "9173" —
    summing them would silently produce "20". The one exception is a connector
    between unit and tens in ``unit_then_tens`` languages, where German's
    "ein und zwanzig" genuinely means 21.

    Args:
        tokens: Whitespace-separated tokens.
        lex: Number-word tables for the language.

    Returns:
        Tokens with number phrases replaced by their numeric value.
    """
    output: list[str] = []
    total = 0
    current = 0
    active = False
    saw_unit = False
    saw_tens = False
    pending_connector = False

    def flush() -> None:
        nonlocal total, current, active, saw_unit, saw_tens, pending_connector
        if active:
            output.append(str(total + current))
        total = current = 0
        active = saw_unit = saw_tens = False
        pending_connector = False

    for index, token in enumerate(tokens):
        if token in lex.units:
            if saw_unit:
                flush()
            current += lex.units[token]
            active = True
            saw_unit = True
            pending_connector = False
        elif token in lex.tens:
            if saw_unit and pending_connector and lex.unit_then_tens:
                current += lex.tens[token]
                saw_unit = False
                saw_tens = True
                pending_connector = False
                continue
            if saw_unit or saw_tens:
                flush()
            current += lex.tens[token]
            active = True
            saw_tens = True
            pending_connector = False
        elif token in lex.hundreds:
            if saw_unit or saw_tens:
                flush()
            current += lex.hundreds[token]
            active = True
            pending_connector = False
        elif token in lex.hundred_words and (active or lex.hundred_standalone):
            current = (current or 1) * 100
            saw_unit = saw_tens = False
            pending_connector = False
            active = True
        elif token in lex.scales and (active or lex.scales_standalone):
            total += (current or 1) * lex.scales[token]
            current = 0
            active = True
            saw_unit = saw_tens = False
            pending_connector = False
        elif token in lex.connectors and active and _next_is_number(tokens, index, lex):
            pending_connector = True
            continue
        else:
            flush()
            output.append(token)
    flush()
    return output


def _next_is_number(tokens: list[str], index: int, lex: NumberLexicon) -> bool:
    """Whether the token after ``index`` continues a number phrase."""
    return index + 1 < len(tokens) and lex.is_number_word(tokens[index + 1])


def _decompose_number_words(tokens: list[str], morphemes: tuple[str, ...]) -> list[str]:
    """Split agglutinated number tokens into their morphemes.

    German writes 420 as one word, ``vierhundertzwanzig``. A token is split
    only when the whole token is consumed by number morphemes, so ordinary
    words that merely start with one — ``achtung``, ``elfmeter`` — pass
    through untouched.

    Args:
        tokens: Whitespace-separated tokens.
        morphemes: Number morphemes sorted longest first for greedy matching.

    Returns:
        Tokens with decomposable number words expanded in place.
    """
    output: list[str] = []
    for token in tokens:
        parts: list[str] = []
        rest = token
        while rest:
            for morpheme in morphemes:
                if rest.startswith(morpheme):
                    parts.append(morpheme)
                    rest = rest[len(morpheme) :]
                    break
            else:
                parts = []
                break
        output.extend(parts if len(parts) > 1 else [token])
    return output


def _normalize_ordinals(tokens: list[str], rules: WordRules) -> list[str]:
    """Reduce ordinals to their cardinal digits.

    Dates are the common case: a corpus writes "February third" where a
    recognizer writes "February 3rd". Mapping both to "3" removes a difference
    that has nothing to do with what was said.
    """
    suffix = rules.ordinal_suffix
    return [
        rules.ordinals.get(token, suffix.sub(r"\1", token) if suffix else token)
        for token in tokens
    ]


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


def _normalize_words(text: str, rules: WordRules) -> str:
    """Run the shared word-level pipeline over one language's tables."""
    text = unicodedata.normalize("NFKC", text).lower()
    if rules.fold_hyphens:
        text = _fold_hyphens(text, rules.hyphen_join_fragments)
    for old, new in rules.replacements:
        text = text.replace(old, new)
    for pattern, new in rules.regex_replacements:
        text = pattern.sub(new, text)
    if rules.currency_words:
        text = _spell_currency(text, rules)
    text = text.replace(",", "")
    text = _strip_punctuation(text)

    tokens = _WHITESPACE_RUN.sub(" ", text).strip().split()
    if rules.number_morphemes:
        tokens = _decompose_number_words(tokens, rules.number_morphemes)
    if rules.ordinals or rules.ordinal_suffix:
        tokens = _normalize_ordinals(tokens, rules)
    return " ".join(_merge_digit_runs(_words_to_digits(tokens, rules.lexicon)))


# --- English ---------------------------------------------------------------

_EN_CURRENCY = {"$": "dollars", "¥": "yen", "€": "euros", "£": "pounds"}
_EN_CURRENCY_LEAD, _EN_CURRENCY_TRAIL = _currency_patterns(_EN_CURRENCY)

_ENGLISH_RULES = WordRules(
    lexicon=NumberLexicon(
        units={
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
        },
        tens={
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        },
        hundred_words=frozenset({"hundred"}),
        scales={"thousand": 1000, "million": 1_000_000, "billion": 10**9},
        connectors=frozenset({"and"}),
    ),
    replacements=(
        ("won't", "will not"),
        ("can't", "cannot"),
        ("n't", " not"),
        ("'re", " are"),
        ("'ve", " have"),
        ("'ll", " will"),
        ("'d", " would"),
        ("'m", " am"),
    ),
    currency_words=_EN_CURRENCY,
    currency_lead=_EN_CURRENCY_LEAD,
    currency_trail=_EN_CURRENCY_TRAIL,
    ordinals={
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
    },
    ordinal_suffix=re.compile(r"^(\d+)(?:st|nd|rd|th)$"),
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
    return _normalize_words(text, _ENGLISH_RULES)


# --- German ----------------------------------------------------------------

_DE_CURRENCY = {"€": "euro", "$": "dollar", "£": "pfund"}
_DE_CURRENCY_LEAD, _DE_CURRENCY_TRAIL = _currency_patterns(_DE_CURRENCY)

_DE_UNITS = {
    "null": 0,
    "ein": 1,
    "eins": 1,
    "eine": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "fünf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
    "elf": 11,
    "zwölf": 12,
    "dreizehn": 13,
    "vierzehn": 14,
    "fünfzehn": 15,
    "sechzehn": 16,
    "siebzehn": 17,
    "achtzehn": 18,
    "neunzehn": 19,
}
_DE_TENS = {
    "zwanzig": 20,
    "dreißig": 30,
    "dreissig": 30,
    "vierzig": 40,
    "fünfzig": 50,
    "sechzig": 60,
    "siebzig": 70,
    "achtzig": 80,
    "neunzig": 90,
}
_DE_SCALES = {"tausend": 1000, "million": 1_000_000, "millionen": 1_000_000}

_GERMAN_RULES = WordRules(
    lexicon=NumberLexicon(
        units=_DE_UNITS,
        tens=_DE_TENS,
        hundred_words=frozenset({"hundert"}),
        scales=_DE_SCALES,
        connectors=frozenset({"und"}),
        unit_then_tens=True,
        hundred_standalone=True,
        scales_standalone=True,
    ),
    currency_words=_DE_CURRENCY,
    currency_lead=_DE_CURRENCY_LEAD,
    currency_trail=_DE_CURRENCY_TRAIL,
    fold_hyphens=True,
    # "ge-" is a bound participle prefix, never a free word: "ge-emailt" is a
    # hyphenated spelling of "geemailt", not two words.
    hyphen_join_fragments=frozenset({"ge"}),
    # Longest first, so "sechzehn" wins over "sechs" during greedy splitting.
    number_morphemes=tuple(
        sorted(
            {*_DE_UNITS, *_DE_TENS, "hundert", "tausend", "und"},
            key=len,
            reverse=True,
        )
    ),
)


def normalize_german(text: str) -> str:
    """Normalize a German transcript for word-level comparison.

    German agglutinates numbers into single words ("vierhundertzwanzig") and
    reverses unit and tens ("einundzwanzig" is 21), so folding decomposes the
    token before the shared engine composes the value.

    Args:
        text: Raw transcript.

    Returns:
        Normalized transcript with single-space token separators.
    """
    return _normalize_words(text, _GERMAN_RULES)


# --- French ----------------------------------------------------------------

_FR_CURRENCY = {"€": "euros", "$": "dollars", "£": "livres"}
_FR_CURRENCY_LEAD, _FR_CURRENCY_TRAIL = _currency_patterns(_FR_CURRENCY)

_FRENCH_RULES = WordRules(
    lexicon=NumberLexicon(
        units={
            "zéro": 0,
            "un": 1,
            "une": 1,
            "deux": 2,
            "trois": 3,
            "quatre": 4,
            "cinq": 5,
            "six": 6,
            "sept": 7,
            "huit": 8,
            "neuf": 9,
            "dix": 10,
            "onze": 11,
            "douze": 12,
            "treize": 13,
            "quatorze": 14,
            "quinze": 15,
            "seize": 16,
            # Joined by the regex replacements below; left split, "dix sept"
            # would read as a dictated 10 7.
            "dixsept": 17,
            "dixhuit": 18,
            "dixneuf": 19,
        },
        tens={
            "vingt": 20,
            "trente": 30,
            "quarante": 40,
            "cinquante": 50,
            "soixante": 60,
            # Joined by the replacements below so each folds as one token;
            # left split, "quatre vingt dix" would sum to 34.
            "soixantedix": 70,
            "quatrevingt": 80,
            "quatrevingts": 80,
            "quatrevingtdix": 90,
        },
        hundred_words=frozenset({"cent", "cents"}),
        scales={"mille": 1000, "million": 1_000_000, "millions": 1_000_000},
        connectors=frozenset({"et"}),
        hundred_standalone=True,
        scales_standalone=True,
    ),
    # Boundary-anchored and ordered longest first: "quatre vingt dix" must
    # join before "quatre vingt", and the anchors stop "dix sept" from firing
    # inside an already-joined "quatrevingtdix sept" (which is 97, not 17).
    regex_replacements=(
        (re.compile(r"\bquatre vingt dix\b"), "quatrevingtdix"),
        (re.compile(r"\bquatre vingts\b"), "quatrevingts"),
        (re.compile(r"\bquatre vingt\b"), "quatrevingt"),
        (re.compile(r"\bsoixante dix\b"), "soixantedix"),
        (re.compile(r"\bdix sept\b"), "dixsept"),
        (re.compile(r"\bdix huit\b"), "dixhuit"),
        (re.compile(r"\bdix neuf\b"), "dixneuf"),
    ),
    currency_words=_FR_CURRENCY,
    currency_lead=_FR_CURRENCY_LEAD,
    currency_trail=_FR_CURRENCY_TRAIL,
    fold_hyphens=True,
)


def normalize_french(text: str) -> str:
    """Normalize a French transcript for word-level comparison.

    French hyphenates compound numbers and builds 70/80/90 from smaller words
    ("quatre-vingt-dix"), so hyphens become spaces and the vigesimal phrases
    are joined into single tokens before the shared engine folds them.

    Args:
        text: Raw transcript.

    Returns:
        Normalized transcript with single-space token separators.
    """
    return _normalize_words(text, _FRENCH_RULES)


# --- Spanish ---------------------------------------------------------------

_ES_CURRENCY = {"€": "euros", "$": "dólares", "£": "libras"}
_ES_CURRENCY_LEAD, _ES_CURRENCY_TRAIL = _currency_patterns(_ES_CURRENCY)

_SPANISH_RULES = WordRules(
    lexicon=NumberLexicon(
        units={
            "cero": 0,
            "un": 1,
            "una": 1,
            "uno": 1,
            "dos": 2,
            "tres": 3,
            "cuatro": 4,
            "cinco": 5,
            "seis": 6,
            "siete": 7,
            "ocho": 8,
            "nueve": 9,
            "diez": 10,
            "once": 11,
            "doce": 12,
            "trece": 13,
            "catorce": 14,
            "quince": 15,
            "dieciséis": 16,
            "dieciseis": 16,
            "diecisiete": 17,
            "dieciocho": 18,
            "diecinueve": 19,
            "veintiuno": 21,
            "veintiuna": 21,
            "veintiún": 21,
            "veintiun": 21,
            "veintidós": 22,
            "veintidos": 22,
            "veintitrés": 23,
            "veintitres": 23,
            "veinticuatro": 24,
            "veinticinco": 25,
            "veintiséis": 26,
            "veintiseis": 26,
            "veintisiete": 27,
            "veintiocho": 28,
            "veintinueve": 29,
        },
        tens={
            "veinte": 20,
            "treinta": 30,
            "cuarenta": 40,
            "cincuenta": 50,
            "sesenta": 60,
            "setenta": 70,
            "ochenta": 80,
            "noventa": 90,
        },
        hundreds={
            "cien": 100,
            "ciento": 100,
            "doscientos": 200,
            "doscientas": 200,
            "trescientos": 300,
            "trescientas": 300,
            "cuatrocientos": 400,
            "cuatrocientas": 400,
            "quinientos": 500,
            "quinientas": 500,
            "seiscientos": 600,
            "seiscientas": 600,
            "setecientos": 700,
            "setecientas": 700,
            "ochocientos": 800,
            "ochocientas": 800,
            "novecientos": 900,
            "novecientas": 900,
        },
        scales={
            "mil": 1000,
            "millón": 1_000_000,
            "millon": 1_000_000,
            "millones": 1_000_000,
        },
        connectors=frozenset({"y"}),
        scales_standalone=True,
    ),
    currency_words=_ES_CURRENCY,
    currency_lead=_ES_CURRENCY_LEAD,
    currency_trail=_ES_CURRENCY_TRAIL,
    fold_hyphens=True,
)


def normalize_spanish(text: str) -> str:
    """Normalize a Spanish transcript for word-level comparison.

    Spanish lexicalizes 21-29 and the hundreds ("veintiuno", "quinientos"),
    so those live in the tables as whole words rather than compositions.

    Args:
        text: Raw transcript.

    Returns:
        Normalized transcript with single-space token separators.
    """
    return _normalize_words(text, _SPANISH_RULES)


# --- Japanese --------------------------------------------------------------

# Katakana ァ..ヶ fold onto the hiragana block 0x60 below it. The prolonged
# sound mark ー is shared by both scripts and passes through.
_KANA_FOLD = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}


def fold_kana(text: str) -> str:
    """Fold katakana onto hiragana so script choice scores as identical.

    Vendors disagree about which script to emit — one writes コーヒー where
    another writes こーひー for the same audio. The choice is orthographic
    policy, not recognition, so CER must not charge for it. The fold is a
    scoring equivalence only: it is applied at comparison time rather than
    inside :func:`normalize_japanese`, whose output keeps the script
    distinction a human reader expects.
    """
    return text.translate(_KANA_FOLD)


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


# --- Korean ----------------------------------------------------------------

_KO_SINO_DIGITS = {
    "영": 0,
    "공": 0,
    "일": 1,
    "이": 2,
    "삼": 3,
    "사": 4,
    "오": 5,
    "육": 6,
    "칠": 7,
    "팔": 8,
    "구": 9,
}
_KO_SINO_SMALL = {"십": 10, "백": 100, "천": 1000}
_KO_SINO_LARGE = {"만": 10_000, "억": 100_000_000}

_KO_COUNTERS = (
    "퍼센트",
    "시간",
    "켤레",
    "그릇",
    "마리",
    "가지",
    "월",
    "년",
    "일",
    "시",
    "분",
    "초",
    "개",
    "명",
    "번",
    "원",
    "살",
    "주",
    "달",
    "회",
    "층",
    "도",
    "권",
    "잔",
    "대",
    "병",
    "장",
    "곳",
    "배",
)
"""Counter words that license a numeral fold, longest first so 시간 wins
over 시. The gate exists because most Sino-Korean numeral syllables are also
ordinary morphemes — folding a bare 사 or 이 would corrupt words like 사이."""

_KO_NUMERAL_STOPWORDS = frozenset({"만일", "천사"})
"""Common words spelled entirely in numeral syllables — 만일 "in case",
천사 "angel" — that would otherwise fold when a counter happens to follow."""

# Native-Korean numbers pair with a smaller counter set (hours, people,
# objects); dates and money always take Sino-Korean, so keeping the sets
# separate avoids folding 네 ("your") before 원.
_KO_NATIVE = {
    "스물한": 21,
    "스물두": 22,
    "스물세": 23,
    "스물네": 24,
    "열한": 11,
    "열두": 12,
    "열세": 13,
    "열네": 14,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "스무": 20,
    "한": 1,
    "두": 2,
    "세": 3,
    "네": 4,
    "열": 10,
}
_KO_NATIVE_COUNTERS = (
    "시간",
    "켤레",
    "그릇",
    "마리",
    "가지",
    "시",
    "개",
    "명",
    "번",
    "살",
    "잔",
    "병",
    "대",
    "장",
    "권",
    "곳",
    "배",
)

_KO_COUNTER_ALT = "|".join(_KO_COUNTERS)
_KO_SINO_PATTERN = re.compile(
    rf"(?<![가-힣])([영공일이삼사오육칠팔구십백천만억]+)\s?({_KO_COUNTER_ALT})"
)
_KO_NATIVE_PATTERN = re.compile(
    rf"(?<![가-힣])({'|'.join(_KO_NATIVE)})\s?({'|'.join(_KO_NATIVE_COUNTERS)})"
)
_KO_DIGIT_COUNTER = re.compile(rf"(\d+)\s+({_KO_COUNTER_ALT})")
_KO_MONTH_NAMES = (
    # 6월 and 10월 have irregular month readings that drop a final consonant,
    # so the numeral parser alone can never reach them.
    (re.compile(r"(?<![가-힣])유월(?![가-힣])"), "6월"),
    (re.compile(r"(?<![가-힣])시월(?![가-힣])"), "10월"),
)


def _parse_sino_korean(run: str) -> int:
    """Evaluate a Sino-Korean numeral run positionally (이십삼 → 23)."""
    total = 0
    section = 0
    digit = 0
    for char in run:
        if char in _KO_SINO_DIGITS:
            digit = _KO_SINO_DIGITS[char]
        elif char in _KO_SINO_SMALL:
            section += (digit or 1) * _KO_SINO_SMALL[char]
            digit = 0
        else:
            total += ((section + digit) or 1) * _KO_SINO_LARGE[char]
            section = digit = 0
    return total + section + digit


def _fold_sino_korean(match: re.Match[str]) -> str:
    """Fold one counter-gated numeral run, refusing the ambiguous cases.

    A multi-syllable run without a multiplier (사이, 이사) is almost always an
    ordinary word that happens to be spelled in numeral syllables, so only
    single digits and runs containing 십/백/천/만/억 fold.
    """
    run, counter = match.group(1), match.group(2)
    if run in _KO_NUMERAL_STOPWORDS:
        return match.group(0)
    if len(run) > 1 and not any(char in "십백천만억" for char in run):
        return match.group(0)
    return f"{_parse_sino_korean(run)}{counter}"


def normalize_korean(text: str) -> str:
    """Normalize a Korean transcript for word-level comparison.

    The known blocker is numeral formatting: corpora write 3월 where
    recognizers write 삼월 (and vice versa), which scores as an error on a
    perfectly heard date. Sino-Korean and native-Korean numerals fold to
    digits when a counter word licenses the reading, and a digit already in
    place is attached to its counter so spacing differences are free too.

    Args:
        text: Raw transcript.

    Returns:
        Normalized transcript with single-space token separators.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    for pattern, replacement in _KO_MONTH_NAMES:
        text = pattern.sub(replacement, text)
    text = _KO_SINO_PATTERN.sub(_fold_sino_korean, text)
    text = _KO_NATIVE_PATTERN.sub(
        lambda m: f"{_KO_NATIVE[m.group(1)]}{m.group(2)}", text
    )
    text = _KO_DIGIT_COUNTER.sub(r"\1\2", text)
    text = _strip_punctuation(text)
    return _WHITESPACE_RUN.sub(" ", text).strip()


# --- Registry --------------------------------------------------------------


def normalize_generic(text: str) -> str:
    """Normalize a transcript in a language the harness has no rules for."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _strip_punctuation(text)
    return _WHITESPACE_RUN.sub(" ", text).strip()


def _identity(text: str) -> str:
    """Comparison fold for languages that need none."""
    return text


_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "en": normalize_english,
    "ja": normalize_japanese,
    "ko": normalize_korean,
    "de": normalize_german,
    "fr": normalize_french,
    "es": normalize_spanish,
}

_COMPARISON_FOLDS: dict[str, Callable[[str], str]] = {
    "ja": fold_kana,
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


def comparison_fold_for(language: str) -> Callable[[str], str]:
    """Return the fold applied to both sides right before edit distance.

    Distinct from the normalizer: a normalizer output is the canonical form
    of a transcript, while a comparison fold is an equivalence class used
    only for scoring — Japanese kana folding erases a script distinction a
    human reader still wants to see.

    Args:
        language: BCP-47 tag; only the primary subtag is consulted.

    Returns:
        The fold, or an identity function when the language needs none.
    """
    primary = language.split("-", 1)[0].lower()
    return _COMPARISON_FOLDS.get(primary, _identity)


def uses_character_metric(language: str) -> bool:
    """Whether accuracy for a language should be scored per character.

    Word error rate needs reliable word boundaries. For languages written
    without spaces, tokenization would dominate the score, so character error
    rate is the honest primary metric.
    """
    return language.split("-", 1)[0].lower() in {"ja", "zh", "th", "lo", "my", "km"}
