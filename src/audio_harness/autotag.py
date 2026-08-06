"""Automatic entity annotation of corpus references.

The entity scorer (``entities.py``) needs tagged references, and hand-tagging
thousands of saved clips is not going to happen. Two sources fill the gap:

* **Gold slots** where the corpus already carries them: Speech-MASSIVE's
  ``annot_utt`` field annotates spans as ``[slot : value]`` in every
  language, and human slot labels beat any tagger. Slots that map onto the
  harness's entity classes become tags; the rest stay plain text.
* **Deterministic rules** everywhere else. Numbers are detected with the
  scoring normalizer itself — a span is a number precisely when the P0.2
  lexicon folds it to digits — so tagging can never disagree with scoring.
  Dates, currency and IDs are per-language regex tables.

Proper names are rule-based, not model-based, by explicit choice:
Speech-MASSIVE references are lowercased, which defeats both capitalization
heuristics and NER models trained on cased text — and its gold ``person`` /
``artist_name`` / ``place_name`` slots are already better than a model. The
cased pipecat corpus gets a capitalization heuristic instead. No model
download, no model license, fully reproducible.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from .entities import strip_tags
from .normalize import (
    _KO_COUNTERS,
    _KO_NATIVE,
    _KO_NATIVE_COUNTERS,
    _KO_NUMERAL_STOPWORDS,
    normalizer_for,
)

MASSIVE_SLOT_CLASSES: dict[str, str] = {
    "date": "date",
    # Times are temporal expressions; the harness folds them into the date
    # class rather than leaving them untagged.
    "time": "date",
    "person": "name",
    "artist_name": "name",
    "business_name": "name",
    "place_name": "name",
    "currency_name": "currency",
}
"""Speech-MASSIVE slots that map onto harness entity classes. The mapping is
deliberately conservative: media titles, genres and descriptors are proper
nouns of a sort, but scoring them as names would drown the class in long
free-text spans."""

_MASSIVE_SLOT = re.compile(r"\[\s*([a-z_]+)\s*:\s*([^\]]*?)\s*\]")

_PRIORITY = {"gold": -1, "currency": 0, "date": 1, "id": 2, "number": 3, "name": 4}

_DIGITS_ONLY = re.compile(r"\d+")
_TOKEN = re.compile(r"\S+")
_MAX_NUMBER_TOKENS = 6

# A bare indefinite article folds to 1 in these languages ("un minuteur"),
# which is correct for edit distance but wrong as a tagged number entity.
_ARTICLES = {
    "fr": frozenset({"un", "une"}),
    "de": frozenset({"ein", "eine", "einen", "einem", "einer", "eines"}),
    "es": frozenset({"un", "una"}),
}

_MONTHS = {
    "en": [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ],
    "de": [
        "januar",
        "februar",
        "märz",
        "april",
        "mai",
        "juni",
        "juli",
        "august",
        "september",
        "oktober",
        "november",
        "dezember",
    ],
    "fr": [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ],
    "es": [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ],
}

# Month words that are everyday words too; alone they are not a date.
_AMBIGUOUS_MONTHS = {"en": {"may"}, "fr": {"mars"}, "de": {"mai"}, "es": {"mayo"}}

_CURRENCY_SYMBOL = re.compile(r"[$€£¥]\s?\d[\d.,]*|\d[\d.,]*\s?[$€£¥]")
_CURRENCY_WORDS = {
    "en": (
        "dollars",
        "dollar",
        "euros",
        "euro",
        "pounds",
        "pound",
        "cents",
        "cent",
        "yen",
    ),
    "de": ("euros", "euro", "dollars", "dollar", "cents", "cent", "pfund"),
    "fr": ("euros", "euro", "dollars", "dollar", "centimes", "centime", "livres"),
    "es": (
        "euros",
        "euro",
        "dólares",
        "dólar",
        "dolares",
        "dolar",
        "céntimos",
        "centavos",
        "libras",
    ),
}

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_NUMERIC_DATE = re.compile(r"\b\d{1,2}[/.]\d{1,2}(?:[/.]\d{2,4})?\b")
_MIXED_ID = re.compile(
    r"\b(?=[0-9a-z]*\d)(?=[0-9a-z]*[a-z])[0-9a-z]{3,}\b", re.IGNORECASE
)
_LONG_DIGITS = re.compile(r"\b\d{7,}\b")

_KO_SINO_RUN = r"[영공일이삼사오육칠팔구십백천만억]+"
# A counter may be followed by a particle (십오일에, 시간이) but not by more
# stem hangul — 시작 must not read as the counter 시 plus 작.
_KO_AFTER = r"(?=[에은는이가을를도로의와과부까마쯤]|[^가-힣]|$)"
_KO_DATE = re.compile(rf"(?<![가-힣])(?:{_KO_SINO_RUN}|\d+)\s?[년월일]{_KO_AFTER}")
_KO_CURRENCY = re.compile(rf"(?<![가-힣])(?:{_KO_SINO_RUN}|\d+)\s?원{_KO_AFTER}")
_KO_OTHER_COUNTERS = "|".join(
    counter
    for counter in dict.fromkeys(_KO_COUNTERS + _KO_NATIVE_COUNTERS)
    if counter not in {"년", "월", "일", "원"}
)
_KO_NUMBER = re.compile(
    rf"(?<![가-힣])(?:{_KO_SINO_RUN}|{'|'.join(_KO_NATIVE)})\s?"
    rf"(?:{_KO_OTHER_COUNTERS}){_KO_AFTER}"
)

_CAP_WORD = re.compile(r"^[A-ZÀ-Þ][a-zà-þ'’]+$")  # noqa: RUF001 - vendors emit typographic apostrophes in names
_SENTENCE_END = re.compile(r"[.!?]$")


@dataclass(slots=True, frozen=True)
class _Span:
    """One candidate entity span over the raw reference text."""

    start: int
    end: int
    label: str
    priority: int


def annotate_massive(annot_utt: str, language: str) -> str:
    """Annotate a Speech-MASSIVE reference from its gold slot annotation.

    Mapped slots become entity tags; unmapped slots dissolve into plain text.
    The untagged remainder still goes through the rule pass, because MASSIVE
    has no generic number slot — "quatre omelette" tags nothing, yet the
    quantity is exactly what entity-WER should watch.

    Args:
        annot_utt: MASSIVE ``[slot : value]`` annotated utterance.
        language: BCP-47 tag of the utterance.

    Returns:
        The reference with inline entity tags; tag-free when nothing matched.
    """
    plain: list[str] = []
    gold: list[_Span] = []
    position = 0
    offset = 0
    for match in _MASSIVE_SLOT.finditer(annot_utt):
        plain.append(annot_utt[position : match.start()])
        offset += match.start() - position
        value = match.group(2)
        label = MASSIVE_SLOT_CLASSES.get(match.group(1))
        if label is not None and value:
            gold.append(_Span(offset, offset + len(value), label, _PRIORITY["gold"]))
        plain.append(value)
        offset += len(value)
        position = match.end()
    plain.append(annot_utt[position:])
    text = "".join(plain)

    spans = _resolve(gold + _rule_spans(text, language))
    return _render(text, spans)


def autotag_reference(reference: str, language: str) -> str:
    """Annotate a plain reference with rule-detected entities.

    Args:
        reference: Untagged reference transcript.
        language: BCP-47 tag of the reference.

    Returns:
        The reference with inline entity tags; unchanged when nothing matched.
    """
    return _render(reference, _resolve(_rule_spans(reference, language)))


def annotate_record(
    record: dict[str, object],
    gold: Mapping[tuple[str, str], str] | None = None,
    *,
    force: bool = False,
) -> bool:
    """Set ``reference_annotated`` on one saved-result record, in place.

    Gold slot annotations are used when the record's ``(language, clip_id)``
    has one — but only after verifying that the annotation's plain text is
    the record's reference. A drifted annotation would mis-place every span,
    so on mismatch the rule pass over the actual reference wins instead.

    Args:
        record: One JSONL record from ``stt-results.jsonl``.
        gold: Optional ``(language, clip_id)`` → MASSIVE ``annot_utt`` map.
        force: Re-annotate even when ``reference_annotated`` is present.

    Returns:
        Whether the record now carries an annotation with at least one tag.
    """
    reference = record.get("reference")
    if not isinstance(reference, str) or not reference:
        return False
    if record.get("reference_annotated") and not force:
        return True

    language = str(record.get("language") or "")
    annotated: str | None = None
    if gold is not None:
        annot_utt = gold.get((language, str(record.get("clip_id"))))
        if annot_utt:
            candidate = annotate_massive(annot_utt, language)
            if _same_text(strip_tags(candidate), reference):
                annotated = candidate
    if annotated is None:
        annotated = autotag_reference(reference, language)

    if annotated == reference:
        if force:
            record.pop("reference_annotated", None)
        return False
    record["reference_annotated"] = annotated
    return True


def _same_text(left: str, right: str) -> bool:
    """Whether two references differ only in whitespace."""
    return " ".join(left.split()) == " ".join(right.split())


def _rule_spans(text: str, language: str) -> list[_Span]:
    """Collect deterministic entity candidates for one reference."""
    primary = language.split("-", 1)[0].lower()
    spans: list[_Span] = []
    if primary == "ko":
        for pattern, label in (
            (_KO_DATE, "date"),
            (_KO_CURRENCY, "currency"),
            (_KO_NUMBER, "number"),
        ):
            spans += [
                span
                for span in _regex_spans(text, pattern, label)
                if _ko_licensed(text[span.start : span.end])
            ]
    spans += _regex_spans(text, _YEAR, "date")
    spans += _regex_spans(text, _NUMERIC_DATE, "date")
    spans += _month_spans(text, primary)
    spans += _regex_spans(text, _CURRENCY_SYMBOL, "currency")
    spans += _regex_spans(text, _MIXED_ID, "id")
    spans += _regex_spans(text, _LONG_DIGITS, "id")
    spans += _with_currency_words(text, _number_spans(text, language, primary), primary)
    spans += _name_spans(text, primary)
    return spans


def _with_currency_words(text: str, numbers: list[_Span], primary: str) -> list[_Span]:
    """Promote a number to currency when a currency word follows it.

    "vierhundertzwanzig euro" is an amount, not a bare number, and the
    reference side of the corpus spells the unit rather than the symbol.
    """
    words = _CURRENCY_WORDS.get(primary)
    if not words:
        return numbers
    alternation = "|".join(words)
    promoted: list[_Span] = []
    for span in numbers:
        trailing = re.match(rf"\s+({alternation})\b", text[span.end :], re.IGNORECASE)
        if trailing is None:
            promoted.append(span)
        else:
            promoted.append(
                _Span(
                    span.start,
                    span.end + trailing.end(),
                    "currency",
                    _PRIORITY["currency"],
                )
            )
    return promoted


def _regex_spans(text: str, pattern: re.Pattern[str], label: str) -> list[_Span]:
    return [
        _Span(m.start(), m.end(), label, _PRIORITY[label])
        for m in pattern.finditer(text)
    ]


def _ko_licensed(span_text: str) -> bool:
    """Apply the same fold guards the Korean normalizer uses.

    A multi-syllable Sino run without a multiplier is almost always an
    ordinary word (사이, 이사), and the stopwords are common words spelled
    entirely in numeral syllables — 만일 "in case" parses as run 만 plus the
    day counter 일, so the attached form is checked against the stopword
    table too.
    """
    if span_text.replace(" ", "") in _KO_NUMERAL_STOPWORDS:
        return False
    run = re.match(_KO_SINO_RUN, span_text)
    if run is None:
        return True
    numerals = run.group(0)
    if numerals in _KO_NUMERAL_STOPWORDS:
        return False
    return len(numerals) == 1 or any(char in "십백천만억" for char in numerals)


def _month_spans(text: str, primary: str) -> list[_Span]:
    """Tag month names, requiring context for the ambiguous ones.

    "may" and "mars" are everyday words; they only read as months next to a
    day or year ("may third", "3 mars"). Unambiguous month names tag alone.
    """
    months = _MONTHS.get(primary)
    if not months:
        return []
    ambiguous = _AMBIGUOUS_MONTHS.get(primary, set())
    alternation = "|".join(months)
    day = r"\d{1,2}(?:st|nd|rd|th)?"
    with_context = re.compile(
        rf"\b(?:{day}\s+(?:of\s+)?)?({alternation})(?:\s+(?:the\s+)?{day})?\b",
        re.IGNORECASE,
    )
    spans: list[_Span] = []
    for match in with_context.finditer(text):
        bare = match.group(0).lower() == match.group(1).lower()
        if bare and match.group(1).lower() in ambiguous:
            continue
        spans.append(_Span(match.start(), match.end(), "date", _PRIORITY["date"]))
    return spans


def _number_spans(text: str, language: str, primary: str) -> list[_Span]:
    """Detect number phrases with the scoring normalizer itself.

    A window of tokens is a number exactly when the language's normalizer
    folds it to a digit string — the same fold the metric applies — so the
    tagger and the scorer can never disagree about what a number is.
    """
    normalize = normalizer_for(language)
    articles = _ARTICLES.get(primary, frozenset())
    tokens = list(_TOKEN.finditer(text))
    spans: list[_Span] = []
    index = 0
    while index < len(tokens):
        matched = None
        for stop in range(min(index + _MAX_NUMBER_TOKENS, len(tokens)), index, -1):
            window = text[tokens[index].start() : tokens[stop - 1].end()]
            if stop == index + 1 and _trim(window).lower() in articles:
                continue
            if _DIGITS_ONLY.fullmatch(normalize(window)):
                matched = stop
                break
        if matched is None:
            index += 1
            continue
        start = tokens[index].start()
        end = tokens[matched - 1].end()
        end -= len(text[start:end]) - len(_trim(text[start:end]))
        spans.append(_Span(start, end, "number", _PRIORITY["number"]))
        index = matched
    return spans


def _trim(token: str) -> str:
    """Strip trailing punctuation so tags do not swallow a comma."""
    while token and unicodedata.category(token[-1]).startswith("P"):
        token = token[:-1]
    return token


def _name_spans(text: str, primary: str) -> list[_Span]:
    """Tag capitalized runs as proper names, on cased references only.

    Lowercased corpora carry no case signal, and German capitalizes every
    noun, so both are excluded — for those, names come from gold slots or
    not at all. Sentence-initial capitals are skipped because they say
    nothing about the word.
    """
    if primary == "de" or text == text.lower():
        return []
    months = set(_MONTHS.get(primary, ()))
    spans: list[_Span] = []
    run_start: int | None = None
    run_end = 0
    previous_end = ""
    for index, token in enumerate(_TOKEN.finditer(text)):
        word = _trim(token.group(0))
        sentence_initial = index == 0 or bool(_SENTENCE_END.search(previous_end))
        is_name = (
            bool(_CAP_WORD.match(word))
            and word.lower() not in months
            and not (sentence_initial and run_start is None)
        )
        if is_name:
            if run_start is None:
                run_start = token.start()
            run_end = token.start() + len(word)
        else:
            if run_start is not None:
                spans.append(_Span(run_start, run_end, "name", _PRIORITY["name"]))
                run_start = None
        previous_end = token.group(0)
    if run_start is not None:
        spans.append(_Span(run_start, run_end, "name", _PRIORITY["name"]))
    return spans


def _resolve(spans: list[_Span]) -> list[_Span]:
    """Keep the best non-overlapping spans: priority first, then length."""
    chosen: list[_Span] = []
    for span in sorted(spans, key=lambda s: (s.priority, s.start - s.end, s.start)):
        if span.end <= span.start:
            continue
        if all(span.end <= kept.start or span.start >= kept.end for kept in chosen):
            chosen.append(span)
    return sorted(chosen, key=lambda s: s.start)


def _render(text: str, spans: list[_Span]) -> str:
    """Insert tags around the chosen spans, right to left."""
    for span in reversed(spans):
        text = (
            f"{text[: span.start]}<{span.label}>{text[span.start : span.end]}"
            f"</{span.label}>{text[span.end :]}"
        )
    return text
