"""The "entities" layer of the TTS prompt suite (plan step P2.12c).

Generates templated hard cases per locale for every class in
:data:`audio_harness.entities.ENTITY_CLASSES` — numbers, dates, currency
amounts, alphanumeric IDs, and names — sharing that exact tag vocabulary so
these prompts can feed the entity scorer once a roundtrip lane wires them in.

Each hard case is written twice, deliberately in different forms:

* ``text`` is the natural *written* form (digits, currency symbols, a
  localized date string) — what actually gets fed to the TTS engine, so the
  engine has to do its own verbalization instead of being handed the answer.
* ``annotated`` is the canonical *spoken word* form, tagged
  (``<currency>four hundred and twenty dollars</currency>``) the same way
  ``entities.py`` already expects references to look. This is the ground
  truth a future roundtrip-STT hypothesis would be scored against.

Ingredients, one per class of problem:

* **num2words** supplies cardinal/ordinal/currency word forms per locale.
  Its currency converter is not uniform across languages — some treat a
  plain int as "already split" major.minor, Turkish hardcodes Lira and
  takes no currency code, and Vietnamese has no currency converter at all —
  so :func:`spoken_currency` and :data:`WHOLE_UNIT_CURRENCY` document and
  route around each quirk explicitly rather than assuming one calling
  convention works everywhere (verified empirically per locale).
* **Babel (CLDR)** supplies locale-correct date and currency *display*
  formatting (``format_date``, ``format_currency``, ``get_month_names``) for
  the written form.
* **Faker** supplies locale-appropriate names.
* **Templates** are short, simple hand-written carrier sentences per locale
  and per class (below) — much lower-risk than the interview-question layer,
  since these are common, formulaic phrases rather than nuanced dialogue.

Known simplification: the spoken date form is composed uniformly as
"<month name> <ordinal day>" (matching the convention P0's ``entities.py``
already tests for English — "February third") for every locale. This reads
naturally in English; elsewhere it is a defensible approximation, not a
claim of native grammatical correctness — languages with their own day-
counting conventions (e.g. Japanese irregular day readings) are a follow-up,
not silently pretended away. Alphanumeric IDs are digit-only for the same
reason: per-language letter-name pronunciation is out of scope here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
import random

from babel.dates import format_date, get_month_names
from babel.numbers import format_currency
from faker import Faker
import num2words

from .entities import ENTITY_CLASSES
from .prompt_suite import LOCALES, SuitePrompt


CATEGORY = "entities"
LICENSE = "generated"
SOURCE = "tools/gen_entity_prompts.py (num2words + Babel/CLDR + Faker + templates)"

FAKER_LOCALE = {
    "en": "en_US",
    "ja": "ja_JP",
    "ar": "ar_SA",
    "de": "de_DE",
    "es": "es_ES",
    "fr": "fr_FR",
    "hu": "hu_HU",
    "ko": "ko_KR",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "pt": "pt_PT",
    "ru": "ru_RU",
    "tr": "tr_TR",
    "vi": "vi_VN",
}

CURRENCY_CODE = {
    "en": "USD",
    "ja": "JPY",
    "ar": "SAR",
    "de": "EUR",
    "es": "EUR",
    "fr": "EUR",
    "hu": "HUF",
    "ko": "KRW",
    "nl": "EUR",
    "pl": "PLN",
    "pt": "EUR",
    "ru": "RUB",
    "tr": "TRY",
    "vi": "VND",
}

WHOLE_UNIT_CURRENCY = frozenset({"ja", "ko", "vi"})
"""Locales generated with no minor-unit (cents) amount: JPY/KRW have no
minor unit in ordinary use, and VND's is impractically small — this also
sidesteps num2words rejecting fractional JPY/KRW outright."""

# One simple carrier sentence per class per locale. Kept to a single
# template each: formulaic and low-risk, but not proofread by a native
# speaker of every locale, so treat unnatural phrasing as a bug report
# rather than a hidden translation claim.
TEMPLATES: dict[str, dict[str, str]] = {
    "number": {
        "en": "Please confirm the quantity: {}.",
        "ja": "数量は{}でお願いします。",
        "ar": "الرجاء تأكيد الكمية: {}.",
        "de": "Bitte bestätigen Sie die Menge: {}.",
        "es": "Por favor confirme la cantidad: {}.",
        "fr": "Merci de confirmer la quantité : {}.",
        "hu": "Kérjük, erősítse meg a mennyiséget: {}.",
        "ko": "수량을 확인해 주세요: {}.",
        "nl": "Bevestig de hoeveelheid: {}.",
        "pl": "Proszę potwierdzić ilość: {}.",
        "pt": "Por favor confirme a quantidade: {}.",
        "ru": "Пожалуйста, подтвердите количество: {}.",
        "tr": "Lütfen miktarı onaylayın: {}.",  # ruff: ignore[ambiguous-unicode-character-string] - Turkish dotless i
        "vi": "Vui lòng xác nhận số lượng: {}.",
    },
    "date": {
        "en": "The appointment is scheduled for {}.",
        "ja": "予約は{}に予定されています。",
        "ar": "الموعد محدد في {}.",
        "de": "Der Termin ist für {} geplant.",
        "es": "La cita está programada para el {}.",
        "fr": "Le rendez-vous est prévu le {}.",
        "hu": "Az időpont {} napra van tervezve.",
        "ko": "예약은 {}로 예정되어 있습니다.",
        "nl": "De afspraak staat gepland voor {}.",
        "pl": "Spotkanie jest zaplanowane na {}.",
        "pt": "A consulta está marcada para {}.",
        "ru": "Встреча назначена на {}.",
        "tr": "Randevu {} tarihine planlandı.",  # ruff: ignore[ambiguous-unicode-character-string] - Turkish dotless i
        "vi": "Cuộc hẹn được lên lịch vào {}.",
    },
    "currency": {
        "en": "The total comes to {}.",
        "ja": "合計金額は{}です。",
        "ar": "الإجمالي هو {}.",
        "de": "Die Summe beträgt {}.",
        "es": "El total es de {}.",
        "fr": "Le total s'élève à {}.",
        "hu": "A végösszeg {}.",
        "ko": "총 금액은 {}입니다.",
        "nl": "Het totaal komt neer op {}.",
        "pl": "Suma wynosi {}.",
        "pt": "O total é de {}.",
        "ru": "Итого: {}.",
        "tr": "Toplam tutar {}.",
        "vi": "Tổng cộng là {}.",
    },
    "id": {
        "en": "Your confirmation code is {}.",
        "ja": "確認コードは{}です。",
        "ar": "رمز التأكيد الخاص بك هو {}.",
        "de": "Ihr Bestätigungscode lautet {}.",
        "es": "Su código de confirmación es {}.",
        "fr": "Votre code de confirmation est {}.",
        "hu": "Az igazoló kódja: {}.",
        "ko": "확인 코드는 {}입니다.",
        "nl": "Uw bevestigingscode is {}.",
        "pl": "Twój kod potwierdzający to {}.",
        "pt": "O seu código de confirmação é {}.",
        "ru": "Ваш код подтверждения: {}.",
        "tr": "Onay kodunuz {}.",
        "vi": "Mã xác nhận của bạn là {}.",
    },
    "name": {
        "en": "Please ask for {} when you arrive.",
        "ja": "到着したら{}をお呼びください。",
        "ar": "يرجى السؤال عن {} عند وصولك.",
        "de": "Bitte fragen Sie bei Ihrer Ankunft nach {}.",
        "es": "Por favor pregunte por {} al llegar.",
        "fr": "Merci de demander {} à votre arrivée.",
        "hu": "Kérjük, érkezéskor keresse: {}.",
        "ko": "도착하시면 {}님을 찾아주세요.",
        "nl": "Vraag bij aankomst naar {}.",
        "pl": "Proszę zapytać o {} po przyjeździe.",
        "pt": "Por favor, pergunte por {} quando chegar.",
        "ru": "Пожалуйста, спросите {} по прибытии.",
        "tr": "Vardığınızda lütfen {} kişisini sorun.",  # ruff: ignore[ambiguous-unicode-character-string] - Turkish dotless i
        "vi": "Vui lòng hỏi {} khi bạn đến.",
    },
}

DATE_EPOCH = date(2026, 1, 1)
DATE_WINDOW_DAYS = 729
"""Two-year window from a fixed epoch, not ``date.today()`` — generation
must be reproducible for the same seed regardless of when it runs."""


def spoken_currency(major: int, minor: int, lang: str) -> str:
    """Return the spoken word form of a currency amount for ``lang``.

    Routes around three confirmed num2words inconsistencies rather than
    assuming a single calling convention: Vietnamese has no currency
    converter at all (falls back to a cardinal + the VND word "đồng"),
    Turkish's hardcodes Lira and rejects a ``currency`` keyword, and most
    others expect the value pre-multiplied into minor units (cents) even
    when given a plain integer.

    Args:
        major: Whole currency units.
        minor: Minor units (cents). Ignored for a locale in
            :data:`WHOLE_UNIT_CURRENCY` — JPY/KRW/VND have no minor unit in
            ordinary use, so this never raises on a nonzero value; it just
            has no effect, the same way a caller passing ``minor=0`` would.
        lang: Bare primary-subtag key into :data:`CURRENCY_CODE`.

    Returns:
        The spoken form, e.g. ``"four hundred and twenty dollars, zero cents"``.
    """
    if lang == "vi":
        return f"{num2words.num2words(major, lang='vi')} đồng"
    if lang == "tr":
        return str(num2words.num2words(major, lang="tr", to="currency"))
    if lang in WHOLE_UNIT_CURRENCY:  # ja, ko
        return str(num2words.num2words(major, lang=lang, to="currency", currency=CURRENCY_CODE[lang]))
    value = major + minor / 100
    return str(num2words.num2words(value, lang=lang, to="currency", currency=CURRENCY_CODE[lang]))


def spoken_date(day: date, lang: str) -> str:
    """Return the spoken word form of a date for ``lang``.

    Composes a CLDR month name (Babel) with a num2words ordinal day,
    uniformly across locales — see the module docstring's simplification
    note.
    """
    month_name = get_month_names("wide", locale=lang)[day.month]
    day_word = num2words.num2words(day.day, lang=lang, to="ordinal")
    return f"{month_name} {day_word}"


def spell_digits(digits: str, lang: str) -> str:
    """Return each digit of ``digits`` spoken individually, space-separated."""
    return " ".join(num2words.num2words(int(digit), lang=lang) for digit in digits)


def _build_number(rng: random.Random, lang: str) -> tuple[str, str]:
    value = rng.randint(100, 9999)
    spoken = num2words.num2words(value, lang=lang)
    template = TEMPLATES["number"][lang]
    return template.format(value), template.format(f"<number>{spoken}</number>")


def _build_date(rng: random.Random, lang: str) -> tuple[str, str]:
    day = DATE_EPOCH + timedelta(days=rng.randint(0, DATE_WINDOW_DAYS))
    written = format_date(day, format="long", locale=lang)
    spoken = spoken_date(day, lang)
    template = TEMPLATES["date"][lang]
    return template.format(written), template.format(f"<date>{spoken}</date>")


def _build_currency(rng: random.Random, lang: str) -> tuple[str, str]:
    major = rng.randint(1, 999)
    minor = 0 if lang in WHOLE_UNIT_CURRENCY else rng.randint(0, 99)
    written = format_currency(major + minor / 100, CURRENCY_CODE[lang], locale=lang)
    spoken = spoken_currency(major, minor, lang)
    template = TEMPLATES["currency"][lang]
    return template.format(written), template.format(f"<currency>{spoken}</currency>")


def _build_id(rng: random.Random, lang: str) -> tuple[str, str]:
    digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(4, 6)))
    spoken = spell_digits(digits, lang)
    template = TEMPLATES["id"][lang]
    return template.format(digits), template.format(f"<id>{spoken}</id>")


def _build_name(rng: random.Random, faker: Faker, lang: str) -> tuple[str, str]:
    del rng  # Faker owns its own PRNG state; kept for a uniform signature.
    person = faker.name()
    template = TEMPLATES["name"][lang]
    return template.format(person), template.format(f"<name>{person}</name>")


_BUILDERS: dict[str, Callable[[random.Random, str], tuple[str, str]]] = {
    "number": _build_number,
    "date": _build_date,
    "currency": _build_currency,
    "id": _build_id,
}


def generate_locale(subtag: str, *, count_per_class: int, seed: int) -> list[SuitePrompt]:
    """Generate every entity-class hard case for one locale.

    Args:
        subtag: Bare primary-subtag key into :data:`LOCALES`.
        count_per_class: Hard cases to generate per entity class.
        seed: Base seed; combined with ``(subtag, class)`` so every class's
            stream is independent and reproducible.

    Returns:
        ``count_per_class * len(ENTITY_CLASSES)`` prompts, grouped by class
        in :data:`audio_harness.entities.ENTITY_CLASSES` order.
    """
    language = LOCALES[subtag]
    prompts: list[SuitePrompt] = []
    for entity_class in ENTITY_CLASSES:
        # random.Random only accepts None/int/float/str/bytes/bytearray, so
        # the composite key is joined into a single reproducible string
        # rather than passed as a tuple.
        class_seed = f"{seed}:{subtag}:{entity_class}"
        rng = random.Random(class_seed)
        faker: Faker | None = None
        if entity_class == "name":
            faker = Faker(FAKER_LOCALE[subtag])
            faker.seed_instance(class_seed)

        for index in range(count_per_class):
            if faker is not None:
                text, annotated = _build_name(rng, faker, subtag)
            else:
                text, annotated = _BUILDERS[entity_class](rng, subtag)
            prompts.append(
                SuitePrompt(
                    prompt_id=f"entities-{entity_class}-{index:04d}",
                    text=text,
                    language=language,
                    category=CATEGORY,
                    license=LICENSE,
                    source=SOURCE,
                    annotated=annotated,
                    entity_class=entity_class,
                )
            )
    return prompts
