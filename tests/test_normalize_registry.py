"""Tests for the per-language normalization registry.

Each language's rules are data tables driven by shared engines, so the tests
are vector tables too: raw text in, canonical form out, with the cases chosen
to pin the formatting differences that would otherwise score as recognition
errors — hangul numerals against digits, agglutinated German numbers,
vigesimal French, kana script choice.
"""

from __future__ import annotations

from audio_harness.metrics import score_pair
from audio_harness.normalize import (
    comparison_fold_for,
    fold_kana,
    normalize_french,
    normalize_german,
    normalize_japanese,
    normalize_korean,
    normalize_spanish,
    normalizer_for,
    uses_character_metric,
)


class TestKoreanNormalization:
    """Hangul numerals and digits must meet in one canonical form."""

    def test_numeral_folding_cases(self) -> None:
        tests = {
            "hangul month folds to digit month (the ko blocker)": {
                "raw": "삼월",
                "expected": "3월",
            },
            "compound numeral with counter": {
                "raw": "이십삼명",
                "expected": "23명",
            },
            "spaced numeral with counter": {
                "raw": "오백 원",
                "expected": "500원",
            },
            "positional thousands": {
                "raw": "천이백삼십사원",
                "expected": "1234원",
            },
            "large scale word alone": {
                "raw": "만원",
                "expected": "10000원",
            },
            "day counter backtracks out of the numeral run": {
                "raw": "삼십일 일",
                "expected": "31일",
            },
            "irregular june reading": {
                "raw": "유월",
                "expected": "6월",
            },
            "irregular october reading": {
                "raw": "시월",
                "expected": "10월",
            },
        }
        for name, case in tests.items():
            assert normalize_korean(case["raw"]) == case["expected"], name

    def test_native_numerals_fold_with_their_counters(self) -> None:
        tests = {
            "three o'clock": {"raw": "세 시", "expected": "3시"},
            "once": {"raw": "한 번", "expected": "1번"},
            "twelve hours": {"raw": "열두 시간", "expected": "12시간"},
            "attached hour": {"raw": "두시", "expected": "2시"},
        }
        for name, case in tests.items():
            assert normalize_korean(case["raw"]) == case["expected"], name

    def test_digit_and_counter_spacing_is_free(self) -> None:
        assert normalize_korean("30 분") == "30분"
        assert normalize_korean("2026 년") == "2026년"

    def test_ordinary_words_in_numeral_syllables_survive(self) -> None:
        """Most numeral syllables are also common morphemes; folding them
        outside a licensed reading would corrupt the transcript."""
        tests = {
            "stopword before a counter": {
                "raw": "만일 시간이 되면",
                "expected": "만일 시간이 되면",
            },
            "angel is not 1004": {
                "raw": "천사 명",
                "expected": "천사 명",
            },
            "between is not 42": {
                "raw": "사이",
                "expected": "사이",
            },
            "no counter means no fold": {
                "raw": "삼삼오오",
                "expected": "삼삼오오",
            },
        }
        for name, case in tests.items():
            assert normalize_korean(case["raw"]) == case["expected"], name

    def test_hangul_and_digit_forms_score_identically(self) -> None:
        counts = score_pair("3월 15일에 만나요", "삼월 십오일에 만나요", "ko-KR")
        assert counts.errors == 0, (
            "the reference writes digits and the recognizer writes hangul "
            "numerals for the same audio; that is formatting, not an error"
        )

    def test_korean_stays_a_word_metric(self) -> None:
        assert not uses_character_metric("ko-KR")


class TestGermanNormalization:
    """Agglutinated numbers must decompose before folding."""

    def test_number_folding_cases(self) -> None:
        tests = {
            "reversed unit and tens": {"raw": "einundzwanzig", "expected": "21"},
            "agglutinated hundreds": {
                "raw": "vierhundertzwanzig",
                "expected": "420",
            },
            "thousands": {"raw": "zweitausendfünf", "expected": "2005"},
            "sharp s tens": {"raw": "dreißig", "expected": "30"},
            "spelled with und over hundreds": {
                "raw": "dreihundertfünfundvierzig",
                "expected": "345",
            },
        }
        for name, case in tests.items():
            assert normalize_german(case["raw"]) == case["expected"], name

    def test_currency_symbol_matches_spoken_form(self) -> None:
        assert normalize_german("420 €") == normalize_german("vierhundertzwanzig Euro")

    def test_hyphenated_spelling_variants_join(self) -> None:
        """Golden-diff clips 16144/16214: the vendor wrote orthographically
        correct "E-Mail"/"ge-emailt" against unhyphenated references and was
        punished with false errors."""
        tests = {
            "single-letter fragment joins": {
                "raw": "E-Mail",
                "same_as": "email",
            },
            "bound prefix joins": {
                "raw": "ge-emailt",
                "same_as": "geemailt",
            },
            "prefix and single letter together": {
                "raw": "ge-e-mailt",
                "same_as": "geemailt",
            },
        }
        for name, case in tests.items():
            assert normalize_german(case["raw"]) == normalize_german(case["same_as"]), (
                name
            )

    def test_hyphenated_compounds_still_split_to_match_spaced_refs(self) -> None:
        """Golden-diff clips 11131/1871: references space these compounds, so
        joining them (the old behavior) scored false errors the other way."""
        tests = {
            "to-do list": {"raw": "To-do-Liste", "expected": "to do liste"},
            "top ten": {
                "raw": "Top-Ten-Favoriten",
                "expected": "top ten favoriten",
            },
            "place name": {
                "raw": "Potsdam-Brandenburg",
                "expected": "potsdam brandenburg",
            },
        }
        for name, case in tests.items():
            assert normalize_german(case["raw"]) == case["expected"], name

    def test_words_that_start_with_number_morphemes_survive(self) -> None:
        tests = {
            "achtung is not acht": {"raw": "Achtung", "expected": "achtung"},
            "elfmeter is not elf": {"raw": "Elfmeter", "expected": "elfmeter"},
            "und outside a number survives": {
                "raw": "Katzen und Hunde",
                "expected": "katzen und hunde",
            },
        }
        for name, case in tests.items():
            assert normalize_german(case["raw"]) == case["expected"], name


class TestFrenchNormalization:
    """The vigesimal 70/80/90 forms must fold as single numbers."""

    def test_number_folding_cases(self) -> None:
        tests = {
            "ninety nine": {"raw": "quatre-vingt-dix-neuf", "expected": "99"},
            "eighty": {"raw": "quatre-vingts", "expected": "80"},
            "seventy five": {"raw": "soixante-quinze", "expected": "75"},
            "et connector": {"raw": "vingt et un", "expected": "21"},
            "bare hundred": {"raw": "cent vingt-trois", "expected": "123"},
            "bare thousand": {"raw": "mille neuf cents", "expected": "1900"},
            "lexicalized seventeen": {"raw": "dix-sept", "expected": "17"},
            "lexicalized eighteen": {"raw": "dix-huit heures", "expected": "18 heures"},
            "ninety seven keeps its join order": {
                "raw": "quatre-vingt-dix-sept",
                "expected": "97",
            },
        }
        for name, case in tests.items():
            assert normalize_french(case["raw"]) == case["expected"], name

    def test_currency_symbol_matches_spoken_form(self) -> None:
        assert normalize_french("20 €") == normalize_french("vingt euros")

    def test_et_outside_a_number_survives(self) -> None:
        assert normalize_french("chats et chiens") == "chats et chiens"

    def test_hyphen_folding_matches_both_vendor_conventions(self) -> None:
        """Golden-diff fr-FR: "e-mail"/"email" are one word (join), while
        inversion clitics hyphenate what references space (split)."""
        tests = {
            "email variant joins": {
                "raw": "envoie un e-mail",
                "same_as": "envoie un email",
            },
            "elided email joins": {
                "raw": "enregistrer l'e-mail",
                "same_as": "enregistrer l'email",
            },
            "inversion clitic splits": {
                "raw": "Pourrais-tu venir",
                "same_as": "pourrais tu venir",
            },
        }
        for name, case in tests.items():
            assert normalize_french(case["raw"]) == normalize_french(case["same_as"]), (
                name
            )


class TestSpanishNormalization:
    """Lexicalized twenties and hundreds fold from the tables."""

    def test_number_folding_cases(self) -> None:
        tests = {
            "four hundred twenty": {
                "raw": "cuatrocientos veinte",
                "expected": "420",
            },
            "y connector": {"raw": "treinta y cinco", "expected": "35"},
            "lexicalized twenties": {"raw": "veintitrés", "expected": "23"},
            "bare thousand": {"raw": "mil novecientos", "expected": "1900"},
            "ciento composes": {"raw": "ciento veintitrés", "expected": "123"},
            "feminine hundreds": {
                "raw": "quinientas personas",
                "expected": "500 personas",
            },
        }
        for name, case in tests.items():
            assert normalize_spanish(case["raw"]) == case["expected"], name

    def test_currency_symbol_matches_spoken_form(self) -> None:
        assert normalize_spanish("$25") == normalize_spanish("veinticinco dólares")

    def test_y_outside_a_number_survives(self) -> None:
        assert normalize_spanish("blanco y negro") == "blanco y negro"

    def test_email_spelling_variants_join(self) -> None:
        assert normalize_spanish("un e-mail") == normalize_spanish("un email")


class TestJapaneseKanaFold:
    """Script choice is vendor policy; CER must not charge for it."""

    def test_katakana_folds_onto_hiragana(self) -> None:
        assert fold_kana("アイウ") == "あいう"
        assert fold_kana("コーヒー") == "こーひー", (
            "the prolonged sound mark is shared by both scripts"
        )

    def test_normalizer_output_keeps_the_script_distinction(self) -> None:
        """The fold is a scoring equivalence, not a canonical form — a human
        reading the normalized transcript still expects katakana loanwords."""
        assert normalize_japanese("ｱｲｳ") == "アイウ"

    def test_script_choice_scores_as_identical(self) -> None:
        counts = score_pair("コーヒーを飲む", "こーひーを飲む", "ja-JP")
        assert counts.errors == 0

    def test_real_differences_still_count(self) -> None:
        counts = score_pair("コーヒー", "こーちー", "ja-JP")
        assert counts.errors == 1


class TestRegistry:
    """One registry keyed by BCP-47 primary subtag resolves every language."""

    def test_new_languages_resolve_from_the_registry(self) -> None:
        tests = {
            "ko-KR": normalize_korean,
            "de-DE": normalize_german,
            "fr-FR": normalize_french,
            "es-ES": normalize_spanish,
        }
        for tag, expected in tests.items():
            assert normalizer_for(tag) is expected, tag

    def test_comparison_fold_defaults_to_identity(self) -> None:
        assert comparison_fold_for("ja-JP") is fold_kana
        assert comparison_fold_for("en-US")("Text") == "Text"
        assert comparison_fold_for("ko-KR")("삼월") == "삼월"
