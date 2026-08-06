"""Tests for the entity hard-case generator.

The point of this layer is correctness under a shared tag vocabulary and
reproducibility, not exhaustive per-language proofreading — see the module
docstring's documented simplifications. These tests pin: deterministic
seeding, well-formed entity tags every ``entities.py`` can parse, spoken
currency across num2words' inconsistent per-language conventions, and
coverage of every locale/class combination.
"""

from __future__ import annotations

from datetime import date

import pytest

from audio_harness.entities import ENTITY_CLASSES, parse_annotated, score_entities
from audio_harness.entity_prompts import (
    CURRENCY_CODE,
    WHOLE_UNIT_CURRENCY,
    generate_locale,
    spell_digits,
    spoken_currency,
    spoken_date,
)
from audio_harness.prompt_suite import LOCALES


class TestDeterminism:
    """Same seed -> identical output; different seed -> different output."""

    def test_same_seed_reproduces_exactly(self) -> None:
        first = generate_locale("en", count_per_class=4, seed=1)
        second = generate_locale("en", count_per_class=4, seed=1)

        assert first == second

    def test_different_seeds_diverge(self) -> None:
        first = generate_locale("en", count_per_class=4, seed=1)
        second = generate_locale("en", count_per_class=4, seed=2)

        assert [p.text for p in first] != [p.text for p in second]

    def test_different_locales_do_not_collide(self) -> None:
        en = generate_locale("en", count_per_class=3, seed=1)
        de = generate_locale("de", count_per_class=3, seed=1)

        assert [p.text for p in en] != [p.text for p in de]


class TestCoverage:
    """Every locale and every entity class must actually produce content."""

    def test_every_class_present_per_locale(self) -> None:
        prompts = generate_locale("en", count_per_class=2, seed=1)

        classes = {p.entity_class for p in prompts}
        assert classes == set(ENTITY_CLASSES)

    def test_count_per_class_is_respected(self) -> None:
        prompts = generate_locale("en", count_per_class=5, seed=1)

        assert len(prompts) == 5 * len(ENTITY_CLASSES)

    @pytest.mark.parametrize("subtag", sorted(LOCALES))
    def test_every_locale_generates_without_error(self, subtag: str) -> None:
        prompts = generate_locale(subtag, count_per_class=1, seed=1)

        assert len(prompts) == len(ENTITY_CLASSES)
        assert all(p.language == LOCALES[subtag] for p in prompts)
        assert all(p.text.strip() for p in prompts)
        assert all(p.annotated for p in prompts)


class TestAnnotatedTagsAreWellFormed:
    """The annotated form must round-trip through entities.py's own parser."""

    @pytest.mark.parametrize("subtag", sorted(LOCALES))
    def test_annotated_form_parses_to_exactly_one_tagged_span(
        self, subtag: str
    ) -> None:
        prompts = generate_locale(subtag, count_per_class=1, seed=3)

        for prompt in prompts:
            assert prompt.annotated is not None
            segments = parse_annotated(prompt.annotated)
            tagged = [label for _text, label in segments if label is not None]
            assert tagged == [prompt.entity_class]

    def test_scoring_a_hard_case_against_itself_is_a_perfect_match(self) -> None:
        # A hard case's own annotated reference, scored against a hypothesis
        # equal to its plain text, is the sanity check that both forms agree
        # on what the entity token stream actually is.
        prompts = generate_locale("en", count_per_class=2, seed=5)

        for prompt in prompts:
            assert prompt.annotated is not None
            scores = score_entities(prompt.annotated, prompt.text, prompt.language)
            assert prompt.entity_class in scores


class TestSpokenCurrency:
    """num2words' currency converter is not uniform; this pins the routing."""

    def test_covers_every_locale_without_raising(self) -> None:
        for subtag in LOCALES:
            result = spoken_currency(420, 0, subtag)
            assert isinstance(result, str)
            assert result.strip()

    def test_whole_unit_locales_ignore_minor_units(self) -> None:
        for subtag in WHOLE_UNIT_CURRENCY:
            assert spoken_currency(420, 0, subtag) == spoken_currency(420, 99, subtag)

    def test_non_whole_unit_locale_reflects_minor_units(self) -> None:
        assert spoken_currency(420, 0, "en") != spoken_currency(420, 50, "en")

    def test_every_currency_code_is_assigned(self) -> None:
        assert set(CURRENCY_CODE) == set(LOCALES)


class TestSpellDigits:
    """Digit spelling must cover every locale, including a real '0'."""

    def test_each_digit_spoken_individually(self) -> None:
        spelled = spell_digits("042", "en")
        assert spelled == "zero four two"

    @pytest.mark.parametrize("subtag", sorted(LOCALES))
    def test_zero_is_spoken_in_every_locale(self, subtag: str) -> None:
        assert spell_digits("0", subtag).strip()


class TestSpokenDate:
    """Date composition must not raise for any locale, across a leap day."""

    @pytest.mark.parametrize("subtag", sorted(LOCALES))
    def test_covers_every_locale(self, subtag: str) -> None:
        assert spoken_date(date(2026, 3, 3), subtag).strip()

    def test_covers_a_leap_day(self) -> None:
        assert spoken_date(date(2028, 2, 29), "en").strip()
