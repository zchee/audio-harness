"""Tests for automatic entity annotation of corpus references.

The tagger feeds the entity scorer, so its failure modes matter in one
direction only: a missed entity is a smaller sample, a false tag scores
noise as fact. The cases below therefore pin both the detections and the
guards — articles that fold to 1, Korean words spelled in numeral
syllables, ambiguous month names, German noun capitalization.
"""

from __future__ import annotations

from audio_harness.autotag import (
    MASSIVE_SLOT_CLASSES,
    annotate_massive,
    annotate_record,
    autotag_reference,
)
from audio_harness.entities import score_entities, strip_tags
from audio_harness.metrics import summarize
from audio_harness.types import Mode, SttResult


class TestRuleTagging:
    """Deterministic per-language detections and their guards."""

    def test_english_cases(self) -> None:
        tests = {
            "cased name run": {
                "raw": "This is Samantha Lee with a night to",
                "expected": "This is <name>Samantha Lee</name> with a night to",
            },
            "number word via the scoring fold": {
                "raw": "How much juice is in one lime?",
                "expected": "How much juice is in <number>one</number> lime?",
            },
            "currency words after a number": {
                "raw": "that costs twenty five dollars",
                "expected": "that costs <currency>twenty five dollars</currency>",
            },
            "currency symbol": {
                "raw": "I paid $25 there",
                "expected": "I paid <currency>$25</currency> there",
            },
            "month with day": {
                "raw": "see you March 3rd",
                "expected": "see you <date>March 3rd</date>",
            },
            "ambiguous month alone is not a date": {
                "raw": "you may go now",
                "expected": "you may go now",
            },
            "year": {
                "raw": "back in 2026 already",
                "expected": "back in <date>2026</date> already",
            },
            "mixed alphanumeric id": {
                "raw": "my code is AB1234 ok",
                "expected": "my code is <id>AB1234</id> ok",
            },
            "long digit string is an id": {
                "raw": "call 5551234567 now",
                "expected": "call <id>5551234567</id> now",
            },
            "sentence-initial capital is not a name": {
                "raw": "Remind me later",
                "expected": "Remind me later",
            },
        }
        for name, case in tests.items():
            assert autotag_reference(case["raw"], "en-US") == case["expected"], name

    def test_german_cases(self) -> None:
        tests = {
            "number word": {
                "raw": "buche mir ein ticket vor zehn uhr",
                "expected": "buche mir ein ticket vor <number>zehn</number> uhr",
            },
            "agglutinated amount": {
                "raw": "das kostet vierhundertzwanzig euro",
                "expected": "das kostet <currency>vierhundertzwanzig euro</currency>",
            },
            "capitalized nouns are never names": {
                "raw": "Zeige mir das Wetter in Hamburg",
                "expected": "Zeige mir das Wetter in Hamburg",
            },
        }
        for name, case in tests.items():
            assert autotag_reference(case["raw"], "de-DE") == case["expected"], name

    def test_french_cases(self) -> None:
        tests = {
            "article un is not a number": {
                "raw": "régler un minuteur sur cinq minutes",
                "expected": "régler un minuteur sur <number>cinq</number> minutes",
            },
            "vigesimal compound": {
                "raw": "il y en a quatre-vingt-dix-neuf",
                "expected": "il y en a <number>quatre-vingt-dix-neuf</number>",
            },
        }
        for name, case in tests.items():
            assert autotag_reference(case["raw"], "fr-FR") == case["expected"], name

    def test_spanish_cases(self) -> None:
        tests = {
            "article una is not a number": {
                "raw": "me gustaría una alarma a las siete",
                "expected": "me gustaría una alarma a las <number>siete</number>",
            },
            "amount with currency word": {
                "raw": "convierte veinticinco dólares ahora",
                "expected": "convierte <currency>veinticinco dólares</currency> ahora",
            },
        }
        for name, case in tests.items():
            assert autotag_reference(case["raw"], "es-ES") == case["expected"], name

    def test_korean_cases(self) -> None:
        tests = {
            "month and day with particle": {
                "raw": "삼월 십오일에 만나요",
                "expected": "<date>삼월</date> <date>십오일</date>에 만나요",
            },
            "currency and native hour": {
                "raw": "오백원 내고 세 시에 만나",
                "expected": ("<currency>오백원</currency> 내고 <number>세 시</number>에 만나"),
            },
            "stopwords and bare syllables stay untagged": {
                "raw": "만일 시간이 되면 사이에",
                "expected": "만일 시간이 되면 사이에",
            },
            "counter inside a longer word is not a match": {
                "raw": "지금 시작하자",
                "expected": "지금 시작하자",
            },
            "angel is not a thousand and four people": {
                "raw": "천사 명이 왔다",
                "expected": "천사 명이 왔다",
            },
        }
        for name, case in tests.items():
            assert autotag_reference(case["raw"], "ko-KR") == case["expected"], name

    def test_tags_never_alter_the_text(self) -> None:
        references = {
            "en-US": "Pay Anna Lee $25 on March 3rd, code AB12, call 5551234567",
            "de-DE": "das kostet einundzwanzig euro am 12.03.2026",
            "fr-FR": "quatre-vingt-dix-neuf euros le 3/4",
            "es-ES": "quinientas personas y veinticinco dólares",
            "ko-KR": "삼월 십오일에 오백원 내고 세 시에",
        }
        for language, reference in references.items():
            assert strip_tags(autotag_reference(reference, language)) == reference, language


class TestMassiveGold:
    """Gold slots outrank rules; unmapped slots dissolve into plain text."""

    def test_mapped_slots_become_tags(self) -> None:
        annotated = annotate_massive(
            "réveille-moi à [time : neuf heures] et appelle [person : olivier]",
            "fr-FR",
        )
        assert "<date>neuf heures</date>" in annotated
        assert "<name>olivier</name>" in annotated

    def test_unmapped_slots_dissolve(self) -> None:
        annotated = annotate_massive("joue de la musique [music_genre : classique]", "fr-FR")
        assert annotated == "joue de la musique classique"

    def test_rules_still_tag_the_untagged_remainder(self) -> None:
        annotated = annotate_massive("commande quatre pizzas pour [person : anna]", "fr-FR")
        assert "<number>quatre</number>" in annotated
        assert "<name>anna</name>" in annotated

    def test_gold_beats_a_conflicting_rule_span(self) -> None:
        """ "neuf" inside a gold time span must not be re-tagged a number."""
        annotated = annotate_massive("à [time : neuf heures] pile", "fr-FR")
        assert annotated == "à <date>neuf heures</date> pile"

    def test_the_slot_map_only_targets_harness_classes(self) -> None:
        assert set(MASSIVE_SLOT_CLASSES.values()) <= {"date", "name", "currency"}


class TestAnnotateRecord:
    """The saved-results path: gold join, drift fallback, idempotence."""

    def _record(self, reference: str, language: str = "fr-FR") -> dict[str, object]:
        return {
            "provider": "p1",
            "clip_id": "42",
            "language": language,
            "reference": reference,
        }

    def test_gold_annotation_is_used_when_text_matches(self) -> None:
        record = self._record("appelle olivier maintenant")
        gold = {("fr-FR", "42"): "appelle [person : olivier] maintenant"}

        assert annotate_record(record, gold)
        assert record["reference_annotated"] == ("appelle <name>olivier</name> maintenant")

    def test_drifted_gold_falls_back_to_rules(self) -> None:
        """A gold annotation for different text would mis-place every span."""
        record = self._record("règle un minuteur sur cinq minutes")
        gold = {("fr-FR", "42"): "appelle [person : olivier] maintenant"}

        assert annotate_record(record, gold)
        assert record["reference_annotated"] == ("règle un minuteur sur <number>cinq</number> minutes")

    def test_untaggable_reference_sets_nothing(self) -> None:
        record = self._record("va te coucher")
        assert not annotate_record(record, {})
        assert "reference_annotated" not in record

    def test_existing_annotation_is_kept_without_force(self) -> None:
        record = self._record("cinq minutes")
        record["reference_annotated"] = "<number>cinq</number> minutes"
        annotate_record(record, {})
        assert record["reference_annotated"] == "<number>cinq</number> minutes"


class TestScoringIntegration:
    """Auto-tags must flow through the entity scorer end to end."""

    def test_autotagged_reference_scores_entities(self) -> None:
        reference = "call me on March 3rd about the twenty five dollars"
        annotated = autotag_reference(reference, "en-US")

        scores = score_entities(annotated, "call me on march 3 about the $25", "en-US")

        assert scores["date"].exact_match_rate == 1.0
        assert scores["currency"].exact_match_rate == 1.0

    def test_summarize_populates_entity_columns_from_auto_tags(self) -> None:
        reference = "the code is AB1234"
        result = SttResult(provider="p1", clip_id="c1", mode=Mode.STREAM, text="the code is AB1234")
        result.audio_s = 1.0
        result.raw["reference"] = reference
        result.raw["language"] = "en-US"
        result.raw["reference_annotated"] = autotag_reference(reference, "en-US")

        summaries = summarize([result], "en-US")

        assert summaries[0].entities["id"].exact_match_rate == 1.0
