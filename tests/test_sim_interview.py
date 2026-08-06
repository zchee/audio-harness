"""Tests for the simulated-interview E2E lane (plan P4 step 18, lane E3)."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re

import numpy as np
import orjson
import pytest

from audio_harness.config import BenchmarkConfig, ConfigError
from audio_harness.sim.interview import (
    DEGRADATION,
    KOKORO_SAMPLE_RATE,
    SimConfig,
    SlotValue,
    Turn,
    VendorScore,
    average_ranks,
    canonical_value,
    clips_from_turns,
    composite_ranking,
    conduct_interview,
    ensure_within_cap,
    estimate_spend,
    evaluate_gate,
    exact_permutation_p,
    extract_slot,
    load_canonical,
    load_scenarios,
    run_sim,
    sample_personas,
    spearman_rho,
    write_sim_outputs,
)
from audio_harness.types import Mode, SttResult


SCENARIOS_EN = Path("data/sim/scenarios-en.yaml")


# --------------------------------------------------------------------------
# Scenario parsing
# --------------------------------------------------------------------------


class TestScenarioParsing:
    """The committed spec parses; malformed specs fail loudly."""

    def test_committed_spec_loads(self) -> None:
        scenarios = load_scenarios(SCENARIOS_EN)

        assert [s.scenario_id for s in scenarios] == [
            "job-application",
            "insurance-claim",
            "clinic-intake",
        ]
        assert all(s.language == "en-US" for s in scenarios)
        assert all(len(s.slots) == 8 for s in scenarios)
        assert all(slot.entity for s in scenarios for slot in s.slots)

    def test_malformed_specs_are_rejected(self, tmp_path: Path) -> None:
        tests = {
            "error: unknown kind": ("scenarios:\n  - id: x\n    slots:\n      - {name: a, kind: nope, question: q}\n"),
            "error: missing question": ("scenarios:\n  - id: x\n    slots:\n      - {name: a, kind: phone}\n"),
            "error: no scenarios": "scenarios: []\n",
            "error: no slots": "scenarios:\n  - id: x\n    slots: []\n",
        }
        for name, text in tests.items():
            spec = tmp_path / "spec.yaml"
            spec.write_text(text, encoding="utf-8")
            with pytest.raises(ConfigError):
                load_scenarios(spec)
            assert name  # test-case label used


class TestPersonaSampling:
    """Ground truth is seeded, coherent and self-verifying."""

    def test_same_seed_same_personas(self) -> None:
        scenario = load_scenarios(SCENARIOS_EN)[0]
        first = sample_personas(scenario, 3, np.random.default_rng(11))
        second = sample_personas(scenario, 3, np.random.default_rng(11))

        assert [p.name for p in first] == [p.name for p in second]
        assert [p.values["phone"].canonical for p in first] == [p.values["phone"].canonical for p in second]

    def test_person_slot_reuses_the_identity(self) -> None:
        scenario = load_scenarios(SCENARIOS_EN)[0]
        persona = sample_personas(scenario, 1, np.random.default_rng(3))[0]

        assert persona.values["full_name"].written == persona.name

    def test_spoken_forms_verify_through_the_extractor(self) -> None:
        """Every sampled spoken form must pass its own slot's extractor —
        otherwise no persona answer could ever verify."""
        for scenario in load_scenarios(SCENARIOS_EN):
            personas = sample_personas(scenario, 4, np.random.default_rng(5))
            for persona in personas:
                for slot in scenario.slots:
                    truth = persona.values[slot.name]
                    extracted = extract_slot(
                        f"well um it is {truth.spoken} I think",
                        truth,
                        slot.entity,
                        scenario.language,
                    )
                    assert extracted == truth.canonical, (
                        scenario.scenario_id,
                        slot.name,
                        truth.spoken,
                    )


# --------------------------------------------------------------------------
# Deterministic extraction
# --------------------------------------------------------------------------


class TestExtraction:
    """Schema-aware candidate extraction over the shared normalization."""

    def test_extraction_fixtures(self) -> None:
        tests = {
            "success: spoken digits": (
                "Um, it is seven nine zero, five five five, zero one six two.",
                SlotValue("7905550162", "790-555-0162", ""),
                "number",
                True,
            ),
            "success: formatted digits": (
                "Reach me at 790-555-0162 anytime.",
                SlotValue("7905550162", "790-555-0162", ""),
                "number",
                True,
            ),
            "success: grouped digits": (
                "My number is 790 555 0162.",
                SlotValue("7905550162", "", ""),
                "number",
                True,
            ),
            "error: one digit off": (
                "It was 790-555-0163.",
                SlotValue("7905550162", "", ""),
                "number",
                False,
            ),
            "error: no token-internal match": (
                "The answer is 125.",
                SlotValue("12", "", ""),
                "number",
                False,
            ),
            "success: spoken amount": (
                "I want ninety five thousand dollars a year.",
                SlotValue("95000 dollars", "", ""),
                "currency",
                True,
            ),
            "success: symbol amount": (
                "Around $95,000 I think.",
                SlotValue("95000 dollars", "", ""),
                "currency",
                True,
            ),
            "error: wrong amount": (
                "Around $96,000 I think.",
                SlotValue("95000 dollars", "", ""),
                "currency",
                False,
            ),
            "success: ordinal date": (
                "I could start March fourteenth.",
                SlotValue("march 14", "", ""),
                "date",
                True,
            ),
            "success: day-first date": (
                "the fourteenth of March works",
                SlotValue("march 14", "", ""),
                "date",
                True,
            ),
            "error: wrong day": (
                "March 15 works.",
                SlotValue("march 14", "", ""),
                "date",
                False,
            ),
            "success: spelled code": (
                "the code is W X one zero four seven M.",
                SlotValue("wx1047m", "", ""),
                "id",
                True,
            ),
            "success: compact code": (
                "sure, WX1047M.",
                SlotValue("wx1047m", "", ""),
                "id",
                True,
            ),
            "success: split code": (
                "sure, WX 1047 M.",
                SlotValue("wx1047m", "", ""),
                "id",
                True,
            ),
            "error: wrong code": (
                "sure, WX1048M.",
                SlotValue("wx1047m", "", ""),
                "id",
                False,
            ),
            "success: name present": (
                "My name is Teresa Mercado.",
                SlotValue("teresa mercado", "", ""),
                "name",
                True,
            ),
            "error: name misheard": (
                "Teresa Mercano, that is me.",
                SlotValue("teresa mercado", "", ""),
                "name",
                False,
            ),
        }
        for name, (text, truth, entity, expect) in tests.items():
            extracted = extract_slot(text, truth, entity, "en-US")
            assert (extracted is not None) == expect, name
            if expect:
                assert extracted == truth.canonical, name

    def test_canonical_value_matches_extraction_fold(self) -> None:
        tests = {
            "number strips punctuation": ("number", "790-555-0162", "7905550162"),
            "currency folds symbol": ("currency", "$95,000", "95000 dollars"),
            "date folds ordinal suffix": ("date", "March 14", "march 14"),
            "id squashes case": ("id", "WX1047M", "wx1047m"),
            "name lowercases": ("name", "Teresa Mercado", "teresa mercado"),
        }
        for name, (entity, written, expected) in tests.items():
            assert canonical_value(entity, written, "en-US") == expected, name


# --------------------------------------------------------------------------
# Ranking mathematics
# --------------------------------------------------------------------------


class TestRankingMath:
    """Hand-computed fixtures for ranks, rho and the exact permutation p."""

    def test_average_ranks_with_ties(self) -> None:
        ranks = average_ranks({"a": 0.9, "b": 0.8, "c": 0.8, "d": 0.1}, higher_is_better=True)

        assert ranks == {"a": 1.0, "b": 2.5, "c": 2.5, "d": 4.0}

    def test_missing_measurements_share_the_worst_ranks(self) -> None:
        ranks = average_ranks({"a": 1.0, "b": None, "c": 2.0, "d": None})

        # b and d occupy positions 3 and 4, tied: (3 + 4) / 2.
        assert ranks == {"a": 1.0, "c": 2.0, "b": 3.5, "d": 3.5}

    def test_spearman_hand_computed(self) -> None:
        a = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}

        assert spearman_rho(a, a) == pytest.approx(1.0)
        assert spearman_rho(a, {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}) == pytest.approx(-1.0)
        # One adjacent swap on n=4: rho = 1 - 6*2/(4*15) = 0.8.
        assert spearman_rho(a, {"a": 2.0, "b": 1.0, "c": 3.0, "d": 4.0}) == pytest.approx(0.8)

    def test_spearman_degenerate_is_nan(self) -> None:
        constant = {"a": 1.0, "b": 1.0, "c": 1.0}

        assert math.isnan(spearman_rho(constant, {"a": 1.0, "b": 2.0, "c": 3.0}))

    def test_exact_permutation_p_hand_computed(self) -> None:
        a = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}

        # Perfect agreement: only the identity of 24 permutations reaches 1.
        assert exact_permutation_p(a, a) == pytest.approx(1 / 24)
        # Perfect disagreement: every permutation is at least -1.
        assert exact_permutation_p(a, {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}) == pytest.approx(1.0)

    def test_gate_verdict_thresholds(self) -> None:
        composite = composite_ranking([], ["a", "b", "c"])
        composite.mean_rank = {"a": 1.0, "b": 2.0, "c": 3.0}

        def score(provider: str, correct: int) -> VendorScore:
            return VendorScore(provider=provider, scorable=10, correct=correct)

        passing = evaluate_gate(
            {"a": score("a", 9), "b": score("b", 5), "c": score("c", 1)},
            composite,
        )
        failing = evaluate_gate(
            {"a": score("a", 1), "b": score("b", 5), "c": score("c", 9)},
            composite,
        )

        assert passing.rho == pytest.approx(1.0)
        assert passing.passed
        assert failing.rho == pytest.approx(-1.0)
        assert not failing.passed
        assert failing.divergence[0]["delta"] != 0


# --------------------------------------------------------------------------
# Composite from canonical results
# --------------------------------------------------------------------------


def _canonical_result(
    provider: str,
    clip_id: str,
    text: str,
    *,
    reference: str,
    annotated: str | None = None,
    finalize_s: float = 0.3,
) -> SttResult:
    result = SttResult(
        provider=provider,
        clip_id=clip_id,
        mode=Mode.STREAM,
        text=text,
        finalize_s=finalize_s,
    )
    result.raw["reference"] = reference
    result.raw["language"] = "en-US"
    if annotated:
        result.raw["reference_annotated"] = annotated
    return result


class TestComposite:
    """The pre-registered composite over synthetic canonical results."""

    def test_annotations_propagate_across_lanes(self) -> None:
        reference = "my number is 42"
        annotated = "my number is <number>42</number>"
        results = [
            _canonical_result(
                "good",
                "c1",
                "my number is 42",
                reference=reference,
                annotated=annotated,
            ),
            # The superseding lane lost its annotation; it must be scored
            # over the same spans as the annotated lane.
            _canonical_result("bad", "c1", "my number is 43", reference=reference),
        ]

        composite = composite_ranking(results, ["good", "bad"])

        entity_cells = [c for c in composite.cells if c.metric == "entity-wer"]
        assert len(entity_cells) == 1
        assert entity_cells[0].scores["good"] == pytest.approx(0.0)
        assert entity_cells[0].scores["bad"] == pytest.approx(1.0)

    def test_mean_rank_over_both_metrics(self) -> None:
        results = [
            _canonical_result(
                "fast-sloppy",
                "c1",
                "my number is 43",
                reference="my number is 42",
                annotated="my number is <number>42</number>",
                finalize_s=0.1,
            ),
            _canonical_result(
                "slow-exact",
                "c1",
                "my number is 42",
                reference="my number is 42",
                annotated="my number is <number>42</number>",
                finalize_s=0.9,
            ),
        ]

        composite = composite_ranking(results, ["fast-sloppy", "slow-exact"])

        # Each vendor wins one cell: mean ranks tie at 1.5.
        assert composite.mean_rank["fast-sloppy"] == pytest.approx(1.5)
        assert composite.mean_rank["slow-exact"] == pytest.approx(1.5)

    def test_load_canonical_supersedes_lanes(self, tmp_path: Path) -> None:
        def write(path: Path, provider: str, text: str) -> Path:
            record = {
                "provider": provider,
                "clip_id": "c1",
                "mode": "stream",
                "text": text,
                "reference": "ref",
                "language": "en-US",
                "audio_s": 1.0,
                "partials": [],
            }
            path.write_bytes(orjson.dumps(record) + b"\n")
            return path

        early = write(tmp_path / "early.jsonl", "v1", "old")
        late = write(tmp_path / "late.jsonl", "v1", "new")

        merged = load_canonical([early, late])

        assert len(merged) == 1
        assert merged[0].text == "new"


# --------------------------------------------------------------------------
# Pipeline dry-run with all-local stubs
# --------------------------------------------------------------------------

_FACT_RE = re.compile(r"clearly state .*?: (?P<spoken>.+?)\nAnswer", re.DOTALL)
_RETRY_RE = re.compile(r"word for word: (?P<spoken>.+)$", re.DOTALL)


async def _stub_llm(system: str, prompt: str, temperature: float) -> str:  # ruff: ignore[unused-async] -- awaited through the LLM callable contract
    """Local dialogue stub: interviewer echoes the script, persona answers
    with the required spoken value woven into a disfluent sentence."""
    if "interviewer" in system:
        match = re.search(r"Scripted question to ask next: (.+)$", prompt, re.DOTALL)
        assert match is not None
        return match.group(1).strip()
    match = _RETRY_RE.search(prompt) or _FACT_RE.search(prompt)
    assert match is not None, prompt
    return f"Um, sure — it is {match.group('spoken').strip()}, I believe."


def _stub_tts(text: str, voice: str) -> np.ndarray:
    """One second of tone per answer; content is irrelevant to the stubs."""
    t = np.linspace(0.0, 1.0, KOKORO_SAMPLE_RATE, endpoint=False)
    return (0.3 * np.sin(2 * math.pi * 220.0 * t)).astype(np.float32)


def _fake_run_stt(perfect: str, corrupted: str):
    """Fake executor: one vendor echoes the reference, one loses digits."""

    async def run(bench, clips, progress=None):  # ruff: ignore[unused-async] -- stands in for runner.run_stt, which callers await
        results = []
        for entry in bench.stt:
            for clip in clips:
                text = clip.reference or ""
                if entry.name == corrupted:
                    text = re.sub(
                        r"\b(zero|one|two|three|four|five|six|seven|eight|nine)\b",
                        "hmm",
                        text,
                    )
                result = SttResult(
                    provider=entry.name,
                    clip_id=clip.clip_id,
                    mode=Mode.STREAM,
                    text=text,
                    audio_s=clip.duration_s,
                    finalize_s=0.2 if entry.name == perfect else 0.5,
                )
                result.raw["reference"] = clip.reference or ""
                result.raw["language"] = clip.language
                results.append(result)
        return results

    return run


def _mini_spec(tmp_path: Path) -> Path:
    spec = tmp_path / "scenarios.yaml"
    spec.write_text(
        """
scenarios:
  - id: mini
    language: en-US
    description: a tiny screening call
    slots:
      - {name: phone, kind: phone, question: "What's your phone number?"}
      - {name: when, kind: date, question: "When can you start?"}
""",
        encoding="utf-8",
    )
    return spec


class TestPipelineDryRun:
    """End-to-end with pre-generated dialogue, fake TTS and fake STT."""

    async def test_dry_run_produces_a_scored_interview(self, tmp_path: Path) -> None:
        bench = BenchmarkConfig.from_dict({
            "stt": [
                {"name": "vendor-perfect", "modes": ["stream"]},
                {"name": "vendor-broken", "modes": ["stream"]},
            ],
            "run": {"repeats": 1, "warmup": 0, "output_dir": str(tmp_path)},
        })
        sim = SimConfig(
            scenarios_path=str(_mini_spec(tmp_path)),
            personas_per_scenario=2,
            seed=1234,
        )

        outcome = await run_sim(
            bench,
            sim,
            llm=_stub_llm,
            tts=_stub_tts,
            run_stt=_fake_run_stt("vendor-perfect", "vendor-broken"),
        )

        assert outcome.seed == 1234
        assert len(outcome.turns) == 4
        assert all(t.verified for t in outcome.turns)
        perfect = outcome.vendor_scores["vendor-perfect"]
        broken = outcome.vendor_scores["vendor-broken"]
        assert perfect.success == pytest.approx(1.0)
        assert broken.success is not None
        assert broken.success < 1.0
        assert outcome.spend["total"] >= 0.0

        results_path = tmp_path / "stt-results.jsonl"
        results_path.write_bytes(b"")
        sim_path, gate_path, report_path = write_sim_outputs(outcome, None, results_path)
        records = [orjson.loads(line) for line in sim_path.read_bytes().splitlines() if line]
        assert len(records) == 4
        assert all(r["vendors"] for r in records)
        assert orjson.loads(gate_path.read_bytes())["gate"] is None
        assert "Task success" in report_path.read_text(encoding="utf-8")

    async def test_unverified_turns_are_not_voiced(self) -> None:
        async def hopeless_llm(system: str, prompt: str, temperature: float) -> str:  # ruff: ignore[unused-async] -- awaited through the LLM callable contract
            return "question?" if "interviewer" in system else "no comment."

        scenario = load_scenarios(SCENARIOS_EN)[0]
        persona = sample_personas(scenario, 1, np.random.default_rng(2))[0]
        turns = await conduct_interview(scenario, persona, hopeless_llm, voice="af_heart")

        assert all(not t.verified for t in turns)
        assert all(t.attempts == 3 for t in turns)
        assert clips_from_turns(turns, _stub_tts, language="en-US") == []

    async def test_unknown_degradation_is_rejected(self) -> None:
        turn = Turn(
            scenario_id="s",
            persona_id="p",
            index=0,
            slot=load_scenarios(SCENARIOS_EN)[0].slots[0],
            truth=SlotValue("x", "x", "x"),
            verified=True,
            voice="af_heart",
            clip_id="c",
            answer="hello",
        )
        with pytest.raises(ConfigError):
            clips_from_turns([turn], _stub_tts, language="en-US", degradation="babble")
        assert DEGRADATION == "tel8k"


class TestSpendGuards:
    """The estimate prints before spend, and the hard cap aborts."""

    def test_estimate_counts_turns_and_lanes(self) -> None:
        bench = BenchmarkConfig.from_dict({"stt": [{"name": "deepgram-nova3", "modes": ["stream"]}]})
        scenarios = load_scenarios(SCENARIOS_EN)

        estimate = estimate_spend(bench, scenarios, 8, est_answer_s=12.0)

        assert estimate.turns == 3 * 8 * 8
        assert estimate.stt_usd["deepgram-nova3"] > 0.0
        assert estimate.total_usd < 50.0
        assert "expected total" in estimate.render()

    def test_hard_cap_aborts(self) -> None:
        bench = BenchmarkConfig.from_dict({"stt": [{"name": "google-chirp3", "modes": ["stream"]}]})
        estimate = estimate_spend(bench, load_scenarios(SCENARIOS_EN), 8, est_answer_s=12.0)

        with pytest.raises(ConfigError):
            ensure_within_cap(estimate, 0.01)


# --------------------------------------------------------------------------
# Env-gated live smoke (minimal-API policy: one mini-interview, one vendor)
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AUDIO_HARNESS_SIM_SMOKE") != "1",
    reason="live smoke needs AUDIO_HARNESS_SIM_SMOKE=1 (paid API calls)",
)
class TestLiveSmoke:
    """One two-slot interview, one vendor, real Gemini + Kokoro (~$0.01)."""

    async def test_mini_interview_end_to_end(self, tmp_path: Path) -> None:
        from audio_harness.cli import load_env_file
        from audio_harness.sim.interview import KokoroSynth, gemini_llm

        load_env_file()
        bench = BenchmarkConfig.from_dict({
            "stt": [{"name": "deepgram-nova3", "modes": ["stream"]}],
            "run": {
                "repeats": 1,
                "warmup": 0,
                "provider_concurrency": 1,
                "output_dir": str(tmp_path),
            },
        })
        sim = SimConfig(
            scenarios_path=str(_mini_spec(tmp_path)),
            personas_per_scenario=1,
            voices=("af_heart",),
        )

        outcome = await run_sim(bench, sim, llm=gemini_llm(sim.model), tts=KokoroSynth(("af_heart",)))

        assert outcome.usage.calls > 0
        score = outcome.vendor_scores["deepgram-nova3"]
        assert score.scorable >= 1
        assert outcome.spend["total"] < 0.10
