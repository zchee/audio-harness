"""Tests for the audio-LLM pairwise TTS arena (plan P4 step 17).

Every test except the env-gated live smoke injects a stub judge, so the
Bradley-Terry fit, bootstrap CIs, order-swap bookkeeping, panel decision rule
and cache behaviour are all verified against hand-computed fixtures without
spending a single API call.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import math
import os
from pathlib import Path

import numpy as np
import orjson
import pytest
import soundfile as sf

from audio_harness.judge import tts_arena
from audio_harness.judge.tts_arena import (
    ArenaError,
    BtScore,
    Comparison,
    JudgeReply,
    OrderFlipStats,
    PanelVote,
    Verdict,
    bootstrap_bt,
    bt_scores,
    bt_table,
    ci_separated_pairs,
    comparisons_from_verdicts,
    evaluate_cross_family_gate,
    evaluate_gate,
    exact_permutation_p,
    inter_rater_agreement,
    judge_family,
    load_arena_prompts,
    order_flip_stats,
    pair_wav_bytes,
    parse_verdict,
    render_arena_markdown,
    render_cross_family_gate,
    run_arena,
    spearman_rho,
    write_arena_outputs,
    write_cross_family_gate,
)
from audio_harness.types import TtsPrompt


def _verdict(
    first: str,
    second: str,
    winner: str,
    *,
    aspect: str = "naturalness",
    prompt_id: str = "p1",
) -> Verdict:
    return Verdict(aspect=aspect, prompt_id=prompt_id, first=first, second=second, winner=winner)


def _write_wav(path: Path, *, seconds: float = 0.3, freq: float = 220.0) -> Path:
    rate = 16000
    t = np.arange(int(rate * seconds)) / rate
    sf.write(path, (0.4 * np.sin(2 * np.pi * freq * t)).astype("float32"), rate)
    return path


class CountingJudge:
    """Stub judge that counts calls and replies with a fixed text."""

    def __init__(self, reply: str = "FIRST") -> None:
        self.reply = reply
        self.calls = 0

    async def __call__(self, wav: bytes, instruction: str) -> JudgeReply:
        self.calls += 1
        return JudgeReply(text=self.reply, prompt_tokens=100, output_tokens=1)


class TestLoadArenaPrompts:
    """Prompt ids must be stable across runs so WAVs and caches survive."""

    def test_ids_carry_stem_and_line_number(self, tmp_path: Path) -> None:
        file = tmp_path / "interview.txt"
        file.write_text("one\n\ntwo\nthree\n", encoding="utf-8")

        prompts = load_arena_prompts([str(file)], language="en-US", seed=1)

        assert [p.prompt_id for p in prompts] == [
            "interview-001",
            "interview-003",
            "interview-004",
        ]
        assert prompts[0].text == "one"
        assert prompts[0].language == "en-US"

    def test_sampling_is_seeded_and_reproducible(self, tmp_path: Path) -> None:
        file = tmp_path / "general.txt"
        file.write_text("\n".join(f"line {i}" for i in range(20)), encoding="utf-8")

        once = load_arena_prompts([f"{file}:5"], language="en-US", seed=7)
        again = load_arena_prompts([f"{file}:5"], language="en-US", seed=7)

        assert len(once) == 5
        assert [p.prompt_id for p in once] == [p.prompt_id for p in again]

    def test_count_at_least_file_size_takes_everything(self, tmp_path: Path) -> None:
        file = tmp_path / "entities.txt"
        file.write_text("a\nb\n", encoding="utf-8")

        prompts = load_arena_prompts([f"{file}:10"], language="ja-JP", seed=0)

        assert len(prompts) == 2

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ArenaError, match="not found"):
            load_arena_prompts([str(tmp_path / "nope.txt")], language="en-US", seed=0)

    def test_duplicate_stems_would_collide(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        one = tmp_path / "a" / "general.txt"
        two = tmp_path / "b" / "general.txt"
        one.write_text("x\n", encoding="utf-8")
        two.write_text("y\n", encoding="utf-8")

        with pytest.raises(ArenaError, match="collide"):
            load_arena_prompts([str(one), str(two)], language="en-US", seed=0)


class TestPairAudio:
    """Concatenated pair audio is clip one + gap + clip two at 16 kHz."""

    def test_durations_add_up(self, tmp_path: Path) -> None:
        first = np.zeros(8000, dtype=np.float32)
        second = np.zeros(4000, dtype=np.float32)

        wav = pair_wav_bytes(first, second, gap_s=0.5)

        file = tmp_path / "pair.wav"
        file.write_bytes(wav)
        with sf.SoundFile(file) as handle:
            assert handle.samplerate == tts_arena.ARENA_SAMPLE_RATE
            assert len(handle) == 8000 + 8000 + 4000


class TestParseVerdict:
    """Chatty replies still parse; unrecognizable ones never get guessed."""

    def test_single_word_forms(self) -> None:
        assert parse_verdict("FIRST") == "first"
        assert parse_verdict("second") == "second"
        assert parse_verdict("Tie.") == "tie"

    def test_first_token_in_a_sentence_wins(self) -> None:
        assert parse_verdict("The second clip sounds cleaner.") == "second"

    def test_garbage_returns_none(self) -> None:
        assert parse_verdict("I cannot compare these clips.") is None
        assert parse_verdict("") is None


class TestJudgeFamily:
    """Model ids select a persisted family or fail closed."""

    @pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-live-audio"])
    def test_gemini_prefix(self, model: str) -> None:
        assert judge_family(model) == "gemini"

    @pytest.mark.parametrize(
        "model",
        ["gpt-audio", "gpt-audio-2025-08-28", "gpt-audio-mini-2025-12-15", "gpt-audio-1.5"],
    )
    def test_openai_prefix(self, model: str) -> None:
        assert judge_family(model) == "openai"

    def test_unknown_prefix_is_rejected(self) -> None:
        with pytest.raises(ArenaError, match="unknown judge model family"):
            judge_family("claude-audio")

    @pytest.mark.parametrize("aspect", tts_arena.ASPECTS)
    def test_instruction_preamble_is_family_specific(self, aspect: str) -> None:
        gemini = tts_arena.aspect_instruction(aspect)
        openai = tts_arena.aspect_instruction(aspect, "openai")
        assert gemini.startswith("You will hear one audio file containing")
        assert openai.startswith("The attached audio contains")
        suffix = gemini.removeprefix("You will hear one audio file containing")
        assert openai.removeprefix("The attached audio contains") == suffix
        assert "Answer with exactly one word" in suffix

    def test_cost_estimate_dispatches_by_family(self) -> None:
        gemini = tts_arena.ArenaRun(
            model="gemini-2.5-flash",
            live_prompt_tokens=2_000_000,
            live_output_tokens=1_000_000,
        )
        openai = tts_arena.ArenaRun(
            model="gpt-audio-2025-08-28",
            live_prompt_tokens=2_000_000,
            live_output_tokens=1_000_000,
        )

        assert gemini.est_usd == pytest.approx(4.5)
        assert openai.est_usd == pytest.approx(74.0)


class TestOpenAiJudge:
    """The OpenAI factory uses raw HTTP with the documented audio part."""

    async def test_chat_completions_request_and_usage(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        class StubResponse:
            text = ""

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "id": "chatcmpl_test",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "SECOND"}}],
                    "usage": {
                        "prompt_tokens": 321,
                        "completion_tokens": 7,
                        "total_tokens": 328,
                    },
                }

        class StubClient:
            def __init__(self, *, timeout: None) -> None:
                captured["timeout"] = timeout

            async def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> StubResponse:
                captured.update(url=url, headers=headers, json=json)
                return StubResponse()

        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setattr(tts_arena.httpx, "AsyncClient", StubClient)
        model = "gpt-audio-2025-08-28"
        wav = b"RIFF-test-wav"
        instruction = "Choose FIRST, SECOND, or TIE."

        reply = await tts_arena._openai_judge(model)(wav, instruction)

        assert captured["url"] == tts_arena.OPENAI_CHAT_COMPLETIONS_URL
        assert captured["timeout"] is None
        assert captured["headers"] == {
            "Authorization": "Bearer test-openai-key",
            "Content-Type": "application/json",
        }
        body = captured["json"]
        assert isinstance(body, dict)
        assert body["model"] == model
        assert body["modalities"] == ["text"]
        assert body["temperature"] == 0
        content = body["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": instruction}
        assert content[1] == {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(wav).decode("ascii"),
                "format": "wav",
            },
        }
        assert reply == JudgeReply(text="SECOND", prompt_tokens=321, output_tokens=7)
        assert parse_verdict(reply.text) == "second"


class TestBradleyTerry:
    """Hand-computed fixtures: the 2-system MLE is the win ratio."""

    def test_three_to_one_gives_log_three(self) -> None:
        comparisons = [Comparison("a", "b", 1.0)] * 3 + [Comparison("a", "b", 0.0)]

        scores = bt_scores(comparisons, ["a", "b"])

        assert scores["a"] - scores["b"] == pytest.approx(math.log(3), abs=1e-6)
        assert scores["a"] + scores["b"] == pytest.approx(0.0, abs=1e-6)

    def test_ties_count_half_a_win_each(self) -> None:
        comparisons = [
            Comparison("a", "b", 1.0),
            Comparison("a", "b", 1.0),
            Comparison("a", "b", 0.5),
        ]

        scores = bt_scores(comparisons, ["a", "b"])

        # Win credit 2.5 vs 0.5: the two-system MLE is the ratio 5.
        assert scores["a"] - scores["b"] == pytest.approx(math.log(5), abs=1e-6)

    def test_no_data_scores_everyone_equal(self) -> None:
        scores = bt_scores([], ["a", "b", "c"])

        assert scores == {"a": 0.0, "b": 0.0, "c": 0.0}

    def test_comparison_order_does_not_matter(self) -> None:
        comparisons = [
            Comparison("a", "b", 1.0),
            Comparison("b", "c", 1.0),
            Comparison("c", "a", 0.0),
            Comparison("a", "b", 0.0),
        ]

        forward = bt_scores(comparisons, ["a", "b", "c"])
        backward = bt_scores(list(reversed(comparisons)), ["a", "b", "c"])

        for system in ("a", "b", "c"):
            assert forward[system] == pytest.approx(backward[system], abs=1e-8)

    def test_shutout_stays_finite_and_ranked_last(self) -> None:
        comparisons = [Comparison("a", "b", 1.0)] * 4

        scores = bt_scores(comparisons, ["a", "b"])

        assert scores["a"] > scores["b"]
        assert math.isfinite(scores["b"])


class TestBootstrapCi:
    """Cluster-bootstrap CIs: seeded, and separated only when the data is."""

    def _sweep(self, prompts: int, winner_by_prompt) -> list[Verdict]:
        verdicts = []
        for i in range(prompts):
            pid = f"p{i:02d}"
            better = winner_by_prompt(i)
            other = "b" if better == "a" else "a"
            verdicts.extend((
                _verdict(better, other, "first", prompt_id=pid),
                _verdict(other, better, "second", prompt_id=pid),
            ))
        return verdicts

    def test_same_seed_reproduces_the_interval(self) -> None:
        verdicts = self._sweep(6, lambda i: "a" if i % 2 else "b")

        one = bootstrap_bt(verdicts, ["a", "b"], n_boot=100, seed=42)
        two = bootstrap_bt(verdicts, ["a", "b"], n_boot=100, seed=42)

        assert one == two

    def test_lopsided_data_separates_the_cis(self) -> None:
        verdicts = self._sweep(10, lambda i: "a")

        table = bt_table(verdicts, ["a", "b"], n_boot=100, seed=0)

        assert [score.system for score in table] == ["a", "b"]
        assert ci_separated_pairs(table) == [("a", "b")]

    def test_balanced_data_overlaps(self) -> None:
        verdicts = self._sweep(10, lambda i: "a" if i % 2 else "b")

        table = bt_table(verdicts, ["a", "b"], n_boot=100, seed=0)

        assert ci_separated_pairs(table) == []

    def test_no_verdicts_gives_degenerate_zero_intervals(self) -> None:
        assert bootstrap_bt([], ["a", "b"], n_boot=10, seed=0) == {
            "a": (0.0, 0.0),
            "b": (0.0, 0.0),
        }


class TestOrderFlip:
    """Criterion (iii): a verdict that follows position, not audio, flips."""

    def test_positional_judge_flips_every_group(self) -> None:
        verdicts = [
            _verdict("a", "b", "first"),
            _verdict("b", "a", "first"),
        ]

        stats = order_flip_stats(verdicts)

        assert stats == OrderFlipStats(judged=1, flips=1, dropped=0)
        assert stats.rate == 1.0

    def test_consistent_winner_never_flips(self) -> None:
        verdicts = [
            _verdict("a", "b", "first"),
            _verdict("b", "a", "second"),
        ]

        assert order_flip_stats(verdicts).rate == 0.0

    def test_double_tie_is_consistent(self) -> None:
        verdicts = [
            _verdict("a", "b", "tie"),
            _verdict("b", "a", "tie"),
        ]

        assert order_flip_stats(verdicts).rate == 0.0

    def test_tie_against_a_win_counts_as_a_flip(self) -> None:
        verdicts = [
            _verdict("a", "b", "first"),
            _verdict("b", "a", "tie"),
        ]

        assert order_flip_stats(verdicts).rate == 1.0

    def test_errored_groups_are_dropped_not_judged(self) -> None:
        verdicts = [
            _verdict("a", "b", "first"),
            _verdict("b", "a", "error"),
            _verdict("a", "b", "first", aspect="prosody"),
            _verdict("b", "a", "second", aspect="prosody"),
        ]

        stats = order_flip_stats(verdicts)

        assert stats == OrderFlipStats(judged=1, flips=0, dropped=1)


class TestSpearman:
    """Rank correlation helpers used descriptively by the gate."""

    def test_perfect_agreement(self) -> None:
        assert spearman_rho([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)

    def test_perfect_disagreement(self) -> None:
        assert spearman_rho([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)

    def test_constant_vector_is_undefined(self) -> None:
        assert spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    def test_exact_permutation_p_for_perfect_match(self) -> None:
        # Only the identity permutation of three distinct values reaches
        # rho = 1, so the one-sided exact p is 1/6.
        p = exact_permutation_p([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        assert p == pytest.approx(1 / 6)


class TestInterRater:
    """Panel agreement is computed on orientation-normalized shared items."""

    def test_agreement_over_shared_items(self) -> None:
        votes = [
            # Item 1: both prefer system a (opposite presentation orders).
            PanelVote("r1", "p1", "a", "b", "first"),
            PanelVote("r2", "p1", "b", "a", "second"),
            # Item 2: r1 prefers a, r2 says tie.
            PanelVote("r1", "p2", "a", "b", "first"),
            PanelVote("r2", "p2", "a", "b", "tie"),
        ]

        percent, kappa = inter_rater_agreement(votes)

        assert percent == pytest.approx(0.5)
        assert kappa is not None

    def test_no_shared_items_is_undefined(self) -> None:
        votes = [
            PanelVote("r1", "p1", "a", "b", "first"),
            PanelVote("r2", "p2", "a", "b", "first"),
        ]

        assert inter_rater_agreement(votes) == (None, None)


def _scores_separated() -> list[BtScore]:
    return [
        BtScore("a", 1.0, 0.5, 1.5, 30.0, 40.0),
        BtScore("b", -1.0, -1.5, -0.5, 10.0, 40.0),
    ]


def _panel(winner: str, *, votes_per_rater: int = 100) -> list[PanelVote]:
    return [PanelVote(f"r{rater}", f"p{i:03d}", "a", "b", winner) for rater in (1, 2) for i in range(votes_per_rater)]


class TestGate:
    """The three-criterion gate records everything and fakes nothing."""

    def _flips_pass(self) -> OrderFlipStats:
        return OrderFlipStats(judged=100, flips=10, dropped=0)

    def test_no_panel_renders_panel_pending(self) -> None:
        gate = evaluate_gate(_scores_separated(), self._flips_pass(), None)

        human, family, flip = gate.criteria
        assert human.status == "pending"
        assert family.status == "uncomputable"
        assert flip.status == "pass"
        assert not gate.passed
        assert gate.label.startswith("experimental (panel pending")

    def test_concordant_sized_panel_passes_criterion_one(self) -> None:
        gate = evaluate_gate(_scores_separated(), self._flips_pass(), _panel("first"))

        assert gate.criteria[0].status == "pass"
        # Criterion (ii) stays uncomputable, so the lane still never ranks.
        assert not gate.passed
        assert "not ranked" not in gate.criteria[0].detail

    def test_discordant_ci_separated_pair_fails(self) -> None:
        gate = evaluate_gate(_scores_separated(), self._flips_pass(), _panel("second"))

        assert gate.criteria[0].status == "fail"
        assert "a vs b" in gate.criteria[0].detail
        assert "failed: human-panel agreement" in gate.label

    def test_undersized_panel_stays_pending(self) -> None:
        panel = _panel("first", votes_per_rater=5)

        gate = evaluate_gate(_scores_separated(), self._flips_pass(), panel)

        assert gate.criteria[0].status == "pending"
        assert "below the pre-registered size" in gate.criteria[0].detail

    def test_overlapping_cis_have_nothing_to_be_discordant_about(self) -> None:
        overlapping = [
            BtScore("a", 0.2, -0.5, 0.9, 22.0, 40.0),
            BtScore("b", -0.2, -0.9, 0.5, 18.0, 40.0),
        ]

        gate = evaluate_gate(overlapping, self._flips_pass(), _panel("second"))

        assert gate.criteria[0].status == "pass"

    def test_second_family_makes_criterion_two_computable(self) -> None:
        gate = evaluate_gate(
            _scores_separated(),
            self._flips_pass(),
            _panel("first"),
            second_family_scores={"a": 2.0, "b": -2.0},
        )

        assert gate.criteria[1].status == "pass"
        assert gate.passed
        assert gate.label.startswith("gate passed")

    def test_order_flip_above_gate_fails(self) -> None:
        flips = OrderFlipStats(judged=100, flips=20, dropped=0)

        gate = evaluate_gate(_scores_separated(), flips, None)

        assert gate.criteria[2].status == "fail"
        assert "20.0%" in gate.criteria[2].detail


class TestRunArenaCache:
    """Judge calls are order-counterbalanced, cached, and never re-billed."""

    def _stage(self, tmp_path: Path) -> tuple[Path, list[TtsPrompt]]:
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for system, freq in (("sysa", 220.0), ("sysb", 440.0)):
            for pid in ("p1", "p2"):
                _write_wav(audio_dir / f"{system}-batch-{pid}.wav", freq=freq)
        prompts = [
            TtsPrompt(prompt_id="p1", text="hello", language="en-US"),
            TtsPrompt(prompt_id="p2", text="world", language="en-US"),
        ]
        return audio_dir, prompts

    async def test_call_count_and_cache_hits(self, tmp_path: Path) -> None:
        audio_dir, prompts = self._stage(tmp_path)
        cache = tmp_path / "judge-cache.jsonl"

        first_judge = CountingJudge()
        first = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=cache,
            judge=first_judge,
        )

        # 1 pair x 2 prompts x 3 aspects x 2 orders = 12 calls.
        assert first_judge.calls == 12
        assert (first.live_calls, first.cached_calls, first.error_calls) == (12, 0, 0)
        assert first.language == "en-US"
        assert first.est_usd > 0

        second_judge = CountingJudge()
        second = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=cache,
            judge=second_judge,
        )

        assert second_judge.calls == 0
        assert (second.live_calls, second.cached_calls) == (0, 12)
        assert sorted(v.winner for v in second.verdicts) == sorted(v.winner for v in first.verdicts)
        assert all(v.cached for v in second.verdicts)

    async def test_cache_keys_are_separate_across_judge_models(self, tmp_path: Path) -> None:
        audio_dir, prompts = self._stage(tmp_path)
        cache = tmp_path / "judge-cache.jsonl"
        gemini_model = "gemini-2.5-flash"
        openai_model = "gpt-audio-2025-08-28"

        gemini_judge = CountingJudge()
        gemini_first = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=cache,
            model=gemini_model,
            judge=gemini_judge,
        )
        openai_judge = CountingJudge()
        openai_first = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=cache,
            model=openai_model,
            judge=openai_judge,
        )

        assert gemini_judge.calls == openai_judge.calls == 12
        assert (gemini_first.live_calls, gemini_first.cached_calls) == (12, 0)
        assert (openai_first.live_calls, openai_first.cached_calls) == (12, 0)

        gemini_cached = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=cache,
            model=gemini_model,
            judge=CountingJudge(),
        )
        openai_cached = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=cache,
            model=openai_model,
            judge=CountingJudge(),
        )

        assert (gemini_cached.live_calls, gemini_cached.cached_calls) == (0, 12)
        assert (openai_cached.live_calls, openai_cached.cached_calls) == (0, 12)
        assert {verdict.model for verdict in gemini_cached.verdicts} == {gemini_model}
        assert {verdict.model for verdict in openai_cached.verdicts} == {openai_model}

    async def test_positional_stub_judge_flips_everything(self, tmp_path: Path) -> None:
        audio_dir, prompts = self._stage(tmp_path)

        run = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=tmp_path / "cache.jsonl",
            judge=CountingJudge(reply="FIRST"),
        )

        assert order_flip_stats(run.verdicts).rate == 1.0

    async def test_missing_audio_skips_the_pair_and_says_so(self, tmp_path: Path) -> None:
        audio_dir, prompts = self._stage(tmp_path)
        (audio_dir / "sysb-batch-p2.wav").unlink()

        run = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=tmp_path / "cache.jsonl",
            judge=CountingJudge(),
        )

        assert run.missing_audio == ["sysb:p2"]
        assert len(run.verdicts) == 6

    async def test_unparseable_replies_error_and_are_not_cached(self, tmp_path: Path) -> None:
        audio_dir, prompts = self._stage(tmp_path)
        cache = tmp_path / "cache.jsonl"

        run = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts,
            cache_path=cache,
            judge=CountingJudge(reply="no idea"),
        )

        assert run.error_calls == 12
        assert all(v.winner == "error" for v in run.verdicts)
        assert comparisons_from_verdicts(run.verdicts) == []
        assert not cache.is_file() or not cache.read_text(encoding="utf-8").strip()

    async def test_hung_judge_times_out_instead_of_deadlocking(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without a per-attempt deadline, hung calls fill every concurrency
        # slot and the run freezes forever (observed live on the ja lane).
        audio_dir, prompts = self._stage(tmp_path)
        monkeypatch.setattr(tts_arena, "_RETRY_BASE_S", 0.01)

        async def hang(wav: bytes, instruction: str) -> JudgeReply:
            await asyncio.sleep(60)
            return JudgeReply(text="FIRST")

        run = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts[:1],
            cache_path=tmp_path / "cache.jsonl",
            judge=hang,
            call_timeout_s=0.05,
        )

        assert run.error_calls == 6
        assert all("Timeout" in (v.error or "") for v in run.verdicts)

    async def test_httpx_status_uses_the_shared_retry_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audio_dir, prompts = self._stage(tmp_path)
        monkeypatch.setattr(tts_arena, "_RETRY_BASE_S", 0.0)

        class RateLimitedOncePerJob:
            def __init__(self) -> None:
                self.calls = 0

            async def __call__(self, wav: bytes, instruction: str) -> JudgeReply:
                self.calls += 1
                if self.calls % 2:
                    request = tts_arena.httpx.Request("POST", tts_arena.OPENAI_CHAT_COMPLETIONS_URL)
                    response = tts_arena.httpx.Response(429, request=request)
                    raise tts_arena.httpx.HTTPStatusError(
                        "rate limited",
                        request=request,
                        response=response,
                    )
                return JudgeReply(text="FIRST", prompt_tokens=100, output_tokens=1)

        judge = RateLimitedOncePerJob()
        run = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts[:1],
            cache_path=tmp_path / "cache.jsonl",
            model="gpt-audio",
            judge=judge,
            concurrency=1,
        )

        assert judge.calls == 12
        assert (run.live_calls, run.error_calls) == (6, 0)

    async def test_malformed_openai_usage_is_retried_and_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        audio_dir, prompts = self._stage(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setattr(tts_arena, "_RETRY_BASE_S", 0.0)
        posts = 0

        class StubResponse:
            text = ""

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "choices": [{"message": {"content": "FIRST"}}],
                    "usage": "not-an-object",
                }

        class StubClient:
            def __init__(self, *, timeout: None) -> None:
                assert timeout is None

            async def post(self, url: str, **kwargs: object) -> StubResponse:
                nonlocal posts
                posts += 1
                return StubResponse()

        monkeypatch.setattr(tts_arena.httpx, "AsyncClient", StubClient)

        run = await run_arena(
            audio_dir=audio_dir,
            systems=("sysa", "sysb"),
            prompts=prompts[:1],
            cache_path=tmp_path / "cache.jsonl",
            model="gpt-audio",
        )

        assert posts == 6 * tts_arena._JUDGE_ATTEMPTS
        assert run.error_calls == 6
        assert all("usage is not an object" in (verdict.error or "") for verdict in run.verdicts)

    async def test_one_system_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ArenaError, match="at least two"):
            await run_arena(
                audio_dir=tmp_path,
                systems=("sysa",),
                prompts=[],
                cache_path=tmp_path / "cache.jsonl",
                judge=CountingJudge(),
            )


def _write_cross_family_fixture(
    directory: Path,
    *,
    model: str,
    family: str,
    systems: list[str],
    strengths: dict[str, float],
    winner_overrides: dict[tuple[str, str], str] | None = None,
    language: str = "en-US",
) -> Path:
    """Write one completed summary plus order-counterbalanced verdicts."""
    directory.mkdir()
    summary = directory / "arena-summary.json"
    summary.write_bytes(
        orjson.dumps(
            {
                "language": language,
                "model": model,
                "family": family,
                "systems": systems,
                "bt": [
                    {
                        "system": system,
                        "log_strength": strength,
                        "ci_low": strength - 0.1,
                        "ci_high": strength + 0.1,
                        "wins": 10.0,
                        "games": 12.0,
                    }
                    for system, strength in strengths.items()
                ],
            },
            option=orjson.OPT_INDENT_2,
        )
    )

    verdicts = []
    overrides = winner_overrides or {}
    for one, two in itertools.combinations(sorted(systems), 2):
        outcome = overrides.get((one, two), one)
        for first, second in ((one, two), (two, one)):
            if outcome == "tie":
                winner = "tie"
            elif outcome == first:
                winner = "first"
            elif outcome == second:
                winner = "second"
            else:
                raise AssertionError(f"invalid fixture outcome {outcome!r} for {one!r}/{two!r}")
            verdicts.append({
                "aspect": "naturalness",
                "prompt_id": "p1",
                "first": first,
                "second": second,
                "winner": winner,
                "model": model,
            })
    (directory / "arena-results.jsonl").write_bytes(b"".join(orjson.dumps(verdict) + b"\n" for verdict in verdicts))
    return summary


class TestCrossFamilyGate:
    """Persisted summaries make criterion (ii) independently computable."""

    def test_perfect_bt_rho_and_pair_agreement_diagnostic(self, tmp_path: Path) -> None:
        systems = ["a", "b", "c", "d"]
        summary_a = _write_cross_family_fixture(
            tmp_path / "gemini",
            model="gemini-2.5-flash",
            family="gemini",
            systems=systems,
            strengths={"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0},
            winner_overrides={("a", "b"): "tie"},
        )
        summary_b = _write_cross_family_fixture(
            tmp_path / "openai",
            model="gpt-audio-2025-08-28",
            family="openai",
            systems=list(reversed(systems)),
            strengths={"d": 10.0, "b": 30.0, "a": 40.0, "c": 20.0},
            winner_overrides={("a", "b"): "tie", ("c", "d"): "d"},
        )

        result = evaluate_cross_family_gate(summary_a, summary_b)

        assert result.rho == pytest.approx(1.0)
        assert result.criterion.status == "pass"
        assert result.pair_agreement.compared == 6
        assert result.pair_agreement.agreements == 5
        assert result.pair_agreement.rate == pytest.approx(5 / 6)
        markdown = render_cross_family_gate(result)
        assert "| cross-family judge agreement | pass |" in markdown
        assert "5/6 comparable pair groups" in markdown
        assert "agree = 83.3%" in markdown
        report = write_cross_family_gate(result)
        assert report == summary_a.with_name("arena-cross-family-gate.md")
        assert report.read_text(encoding="utf-8") == markdown + "\n"

    def test_same_family_is_rejected(self, tmp_path: Path) -> None:
        systems = ["a", "b"]
        summary_a = _write_cross_family_fixture(
            tmp_path / "one",
            model="gemini-2.5-flash",
            family="gemini",
            systems=systems,
            strengths={"a": 1.0, "b": -1.0},
        )
        summary_b = _write_cross_family_fixture(
            tmp_path / "two",
            model="gemini-2.5-pro",
            family="gemini",
            systems=systems,
            strengths={"a": 2.0, "b": -2.0},
        )

        with pytest.raises(ArenaError, match=r"different judge families.*gemini.*gemini"):
            evaluate_cross_family_gate(summary_a, summary_b)

    def test_mismatched_system_sets_name_the_difference(self, tmp_path: Path) -> None:
        summary_a = _write_cross_family_fixture(
            tmp_path / "one",
            model="gemini-2.5-flash",
            family="gemini",
            systems=["a", "b", "c"],
            strengths={"a": 2.0, "b": 1.0, "c": 0.0},
        )
        summary_b = _write_cross_family_fixture(
            tmp_path / "two",
            model="gpt-audio",
            family="openai",
            systems=["a", "b", "d"],
            strengths={"a": 2.0, "b": 1.0, "d": 0.0},
        )

        with pytest.raises(ArenaError, match=r"symmetric difference: \['c', 'd'\]"):
            evaluate_cross_family_gate(summary_a, summary_b)

    def test_mismatched_languages_are_rejected(self, tmp_path: Path) -> None:
        systems = ["a", "b"]
        summary_a = _write_cross_family_fixture(
            tmp_path / "one",
            model="gemini-2.5-flash",
            family="gemini",
            systems=systems,
            strengths={"a": 1.0, "b": -1.0},
            language="en-US",
        )
        summary_b = _write_cross_family_fixture(
            tmp_path / "two",
            model="gpt-audio",
            family="openai",
            systems=systems,
            strengths={"a": 1.0, "b": -1.0},
            language="ja-JP",
        )

        with pytest.raises(ArenaError, match=r"languages differ: 'en-US' vs 'ja-JP'"):
            evaluate_cross_family_gate(summary_a, summary_b)


class TestOutputs:
    """Reports lead with the gate label; the summary carries AC8 metrics."""

    def _fixture(self):
        verdicts = [_verdict("a", "b", "first", prompt_id=f"p{i:02d}") for i in range(4)] + [
            _verdict("b", "a", "second", prompt_id=f"p{i:02d}") for i in range(4)
        ]
        run = tts_arena.ArenaRun(
            verdicts=verdicts,
            systems=("a", "b"),
            language="en-US",
            live_calls=8,
        )
        scores = bt_table(verdicts, ["a", "b"], n_boot=50, seed=0)
        flips = order_flip_stats(verdicts)
        gate = evaluate_gate(scores, flips, None)
        return run, scores, flips, gate

    def test_markdown_leads_with_the_gate_label(self) -> None:
        run, scores, _flips, gate = self._fixture()

        markdown = render_arena_markdown(run, scores, gate, notes=["ja interview prompts are PENDING"])

        assert markdown.startswith("## TTS arena (en-US) - experimental (panel")
        assert "| a |" in markdown
        assert "uncomputable" in markdown
        assert "Note: ja interview prompts are PENDING" in markdown

    def test_outputs_round_trip(self, tmp_path: Path) -> None:
        import orjson

        run, scores, flips, gate = self._fixture()

        results, summary, report = write_arena_outputs(tmp_path, run, scores, flips, gate, notes=["note"])

        assert results.is_file()
        assert report.is_file()
        payload = orjson.loads(summary.read_bytes())
        assert payload["language"] == "en-US"
        assert payload["family"] == "gemini"
        assert payload["gate"]["criteria"][1]["status"] == "uncomputable"
        # One aspect over four prompts: four swap-groups, none flipped.
        assert payload["order_flip"]["judged"] == 4
        assert payload["order_flip"]["flips"] == 0
        assert [row["system"] for row in payload["bt"]] == ["a", "b"]
        lines = results.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(run.verdicts)


@pytest.mark.skipif(
    not os.environ.get("AUDIO_HARNESS_TEST_ARENA_LIVE"),
    reason="live Gemini judge smoke is opt-in: AUDIO_HARNESS_TEST_ARENA_LIVE=1",
)
class TestLiveJudge:
    """One real judge call (fractions of a cent) verifying the API contract."""

    async def test_judge_reply_parses(self) -> None:
        rate = tts_arena.ARENA_SAMPLE_RATE
        t = np.arange(rate) / rate
        clean = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        rng = np.random.default_rng(0)
        harsh = np.clip(clean * 5.0 + rng.normal(0, 0.3, clean.shape), -1.0, 1.0).astype(np.float32)

        judge = tts_arena._gemini_judge(tts_arena.JUDGE_MODEL)
        reply = await judge(pair_wav_bytes(clean, harsh), tts_arena.aspect_instruction("artifacts"))

        assert parse_verdict(reply.text) is not None
        assert reply.prompt_tokens > 0
