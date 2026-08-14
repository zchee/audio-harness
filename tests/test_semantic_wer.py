"""Tests for the pipecat-style semantic WER judge (network-free plus a gated live smoke)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import orjson
import pytest

from audio_harness.judge import semantic_wer
from audio_harness.judge.semantic import JudgeItem


def _item(
    clip_id: str = "0001",
    provider: str = "deepgram-nova3",
    mode: str = "batch",
    reference: str = "I must initiate an immediate trace.",
    hypothesis: str = "I must initiate an immediate trade.",
) -> JudgeItem:
    return JudgeItem(
        provider=provider,
        mode=mode,
        clip_id=clip_id,
        language="en-US",
        reference=reference,
        hypothesis=hypothesis,
    )


def _tool_response(substitutions: int, deletions: int, insertions: int, reference_words: int) -> httpx.Response:
    """A Messages API reply whose only content is a calculate_wer call."""
    return httpx.Response(
        200,
        json={
            "content": [
                {"type": "text", "text": "Semantic check complete."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "calculate_wer",
                    "input": {
                        "substitutions": substitutions,
                        "deletions": deletions,
                        "insertions": insertions,
                        "reference_words": reference_words,
                        "normalized_reference": "ref",
                        "normalized_hypothesis": "hyp",
                        "errors": [
                            {"type": "substitution", "reference": "trace", "hypothesis": "trade", "position": 5}
                        ],
                    },
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 120, "output_tokens": 400, "cache_read_input_tokens": 3000},
        },
    )


class _CountingTransport(httpx.MockTransport):
    """MockTransport that counts the requests it served."""

    def __init__(self, handler) -> None:
        self.calls = 0

        def counted(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            return handler(request)

        super().__init__(counted)


def test_judge_pair_computes_rate_programmatically() -> None:
    """The division comes from the tool input, not from any model text."""
    transport = _CountingTransport(lambda request: _tool_response(1, 0, 0, 32))

    async def run() -> semantic_wer.SemanticWerVerdict:
        async with httpx.AsyncClient(transport=transport) as client:
            return await semantic_wer.judge_pair(client, "ref text", "hyp text", api_key="test-key")

    verdict = asyncio.run(run())
    assert verdict.wer == pytest.approx(1 / 32)
    assert verdict.total_errors == 1
    assert verdict.reference_words == 32
    assert verdict.num_turns == 1
    assert verdict.output_tokens == 400
    assert transport.calls == 1


def test_judge_pair_empty_cases_never_bill() -> None:
    """Empty-text pairs resolve deterministically, exactly as upstream."""
    transport = _CountingTransport(lambda request: _tool_response(0, 0, 0, 0))

    tests = {
        "success: both empty scores zero": ("", "", 0, 0, 0, 0, 0.0),
        "success: empty hypothesis is full deletion": ("three word reference", "", 0, 3, 0, 3, 1.0),
        "success: empty reference is pure insertion": ("", "two words", 0, 0, 2, 0, None),
    }

    async def run(reference: str, hypothesis: str) -> semantic_wer.SemanticWerVerdict:
        async with httpx.AsyncClient(transport=transport) as client:
            return await semantic_wer.judge_pair(client, reference, hypothesis, api_key="test-key")

    for name, (reference, hypothesis, subs, dels, ins, words, wer) in tests.items():
        verdict = asyncio.run(run(reference, hypothesis))
        assert verdict.substitutions == subs, name
        assert verdict.deletions == dels, name
        assert verdict.insertions == ins, name
        assert verdict.reference_words == words, name
        assert verdict.wer == wer, name
    assert transport.calls == 0


def test_judge_pair_without_tool_call_is_an_error() -> None:
    """A conversation that ends in prose is a failure, never a silent zero."""
    transport = _CountingTransport(
        lambda request: httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "The WER is 0%."}], "stop_reason": "end_turn", "usage": {}},
        )
    )

    async def run() -> None:
        async with httpx.AsyncClient(transport=transport) as client:
            await semantic_wer.judge_pair(client, "ref", "hyp", api_key="test-key")

    with pytest.raises(semantic_wer.SemanticWerError):
        asyncio.run(run())


def test_judge_items_serves_repeats_from_cache(tmp_path: Path) -> None:
    """A second run over the same merge bills nothing."""
    transport = _CountingTransport(lambda request: _tool_response(1, 0, 0, 32))
    cache_path = tmp_path / "semantic-wer-cache.jsonl"
    items = [_item()]

    first, first_stats = asyncio.run(
        semantic_wer.judge_items(items, semantic_wer.VerdictCache(cache_path), api_key="test-key", transport=transport)
    )
    assert first_stats.live_calls == 1
    assert transport.calls == 1

    second, second_stats = asyncio.run(
        semantic_wer.judge_items(items, semantic_wer.VerdictCache(cache_path), api_key="test-key", transport=transport)
    )
    assert second_stats.live_calls == 0
    assert second_stats.cached_verdicts == 1
    assert transport.calls == 1
    assert second[0].verdict.from_cache is True
    assert second[0].verdict.wer == first[0].verdict.wer


def test_judge_items_counts_failures(tmp_path: Path) -> None:
    """A dead conversation is dropped and counted, never scored."""
    transport = _CountingTransport(
        lambda request: httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn", "usage": {}},
        )
    )
    verdicts, stats = asyncio.run(
        semantic_wer.judge_items(
            [_item()], semantic_wer.VerdictCache(tmp_path / "cache.jsonl"), api_key="test-key", transport=transport
        )
    )
    assert verdicts == []
    assert stats.failures == 1


def test_summarize_pools_counts_and_pairs_deterministic_wer() -> None:
    """Lane rates pool edits before dividing, and both columns share clips."""
    entries = [
        semantic_wer.ItemVerdict(
            item=_item(clip_id="0001", reference="an immediate trace", hypothesis="an immediate trade"),
            verdict=semantic_wer.SemanticWerVerdict(1, 0, 0, 3, 1 / 3),
        ),
        semantic_wer.ItemVerdict(
            item=_item(clip_id="0002", reference="the coastal areas", hypothesis="the coastal areas"),
            verdict=semantic_wer.SemanticWerVerdict(0, 0, 0, 3, 0.0),
        ),
    ]
    summaries = semantic_wer.summarize(entries)
    assert len(summaries) == 1
    lane = summaries[0]
    assert lane.clips == 2
    assert lane.semantic_errors == 1
    assert lane.semantic_reference_words == 6
    assert lane.semantic_wer == pytest.approx(1 / 6)
    assert lane.deterministic_wer == pytest.approx(1 / 6)


def test_markdown_carries_the_experimental_banner(tmp_path: Path) -> None:
    """Judge-based numbers must never render without the not-ranked banner."""
    entries = [
        semantic_wer.ItemVerdict(
            item=_item(),
            verdict=semantic_wer.SemanticWerVerdict(1, 0, 0, 32, 1 / 32),
        )
    ]
    summaries = semantic_wer.summarize(entries)
    markdown = semantic_wer.render_markdown(summaries)
    assert semantic_wer.EXPERIMENTAL_BANNER in markdown
    assert "| deepgram-nova3 | batch | 1 | 3.12% |" in markdown

    out = semantic_wer.write_results(entries, summaries, semantic_wer.JudgeRunStats(), tmp_path / "semantic-wer.json")
    payload = orjson.loads(out.read_bytes())
    assert payload["banner"] == semantic_wer.EXPERIMENTAL_BANNER
    assert payload["lanes"][0]["semantic_wer"] == pytest.approx(1 / 32)
    assert payload["items"][0]["clip_id"] == "0001"


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("AUDIO_HARNESS_LIVE")),
    reason="live smoke needs ANTHROPIC_API_KEY and AUDIO_HARNESS_LIVE",
)
def test_live_smoke_single_pair() -> None:
    """One real judged pair: the trace/trade example must count one error."""

    async def run() -> semantic_wer.SemanticWerVerdict:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            return await semantic_wer.judge_pair(
                client,
                "I must initiate an immediate trace.",
                "I must initiate an immediate trade.",
                api_key=os.environ["ANTHROPIC_API_KEY"],
            )

    verdict = asyncio.run(run())
    assert verdict.total_errors >= 1
    assert verdict.reference_words > 0
    assert verdict.num_turns >= 1
