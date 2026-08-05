"""Benchmark orchestration.

Two scheduling rules keep the measurements honest:

* Clips within a provider run **sequentially**. Issuing them concurrently would
  make each request queue behind the others and inflate the very latency the
  benchmark is trying to measure.
* Providers run **concurrently**, because they are independent services and
  serializing them would multiply wall-clock time for no gain in accuracy.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import orjson

from . import stt, tts
from .audio import decode_audio_duration, wrap_wav
from .config import BenchmarkConfig, ProviderConfig, RunConfig
from .types import AudioClip, Mode, SttResult, TtsPrompt, TtsResult


@dataclass(slots=True)
class Progress:
    """Callbacks the CLI uses to render progress without owning the loop."""

    on_start: object = None
    on_result: object = None

    def start(self, provider: str, mode: str, total: int) -> None:
        """Announce that a provider/mode lane is beginning."""
        if callable(self.on_start):
            self.on_start(provider, mode, total)

    def result(self, provider: str, mode: str, ok: bool) -> None:
        """Announce that one run finished."""
        if callable(self.on_result):
            self.on_result(provider, mode, ok)


async def run_stt(
    config: BenchmarkConfig, clips: list[AudioClip], progress: Progress | None = None
) -> list[SttResult]:
    """Benchmark every configured STT provider over every clip.

    Args:
        config: Benchmark definition.
        clips: Evaluation clips.
        progress: Optional progress sink.

    Returns:
        One result per provider, mode, clip and repeat. Failed runs are
        included with their ``error`` set so failure rates stay visible.
    """
    limiter = asyncio.Semaphore(config.run.provider_concurrency)
    vendors = _VendorLocks(config.run.vendor_concurrency)
    lanes = [
        _stt_lane(entry, mode, clips, config.run, limiter, vendors, progress)
        for entry in config.stt
        for mode in entry.modes
    ]
    batches = await asyncio.gather(*lanes)
    return [result for batch in batches for result in batch]


class _VendorLocks:
    """Caps how many lanes may hold sessions against one vendor account.

    Two benchmark entries can be one account — Speechmatics Standard and
    Enhanced, or Deepgram STT and TTS — and free plans cap concurrent
    sessions. Without this, those lanes race each other and the vendor
    rejects half of them, which looks like a provider failure rather than a
    scheduling mistake in the harness.
    """

    __slots__ = ("_limit", "_locks")

    def __init__(self, limit: int) -> None:
        self._limit = max(1, limit)
        self._locks: dict[str, asyncio.Semaphore] = {}

    def get(self, vendor: str) -> asyncio.Semaphore:
        """Return the semaphore guarding one vendor account."""
        if vendor not in self._locks:
            self._locks[vendor] = asyncio.Semaphore(self._limit)
        return self._locks[vendor]


async def _stt_lane(
    entry: ProviderConfig,
    mode: str,
    clips: list[AudioClip],
    run: RunConfig,
    limiter: asyncio.Semaphore,
    vendors: _VendorLocks,
    progress: Progress | None,
) -> list[SttResult]:
    """Run one provider in one mode across every clip, sequentially."""
    provider = stt.create(entry.name, entry.options)
    async with limiter, vendors.get(provider.billing_group):
        transport = Mode(mode)
        results: list[SttResult] = []

        if progress is not None:
            progress.start(entry.name, mode, len(clips) * run.repeats)

        try:
            await _warmup_stt(provider, clips[0], transport, run)
            for _ in range(run.repeats):
                for clip in clips:
                    result = await _one_stt(provider, clip, transport, run)
                    results.append(result)
                    if progress is not None:
                        progress.result(entry.name, mode, result.ok)
                    # Let the vendor release the finished session before the
                    # next one opens; without this the lane races itself.
                    if transport is Mode.STREAM and run.settle_ms > 0:
                        await asyncio.sleep(run.settle_ms / 1000)
        finally:
            await provider.aclose()
        return results


async def _warmup_stt(
    provider: stt.SttProvider, clip: AudioClip, mode: Mode, run: RunConfig
) -> None:
    """Absorb DNS, TLS and cold-start cost so it does not skew run one."""
    for _ in range(run.warmup):
        await _one_stt(provider, clip, mode, run)


async def _one_stt(
    provider: stt.SttProvider, clip: AudioClip, mode: Mode, run: RunConfig
) -> SttResult:
    """Execute a single transcription, converting any failure into a result.

    The requested frame size is clamped to what the provider accepts and the
    effective value is recorded, so a vendor that had to be fed larger frames
    is visible in the report rather than silently penalised on latency.
    """
    chunk_ms = provider.effective_chunk_ms(run.chunk_ms)
    attempts = max(1, run.transient_retries + 1)

    for attempt in range(attempts):
        try:
            async with asyncio.timeout(run.timeout_s):
                if mode is Mode.STREAM:
                    result = await provider.transcribe_stream(
                        clip, chunk_ms=chunk_ms, realtime=run.realtime
                    )
                    result.raw["chunk_ms"] = chunk_ms
                    _rebase_finalize(result, clip, realtime=run.realtime)
                    if attempt:
                        result.raw["retries"] = attempt
                    return result
                return await provider.transcribe_batch(clip)
        except TimeoutError, asyncio.CancelledError:
            result = SttResult(provider=provider.key, clip_id=clip.clip_id, mode=mode)
            result.audio_s = clip.duration_s
            result.error = f"timeout after {run.timeout_s}s"
            return result
        except Exception as exc:
            if attempt + 1 < attempts and _is_transient(exc):
                await asyncio.sleep(run.retry_backoff_s * (2**attempt))
                continue
            result = SttResult(provider=provider.key, clip_id=clip.clip_id, mode=mode)
            result.audio_s = clip.duration_s
            result.error = f"{type(exc).__name__}: {exc}"
            return result

    raise AssertionError("unreachable: the retry loop always returns")


_TRANSIENT_MARKERS = (
    "concurrent",
    "concurrency",
    "too many",
    "rate limit",
    "rate_limit",
    "429",
    "503",
    "temporarily unavailable",
)


def _is_transient(exc: Exception) -> bool:
    """Whether a failure is a capacity refusal rather than a real error.

    Serializing lanes per account is not always enough: a vendor can still
    refuse a session while the previous one finishes winding down. Those
    refusals say nothing about the provider's accuracy or latency, and
    recording them as failures would blame the vendor for the harness'
    scheduling. Genuine errors — bad audio, auth, malformed requests — do not
    match and are reported as-is.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _rebase_finalize(result: SttResult, clip: AudioClip, *, realtime: bool) -> None:
    """Re-measure finalization latency from end of speech, not end of file.

    Recorded clips carry trailing silence. A provider that endpoints during it
    emits its final transcript before the file runs out, which reads as zero —
    or negative — latency when measured from the last audio byte. Ranking
    providers on that number rewards whoever ignores the tail of the clip.

    Timing from the last voiced frame instead answers the question a voice
    agent actually asks: once the user stopped talking, how long until the
    transcript was final.

    Only valid under real-time pacing, where elapsed stream time equals
    playback position. The end-of-file figure is preserved in ``raw`` so the
    two can be compared.

    Args:
        result: Streaming result to adjust in place.
        clip: Clip that was streamed, carrying its speech-end offset.
        realtime: Whether audio was paced at playback speed.
    """
    result.raw["finalize_from_eof_s"] = result.finalize_s
    result.raw["speech_end_s"] = clip.speech_end_s
    if not realtime or clip.speech_end_s <= 0:
        return

    finals = [p for p in result.partials if p.is_final]
    if not finals:
        result.finalize_s = None
        return
    result.finalize_s = max(0.0, finals[-1].t_s - clip.speech_end_s)


async def run_tts(
    config: BenchmarkConfig,
    prompts: list[TtsPrompt],
    progress: Progress | None = None,
) -> list[TtsResult]:
    """Benchmark every configured TTS provider over every prompt.

    Args:
        config: Benchmark definition.
        prompts: Texts to synthesize.
        progress: Optional progress sink.

    Returns:
        One result per provider, mode, prompt and repeat.
    """
    limiter = asyncio.Semaphore(config.run.provider_concurrency)
    vendors = _VendorLocks(config.run.vendor_concurrency)
    lanes = [
        _tts_lane(entry, mode, prompts, config.run, limiter, vendors, progress)
        for entry in config.tts
        for mode in entry.modes
    ]
    batches = await asyncio.gather(*lanes)
    return [result for batch in batches for result in batch]


async def _tts_lane(
    entry: ProviderConfig,
    mode: str,
    prompts: list[TtsPrompt],
    run: RunConfig,
    limiter: asyncio.Semaphore,
    vendors: _VendorLocks,
    progress: Progress | None,
) -> list[TtsResult]:
    """Run one provider in one mode across every prompt, sequentially."""
    provider = tts.create(entry.name, entry.options)
    async with limiter, vendors.get(provider.billing_group):
        transport = Mode(mode)
        results: list[TtsResult] = []

        if progress is not None:
            progress.start(entry.name, mode, len(prompts) * run.repeats)

        try:
            for _ in range(run.warmup):
                await _one_tts(provider, prompts[0], transport, run)
            for _ in range(run.repeats):
                for prompt in prompts:
                    result = await _one_tts(provider, prompt, transport, run)
                    results.append(result)
                    if progress is not None:
                        progress.result(entry.name, mode, result.ok)
        finally:
            await provider.aclose()
        return results


async def _one_tts(
    provider: tts.TtsProvider, prompt: TtsPrompt, mode: Mode, run: RunConfig
) -> TtsResult:
    """Execute a single synthesis, converting any failure into a result."""
    try:
        async with asyncio.timeout(run.timeout_s):
            if mode is Mode.STREAM:
                return await provider.synthesize_stream(prompt)
            return await provider.synthesize(prompt)
    except TimeoutError, asyncio.CancelledError:
        result = TtsResult(provider=provider.key, prompt_id=prompt.prompt_id, mode=mode)
        result.chars = prompt.chars
        result.error = f"timeout after {run.timeout_s}s"
        return result
    except Exception as exc:
        result = TtsResult(provider=provider.key, prompt_id=prompt.prompt_id, mode=mode)
        result.chars = prompt.chars
        result.error = f"{type(exc).__name__}: {exc}"
        return result


async def score_roundtrip(
    config: BenchmarkConfig,
    results: list[TtsResult],
    prompts: dict[str, TtsPrompt],
) -> None:
    """Transcribe synthesized audio to estimate intelligibility.

    Naturalness cannot be measured without listeners, but intelligibility can:
    feed the synthesized audio back through one fixed recognizer and compare
    against the prompt. A high round-trip error rate means the voice is hard to
    decode. The recognizer is held constant across TTS providers so the score
    is comparative, not absolute — it is a proxy, not a MOS.

    Configure the recognizer with text formatting off. Otherwise its inverse
    text normalization rewrites spoken numbers into digits, every engine
    inherits the same rewrite, and the metric collapses to a constant that
    says nothing about the voices being compared.

    The transcript is written to each result's ``raw`` under
    ``roundtrip_text``; the metrics layer turns that into an error rate.

    Args:
        config: Benchmark definition, providing ``roundtrip_stt``.
        results: TTS results to score, mutated in place.
        prompts: Prompt lookup keyed by prompt id.
    """
    if config.roundtrip_stt is None:
        return

    entry = config.roundtrip_stt
    provider = stt.create(entry.name, entry.options)
    try:
        for result in results:
            if not result.ok:
                continue
            prompt = prompts.get(result.prompt_id)
            if prompt is None:
                continue
            clip = AudioClip(
                clip_id=f"{result.provider}-{result.prompt_id}",
                pcm=result.audio,
                sample_rate=result.sample_rate,
                duration_s=decode_audio_duration(
                    result.audio,
                    encoding=result.encoding,
                    sample_rate=result.sample_rate,
                ),
                reference=prompt.text,
                language=prompt.language,
                source_path="<synthesized>",
            )
            try:
                transcription = await provider.transcribe_batch(clip)
                result.raw["roundtrip_text"] = transcription.text
                result.raw["roundtrip_provider"] = entry.name
            except Exception as exc:
                result.raw["roundtrip_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        await provider.aclose()


def write_stt_results(results: list[SttResult], output_dir: str | Path) -> Path:
    """Persist STT results as JSONL and return the file path.

    Interim hypotheses are kept so churn can be re-analysed without re-running
    the benchmark, which is the expensive part.
    """
    path = _prepare(output_dir, "stt-results.jsonl")
    with path.open("wb") as handle:
        for result in results:
            handle.write(
                orjson.dumps(
                    {
                        "provider": result.provider,
                        "clip_id": result.clip_id,
                        "mode": str(result.mode),
                        "text": result.text,
                        "reference": result.raw.get("reference", ""),
                        "audio_s": result.audio_s,
                        "total_s": result.total_s,
                        "ttft_s": result.ttft_s,
                        "finalize_s": result.finalize_s,
                        "rtf": result.rtf,
                        "chunk_ms": result.raw.get("chunk_ms"),
                        "error": result.error,
                        "partials": [
                            {"t_s": p.t_s, "text": p.text, "is_final": p.is_final}
                            for p in result.partials
                        ],
                    }
                )
            )
            handle.write(b"\n")
    return path


def write_tts_results(
    results: list[TtsResult], output_dir: str | Path, *, save_audio: bool
) -> Path:
    """Persist TTS results as JSONL, optionally writing the audio alongside."""
    path = _prepare(output_dir, "tts-results.jsonl")
    audio_dir = path.parent / "audio"
    if save_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as handle:
        for result in results:
            audio_path: str | None = None
            if save_audio and result.ok:
                name = f"{result.provider}-{result.mode}-{result.prompt_id}.wav"
                target = audio_dir / name
                target.write_bytes(wrap_wav(result.audio, result.sample_rate))
                audio_path = str(target)

            handle.write(
                orjson.dumps(
                    {
                        "provider": result.provider,
                        "prompt_id": result.prompt_id,
                        "mode": str(result.mode),
                        "chars": result.chars,
                        "audio_s": result.audio_s,
                        "ttfb_s": result.ttfb_s,
                        "total_s": result.total_s,
                        "rtf": result.rtf,
                        "error": result.error,
                        "text": result.raw.get("text", ""),
                        "roundtrip_text": result.raw.get("roundtrip_text"),
                        "roundtrip_provider": result.raw.get("roundtrip_provider"),
                        "audio_path": audio_path,
                    }
                )
            )
            handle.write(b"\n")
    return path


def _prepare(output_dir: str | Path, filename: str) -> Path:
    """Create the output directory and return a timestamped file path."""
    directory = Path(output_dir) / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename
