"""Benchmark orchestration.

Two scheduling rules keep the measurements honest:

* Clips within a provider run **sequentially**. Issuing them concurrently would
  make each request queue behind the others and inflate the very latency the
  benchmark is trying to measure.
* Providers run **concurrently**, because they are independent services and
  serializing them would multiply wall-clock time for no gain in accuracy.

One persistence rule keeps the measurements recoverable: completed lanes are
**flushed to disk immediately**. The API calls are the expensive part of a
benchmark, and writing results only at end-of-run meant an interrupted run
lost every finished lane. Each lane's records now land in the run's results
file the moment the lane completes, so a crash preserves finished lanes and a
rerun of only the missing ones can be folded in by the report's supersede
merge.
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
from .types import AudioClip, Mode, Partial, SttResult, TtsPrompt, TtsResult


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

    Completed lanes are persisted under ``config.run.output_dir`` as the run
    progresses, so an aborted run keeps every finished lane on disk in a file
    :func:`read_stt_results` can load. :func:`write_stt_results` reuses the
    same directory to land the canonical end-of-run file.
    """
    limiter = asyncio.Semaphore(config.run.provider_concurrency)
    vendors = _VendorLocks(config.run.vendor_concurrency)
    sink = _LaneSink(_begin_run(config.run.output_dir, "stt-results.jsonl"))
    lanes = [
        _stt_lane(entry, mode, clips, config.run, limiter, vendors, progress, sink)
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
    sink: _LaneSink,
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
                    settle_ms = provider.settle_ms or run.settle_ms
                    if transport is Mode.STREAM and settle_ms > 0:
                        await asyncio.sleep(settle_ms / 1000)
        finally:
            await provider.aclose()
        await sink.lane_done(b"".join(_stt_record(result) for result in results))
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
            result.raw["language"] = clip.language
            result.audio_s = clip.duration_s
            result.error = f"timeout after {run.timeout_s}s"
            return result
        except Exception as exc:
            if attempt + 1 < attempts and _is_transient(exc):
                await asyncio.sleep(run.retry_backoff_s * (2**attempt))
                continue
            result = SttResult(provider=provider.key, clip_id=clip.clip_id, mode=mode)
            result.raw["language"] = clip.language
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
    "quota",
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

    Completed lanes are persisted under ``config.run.output_dir`` as the run
    progresses, so an aborted run keeps every finished lane on disk in a file
    :func:`read_tts_results` can load. The snapshots record metrics only —
    synthesized audio files and round-trip verdicts are added by the
    end-of-run :func:`write_tts_results`, which reuses the same directory.
    """
    limiter = asyncio.Semaphore(config.run.provider_concurrency)
    vendors = _VendorLocks(config.run.vendor_concurrency)
    sink = _LaneSink(_begin_run(config.run.output_dir, "tts-results.jsonl"))
    lanes = [
        _tts_lane(entry, mode, prompts, config.run, limiter, vendors, progress, sink)
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
    sink: _LaneSink,
) -> list[TtsResult]:
    """Run one provider in one mode across every prompt, sequentially.

    The warmup pass is recorded rather than discarded: flagged ``cold``,
    those runs carry the connection-establishment cost the first real user
    request pays, and the report splits them out instead of blending them
    into the warm percentiles. When the config asks for a load pass, the
    prompt set is repeated with several syntheses in flight, tagged so the
    report keeps them out of the sequential lane.
    """
    provider = tts.create(entry.name, entry.options)
    async with limiter, vendors.get(provider.billing_group):
        transport = Mode(mode)
        results: list[TtsResult] = []

        if progress is not None:
            progress.start(entry.name, mode, len(prompts) * run.repeats)

        try:
            for _ in range(run.warmup):
                cold = await _one_tts(provider, prompts[0], transport, run)
                cold.cold = True
                results.append(cold)
            for _ in range(run.repeats):
                for prompt in prompts:
                    result = await _one_tts(provider, prompt, transport, run)
                    results.append(result)
                    if progress is not None:
                        progress.result(entry.name, mode, result.ok)
            if transport is Mode.STREAM and run.tts_load_concurrency > 1:
                results.extend(await _tts_load_pass(provider, prompts, run))
        finally:
            await provider.aclose()
        await sink.lane_done(b"".join(_tts_record(result, None) for result in results))
        return results


async def _tts_load_pass(
    provider: tts.TtsProvider, prompts: list[TtsPrompt], run: RunConfig
) -> list[TtsResult]:
    """Repeat the prompt set with several syntheses in flight at once.

    A voice agent under load holds concurrent sessions, and a vendor that
    looks fast sequentially can queue under concurrency. Each prompt is
    issued ``tts_load_concurrency`` times simultaneously; every result is
    tagged with the load factor so the report renders these as their own
    lane instead of folding them into the sequential percentiles.
    """
    load = run.tts_load_concurrency
    results: list[TtsResult] = []
    for prompt in prompts:
        batch = await asyncio.gather(
            *(_one_tts(provider, prompt, Mode.STREAM, run) for _ in range(load))
        )
        for result in batch:
            result.raw["load"] = load
        results.extend(batch)
    return results


async def _one_tts(
    provider: tts.TtsProvider, prompt: TtsPrompt, mode: Mode, run: RunConfig
) -> TtsResult:
    """Execute a single synthesis, converting any failure into a result.

    With incremental text enabled, streaming lanes whose protocol accepts
    appended text are fed at LLM-token cadence; lanes without that support
    fall back to whole-prompt streaming and record ``input_streaming: false``
    so the report never presents a fallback as the real thing.
    """
    try:
        async with asyncio.timeout(run.timeout_s):
            if mode is Mode.STREAM:
                if run.tts_incremental_text:
                    if provider.supports_input_streaming:
                        return await provider.synthesize_incremental(
                            prompt, token_rate=run.tts_token_rate
                        )
                    result = await provider.synthesize_stream(prompt)
                    result.raw["input_streaming"] = False
                    return result
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
    feed the synthesized audio back through fixed recognizers and compare
    against the prompt. A high round-trip error rate means the voice is hard to
    decode. The judges are held constant across TTS providers so the score is
    comparative, not absolute — it is a proxy, not a MOS.

    Every configured judge transcribes every clip. Judges must span vendor
    families because a recognizer decodes its own vendor's voices best — the
    report ranks each lane only by judges outside that lane's family and
    demotes same-family scores to diagnostics.

    Configure each recognizer with text formatting off. Otherwise its inverse
    text normalization rewrites spoken numbers into digits, every engine
    inherits the same rewrite, and the metric collapses to a constant that
    says nothing about the voices being compared.

    Verdicts land in each result's ``raw["roundtrip"]`` as a list of
    ``{provider, text, error}`` mappings in config order; the report layer
    turns those into error rates.

    Cold-lane warmup runs are skipped: the same voice is already scored on
    its warm runs, and judging the duplicate would double-bill the judges.

    Args:
        config: Benchmark definition, providing ``roundtrip_stt``.
        results: TTS results to score, mutated in place.
        prompts: Prompt lookup keyed by prompt id.
    """
    for entry in config.roundtrip_stt:
        provider = stt.create(entry.name, entry.options)
        try:
            for result in results:
                if not result.ok or result.cold:
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
                verdict: dict[str, object] = {
                    "provider": entry.name,
                    "text": None,
                    "error": None,
                }
                try:
                    transcription = await provider.transcribe_batch(clip)
                    verdict["text"] = transcription.text
                except Exception as exc:
                    verdict["error"] = f"{type(exc).__name__}: {exc}"
                verdicts = result.raw.get("roundtrip")
                if not isinstance(verdicts, list):
                    verdicts = []
                    result.raw["roundtrip"] = verdicts
                verdicts.append(verdict)
        finally:
            await provider.aclose()


def _stt_record(result: SttResult) -> bytes:
    """Serialize one STT result to its JSONL line, newline included.

    Shared by the per-lane snapshots and :func:`write_stt_results` so the
    crash artifact and the canonical file can never drift apart in format.
    """
    return (
        orjson.dumps(
            {
                "provider": result.provider,
                "clip_id": result.clip_id,
                "mode": str(result.mode),
                "text": result.text,
                "reference": result.raw.get("reference", ""),
                "reference_annotated": result.raw.get("reference_annotated"),
                "license": result.raw.get("license"),
                "gold_status": result.raw.get("gold_status"),
                "language": result.raw.get("language", ""),
                "audio_s": result.audio_s,
                "total_s": result.total_s,
                "ttft_s": result.ttft_s,
                "finalize_s": result.finalize_s,
                "rtf": result.rtf,
                "chunk_ms": result.raw.get("chunk_ms"),
                "speech_end_s": result.raw.get("speech_end_s"),
                "pauses": result.raw.get("pauses"),
                "ws_rtt_s": result.raw.get("ws_rtt_s"),
                "eou_source": result.raw.get("eou_source"),
                "endpoint_config": result.raw.get("endpoint_config"),
                "error": result.error,
                "partials": [
                    {
                        "t_s": p.t_s,
                        "text": p.text,
                        "is_final": p.is_final,
                        "kind": p.kind,
                    }
                    for p in result.partials
                ],
            }
        )
        + b"\n"
    )


def write_stt_results(results: list[SttResult], output_dir: str | Path) -> Path:
    """Persist STT results as JSONL and return the file path.

    Interim hypotheses are kept so churn can be re-analysed without re-running
    the benchmark, which is the expensive part. When the results come from a
    run that persisted its lanes incrementally, the canonical file lands in
    that run's directory, replacing the lane snapshots.
    """
    path = _prepare(output_dir, "stt-results.jsonl")
    with path.open("wb") as handle:
        for result in results:
            handle.write(_stt_record(result))
    return path


def _tts_record(result: TtsResult, audio_path: str | None) -> bytes:
    """Serialize one TTS result to its JSONL line, newline included.

    Shared by the per-lane snapshots — which pass no audio path, because
    audio files are written at end of run — and :func:`write_tts_results`,
    so the crash artifact and the canonical file can never drift in format.
    """
    return (
        orjson.dumps(
            {
                "provider": result.provider,
                "prompt_id": result.prompt_id,
                "mode": str(result.mode),
                "chars": result.chars,
                "audio_s": result.audio_s,
                "ttfb_s": result.ttfb_s,
                "ttfa_s": result.ttfa_s,
                "gap_p99_s": result.gap_p99_s,
                "cold": result.cold,
                "chunk_t_s": result.chunk_t_s,
                "total_s": result.total_s,
                "rtf": result.rtf,
                "load": result.raw.get("load"),
                "input_streaming": result.raw.get("input_streaming"),
                "error": result.error,
                "text": result.raw.get("text", ""),
                "roundtrip": result.raw.get("roundtrip"),
                "audio_path": audio_path,
            }
        )
        + b"\n"
    )


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

            handle.write(_tts_record(result, audio_path))
    return path


def read_stt_results(path: str | Path) -> list[SttResult]:
    """Load STT results back from a saved JSONL file.

    Re-scoring saved runs is the point: the expensive part of a benchmark is
    the API calls, so a normalization fix or a merge of two runs should not
    require paying for them again. Interim hypotheses are restored too, which
    is what makes churn re-computable after the fact.

    Args:
        path: JSONL file written by :func:`write_stt_results`.

    Returns:
        The reconstructed results.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is not valid JSON.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"results file not found: {file}")

    results: list[SttResult] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"{file}:{number}: invalid JSON") from exc

        result = SttResult(
            provider=record["provider"],
            clip_id=record["clip_id"],
            mode=Mode(record["mode"]),
            text=record.get("text", ""),
            audio_s=float(record.get("audio_s") or 0.0),
            total_s=float(record.get("total_s") or 0.0),
            ttft_s=record.get("ttft_s"),
            finalize_s=record.get("finalize_s"),
            error=record.get("error"),
        )
        result.partials = [
            # Records written before the EOU migration carry no kind; the
            # Partial default derives it from is_final, which is exactly the
            # pre-migration semantics.
            Partial(
                t_s=p["t_s"],
                text=p["text"],
                is_final=p["is_final"],
                kind=p.get("kind") or "",
            )
            for p in record.get("partials", [])
        ]
        result.raw["reference"] = record.get("reference", "")
        result.raw["language"] = record.get("language", "")
        if record.get("reference_annotated"):
            result.raw["reference_annotated"] = record["reference_annotated"]
        if record.get("license"):
            result.raw["license"] = record["license"]
        if record.get("gold_status"):
            result.raw["gold_status"] = record["gold_status"]
        if record.get("chunk_ms") is not None:
            result.raw["chunk_ms"] = record["chunk_ms"]
        if record.get("speech_end_s") is not None:
            result.raw["speech_end_s"] = record["speech_end_s"]
        if record.get("pauses"):
            result.raw["pauses"] = record["pauses"]
        if record.get("ws_rtt_s") is not None:
            result.raw["ws_rtt_s"] = record["ws_rtt_s"]
        if record.get("eou_source"):
            result.raw["eou_source"] = record["eou_source"]
        if record.get("endpoint_config"):
            result.raw["endpoint_config"] = record["endpoint_config"]
        results.append(result)
    return results


def read_tts_results(path: str | Path) -> list[TtsResult]:
    """Load TTS results back from a saved JSONL file.

    Audio bytes are not restored — the JSONL stores a path, not samples — but
    every scored field is, so a report can be re-rendered without paying for
    synthesis again. Records written before the two-judge migration carry a
    scalar ``roundtrip_text``/``roundtrip_provider`` pair; those are folded
    into the ``roundtrip`` list so historical runs merge as one-judge lanes
    rather than losing their score.

    Args:
        path: JSONL file written by :func:`write_tts_results`.

    Returns:
        The reconstructed results.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line is not valid JSON.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"results file not found: {file}")

    results: list[TtsResult] = []
    for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"{file}:{number}: invalid JSON") from exc

        result = TtsResult(
            provider=record["provider"],
            prompt_id=record["prompt_id"],
            mode=Mode(record["mode"]),
            chars=int(record.get("chars") or 0),
            audio_s=float(record.get("audio_s") or 0.0),
            ttfb_s=record.get("ttfb_s"),
            ttfa_s=record.get("ttfa_s"),
            gap_p99_s=record.get("gap_p99_s"),
            cold=bool(record.get("cold", False)),
            chunk_t_s=[float(t) for t in record.get("chunk_t_s") or []],
            total_s=float(record.get("total_s") or 0.0),
            error=record.get("error"),
        )
        result.raw["text"] = record.get("text", "")
        if record.get("load") is not None:
            result.raw["load"] = int(record["load"])
        if record.get("input_streaming") is not None:
            result.raw["input_streaming"] = bool(record["input_streaming"])
        roundtrip = record.get("roundtrip")
        if isinstance(roundtrip, list):
            result.raw["roundtrip"] = roundtrip
        elif isinstance(record.get("roundtrip_text"), str):
            result.raw["roundtrip"] = [
                {
                    "provider": record.get("roundtrip_provider") or "",
                    "text": record["roundtrip_text"],
                    "error": None,
                }
            ]
        if record.get("audio_path"):
            result.raw["audio_path"] = record["audio_path"]
        results.append(result)
    return results


class _LaneSink:
    """Persists each completed lane so an aborted run keeps its finished work.

    Every lane completion rewrites the results file through a temporary path
    and an atomic replace, so the file on disk always holds exactly the lanes
    that finished — never a torn line from a crash mid-write. Lanes land in
    completion order; the end-of-run writer replaces the file in canonical
    config order, and the report's supersede merge keys on provider and mode,
    so an interim file reports correctly either way.
    """

    __slots__ = ("_chunks", "_lock", "path")

    def __init__(self, path: Path) -> None:
        """Remember the file to maintain; nothing is written until a lane ends.

        Args:
            path: Results file inside the run's directory.
        """
        self.path = path
        self._chunks: list[bytes] = []
        self._lock = asyncio.Lock()

    async def lane_done(self, records: bytes) -> None:
        """Add one completed lane's serialized records and flush to disk.

        The write runs off the event loop so a large flush cannot stall
        concurrently streaming lanes and distort their latency measurements;
        the lock serializes flushes from lanes finishing at the same moment.

        Args:
            records: The lane's JSONL lines, newline-terminated.
        """
        async with self._lock:
            self._chunks.append(records)
            payload = b"".join(self._chunks)
            await asyncio.to_thread(self._flush, payload)

    def _flush(self, payload: bytes) -> None:
        """Replace the results file atomically with the given content."""
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(self.path)


_ACTIVE_RUNS: dict[tuple[str, str], Path] = {}


def _begin_run(output_dir: str | Path, filename: str) -> Path:
    """Open a run's results file path at run start, remembering the directory.

    Creating the timestamped directory up front lets lanes persist as they
    finish; the registry entry lets the end-of-run writer land the canonical
    file over the lane snapshots instead of opening a second directory.

    Args:
        output_dir: Root results directory from the run configuration.
        filename: Results file name, which also namespaces the registry so
            an STT and a TTS run against one root cannot collide.

    Returns:
        The results file path inside the freshly created run directory.
    """
    directory = Path(output_dir) / time.strftime("%Y%m%d-%H%M%S")
    directory.mkdir(parents=True, exist_ok=True)
    _ACTIVE_RUNS[(str(Path(output_dir)), filename)] = directory
    return directory / filename


def _prepare(output_dir: str | Path, filename: str) -> Path:
    """Return the run's file path, creating a timestamped directory if needed.

    A run that persisted its lanes incrementally already owns a directory;
    the final write reuses it so the canonical file replaces the snapshots.
    A standalone write — re-scoring, merging, tests — creates a fresh one.
    """
    directory = _ACTIVE_RUNS.pop((str(Path(output_dir)), filename), None)
    if directory is None:
        directory = Path(output_dir) / time.strftime("%Y%m%d-%H%M%S")
        directory.mkdir(parents=True, exist_ok=True)
    return directory / filename
