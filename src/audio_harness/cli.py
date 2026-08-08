"""Command-line interface."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import random
from typing import Annotated

import orjson
from rich.console import Console
from rich.table import Table
import typer

from . import agreement, doctor as doctor_module, realdata, report, runner, stt, tts
from .config import PRICING_CHECKED, BenchmarkConfig, ConfigError
from .dataset import DatasetError, load_clips, load_prompts
from .types import SttResult, TtsResult


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Benchmark speech-to-text and text-to-speech providers.",
)
console = Console()

DEFAULT_ENV_FILE = ".env"


def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> int:
    """Load ``KEY=value`` pairs from a dotenv file into the environment.

    Existing environment variables win, so an explicit export always overrides
    the file. Blank lines, ``#`` comments and quoted values are handled.

    Args:
        path: Dotenv file to read; a missing file is not an error.

    Returns:
        The number of variables set.
    """
    file = Path(path)
    if not file.is_file():
        return 0

    loaded = 0
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if name and value and name not in os.environ:
            os.environ[name] = value
            loaded += 1
    return loaded


def _load_config(path: Path) -> BenchmarkConfig:
    """Load a benchmark config, exiting cleanly on a configuration error."""
    try:
        return BenchmarkConfig.from_yaml(path)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def validate_roundtrip_judges(config: BenchmarkConfig) -> None:
    """Reject judge sets that cannot score every TTS lane fairly.

    A recognizer decodes its own vendor family's voices best, so a lane ranked
    by a same-family judge is flattered rather than measured. Every candidate
    therefore needs at least one judge from a different family; same-family
    judges are allowed but only ever count as diagnostics.

    This lives in the CLI load path rather than in ``config.py`` because
    resolving a family needs the adapter registries, and ``config.py`` cannot
    import those without a circular import through ``require_env``.

    Args:
        config: Parsed benchmark definition.

    Raises:
        ConfigError: If a judge or candidate key is unknown, or if some
            candidate has no cross-family judge.
    """
    if not config.roundtrip_stt or not config.tts:
        return

    judge_families: dict[str, str] = {}
    for judge in config.roundtrip_stt:
        if judge.name not in stt.available():
            raise ConfigError(
                f"roundtrip_stt: unknown STT provider {judge.name!r}; available: {', '.join(stt.available())}"
            )
        judge_families[judge.name] = stt.family_of(judge.name)

    for candidate in config.tts:
        if candidate.name not in tts.available():
            raise ConfigError(f"tts: unknown TTS provider {candidate.name!r}; available: {', '.join(tts.available())}")
        family = tts.family_of(candidate.name)
        if all(judge_family == family for judge_family in judge_families.values()):
            raise ConfigError(
                f"roundtrip_stt: every judge shares family {family!r} with "
                f"TTS candidate {candidate.name!r}, so its lane would be "
                f"scored by its own vendor's recognizer. Add a judge from "
                f"another family (e.g. whisper-local)."
            )


def _progress() -> runner.Progress:
    """Build a progress sink that prints one line per completed lane."""
    totals: dict[tuple[str, str], int] = {}
    done: dict[tuple[str, str], int] = {}
    failed: dict[tuple[str, str], int] = {}

    def on_start(provider: str, mode: str, total: int) -> None:
        totals[provider, mode] = total
        done[provider, mode] = 0
        failed[provider, mode] = 0
        console.print(f"[dim]start[/dim] {provider} [cyan]{mode}[/cyan] ({total} runs)")

    def on_result(provider: str, mode: str, ok: bool) -> None:
        key = (provider, mode)
        done[key] += 1
        if not ok:
            failed[key] += 1
        if done[key] == totals.get(key):
            note = f" [red]{failed[key]} failed[/red]" if failed[key] else ""
            console.print(f"[green]done[/green]  {provider} [cyan]{mode}[/cyan]{note}")

    return runner.Progress(on_start=on_start, on_result=on_result)


@app.command()
def doctor(
    env_file: Annotated[Path, typer.Option(help="Dotenv file to load before checking.")] = Path(DEFAULT_ENV_FILE),
) -> None:
    """Verify every provider credential with a cheap authenticated request."""
    load_env_file(env_file)
    results = asyncio.run(doctor_module.run_checks())

    table = Table(title="Credential check", show_lines=False)
    table.add_column("Provider")
    table.add_column("Env var", style="dim")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")

    for result in results:
        if result.skipped:
            status = "[yellow]skipped[/yellow]"
        elif result.ok:
            status = "[green]ok[/green]"
        else:
            status = "[red]failed[/red]"
        table.add_row(result.provider, result.env_var, status, result.detail)

    console.print(table)

    broken = [r for r in results if not r.ok and not r.skipped]
    missing = [r for r in results if r.skipped]
    if missing:
        console.print(f"[yellow]{len(missing)} credential(s) not set.[/yellow]")
    if broken:
        console.print(f"[red]{len(broken)} credential(s) failed.[/red]")
        raise typer.Exit(code=1)
    console.print("[green]All configured credentials authenticated.[/green]")


@app.command()
def providers() -> None:
    """List every registered provider and the modes it supports."""
    table = Table(title="Registered providers")
    table.add_column("Kind")
    table.add_column("Key")
    table.add_column("Batch")
    table.add_column("Stream")

    for key in stt.available():
        adapter = stt.create(key)
        table.add_row(
            "stt",
            key,
            "yes" if adapter.supports_batch else "—",
            "yes" if adapter.supports_stream else "—",
        )
    for key in tts.available():
        adapter_tts = tts.create(key)
        table.add_row(
            "tts",
            key,
            "yes" if adapter_tts.supports_batch else "—",
            "yes" if adapter_tts.supports_stream else "—",
        )

    console.print(table)
    console.print(f"[dim]Pricing table last verified {PRICING_CHECKED}.[/dim]")


@app.command("stt")
def stt_command(
    config_path: Annotated[Path, typer.Argument(help="Benchmark YAML config.")],
    env_file: Annotated[Path, typer.Option(help="Dotenv file.")] = Path(DEFAULT_ENV_FILE),
) -> None:
    """Run the speech-to-text benchmark."""
    load_env_file(env_file)
    config = _load_config(config_path)
    if not config.stt:
        console.print("[red]config error:[/red] no stt providers configured")
        raise typer.Exit(code=2)

    try:
        clips = load_clips(config.dataset)
    except DatasetError as exc:
        console.print(f"[red]dataset error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    total_audio = sum(clip.duration_s for clip in clips)
    console.print(f"Loaded {len(clips)} clips ({total_audio:.1f}s audio), language {config.dataset.language}")

    results = asyncio.run(runner.run_stt(config, clips, _progress()))
    path = runner.write_stt_results(results, config.run.output_dir)

    frame = report.stt_summary_frame(results, config.dataset.language)
    markdown = report.render_stt_markdown(frame)
    _emit(path, "STT", markdown)


@app.command("tts")
def tts_command(
    config_path: Annotated[Path, typer.Argument(help="Benchmark YAML config.")],
    env_file: Annotated[Path, typer.Option(help="Dotenv file.")] = Path(DEFAULT_ENV_FILE),
    save_audio: Annotated[bool, typer.Option(help="Write synthesized WAV files next to results.")] = True,
) -> None:
    """Run the text-to-speech benchmark."""
    load_env_file(env_file)
    config = _load_config(config_path)
    if not config.tts:
        console.print("[red]config error:[/red] no tts providers configured")
        raise typer.Exit(code=2)

    try:
        validate_roundtrip_judges(config)
    except ConfigError as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    try:
        prompts = load_prompts(config.dataset)
    except DatasetError as exc:
        console.print(f"[red]dataset error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(f"Loaded {len(prompts)} prompts, language {config.dataset.language}")

    async def execute() -> list[TtsResult]:
        results = await runner.run_tts(config, prompts, _progress())
        if config.roundtrip_stt:
            judges = ", ".join(judge.name for judge in config.roundtrip_stt)
            console.print(f"Scoring intelligibility via [cyan]{judges}[/cyan]")
            await runner.score_roundtrip(config, results, {p.prompt_id: p for p in prompts})
        return results

    results = asyncio.run(execute())
    path = runner.write_tts_results(results, config.run.output_dir, save_audio=save_audio)

    frame = report.tts_summary_frame(results, config.dataset.language)
    markdown = report.render_tts_markdown(frame)
    _emit(path, "TTS", markdown)

    delta = report.tts_mode_delta_frame(results, config.dataset.language)
    if not delta.is_empty():
        _emit(path, "TTS-delta", report.render_tts_mode_delta_markdown(delta))


def _results_kind(path: Path) -> str:
    """Detect whether a results file holds STT or TTS runs.

    The two schemas share a file naming convention but not a key set, so the
    first record decides: TTS records carry a ``prompt_id``, STT records a
    ``clip_id``. An empty file defaults to STT and fails later with the
    ordinary "no results" path.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = orjson.loads(line)
        return "tts" if "prompt_id" in record else "stt"
    return "stt"


def _fold_lanes[R: SttResult | TtsResult](by_lane: dict[tuple[str, str], list[R]], loaded: list[R]) -> int:
    """Fold one results file into the per-lane map, later files winning.

    Returns:
        The number of runs folded in.
    """
    lanes: dict[tuple[str, str], list[R]] = {}
    for item in loaded:
        lanes.setdefault((item.provider, str(item.mode)), []).append(item)
    for lane, runs in lanes.items():
        earlier = by_lane.get(lane)
        if earlier is not None:
            console.print(
                f"[yellow]superseded[/yellow] {lane[0]} {lane[1]} — {len(earlier)} earlier runs replaced by {len(runs)}"
            )
        by_lane[lane] = runs
    return len(loaded)


@app.command("report")
def report_command(
    results: Annotated[
        list[Path],
        typer.Argument(help="One or more stt-results.jsonl / tts-results.jsonl."),
    ],
    language: Annotated[str, typer.Option(help="BCP-47 tag driving normalization and metric choice.")] = "en-US",
    plots: Annotated[bool, typer.Option(help="Render Pareto and latency charts as PNGs.")] = True,
    latency_metric: Annotated[str, typer.Option(help="Latency axis for charts: finalize or ttft.")] = "finalize",
) -> None:
    """Re-render a report from saved results, merging several runs if given.

    The API calls are the expensive part of a benchmark. This re-scores what
    is already on disk, so a normalization change or a provider added in a
    later run costs nothing to fold in. STT and TTS files may be mixed; each
    kind merges and reports separately, and legacy single-judge TTS files
    fold in as one-judge lanes.

    When the same provider and mode appears in more than one file, the later
    file wins. That is what makes "re-run the one provider that failed, then
    merge" work: without it the superseded run would be averaged back in and
    the fix would look half-effective.
    """
    stt_by_lane: dict[tuple[str, str], list[SttResult]] = {}
    tts_by_lane: dict[tuple[str, str], list[TtsResult]] = {}
    for path in results:
        try:
            kind = _results_kind(path)
            if kind == "tts":
                count = _fold_lanes(tts_by_lane, runner.read_tts_results(path))
            else:
                count = _fold_lanes(stt_by_lane, runner.read_stt_results(path))
        except (ValueError, OSError) as exc:
            console.print(f"[red]results error:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        console.print(f"[dim]loaded[/dim] {count:4d} {kind} runs from {path}")

    merged_stt = [item for runs in stt_by_lane.values() for item in runs]
    merged_tts = [item for runs in tts_by_lane.values() for item in runs]
    if not merged_stt and not merged_tts:
        console.print("[red]no results to report[/red]")
        raise typer.Exit(code=2)

    if merged_stt:
        frame = report.stt_summary_frame(merged_stt, language)
        markdown = report.render_stt_markdown(frame)
        _emit(results[0], "STT", markdown)

        from . import snr

        snr_rows = snr.summarize_snr(merged_stt, language)
        if snr_rows:
            snr_markdown = snr.render_snr_markdown(snr_rows)
            console.print()
            console.print(snr_markdown)
            snr_path = results[0].parent / "snr-report.md"
            snr_path.write_text(f"# SNR robustness\n\n{snr_markdown}\n", encoding="utf-8")
            console.print(f"[dim]snr report:[/dim]  {snr_path}")

        from . import endpointing

        endpoint_summaries = endpointing.summarize_endpointing(merged_stt, language)
        if any(s.hold_pauses or s.eou_source for s in endpoint_summaries):
            _emit(
                results[0],
                "Endpointing",
                endpointing.render_endpointing_markdown(endpoint_summaries),
            )

        if plots:
            from . import metrics, plot

            charts = plot.render_all(frame, results[0].parent / "charts", metric=latency_metric)
            hallucination = plot.plot_hallucination(
                list(metrics.summarize_hallucination(merged_stt, language)),
                results[0].parent / "charts" / "hallucination.png",
            )
            if hallucination is not None:
                charts.append(hallucination)
            for chart in charts:
                console.print(f"[dim]chart:[/dim]       {chart}")
            if not charts:
                console.print(
                    "[yellow]no charts rendered[/yellow] — charts need at least "
                    "two streaming providers with latency and accuracy"
                )

    if merged_tts:
        tts_frame = report.tts_summary_frame(merged_tts, language)
        tts_markdown = report.render_tts_markdown(tts_frame)
        _emit(results[0], "TTS", tts_markdown)

        delta = report.tts_mode_delta_frame(merged_tts, language)
        if not delta.is_empty():
            _emit(results[0], "TTS-delta", report.render_tts_mode_delta_markdown(delta))

        if plots:
            from . import plot

            tts_chart = plot.plot_tts_latency(tts_frame, results[0].parent / "charts" / "tts_latency.png")
            if tts_chart is not None:
                console.print(f"[dim]chart:[/dim]       {tts_chart}")


def _emit(results_path: Path, title: str, markdown: str) -> None:
    """Print a report and write it next to the raw results."""
    console.print()
    console.print(markdown)
    console.print()
    console.print(report.LEGEND)

    report_path = results_path.parent / f"{title.lower()}-report.md"
    report_path.write_text(f"# {title} benchmark\n\n{markdown}\n\n{report.LEGEND}\n", encoding="utf-8")
    console.print()
    console.print(f"[dim]raw results:[/dim] {results_path}")
    console.print(f"[dim]report:[/dim]      {report_path}")


@app.command("guardrail")
def guardrail_command(
    source: Annotated[
        Path,
        typer.Argument(help="Directory of saved TTS audio, or a tts-results.jsonl."),
    ],
    baseline: Annotated[
        Path,
        typer.Option(help="JSON file storing each provider's baseline mean MOS."),
    ] = Path("mos-baseline.json"),
    update_baseline: Annotated[
        bool,
        typer.Option(
            "--update-baseline",
            help="Overwrite the baseline with this run's means.",
        ),
    ] = False,
) -> None:
    """Score saved TTS audio with Distill-MOS as a regression tripwire.

    This is a regression guardrail, NOT a quality ranking across providers:
    single-utterance MOS predictors collapse out-of-domain, so this only ever
    compares a provider's audio against its own recorded baseline, alerting
    when the mean drops by more than the guardrail's threshold.

    Requires the optional dependency group: uv sync --extra guardrail-mos.
    """
    from . import tts_quality

    if not source.exists():
        console.print(f"[red]guardrail error:[/red] source not found: {source}")
        raise typer.Exit(code=2)

    try:
        summaries = tts_quality.run_guardrail(source, baseline_path=baseline, update_baseline=update_baseline)
    except RuntimeError as exc:
        console.print(f"[red]guardrail error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if not summaries:
        console.print("[yellow]no scoreable audio found[/yellow]")
        raise typer.Exit(code=0)

    table = Table(
        title="Distill-MOS regression guardrail — tripwire only, not a ranking",
        show_lines=False,
    )
    table.add_column("Provider")
    table.add_column("Clips", justify="right")
    table.add_column("Mean MOS", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status")

    for summary in summaries:
        delta = summary.delta
        if summary.alert:
            status = "[red]ALERT[/red]"
        elif delta is None:
            status = "[dim]no baseline[/dim]"
        else:
            status = "[green]ok[/green]"
        table.add_row(
            summary.provider,
            str(summary.clips),
            f"{summary.mean_mos:.2f}",
            f"{summary.baseline_mos:.2f}" if summary.baseline_mos is not None else "—",
            f"{delta:+.2f}" if delta is not None else "—",
            status,
        )

    console.print(table)
    if update_baseline:
        console.print(f"[dim]baseline updated:[/dim] {baseline}")

    alerted = [s for s in summaries if s.alert]
    if alerted:
        names = ", ".join(s.provider for s in alerted)
        console.print(
            f"[red]{len(alerted)} provider(s) dropped more than "
            f"{tts_quality.REGRESSION_THRESHOLD} MOS vs baseline:[/red] {names}"
        )
        raise typer.Exit(code=1)


@app.command("tts-arena")
def tts_arena_command(
    config_path: Annotated[Path, typer.Argument(help="Benchmark YAML config listing the TTS candidates.")],
    prompt_files: Annotated[
        list[str],
        typer.Argument(help="Prompt files, each optionally suffixed :count for a seeded sample."),
    ],
    env_file: Annotated[Path, typer.Option(help="Dotenv file.")] = Path(DEFAULT_ENV_FILE),
    panel: Annotated[
        Path | None,
        typer.Option(help="Human-panel votes JSONL for gate criterion (i)."),
    ] = None,
    judge_model: Annotated[
        str, typer.Option(help="Audio-in judge model; pinned, part of cache keys.")
    ] = "gemini-2.5-flash",
    seed: Annotated[int, typer.Option(help="Seed for prompt sampling and bootstrap CIs.")] = 20260806,
    max_calls: Annotated[int, typer.Option(help="Refuse to schedule more judge calls than this.")] = 2000,
    concurrency: Annotated[int, typer.Option(help="Concurrent judge calls in flight.")] = 8,
    reuse_audio: Annotated[
        Path | None,
        typer.Option(
            help="Judge saved WAVs from this directory instead of synthesizing "
            "(useful when a candidate's lane fails for a language)."
        ),
    ] = None,
) -> None:
    """Rank TTS candidates with the pairwise audio-LLM arena (experimental).

    AudioJudge protocol (arXiv:2507.12705): every unordered pair of
    candidates is judged per prompt in both presentation orders by three
    pinned aspect judges over concatenated pair audio, aggregated to a
    system-level Bradley-Terry ranking with bootstrap CIs. The lane renders
    "experimental - not ranked" until its three-criterion validity gate
    passes; judge calls are cached, so re-runs only pay for what is new.

    Synthesis is skipped when a timestamped run directory under the config's
    output directory already holds every candidate WAV (or --reuse-audio
    names one explicitly); otherwise a full batch synthesis pass runs (judge
    caching makes the re-judging free).
    """
    from .judge import tts_arena

    load_env_file(env_file)
    config = _load_config(config_path)
    systems = sorted({entry.name for entry in config.tts})
    if len(systems) < 2:
        console.print("[red]config error:[/red] the arena needs >= 2 tts candidates")
        raise typer.Exit(code=2)
    for entry in config.tts:
        if "batch" not in entry.modes:
            console.print(
                f"[red]config error:[/red] arena candidate {entry.name} must "
                "include batch mode — the arena judges saved batch audio"
            )
            raise typer.Exit(code=2)

    try:
        prompts = tts_arena.load_arena_prompts(prompt_files, language=config.dataset.language, seed=seed)
        votes = tts_arena.load_panel(panel) if panel is not None else None
    except tts_arena.ArenaError as exc:
        console.print(f"[red]arena error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    output_dir = Path(config.run.output_dir)
    expected = [f"{system}-batch-{prompt.prompt_id}.wav" for system in systems for prompt in prompts]

    def complete_audio_dir() -> Path | None:
        """Newest timestamped run directory already holding every WAV."""
        candidates = sorted(output_dir.glob("*/audio"), reverse=True)
        for candidate in candidates:
            if all((candidate / name).is_file() for name in expected):
                return candidate
        return None

    audio_dir = reuse_audio if reuse_audio is not None else complete_audio_dir()
    if audio_dir is not None:
        if not audio_dir.is_dir():
            console.print(f"[red]arena error:[/red] audio dir not found: {audio_dir}")
            raise typer.Exit(code=2)
        console.print(f"[dim]reusing synthesized audio:[/dim] {audio_dir}")
    else:
        console.print(f"Synthesizing {len(prompts)} prompts x {len(systems)} candidates (batch)")
        results = asyncio.run(runner.run_tts(config, prompts, _progress()))
        results_path = runner.write_tts_results(results, config.run.output_dir, save_audio=True)
        audio_dir = results_path.parent / "audio"
        failed = [r for r in results if not r.ok]
        if failed:
            console.print(
                f"[yellow]{len(failed)} synthesis run(s) failed; their pairs "
                "will be skipped and reported as missing audio[/yellow]"
            )

    n_pairs = len(systems) * (len(systems) - 1) // 2
    total_calls = n_pairs * len(prompts) * len(tts_arena.ASPECTS) * 2
    if total_calls > max_calls:
        console.print(
            f"[red]arena error:[/red] {total_calls} judge calls scheduled "
            f"exceeds --max-calls {max_calls}; shrink the prompt mix or "
            "raise the cap deliberately"
        )
        raise typer.Exit(code=2)
    console.print(
        f"Judging {total_calls} calls ({n_pairs} pairs x {len(prompts)} prompts "
        f"x {len(tts_arena.ASPECTS)} aspects x 2 orders) via "
        f"[cyan]{judge_model}[/cyan]"
    )

    def on_progress(done: int, total: int) -> None:
        if done % 100 == 0 or done == total:
            console.print(f"[dim]judged[/dim] {done}/{total}")

    run = asyncio.run(
        tts_arena.run_arena(
            audio_dir=audio_dir,
            systems=systems,
            prompts=prompts,
            cache_path=output_dir / "judge-cache.jsonl",
            model=judge_model,
            concurrency=concurrency,
            on_progress=on_progress,
        )
    )

    scores = tts_arena.bt_table(run.verdicts, systems, seed=seed)
    flips = tts_arena.order_flip_stats(run.verdicts)
    gate = tts_arena.evaluate_gate(scores, flips, votes)
    notes = []
    if config.dataset.language.split("-")[0] == "ja":
        notes.append(
            "ja interview prompts are PENDING "
            "(data/prompts-ja/interview.PENDING.md); the ja mix covers "
            "general + entities only"
        )

    # Keep each judge model's persisted evidence beside the audio run without
    # letting the second family overwrite the first; the cache spans run dirs.
    judge_output_dir = audio_dir.parent / "judge-results" / judge_model.replace("/", "_")
    results_path, summary_path, report_path = tts_arena.write_arena_outputs(
        judge_output_dir, run, scores, flips, gate, notes=notes
    )
    console.print()
    console.print(tts_arena.render_arena_markdown(run, scores, gate, notes=notes))
    console.print()
    console.print(
        f"Spend: {run.live_calls} live calls ({run.cached_calls} cached, "
        f"{run.error_calls} errored), {run.live_prompt_tokens} prompt + "
        f"{run.live_output_tokens} output tokens, est. ${run.est_usd:.2f}"
    )
    console.print(f"[dim]raw verdicts:[/dim] {results_path}")
    console.print(f"[dim]summary:[/dim]      {summary_path}")
    console.print(f"[dim]report:[/dim]       {report_path}")


@app.command("arena-gate")
def arena_gate_command(
    summary_a: Annotated[Path, typer.Argument(help="First completed arena-summary.json.")],
    summary_b: Annotated[
        Path,
        typer.Argument(help="Second completed arena-summary.json from another judge family."),
    ],
) -> None:
    """Compare completed Gemini/OpenAI arena runs for gate criterion (ii).

    Bradley-Terry vectors come from the two summaries. The diagnostic
    per-pair verdict agreement comes from each summary's sibling
    ``arena-results.jsonl``. The Markdown result is printed and written as
    ``arena-cross-family-gate.md`` beside the first summary.
    """
    from .judge import tts_arena

    try:
        result = tts_arena.evaluate_cross_family_gate(summary_a, summary_b)
        report_path = tts_arena.write_cross_family_gate(result)
    except tts_arena.ArenaError as exc:
        console.print(f"[red]arena gate error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(tts_arena.render_cross_family_gate(result))
    console.print()
    console.print(f"[dim]cross-family gate report:[/dim] {report_path}")
    if result.criterion.status != "pass":
        raise typer.Exit(code=1)


@app.command("sim")
def sim_command(
    config_path: Annotated[Path, typer.Argument(help="Simulated-interview YAML config.")],
    env_file: Annotated[Path, typer.Option(help="Dotenv file.")] = Path(DEFAULT_ENV_FILE),
    estimate_only: Annotated[
        bool,
        typer.Option("--estimate-only", help="Print the expected spend and exit."),
    ] = False,
) -> None:
    """Run the simulated-interview E2E lane (experimental, gated).

    Interviewer LLM -> persona LLM -> pinned Kokoro voice -> pinned tel8k
    degradation -> candidate streaming STT -> deterministic field-extraction
    scoring. The expected spend is computed and printed before anything paid
    executes, and the run aborts above the config's hard cap. The vendor
    ranking is compared against the pre-registered real-corpus composite;
    below the gate the lane renders "experimental - not ranked".

    Requires the optional dependency group: uv sync --extra sim-kokoro.
    """
    import yaml

    from .sim import interview as sim_interview

    load_env_file(env_file)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        bench = BenchmarkConfig.from_dict(raw)
        sim = sim_interview.SimConfig.from_mapping(raw.get("sim") or {})
        scenarios = sim_interview.load_scenarios(sim.scenarios_path)
    except (OSError, ConfigError) as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    if not bench.stt:
        console.print("[red]config error:[/red] no stt providers configured")
        raise typer.Exit(code=2)

    estimate = sim_interview.estimate_spend(
        bench,
        scenarios,
        sim.personas_per_scenario,
        est_answer_s=sim.est_answer_s,
    )
    console.print(f"[bold]Expected spend (hard cap ${sim.hard_cap_usd:.2f})[/bold]")
    console.print(estimate.render())
    if estimate_only:
        return
    try:
        sim_interview.ensure_within_cap(estimate, sim.hard_cap_usd)
    except ConfigError as exc:
        console.print(f"[red]spend cap:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    try:
        llm = sim_interview.gemini_llm(sim.model)
        console.print("[dim]loading Kokoro (pinned revision)…[/dim]")
        tts_fn = sim_interview.KokoroSynth(sim.voices)
    except (RuntimeError, ConfigError) as exc:
        console.print(f"[red]sim error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    try:
        outcome = asyncio.run(sim_interview.run_sim(bench, sim, llm=llm, tts=tts_fn, progress=_progress()))
    except ConfigError as exc:
        console.print(f"[red]sim error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    results_path = runner.write_stt_results(outcome.results, bench.run.output_dir)

    gate = None
    gate_raw = raw.get("gate") or {}
    canonical = [Path(p) for p in gate_raw.get("canonical") or []]
    if canonical:
        try:
            merged = sim_interview.load_canonical(canonical)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]gate error:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        composite = sim_interview.composite_ranking(merged, [entry.name for entry in bench.stt])
        gate = sim_interview.evaluate_gate(
            outcome.vendor_scores,
            composite,
            threshold=float(gate_raw.get("rho_threshold", 0.8)),
        )

    sim_path, gate_path, report_path = sim_interview.write_sim_outputs(outcome, gate, results_path)
    console.print()
    console.print(sim_interview.render_sim_markdown(outcome, gate))
    console.print(
        "Actual spend: "
        + ", ".join(f"{name} ${usd:.2f}" for name, usd in sorted(outcome.spend.items()) if name != "total")
        + f" — total ${outcome.spend['total']:.2f} "
        f"({outcome.usage.calls} LLM calls, seed {outcome.seed})"
    )
    if gate is not None and not gate.passed:
        console.print(
            "[yellow]gate FAILED — write the divergence analysis "
            "(sim-divergence.md) before any kill/demote verdict[/yellow]"
        )
    console.print(f"[dim]raw results:[/dim] {results_path}")
    console.print(f"[dim]sim results:[/dim] {sim_path}")
    console.print(f"[dim]gate:[/dim]        {gate_path}")
    console.print(f"[dim]report:[/dim]      {report_path}")


@app.command("judge-semantic")
def judge_semantic_command(
    results: Annotated[
        list[Path],
        typer.Argument(help="One or more stt-results.jsonl files to merge and judge."),
    ],
    language: Annotated[str, typer.Option(help="Fallback BCP-47 tag for results that recorded none.")] = "en-US",
    anchor: Annotated[
        Path | None,
        typer.Option(help="Human anchor CSV (clip_id, provider, human_label)."),
    ] = None,
    cache_file: Annotated[
        Path | None,
        typer.Option(help="Vote cache JSONL; defaults next to the first results file."),
    ] = None,
    max_calls: Annotated[int, typer.Option(help="Hard cap on live judge calls for this run.")] = 4800,
    semascore: Annotated[bool, typer.Option(help="Compute the deterministic SeMaScore fallback.")] = True,
    env_file: Annotated[Path, typer.Option(help="Dotenv file.")] = Path(DEFAULT_ENV_FILE),
) -> None:
    """Judge saved STT transcripts for semantic fidelity (experimental lane).

    Reads saved results only — no audio is ever re-submitted — and caches
    every vote by content, so re-running an already-judged merge bills
    nothing. Files merge with the same supersede rule as ``report``: when a
    provider and mode appears in several files, the later file wins. Until a
    language's human anchor exists and its Cohen's-kappa gate passes, every
    number renders as "experimental — not ranked" and must not feed a
    vendor recommendation.
    """
    from collections.abc import Callable

    from .judge import semantic

    load_env_file(env_file)

    by_lane: dict[tuple[str, str], list[SttResult]] = {}
    for path in results:
        try:
            loaded = runner.read_stt_results(path)
        except (ValueError, OSError) as exc:
            console.print(f"[red]results error:[/red] {exc}")
            raise typer.Exit(code=2) from exc
        lanes: dict[tuple[str, str], list[SttResult]] = {}
        for item in loaded:
            lanes.setdefault((item.provider, str(item.mode)), []).append(item)
        for lane, runs in lanes.items():
            if lane in by_lane:
                console.print(f"[yellow]superseded[/yellow] {lane[0]} {lane[1]}")
            by_lane[lane] = runs
        console.print(f"[dim]loaded[/dim] {len(loaded):4d} stt runs from {path}")

    merged = [item for runs in by_lane.values() for item in runs]
    items = semantic.judgeable_items(merged, language)
    if not items:
        console.print("[red]no judgeable results[/red] (need ok runs with references)")
        raise typer.Exit(code=2)

    anchor_map = None
    if anchor is not None:
        try:
            anchor_map = semantic.load_anchor(anchor)
        except (ValueError, OSError) as exc:
            console.print(f"[red]anchor error:[/red] {exc}")
            raise typer.Exit(code=2) from exc

    semascore_fn: Callable[[semantic.JudgeItem], float | None] | None = None
    if semascore:
        try:
            embed = semantic.roberta_embedder()
        except RuntimeError as exc:
            # Optional feature: the judge lane still works without the
            # fallback metric, so a missing extra degrades instead of failing.
            console.print(f"[yellow]semascore skipped:[/yellow] {exc}")
        else:

            def semascore_fn(item: semantic.JudgeItem) -> float | None:
                return semantic.semascore(item.reference, item.hypothesis, item.language, embed)

    console.print(
        f"Judging {len(items)} items with [cyan]{semantic.JUDGE_MODEL}[/cyan] "
        f"({len(items) * semantic.VOTES_PER_ITEM} votes max, cap {max_calls})"
    )
    cache_path = cache_file if cache_file is not None else results[0].parent / "semantic-cache.jsonl"
    try:
        judgements, stats = semantic.run_judge(
            items,
            semantic.GeminiJudge(),
            semantic.VoteCache(cache_path),
            max_calls=max_calls,
            semascore_fn=semascore_fn,
        )
    except (semantic.JudgeBudgetError, ConfigError, RuntimeError) as exc:
        console.print(f"[red]judge error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    gates = semantic.evaluate_gates(judgements, anchor_map)
    out_path = semantic.write_semantic_results(judgements, gates, stats, results[0].parent / "semantic-results.jsonl")
    markdown = semantic.render_semantic_markdown(semantic.summarize_semantic(judgements), gates)
    console.print()
    console.print(markdown)
    report_path = results[0].parent / "semantic-report.md"
    report_path.write_text(f"# Semantic-fidelity judge (experimental)\n\n{markdown}\n", encoding="utf-8")
    console.print()
    console.print(
        f"[dim]judge spend:[/dim] {stats.live_calls} live calls "
        f"({stats.cached_votes} cached), {stats.input_tokens} in / "
        f"{stats.output_tokens} out tokens, est ${stats.estimated_usd:.4f}"
    )
    console.print(f"[dim]raw results:[/dim] {out_path}")
    console.print(f"[dim]report:[/dim]      {report_path}")


@app.command("agree")
def agree_command(
    runs: Annotated[list[Path], typer.Argument(help="Two or more completed STT run directories.")],
    output_dir: Annotated[Path, typer.Option(help="Directory receiving agreement.md and agreement.json.")] = Path(
        "results/agreement"
    ),
) -> None:
    """Compare completed STT runs on pairwise inter-model agreement."""
    loaded = agreement.load_agreement_runs(runs)
    if len(loaded) < 2:
        console.print("[red]agreement error:[/red] need results from at least two lanes")
        raise typer.Exit(code=2)

    agreement_report = agreement.compute_agreement(loaded)
    markdown_path, json_path = agreement.write_agreement_report(agreement_report, output_dir)
    console.print(agreement.render_agreement_markdown(agreement_report))
    console.print(f"[green]wrote[/green] {markdown_path} and {json_path}")


@app.command("realdata")
def realdata_command(
    video_prefix: Annotated[str, typer.Argument(help="GCS prefix holding LiveKit video egress recordings.")],
    dest: Annotated[Path, typer.Option(help="Local real-data root; must stay gitignored.")] = Path("data/realdata"),
    sessions: Annotated[int, typer.Option(help="Sessions to download before clip selection.")] = 40,
    pilot: Annotated[int, typer.Option(help="Pilot manifest size.")] = 30,
    label: Annotated[int, typer.Option(help="Human-labeling sheet size.")] = 50,
    audio_prefix: Annotated[str, typer.Option(help="Optional GCS prefix with MP3-only recordings to include.")] = "",
    seed: Annotated[int, typer.Option(help="Deterministic sampling seed.")] = 20260808,
    dry_run: Annotated[bool, typer.Option(help="Reuse files already under dest; never touch the network.")] = False,
) -> None:
    """Stage real recordings into a reference-free local benchmark corpus."""
    video_dir = dest / "video"
    clips_dir = dest / "clips"
    clips_path = dest / "clips.jsonl"
    join_path = dest / "join.jsonl"

    manifests = realdata.list_manifests(video_dir, video_prefix, dry_run=dry_run)
    realdata.build_join(manifests, output_path=join_path)
    objects = realdata.dedupe_sessions(realdata.list_video_objects(video_dir, video_prefix, dry_run=dry_run))
    all_sessions = sorted({item.session_id for item in objects})
    chosen = sorted(random.Random(seed).sample(all_sessions, k=min(sessions, len(all_sessions))))
    console.print(f"{len(all_sessions)} sessions listed, staging {len(chosen)}")

    videos = realdata.ingest_video(video_dir, video_prefix, chosen, dry_run=dry_run)
    session_by_name = {item.filename: item.session_id for item in objects}

    clips_path.unlink(missing_ok=True)
    for video_path in videos:
        video_session = session_by_name.get(video_path.name)
        if video_session is None:
            continue
        cut = realdata.cut_video_clips(video_path, video_session, clips_dir, metadata_path=clips_path)
        console.print(f"  {video_session}: {len(cut)} clips (video)")

    if audio_prefix:
        audio_dir = dest / "audio"
        realdata.ingest(audio_dir, audio_prefix, dry_run=dry_run)
        merged_by_session: dict[str, Path] = {}
        track_by_session: dict[str, Path] = {}
        for mp3_path in sorted(audio_dir.rglob("*.mp3")):
            mp3_session = mp3_path.name.split("_", 1)[0]
            if mp3_path.stem.endswith("_merged"):
                merged_by_session[mp3_session] = mp3_path
            else:
                track_by_session[mp3_session] = mp3_path
        for mp3_session, merged_path in sorted(merged_by_session.items()):
            triage = realdata.triage_session(track_by_session.get(mp3_session), merged_path)
            # An agent-classified track is synthesized speech, so the merged
            # mix is the only file that still contains the user's voice.
            if triage.source == "user":
                cut = realdata.cut_clips(
                    track_by_session[mp3_session], mp3_session, clips_dir, source="user", metadata_path=clips_path
                )
            else:
                cut = realdata.cut_clips(merged_path, mp3_session, clips_dir, source="merged", metadata_path=clips_path)
            console.print(f"  {mp3_session}: {len(cut)} clips ({triage.source})")

    rows = [orjson.loads(line) for line in clips_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tagged = 0
    for row in rows:
        if row.get("language") not in {"en", "ja", "other"}:
            row["language"] = realdata.identify_language(Path(str(row["clip_path"])))
            tagged += 1
    clips_path.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    console.print(f"language-tagged {tagged} of {len(rows)} clips")

    pilot_path = dest / f"pilot-{pilot}.jsonl"
    selected = realdata.select_pilot(
        pilot, seed=seed, clips_path=clips_path, join_path=join_path, output_path=pilot_path
    )
    sheet = realdata.select_label_candidates(label, seed=seed, clips_path=clips_path, join_path=join_path)
    console.print(f"[green]pilot[/green] {len(selected)} clips -> {pilot_path}")
    console.print(f"[green]label sheet[/green] {len(sheet)} rows -> data/anchors/realdata/")


def main() -> None:
    """Entry point for the ``audio-harness`` console script."""
    app()


if __name__ == "__main__":
    main()
