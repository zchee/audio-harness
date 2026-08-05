"""Command-line interface."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import doctor as doctor_module
from . import report, runner, stt, tts
from .config import PRICING_CHECKED, BenchmarkConfig, ConfigError
from .dataset import DatasetError, load_clips, load_prompts

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


def _progress() -> runner.Progress:
    """Build a progress sink that prints one line per completed lane."""
    totals: dict[tuple[str, str], int] = {}
    done: dict[tuple[str, str], int] = {}
    failed: dict[tuple[str, str], int] = {}

    def on_start(provider: str, mode: str, total: int) -> None:
        totals[(provider, mode)] = total
        done[(provider, mode)] = 0
        failed[(provider, mode)] = 0
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
    env_file: Annotated[
        Path, typer.Option(help="Dotenv file to load before checking.")
    ] = Path(DEFAULT_ENV_FILE),
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
    env_file: Annotated[Path, typer.Option(help="Dotenv file.")] = Path(
        DEFAULT_ENV_FILE
    ),
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
    console.print(
        f"Loaded {len(clips)} clips ({total_audio:.1f}s audio), "
        f"language {config.dataset.language}"
    )

    results = asyncio.run(runner.run_stt(config, clips, _progress()))
    path = runner.write_stt_results(results, config.run.output_dir)

    frame = report.stt_summary_frame(results, config.dataset.language)
    markdown = report.render_stt_markdown(frame)
    _emit(path, "STT", markdown)


@app.command("tts")
def tts_command(
    config_path: Annotated[Path, typer.Argument(help="Benchmark YAML config.")],
    env_file: Annotated[Path, typer.Option(help="Dotenv file.")] = Path(
        DEFAULT_ENV_FILE
    ),
    save_audio: Annotated[
        bool, typer.Option(help="Write synthesized WAV files next to results.")
    ] = True,
) -> None:
    """Run the text-to-speech benchmark."""
    load_env_file(env_file)
    config = _load_config(config_path)
    if not config.tts:
        console.print("[red]config error:[/red] no tts providers configured")
        raise typer.Exit(code=2)

    try:
        prompts = load_prompts(config.dataset)
    except DatasetError as exc:
        console.print(f"[red]dataset error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(f"Loaded {len(prompts)} prompts, language {config.dataset.language}")

    async def execute() -> list[object]:
        results = await runner.run_tts(config, prompts, _progress())
        if config.roundtrip_stt is not None:
            console.print(
                f"Scoring intelligibility via [cyan]{config.roundtrip_stt.name}[/cyan]"
            )
            await runner.score_roundtrip(
                config, results, {p.prompt_id: p for p in prompts}
            )
        return results

    results = asyncio.run(execute())
    path = runner.write_tts_results(
        results, config.run.output_dir, save_audio=save_audio
    )

    frame = report.tts_summary_frame(results, config.dataset.language)
    markdown = report.render_tts_markdown(frame)
    _emit(path, "TTS", markdown)


@app.command("report")
def report_command(
    results: Annotated[
        list[Path], typer.Argument(help="One or more stt-results.jsonl files.")
    ],
    language: Annotated[
        str, typer.Option(help="BCP-47 tag driving normalization and metric choice.")
    ] = "en-US",
) -> None:
    """Re-render a report from saved results, merging several runs if given.

    The API calls are the expensive part of a benchmark. This re-scores what
    is already on disk, so a normalization change or a provider added in a
    later run costs nothing to fold in.

    When the same provider and mode appears in more than one file, the later
    file wins. That is what makes "re-run the one provider that failed, then
    merge" work: without it the superseded run would be averaged back in and
    the fix would look half-effective.
    """
    by_lane: dict[tuple[str, str], list[object]] = {}
    for path in results:
        try:
            loaded = runner.read_stt_results(path)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[red]results error:[/red] {exc}")
            raise typer.Exit(code=2) from exc

        lanes: dict[tuple[str, str], list[object]] = {}
        for item in loaded:
            lanes.setdefault((item.provider, str(item.mode)), []).append(item)
        for lane, runs in lanes.items():
            if lane in by_lane:
                console.print(
                    f"[yellow]superseded[/yellow] {lane[0]} {lane[1]}"
                    f" — {len(by_lane[lane])} earlier runs replaced by {len(runs)}"
                )
            by_lane[lane] = runs
        console.print(f"[dim]loaded[/dim] {len(loaded):4d} runs from {path}")

    merged = [item for runs in by_lane.values() for item in runs]
    if not merged:
        console.print("[red]no results to report[/red]")
        raise typer.Exit(code=2)

    frame = report.stt_summary_frame(merged, language)
    markdown = report.render_stt_markdown(frame)
    _emit(results[0], "STT", markdown)


def _emit(results_path: Path, title: str, markdown: str) -> None:
    """Print a report and write it next to the raw results."""
    console.print()
    console.print(markdown)
    console.print()
    console.print(report.LEGEND)

    report_path = results_path.parent / f"{title.lower()}-report.md"
    report_path.write_text(
        f"# {title} benchmark\n\n{markdown}\n\n{report.LEGEND}\n", encoding="utf-8"
    )
    console.print()
    console.print(f"[dim]raw results:[/dim] {results_path}")
    console.print(f"[dim]report:[/dim]      {report_path}")


def main() -> None:
    """Entry point for the ``audio-harness`` console script."""
    app()


if __name__ == "__main__":
    main()
