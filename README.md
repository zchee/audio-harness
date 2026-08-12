# audio-harness

Cross-vendor STT/TTS benchmark harness. Measures recognition accuracy,
streaming latency and cost across speech vendors under identical audio,
identical pacing and identical text normalization.

Built for choosing a speech stack for a **realtime voice agent**, so the
headline metric is not word error rate — it is how long the user waits after
they stop talking.

## Architecture

```mermaid
flowchart TB
    subgraph ENTRY["Entry / configuration"]
        CLI["cli.py<br/>stt / tts / sim / agree / report /<br/>realdata / arena-gate / doctor"]
        CFG["config.py<br/>BenchmarkConfig + dated pricing tables"]
        YAML[("configs/*.yaml<br/>experiment definitions<br/>(corpus x lanes x modes)")]
    end

    subgraph DATAIN["Dataset supply"]
        DS["dataset.py<br/>parquet / manifest / prompt loaders"]
        CUR["curated.py + curate.py<br/>YODAS / Granary (IDs only, no audio redistribution)"]
        RDM["realdata_manifest.py<br/>reference-free real-recording loader"]
        RD["realdata.py<br/>GCS ingest, clip cutting, language ID,<br/>PII-trimmed stratified selection"]
        SNR["snr.py / synthetic.py<br/>degradation conditions"]
    end

    subgraph CORE["Execution core"]
        RUN["runner.py<br/>lanes concurrent, clips sequential,<br/>per-vendor session locks, crash-safe persistence"]
        AUD["audio.py<br/>PCM/WAV, realtime pacing, speech-end detection"]
    end

    subgraph ADAPT["Vendor adapters (registry)"]
        STT["stt/ - 19 adapters<br/>direct WS/HTTP wire protocols;<br/>privacy controls built in<br/>(mip_opt_out, delete_after + evidence, region)"]
        TTS["tts/ - 13 adapters"]
        WS["stt/ws.py<br/>shared streaming base"]
    end

    subgraph SCORE["Scoring and reporting"]
        NORM["normalize / entities / autotag<br/>per-language normalization, entity scoring"]
        MET["metrics.py<br/>WER/CER, latency percentiles, churn"]
        AGR["agreement.py<br/>cross-lane agreement (reference-free)"]
        REP["report.py<br/>lane-keyed supersede merge -> canonical;<br/>quantified comparability caveats"]
    end

    subgraph GATED["Gated experimental lanes (never ranked until their validity gate passes)"]
        SIM["sim/interview.py - E3<br/>LLM-generated interviews, task-success,<br/>gate rho >= 0.8; audio persisted locally"]
        ARENA["judge/tts_arena.py - E2<br/>pairwise Bradley-Terry + cross-family judge gate"]
        SEM["judge/semantic.py - E1<br/>semantic-fidelity judge"]
    end

    VENDORS(("cloud vendor APIs"))
    LOCAL(("on-device inference<br/>ANE / MLX / SFSpeech / Kokoro"))
    OUT[("results/<br/>per-run jsonl + reports + audio;<br/>canonical merges")]

    CLI --> CFG --> RUN
    YAML --> CFG
    CLI --> DS --> RUN
    CUR --> DS
    RD --> RDM --> DS
    SNR --> DS
    RUN --> STT & TTS
    STT --> WS
    RUN --> AUD
    STT --> VENDORS
    TTS --> VENDORS
    STT --> LOCAL
    TTS -.-> LOCAL
    RUN --> OUT
    OUT --> REP
    REP --> MET --> NORM
    OUT --> AGR
    SIM --> RUN
    ARENA --> OUT
    SEM --> OUT
```

Principles that cut across every layer: no same-family judging, lane-keyed
supersede (never mixing corpora in one canonical row), quantified
comparability caveats (hosted-proxy, SDK-buffered, local-compute), gated
isolation for judge-based lanes, and pricing treated as dated data that rots.

## Providers

| Kind | Key | Model | Batch | Stream |
| --- | --- | --- | --- | --- |
| STT | `deepgram-nova3` | Nova-3 | yes | yes |
| STT | `elevenlabs-scribe2` | Scribe v2 / v2 realtime | yes | yes |
| STT | `google-chirp3` | Chirp 3 (STT v2, gRPC) | yes | yes |
| STT | `xai-grok-stt` | Grok STT | yes | yes |
| STT | `assemblyai-universal35pro` | Universal-3.5 pro | yes | yes |
| STT | `speechmatics-enhanced` | Enhanced operating point | yes | yes |
| STT | `speechmatics-standard` | Standard operating point | yes | yes |
| STT | `soniox-rt-v5` | stt-rt-v5 | — | yes |
| TTS | `cartesia-sonic3` | Sonic 3.0 | yes | yes |
| TTS | `cartesia-sonic35` | Sonic 3.5 | yes | yes |
| TTS | `deepgram-aura2` | Aura-2 | yes | yes |
| TTS | `gemini-tts` | Gemini TTS | yes | yes |

Adapters talk to raw HTTP and WebSocket endpoints rather than vendor SDKs.
SDKs buffer and retry on their own schedule, which is exactly the behaviour a
latency benchmark must not measure. Chirp 3 is the one exception — it is
gRPC-only — and its numbers carry that caveat.

## Quick start

```bash
uv sync --extra dev
```

```bash
cp .env.example .env
```

Fill in `.env` (see [docs/ACCOUNTS.md](docs/ACCOUNTS.md) for where each key is
issued), then verify every credential without spending anything:

```bash
uv run audio-harness doctor
```

Run the benchmarks:

```bash
uv run audio-harness stt configs/stt-en.yaml
```

```bash
uv run audio-harness tts configs/tts-en.yaml
```

Each run writes `stt-results.jsonl` (including every interim hypothesis), a
markdown report, and — for TTS — the synthesized WAVs, under a timestamped
directory in `results/`.

## Datasets

### Pipecat STT benchmark corpus (default)

1,000 clips of real conversational voice-agent speech — support calls,
scheduling, product questions — 1–16 s, 16 kHz, mean 9.6 s, with human
references. Fetch it once:

```bash
hf download pipecat-ai/stt-benchmark-data --repo-type dataset --local-dir data/hf/stt-benchmark-data
```

```bash
uv run audio-harness stt configs/stt-pipecat.yaml
```

The parquet is read directly — audio is decoded from the embedded bytes rather
than exploded into a second copy of the corpus on disk.

`limit` with `sample_seed` takes a reproducible random subset. Use the seed:
corpora are rarely shuffled on disk, so a bare `limit` biases toward whatever
the first N rows happen to be.

*Corpus caveat:* 51 of the 1,000 clips sit exactly at a 16.0 s ceiling and
appear to be cut there. Every provider sees the same truncation, so the ranking
is unaffected, but the absolute error rate carries a small floor from them.

### Your own audio

Either point `dataset.parquet` at any parquet with embedded audio (column names
are configurable), or use a JSONL manifest, one clip per line:

```json
{"id": "utt-001", "audio": "clips/utt-001.wav", "text": "the reference transcript", "language": "en-US"}
```

`audio` may be absolute or relative to the manifest. Any format libsndfile
reads works; everything is decoded to mono 16 kHz PCM before it reaches a
vendor, so no provider gets a codec advantage.

TTS input is a plain text file, one prompt per line — see
[data/prompts-en.txt](data/prompts-en.txt) and
[data/prompts-ja.txt](data/prompts-ja.txt).

**Use real recorded speech.** Synthesized audio is unrealistically clean and
every provider scores near 0% on it. For Japanese, Common Voice ja or
ReazonSpeech plus a sample of your own production audio is the combination that
actually predicts behaviour.

## What is measured

| Metric | Meaning |
| --- | --- |
| Error rate | WER, or CER for Japanese and other languages written without spaces |
| TTFT | First interim hypothesis, from the first audio byte |
| **Finalize** | **Last audio byte → final transcript** |
| RTF | Processing seconds per audio second; above 1.0x falls behind live audio |
| Churn | Share of interim hypotheses that rewrote already-displayed text |
| TTFB | Time to first audio byte (TTS) |
| Round-trip err | Synthesized audio re-transcribed and scored against the prompt |

**Finalize is the number to optimize.** TTFT tells you when a UI can show that
it is listening; finalize sets how long the user waits before the agent can
respond. A provider can win on WER and still feel sluggish.

Churn matters for barge-in: a provider that retracts text it already emitted
can cause turn-taking logic to act on a phrase that later disappears.

## Methodology

These decisions are what make the numbers comparable, and each one is a way
the benchmark could otherwise have lied:

- **Audio is streamed at 1x wall-clock.** Feeding a socket as fast as it
  accepts measures throughput and reports fictitiously low latency. Setting
  `realtime: false` invalidates every latency figure and exists only for smoke
  tests.
- **Clips run sequentially within a provider.** Issuing them concurrently makes
  each request queue behind the others, inflating the latency being measured.
  Providers run in parallel because they are independent services.
- **Error rates aggregate at corpus level** — total edits over total reference
  length — never as a mean of per-utterance rates, which lets a three-word clip
  outweigh a thirty-second one.
- **Latency is reported as p50 and p95**, not a mean. Speech APIs have long
  right tails, and a mean blends the tail into the typical case.
- **Numbers are normalized on both sides.** Corpora write "four hundred and
  twenty dollars"; recognizers return "$420". Without folding both to a common
  form the accuracy column measures inverse text normalization rather than
  recognition — in early runs of this harness that alone produced a flat 36%
  WER across every vendor.
- **Frame size is clamped per vendor and reported.** AssemblyAI rejects frames
  below 50 ms, so it runs at 50 ms while others use the 20 ms telephony
  default. The `Frame` column makes that handicap visible instead of silently
  charging it to the provider's latency.
- **Lanes sharing one vendor account are serialized.** Speechmatics Standard
  and Enhanced are one account, and free plans cap concurrent sessions; without
  this the vendor rejects half the runs and it reads as a provider failure
  rather than a scheduling mistake. Raise `vendor_concurrency` if your plan
  allows it.
- **Round-trip TTS scoring needs the recognizer's formatting off.** Otherwise
  every engine inherits the same rewrite and the metric stops discriminating.
  It is an intelligibility proxy, not naturalness — only row-to-row
  comparisons mean anything, and it is no substitute for listening.

## Cost

Every run reports estimated spend from the rates in
[config.py](src/audio_harness/config.py), which carry a `PRICING_CHECKED` date.
Treat a stale date as a signal to re-verify before quoting a figure. A
30-minute corpus across all STT providers in both modes costs a few dollars,
and most of it lands inside free tiers.

## Development

```bash
uv run pytest tests/ -q
```

```bash
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

The streaming driver is tested against a real local WebSocket server rather
than a mock: whether send and receive genuinely overlap, and whether
finalization latency excludes the clip's own duration, are properties that only
exist in a real socket. A mock would confirm whatever timing the driver
claimed.
