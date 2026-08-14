# MaaS account setup checklist

日本語版: [ACCOUNTS.ja.md](ACCOUNTS.ja.md)

Accounts and API keys needed for the benchmarked models. Sign-ups must be
done by a human (an agent cannot do them for you). Once the keys are in
`.env`, run `uv run audio-harness doctor` to verify every credential with a
cheap authenticated request.

---

## Already held

| Provider | Used for | Required environment variables |
| --- | --- | --- |
| Google Cloud | STT: Chirp 3 | `GOOGLE_APPLICATION_CREDENTIALS` (SA JSON) or ADC, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` |
| Gemini | TTS: Gemini TTS | `GEMINI_API_KEY` |
| xAI | STT: Grok STT | `XAI_API_KEY` |

Google Cloud needs the Speech-to-Text API v2 enabled:

```bash
gcloud services enable speech.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
```

Chirp 3 is served from multi-regions such as `us` / `eu`, so set
`GOOGLE_CLOUD_LOCATION` to a region, not `global`.

---

## Sign-up required

Ordered by "largest free tier, no credit card first".

### 1. Deepgram — STT: Nova-3 / TTS: Aura-2

- Sign up: <https://console.deepgram.com/signup>
- Free tier: **$200 credit, no credit card required** (enough to cover every
  STT run in this benchmark)
- Key issuance: Console → target project → **API Keys** → *Create a New API Key*
  - The `Member` scope is sufficient
- Environment variable: `DEEPGRAM_API_KEY`
- Note: one account covers both STT and TTS

### 2. AssemblyAI — STT: Universal-3.5 pro

- Sign up: <https://www.assemblyai.com/dashboard/signup>
- Free tier: **$50 credit** (one-time, on first registration)
- Key issuance: the API key is shown on the dashboard front page
- Environment variable: `ASSEMBLYAI_API_KEY`

### 3. Speechmatics — STT: Enhanced / Standard

- Sign up: <https://portal.speechmatics.com/signup>
- Free tier: **20 hours/month** (up to 2 concurrent sessions) — effectively
  free for this benchmark
- Key issuance: Portal → **API Keys** → *Create API Key*
  - The same key works for Batch and Real-time
- Environment variable: `SPEECHMATICS_API_KEY`
- Note: `enhanced` and `standard` are an `operating_point` switch on the
  same key, so one account covers both models

### 4. ElevenLabs — STT: Scribe v2

- Sign up: <https://elevenlabs.io/app/sign-up>
- Free tier: **10,000 credits/month (roughly 10 minutes)** — not enough for
  the benchmark, so plan on a paid plan (Starter, from $5/month)
- Key issuance: account menu (bottom right) → **API Keys** → *Create API Key*
- Environment variable: `ELEVENLABS_API_KEY`
- Note: `scribe_v2` (batch) and `scribe_v2_realtime` (streaming) both work
  with the same key

### 5. Cartesia — TTS: Sonic 3.0 / Sonic 3.5

- Sign up: <https://play.cartesia.ai/sign-up>
- Free tier: a small trial credit. For sustained use, Pro at $5/month is the
  realistic option
- Key issuance: Dashboard → **API Keys** → *Create API Key*
- Environment variable: `CARTESIA_API_KEY`
- Note: `sonic-3` and `sonic-3.5` are a `model_id` switch on the same key.
  Pick one `voice_id` before benchmarking and keep it fixed — changing the
  voice makes round-trip WER incomparable

### 6. Soniox — STT: stt-rt-v5

- Sign up: <https://console.soniox.com/signup>
- Free tier: **none** (free credits for new API registrations were
  discontinued as an anti-abuse measure). Usage-billed, so a payment method
  must be registered up front
- Key issuance: Console → **API Keys** → *New API Key*
- Environment variable: `SONIOX_API_KEY`
- Note: the unit price is the cheapest of the field (about $0.12/hour), so
  the benchmark cost stays tiny even after registering billing

---

## Verify

```bash
cp .env.example .env
# after filling the keys into .env:
uv run audio-harness doctor
```

`doctor` sends each provider one cheap authenticated GET and reports key
validity, balances where visible, and reachability. No audio is sent, so
essentially nothing is billed.

---

## Cost estimate

Running 30 minutes of evaluation audio through 8 STT models in streaming +
batch:

| Provider | Rate (streaming, USD/hr) | 30 min × 2 modes |
| --- | ---: | ---: |
| xAI Grok STT | 0.20 | $0.15 |
| Soniox stt-rt-v5 | 0.12 | $0.12 |
| Speechmatics Standard | 0.24 | within free tier |
| Speechmatics Enhanced | 0.43 | within free tier |
| AssemblyAI Universal-3.5 pro | 0.45 | within free credit |
| Deepgram Nova-3 | 0.46 | within free credit |
| ElevenLabs Scribe v2 | 0.36 | paid plan required |
| Google Chirp 3 | 0.96 | $0.72 |

TTS bills per character, so a prompt set of about 1,000 characters totals
under $1. In other words, **the whole benchmark completes for a few dollars
when the free tiers are used**.

Rates are public list prices as of 2026-08-05. Cross-check against
`PRICING_CHECKED` in `src/audio_harness/config.py` and re-verify on each
vendor's pricing page if stale.
