# MaaS account setup checklist

日本語版: [ACCOUNTS.ja.md](ACCOUNTS.ja.md)

Accounts and API keys needed for the benchmarked lanes (51 registered — see
the provider tables in the README). Sign-ups must be done by a human (an
agent cannot do them for you). Once the keys are in `.env`, run
`uv run audio-harness doctor` to verify every credential with a cheap
authenticated request.

---

## Already held

| Provider | Used for | Required environment variables |
| --- | --- | --- |
| Google Cloud | STT: Chirp 3 | `GOOGLE_APPLICATION_CREDENTIALS` (SA JSON) or ADC, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` |
| Gemini | TTS: Gemini TTS / 3.1 preview; sim-interview generator | `GEMINI_API_KEY` (optional `GEMINI_TTS_VOICE`) |
| xAI | STT: Grok STT / TTS: Grok TTS | `XAI_API_KEY` |

Google Cloud needs the Speech-to-Text API v2 enabled:

```bash
gcloud services enable speech.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
```

Chirp 3 is served from multi-regions such as `us` / `eu`, so set
`GOOGLE_CLOUD_LOCATION` to a region, not `global`. Note chirp_3 is not
served from `asia-northeast1` (verified 2026-08-12).

---

## Sign-up required

Ordered by "largest free tier, no credit card first".

### 1. Deepgram — STT: Nova-3, Flux / TTS: Aura-2

- Sign up: <https://console.deepgram.com/signup>
- Free tier: **$200 credit, no credit card required**
- Key issuance: Console → target project → **API Keys** → *Create a New API Key*
  (the `Member` scope is sufficient)
- Environment variables: `DEEPGRAM_API_KEY` (optional `DEEPGRAM_TTS_VOICE`)
- Notes: one account covers STT and TTS. The harness always sends
  `mip_opt_out=true`, keeping audio out of Deepgram's Model Improvement
  Program. Model access differs per project — `nova-3-multilingual` may 403
  on newer projects while `nova-3` now serves ja directly (2026-08-12).

### 2. AssemblyAI — STT: Universal-3.5 pro

- Sign up: <https://www.assemblyai.com/dashboard/signup>
- Free tier: **$50 credit** (one-time)
- Key issuance: shown on the dashboard front page
- Environment variable: `ASSEMBLYAI_API_KEY`
- Note: the batch lane supports `delete_after` — the harness deletes stored
  transcripts right after retrieval (content is scrubbed server-side)

### 3. Speechmatics — STT: Enhanced / Standard

- Sign up: <https://portal.speechmatics.com/signup>
- Free tier: **20 hours/month** (up to 2 concurrent sessions)
- Key issuance: Portal → **API Keys** → *Create API Key* (Batch and
  Real-time share the key)
- Environment variable: `SPEECHMATICS_API_KEY`
- Notes: `enhanced` / `standard` are an `operating_point` switch on one key.
  `delete_after` removes batch jobs right after retrieval (verified 404)

### 4. OpenRouter — one key, many hosted lanes

- Sign up: <https://openrouter.ai>
- Covers: STT `or-parakeet` / `or-fish-transcribe` / `or-mai-transcribe`;
  TTS `or-qwen-tts-*`, `or-fish-s*`, `or-mai-voice-2*`, `or-minimax-*`,
  `or-flux-tts` (**:free — bills nothing**) and the OSS set
  (`or-kokoro` / `or-orpheus` / `or-csm` / `or-zonos`)
- Environment variable: `OPENROUTER_API_KEY`
- Notes: hosted-proxy lanes — medians are direct-comparable (+~0.1s,
  measured 2026-08-12), tails are not. Never used for production audio
  (double-hop processing keeps OpenRouter off the real-data allowlist)

### 5. OpenAI — STT: gpt-transcribe family / TTS: gpt-4o-mini-tts

- Sign up: <https://platform.openai.com>
- Environment variable: `OPENAI_API_KEY`
- Note: API inputs are not used for training, but standard accounts retain
  payloads ~30 days for abuse monitoring (ZDR is enterprise/approval-gated)

### 6. ElevenLabs — STT: Scribe v2 / TTS: Eleven v3, Flash v2.5

- Sign up: <https://elevenlabs.io/app/sign-up>
- Free tier: **10,000 credits/month (~10 minutes)** — plan on a paid plan
  (Starter, from $5/month) for real runs
- Environment variables: `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`
  (pin one voice; changing it breaks round-trip comparability)

### 7. Cartesia — TTS: Sonic 3.0 / 3.5, STT: Ink-2

- Sign up: <https://play.cartesia.ai/sign-up>
- Free tier: small trial credit; Pro at $5/month for sustained use
- Environment variables: `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID` (pin one)
- Notes: Ink-2 streaming is English-only; its batch mode runs the separate
  `ink-whisper` model

### 8. Soniox — STT: stt-rt-v5 / TTS: tts-rt-v2

- Sign up: <https://console.soniox.com/signup>
- Free tier: **none** (usage-billed; register a payment method up front)
- Environment variable: `SONIOX_API_KEY`
- Notes: cheapest STT of the field (~$0.12/hr). The batch STT mode runs the
  separate `stt-async-v5` lineage and the harness deletes uploaded assets
  unconditionally after each request

### 9. Mistral — STT: Voxtral realtime / TTS: Voxtral TTS

- Sign up: <https://console.mistral.ai>
- Environment variable: `MISTRAL_API_KEY`

### 10. Gladia — STT: Solaria-1 / Solaria-3

- Sign up: <https://app.gladia.io>
- Environment variable: `GLADIA_API_KEY`
- Note: paid tiers never train on customer audio; the free tier does and
  retains up to 12 months — use a paid tier for anything sensitive

### 11. Inworld — TTS: Inworld TTS 2

- Sign up: <https://studio.inworld.ai>
- Environment variable: `INWORLD_API_KEY`

### 12. Azure — STT: Azure Speech / TTS: Azure Neural

- Sign up: Azure portal, create a Speech resource
- Environment variables: `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`
- Note: both lanes go through the vendor SDK and carry the `sdk_buffered`
  latency caveat

---

## Local lanes (no accounts, $0)

| Lane | Requirement |
| --- | --- |
| `whisper-local` | `uv sync --extra judge-whisper` (mlx-whisper) |
| `parakeet-ane` | build the Swift sidecar under `sidecars/parakeet-ane/` |
| `apple-speech-stt` | macOS dictation language installed per language (System Settings → Keyboard → Dictation) |
| Kokoro (sim persona voice) | `uv sync --extra sim-kokoro` |

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

## Cost

Pricing is data that rots, so the canonical rates live in
`src/audio_harness/config.py` (`STT_PRICING` / `TTS_PRICING`) with the date
each rate was last verified; cross-check `PRICING_CHECKED` before quoting a
figure. As orientation: a full English STT sweep (30 clips, both modes,
every cloud lane) lands in single-digit dollars, TTS sweeps bill per
character (a 1,000-character prompt set totals under $1 per lane for most
vendors), and free tiers cover a large share of the field.
