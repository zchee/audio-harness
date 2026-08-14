# MaaS アカウント準備チェックリスト

English version: [ACCOUNTS.md](ACCOUNTS.md)

ベンチマーク対象レーン(登録 51 本 — README の Provider 表を参照)に必要な
アカウントと API キーの一覧。アカウント作成そのものは各自で実施してください
(エージェントによる代行不可)。キーを `.env` に書いたら
`uv run audio-harness doctor` で疎通確認できます。

---

## 既に保有

| Provider | 用途 | 必要な環境変数 |
| --- | --- | --- |
| Google Cloud | STT: Chirp 3 | `GOOGLE_APPLICATION_CREDENTIALS` (SA JSON) または ADC、`GOOGLE_CLOUD_PROJECT`、`GOOGLE_CLOUD_LOCATION` |
| Gemini | TTS: Gemini TTS / 3.1 preview、sim-interview の生成 | `GEMINI_API_KEY`(任意: `GEMINI_TTS_VOICE`) |
| xAI | STT: Grok STT / TTS: Grok TTS | `XAI_API_KEY` |

Google Cloud は Speech-to-Text API v2 の有効化が必要です:

```bash
gcloud services enable speech.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
```

Chirp 3 は `us` / `eu` などのマルチリージョン提供のため、
`GOOGLE_CLOUD_LOCATION` は `global` ではなくリージョンを指定してください。
なお chirp_3 は `asia-northeast1`(東京)では提供されていません(2026-08-12 確認)。

---

## 新規作成が必要

作成順は「無料枠が大きく、クレカ不要なもの」から並べてあります。

### 1. Deepgram — STT: Nova-3, Flux / TTS: Aura-2

- サインアップ: <https://console.deepgram.com/signup>
- 無料枠: **$200 クレジット、クレジットカード不要**
- キー発行: Console → 対象 Project → **API Keys** → *Create a New API Key*
  (スコープは `Member` で十分)
- 環境変数: `DEEPGRAM_API_KEY`(任意: `DEEPGRAM_TTS_VOICE`)
- 備考: 1 アカウントで STT/TTS 両対応。ハーネスは常時 `mip_opt_out=true` を
  付与し、音声を Model Improvement Program に入れません。モデルアクセスは
  プロジェクト依存 — 新しめのプロジェクトでは `nova-3-multilingual` が 403 に
  なる一方、`nova-3` が ja を直接サポートします(2026-08-12 確認)

### 2. AssemblyAI — STT: Universal-3.5 pro

- サインアップ: <https://www.assemblyai.com/dashboard/signup>
- 無料枠: **$50 クレジット**(初回一度きり)
- キー発行: Dashboard トップに表示
- 環境変数: `ASSEMBLYAI_API_KEY`
- 備考: batch レーンは `delete_after` 対応 — 取得直後に保存 transcript を
  削除します(サーバ側でコンテンツ消去)

### 3. Speechmatics — STT: Enhanced / Standard

- サインアップ: <https://portal.speechmatics.com/signup>
- 無料枠: **月 20 時間**(同時実行 2 まで)
- キー発行: Portal → **API Keys** → *Create API Key*(Batch/Real-time 共通)
- 環境変数: `SPEECHMATICS_API_KEY`
- 備考: `enhanced` / `standard` は同一キーの `operating_point` 切り替え。
  `delete_after` で batch ジョブを取得直後に削除(404 検証済み)

### 4. OpenRouter — 1 キーで多数のホスト型レーン

- サインアップ: <https://openrouter.ai>
- 対象: STT `or-parakeet` / `or-fish-transcribe` / `or-mai-transcribe`、
  TTS `or-qwen-tts-*`、`or-fish-s*`、`or-mai-voice-2*`、`or-minimax-*`、
  `or-flux-tts`(**:free — 課金ゼロ**)、OSS 系
  (`or-kokoro` / `or-orpheus` / `or-csm` / `or-zonos`)
- 環境変数: `OPENROUTER_API_KEY`
- 備考: hosted-proxy レーン — 中央値は直行比較可(+~0.1s、2026-08-12 実測)、
  テールは不可。本番音声には使用しない(二重ホップのため実データ許可リスト外)

### 5. OpenAI — STT: gpt-transcribe 系 / TTS: gpt-4o-mini-tts

- サインアップ: <https://platform.openai.com>
- 環境変数: `OPENAI_API_KEY`
- 備考: API 入力は学習に使われませんが、標準アカウントは不正利用監視のため
  約 30 日ペイロードを保持します(ZDR は Enterprise/承認制)

### 6. ElevenLabs — STT: Scribe v2 / TTS: Eleven v3, Flash v2.5

- サインアップ: <https://elevenlabs.io/app/sign-up>
- 無料枠: **月 10,000 クレジット(約 10 分)** — 実運用は有料プラン
  (Starter $5/月〜)前提
- 環境変数: `ELEVENLABS_API_KEY`、`ELEVENLABS_VOICE_ID`
  (ボイスは 1 つに固定 — 変えると往復比較が壊れます)

### 7. Cartesia — TTS: Sonic 3.0 / 3.5、STT: Ink-2

- サインアップ: <https://play.cartesia.ai/sign-up>
- 無料枠: 少量の試用クレジット。継続は Pro $5/月
- 環境変数: `CARTESIA_API_KEY`、`CARTESIA_VOICE_ID`(1 つに固定)
- 備考: Ink-2 ストリーミングは英語専用。batch モードは別モデル
  `ink-whisper` が動きます

### 8. Soniox — STT: stt-rt-v5 / TTS: tts-rt-v2

- サインアップ: <https://console.soniox.com/signup>
- 無料枠: **なし**(従量課金 — 事前に支払い方法の登録が必要)
- 環境変数: `SONIOX_API_KEY`
- 備考: STT 単価は最安(約 $0.12/時)。batch STT は別系列 `stt-async-v5` が
  動き、ハーネスはアップロード資産をリクエスト毎に無条件削除します

### 9. Mistral — STT: Voxtral realtime / TTS: Voxtral TTS

- サインアップ: <https://console.mistral.ai>
- 環境変数: `MISTRAL_API_KEY`

### 10. Gladia — STT: Solaria-1 / Solaria-3

- サインアップ: <https://app.gladia.io>
- 環境変数: `GLADIA_API_KEY`
- 備考: 有料ティアは顧客音声を学習に使いませんが、**無料ティアは学習利用 +
  最長 12 ヶ月保持** — センシティブな音声には有料ティアを使ってください

### 11. Inworld — TTS: Inworld TTS 2

- サインアップ: <https://studio.inworld.ai>
- 環境変数: `INWORLD_API_KEY`

### 12. Azure — STT: Azure Speech / TTS: Azure Neural

- サインアップ: Azure ポータルで Speech リソースを作成
- 環境変数: `AZURE_SPEECH_KEY`、`AZURE_SPEECH_REGION`
- 備考: 両レーンともベンダ SDK 経由のため `sdk_buffered` レイテンシ注記付き

---

## ローカルレーン(アカウント不要・$0)

| レーン | 必要なもの |
| --- | --- |
| `whisper-local` | `uv sync --extra judge-whisper`(mlx-whisper) |
| `parakeet-ane` | `sidecars/parakeet-ane/` の Swift サイドカーをビルド |
| `apple-speech-stt` | macOS の音声入力言語を言語毎に導入(システム設定 → キーボード → 音声入力) |
| Kokoro(sim のペルソナ音声) | `uv sync --extra sim-kokoro` |

---

## 作成後の確認

```bash
cp .env.example .env
# .env に取得したキーを記入してから:
uv run audio-harness doctor
```

`doctor` は各プロバイダに 1 回ずつ安価な認証済み GET を投げ、
キーの有効性・残高の見えるものは残高・到達可能性を表示します。
音声は一切送らないので課金はほぼ発生しません。

---

## 費用

料金は「腐るデータ」なので、正典は `src/audio_harness/config.py` の
`STT_PRICING` / `TTS_PRICING`(各レートに検証日付き)です。引用前に
`PRICING_CHECKED` と突き合わせてください。目安: 英語 STT フルスイープ
(30 クリップ × 両モード × 全クラウドレーン)で 1 桁ドル、TTS は文字数課金で
1,000 文字規模のプロンプト集なら大半のベンダで 1 レーン $1 未満、
無料枠でかなりの範囲を賄えます。
