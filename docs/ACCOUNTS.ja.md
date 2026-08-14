# MaaS アカウント準備チェックリスト

English version: [ACCOUNTS.md](ACCOUNTS.md)

ベンチマーク対象モデルに必要なアカウントと API キーの一覧。
アカウント作成そのものは各自で実施してください（エージェントによる代行不可）。
キーを \`.env\` に書いたら \`uv run audio-harness doctor\` で疎通確認できます。

---

## 既に保有

| Provider | 用途 | 必要な環境変数 |
| --- | --- | --- |
| Google Cloud | STT: Chirp 3 | \`GOOGLE_APPLICATION_CREDENTIALS\` (SA JSON) または ADC、\`GOOGLE_CLOUD_PROJECT\`、\`GOOGLE_CLOUD_LOCATION\` |
| Gemini | TTS: Gemini TTS | \`GEMINI_API_KEY\` |
| xAI | STT: Grok STT | \`XAI_API_KEY\` |

Google Cloud は Speech-to-Text API v2 の有効化が必要です:

\`\`\`bash
gcloud services enable speech.googleapis.com --project "$GOOGLE_CLOUD_PROJECT"
\`\`\`

Chirp 3 は \`us\` / \`eu\` などのマルチリージョンで提供されるため、
\`GOOGLE_CLOUD_LOCATION\` は \`global\` ではなくリージョンを指定してください。

---

## 新規作成が必要

作成順は「無料枠が大きく、クレカ不要なもの」から並べてあります。

### 1. Deepgram — STT: Nova-3 / TTS: Aura-2

- サインアップ: <https://console.deepgram.com/signup>
- 無料枠: **$200 クレジット、クレジットカード不要**（本ベンチの全 STT 実行を賄える規模）
- キー発行: Console → 対象 Project → **API Keys** → *Create a New API Key*
  - スコープは \`Member\` で十分
- 環境変数: \`DEEPGRAM_API_KEY\`
- 備考: 1 アカウントで STT と TTS の両方をカバーできます

### 2. AssemblyAI — STT: Universal-3.5 pro

- サインアップ: <https://www.assemblyai.com/dashboard/signup>
- 無料枠: **$50 クレジット**（初回登録時、一度きり）
- キー発行: Dashboard トップに API key が表示されます
- 環境変数: \`ASSEMBLYAI_API_KEY\`

### 3. Speechmatics — STT: Enhanced / Standard

- サインアップ: <https://portal.speechmatics.com/signup>
- 無料枠: **月 20 時間**（同時実行 2 まで）— 本ベンチでは実質無料
- キー発行: Portal → **API Keys** → *Create API Key*
  - Batch と Real-time で同じキーが使えます
- 環境変数: \`SPEECHMATICS_API_KEY\`
- 備考: \`enhanced\` と \`standard\` は同一キーの \`operating_point\` 切り替えなので、
  アカウントは 1 つで 2 モデル分カバーできます

### 4. ElevenLabs — STT: Scribe v2

- サインアップ: <https://elevenlabs.io/app/sign-up>
- 無料枠: **月 10,000 クレジット（約 10 分相当）** — ベンチには不足するため
  有料プラン（Starter $5/月〜）への切り替えを想定してください
- キー発行: 右下のアカウントメニュー → **API Keys** → *Create API Key*
- 環境変数: \`ELEVENLABS_API_KEY\`
- 備考: \`scribe_v2\`（バッチ）と \`scribe_v2_realtime\`（ストリーミング）は
  同一キーで両方利用できます

### 5. Cartesia — TTS: Sonic 3.0 / Sonic 3.5

- サインアップ: <https://play.cartesia.ai/sign-up>
- 無料枠: 試用クレジットあり（少量）。継続利用は Pro $5/月 が現実的
- キー発行: Dashboard → **API Keys** → *Create API Key*
- 環境変数: \`CARTESIA_API_KEY\`
- 備考: \`sonic-3\` と \`sonic-3.5\` は同一キーの \`model_id\` 切り替えです。
  ベンチ実行前に使用する \`voice_id\` を 1 つ決めて固定してください
  （声質が変わると round-trip WER が比較不能になります）

### 6. Soniox — STT: stt-rt-v5

- サインアップ: <https://console.soniox.com/signup>
- 無料枠: **なし**（不正利用対策で新規 API 登録への無料クレジット提供は終了）
  従量課金のため、事前に支払い方法の登録が必要です
- キー発行: Console → **API Keys** → *New API Key*
- 環境変数: \`SONIOX_API_KEY\`
- 備考: 単価は最安（約 $0.12/時）なので、課金登録してもベンチ費用は僅少です

---

## 作成後の確認

\`\`\`bash
cp .env.example .env
# .env に取得したキーを記入してから:
uv run audio-harness doctor
\`\`\`

\`doctor\` は各プロバイダに 1 回ずつ安価な認証済み GET を投げ、
キーの有効性・残高の見えるものは残高・到達可能性を表示します。
音声は一切送らないので課金はほぼ発生しません。

---

## 費用見積

30 分の評価音声を STT 8 モデル × ストリーミング + バッチで回した場合:

| Provider | 単価 (streaming, USD/hr) | 30 分 × 2 モード |
| --- | ---: | ---: |
| xAI Grok STT | 0.20 | $0.15 |
| Soniox stt-rt-v5 | 0.12 | $0.12 |
| Speechmatics Standard | 0.24 | 無料枠内 |
| Speechmatics Enhanced | 0.43 | 無料枠内 |
| AssemblyAI Universal-3.5 pro | 0.45 | 無料クレジット内 |
| Deepgram Nova-3 | 0.46 | 無料クレジット内 |
| ElevenLabs Scribe v2 | 0.36 | 要有料プラン |
| Google Chirp 3 | 0.96 | $0.72 |

TTS 側は文字数課金のため、1,000 文字程度のプロンプト集なら合計 $1 未満です。
つまり **無料枠を使えばベンチ全体を数ドル以内で完走できます**。

料金は 2026-08-05 時点の公開値です。\`src/audio_harness/config.py\` の
\`PRICING_CHECKED\` と突き合わせ、古い場合は各社の料金ページで再確認してください。
