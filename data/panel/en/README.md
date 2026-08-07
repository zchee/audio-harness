# E2 TTS arena human panel — English (100 pairs, 2 raters)

This kit collects the human ranking for arena gate criterion (i) (plan P4 step 17): a
panel of two raters, 100 concatenated pairs, Bradley-Terry aggregation. The gate rule:
no pair of systems whose bootstrap CIs are separated may be ranked discordantly versus
the human-panel Bradley-Terry ranking.

## Files

- `rating-sheet-rater1.csv` / `rating-sheet-rater2.csv` — one row per pair; fill
  `winner` with `first`, `second`, or `tie`. Both raters rate the same 100 pairs.
- `answer-key.jsonl` — pair_id -> (prompt, first system, second system, wav path).
  RATERS MUST NOT OPEN THIS FILE; it exists to convert filled sheets into the
  `PanelVote` JSONL the arena consumes (fields: rater, prompt_id, first, second,
  winner) after rating is complete.
- Audio: `results/panel-kit-en/pairs/pair-001.wav` … `pair-100.wav` (not committed;
  regenerate with the seed below if missing).

## Rating protocol (blind)

Each WAV holds two renditions of the same text: clip one, a short pause, clip two.
Judge which rendition is the better text-to-speech output OVERALL, weighing equally:

- naturalness — closer to a fluent human speaker, less obviously synthetic;
- prosody — rhythm, stress, intonation and phrasing appropriate to the text;
- artifacts — fewer glitches, clicks, buzzing, distortion, dropouts, robotic timbre.

Rules: rate independently (no discussion between raters until both sheets are done);
`tie` is allowed but should be a genuine coin-flip, not a default; use headphones; you
may replay a pair, always in the given order; never look at the answer key.

## Provenance

100 pairs sampled with seed 20260808 from the en arena run
(results/arena-en/20260806-151700): 4 systems (cartesia-sonic3, cartesia-sonic35,
deepgram-aura2, gemini-tts), all 6 system pairings x sampled prompts, presentation
order randomized per pair, clips concatenated by the arena's own pair builder
(same gap and sample rate the LLM judges hear). Generated 2026-08-08.
