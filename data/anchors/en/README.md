# E1 semantic-judge human anchor — English (100 items)

This kit anchors the E1 semantic-fidelity judge (plan P4 step 16). A human labels each
item; Cohen's kappa between the judge's majority vote and these labels gates whether the
judge's numbers may be ranked (gate: kappa >= 0.75 point estimate; bootstrap CI reported).

## Files

- `anchor-review-sheet.csv` — what you read while labeling: row number, clip/provider,
  the ground-truth `reference`, and the provider's `hypothesis` transcript.
- `anchor-items.csv` — what you fill in: put exactly one label per row in `human_label`.
  Row order matches the review sheet by `clip_id` + `provider`.

## Labels (pick exactly one per item)

- `meaning-changing` — the hypothesis changes what the utterance means for a listener
  acting on it (wrong action, wrong assertion, dropped/negated content, fabricated
  content, or so garbled the meaning is lost).
- `entity` — a concrete fact is corrupted: number, date, currency amount, identifier,
  or proper name is wrong, missing, or invented, while the rest of the meaning survives.
  Entity damage outranks general meaning change: if both apply, label `entity`.
- `harmless` — differences are cosmetic: casing, punctuation, filler words,
  contractions, benign spelling variants, or small omissions a listener would not act on
  differently.

## Protocol

- Label from the text alone (the judge sees the same evidence — stream-final transcripts
  from saved results; no audio is replayed).
- An empty hypothesis is `meaning-changing` (dropping everything changes meaning), never
  a skip.
- Do not consult the judge's output or another person's labels; the anchor must be
  independent.
- Provenance: 100 items stratified across 11 English stream lanes, sampled with seed
  20260808 from the canonical saved results (pipecat + endpointing corpora, verified
  references only). Generated 2026-08-08.
