# Real-recording transcript anchor (50 items)

This kit prepares offline interview clips for independent human transcription. No
provider transcript exists at this stage, so the review sheet contains only clip
identity and recording metadata plus one empty `true_transcript` field.

## Files

- `anchor-review-sheet.csv` — replay each local clip identified by row number, path,
  session, language, and duration; enter the verbatim transcript in `true_transcript`.
- `README.md` — this labeling protocol.

## Protocol

- Replay the referenced audio locally; do not upload production audio or identifiers.
- Put exactly one verbatim human transcript in `true_transcript` for every completed row.
- Preserve hesitations, repetitions, numbers, names, and code-switching as spoken.
- Leave a row empty only while it is genuinely unreviewed; do not invent placeholder text.
- Selection is stratified over English/Japanese and 2--30 second duration buckets,
  limited to two clips per session, video-first, with seed 20260808.
