# FLEURS multilingual TTS guardrail prompts

- Source: https://huggingface.co/datasets/google/fleurs (``data/<locale>/test.tsv``)
- License: CC-BY-4.0 (FLEURS; Conneau et al., arXiv:2205.12446)
- Retrieved: 2026-08-12 by scripts/extract_fleurs_prompts.py
- Seed: 20260812 — sentences deduplicated, length-filtered to
  60-120 characters, sorted, then sampled with
  ``random.Random(seed)``; 10 per language
- Text column: ``raw_transcription`` (original punctuation kept for TTS)

Language set: the 12 Speech-MASSIVE languages of
configs/stt-speech-massive-internal.yaml. FLEURS regional variants differ
for three of them, so the prompt text region does not always match the
Speech-MASSIVE tag:

| file | FLEURS locale | Speech-MASSIVE tag | region note |
|------|---------------|--------------------|-------------|
| ar.txt | ar_eg | ar-SA | FLEURS Arabic is Egyptian, not Saudi |
| de.txt | de_de | de-DE | same region |
| es.txt | es_419 | es-ES | FLEURS Spanish is Latin American, not Castilian |
| fr.txt | fr_fr | fr-FR | same region |
| hu.txt | hu_hu | hu-HU | same region |
| ko.txt | ko_kr | ko-KR | same region |
| nl.txt | nl_nl | nl-NL | same region |
| pl.txt | pl_pl | pl-PL | same region |
| pt.txt | pt_br | pt-PT | FLEURS Portuguese is Brazilian, not European |
| ru.txt | ru_ru | ru-RU | same region |
| tr.txt | tr_tr | tr-TR | same region |
| vi.txt | vi_vn | vi-VN | same region |

Regenerate with:

    uv run python scripts/extract_fleurs_prompts.py
