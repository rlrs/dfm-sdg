# translation

`translation` builds translation instruction rows from parallel Danish-English text pairs.

It is aimed at datasets like `oliverkinch/machine-translation-da-en` and `oliverkinch/eur-lex`, where each source row already contains aligned Danish and English text.

## Current shape

- source pairs can come from a local JSONL file or a Hugging Face dataset
- configs can use either a single `source` / `generation` pair or a `sources:` list with one `source` / `generation` block per corpus
- Hugging Face loading can stream rows so large corpora do not need to be materialized locally
- `generation.max_pairs_per_source` can cap rows per upstream source label when the corpus mixes multiple sources in one dataset
- `generation.max_pair_chars` can skip rows that are too large to be useful as prompts
- `generation.long_text_threshold_chars` with `generation.long_text_template_styles` can keep long rows on a simpler prompt subset
- each source pair produces one row
- a multi-source config emits one combined dataset artifact and applies train/eval splitting per source config
- prompt surfaces are distributed across rows over:
  - translation direction: `da-en` and `en-da`
  - prompt language: Danish and English
  - template style: plain prompt variants, marked-span input, JSON-field input, JSON object output, XML output, YAML output, and JSON-string output
- outputs are standard `prompt` / `target` rows, where `target` is the gold translation text

## Configs

- `configs/base.yaml`: combined `oliverkinch/machine-translation-da-en` and `oliverkinch/eur-lex`
- `configs/machine_translation_da_en.yaml`: `oliverkinch/machine-translation-da-en`
- `configs/eur_lex.yaml`: `oliverkinch/eur-lex` using `text_da` / `text_en`, with long-document filtering and long-text template fallback

`oliverkinch/eur-lex` exposes one `default` config and one `train` split. Its rows include `celex`, `resource_type`, `text_da`, `text_en`, `chars_da`, and `chars_en`.

## Usage

```bash
uv run sdg build sdg/packs/translation/configs/base.yaml
uv run sdg build sdg/packs/translation/configs/machine_translation_da_en.yaml
uv run sdg build sdg/packs/translation/configs/eur_lex.yaml
uv run sdg verify <run-id>
uv run sdg publish <run-id>
```
