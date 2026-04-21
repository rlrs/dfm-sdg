# backtranslation

`backtranslation` is a pack for instruction backtranslation. It takes existing text and uses a model to generate the user prompt that would naturally produce that text, creating `prompt` / `target` pairs for instruction tuning.

The pack supports multiple generation modes depending on the source data.

## Modes

### `ArticleMode` (default)

Reverse-engineers a realistic prompt from a source article. Used when the config has no `summary_field`.

The target is the source article text.

### `EurLexSumMode`

Generates varied summarization requests paired with reference summaries as targets. Used when the config specifies a `summary_field`.

Prompt diversity is driven by 12 style seeds covering different structures (request-first, document-first, document mid-message) and tones (direct, polite, formal, informal, terse). EUR-Lex metadata headers are stripped from targets so they begin directly with prose.

## Config reference

| Field | Description |
|---|---|
| `source.dataset` | HuggingFace dataset path |
| `source.text_field` | Field containing the main document text |
| `source.summary_field` | *(optional)* Field containing the reference summary — enables `EurLexSumMode` |
| `source.title_field` | *(optional)* Field containing the document title |
| `source.url_field` | *(optional)* Field containing the document URL |
| `source.id_field` | *(optional)* Field to use as a stable row identifier |
| `generation.min_article_chars` | Minimum character length for source documents |
| `generation.temperature` | Sampling temperature for the instruction writer |
| `generation.train_fraction` | Fraction of rows allocated to the train split on publish |

## Configs

- `configs/base.yaml` — Danish Wikipedia articles (`oliverkinch/danish_wikipedia`)
- `configs/eur_lex_sum_da.yaml` — EUR-Lex Summary Danish (`oliverkinch/eur-lex-sum`)

## Usage

```bash
uv run sdg build sdg/packs/backtranslation/configs/base.yaml
uv run sdg verify <run-id>
uv run sdg publish <run-id>
uv run sdg upload-hf <run-id> --repo <org>/<dataset> --split train --private
```
