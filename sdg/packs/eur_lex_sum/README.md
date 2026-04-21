# eur_lex_sum

`eur_lex_sum` builds Danish summarization rows from EUR-Lex source-summary pairs.

Each row contains:

- `prompt`: a natural Danish summarization request with the source document inserted
- `target`: the reference summary with the leading EUR-Lex boilerplate removed

The pack is dedicated to this task shape. It does not share task logic with `backtranslation`.

## Source fields

The source config must provide:

- `text_field` for the document text
- `summary_field` for the gold summary

Optional fields:

- `title_field`
- `url_field`
- `id_field`

## Prompt variation

Prompt framing is generated with one `instruction_writer` model and a fixed pool of 12 Danish style seeds covering:

- request-first prompts
- document-first prompts
- document-in-the-middle prompts
- terse and formal variants

## Usage

```bash
uv run sdg build sdg/packs/eur_lex_sum/configs/base.yaml
uv run sdg verify <run-id>
uv run sdg publish <run-id>
```
