# backtranslation_passages_eurlex

Backtranslation-style prompt generation from Danish EUR-Lex records.

The pack builds Danish instruction rows where:

- `target` is a Danish legal text chunk from `oliverkinch/eur-lex`
- `prompt` is an LLM-generated Danish user request that could plausibly produce that text

Current defaults focus on quality for instruction finetuning:

- uses only Danish fields (`title_da`, `text_da`)
- keeps all `resource_type` categories
- filters to `text_source_da=html` in source config
- uses higher minimum article length than DSK (`min_article_chars: 900`)

Run smoke:

```bash
uv run sdg build sdg/packs/backtranslation_passages_eurlex/configs/eurlex_html_smoke.yaml
```

Run full:

```bash
uv run sdg build sdg/packs/backtranslation_passages_eurlex/configs/eurlex_html.yaml
```
