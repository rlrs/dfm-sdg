# backtranslation_passages_dsk

Backtranslation-style prompt generation from DSK (Dansk Sprogteknologi Konsortium) passage chunks.

The source data is **private** and not publicly available. Place the parquet files in a local
directory and update the `path` fields in the config files accordingly.

Run smoke:

```bash
uv run sdg build sdg/packs/backtranslation_passages_dsk/configs/dsk_news_smoke.yaml
uv run sdg build sdg/packs/backtranslation_passages_dsk/configs/dsk_promo_smoke.yaml
```

Run full:

```bash
uv run sdg build sdg/packs/backtranslation_passages_dsk/configs/dsk_news.yaml
uv run sdg build sdg/packs/backtranslation_passages_dsk/configs/dsk_promo.yaml
```
