# tidsskrift

`tidsskrift` generates Danish instruction-following rows from academic journal articles published on tidsskrift.dk via backtranslation.

It starts from finished articles, extracts prose passages, and asks one model to write the user prompt that would have produced each passage.

## Current shape

- source articles come from a Hugging Face dataset (default: `oliverkinch/tidsskrift-dk`)
- passages are extracted heuristically — short blocks, headers, reference lists, and non-Danish text are filtered out at extraction time
- one model role, `instruction_writer`, generates the prompt for each passage
- prompt length is varied deterministically across three buckets (short / medium / long) to encourage diversity
- each row in the output carries a persona (sampled from `nvidia/Nemotron-Personas-USA`) that shapes the register of the generated prompt
- outputs are standard `prompt` / `target` rows, where `target` is the source passage text

## Base config

The starter config points at `oliverkinch/tidsskrift-dk`.

That dataset has one `train` split and exposes `url`, `title`, `text`, `journal`, and `journal_description` fields. If you switch to a different source, update the `*_field` keys in the config accordingly.

## Usage

```bash
uv run sdg build sdg/packs/tidsskrift/configs/base.yaml
uv run sdg verify <run-id>
uv run sdg publish <run-id>
```
