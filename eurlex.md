# EUR-Lex Plan

## Goal

Add one new backtranslation profile for Danish legal summarization using `dennlinger/eur-lex-sum`.

The task shape is:

- input: Danish legal text
- target: Danish summary
- prompt: a fixed Danish instruction asking for a formal legal summary

## Why This Fits The Existing Pack

The current backtranslation pack already does the core thing we need:

- take a finished target
- ask a model to infer a prompt that would have produced it

For EUR-Lex summarization, the final prompt must include a source document in addition to the instruction.

The important difference is that this profile should not use an LLM prompt generator.

The EUR-Lex documents are long enough that sending the full source text to another model just to synthesize an instruction would be unnecessarily expensive.

So this profile should be template-backed, not model-backed.

## Minimal Refactor

Keep this in the same pack, but generalize the internal normalized example shape.

Introduce a normalized record with fields like:

- `source_id`
- `task_profile`
- `input_text`
- `target_text`
- `title`
- `meta`

Then add one small profile layer:

- `wikipedia_article_da`
- `legal_summary_da`

## EUR-Lex Profile

Profile name:

- `legal_summary_da`

Dataset source:

- `dennlinger/eur-lex-sum`

Preferred output style:

- short
- formal
- neutral
- Danish
- focused on purpose, scope, key provisions, and affected parties

Example instruction shape:

`Skriv en kort, saglig opsummering på dansk af nedenstående EU-retsakt. Fremhæv formål, anvendelsesområde, centrale bestemmelser og berørte aktører. Hold stilen neutral og officiel.`

This should come from a fixed template or a very small hand-written variant set.

No LLM generation step is needed for the instruction itself.

## Length First

Length should be a first-class concern for this profile.

Before choosing final thresholds, we should scan the dataset and measure:

- input character lengths
- input token lengths
- summary character lengths
- summary token lengths
- compression ratio between source and summary

Then filter explicitly instead of guessing.

Likely controls:

- `min_input_chars`
- `max_input_chars`
- `min_target_chars`
- `max_target_chars`
- optional token-based equivalents if we want model-specific context control

The first implementation should also record length metadata on each row so we can tighten filters later without rerunning the analysis step.

## Row Construction

For this profile:

- fixed instruction becomes the first part of `prompt`
- the legal text is appended after the instruction
- the gold summary becomes `target`

Conceptually:

- `prompt = <fixed instruction> + "\\n\\nTekst:\\n" + <input_text>`
- `target = <target_text>`

## Config Shape

Add a separate config file instead of mixing this into the Wikipedia config.

Example:

```yaml
pack: backtranslation

source:
  dataset: dennlinger/eur-lex-sum
  config_name: da

task:
  profile: legal_summary_da

generation:
  min_input_chars: 1000
  min_target_chars: 200
  max_input_chars: 12000
  train_fraction: 0.9
```

## Implementation Steps

1. Add a generic example shape in the backtranslation pack that supports `input_text` and `target_text`.
2. Add a `legal_summary_da` profile that loads EUR-Lex source-summary pairs directly.
3. Add a fixed Danish instruction template for the profile instead of using an LLM.
4. Add a small stats step or helper to inspect source and summary length distributions.
5. Add explicit min/max length filtering for both input and target.
6. Build final prompts by combining the fixed instruction with the legal source text.
7. Add one dedicated config file for EUR-Lex.
8. Add a focused test that uses a tiny fake EUR-Lex-style source example.

## Non-Goals

Not part of the first version:

- LLM-based prompt generation
- translation backtranslation
- multilingual alignment logic
- using Dynaword as the EUR-Lex source
- multiple legal prompt styles

Those can be added later once the single Danish summarization profile works cleanly.
