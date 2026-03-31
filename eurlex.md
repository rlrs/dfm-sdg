# EUR-Lex Plan

## Goal

Add one new backtranslation profile for Danish legal summarization using `dennlinger/eur-lex-sum`.

The task shape is:

- input: Danish legal text
- target: Danish summary
- generated instruction: a short Danish prompt asking for a formal legal summary

## Why This Fits The Existing Pack

The current backtranslation pack already does the core thing we need:

- take a finished target
- ask a model to infer a prompt that would have produced it

For EUR-Lex summarization, the only difference is that the final prompt must include a source document in addition to the generated instruction.

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

Example generated instruction shape:

`Skriv en kort, saglig opsummering på dansk af nedenstående EU-retsakt. Fremhæv formål, anvendelsesområde, centrale bestemmelser og berørte aktører. Hold stilen neutral og officiel.`

## Row Construction

For this profile:

- generated instruction becomes the first part of `prompt`
- the legal text is appended after the instruction
- the gold summary becomes `target`

Conceptually:

- `prompt = <generated instruction> + "\\n\\nTekst:\\n" + <input_text>`
- `target = <target_text>`

## Config Shape

Add a separate config file instead of mixing this into the Wikipedia config.

Example:

```yaml
pack: backtranslation

models:
  instruction_writer: openai

source:
  dataset: dennlinger/eur-lex-sum
  config_name: da

task:
  profile: legal_summary_da

generation:
  min_input_chars: 1000
  min_target_chars: 200
  max_input_chars: 12000
  temperature: 0.2
  train_fraction: 0.9
```

## Implementation Steps

1. Generalize the normalized source record from article-specific fields to generic example fields.
2. Add profile-specific record loading for `legal_summary_da`.
3. Add profile-specific instruction prompting for EUR-Lex summarization.
4. Build final prompts by combining generated instruction with the legal source text.
5. Add one dedicated config file for EUR-Lex.
6. Add a focused test that uses a tiny fake EUR-Lex-style source example.

## Non-Goals

Not part of the first version:

- translation backtranslation
- multilingual alignment logic
- using Dynaword as the EUR-Lex source
- multiple legal prompt styles

Those can be added later once the single Danish summarization profile works cleanly.
