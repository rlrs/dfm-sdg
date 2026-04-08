# Verifiable Reasoning Pack

`verifiable_reasoning` now uses an explicit recipe-planning system instead of relying on ad hoc randomness.

Each family defines a small recipe catalog. A build first creates a balanced `plan.json`, then generates prompt-only puzzle rows from those planned recipes, and finally verifies both puzzle quality and dataset-level coverage. The hidden solution stays internal to the row so a later LLM solve step can attach a response and run answer verification separately.

If you want the pack to attach model-generated answers during build, set `generation.attach_targets: true` and provide `models.answer_teacher`. The solve scaffold asks the model to reason step by step and finish with a final `Answer:` block. The build keeps the reasoning text, parses the final answer through the family parser, normalizes it to the family's canonical answer format, and then lets normal verification check correctness.

## Current families

- `zebra_logic`
  - bilingual English/Danish
  - four-house and five-house zebra puzzles
  - multiple prompt styles, axis profiles, clue profiles, and difficulty tiers
  - house-ledger answer contract
- `lineup_logic`
  - bilingual English/Danish
  - queue, chair, and ranking scenarios
  - four-, five-, and six-person variants
  - broader ordering clue types, including edge clues, negative position clues, middle-position clues, `between`, `not_adjacent`, and `one_between`
- `futoshiki_logic`
  - bilingual English/Danish
  - 4x4 and 5x5 board puzzles with fixed cells and inequality clues
  - multiple prompt styles and shared surface variation
  - number-grid answer contract
- `skyscraper_logic`
  - bilingual English/Danish
  - 4x4 and 5x5 board puzzles with border visibility clues
  - multiple prompt styles and clue-density profiles
  - number-grid answer contract
- `numbrix_logic`
  - bilingual English/Danish
  - square and rectangular boards with 0-based number paths
  - sparse, balanced, and near-complete given regimes inspired by Dolci
  - number-grid answer contract
- `hitori_logic`
  - bilingual English/Danish
  - rectangular and square boards with full matrix givens
  - duplicate-elimination, non-adjacent black cells, and white-cell connectivity
  - mask-grid answer contract using `.` and `*`

## Quality system

Every run now carries both row-level and dataset-level checks.

Row-level checks:
- clue set resolves to exactly one solution

If solved rows later attach a `target`, verification also checks:
- response is parseable
- response matches the hidden solution

Dataset-level checks:
- planned recipe coverage
- planned difficulty coverage
- planned prompt-style coverage
- family-specific coverage, such as clue-kind minimums
- unique prompt coverage

The build writes `plan.json`, and verify writes `dataset_checks.json`.

## Start here

1. Run `sdg/packs/verifiable_reasoning/configs/base.yaml` for English zebra puzzles.
2. Run `sdg/packs/verifiable_reasoning/configs/base_da.yaml` for Danish zebra puzzles.
3. Run `sdg/packs/verifiable_reasoning/configs/futoshiki.yaml` for English futoshiki puzzles.
4. Run `sdg/packs/verifiable_reasoning/configs/futoshiki_da.yaml` for Danish futoshiki puzzles.
5. Run `sdg/packs/verifiable_reasoning/configs/skyscraper.yaml` for English skyscraper puzzles.
6. Run `sdg/packs/verifiable_reasoning/configs/skyscraper_da.yaml` for Danish skyscraper puzzles.
7. Run `sdg/packs/verifiable_reasoning/configs/numbrix.yaml` for English numbrix puzzles.
8. Run `sdg/packs/verifiable_reasoning/configs/numbrix_da.yaml` for Danish numbrix puzzles.
9. Run `sdg/packs/verifiable_reasoning/configs/hitori.yaml` for English hitori puzzles.
10. Run `sdg/packs/verifiable_reasoning/configs/hitori_da.yaml` for Danish hitori puzzles.
11. Run `sdg/packs/verifiable_reasoning/configs/mixed.yaml` for a mixed English/Danish and zebra/lineup dataset.
12. Use `lineup.yaml` or `lineup_da.yaml` if you want only the ordering family.
13. Use `base_solved.yaml`, `futoshiki_solved.yaml`, `skyscraper_solved.yaml`, `numbrix_solved.yaml`, or `hitori_solved.yaml` for English, and the corresponding `_da` config for Danish, if you want the scaffolded `answer_teacher` solve pass enabled during build.

## Dolci inspirations

- `zebralogics`
- `futoshikipuzzle`
- `hitoripuzzle`
- `numbrix`
- `skyscraperpuzzle`
- `sudoku`

The next strong extension is another exact board family such as `sudoku`, built on the same recipe-planning and dataset-check pattern.
