# Verifiable Reasoning Pack

`verifiable_reasoning` now uses an explicit recipe-planning system instead of relying on ad hoc randomness.

Each family defines a small recipe catalog. A build first creates a balanced `plan.json`, then generates rows from those planned recipes, and finally verifies both row-level correctness and dataset-level coverage. That gives a reusable pattern for future exact families: declare the family variants you want, then fail the run if the generated dataset does not actually cover them.

## Current families

- `zebra_logic`
  - bilingual English/Danish
  - four-house zebra puzzles
  - multiple prompt styles, axis profiles, clue profiles, and difficulty tiers
  - structured house-table targets
- `lineup_logic`
  - bilingual English/Danish
  - queue, chair, and ranking scenarios
  - four-person and five-person variants
  - broader ordering clue types, including `between`, `not_adjacent`, and `one_between`

## Quality system

Every run now carries both row-level and dataset-level checks.

Row-level checks:
- answer is parseable
- answer matches the hidden solution
- clue set resolves to exactly one solution

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
3. Run `sdg/packs/verifiable_reasoning/configs/mixed.yaml` for a mixed English/Danish and zebra/lineup dataset.
4. Use `lineup.yaml` or `lineup_da.yaml` if you want only the ordering family.

## Dolci inspirations

- `zebralogics`
- `futoshikipuzzle`
- `hitoripuzzle`
- `numbrix`
- `skyscraperpuzzle`
- `sudoku`

The next strong extension is another exact board family such as `futoshiki` or `skyscraper`, built on the same recipe-planning and dataset-check pattern.
