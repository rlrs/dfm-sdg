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
- `knightsandknaves_logic`
  - bilingual English/Danish
  - two-, three-, four-, and five-speaker dialogue puzzles
  - exact role-assignment solver over direct claims, count statements, relation statements, conditional statements, and quoted-speech statements
  - role-assignment answer contract
- `jugpuzzle_logic`
  - bilingual English/Danish
  - shortest-sequence water-jug puzzles with fill, empty, and pour actions
  - exact BFS solver over jug states with unique optimal solutions
  - action-sequence answer contract
- `countdownequal_logic`
  - bilingual English/Danish
  - shortest-expression arithmetic puzzles over a small number set and target
  - exact subset-expression solver with `+`, `-`, `*`, and integer `/`
  - expression-string answer contract
- `setsplitting_logic`
  - bilingual English/Danish
  - exact two-group split puzzles over small element sets
  - unique split under the convention that element 1 is fixed in group A
  - group-assignment answer contract
- `cryptarithmetic_logic`
  - bilingual English/Danish
  - exact unknown-base digit-system equations inspired by Dolci
  - two- and three-addend addition with `d[i]` symbols
  - digit-sequence answer contract
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
- `kakurasu_logic`
  - bilingual English/Danish
  - 4x4, 5x5, and 6x6 weighted-sum board puzzles
  - row and column sum clues presented as arrays `A` and `B`
  - binary-grid answer contract using `0` and `1`
- `lightuppuzzle_logic`
  - bilingual English/Danish
  - 5x5 and 6x6 Akari / Light Up boards with walls and numbered walls
  - exact lamp-placement verification with illumination and line-of-sight constraints
  - annotated-grid answer contract using `#`, digits, `.`, and `*`
- `blocked_star_logic`
  - bilingual English/Danish
  - rectangular and square blocked-cell star placement puzzles
  - one star per row, at most one per column, and no 8-neighbor adjacency
  - annotated-grid answer contract using `X`, `.`, and `*`
- `starbattle_logic`
  - bilingual English/Danish
  - classic region-based 5x5 and 6x6 star battle puzzles
  - exactly one star per row, column, and region, with no 8-neighbor adjacency
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
3. Run `sdg/packs/verifiable_reasoning/configs/knightsandknaves.yaml` for English knights-and-knaves puzzles.
4. Run `sdg/packs/verifiable_reasoning/configs/knightsandknaves_da.yaml` for Danish knights-and-knaves puzzles.
5. Run `sdg/packs/verifiable_reasoning/configs/cryptarithmetic.yaml` for English cryptarithmetic puzzles.
6. Run `sdg/packs/verifiable_reasoning/configs/cryptarithmetic_da.yaml` for Danish cryptarithmetic puzzles.
7. Run `sdg/packs/verifiable_reasoning/configs/jugpuzzle.yaml` for English jug puzzles.
8. Run `sdg/packs/verifiable_reasoning/configs/jugpuzzle_da.yaml` for Danish jug puzzles.
9. Run `sdg/packs/verifiable_reasoning/configs/countdownequal.yaml` for English countdown-equal puzzles.
10. Run `sdg/packs/verifiable_reasoning/configs/countdownequal_da.yaml` for Danish countdown-equal puzzles.
11. Run `sdg/packs/verifiable_reasoning/configs/setsplitting.yaml` for English set-splitting puzzles.
12. Run `sdg/packs/verifiable_reasoning/configs/setsplitting_da.yaml` for Danish set-splitting puzzles.
13. Run `sdg/packs/verifiable_reasoning/configs/futoshiki.yaml` for English futoshiki puzzles.
14. Run `sdg/packs/verifiable_reasoning/configs/futoshiki_da.yaml` for Danish futoshiki puzzles.
15. Run `sdg/packs/verifiable_reasoning/configs/skyscraper.yaml` for English skyscraper puzzles.
16. Run `sdg/packs/verifiable_reasoning/configs/skyscraper_da.yaml` for Danish skyscraper puzzles.
17. Run `sdg/packs/verifiable_reasoning/configs/numbrix.yaml` for English numbrix puzzles.
18. Run `sdg/packs/verifiable_reasoning/configs/numbrix_da.yaml` for Danish numbrix puzzles.
19. Run `sdg/packs/verifiable_reasoning/configs/hitori.yaml` for English hitori puzzles.
20. Run `sdg/packs/verifiable_reasoning/configs/hitori_da.yaml` for Danish hitori puzzles.
21. Run `sdg/packs/verifiable_reasoning/configs/kakurasu.yaml` for English kakurasu puzzles.
22. Run `sdg/packs/verifiable_reasoning/configs/kakurasu_da.yaml` for Danish kakurasu puzzles.
23. Run `sdg/packs/verifiable_reasoning/configs/lightuppuzzle.yaml` for English Light Up puzzles.
24. Run `sdg/packs/verifiable_reasoning/configs/lightuppuzzle_da.yaml` for Danish Light Up puzzles.
25. Run `sdg/packs/verifiable_reasoning/configs/blocked_star.yaml` for English blocked-star puzzles.
26. Run `sdg/packs/verifiable_reasoning/configs/blocked_star_da.yaml` for Danish blocked-star puzzles.
27. Run `sdg/packs/verifiable_reasoning/configs/starbattle.yaml` for English star battle puzzles.
28. Run `sdg/packs/verifiable_reasoning/configs/starbattle_da.yaml` for Danish star battle puzzles.
29. Run `sdg/packs/verifiable_reasoning/configs/mixed.yaml` for a mixed English/Danish and zebra/lineup dataset.
30. Use `lineup.yaml` or `lineup_da.yaml` if you want only the ordering family.
31. Use `base_solved.yaml`, `knightsandknaves_solved.yaml`, `cryptarithmetic_solved.yaml`, `jugpuzzle_solved.yaml`, `countdownequal_solved.yaml`, `setsplitting_solved.yaml`, `futoshiki_solved.yaml`, `skyscraper_solved.yaml`, `numbrix_solved.yaml`, `hitori_solved.yaml`, `kakurasu_solved.yaml`, `blocked_star_solved.yaml`, or `starbattle_solved.yaml` for English, and the corresponding `_da` config for Danish, if you want the scaffolded `answer_teacher` solve pass enabled during build.
32. `lightuppuzzle_logic` is currently prompt-only. It stays verifier-backed, but `answer_teacher` attachment is disabled for that family for now.
33. Use `all.yaml`, `all_da.yaml`, or `all_mixed.yaml` to build across all currently included families. These deliberately exclude `lightuppuzzle_logic`.

## Dolci inspirations

- `zebralogics`
- `futoshikipuzzle`
- `hitoripuzzle`
- `kakurasu`
- `knightsandknaves`
- `jugpuzzle`
- `countdownequal`
- `setsplitting`
- `cryptarithmetic`
- `lightuppuzzle`
- `numbrix`
- `skyscraperpuzzle`
- `starbattle`

## Scaffolded next families

- `minesweeping_logic`
  - scaffolded in `minesweeping.py`
  - next step: clue-board Minesweeper puzzles with exact mine-mask output
