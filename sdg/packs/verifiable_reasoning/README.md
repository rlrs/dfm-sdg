# Verifiable Reasoning Pack

`verifiable_reasoning` is a starter pack for synthetic tasks with exact or near-exact verification.

The goal is to build small reasoning families where generation, hidden state, and checking all stay under our control. This is the strongest candidate for a full pack because the verification contract is clear from the start.

## Start here

1. Start with `logic_puzzles` as the first slice of the pack.
2. Pick one family such as `kakurasu`, `futoshikipuzzle`, `numbrix`, or `sudoku`.
3. Implement four small pieces in order: instance generator, canonical solver, prompt formatter, and exact verifier.
4. Once one family is stable, add a second family that reuses the same row shape and publish path.

## Dolci inspirations

- logical puzzle families like `kakurasu`, `futoshikipuzzle`, `hitoripuzzle`, `numbrix`, `skyscraperpuzzle`, `starbattle`, `survopuzzle`, `sudoku`, and `zebralogics`
- graph and routing families like `gridbfs`, `pipelinearrangement`, `shortestpath`, and `topologicalsort`
- symbolic math and number theory families like `bezoutidentity`, `crt`, `discretelogarithm`, and `euclidgame`

The scaffold build produces starter rows and a guide artifact. It does not implement real generation yet.
