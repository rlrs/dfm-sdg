# Python Algorithms Pack

`python_algorithms` is a starter pack for coding tasks with executable verification.

The goal is to synthesize Python algorithm prompts where the generated solution can be checked with hidden tests, reference implementations, or both. This should be a separate pack because the artifacts and verification loop are code-first rather than text-first.

## Start here

1. Pick one narrow topic such as BFS, two pointers, or basic dynamic programming.
2. Write the reference solution and hidden tests before generating prompt variants.
3. Keep the first row schema simple: prompt, canonical solution, public examples, hidden tests.
4. Add harder transformations like paraphrases or complexity constraints only after the test harness is stable.

## Dolci inspirations

- `allenai/correct-python-sft-187k-decontam-v2_tmp_ids`

This Dolci slice is large but monolithic, so the pack should split it into clear internal slices like graph algorithms, string algorithms, and dynamic programming.

The scaffold build produces starter rows and a guide artifact. It does not implement real code execution yet.
