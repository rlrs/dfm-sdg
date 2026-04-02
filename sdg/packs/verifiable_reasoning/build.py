from typing import Any

from sdg.commons.starter_pack import (
    StarterPackSpec,
    StarterRowSpec,
    build_starter_pack,
    publish_starter_pack,
    summarize_starter_pack,
    verify_starter_pack,
)

SPEC = StarterPackSpec(
    name="verifiable_reasoning",
    family="verifiable_reasoning",
    description="Starter scaffold for exactly checkable synthetic reasoning tasks.",
    goal=(
        "Build narrow reasoning families with hidden state and deterministic verification "
        "before scaling to broader synthetic mixes."
    ),
    getting_started=(
        "Start with logic_puzzles and keep the first family solver-backed.",
        "Make the prompt formatter and verifier agree on one explicit row schema.",
        "Add harder families only after one small family publishes clean train and eval splits.",
    ),
    starter_rows=(
        StarterRowSpec(
            title="Logic puzzle pilot",
            prompt=(
                "Start the pack with a logic puzzle family such as Kakurasu or Futoshiki. "
                "Generate one instance, keep the solved board in hidden state, and expose only "
                "the natural-language puzzle text to the model."
            ),
            target=(
                "Success looks like a solver-backed row format with exact verification: prompt, "
                "target solution, hidden canonical state, and a verifier that can re-solve the "
                "instance from the prompt alone."
            ),
            primary_subset="logic_puzzles",
            subset_inspirations=(
                "kakurasu",
                "futoshikipuzzle",
                "hitoripuzzle",
                "numbrix",
                "skyscraperpuzzle",
                "starbattle",
                "survopuzzle",
                "sudoku",
                "zebralogics",
            ),
            verification="Exact solver check against the canonical hidden solution.",
        ),
        StarterRowSpec(
            title="Graph path family",
            prompt=(
                "Add a small graph reasoning family where the hidden state is a generated graph "
                "instance and the visible prompt asks for a shortest path, traversal order, or "
                "feasible pipeline arrangement."
            ),
            target=(
                "The first graph family should publish rows with machine-checkable answers, such "
                "as exact shortest-path cost, canonical node order, or a verifier that validates "
                "the returned arrangement against graph constraints."
            ),
            primary_subset="graph_paths",
            subset_inspirations=(
                "gridbfs",
                "pipelinearrangement",
                "shortestpath",
                "topologicalsort",
            ),
            verification="Recompute the expected answer directly from the hidden graph state.",
        ),
        StarterRowSpec(
            title="Number theory slice",
            prompt=(
                "Prototype a symbolic math family where each row is derived from structured "
                "integer state and the answer can be checked by recomputation."
            ),
            target=(
                "Keep the first slice narrow: one template, one solver, one verifier. Prefer "
                "families where exact equality is enough, such as CRT, Bezout identity, or "
                "Euclid-game style outputs."
            ),
            primary_subset="number_theory",
            subset_inspirations=(
                "bezoutidentity",
                "crt",
                "discretelogarithm",
                "euclidgame",
            ),
            verification="Re-run the symbolic computation and compare against the returned answer.",
        ),
        StarterRowSpec(
            title="Counting and DP slice",
            prompt=(
                "Add one counting family where the hidden instance can be solved with a reference "
                "dynamic program and the visible prompt asks for the final count only."
            ),
            target=(
                "The pack should treat counting families as the second wave after logic puzzles: "
                "they are still exactly checkable, but the verifier should come from a trusted "
                "reference implementation rather than heuristics."
            ),
            primary_subset="counting_dp",
            subset_inspirations=(
                "gcdlcmcounting",
                "palindromepartitioncounting",
                "stirlingsecond",
                "subsetsumsequence",
            ),
            verification="Run the reference DP and compare the final numeric answer.",
        ),
    ),
)


def build(cfg: dict[str, Any]):
    return build_starter_pack(cfg, spec=SPEC)


def verify(run_id_or_path: str):
    return verify_starter_pack(run_id_or_path, spec=SPEC)


def summarize(run_id_or_path: str):
    return summarize_starter_pack(run_id_or_path)


def publish(run_id_or_path: str, out_dir: str | None = None):
    return publish_starter_pack(run_id_or_path, out_dir=out_dir)
