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
    name="python_algorithms",
    family="python_algorithms",
    description="Starter scaffold for Python algorithm tasks with executable tests.",
    goal=(
        "Build code-generation rows where every target solution can be checked against hidden "
        "tests or a trusted reference implementation."
    ),
    getting_started=(
        "Start with one topic and a tiny trusted test harness.",
        "Generate prompt variants only after the reference solution and hidden tests are stable.",
        "Treat execution traces, failure cases, and edge-case coverage as part of the row design.",
    ),
    starter_rows=(
        StarterRowSpec(
            title="Array and hashing slice",
            prompt=(
                "Prototype a small coding family with array or hashing tasks where a canonical "
                "solution is easy to explain and easy to test."
            ),
            target=(
                "The first version should publish prompt text, a canonical Python solution, public "
                "examples, and a hidden test bundle that checks edge cases and duplicate handling."
            ),
            primary_subset="array_hashing",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Run the generated solution against hidden unit tests.",
        ),
        StarterRowSpec(
            title="Graph algorithms slice",
            prompt=(
                "Add graph tasks such as BFS or shortest-path traversal where correctness is easier "
                "to verify than stylistic code quality."
            ),
            target=(
                "Use a row shape that keeps algorithm statement, complexity expectation, reference "
                "solution, and hidden tests together so the verifier can execute the code directly."
            ),
            primary_subset="graph_algorithms",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Execute hidden graph cases and compare outputs to the reference solver.",
        ),
        StarterRowSpec(
            title="Dynamic programming slice",
            prompt=(
                "Add one DP family with constrained input sizes and clear recurrence structure so "
                "the test harness can cover both correctness and common off-by-one failures."
            ),
            target=(
                "The pack should separate public examples from hidden adversarial cases and keep a "
                "small reference implementation for expected outputs."
            ),
            primary_subset="dynamic_programming",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Run hidden DP cases and compare exact outputs against the reference.",
        ),
        StarterRowSpec(
            title="String algorithms slice",
            prompt=(
                "Reserve a later slice for string algorithms where prompt wording matters more, "
                "but the first implementation still relies on executable tests instead of judges."
            ),
            target=(
                "Keep this as an expansion path after the simpler array and graph slices are stable. "
                "The important part is to reuse the same executable row contract."
            ),
            primary_subset="string_algorithms",
            subset_inspirations=("allenai/correct-python-sft-187k-decontam-v2_tmp_ids",),
            verification="Execute hidden cases that stress indexing, slicing, and corner cases.",
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
