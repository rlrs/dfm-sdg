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
    name="tool_use",
    family="tool_use",
    description="Starter scaffold for structured tool-calling synthetic data.",
    goal=(
        "Build synthetic rows where tool selection, argument validity, tool execution, and final "
        "answer grounding can all be checked explicitly."
    ),
    getting_started=(
        "Start with one narrow schema such as calculator or retrieval before multi-tool traces.",
        "Treat argument validation and grounded final answers as first-class checks.",
        "Add repair and retry turns only after the one-tool path is stable.",
    ),
    starter_rows=(
        StarterRowSpec(
            title="Single tool dispatch",
            prompt=(
                "Define one small tool schema and generate rows where the user intent maps cleanly "
                "to exactly one tool call."
            ),
            target=(
                "The first milestone is a row shape that separates user request, assistant tool "
                "call, tool result, and final assistant answer, with deterministic validation for "
                "tool name and argument keys."
            ),
            primary_subset="single_tool_dispatch",
            subset_inspirations=(
                "olmo-toolu-s2-sft-m3",
                "olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated_S2",
            ),
            verification="Validate tool choice and argument schema against the declared tool spec.",
        ),
        StarterRowSpec(
            title="Two-step tool chain",
            prompt=(
                "Add a second stage where the model must call one tool, inspect the returned state, "
                "and then decide whether a second tool call is required."
            ),
            target=(
                "Keep the chain short and explicit. The verifier should reconstruct the expected "
                "call order and ensure each step is licensed by the prior tool result."
            ),
            primary_subset="two_step_tool_chain",
            subset_inspirations=(
                "olmo-toolu-s2-sft-m4v2",
                "olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated_T2",
            ),
            verification="Replay the allowed call sequence and reject unsupported extra calls.",
        ),
        StarterRowSpec(
            title="Result-grounded answer",
            prompt=(
                "Create rows where the tool result contains the facts needed for the final answer, "
                "and the assistant must respond using only those returned facts."
            ),
            target=(
                "The pack should explicitly check that the final answer cites or reflects the tool "
                "output instead of hallucinating unsupported details."
            ),
            primary_subset="result_grounded_answer",
            subset_inspirations=("olmo-toolu-s2-sft-m5v2",),
            verification="Compare the final answer against the tool output and declared grounding rules.",
        ),
        StarterRowSpec(
            title="Deep research routing",
            prompt=(
                "Reserve a later slice for deeper search or browse traces where a planner chooses "
                "sub-queries, gathers evidence, and synthesizes a final answer."
            ),
            target=(
                "Keep this out of the first implementation. It is a good expansion target after the "
                "single-tool and two-step rows are stable and well-validated."
            ),
            primary_subset="deep_research_routing",
            subset_inspirations=("olmo-toolu_deepresearch_no_thinking_DRv4_DS",),
            verification="Check planner structure, evidence coverage, and final answer grounding.",
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
