from __future__ import annotations

from pathlib import Path
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons.run import load, run
from sdg.commons.store import read_jsonl
from sdg.commons.utils import read_json, write_json
from sdg.packs.pleias_synth.build_memory_core import build_memory_core
from sdg.packs.pleias_synth.export import (
    load_existing_quality,
    publish_generated_dataset,
    publish_run,
)
from sdg.packs.pleias_synth.gen_memorization import generate_memorization
from sdg.packs.pleias_synth.rows import load_generated_rows
from sdg.packs.pleias_synth.verify import verify_memorization, verify_memory_core


def build(cfg: dict[str, Any]) -> BuildResult:
    """Build the initial PleIAs memory core artifacts."""

    return run(
        _build_run,
        pack="pleias_synth",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    """Verify the memory-core artifacts for one completed run."""

    result = load(run_id_or_path)
    chunks = read_jsonl(result.artifacts["memory_chunks"].path)
    source_table = read_jsonl(result.artifacts["source_table"].path)
    index = read_json(result.artifacts["retrieval_index"].path)

    metrics, failure_summary = verify_memory_core(chunks, source_table, index)
    outputs_dir = Path(result.run_dir) / "outputs"
    all_metrics: dict[str, Any] = {"memory_core": metrics}
    all_failures: dict[str, Any] = {"memory_core": failure_summary}

    if "memorization_rows" in result.artifacts:
        memorization_rows = store.read_jsonl(result.artifacts["memorization_rows"].path)
        verified_rows, memorization_metrics, memorization_failures = verify_memorization(memorization_rows)
        store.write_jsonl(verified_rows, outputs_dir / "memorization_verified.jsonl")
        store.write_jsonl(
            [row for row in verified_rows if any(not passed for passed in row["checks"].values())],
            outputs_dir / "memorization_failures.jsonl",
        )
        all_metrics["memorization"] = memorization_metrics
        all_failures["memorization"] = memorization_failures

    write_json(all_metrics, outputs_dir / "metrics.json")
    write_json(all_failures, outputs_dir / "failure_summary.json")

    return {
        "run_id": result.run_id,
        "metrics": all_metrics,
        "failure_summary": all_failures,
    }


def summarize(run_id_or_path: str) -> dict[str, Any]:
    """Summarize the current state of a PleIAs memory-core run."""

    result = load(run_id_or_path)
    chunks = read_jsonl(result.artifacts["memory_chunks"].path)
    source_table = read_jsonl(result.artifacts["source_table"].path)
    generated_rows = load_generated_rows(result.run_dir)
    metrics, failure_summary = load_existing_quality(result)

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "sources": len(source_table),
        "chunks": len(chunks),
        "generated_rows": len(generated_rows),
        "metrics": metrics,
        "failure_summary": failure_summary,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    """Publish the current PleIAs memory-core artifacts and reports."""

    result = load(run_id_or_path)
    metrics, failure_summary = load_existing_quality(result)
    if not metrics and not failure_summary:
        verification = verify(run_id_or_path)
        metrics = verification["metrics"]
        failure_summary = verification["failure_summary"]

    publication = publish_run(result, metrics=metrics, failure_summary=failure_summary, out_dir=out_dir)
    dataset_stats = publish_generated_dataset(result, Path(publication["out_dir"]))
    publication.update(dataset_stats)
    return publication


def _build_run(
    *,
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
) -> dict[str, Artifact]:
    memory = build_memory_core(cfg, outputs_dir)
    artifacts = dict(memory["artifacts"])

    generation = cfg["generation"]
    families = {str(item) for item in generation["families"]}
    memorization_stats: dict[str, Any] = {}
    if "memorization" in families:
        memorization_artifacts, memorization_stats = generate_memorization(
            cfg,
            memory,
            outputs_dir,
            seed=seed,
        )
        artifacts.update(memorization_artifacts)

    write_json(
        {
            "memory_core": {
                "sources": len(memory["source_table"]),
                "chunks": len(memory["chunks"]),
                "index_type": memory["index"]["type"],
            },
            "memorization": {
                "rows": memorization_stats["rows"],
            }
            if memorization_stats
            else {},
        },
        outputs_dir / "metrics.json",
    )
    return artifacts
