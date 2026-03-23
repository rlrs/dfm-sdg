from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons.run import load, run
from sdg.commons.run_log import log_event
from sdg.commons.store import read_jsonl
from sdg.commons.utils import read_json, write_json
from sdg.packs.synth.build_memory_core import build_memory_core, clean_corpus
from sdg.packs.synth.export import (
    load_existing_quality,
    publish_generated_dataset,
    publish_run,
)
from sdg.packs.synth.gen_grounded_qa import generate_grounded_qa
from sdg.packs.synth.gen_memorization import generate_memorization
from sdg.packs.synth.rows import load_generated_rows
from sdg.packs.synth.sources import load_sources
from sdg.packs.synth.verify import (
    verify_grounded_qa,
    verify_memorization,
    verify_memory_core,
)


def build(cfg: dict[str, Any]) -> BuildResult:
    """Build the initial memory core artifacts."""

    return run(
        _build_run,
        pack="synth",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
        resume_incomplete=cfg.get("resume_incomplete", True),
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

    if "grounded_qa_rows" in result.artifacts:
        grounded_qa_rows = store.read_jsonl(result.artifacts["grounded_qa_rows"].path)
        verified_rows, grounded_qa_metrics, grounded_qa_failures = verify_grounded_qa(grounded_qa_rows)
        store.write_jsonl(verified_rows, outputs_dir / "grounded_qa_verified.jsonl")
        store.write_jsonl(
            [row for row in verified_rows if any(not passed for passed in row["checks"].values())],
            outputs_dir / "grounded_qa_failures.jsonl",
        )
        all_metrics["grounded_qa"] = grounded_qa_metrics
        all_failures["grounded_qa"] = grounded_qa_failures

    write_json(all_metrics, outputs_dir / "metrics.json")
    write_json(all_failures, outputs_dir / "failure_summary.json")

    return {
        "run_id": result.run_id,
        "metrics": all_metrics,
        "failure_summary": all_failures,
    }


def summarize(run_id_or_path: str) -> dict[str, Any]:
    """Summarize the current state of a memory-core run."""

    result = load(run_id_or_path)
    outputs_dir = Path(result.run_dir) / "outputs"
    chunks = _read_jsonl(_output_path(result, "memory_chunks", "memory_chunks.jsonl"))
    source_table = _read_jsonl(_output_path(result, "source_table", "source_table.jsonl"))
    generated_rows = load_generated_rows(result.run_dir)
    metrics, failure_summary = load_existing_quality(result)

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "sources": len(source_table),
        "chunks": len(chunks),
        "generated_rows": len(generated_rows),
        "artifacts": sorted(result.artifacts),
        "memorization": _summarize_memorization_outputs(outputs_dir),
        "grounded_qa": _summarize_grounded_qa_outputs(outputs_dir),
        "progress": _default_progress(outputs_dir),
        "family_progress": _family_progress(outputs_dir),
        "model_metrics": _read_optional_json(outputs_dir / "model_metrics.json"),
        "llm_json_metrics": _read_optional_json(outputs_dir / "llm_json_metrics.json"),
        "metrics": metrics,
        "failure_summary": failure_summary,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    """Publish the current memory-core artifacts and reports."""

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
    memory = _load_or_build_memory(cfg, outputs_dir)
    artifacts = dict(memory["artifacts"])

    generation = cfg["generation"]
    families = {str(item) for item in generation["families"]}
    memorization_stats: dict[str, Any] = {}
    grounded_qa_stats: dict[str, Any] = {}
    if "memorization" in families:
        memorization_artifacts, memorization_stats = generate_memorization(
            cfg,
            memory,
            outputs_dir,
            seed=seed,
        )
        artifacts.update(memorization_artifacts)
    if "grounded_qa" in families:
        grounded_qa_artifacts, grounded_qa_stats = generate_grounded_qa(
            cfg,
            memory,
            outputs_dir,
            seed=seed,
        )
        artifacts.update(grounded_qa_artifacts)

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
            "grounded_qa": {
                "rows": grounded_qa_stats["rows"],
            }
            if grounded_qa_stats
            else {},
        },
        outputs_dir / "metrics.json",
    )
    return artifacts


def _load_or_build_memory(cfg: dict[str, Any], outputs_dir: Path) -> dict[str, Any]:
    existing = _load_existing_memory(cfg, outputs_dir)
    if existing is not None:
        log_event(
            "synth",
            "memory_core_resumed",
            sources=len(existing["source_table"]),
            chunks=len(existing["chunks"]),
            index_type=existing["index"]["type"],
        )
        return existing

    log_event("synth", "memory_core_started")
    memory = build_memory_core(cfg, outputs_dir)
    log_event(
        "synth",
        "memory_core_completed",
        sources=len(memory["source_table"]),
        chunks=len(memory["chunks"]),
        index_type=memory["index"]["type"],
    )
    return memory


def _load_existing_memory(cfg: dict[str, Any], outputs_dir: Path) -> dict[str, Any] | None:
    chunks_path = outputs_dir / "memory_chunks.jsonl"
    source_table_path = outputs_dir / "source_table.jsonl"
    index_path = outputs_dir / "retrieval_index.json"
    manifest_path = outputs_dir / "memory_manifest.json"
    if not chunks_path.exists() or not source_table_path.exists() or not index_path.exists() or not manifest_path.exists():
        return None

    docs = clean_corpus(load_sources(cfg))
    chunks = store.read_jsonl(chunks_path)
    source_table = store.read_jsonl(source_table_path)
    index = read_json(index_path)
    return {
        "docs": docs,
        "sections": [],
        "chunks": chunks,
        "index": index,
        "source_table": source_table,
        "artifacts": {
            "memory_chunks": Artifact(
                name="memory_chunks",
                path=str(chunks_path),
                kind="jsonl",
                meta={"rows": len(chunks)},
            ),
            "source_table": Artifact(
                name="source_table",
                path=str(source_table_path),
                kind="jsonl",
                meta={"rows": len(source_table)},
            ),
            "retrieval_index": Artifact(
                name="retrieval_index",
                path=str(index_path),
                kind="blob",
                meta={"type": index["type"]},
            ),
            "memory_manifest": Artifact(
                name="memory_manifest",
                path=str(manifest_path),
                kind="blob",
                meta={"source_count": len(source_table), "chunk_count": len(chunks)},
            ),
        },
    }


def _summarize_memorization_outputs(outputs_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(outputs_dir / "memorization_rows.jsonl")
    candidates = _read_jsonl(outputs_dir / "memorization_candidates.jsonl")
    rejected = _read_jsonl(outputs_dir / "memorization_rejected.jsonl")

    return {
        "candidate_rows": len(candidates),
        "rows": len(rows),
        "rejected_rows": len(rejected),
        "question_types": _count_values(rows, lambda row: row["meta"].get("question_type")),
        "query_angles": _count_values(rows, lambda row: row["hidden"].get("query_angle")),
        "source_titles": _count_values(rows, lambda row: row["hidden"].get("source_title"), limit=10),
        "reject_reasons": _count_reject_reasons(rejected),
        "kept_preview": _preview_rows(outputs_dir / "memorization_preview.jsonl"),
        "rejected_preview": _preview_rows(outputs_dir / "memorization_rejected_preview.jsonl"),
    }


def _summarize_grounded_qa_outputs(outputs_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(outputs_dir / "grounded_qa_rows.jsonl")
    candidates = _read_jsonl(outputs_dir / "grounded_qa_candidates.jsonl")
    rejected = _read_jsonl(outputs_dir / "grounded_qa_rejected.jsonl")

    return {
        "candidate_rows": len(candidates),
        "rows": len(rows),
        "rejected_rows": len(rejected),
        "question_types": _count_values(rows, lambda row: row["meta"].get("question_type")),
        "query_angles": _count_values(rows, lambda row: row["hidden"].get("query_angle")),
        "source_titles": _count_values(rows, lambda row: row["hidden"].get("source_title"), limit=10),
        "source_counts": _count_values(rows, lambda row: len(row.get("sources", []))),
        "reject_reasons": _count_reject_reasons(rejected),
        "kept_preview": _preview_rows(outputs_dir / "grounded_qa_preview.jsonl"),
        "rejected_preview": _preview_rows(outputs_dir / "grounded_qa_rejected_preview.jsonl"),
    }


def _count_values(
    rows: list[dict[str, Any]],
    value_for: Callable[[dict[str, Any]], Any],
    *,
    limit: int | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = value_for(row)
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    ordered = dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    if limit is None:
        return ordered
    return dict(list(ordered.items())[:limit])


def _count_reject_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reasons = row.get("hidden", {}).get("generation_filter", {}).get("reasons", [])
        for reason in reasons:
            key = str(reason)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _preview_rows(path: Path, *, limit: int = 3) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    preview: list[dict[str, Any]] = []
    for row in rows[:limit]:
        preview.append(
            {
                "id": row.get("id"),
                "source_title": row.get("hidden", {}).get("source_title"),
                "question_type": row.get("meta", {}).get("question_type"),
                "query_angle": row.get("hidden", {}).get("query_angle"),
                "prompt": row.get("prompt"),
                "target": row.get("target"),
                "reasons": row.get("hidden", {}).get("generation_filter", {}).get("reasons", []),
            }
        )
    return preview


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return store.read_jsonl(path)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _family_progress(outputs_dir: Path) -> dict[str, Any]:
    progress: dict[str, Any] = {}
    memorization = _read_optional_json(outputs_dir / "memorization_progress.json")
    if memorization is not None:
        progress["memorization"] = memorization
    grounded_qa = _read_optional_json(outputs_dir / "grounded_qa_progress.json")
    if grounded_qa is not None:
        progress["grounded_qa"] = grounded_qa
    return progress


def _default_progress(outputs_dir: Path) -> dict[str, Any] | None:
    memorization = _read_optional_json(outputs_dir / "memorization_progress.json")
    if memorization is not None:
        return memorization
    return _read_optional_json(outputs_dir / "grounded_qa_progress.json")


def _output_path(result: BuildResult, artifact_name: str, filename: str) -> Path:
    artifact = result.artifacts.get(artifact_name)
    if artifact is not None:
        return Path(artifact.path)
    return Path(result.run_dir) / "outputs" / filename
