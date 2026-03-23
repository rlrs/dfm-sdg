from __future__ import annotations

from pathlib import Path
from typing import Any

from sdg.commons import store

_SIMPLE_GROUNDED_QA_QUESTION_TYPES = {
    "fact_disambiguation",
    "fact_verification",
    "factoid_qa",
    "identification",
    "recall",
}


def load_generated_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    outputs_dir = Path(run_dir) / "outputs"
    return _load_rows(
        outputs_dir / "memorization_rows.jsonl",
        outputs_dir / "grounded_qa_rows.jsonl",
    )


def load_verified_rows(run_dir: str | Path) -> list[dict[str, Any]]:
    outputs_dir = Path(run_dir) / "outputs"
    return _load_rows(
        outputs_dir / "memorization_verified.jsonl",
        outputs_dir / "grounded_qa_verified.jsonl",
    )


def materialize_row(row: dict[str, Any]) -> dict[str, Any]:
    family = row_family(row)
    materialized = dict(row)

    hidden = row.get("hidden")
    if isinstance(hidden, dict):
        materialized_hidden = dict(hidden)
        materialized_hidden.pop("required_cited_sources", None)
        generation_filter = hidden.get("generation_filter")
        if isinstance(generation_filter, dict):
            materialized_hidden["generation_filter"] = {
                "reasons": [str(reason) for reason in generation_filter.get("reasons", [])],
            }
        materialized["hidden"] = materialized_hidden

    meta = row.get("meta")
    if isinstance(meta, dict):
        materialized_meta = dict(meta)
        materialized_meta.pop("source_count", None)
        if family == "grounded_qa":
            materialized_meta["required_cited_sources"] = grounded_qa_required_cited_sources_for_row(row)
        materialized["meta"] = materialized_meta

    messages = row_messages(materialized)
    if messages is not None:
        materialized["messages"] = messages

    return materialized


def row_family(row: dict[str, Any]) -> str | None:
    meta = row.get("meta")
    if isinstance(meta, dict):
        family = meta.get("family")
        if isinstance(family, str) and family:
            return family

    row_id = row.get("id")
    if not isinstance(row_id, str):
        return None
    if row_id.startswith("memorization-"):
        return "memorization"
    if row_id.startswith("grounded_qa-"):
        return "grounded_qa"
    return None


def row_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    family = row_family(row)
    if family == "memorization":
        from sdg.packs.synth.gen_memorization import _memorization_messages

        return _memorization_messages(row)
    if family == "grounded_qa":
        from sdg.packs.synth.gen_grounded_qa import _grounded_qa_messages

        return _grounded_qa_messages(row)
    return None


def grounded_qa_required_cited_sources(question_type: object, source_count: int) -> int:
    normalized_question_type = str(question_type or "").strip()
    if source_count < 2:
        return 1
    if normalized_question_type in _SIMPLE_GROUNDED_QA_QUESTION_TYPES:
        return 1
    return 2


def grounded_qa_required_cited_sources_for_row(row: dict[str, Any]) -> int:
    meta = row.get("meta")
    question_type = meta.get("question_type") if isinstance(meta, dict) else None
    return grounded_qa_required_cited_sources(question_type, len(row.get("sources", [])))


def _load_rows(*paths: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(materialize_row(row) for row in store.read_jsonl(path))
    return rows
