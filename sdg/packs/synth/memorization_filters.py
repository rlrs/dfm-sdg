from __future__ import annotations

from typing import Any

from sdg.packs.synth.languages import row_uses_cross_language
from sdg.packs.synth.memorization_text import (
    as_list,
    meaningful_tokens,
    normalize_text,
)


def row_evidence_tokens(row: dict[str, Any]) -> set[str]:
    language = _source_language(row)
    evidence_tokens = set(meaningful_tokens(row["hidden"]["sentence"], language=language))
    teacher_bundle = row["hidden"].get("teacher_bundle") or {}

    for sentence in teacher_bundle.get("supporting_claims", []):
        evidence_tokens.update(meaningful_tokens(sentence, language=language))
    for item in teacher_bundle.get("structured_context", []):
        evidence_tokens.update(meaningful_tokens(item, language=language))
    for item in teacher_bundle.get("retrieved_claims", []):
        evidence_tokens.update(meaningful_tokens(item, language=language))
    for item in teacher_bundle.get("retrieved_context", []):
        evidence_tokens.update(meaningful_tokens(item.get("snippet", ""), language=language))
    for source in row.get("sources", []):
        evidence_tokens.update(meaningful_tokens(source.get("snippet", ""), language=language))

    return evidence_tokens


def row_retrieval_grounded(row: dict[str, Any]) -> bool:
    language = _source_language(row)
    sources = row.get("sources", [])
    if not sources:
        return False

    source_ids = {source["source_id"] for source in sources}
    if row["hidden"]["source_id"] in source_ids:
        return True

    teacher_bundle = row["hidden"].get("teacher_bundle") or {}
    teacher_tokens = set(meaningful_tokens(teacher_bundle.get("primary_claim", ""), language=language))
    for sentence in teacher_bundle.get("supporting_claims", []):
        teacher_tokens.update(meaningful_tokens(sentence, language=language))
    for item in teacher_bundle.get("structured_context", []):
        teacher_tokens.update(meaningful_tokens(item, language=language))

    retrieved_tokens = set()
    for source in sources:
        retrieved_tokens.update(meaningful_tokens(source.get("snippet", ""), language=language))

    overlap = teacher_tokens.intersection(retrieved_tokens)
    return len(overlap) >= 3


def row_reasoning_grounded(row: dict[str, Any]) -> bool:
    language = _source_language(row)
    reasoning_tokens = meaningful_tokens(_support_reasoning_text(row), language=language)
    if len(reasoning_tokens) < 6:
        return False

    evidence_tokens = row_evidence_tokens(row)
    overlap = reasoning_tokens.intersection(evidence_tokens)
    if len(overlap) < 3:
        return False

    target_tokens = meaningful_tokens(_support_target_text(row), language=language)
    if not target_tokens:
        return False

    return bool(reasoning_tokens.intersection(target_tokens))


def _support_target_text(row: dict[str, Any]) -> str:
    if not row_uses_cross_language(row):
        return str(row.get("target", "")).strip()

    source_target = str(row["hidden"].get("source_target", "")).strip()
    assert source_target, "cross-language rows must define hidden source_target"
    return source_target


def _support_reasoning_text(row: dict[str, Any]) -> str:
    if not row_uses_cross_language(row):
        return str(row.get("reasoning", "")).strip()

    source_reasoning = str(row["hidden"].get("source_reasoning", "")).strip()
    assert source_reasoning, "cross-language rows must define hidden source_reasoning"
    return source_reasoning


def _source_language(row: dict[str, Any]) -> str:
    meta = row.get("meta", {})
    if isinstance(meta.get("source_language"), str) and meta["source_language"]:
        return meta["source_language"]
    if isinstance(meta.get("language"), str) and meta["language"]:
        return meta["language"]
    return "en"


def row_coverage_supported(row: dict[str, Any]) -> bool:
    language = _source_language(row)
    task_plan = row["hidden"].get("task_plan") or {}
    coverage_points = [point for point in as_list(task_plan.get("coverage_points")) if point]
    if not coverage_points:
        return True

    answer_tokens = meaningful_tokens(_support_target_text(row), language=language)
    if not answer_tokens:
        return False

    supported_points = 0
    for point in coverage_points:
        point_tokens = meaningful_tokens(point, language=language)
        if not point_tokens:
            continue

        overlap = answer_tokens.intersection(point_tokens)
        overlap_ratio = len(overlap) / len(point_tokens)
        if len(overlap) >= 2 or overlap_ratio >= 0.4:
            supported_points += 1

    required_points = 1 if len(coverage_points) <= 2 else 2
    return supported_points >= required_points


def row_answer_supported(row: dict[str, Any]) -> bool:
    language = _source_language(row)
    target_tokens = meaningful_tokens(_support_target_text(row), language=language)
    if not target_tokens:
        return False

    question_type = row.get("meta", {}).get("question_type")
    source_title_tokens = meaningful_tokens(row["hidden"].get("source_title", ""), language=language)
    if question_type in {"reverse_definition", "source_clue"}:
        if source_title_tokens.issubset(target_tokens) and row_retrieval_grounded(row):
            return True

    evidence_tokens = row_evidence_tokens(row)
    if target_tokens.issubset(evidence_tokens):
        return True

    overlap = target_tokens.intersection(evidence_tokens)
    if len(overlap) < 3:
        return False

    overlap_ratio = len(overlap) / len(target_tokens)
    if overlap_ratio >= 0.35:
        return True

    return row_coverage_supported(row) and row_retrieval_grounded(row)


def row_language_quality(row: dict[str, Any]) -> bool:
    if not row_uses_cross_language(row):
        return True
    judge = row.get("scores", {}).get("judge")
    if not judge:
        return False
    return bool(judge.get("language_quality", False))


def row_filter_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not row_retrieval_grounded(row):
        reasons.append("retrieval_not_grounded")
    if not str(row.get("reasoning", "")).strip():
        reasons.append("missing_reasoning")
    if not row_reasoning_grounded(row):
        reasons.append("reasoning_not_grounded")
    if not row_answer_supported(row):
        reasons.append("answer_not_supported")
    if not row_coverage_supported(row):
        reasons.append("coverage_not_supported")
    if normalize_text(row.get("target", "")) in normalize_text(row.get("prompt", "")):
        reasons.append("answer_leaked")
    if not row_language_quality(row):
        reasons.append("language_quality_failed")

    judge = row.get("scores", {}).get("judge")
    if judge and not judge.get("pass", False):
        reasons.append("judge_failed")

    return reasons


def annotate_filter_result(row: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    updated = dict(row)
    hidden = dict(row.get("hidden") or {})
    hidden["generation_filter"] = {
        "passed": not reasons,
        "reasons": list(reasons),
    }
    updated["hidden"] = hidden
    return updated
