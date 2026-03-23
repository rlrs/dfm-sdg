from __future__ import annotations

import re
from typing import Any
from xml.etree import ElementTree

from sdg.packs.synth.languages import row_uses_cross_language
from sdg.packs.synth.memorization_text import as_list, meaningful_tokens, normalize_text

STATEMENT_PATTERN = re.compile(r"\s+")


def extract_citation_ids(text: str) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for statement in parse_cited_statements(text):
        for citation in statement["citations"]:
            if citation in seen:
                continue
            seen.add(citation)
            citations.append(citation)
    return citations


def extract_free_text_segments(text: str) -> list[str]:
    raw = str(text).strip()
    assert raw, "cited answer text must not be empty"
    root = ElementTree.fromstring(f"<statements>{raw}</statements>")

    segments: list[str] = []
    head = STATEMENT_PATTERN.sub(" ", str(root.text or "").strip())
    if _has_visible_text(head):
        segments.append(head)

    for node in root:
        tail = STATEMENT_PATTERN.sub(" ", str(node.tail or "").strip())
        if _has_visible_text(tail):
            segments.append(tail)

    return segments


def parse_cited_statements(text: str) -> list[dict[str, Any]]:
    raw = str(text).strip()
    assert raw, "cited answer text must not be empty"
    root = ElementTree.fromstring(f"<statements>{raw}</statements>")

    statements: list[dict[str, Any]] = []
    for node in root:
        assert node.tag == "statement", "cited answer children must be <statement>"
        cites = str(node.attrib.get("cites", "")).split()
        assert cites, "statement must define cites"
        body = STATEMENT_PATTERN.sub(" ", "".join(node.itertext()).strip())
        assert body, "statement text must not be empty"
        statements.append({"text": body, "citations": cites})

    assert statements, "cited answer must contain at least one statement"
    return statements


def row_retrieval_grounded(row: dict[str, Any]) -> bool:
    sources = row.get("sources", [])
    if not sources:
        return False

    retrieved_source_ids = {source["source_id"] for source in sources}
    expected_source_ids = list(_expected_source_ids(row))
    if not expected_source_ids:
        return False
    if expected_source_ids[0] not in retrieved_source_ids:
        return False
    if len(expected_source_ids) == 1:
        return True
    return len(retrieved_source_ids.intersection(expected_source_ids)) >= 2


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


def row_answer_supported(row: dict[str, Any]) -> bool:
    language = _source_language(row)
    target_tokens = meaningful_tokens(_support_target_text(row), language=language)
    if not target_tokens:
        return False

    evidence_tokens = row_evidence_tokens(row)
    if target_tokens.issubset(evidence_tokens):
        return True

    overlap = target_tokens.intersection(evidence_tokens)
    if len(overlap) < 3:
        return False

    if len(overlap) / len(target_tokens) >= 0.35:
        return True

    return row_coverage_supported(row) and row_retrieval_grounded(row)


def row_coverage_supported(row: dict[str, Any]) -> bool:
    language = _source_language(row)
    coverage_points = [point for point in as_list(row["hidden"]["task_plan"].get("coverage_points")) if point]
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
        if len(overlap) >= 2 or len(overlap) / len(point_tokens) >= 0.4:
            supported_points += 1

    required_points = 1 if len(coverage_points) <= 2 else 2
    return supported_points >= required_points


def row_has_citations(row: dict[str, Any]) -> bool:
    target_statements = parse_cited_statements(str(row.get("target", "")))
    source_statements = parse_cited_statements(str(row["hidden"].get("source_target", "")))
    if len(target_statements) != len(source_statements):
        return False

    for target_statement, source_statement in zip(target_statements, source_statements, strict=True):
        if target_statement["citations"] != source_statement["citations"]:
            return False

    citation_ids = extract_citation_ids(str(row.get("target", "")))
    valid_ids = {source["citation_id"] for source in row.get("sources", [])}
    return set(citation_ids).issubset(valid_ids)


def citation_support_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    if not row_has_citations(row):
        return {
            "ok": False,
            "has_citations": False,
            "supported_statements": 0,
            "total_statements": 0,
            "issues": ["invalid_statement_citations"],
            "statements": [],
        }

    language = _source_language(row)
    source_lookup = {source["citation_id"]: source for source in row.get("sources", [])}
    diagnostics: list[dict[str, Any]] = []
    supported = 0

    for statement in parse_cited_statements(str(row["hidden"]["source_target"])):
        statement_tokens = meaningful_tokens(statement["text"], language=language)
        cited_tokens: set[str] = set()
        for citation_id in statement["citations"]:
            source = source_lookup.get(citation_id)
            assert source is not None, f"missing cited source {citation_id}"
            cited_tokens.update(meaningful_tokens(source.get("snippet", ""), language=language))

        overlap = statement_tokens.intersection(cited_tokens)
        statement_supported = _statement_supported(statement_tokens, overlap)
        if statement_supported:
            supported += 1

        diagnostics.append(
            {
                "text": statement["text"],
                "citations": list(statement["citations"]),
                "supported": statement_supported,
                "token_count": len(statement_tokens),
                "overlap_count": len(overlap),
                "overlap_ratio": _overlap_ratio(statement_tokens, overlap),
            }
        )

    issues = [
        f"unsupported_statement_{index + 1}"
        for index, statement in enumerate(diagnostics)
        if not statement["supported"]
    ]
    return {
        "ok": supported == len(diagnostics),
        "has_citations": True,
        "supported_statements": supported,
        "total_statements": len(diagnostics),
        "issues": issues,
        "statements": diagnostics,
    }


def row_citation_supported(row: dict[str, Any]) -> bool:
    return bool(citation_support_diagnostics(row)["ok"])


def row_language_quality(row: dict[str, Any]) -> bool:
    if not row_uses_cross_language(row):
        return True
    judge = row.get("scores", {}).get("judge")
    if not judge:
        return False
    return bool(judge.get("language_quality", False))


def row_filter_reasons(row: dict[str, Any]) -> list[str]:
    if row.get("hidden", {}).get("generation_error"):
        return ["generation_failed"]

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


def row_evidence_tokens(row: dict[str, Any]) -> set[str]:
    language = _source_language(row)
    evidence_tokens: set[str] = set()
    teacher_bundle = row["hidden"].get("teacher_bundle") or {}
    evidence_tokens.update(meaningful_tokens(teacher_bundle.get("primary_claim", ""), language=language))

    for sentence in teacher_bundle.get("supporting_claims", []):
        evidence_tokens.update(meaningful_tokens(sentence, language=language))
    for item in teacher_bundle.get("structured_context", []):
        evidence_tokens.update(meaningful_tokens(item, language=language))
    for source in teacher_bundle.get("bridge_sources", []):
        evidence_tokens.update(meaningful_tokens(source.get("snippet", ""), language=language))
    for source in row.get("sources", []):
        evidence_tokens.update(meaningful_tokens(source.get("snippet", ""), language=language))
    return evidence_tokens


def _support_target_text(row: dict[str, Any]) -> str:
    source_target = str(row["hidden"].get("source_target", "")).strip()
    assert source_target, "grounded_qa rows must define hidden source_target"
    return " ".join(statement["text"] for statement in parse_cited_statements(source_target))


def _statement_supported(statement_tokens: set[str], overlap: set[str]) -> bool:
    token_count = len(statement_tokens)
    if token_count == 0:
        return False
    if overlap == statement_tokens:
        return True
    if token_count <= 2:
        return len(overlap) == token_count
    if token_count <= 4:
        return len(overlap) >= max(2, token_count - 1)
    return len(overlap) >= 3 and _overlap_ratio(statement_tokens, overlap) >= 0.3


def _overlap_ratio(statement_tokens: set[str], overlap: set[str]) -> float:
    if not statement_tokens:
        return 0.0
    return len(overlap) / len(statement_tokens)


def _has_visible_text(text: str) -> bool:
    return any(char.isalnum() for char in text)


def _support_reasoning_text(row: dict[str, Any]) -> str:
    if not row_uses_cross_language(row):
        return str(row.get("reasoning", "")).strip()

    source_reasoning = str(row["hidden"].get("source_reasoning", "")).strip()
    assert source_reasoning, "cross-language grounded_qa rows must define hidden source_reasoning"
    return source_reasoning


def _expected_source_ids(row: dict[str, Any]) -> list[str]:
    expected = row["hidden"].get("expected_source_ids")
    assert isinstance(expected, list) and expected, "grounded_qa rows must define expected_source_ids"
    return [str(item) for item in expected]


def _source_language(row: dict[str, Any]) -> str:
    meta = row.get("meta", {})
    if isinstance(meta.get("source_language"), str) and meta["source_language"]:
        return meta["source_language"]
    if isinstance(meta.get("language"), str) and meta["language"]:
        return meta["language"]
    return "en"
