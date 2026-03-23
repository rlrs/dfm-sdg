from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterable, Iterator
from random import Random
from typing import Any, TextIO, TypedDict
from xml.sax.saxutils import escape

from sdg.commons import Artifact, store
from sdg.commons import publish as common_publish
from sdg.commons.model import LLM, load_clients
from sdg.commons.run_log import log_event, write_snapshot
from sdg.commons.work_queue import map_async_unordered
from sdg.packs.synth.grounded_qa_filters import (
    annotate_filter_result,
    citation_support_diagnostics,
    extract_citation_ids,
    extract_free_text_segments,
    parse_cited_statements,
    row_filter_reasons,
)
from sdg.packs.synth.languages import (
    LanguagePlan,
    language_name,
    load_language_plan,
    row_uses_cross_language,
)
from sdg.packs.synth.llm_json import achat_json
from sdg.packs.synth.memorization_text import (
    as_list,
    clean_recall_list,
    clean_recall_text,
    normalize_text,
    split_sentences,
    tokenize,
)
from sdg.packs.synth.personas import iter_query_plans


class GroundedQASettings(TypedDict):
    max_rows: int
    lead_sentences: int
    max_sentences_per_doc: int
    min_sentence_words: int
    max_sentence_words: int
    structured_facts: int
    min_sources: int
    bridge_sources: int
    retrieve_top_k: int
    language_plan: LanguagePlan


class GroundedQAStats(TypedDict):
    rows: int
    candidate_rows: int
    rejected_rows: int


class _IndexedQueryPlan(TypedDict):
    row_index: int
    plan: dict[str, Any]


class _ResumeState(TypedDict):
    candidate_ids: set[str]
    stats: GroundedQAStats
    reject_reasons: dict[str, int]


class _GroundedQAProgressTracker:
    def __init__(
        self,
        *,
        worker_concurrency: int,
        stats: GroundedQAStats,
        reject_reasons: dict[str, int],
    ):
        self.worker_concurrency = worker_concurrency
        self.started_at = time.monotonic()
        self.stage = "initializing"
        self.completed = 0
        self.total: int | None = None
        self.candidate_rows = stats["candidate_rows"]
        self.rows = stats["rows"]
        self.rejected_rows = stats["rejected_rows"]
        self.reject_reasons = dict(reject_reasons)

    def start(self) -> None:
        self.stage = "generating_rows"
        self._write(force=True)

    def on_progress(self, completed: int, total: int | None, elapsed: int) -> None:
        self.completed = completed
        self.total = total
        self._write(elapsed_seconds=elapsed)

    def on_row(self, reasons: list[str]) -> None:
        self.candidate_rows += 1
        if reasons:
            self.rejected_rows += 1
            for reason in reasons:
                self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1
        else:
            self.rows += 1
        self._write()

    def finish(self) -> None:
        self.stage = "completed"
        self._write(force=True)

    def _write(self, *, elapsed_seconds: int | None = None, force: bool = False) -> None:
        elapsed = elapsed_seconds
        if elapsed is None:
            elapsed = int(time.monotonic() - self.started_at)
        candidates_per_minute = 0.0
        if elapsed > 0:
            candidates_per_minute = round(self.candidate_rows * 60 / elapsed, 2)
        write_snapshot(
            "grounded_qa_progress.json",
            {
                "stage": self.stage,
                "worker_concurrency": self.worker_concurrency,
                "completed": self.completed,
                "total": self.total,
                "elapsed_seconds": elapsed,
                "candidate_rows": self.candidate_rows,
                "rows": self.rows,
                "rejected_rows": self.rejected_rows,
                "reject_reasons": dict(sorted(self.reject_reasons.items())),
                "candidates_per_minute": candidates_per_minute,
            },
            force=force,
            min_interval_seconds=1.0,
        )


def generate_grounded_qa(
    cfg: dict[str, Any],
    memory: dict[str, Any],
    outputs_dir,
    *,
    seed: int | None,
) -> tuple[dict[str, Artifact], GroundedQAStats]:
    models = _load_grounded_qa_models(cfg)
    _progress_log("grounded_qa: loaded models")
    stats = asyncio.run(_write_grounded_qa_outputs_async(memory, cfg, outputs_dir, seed=seed, models=models))

    return (
        {
            "grounded_qa_rows": Artifact(
                name="grounded_qa_rows",
                path=str(outputs_dir / "grounded_qa_rows.jsonl"),
                kind="jsonl",
                meta={"rows": stats["rows"], "family": "grounded_qa"},
            ),
            "grounded_qa_candidates": Artifact(
                name="grounded_qa_candidates",
                path=str(outputs_dir / "grounded_qa_candidates.jsonl"),
                kind="jsonl",
                meta={"rows": stats["candidate_rows"], "family": "grounded_qa"},
            ),
            "grounded_qa_rejected": Artifact(
                name="grounded_qa_rejected",
                path=str(outputs_dir / "grounded_qa_rejected.jsonl"),
                kind="jsonl",
                meta={"rows": stats["rejected_rows"], "family": "grounded_qa"},
            ),
        },
        stats,
    )


async def _write_grounded_qa_outputs_async(
    memory: dict[str, Any],
    cfg: dict[str, Any],
    outputs_dir,
    *,
    seed: int | None,
    models: dict[str, LLM],
) -> GroundedQAStats:
    rows_path = outputs_dir / "grounded_qa_rows.jsonl"
    candidate_path = outputs_dir / "grounded_qa_candidates.jsonl"
    rejected_path = outputs_dir / "grounded_qa_rejected.jsonl"
    settings = _grounded_qa_settings(cfg)
    chunk_lookup = {chunk["id"]: chunk for chunk in memory["chunks"]}
    worker_concurrency = _worker_concurrency(models)
    query_plans = iter_query_plans(
        iter_grounded_specs(memory, settings, seed=seed),
        cfg,
        family="grounded_qa",
        seed=seed,
    )
    resume_state = _load_resume_state(outputs_dir)
    stats = dict(resume_state["stats"])
    tracker = _GroundedQAProgressTracker(
        worker_concurrency=worker_concurrency,
        stats=stats,
        reject_reasons=resume_state["reject_reasons"],
    )
    progress = _progress_reporter("grounded_qa.rows", tracker)
    indexed_query_plans = _indexed_query_plans(query_plans)
    pending_query_plans = _pending_query_plans(indexed_query_plans, resume_state["candidate_ids"])

    resume_count = len(resume_state["candidate_ids"])
    if resume_count:
        _progress_log(
            f"grounded_qa: resuming with {resume_count} existing candidates and worker_concurrency={worker_concurrency}"
        )
        log_event(
            "synth",
            "grounded_qa_resumed",
            existing_candidate_rows=resume_count,
            existing_rows=stats["rows"],
            existing_rejected_rows=stats["rejected_rows"],
            worker_concurrency=worker_concurrency,
        )
    else:
        _progress_log(f"grounded_qa: generating rows with worker_concurrency={worker_concurrency}")
    log_event(
        "synth",
        "grounded_qa_started",
        worker_concurrency=worker_concurrency,
        resumed=bool(resume_count),
    )
    tracker.start()
    file_mode = "a" if resume_count else "w"
    with rows_path.open(file_mode) as rows_handle, candidate_path.open(file_mode) as candidate_handle, rejected_path.open(file_mode) as rejected_handle:
        async for row in map_async_unordered(
            pending_query_plans,
            lambda _ignored_index, item: _generate_candidate_row_async(
                item,
                memory_index=memory["index"],
                chunk_lookup=chunk_lookup,
                settings=settings,
                models=models,
            ),
            concurrency=worker_concurrency,
            progress=progress,
        ):
            stats["candidate_rows"] += 1
            _write_jsonl_line(candidate_handle, row)

            reasons = row_filter_reasons(row)
            annotated = annotate_filter_result(row, reasons)
            tracker.on_row(reasons)
            if reasons:
                stats["rejected_rows"] += 1
                _write_jsonl_line(rejected_handle, annotated)
                continue

            stats["rows"] += 1
            _write_jsonl_line(rows_handle, annotated)

    common_publish.write_preview(
        _jsonl_prefix(rows_path, limit=50),
        outputs_dir / "grounded_qa_preview.jsonl",
        n=50,
    )
    common_publish.write_preview(
        _jsonl_prefix(rejected_path, limit=50),
        outputs_dir / "grounded_qa_rejected_preview.jsonl",
        n=50,
    )
    tracker.finish()
    log_event(
        "synth",
        "grounded_qa_completed",
        worker_concurrency=worker_concurrency,
        candidate_rows=stats["candidate_rows"],
        rows=stats["rows"],
        rejected_rows=stats["rejected_rows"],
    )
    _progress_log(f"grounded_qa: kept {stats['rows']} rows, rejected {stats['rejected_rows']}")
    return stats


def iter_grounded_specs(
    memory: dict[str, Any],
    settings: GroundedQASettings,
    *,
    seed: int | None,
) -> Iterator[dict[str, Any]]:
    docs = list(memory["docs"])
    rng = Random(seed if seed is not None else 0)
    rng.shuffle(docs)
    chunk_lookup = {chunk["id"]: chunk for chunk in memory["chunks"]}

    produced = 0
    for doc in docs:
        candidates = _candidate_sentences(
            doc["text"],
            lead_sentences=settings["lead_sentences"],
            min_words=settings["min_sentence_words"],
            max_words=settings["max_sentence_words"],
        )
        if not candidates:
            continue

        structured_context = _structured_facts(doc, limit=settings["structured_facts"])
        for candidate in candidates[: settings["max_sentences_per_doc"]]:
            seed_source = _seed_source(doc, candidate["text"], chunk_lookup)
            if seed_source is None:
                continue
            bridge_sources = _bridge_sources(
                doc,
                candidate["text"],
                structured_context,
                memory["index"],
                chunk_lookup,
                limit=settings["bridge_sources"],
            )
            source_count = 1 + len(bridge_sources)
            if source_count < settings["min_sources"]:
                continue

            yield {
                "doc": doc,
                "seed_source": seed_source,
                "primary_sentence": candidate["text"],
                "sentence_index": candidate["index"],
                "support_sentences": [source["claim"] for source in bridge_sources],
                "structured_facts": structured_context,
                "bridge_sources": bridge_sources,
            }
            produced += 1
            if produced >= settings["max_rows"]:
                return


def _bridge_sources(
    doc: dict[str, Any],
    primary_sentence: str,
    structured_context: list[str],
    index: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    query_tokens = set(tokenize(" ".join([doc["title"], primary_sentence, *structured_context])))
    scored_chunks: list[tuple[int, dict[str, Any]]] = []
    for chunk_id, entry in index["chunks"].items():
        chunk = chunk_lookup[chunk_id]
        if chunk["source_id"] == doc["id"]:
            continue
        overlap = len(query_tokens.intersection(entry["tokens"]))
        if overlap == 0:
            continue
        scored_chunks.append((overlap, chunk))

    scored_chunks.sort(
        key=lambda item: (item[0], item[1]["meta"]["word_count"]),
        reverse=True,
    )

    bridge_sources: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for score, chunk in scored_chunks:
        if chunk["source_id"] in seen_source_ids:
            continue
        bridge_sources.append(_source_spec(chunk, score=score))
        seen_source_ids.add(chunk["source_id"])
        if len(bridge_sources) >= limit:
            return bridge_sources
    return bridge_sources


def _seed_source(
    doc: dict[str, Any],
    primary_sentence: str,
    chunk_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    sentence_tokens = set(tokenize(primary_sentence))
    best_chunk: dict[str, Any] | None = None
    best_score = -1

    for chunk in chunk_lookup.values():
        if chunk["source_id"] != doc["id"]:
            continue
        score = len(sentence_tokens.intersection(tokenize(chunk["text"])))
        if score <= best_score:
            continue
        best_chunk = chunk
        best_score = score

    if best_chunk is None:
        return None

    return _source_spec(best_chunk, score=max(best_score, 1))


async def _generate_candidate_row_async(
    item: _IndexedQueryPlan,
    *,
    memory_index: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
    settings: GroundedQASettings,
    models: dict[str, LLM],
) -> dict[str, Any]:
    row_index = item["row_index"]
    plan = item["plan"]
    row = await _make_row_async(
        plan,
        row_index,
        llm=models["query_teacher"],
        planner=models["task_planner"],
        language_plan=settings["language_plan"],
    )
    try:
        row = retrieve_support_row(row, memory_index, chunk_lookup, settings)
        trace = await _llm_backreasoning_async(row, models["reasoning_teacher"])
        row = with_backreasoning(row, trace)
        answer = await _generate_answer_async(row, models["answer_teacher"])
        row = with_target(row, answer)
        judge = await _judge_row_async(row, models["judge"])
        return with_judge(row, judge)
    except AssertionError as error:
        log_event(
            "synth",
            "grounded_qa_row_failed",
            row_id=row["id"],
            error=str(error),
        )
        return with_generation_error(row, str(error))


async def _make_row_async(
    plan: dict[str, Any],
    index: int,
    *,
    llm: LLM,
    planner: LLM,
    language_plan: LanguagePlan,
) -> dict[str, Any]:
    bundle = plan["bundle"]
    persona = plan["persona"]
    query_profile = plan["query_profile"]
    assistant_style = plan["assistant_style"]
    doc = bundle["doc"]
    task_plan = await _llm_task_plan_async(plan, planner)
    question = await _llm_question_async(plan, task_plan, language_plan, llm)

    teacher_bundle = _build_teacher_bundle(doc, bundle)
    source_target = _default_source_target(bundle, task_plan)
    bridge_source_ids = [source["source_id"] for source in bundle["bridge_sources"]]
    additional_seed_urls = [source["url"] for source in bundle["bridge_sources"] if source.get("url")]
    required_cited_sources = _required_cited_sources(question["question_type"], bundle)
    row = {
        "id": f"grounded_qa-{index:06d}",
        "query_seed_url": doc.get("url"),
        "query_seed_text": bundle["primary_sentence"],
        "additional_seed_urls": additional_seed_urls,
        "constraints": list(task_plan["constraints"]),
        "prompt": question["prompt"],
        "target": "",
        "reasoning": "",
        "messages": [],
        "sources": [],
        "checks": {},
        "scores": {},
        "hidden": {
            "source_id": doc["id"],
            "source_title": doc["title"],
            "source_url": doc.get("url"),
            "expected_source_ids": [doc["id"], *bridge_source_ids],
            "question_type": question["question_type"],
            "generation_mode": question["generation_mode"],
            "persona": persona,
            "query_angle": plan["query_angle"],
            "query_profile": query_profile,
            "assistant_style": assistant_style,
            "task_plan": task_plan,
            "teacher_bundle": teacher_bundle,
            "source_target": source_target,
            "source_reasoning": "",
            "required_cited_sources": required_cited_sources,
        },
        "meta": {
            "family": "grounded_qa",
            "question_type": question["question_type"],
            "dataset": doc.get("meta", {}).get("dataset"),
            "language": doc.get("meta", {}).get("language"),
            "vital_level": doc.get("meta", {}).get("vital_level"),
            "persona_id": persona["persona_id"],
            "persona_source": persona["source"],
            "persona_tags": persona["tags"],
            "query_angle": plan["query_angle"],
            "task_type": task_plan["task_type"],
            "user_goal": task_plan["user_goal"],
            "source_language": language_plan["source"],
            "prompt_language": language_plan["prompt"],
            "reasoning_language": language_plan["reasoning"],
            "target_language": language_plan["target"],
            "query_profile_id": query_profile["profile_id"],
            "query_profile_source": query_profile["source"],
            "query_profile_tags": query_profile["tags"],
            "assistant_style_id": assistant_style["style_id"],
            "assistant_style_source": assistant_style["source"],
            "assistant_style_tags": assistant_style["tags"],
            "reasoning_style": "grounded_qa_trace_v1",
            "required_cited_sources": required_cited_sources,
        },
    }
    row["messages"] = _grounded_qa_messages(row)
    return row


def _build_teacher_bundle(doc: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "article_title": doc["title"],
        "primary_claim": bundle["primary_sentence"],
        "supporting_claims": list(bundle["support_sentences"]),
        "structured_context": list(bundle["structured_facts"]),
        "seed_source": dict(bundle["seed_source"]),
        "bridge_sources": list(bundle["bridge_sources"]),
    }


def _default_source_target(bundle: dict[str, Any], task_plan: dict[str, Any]) -> str:
    coverage_points = as_list(task_plan["coverage_points"])
    if coverage_points:
        return " ".join(coverage_points[:3])
    return bundle["primary_sentence"]


def _required_cited_sources(question_type: str, bundle: dict[str, Any]) -> int:
    source_count = 1 + len(bundle["bridge_sources"])
    if source_count < 2:
        return 1
    if question_type in _simple_question_types():
        return 1
    return 2


def _grounded_qa_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": _grounded_qa_system_prompt(row),
        },
        {
            "role": "user",
            "content": row["prompt"],
        },
    ]
    if row["target"]:
        messages.append(
            {
                "role": "assistant",
                "content": _grounded_qa_assistant_content(row),
            }
        )
    return messages


def _grounded_qa_assistant_content(row: dict[str, Any]) -> str:
    reasoning = row["reasoning"].strip()
    if not reasoning:
        return row["target"]
    return f"<think>\n{reasoning}\n</think>\n{row['target']}"


def _grounded_qa_system_prompt(row: dict[str, Any]) -> str:
    assistant_style = row["hidden"]["assistant_style"]
    return (
        f"You are {assistant_style['name']}. "
        f"{assistant_style['instructions']} "
        f"Use a {assistant_style['tone']} tone with {assistant_style['detail_level']} detail and a "
        f"{assistant_style['structure']} structure. "
        "Answer the user's question using only facts grounded in the provided sources. "
        "If the sources do not support a claim, do not include it. "
        "Keep any text outside statement tags short and non-factual. "
        'Put factual claims inside <statement cites="...">...</statement> and cite the supporting source ids for each statement.'
    )


def _simple_question_types() -> set[str]:
    return {
        "fact_disambiguation",
        "fact_verification",
        "factoid_qa",
        "identification",
        "recall",
    }


async def _llm_task_plan_async(plan: dict[str, Any], llm: LLM) -> dict[str, Any]:
    parsed = await achat_json(llm, _task_plan_messages(plan), temperature=0.4)
    return _parse_task_plan(plan, parsed)


def _task_plan_messages(plan: dict[str, Any]) -> list[dict[str, str]]:
    bundle = plan["bundle"]
    persona = plan["persona"]
    query_profile = plan["query_profile"]
    bridge_block = "\n".join(f"- {item['title']}: {item['snippet']}" for item in bundle["bridge_sources"])

    return [
        {
            "role": "system",
            "content": (
                "You plan realistic grounded QA tasks from hidden multi-source evidence. "
                "Return strict JSON with keys task_type, user_goal, answer_shape, coverage_points, constraints, and query_brief. "
                "The task should usually need at least two evidence points when they are available. "
                "When the evidence spans multiple documents, broad tasks should genuinely need synthesis across them. "
                "coverage_points must be a list of the main claims the final answer should cover. "
                "constraints must be a list of user-visible answer constraints, including inline citation use."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Persona id: {persona['persona_id']}\n"
                f"Persona intent: {persona['intent']}\n"
                f"Persona knowledge level: {persona['knowledge_level']}\n"
                f"Persona tone: {persona['tone']}\n"
                f"Persona answer granularity: {persona['answer_granularity']}\n"
                f"Query profile channel: {query_profile['channel']}\n"
                f"Query profile register: {query_profile['register']}\n"
                f"Query profile shape: {query_profile['query_shape']}\n"
                f"Query profile instructions: {query_profile['instructions']}\n"
                f"Legacy sampling hint: {plan['query_angle']}\n\n"
                f"Seed article title: {bundle['doc']['title']}\n"
                f"Primary claim: {bundle['primary_sentence']}\n"
                "Structured context:\n"
                + "\n".join(f"- {item}" for item in bundle["structured_facts"])
                + "\n\nEvidence points:\n"
                + "\n".join(f"- {item}" for item in _bundle_evidence_points(bundle))
                + "\n\nBridge evidence:\n"
                + bridge_block
            ),
        },
    ]


def _parse_task_plan(plan: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    task_type = str(parsed.get("task_type", "")).strip() or str(plan["query_angle"]).strip() or "overview"
    user_goal = str(parsed.get("user_goal", "")).strip() or str(plan["persona"]["intent"]).strip()
    answer_shape = str(parsed.get("answer_shape", "")).strip() or "grounded explanation"
    query_brief = str(parsed.get("query_brief", "")).strip() or _default_query_brief(plan)
    coverage_points = as_list(parsed.get("coverage_points"))
    if not coverage_points:
        coverage_points = _default_coverage_points(plan)
    constraints = clean_recall_list(parsed.get("constraints"))
    if not constraints:
        constraints = _default_constraints(plan)

    return {
        "task_type": task_type,
        "user_goal": user_goal,
        "answer_shape": answer_shape,
        "query_brief": query_brief,
        "coverage_points": coverage_points,
        "constraints": constraints,
    }


def _default_query_brief(plan: dict[str, Any]) -> str:
    return (
        f"Ask about {plan['bundle']['doc']['title']} in a way that benefits from grounded evidence "
        "and can be answered as cited XML statements."
    )


def _default_coverage_points(plan: dict[str, Any]) -> list[str]:
    bundle = plan["bundle"]
    return _bundle_evidence_points(bundle)[:5]


def _default_constraints(plan: dict[str, Any]) -> list[str]:
    constraints = list(plan["persona"]["constraints"])
    constraints.append('Write the answer as XML with <statement cites="...">...</statement> blocks.')
    return constraints


async def _llm_question_async(
    plan: dict[str, Any],
    task_plan: dict[str, Any],
    language_plan: LanguagePlan,
    llm: LLM,
) -> dict[str, str]:
    parsed = await achat_json(llm, _question_messages(plan, task_plan, language_plan), temperature=0.4)
    return _parse_question(parsed, task_plan)


def _question_messages(
    plan: dict[str, Any],
    task_plan: dict[str, Any],
    language_plan: LanguagePlan,
) -> list[dict[str, str]]:
    bundle = plan["bundle"]
    persona = plan["persona"]
    query_profile = plan["query_profile"]
    prompt_language = language_name(language_plan["prompt"])
    prompt_guidance = _language_generation_guidance(language_plan["prompt"])
    exemplars = query_profile["exemplars"]
    exemplar_block = ""
    if exemplars:
        exemplar_block = "Query profile exemplars:\n" + "\n".join(f"- {item}" for item in exemplars) + "\n\n"

    return [
        {
            "role": "system",
            "content": (
                "You create user-facing grounded QA prompts from teacher-side multi-source evidence. "
                "Return strict JSON with keys prompt and question_type. "
                "The prompt must feel like something a real user would ask. "
                "It should invite a grounded answer but must not mention hidden snippets, citations, or source labels. "
                f"Write the prompt in {prompt_language}. {prompt_guidance}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt language: {prompt_language}\n"
                f"Persona id: {persona['persona_id']}\n"
                f"Persona name: {persona['name']}\n"
                f"Intent: {persona['intent']}\n"
                f"Knowledge level: {persona['knowledge_level']}\n"
                f"Tone: {persona['tone']}\n"
                f"Question style: {persona['question_style']}\n"
                f"Answer granularity: {persona['answer_granularity']}\n\n"
                f"Query profile id: {query_profile['profile_id']}\n"
                f"Channel: {query_profile['channel']}\n"
                f"Fluency: {query_profile['fluency']}\n"
                f"Register: {query_profile['register']}\n"
                f"Urgency: {query_profile['urgency']}\n"
                f"Query shape: {query_profile['query_shape']}\n"
                f"Instructions: {query_profile['instructions']}\n\n"
                f"Task type: {task_plan['task_type']}\n"
                f"User goal: {task_plan['user_goal']}\n"
                f"Answer shape: {task_plan['answer_shape']}\n"
                "Coverage points:\n"
                + "\n".join(f"- {item}" for item in task_plan["coverage_points"])
                + "\n\n"
                f"Query brief: {task_plan['query_brief']}\n\n"
                f"{exemplar_block}"
                f"Seed article: {bundle['doc']['title']}\n"
                f"Seed claim: {bundle['primary_sentence']}\n"
                "Evidence points:\n"
                + "\n".join(f"- {item}" for item in _bundle_evidence_points(bundle))
                + "\n\n"
                "Bridge evidence titles:\n"
                + "\n".join(f"- {item['title']}" for item in bundle["bridge_sources"])
            ),
        },
    ]


def _parse_question(parsed: dict[str, Any], task_plan: dict[str, Any]) -> dict[str, str]:
    prompt = str(parsed.get("prompt", "")).strip()
    assert prompt, "grounded_qa question prompt must not be empty"
    question_type = str(parsed.get("question_type", task_plan["task_type"])).strip() or task_plan["task_type"]
    return {
        "prompt": prompt,
        "question_type": question_type,
        "generation_mode": "llm",
    }


def retrieve_support_row(
    row: dict[str, Any],
    index: dict[str, Any],
    chunk_lookup: dict[str, dict[str, Any]],
    settings: GroundedQASettings,
) -> dict[str, Any]:
    query_text = row["prompt"]
    if row_uses_cross_language(row):
        query_text = _source_support_query(row)

    query_tokens = set(tokenize(query_text))
    scored_chunks: list[tuple[int, dict[str, Any]]] = []
    for chunk_id, entry in index["chunks"].items():
        overlap = len(query_tokens.intersection(entry["tokens"]))
        if overlap == 0:
            continue
        chunk = chunk_lookup[chunk_id]
        scored_chunks.append((overlap, chunk))

    scored_chunks.sort(
        key=lambda item: (
            item[0],
            item[1]["source_id"] == row["hidden"]["source_id"],
            item[1]["meta"]["word_count"],
        ),
        reverse=True,
    )

    sources: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for preferred in _preferred_sources(row):
        if preferred["source_id"] in seen_source_ids:
            continue
        sources.append(
            {
                "citation_id": str(len(sources) + 1),
                **preferred,
            }
        )
        seen_source_ids.add(preferred["source_id"])
        if len(sources) >= settings["retrieve_top_k"]:
            break

    for score, chunk in scored_chunks:
        if len(sources) >= settings["retrieve_top_k"]:
            break
        if chunk["source_id"] in seen_source_ids:
            continue
        sources.append(
            {
                "citation_id": str(len(sources) + 1),
                **_source_spec(chunk, score=score),
            }
        )
        seen_source_ids.add(chunk["source_id"])

    updated = dict(row)
    updated["sources"] = sources
    updated["meta"] = {**row["meta"], "source_count": len(sources)}
    return updated


def _source_support_query(row: dict[str, Any]) -> str:
    teacher_bundle = row["hidden"]["teacher_bundle"]
    task_plan = row["hidden"]["task_plan"]
    parts = [
        row["hidden"]["source_title"],
        teacher_bundle.get("primary_claim", ""),
        *teacher_bundle.get("supporting_claims", []),
        *teacher_bundle.get("structured_context", []),
        *as_list(task_plan.get("coverage_points")),
    ]
    return " ".join(part for part in parts if part)


async def _llm_backreasoning_async(row: dict[str, Any], llm: LLM) -> dict[str, Any]:
    parsed = await achat_json(llm, _backreasoning_messages(row), temperature=0.2)
    return _parse_backreasoning(row, parsed)


def _backreasoning_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    source_language = language_name(row["meta"]["source_language"])
    reasoning_language = language_name(row["meta"]["reasoning_language"])
    reasoning_guidance = _language_generation_guidance(row["meta"]["reasoning_language"])
    task_plan = row["hidden"]["task_plan"]
    source_block = _format_sources_block(row["sources"])
    return [
        {
            "role": "system",
            "content": (
                "You write grounded reasoning notes for synthetic QA data. "
                "Return strict JSON with keys key_question, delivery_note, source_plan, known_facts, reasoning_steps, caveats, synthesis, and proposed_target. "
                "source_plan, known_facts, reasoning_steps, and caveats must be arrays of short strings. "
                "Use source_plan to sort which packed sources are most useful for the prompt, which ones only add support, "
                "and which ones can be ignored. Mention source ids when they help track support. "
                "Use delivery_note for a short explanation of which language the final answer should use and why. "
                "Base that explanation only on the visible prompt language and obvious conversational context. "
                f"The source snippets are in {source_language}. Write the visible reasoning in {reasoning_language}. "
                f"{reasoning_guidance}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt: {row['prompt']}\n"
                f"Question type: {row['meta']['question_type']}\n"
                f"Task type: {task_plan['task_type']}\n"
                f"User goal: {task_plan['user_goal']}\n"
                f"Answer shape: {task_plan['answer_shape']}\n"
                "Coverage points:\n"
                + "\n".join(f"- {item}" for item in task_plan["coverage_points"])
                + "\n\nPacked sources XML:\n"
                + source_block
            ),
        },
    ]


def _parse_backreasoning(row: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    delivery_note = clean_recall_text(str(parsed.get("delivery_note", "")).strip())
    assert delivery_note, "grounded_qa delivery_note must not be empty"
    source_plan = clean_recall_list(parsed.get("source_plan"))
    assert source_plan, "grounded_qa source_plan must not be empty"
    return {
        "key_question": clean_recall_text(str(parsed.get("key_question", "")).strip()) or _default_key_question(row),
        "delivery_note": delivery_note,
        "source_plan": source_plan,
        "known_facts": clean_recall_list(parsed.get("known_facts")),
        "reasoning_steps": clean_recall_list(parsed.get("reasoning_steps")),
        "caveats": clean_recall_list(parsed.get("caveats")),
        "synthesis": clean_recall_text(str(parsed.get("synthesis", "")).strip())
        or _default_synthesis(row),
        "proposed_target": clean_recall_text(str(parsed.get("proposed_target", "")).strip()),
    }


def with_backreasoning(row: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    hidden = dict(row["hidden"])
    hidden["teacher_backreasoning"] = trace

    updated = dict(row)
    updated["hidden"] = hidden
    updated["reasoning"] = _render_reasoning(row, trace)
    return updated


def _render_reasoning(row: dict[str, Any], trace: dict[str, Any]) -> str:
    if row["meta"]["reasoning_language"] == "da":
        key_question_label = "Nøglespørgsmål"
        delivery_label = "Svarplan"
        source_plan_label = "### 1. Kildeplan"
        known_facts_label = "### 2. Kendte fakta"
        resolution_label = "### 3. Afklaring"
        caveats_label = "### 4. Forbehold"
        synthesis_label = "### 5. Konklusion"
    else:
        key_question_label = "Key question"
        delivery_label = "Response plan"
        source_plan_label = "### 1. Source plan"
        known_facts_label = "### 2. Known facts"
        resolution_label = "### 3. Resolution"
        caveats_label = "### 4. Caveats"
        synthesis_label = "### 5. Synthesis"

    parts = [f"{key_question_label}: {trace['key_question']}"]
    parts.append(f"{delivery_label}: {trace['delivery_note']}")

    if trace["source_plan"]:
        parts.append(source_plan_label + "\n" + "\n".join(f"- {item}" for item in trace["source_plan"]))
    if trace["known_facts"]:
        parts.append(known_facts_label + "\n" + "\n".join(f"- {item}" for item in trace["known_facts"]))
    if trace["reasoning_steps"]:
        parts.append(resolution_label + "\n" + "\n".join(f"- {item}" for item in trace["reasoning_steps"]))
    if trace["caveats"]:
        parts.append(caveats_label + "\n" + "\n".join(f"- {item}" for item in trace["caveats"]))
    if trace["synthesis"]:
        parts.append(synthesis_label + "\n" + trace["synthesis"])

    return "\n\n".join(parts)


def _render_source_reasoning(row: dict[str, Any]) -> str:
    source_language = row["meta"]["source_language"]
    coverage_points = as_list(row["hidden"]["task_plan"].get("coverage_points"))
    if source_language == "da":
        key_question_label = "Nøglespørgsmål"
        source_plan_label = "### 1. Kildeplan"
        known_facts_label = "### 2. Kendte fakta"
        resolution_label = "### 3. Afklaring"
        synthesis_label = "### 4. Konklusion"
    else:
        key_question_label = "Key question"
        source_plan_label = "### 1. Source plan"
        known_facts_label = "### 2. Known facts"
        resolution_label = "### 3. Resolution"
        synthesis_label = "### 4. Synthesis"

    known_facts = [row["hidden"]["teacher_bundle"]["primary_claim"]]
    known_facts.extend(row["hidden"]["teacher_bundle"].get("supporting_claims", [])[:3])
    known_facts.extend(source["snippet"] for source in row["sources"][:2])
    parts = [f"{key_question_label}: {_default_key_question(row)}"]
    source_plan = _default_source_plan(row)
    if source_plan:
        parts.append(source_plan_label + "\n" + "\n".join(f"- {item}" for item in source_plan))
    parts.append(known_facts_label + "\n" + "\n".join(f"- {item}" for item in known_facts if item))
    if coverage_points:
        parts.append(resolution_label + "\n" + "\n".join(f"- {item}" for item in coverage_points[:3]))
    parts.append(synthesis_label + "\n" + row["hidden"]["source_target"])
    return "\n\n".join(parts)


async def _generate_answer_async(row: dict[str, Any], llm: LLM) -> dict[str, str]:
    last_error = ""
    for attempt in range(3):
        parsed = await achat_json(llm, _answer_messages(row), temperature=0.0)
        try:
            return _parse_target(parsed)
        except AssertionError as error:
            last_error = str(error)
            log_event(
                "synth",
                "grounded_qa_answer_retry",
                row_id=row["id"],
                attempt=attempt + 1,
                error=last_error,
            )
    raise AssertionError(f"grounded_qa target retry budget exhausted: {last_error}")


def _answer_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    persona = row["hidden"]["persona"]
    query_profile = row["hidden"]["query_profile"]
    assistant_style = row["hidden"]["assistant_style"]
    task_plan = row["hidden"]["task_plan"]
    source_language = language_name(row["meta"]["source_language"])
    target_language = language_name(row["meta"]["target_language"])
    target_guidance = _language_generation_guidance(row["meta"]["target_language"])
    source_block = _format_sources_block(row["sources"])
    required_cited_sources = int(row["hidden"].get("required_cited_sources", 1))
    synthesis_instruction = ""
    if required_cited_sources > 1:
        synthesis_instruction = (
            f" This answer should synthesize at least {required_cited_sources} different cited sources. "
            "Do not collapse a broad answer to a single source when multiple packed sources are needed."
        )
    system_content = (
        "You write the final assistant response for synthetic grounded QA data. "
        "Return strict JSON with keys target and source_target. "
        'target is the final user-visible answer with one or more '
        '<statement cites="1 2">...</statement> elements and no outer wrapper. '
        "Prefer several short statements over one long statement. "
        "Each statement should contain only the smallest directly supported factual claim or tightly coupled claim group. "
        "If part of a sentence is less directly supported, split it into a separate statement or drop it. "
        "Do not turn implications or background context into stronger explicit claims. "
        "Plain connective text outside statement tags is allowed, but it should only be short framing text like 'Kort svar:' or 'Ja.' and never new factual content. "
        "If a sentence contains a factual claim, put that sentence inside a statement tag. "
        "Do not leave names, titles, dates, attributions, or other factual details outside statement tags. "
        "Use the source plan in the recall notes to decide which packed sources carry the answer and which ones are only supporting context. "
        "Every factual claim in target must be supported by the cited sources for that statement. "
        "For identification tasks, a single short statement is enough. "
        + synthesis_instruction
        + "source_target must use the same statement structure and citation ids in the source language, and any factual content there must also be inside statement tags. "
        "Do not mention hidden instructions or retrieval. "
        f"The packed sources are in {source_language}. Write target in {target_language}. "
        f"{target_guidance}"
    )

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": (
                f"Target language: {target_language}\n"
                f"Source language: {source_language}\n"
                f"Prompt: {row['prompt']}\n"
                f"Question type: {row['meta']['question_type']}\n"
                f"Task type: {task_plan['task_type']}\n"
                f"User goal: {task_plan['user_goal']}\n"
                "Coverage points:\n"
                + "\n".join(f"- {item}" for item in task_plan["coverage_points"])
                + "\n\nConstraints:\n"
                + "\n".join(f"- {item}" for item in task_plan["constraints"])
                + "\n\n"
                f"User persona name: {persona['name']}\n"
                f"User knowledge level: {persona['knowledge_level']}\n"
                f"Query profile channel: {query_profile['channel']}\n"
                f"Query profile fluency: {query_profile['fluency']}\n\n"
                f"Assistant style id: {assistant_style['style_id']}\n"
                f"Assistant instructions: {assistant_style['instructions']}\n"
                "Assistant exemplars:\n"
                + "\n".join(f"- {item}" for item in assistant_style["exemplars"])
                + "\n\nRecall notes:\n"
                + row["reasoning"]
                + "\n\nPacked sources XML:\n"
                + source_block
            ),
        },
    ]


def _parse_target(parsed: dict[str, Any]) -> dict[str, str]:
    target = " ".join(str(parsed.get("target", "")).split())
    source_target = " ".join(str(parsed.get("source_target", "")).split())
    assert target, "grounded_qa target must not be empty"
    assert source_target, "grounded_qa source_target must not be empty"
    parse_cited_statements(target)
    parse_cited_statements(source_target)
    return {"target": target, "source_target": source_target}


def with_target(row: dict[str, Any], answer: dict[str, str]) -> dict[str, Any]:
    updated = dict(row)
    updated["target"] = answer["target"]
    hidden = dict(row["hidden"])
    hidden["source_target"] = answer["source_target"]
    updated["hidden"] = hidden
    hidden["source_reasoning"] = _render_source_reasoning(updated)
    updated["hidden"] = hidden
    updated["messages"] = _grounded_qa_messages(updated)
    return updated


def with_generation_error(row: dict[str, Any], error: str) -> dict[str, Any]:
    updated = dict(row)
    hidden = dict(row["hidden"])
    hidden["generation_error"] = error
    updated["hidden"] = hidden
    updated["target"] = ""
    updated["messages"] = _grounded_qa_messages(updated)
    updated["scores"] = {
        **row["scores"],
        "judge": {
            "pass": False,
            "support": False,
            "citations": False,
            "completeness": False,
            "language_match": True,
            "language_natural": True,
            "language_quality": True,
            "reason": error,
        },
    }
    return updated


async def _judge_row_async(row: dict[str, Any], llm: LLM) -> dict[str, Any]:
    diagnostics = citation_support_diagnostics(row)
    parsed = await achat_json(llm, _judge_messages(row, diagnostics), temperature=0.0)
    return _parse_judge(row, parsed)


def _judge_messages(row: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, str]]:
    source_language = language_name(row["meta"]["source_language"])
    prompt_language = language_name(row["meta"]["prompt_language"])
    reasoning_language = language_name(row["meta"]["reasoning_language"])
    target_language = language_name(row["meta"]["target_language"])
    source_block = _format_sources_block(row["sources"])
    diagnostics_lines = [
        f"- has_citations: {diagnostics['has_citations']}",
        f"- citation_supported: {diagnostics['ok']}",
        f"- supported_statements: {diagnostics['supported_statements']}/{diagnostics['total_statements']}",
    ]
    used_citations = extract_citation_ids(str(row["target"]))
    required_cited_sources = int(row["hidden"].get("required_cited_sources", 1))
    diagnostics_lines.append(f"- required_cited_sources: {required_cited_sources}")
    diagnostics_lines.append(f"- used_cited_sources: {len(set(used_citations))}")
    target_free_text = extract_free_text_segments(str(row["target"]))
    source_free_text = extract_free_text_segments(str(row["hidden"]["source_target"]))
    diagnostics_lines.append(f"- target_free_text_segments: {len(target_free_text)}")
    diagnostics_lines.append(f"- source_free_text_segments: {len(source_free_text)}")
    for issue in diagnostics["issues"][:4]:
        diagnostics_lines.append(f"- issue: {issue}")
    for segment in target_free_text[:2]:
        diagnostics_lines.append(f"- target_free_text: {segment}")
    for segment in source_free_text[:2]:
        diagnostics_lines.append(f"- source_free_text: {segment}")
    return [
        {
            "role": "system",
            "content": (
                "You judge synthetic grounded QA examples. "
                "Return strict JSON with keys pass, support, citations, completeness, language_match, language_natural, and reason. "
                "pass must be true only if the answer is supported, uses valid citations, is reasonably complete for the task, "
                "and the visible text matches the declared languages. "
                "A deterministic citation heuristic is provided below. Treat it as a fallible signal, not a final verdict. "
                f"The packed sources are in {source_language}. The visible prompt is in {prompt_language}, "
                f"the visible reasoning is in {reasoning_language}, and the visible answer is in {target_language}. "
                "Use the packed sources as the evidence base for support and citations. "
                "Do not treat the source-language reference answer as evidence. "
                "Only use the source-language reference answer to check whether the visible answer says the same thing in another language. "
                "Do not use prior knowledge, common knowledge, or plausible inference to rescue an unsupported statement. "
                "If a statement is not supported by the packed sources, citations must fail even if the claim is true in the real world. "
                "If the deterministic heuristic flags an unsupported statement, only overrule it when the packed sources themselves clearly support that statement. "
                "If the row requires multiple cited sources, completeness or citations should fail when the answer relies on too few distinct sources. "
                "language_natural should fail for mixed-language phrasing, obvious literal translation artifacts, or invented title translations. "
                "Citations should fail if factual content appears outside statement tags. "
                "Short framing text outside statements is acceptable only when it adds no facts."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt: {row['prompt']}\n"
                f"Answer: {row['target']}\n"
                f"Reasoning: {row['reasoning']}\n\n"
                f"Source-language reference answer (meaning only, not evidence): {row['hidden']['source_target']}\n\n"
                f"Task type: {row['hidden']['task_plan']['task_type']}\n"
                f"User goal: {row['hidden']['task_plan']['user_goal']}\n"
                "Coverage points:\n"
                + "\n".join(f"- {item}" for item in row["hidden"]["task_plan"]["coverage_points"])
                + "\n\nDeterministic citation heuristic:\n"
                + "\n".join(diagnostics_lines)
                + "\n\nPacked sources XML:\n"
                + source_block
            ),
        },
    ]


def _parse_judge(row: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    if row_uses_cross_language(row):
        assert "language_match" in parsed, "judge language_match must be present for cross-language rows"
        assert "language_natural" in parsed, "judge language_natural must be present for cross-language rows"
    language_quality = bool(parsed.get("language_match", True)) and bool(parsed.get("language_natural", True))
    return {
        "pass": bool(parsed.get("pass", False)),
        "support": bool(parsed.get("support", False)),
        "citations": bool(parsed.get("citations", False)),
        "completeness": bool(parsed.get("completeness", False)),
        "language_match": bool(parsed.get("language_match", True)),
        "language_natural": bool(parsed.get("language_natural", True)),
        "language_quality": language_quality,
        "reason": str(parsed.get("reason", "")).strip(),
    }


def with_judge(row: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    updated["scores"] = {
        **row["scores"],
        "judge": judge,
        "citation_check": citation_support_diagnostics(row),
    }
    return updated


def _default_key_question(row: dict[str, Any]) -> str:
    if row["meta"]["reasoning_language"] == "da":
        return "Hvilke kilder bærer de vigtigste dele af svaret?"
    return "Which sources carry the main parts of the answer?"


def _default_synthesis(row: dict[str, Any]) -> str:
    if row["meta"]["reasoning_language"] == "da":
        return "Svaret skal samle de vigtigste punkter og knytte dem til de rette kilder."
    return "The answer should combine the main points and attach them to the right sources."


def _language_generation_guidance(language: str) -> str:
    if language == "da":
        return (
            "Write idiomatic natural Danish. Avoid mixed English-Danish phrasing, literal translation artifacts, "
            "and stray English fragments unless they are established names or titles. Do not directly translate names, work titles, "
            "or institutions unless there is a clearly established Danish form. If the Danish form is unclear, keep the original name. "
            "Use standard Danish casing and grammar."
        )
    return (
        "Write idiomatic natural English. Avoid translated-sounding phrasing and awkward mixed-language wording. "
        "Do not directly translate names, work titles, or institutions unless there is a clearly established English form. "
        "If the English form is unclear, keep the original name."
    )


def _progress_log(message: str) -> None:
    print(f"[synth] {message}", flush=True)


def _progress_reporter(
    label: str,
    tracker: _GroundedQAProgressTracker,
) -> Callable[[int, int | None, int], None]:
    next_log = {"count": 1}

    def report(completed: int, total: int | None, elapsed: int) -> None:
        tracker.on_progress(completed, total, elapsed)
        total_text = "?" if total is None else str(total)
        if completed == 0:
            _progress_log(f"{label}: 0/{total_text}")
            return
        if completed < next_log["count"]:
            return
        _progress_log(f"{label}: {completed}/{total_text} ({elapsed}s)")
        if total is not None and total <= 10:
            next_log["count"] = completed + 1
            return
        if total is None:
            next_log["count"] = completed + 10
            return
        next_log["count"] = completed + max(1, total // 10)

    return report


def _worker_concurrency(models: dict[str, LLM]) -> int:
    limits = [getattr(getattr(model, "runtime", None), "max_concurrency", 1) for model in models.values()]
    return max(limits)


def _write_jsonl_line(handle: TextIO, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True))
    handle.write("\n")
    handle.flush()


def _load_resume_state(outputs_dir) -> _ResumeState:
    candidate_ids: set[str] = set()
    stats: GroundedQAStats = {"rows": 0, "candidate_rows": 0, "rejected_rows": 0}
    reject_reasons: dict[str, int] = {}

    candidate_path = outputs_dir / "grounded_qa_candidates.jsonl"
    if candidate_path.exists():
        for row in store.read_jsonl(candidate_path):
            candidate_ids.add(str(row["id"]))
        stats["candidate_rows"] = len(candidate_ids)

    rows_path = outputs_dir / "grounded_qa_rows.jsonl"
    if rows_path.exists():
        stats["rows"] = len(store.read_jsonl(rows_path))

    rejected_path = outputs_dir / "grounded_qa_rejected.jsonl"
    if rejected_path.exists():
        rejected_rows = store.read_jsonl(rejected_path)
        stats["rejected_rows"] = len(rejected_rows)
        for row in rejected_rows:
            for reason in row.get("hidden", {}).get("generation_filter", {}).get("reasons", []):
                key = str(reason)
                reject_reasons[key] = reject_reasons.get(key, 0) + 1

    return {"candidate_ids": candidate_ids, "stats": stats, "reject_reasons": reject_reasons}


def _indexed_query_plans(query_plans: Iterable[dict[str, Any]]) -> Iterator[_IndexedQueryPlan]:
    for row_index, plan in enumerate(query_plans):
        yield {"row_index": row_index, "plan": plan}


def _pending_query_plans(
    indexed_query_plans: Iterable[_IndexedQueryPlan],
    existing_candidate_ids: set[str],
) -> Iterator[_IndexedQueryPlan]:
    for item in indexed_query_plans:
        row_id = f"grounded_qa-{item['row_index']:06d}"
        if row_id in existing_candidate_ids:
            continue
        yield item


def _jsonl_prefix(path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def _candidate_sentences(
    text: str,
    *,
    lead_sentences: int,
    min_words: int,
    max_words: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, sentence in enumerate(split_sentences(text)[:lead_sentences]):
        words = len(sentence.split())
        if words < min_words or words > max_words:
            continue
        candidates.append({"index": index, "text": sentence})
    return candidates


def _structured_facts(doc: dict[str, Any], *, limit: int) -> list[str]:
    meta = doc.get("meta") or {}
    wikidata = meta.get("wikidata") or {}
    structured = meta.get("structured_wikipedia") or {}
    facts: list[str] = []

    description = str(wikidata.get("description") or meta.get("wikibase_shortdesc") or "").strip()
    if description:
        facts.append(f"Wikidata description: {description}")

    categories = [str(item).strip() for item in structured.get("categories", []) if str(item).strip()]
    if categories:
        facts.append("Categories: " + ", ".join(categories[:3]))

    outgoing_links = [str(item).strip() for item in structured.get("outgoing_links", []) if str(item).strip()]
    if outgoing_links:
        facts.append("Related links: " + ", ".join(outgoing_links[:3]))

    return facts[:limit]


def _format_sources_block(sources: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'<source id="{source["citation_id"]}"><title>{escape(str(source["title"]))}</title><snippet>{escape(str(source["snippet"]))}</snippet></source>'
        for source in sources
    )


def _bundle_evidence_points(bundle: dict[str, Any]) -> list[str]:
    points = [bundle["primary_sentence"]]
    points.extend(bundle["support_sentences"])
    points.extend(bundle["structured_facts"][:2])
    return [point for point in points if point]


def _default_source_plan(row: dict[str, Any]) -> list[str]:
    source_language = row["meta"]["source_language"]
    required_cited_sources = int(row["hidden"].get("required_cited_sources", 1))
    selected = row["sources"][:required_cited_sources]
    supporting = row["sources"][required_cited_sources:required_cited_sources + 2]
    lines: list[str] = []

    if source_language == "da":
        for source in selected:
            lines.append(
                f"Kilde {source['citation_id']} er central, fordi den dækker {source['claim']}"
            )
        for source in supporting:
            lines.append(
                f"Kilde {source['citation_id']} er sekundær støtte og bør kun bruges til detaljer"
            )
        return lines

    for source in selected:
        lines.append(f"Source {source['citation_id']} is central because it covers {source['claim']}")
    for source in supporting:
        lines.append(f"Source {source['citation_id']} is secondary support and should only add detail")
    return lines


def _preferred_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    teacher_bundle = row["hidden"]["teacher_bundle"]
    sources = [teacher_bundle["seed_source"]]
    sources.extend(teacher_bundle.get("bridge_sources", []))
    return sources


def _source_spec(chunk: dict[str, Any], *, score: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk["id"],
        "source_id": chunk["source_id"],
        "title": chunk["title"],
        "url": chunk.get("url"),
        "score": score,
        "snippet": chunk["text"],
        "claim": _source_claim(chunk["text"]),
    }


def _source_claim(text: str) -> str:
    sentences = split_sentences(text)
    if sentences:
        return sentences[0]
    return text


def _load_grounded_qa_models(cfg: dict[str, Any]) -> dict[str, LLM]:
    model_refs = cfg.get("models", {})
    required_roles = ["query_teacher", "answer_teacher", "judge"]
    missing = [role for role in required_roles if role not in model_refs]
    if missing:
        raise ValueError(f"Missing model roles for grounded_qa LLM path: {', '.join(missing)}")

    requested_roles = dict(model_refs)
    if "reasoning_teacher" not in requested_roles:
        requested_roles["reasoning_teacher"] = requested_roles["answer_teacher"]
    if "task_planner" not in requested_roles:
        requested_roles["task_planner"] = requested_roles["query_teacher"]

    roles = ["task_planner", "query_teacher", "reasoning_teacher", "answer_teacher", "judge"]
    clients = load_clients({role: requested_roles[role] for role in roles})
    return {role: clients[role] for role in roles}


def _grounded_qa_settings(cfg: dict[str, Any]) -> GroundedQASettings:
    generation_cfg = cfg.get("generation", {})
    assert isinstance(generation_cfg, dict), "generation config must be a mapping"

    family_cfg = generation_cfg.get("grounded_qa", {})
    assert isinstance(family_cfg, dict), "grounded_qa config must be a mapping"

    max_rows = _positive_int(
        family_cfg,
        "max_rows",
        default=_positive_int(generation_cfg, "max_rows_per_family", default=200),
    )
    retrieve_top_k = _positive_int(family_cfg, "retrieve_top_k", default=6)
    assert retrieve_top_k <= 10, "grounded_qa retrieve_top_k must be at most 10"

    return {
        "max_rows": max_rows,
        "lead_sentences": _positive_int(family_cfg, "lead_sentences", default=8),
        "max_sentences_per_doc": _positive_int(family_cfg, "max_sentences_per_doc", default=3),
        "min_sentence_words": _positive_int(family_cfg, "min_sentence_words", default=5),
        "max_sentence_words": _positive_int(family_cfg, "max_sentence_words", default=90),
        "structured_facts": _positive_int(family_cfg, "structured_facts", default=4),
        "min_sources": _positive_int(family_cfg, "min_sources", default=2),
        "bridge_sources": _positive_int(family_cfg, "bridge_sources", default=3),
        "retrieve_top_k": retrieve_top_k,
        "language_plan": load_language_plan(cfg, family="grounded_qa"),
    }


def _positive_int(record: dict[str, Any], key: str, *, default: int) -> int:
    value = record.get(key)
    if value is None:
        return default
    assert isinstance(value, int) and value > 0, f"{key} must be a positive integer"
    return value

