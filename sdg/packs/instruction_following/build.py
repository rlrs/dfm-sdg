from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import eval as common_eval
from sdg.commons import model as common_model
from sdg.commons import publish as common_publish
from sdg.commons.model import LLM
from sdg.commons.run import load, run
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json
from sdg.commons.work_queue import map_async_ordered
from sdg.packs.instruction_following.constraints import (
    LanguageCode,
    ResponseShape,
    available_languages,
    available_shapes,
    check_constraints_loose,
    check_constraints_strict,
    constraint_categories,
    constraint_definition,
    render_constraint_lines,
    sample_constraints,
    scenario_kind_pool,
    topic_pool,
)
from sdg.packs.instruction_following.generator import (
    fallback_scenario_bundle,
    follow_up_surface_keys,
    generate_scenario_bundle,
    instruction_surface_keys,
    materialize_messages,
    profile_prompt_seed,
    render_instruction_block,
    render_messages,
    select_prompt_keywords,
)
from sdg.packs.instruction_following.prompt_sources import (
    build_prompt_sampler,
    load_prompt_reservoirs,
    prompt_source_label_for_row,
    sample_balanced_prompt_seed,
)


def build(cfg: dict[str, Any]) -> BuildResult:
    return run(
        _build_run,
        pack="instruction_following",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    plan = _load_plan(result)
    verification = verify_rows(rows, plan=plan)

    outputs_dir = Path(result.run_dir) / "outputs"
    store.write_jsonl(verification["rows"], outputs_dir / "verified.jsonl")
    store.write_jsonl(verification["failures"], outputs_dir / "failures.jsonl")
    write_json(verification["metrics"], outputs_dir / "metrics.json")
    write_json(verification["failure_summary"], outputs_dir / "failure_summary.json")
    write_json(verification["dataset_checks"], outputs_dir / "dataset_checks.json")
    common_publish.write_preview(verification["rows"], outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "run_id": result.run_id,
        "verified_rows": len(verification["rows"]),
        "failed_rows": len(verification["failures"]),
        "response_checks_applied": verification["response_checks_applied"],
        "metrics": verification["metrics"],
        "failure_summary": verification["failure_summary"],
        "dataset_checks": verification["dataset_checks"],
    }


def verify_rows(rows: list[dict[str, Any]], *, plan: dict[str, Any] | None = None) -> dict[str, Any]:
    verified_rows = common_eval.verify(rows, _messages_well_formed, name="messages_well_formed")
    verified_rows = common_eval.verify(verified_rows, _instruction_block_matches_hidden, name="instruction_block_matches_hidden")
    verified_rows = common_eval.verify(verified_rows, _constraints_supported, name="constraints_supported")
    verified_rows = common_eval.verify(verified_rows, _generation_succeeded, name="generation_succeeded")

    response_checks_applied = _has_expected_responses(rows)
    if response_checks_applied:
        verified_rows = common_eval.verify(verified_rows, _response_present, name="response_present")
        verified_rows = common_eval.verify(verified_rows, _response_follows_strict, name="response_follows_strict")
        verified_rows = common_eval.verify(verified_rows, _response_follows_loose, name="response_follows_loose")

    failures = [row for row in verified_rows if _row_failed(row)]
    metrics = common_eval.aggregate_metrics(verified_rows)
    failure_summary = common_eval.summarize_failures(verified_rows)
    dataset_checks = _dataset_checks(verified_rows, plan) if plan is not None else {}

    return {
        "rows": verified_rows,
        "failures": failures,
        "metrics": metrics,
        "failure_summary": failure_summary,
        "dataset_checks": dataset_checks,
        "response_checks_applied": response_checks_applied,
    }


def summarize(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    dataset_checks_path = outputs_dir / "dataset_checks.json"

    metrics = read_json(metrics_path) if metrics_path.exists() else common_eval.aggregate_metrics(rows)
    dataset_checks = read_json(dataset_checks_path) if dataset_checks_path.exists() else {}

    language_counts = Counter(str(row["meta"]["language"]) for row in rows)
    interaction_counts = Counter(str(row["meta"]["interaction_style"]) for row in rows)
    shape_counts = Counter(str(row["meta"]["response_shape"]) for row in rows)
    constraint_counts = Counter(
        str(constraint["id"])
        for row in rows
        for constraint in row["hidden"]["constraints"]
    )
    prompt_source_counts = Counter(str(row["meta"].get("prompt_source", "synthetic")) for row in rows)
    prompt_task_type_counts = Counter(str(row["meta"].get("prompt_task_type", "synthetic")) for row in rows)
    prompt_length_bucket_counts = Counter(str(row["meta"].get("prompt_length_bucket", "synthetic")) for row in rows)

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "artifacts": sorted(result.artifacts),
        "benchmark": "ifbench",
        "language_counts": dict(language_counts),
        "interaction_style_counts": dict(interaction_counts),
        "response_shape_counts": dict(shape_counts),
        "constraint_counts": dict(constraint_counts),
        "prompt_source_counts": dict(prompt_source_counts),
        "prompt_task_type_counts": dict(prompt_task_type_counts),
        "prompt_length_bucket_counts": dict(prompt_length_bucket_counts),
        "metrics": metrics,
        "dataset_checks": dataset_checks,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    export_rows = [_strip_hidden(row) for row in rows]
    failures = [row for row in export_rows if _row_failed(row)]
    train_rows, eval_rows = _split_rows(export_rows, cfg["generation"]["train_fraction"])

    target_dir = _publish_dir(result, out_dir)
    store.ensure_dir(target_dir)
    store.write_parquet(train_rows, target_dir / "train.parquet")
    store.write_parquet(eval_rows, target_dir / "eval.parquet")
    store.write_parquet(failures, target_dir / "failures.parquet")
    common_publish.write_preview(export_rows, target_dir / "sample_preview.jsonl", n=20)

    outputs_dir = Path(result.run_dir) / "outputs"
    metrics = _load_or_compute(outputs_dir / "metrics.json", common_eval.aggregate_metrics(rows))
    failure_summary = _load_or_compute(outputs_dir / "failure_summary.json", common_eval.summarize_failures(rows))
    dataset_checks = _load_or_compute(outputs_dir / "dataset_checks.json", {})

    write_json(metrics, target_dir / "metrics.json")
    write_json(failure_summary, target_dir / "failure_summary.json")
    write_json(dataset_checks, target_dir / "dataset_checks.json")
    common_publish.write_report(metrics, failure_summary, target_dir / "report.json")
    common_publish.write_manifest(
        {
            "pack": result.pack,
            "run_id": result.run_id,
            "source_run_dir": result.run_dir,
            "source_artifacts": sorted(result.artifacts),
            "published_artifacts": [
                "train.parquet",
                "eval.parquet",
                "failures.parquet",
                "sample_preview.jsonl",
                "manifest.json",
                "metrics.json",
                "failure_summary.json",
                "dataset_checks.json",
                "report.json",
            ],
        },
        target_dir / "manifest.json",
    )

    return {
        "run_id": result.run_id,
        "out_dir": str(target_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "failure_rows": len(failures),
    }


def _build_run(
    *,
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
) -> dict[str, Artifact]:
    plan = _create_plan(cfg, seed)
    plan_path = write_json(plan, outputs_dir / "plan.json")
    rows = _generate_rows(plan, cfg)
    dataset_path = store.write_jsonl(rows, outputs_dir / "dataset.jsonl")
    common_publish.write_preview(rows, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={"rows": len(rows), "family": "instruction_following"},
        ),
        "plan": Artifact(
            name="plan",
            path=str(plan_path),
            kind="json",
            meta={"rows": len(plan["rows"])},
        ),
    }


def _create_plan(cfg: dict[str, Any], seed: int | None) -> dict[str, Any]:
    return asyncio.run(_create_plan_async(cfg, seed))


async def _create_plan_async(cfg: dict[str, Any], seed: int | None) -> dict[str, Any]:
    generation = cfg["generation"]
    count = _positive_int(generation, "count", default=20)
    min_constraints = _positive_int(generation, "min_constraints", default=1)
    max_constraints = _positive_int(generation, "max_constraints", default=3)
    assert min_constraints <= max_constraints, "generation min_constraints must be <= max_constraints"
    stress_probability = float(generation.get("semantic_stress_probability", 0.12))

    languages = _languages(generation)
    interaction_styles = _interaction_styles(generation)
    response_shapes = _response_shapes(generation)
    prompt_reservoirs = load_prompt_reservoirs(generation, languages=languages)
    prompt_profiler = _load_prompt_profiler(cfg) if prompt_reservoirs else None
    rng = Random(seed if seed is not None else 0)
    prompt_samplers = {
        language: build_prompt_sampler(prompts, seed=rng.randint(0, 1_000_000_000))
        for language, prompts in prompt_reservoirs.items()
    }
    instruction_surface_orders = {
        language: _shuffled_values(rng, instruction_surface_keys(_language_code(language)))
        for language in languages
    }
    follow_up_surface_orders = {
        language: _shuffled_values(rng, follow_up_surface_keys(_language_code(language)))
        for language in languages
    }
    instruction_surface_offsets: Counter[str] = Counter()
    follow_up_surface_offsets: Counter[str] = Counter()
    variants = [
        {
            "language": language,
            "interaction_style": interaction_style,
            "response_shape": response_shape,
        }
        for response_shape in response_shapes
        for language in languages
        for interaction_style in interaction_styles
    ]
    rng.shuffle(variants)

    draft_rows: list[dict[str, Any]] = []
    for index in range(count):
        variant = variants[index % len(variants)]
        row_rng = Random(rng.randint(0, 1_000_000_000))
        language = str(variant["language"])
        interaction_style = str(variant["interaction_style"])
        prompt_seed = None
        scenario_kind = row_rng.choice(scenario_kind_pool(language))  # type: ignore[arg-type]
        topic = row_rng.choice(topic_pool(language))  # type: ignore[arg-type]
        if language in prompt_samplers:
            prompt_seed = _prepare_prompt_seed(sample_balanced_prompt_seed(prompt_samplers[language], rng=row_rng))
            scenario_kind = "source_prompt"
            topic = ""
        instruction_surface = _cycle_value(
            instruction_surface_orders[language],
            instruction_surface_offsets,
            language,
        )
        follow_up_surface = None
        if interaction_style == "multi_turn_isolation" and isinstance(prompt_seed, dict):
            follow_up_surface = _cycle_value(
                follow_up_surface_orders[language],
                follow_up_surface_offsets,
                language,
            )
        draft_rows.append(
            {
                "language": language,
                "interaction_style": interaction_style,
                "preferred_response_shape": _response_shape(variant["response_shape"]),
                "scenario_kind": scenario_kind,
                "topic": topic,
                "prompt_seed": prompt_seed,
                "instruction_surface": instruction_surface,
                "follow_up_surface": follow_up_surface,
                "planning_seed": row_rng.randint(0, 1_000_000_000),
            }
        )

    rows = await _finalize_plan_rows(
        draft_rows,
        prompt_profiler=prompt_profiler,
        stress_probability=stress_probability,
        min_constraints=min_constraints,
        max_constraints=max_constraints,
        generation=generation,
        response_shapes=available_shapes(),
    )

    return {
        "mode": "planned_rows",
        "rows": rows,
        "benchmark": "ifbench",
        "language_counts": dict(Counter(str(row["language"]) for row in rows)),
        "interaction_style_counts": dict(Counter(str(row["interaction_style"]) for row in rows)),
        "response_shape_counts": dict(Counter(str(row["response_shape"]) for row in rows)),
        "prompt_source_counts": dict(Counter(prompt_source_label_for_row(row) for row in rows)),
        "prompt_task_type_counts": dict(
            Counter(str(row["prompt_seed"]["task_type"]) for row in rows if isinstance(row.get("prompt_seed"), dict))
        ),
        "prompt_length_bucket_counts": dict(
            Counter(str(row["prompt_seed"]["length_bucket"]) for row in rows if isinstance(row.get("prompt_seed"), dict))
        ),
        "constraint_counts": dict(
            Counter(
                str(constraint["id"])
                for row in rows
                for constraint in row["constraints"]
            )
        ),
    }


async def _finalize_plan_rows(
    draft_rows: list[dict[str, Any]],
    *,
    prompt_profiler: LLM | None,
    stress_probability: float,
    min_constraints: int,
    max_constraints: int,
    generation: dict[str, Any],
    response_shapes: tuple[ResponseShape, ...],
) -> list[dict[str, Any]]:
    profiled_rows = await _profile_draft_rows(
        draft_rows,
        prompt_profiler=prompt_profiler,
        generation=generation,
        response_shapes=response_shapes,
    )

    rows: list[dict[str, Any]] = []
    for draft in profiled_rows:
        prompt_seed = draft.get("prompt_seed")
        planning_rng = Random(int(draft["planning_seed"]))
        shape = _planned_response_shape(
            _response_shape(draft["preferred_response_shape"]),
            prompt_seed,
            rng=planning_rng,
            stress_probability=stress_probability,
        )
        constraints = sample_constraints(
            planning_rng,
            language=str(draft["language"]),  # type: ignore[arg-type]
            shape=shape,
            min_count=min_constraints,
            max_count=max_constraints,
            prompt_seed=prompt_seed,
        )
        rows.append(
            {
                "language": str(draft["language"]),
                "interaction_style": str(draft["interaction_style"]),
                "response_shape": shape,
                "scenario_kind": str(draft["scenario_kind"]),
                "topic": str(draft["topic"]),
                "prompt_seed": prompt_seed,
                "instruction_surface": draft.get("instruction_surface"),
                "follow_up_surface": draft.get("follow_up_surface"),
                "constraints": constraints,
            }
        )

    return rows


async def _profile_draft_rows(
    draft_rows: list[dict[str, Any]],
    *,
    prompt_profiler: LLM | None,
    generation: dict[str, Any],
    response_shapes: tuple[ResponseShape, ...],
) -> list[dict[str, Any]]:
    if prompt_profiler is None:
        return draft_rows

    prompt_map: dict[str, dict[str, Any]] = {}
    for row in draft_rows:
        prompt_seed = row.get("prompt_seed")
        if not isinstance(prompt_seed, dict):
            continue
        prompt_map.setdefault(_prompt_seed_key(prompt_seed), prompt_seed)

    if not prompt_map:
        return draft_rows

    profiled: dict[str, dict[str, Any]] = {}
    items = list(prompt_map.items())

    async def worker(index: int, item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        del index
        key, prompt_seed = item
        try:
            profile = await profile_prompt_seed(
                prompt_profiler,
                prompt_seed,
                allowed_shapes=response_shapes,
            )
        except Exception as error:
            fallback = dict(prompt_seed)
            fallback["profile_source"] = "heuristic"
            fallback["profile_error"] = _error_reason(error)
            return key, fallback

        merged = dict(prompt_seed)
        merged.update(profile)
        return key, merged

    async for key, prompt_seed in map_async_ordered(
        items,
        worker,
        concurrency=_prompt_profile_concurrency(generation, prompt_profiler),
        total=len(items),
    ):
        profiled[key] = prompt_seed

    rows: list[dict[str, Any]] = []
    for row in draft_rows:
        prompt_seed = row.get("prompt_seed")
        if not isinstance(prompt_seed, dict):
            rows.append(row)
            continue

        updated = dict(row)
        updated["prompt_seed"] = profiled[_prompt_seed_key(prompt_seed)]
        rows.append(updated)

    return rows


def _generate_rows(plan: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    writer = _load_scenario_writer(cfg)
    answer_teacher = _load_answer_teacher(cfg)
    generation = cfg["generation"]
    return asyncio.run(
        _generate_rows_async(
            plan["rows"],
            writer=writer,
            answer_teacher=answer_teacher,
            scenario_temperature=float(generation.get("scenario_temperature", 0.4)),
            answer_temperature=float(generation.get("answer_temperature", 0.7)),
            concurrency=_row_generation_concurrency(
                generation,
                writer=writer,
                answer_teacher=answer_teacher,
            ),
        )
    )


async def _generate_rows_async(
    row_plans: list[dict[str, Any]],
    *,
    writer: LLM,
    answer_teacher: LLM | None,
    scenario_temperature: float,
    answer_temperature: float,
    concurrency: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    async def worker(index: int, row_plan: dict[str, Any]) -> dict[str, Any]:
        try:
            return await _generate_row_async(
                index,
                row_plan,
                writer=writer,
                answer_teacher=answer_teacher,
                scenario_temperature=scenario_temperature,
                answer_temperature=answer_temperature,
            )
        except Exception as error:
            return _failed_row(
                index,
                row_plan,
                writer=writer,
                error=error,
            )

    async for row in map_async_ordered(
        row_plans,
        worker,
        concurrency=concurrency,
        total=len(row_plans),
    ):
        rows.append(row)

    return rows


def _row_generation_concurrency(
    generation: dict[str, Any],
    *,
    writer: LLM,
    answer_teacher: LLM | None,
) -> int:
    configured = generation.get("row_concurrency")
    if configured is not None:
        return max(int(configured), 1)

    writer_limit = _model_max_concurrency(writer)
    teacher_limit = _model_max_concurrency(answer_teacher)
    return max(writer_limit, teacher_limit, 1)


def _model_max_concurrency(model: LLM | None) -> int:
    if model is None:
        return 0

    runtime = getattr(model, "runtime", None)
    limit = getattr(runtime, "max_concurrency", None)
    if limit is None:
        return 1
    return max(int(limit), 1)


def _prompt_profile_concurrency(
    generation: dict[str, Any],
    prompt_profiler: LLM | None,
) -> int:
    configured = generation.get("prompt_profile_concurrency")
    if configured is not None:
        return max(int(configured), 1)
    return max(_model_max_concurrency(prompt_profiler), 1)


def _prepare_prompt_seed(prompt_seed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(prompt_seed, dict):
        return None
    prepared = dict(prompt_seed)
    prepared.setdefault("profile_source", "heuristic")
    return prepared


def _prompt_seed_key(prompt_seed: dict[str, Any]) -> str:
    language = str(prompt_seed.get("language", ""))
    text = str(prompt_seed.get("text", "")).strip()
    return f"{language}\n{text}"


async def _generate_row_async(
    index: int,
    row_plan: dict[str, Any],
    *,
    writer: LLM,
    answer_teacher: LLM | None,
    scenario_temperature: float,
    answer_temperature: float,
) -> dict[str, Any]:
    language = _language_code(row_plan["language"])
    constraints = [dict(constraint) for constraint in row_plan["constraints"]]
    prompt_seed = row_plan.get("prompt_seed")
    selected_prompt_keywords: list[str] = []
    constraints, selected_prompt_keywords = await _align_constraints_to_prompt(
        writer,
        row_plan,
        constraints,
    )
    scenario_writer_used = not (
        isinstance(prompt_seed, dict) and str(row_plan["interaction_style"]) == "single_turn"
    )
    constraint_lines = render_constraint_lines(constraints, language=language)
    instruction_surface = _string_or_none(row_plan.get("instruction_surface"))
    instruction_block = render_instruction_block(
        language,
        constraint_lines,
        surface_key=instruction_surface,
    )

    scenario_plan = dict(row_plan)
    scenario_plan["constraint_lines"] = constraint_lines
    scenario_writer_error = ""
    try:
        bundle = await generate_scenario_bundle(writer, scenario_plan, temperature=scenario_temperature)
    except Exception as error:
        bundle = fallback_scenario_bundle(row_plan)
        scenario_writer_used = False
        scenario_writer_error = _error_reason(error)

    messages = materialize_messages(bundle, row_plan, instruction_block=instruction_block)
    row = _build_row(
        index,
        row_plan,
        language=language,
        constraints=constraints,
        constraint_lines=constraint_lines,
        instruction_block=instruction_block,
        bundle=bundle,
        messages=messages,
        sources=_row_sources(
            writer=writer,
            row_plan=row_plan,
            scenario_writer_used=scenario_writer_used,
        ),
    )
    if scenario_writer_error:
        hidden = dict(row["hidden"])
        hidden["scenario_writer_error"] = scenario_writer_error
        if selected_prompt_keywords:
            hidden["selected_prompt_keywords"] = selected_prompt_keywords
        row["hidden"] = hidden
    elif selected_prompt_keywords:
        hidden = dict(row["hidden"])
        hidden["selected_prompt_keywords"] = selected_prompt_keywords
        row["hidden"] = hidden

    if answer_teacher is None:
        return row

    try:
        target = await answer_teacher.achat(messages, temperature=answer_temperature)
    except Exception as error:
        return _with_failed_target(row, answer_teacher=answer_teacher, error=error)

    target_text = str(target).strip()
    if not target_text:
        return _with_failed_target(
            row,
            answer_teacher=answer_teacher,
            error=ValueError("answer_teacher returned an empty response"),
        )

    return _with_target(row, answer_teacher=answer_teacher, target_text=target_text)


def _build_row(
    index: int,
    row_plan: dict[str, Any],
    *,
    language: LanguageCode,
    constraints: list[dict[str, Any]],
    constraint_lines: list[str],
    instruction_block: str,
    bundle: dict[str, Any],
    messages: list[dict[str, str]],
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    instruction_surface = _string_or_none(row_plan.get("instruction_surface"))
    follow_up_surface = _string_or_none(row_plan.get("follow_up_surface"))
    prompt_seed = row_plan.get("prompt_seed")

    return {
        "id": f"instruction-following-{index:05d}",
        "prompt": render_messages(messages, language=language),
        "messages": messages,
        "sources": sources,
        "meta": {
            "family": "instruction_following",
            "benchmark": "ifbench",
            "language": language,
            "interaction_style": str(row_plan["interaction_style"]),
            "response_shape": str(row_plan["response_shape"]),
            "instruction_count": len(constraints),
            "constraint_categories": constraint_categories(constraints),
            "scenario_kind": str(row_plan["scenario_kind"]),
            "prompt_source": prompt_source_label_for_row(row_plan),
            "prompt_task_type": _prompt_seed_meta(row_plan, "task_type"),
            "prompt_length_bucket": _prompt_seed_meta(row_plan, "length_bucket"),
            "prompt_semantic_rigidity": _prompt_seed_meta(row_plan, "semantic_rigidity"),
            "prompt_profile_source": _prompt_seed_meta(row_plan, "profile_source"),
            "prompt_naturalness_confidence": _prompt_seed_meta(row_plan, "naturalness_confidence"),
            "instruction_surface": instruction_surface or "default",
            "follow_up_surface": follow_up_surface or "",
        },
        "hidden": {
            "constraints": constraints,
            "instruction_lines": constraint_lines,
            "instruction_block": instruction_block,
            "scenario_bundle": bundle,
            "topic": str(row_plan.get("topic", "")),
            "prompt_seed": prompt_seed,
            "prompt_semantic_rigidity": _prompt_seed_meta(row_plan, "semantic_rigidity"),
            "prompt_profile_source": _prompt_seed_meta(row_plan, "profile_source"),
            "instruction_surface": instruction_surface or "default",
            "follow_up_surface": follow_up_surface or "",
        },
    }


async def _align_constraints_to_prompt(
    writer: LLM,
    row_plan: dict[str, Any],
    constraints: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    prompt_seed = row_plan.get("prompt_seed")
    if not isinstance(prompt_seed, dict):
        return constraints, []

    candidate_keywords = [
        str(item)
        for item in prompt_seed.get("semantic_keywords", [])
        if isinstance(item, str) and item
    ]
    keyword_budget = _prompt_keyword_budget(constraints)
    if keyword_budget <= 0 or not candidate_keywords:
        return constraints, []

    try:
        selected_keywords = await select_prompt_keywords(
            writer,
            row_plan,
            candidate_keywords=candidate_keywords,
            target_count=keyword_budget,
        )
    except Exception:
        selected_keywords = candidate_keywords[:keyword_budget]

    if not selected_keywords:
        return constraints, []
    return _apply_prompt_keywords(constraints, selected_keywords), selected_keywords


def _prompt_keyword_budget(constraints: list[dict[str, Any]]) -> int:
    budget = 0
    for constraint in constraints:
        constraint_id = str(constraint["id"])
        params = dict(constraint.get("params", {}))
        if constraint_id == "count:keywords_multiple":
            budget += len(list(params.get("keywords", [])))
        elif constraint_id in {"words:ordered_keywords", "words:word_positions"}:
            budget += 2
        elif constraint_id in {"start_end:first_word", "start_end:last_word"}:
            budget += 1
    return budget


def _apply_prompt_keywords(
    constraints: list[dict[str, Any]],
    selected_keywords: list[str],
) -> list[dict[str, Any]]:
    if not selected_keywords:
        return constraints

    cursor = 0

    def take(count: int) -> list[str]:
        nonlocal cursor
        values: list[str] = []
        for _ in range(count):
            values.append(selected_keywords[cursor % len(selected_keywords)])
            cursor += 1
        return values

    updated: list[dict[str, Any]] = []
    for constraint in constraints:
        constraint_id = str(constraint["id"])
        params = dict(constraint.get("params", {}))

        if constraint_id == "count:keywords_multiple":
            counts = [int(pair["count"]) for pair in params.get("keywords", [])]
            words = take(len(counts))
            params["keywords"] = [
                {"word": word, "count": count}
                for word, count in zip(words, counts, strict=True)
            ]
        elif constraint_id == "words:ordered_keywords":
            first, second = take(2)
            params["first"] = first
            params["second"] = second
        elif constraint_id == "start_end:first_word":
            params["word"] = take(1)[0]
        elif constraint_id == "start_end:last_word":
            params["word"] = take(1)[0]
        elif constraint_id == "words:word_positions":
            words = take(len(list(params.get("pairs", []))))
            params["pairs"] = [
                {"position": int(pair["position"]), "word": word}
                for pair, word in zip(params.get("pairs", []), words, strict=True)
            ]

        updated.append({"id": constraint_id, "params": params})
    return updated


def _with_target(
    row: dict[str, Any],
    *,
    answer_teacher: LLM,
    target_text: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated["messages"] = [
        *row["messages"],
        {"role": "assistant", "content": target_text},
    ]
    updated["sources"] = [
        *row["sources"],
        {"kind": "target_model", "value": _model_name(answer_teacher)},
    ]
    meta = dict(updated["meta"])
    meta["target_source"] = "answer_teacher"
    updated["meta"] = meta
    return updated


def _with_failed_target(
    row: dict[str, Any],
    *,
    answer_teacher: LLM,
    error: Exception,
) -> dict[str, Any]:
    updated = dict(row)
    updated["sources"] = [
        *row["sources"],
        {"kind": "target_model", "value": _model_name(answer_teacher)},
    ]
    meta = dict(updated["meta"])
    meta["target_source"] = "answer_teacher_failed"
    updated["meta"] = meta
    hidden = dict(updated["hidden"])
    hidden["generation_error"] = _error_reason(error)
    updated["hidden"] = hidden
    return updated


def _failed_row(
    index: int,
    row_plan: dict[str, Any],
    *,
    writer: LLM,
    error: Exception,
) -> dict[str, Any]:
    language = _language_code(row_plan["language"])
    constraints = [dict(constraint) for constraint in row_plan["constraints"]]
    constraint_lines = render_constraint_lines(constraints, language=language)
    instruction_surface = _string_or_none(row_plan.get("instruction_surface"))
    instruction_block = render_instruction_block(
        language,
        constraint_lines,
        surface_key=instruction_surface,
    )
    bundle = fallback_scenario_bundle(row_plan)
    messages = _fallback_messages(bundle, row_plan, instruction_block=instruction_block)
    row = _build_row(
        index,
        row_plan,
        language=language,
        constraints=constraints,
        constraint_lines=constraint_lines,
        instruction_block=instruction_block,
        bundle=bundle,
        messages=messages,
        sources=_row_sources(
            writer=writer,
            row_plan=row_plan,
            scenario_writer_used=False,
        ),
    )
    meta = dict(row["meta"])
    meta["row_status"] = "generation_failed"
    row["meta"] = meta
    hidden = dict(row["hidden"])
    hidden["generation_error"] = _error_reason(error)
    row["hidden"] = hidden
    return row


def _load_rows(result: BuildResult) -> list[dict[str, Any]]:
    dataset_path = Path(result.artifacts["dataset"].path)
    return store.read_jsonl(dataset_path)


def _load_plan(result: BuildResult) -> dict[str, Any]:
    plan_path = Path(result.artifacts["plan"].path)
    return read_json(plan_path)


def _load_verified_rows(result: BuildResult) -> list[dict[str, Any]]:
    verified_path = Path(result.run_dir) / "outputs" / "verified.jsonl"
    if verified_path.exists():
        return store.read_jsonl(verified_path)
    return verify_rows(_load_rows(result), plan=_load_plan(result))["rows"]


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _load_scenario_writer(cfg: dict[str, Any]) -> LLM:
    models = cfg.get("models", {})
    assert "scenario_writer" in models, "instruction_following requires models.scenario_writer"
    loaded = common_model.load_clients({"scenario_writer": models["scenario_writer"]})
    writer = loaded["scenario_writer"]
    assert isinstance(writer, LLM), "models.scenario_writer must resolve to an LLM"
    return writer


def _load_prompt_profiler(cfg: dict[str, Any]) -> LLM | None:
    generation = cfg["generation"]
    if not generation.get("use_prompt_profiler", True):
        return None

    models = cfg.get("models", {})
    if "prompt_profiler" not in models:
        return _load_scenario_writer(cfg)

    loaded = common_model.load_clients({"prompt_profiler": models["prompt_profiler"]})
    profiler = loaded["prompt_profiler"]
    assert isinstance(profiler, LLM), "models.prompt_profiler must resolve to an LLM"
    return profiler


def _load_answer_teacher(cfg: dict[str, Any]) -> LLM | None:
    generation = cfg["generation"]
    if not generation.get("attach_targets", False):
        return None

    models = cfg.get("models", {})
    assert "answer_teacher" in models, "generation.attach_targets requires models.answer_teacher"
    loaded = common_model.load_clients({"answer_teacher": models["answer_teacher"]})
    teacher = loaded["answer_teacher"]
    assert isinstance(teacher, LLM), "models.answer_teacher must resolve to an LLM"
    return teacher


def _messages_well_formed(row: dict[str, Any]) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False

    prompt_roles = ["user"] if row["meta"]["interaction_style"] == "single_turn" else ["user", "assistant", "user"]
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    allowed_roles = [prompt_roles, [*prompt_roles, "assistant"]]
    if roles not in allowed_roles:
        return False
    return all(str(message.get("content", "")).strip() for message in messages)


def _instruction_block_matches_hidden(row: dict[str, Any]) -> bool:
    messages = row["messages"]
    final_user = _final_user_message(messages)
    expected = str(row["hidden"]["instruction_block"])
    return expected in final_user


def _constraints_supported(row: dict[str, Any]) -> bool:
    language = _language_code(row["meta"]["language"])
    shape = _response_shape(row["meta"]["response_shape"])

    constraints = row["hidden"]["constraints"]
    if not isinstance(constraints, list) or not constraints:
        return False

    for constraint in constraints:
        definition = constraint_definition(str(constraint["id"]))
        if language not in definition.supported_languages:
            return False
        if shape not in definition.compatible_shapes:
            return False
    return True


def _response_present(row: dict[str, Any]) -> bool:
    return bool(_response_text(row))


def _generation_succeeded(row: dict[str, Any]) -> bool:
    return str(row.get("meta", {}).get("row_status", "")) != "generation_failed"


def _response_follows_strict(row: dict[str, Any]) -> dict[str, Any]:
    language = _language_code(row["meta"]["language"])
    results = check_constraints_strict(_response_text(row), row["hidden"]["constraints"], language=language)
    return {"passed": all(results), "instruction_results": results}


def _response_follows_loose(row: dict[str, Any]) -> dict[str, Any]:
    language = _language_code(row["meta"]["language"])
    results = check_constraints_loose(_response_text(row), row["hidden"]["constraints"], language=language)
    return {"passed": all(results), "instruction_results": results}


def _dataset_checks(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "row_count": {
            "passed": len(rows) == len(plan["rows"]),
            "expected": len(plan["rows"]),
            "observed": len(rows),
        },
        "language_counts": _count_check(
            expected=plan["language_counts"],
            observed=Counter(str(row["meta"]["language"]) for row in rows),
        ),
        "interaction_style_counts": _count_check(
            expected=plan["interaction_style_counts"],
            observed=Counter(str(row["meta"]["interaction_style"]) for row in rows),
        ),
        "response_shape_counts": _count_check(
            expected=plan["response_shape_counts"],
            observed=Counter(str(row["meta"]["response_shape"]) for row in rows),
        ),
        "constraint_counts": _count_check(
            expected=plan["constraint_counts"],
            observed=Counter(
                str(constraint["id"])
                for row in rows
                for constraint in row["hidden"]["constraints"]
            ),
        ),
        "unique_prompts": {
            "passed": len({str(row["prompt"]) for row in rows}) == len(rows),
            "expected": len(rows),
            "observed": len({str(row["prompt"]) for row in rows}),
        },
    }
    if "prompt_source_counts" in plan:
        checks["prompt_source_counts"] = _count_check(
            expected=plan["prompt_source_counts"],
            observed=Counter(str(row["meta"].get("prompt_source", "synthetic")) for row in rows),
        )
    if plan.get("prompt_task_type_counts"):
        checks["prompt_task_type_counts"] = _count_check(
            expected=plan["prompt_task_type_counts"],
            observed=Counter(str(row["meta"].get("prompt_task_type", "synthetic")) for row in rows),
        )
    if plan.get("prompt_length_bucket_counts"):
        checks["prompt_length_bucket_counts"] = _count_check(
            expected=plan["prompt_length_bucket_counts"],
            observed=Counter(str(row["meta"].get("prompt_length_bucket", "synthetic")) for row in rows),
        )
    return {
        "passed": all(bool(check["passed"]) for check in checks.values()),
        "checks": checks,
    }


def _count_check(*, expected: dict[str, int], observed: Counter[str]) -> dict[str, Any]:
    observed_dict = dict(observed)
    return {
        "passed": observed_dict == expected,
        "expected": expected,
        "observed": observed_dict,
    }


def _row_failed(row: dict[str, Any]) -> bool:
    checks = row.get("checks")
    if not isinstance(checks, dict):
        return False
    return any(not _check_passed(value) for value in checks.values())


def _has_expected_responses(rows: list[dict[str, Any]]) -> bool:
    return any(_expects_response_check(row) for row in rows)


def _expects_response_check(row: dict[str, Any]) -> bool:
    if _response_text(row):
        return True
    target_source = str(row.get("meta", {}).get("target_source", ""))
    return target_source in {"answer_teacher", "answer_teacher_failed"}


def _check_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    assert isinstance(value, dict), "check value must be a bool or a status mapping"
    if "passed" in value:
        return bool(value["passed"])
    if "ok" in value:
        return bool(value["ok"])
    raise AssertionError("check status mapping must contain 'passed' or 'ok'")


def _response_text(row: dict[str, Any]) -> str:
    target = str(row.get("target", "")).strip()
    if target:
        return target

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant":
        return ""
    return str(last.get("content", "")).strip()


def _final_user_message(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _prompt_seed_meta(row_plan: dict[str, Any], key: str) -> str:
    prompt_seed = row_plan.get("prompt_seed")
    if not isinstance(prompt_seed, dict):
        return "synthetic"
    value = prompt_seed.get(key)
    if value is None:
        return "synthetic"
    return str(value)


def _prompt_seed_count(prompt_seed: dict[str, Any], key: str) -> int | None:
    value = prompt_seed.get(key)
    if value is None:
        return None
    assert isinstance(value, int) and value > 0, f"prompt seed {key} must be a positive integer"
    return value


def _strip_hidden(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "hidden"}


def _split_rows(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_at = int(len(rows) * train_fraction)
    return rows[:split_at], rows[split_at:]


def _publish_dir(result: BuildResult, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    return reports_root() / result.pack / result.run_id


def _load_or_compute(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return fallback


def _positive_int(record: dict[str, Any], key: str, *, default: int) -> int:
    value = record.get(key)
    if value is None:
        return default
    assert isinstance(value, int) and value > 0, f"{key} must be a positive integer"
    return value


def _languages(generation: dict[str, Any]) -> tuple[LanguageCode, ...]:
    raw = generation.get("languages")
    if raw is None:
        raw = list(available_languages())
    assert isinstance(raw, list) and raw, "generation languages must be a non-empty list"
    return tuple(_language_code(item) for item in raw)


def _interaction_styles(generation: dict[str, Any]) -> tuple[str, ...]:
    raw = generation.get("interaction_styles")
    if raw is None:
        raw = [generation.get("interaction_style", "multi_turn_isolation")]
    assert isinstance(raw, list) and raw, "generation interaction_styles must be a non-empty list"
    values = tuple(str(item) for item in raw)
    supported = {"single_turn", "multi_turn_isolation"}
    assert set(values).issubset(supported), "unsupported interaction style requested"
    weights = generation.get("interaction_style_weights")
    if weights is None:
        return values

    assert isinstance(weights, dict) and weights, "generation interaction_style_weights must be a non-empty mapping"
    weighted: list[str] = []
    for value in values:
        weight = weights.get(value, 1)
        assert isinstance(weight, int) and weight > 0, f"interaction_style_weights[{value}] must be a positive integer"
        weighted.extend([value] * weight)
    return tuple(weighted)


def _response_shapes(generation: dict[str, Any]) -> tuple[ResponseShape, ...]:
    raw = generation.get("response_shapes")
    if raw is None:
        raw = list(available_shapes())
    assert isinstance(raw, list) and raw, "generation response_shapes must be a non-empty list"
    return tuple(_response_shape(item) for item in raw)


def _language_code(value: object) -> LanguageCode:
    assert value in {"en", "da"}, f"unsupported language: {value}"
    return str(value)  # type: ignore[return-value]


def _response_shape(value: object) -> ResponseShape:
    assert value in {
        "plain_text",
        "numbered_list",
        "bullet_list",
        "json_object",
        "xml_object",
        "separated_responses",
        "indented_lines",
    }, (
        f"unsupported response shape: {value}"
    )
    return str(value)  # type: ignore[return-value]


def _planned_response_shape(
    preferred_shape: ResponseShape,
    prompt_seed: dict[str, Any] | None,
    *,
    rng: Random,
    stress_probability: float,
) -> ResponseShape:
    if not isinstance(prompt_seed, dict):
        return preferred_shape

    preferred, stress = _shape_preferences(prompt_seed)
    if preferred_shape in preferred:
        return preferred_shape
    if preferred_shape in stress and rng.random() < stress_probability:
        return preferred_shape
    if preferred:
        return rng.choice(preferred)
    return preferred_shape


def _shape_preferences(prompt_seed: dict[str, Any]) -> tuple[list[ResponseShape], list[ResponseShape]]:
    task_type = str(prompt_seed.get("task_type", "general_generation"))
    rigidity = str(prompt_seed.get("semantic_rigidity", "medium"))
    numeric_task = bool(prompt_seed.get("numeric_task"))
    sentence_count = _prompt_seed_count(prompt_seed, "requested_sentence_count")
    line_count = _prompt_seed_count(prompt_seed, "requested_line_count")
    item_count = _prompt_seed_count(prompt_seed, "requested_item_count")
    safe_response_shapes = [
        _response_shape(shape)
        for shape in prompt_seed.get("safe_response_shapes", [])
        if isinstance(shape, str)
    ]

    if safe_response_shapes:
        return safe_response_shapes, []

    if sentence_count is not None:
        return ["plain_text"], []

    if line_count is not None:
        preferred: list[ResponseShape] = ["plain_text", "bullet_list", "numbered_list", "separated_responses"]
        if rigidity == "open":
            preferred.append("indented_lines")
        return preferred, []

    if item_count is not None:
        if item_count == 2:
            preferred = ["separated_responses", "bullet_list", "numbered_list", "plain_text"]
        else:
            preferred = ["bullet_list", "numbered_list", "plain_text", "separated_responses"]
        stress = ["json_object", "xml_object"] if task_type == "classification" else []
        return preferred, stress

    if numeric_task:
        return ["json_object", "xml_object", "plain_text", "separated_responses"], ["bullet_list", "numbered_list"]

    if task_type == "translation":
        stress = ["json_object", "xml_object"] if rigidity != "rigid" else []
        return ["bullet_list", "numbered_list", "plain_text", "separated_responses"], stress

    if task_type == "classification":
        return ["json_object", "xml_object", "bullet_list", "numbered_list", "plain_text", "separated_responses"], []

    if task_type in {"analysis", "comparison"}:
        return ["plain_text", "bullet_list", "numbered_list", "json_object", "xml_object", "separated_responses"], []

    if task_type in {"explanation", "question_answering", "rewrite", "summarization"}:
        return ["plain_text", "json_object", "xml_object", "bullet_list", "numbered_list", "separated_responses"], []

    if task_type in {"listing", "recommendation"}:
        return ["bullet_list", "numbered_list", "plain_text", "separated_responses", "json_object", "xml_object"], []

    preferred = list(available_shapes())
    if rigidity in {"medium", "rigid"} and "indented_lines" in preferred:
        preferred.remove("indented_lines")
        return preferred, ["indented_lines"]
    return preferred, []


def _shuffled_values(rng: Random, values: list[str]) -> list[str]:
    shuffled = list(values)
    rng.shuffle(shuffled)
    return shuffled


def _cycle_value(values: list[str], offsets: Counter[str], key: str) -> str:
    assert values, "surface values must not be empty"
    index = offsets[key] % len(values)
    offsets[key] += 1
    return values[index]


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _error_reason(error: Exception) -> str:
    detail = str(error).strip()
    if not detail:
        return error.__class__.__name__
    return f"{error.__class__.__name__}: {detail}"


def _fallback_messages(
    bundle: dict[str, Any],
    row_plan: dict[str, Any],
    *,
    instruction_block: str,
) -> list[dict[str, str]]:
    interaction_style = str(row_plan["interaction_style"])
    if interaction_style == "single_turn":
        user_prefix = str(bundle.get("user_prefix", "")).strip()
        final_user = instruction_block if not user_prefix else f"{user_prefix}\n\n{instruction_block}"
        return [{"role": "user", "content": final_user}]

    base_user = str(bundle.get("base_user", "")).strip()
    assistant_reply = str(bundle.get("assistant_reply", "")).strip()
    final_user_prefix = str(bundle.get("final_user_prefix", "")).strip()
    final_user = instruction_block if not final_user_prefix else f"{final_user_prefix}\n\n{instruction_block}"
    return [
        {"role": "user", "content": base_user},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": final_user},
    ]


def _row_sources(
    *,
    writer: LLM,
    row_plan: dict[str, Any],
    scenario_writer_used: bool,
) -> list[dict[str, str]]:
    sources = [{"kind": "benchmark", "value": "ifbench"}]
    prompt_source = row_plan.get("prompt_seed")
    if isinstance(prompt_source, dict):
        label = str(prompt_source.get("source_label", "")).strip()
        if label:
            sources.append({"kind": "prompt_source", "value": label})
    if scenario_writer_used:
        sources.append({"kind": "scenario_model", "value": _model_name(writer)})
    return sources


def _model_name(model: LLM) -> str:
    name = getattr(model, "model", "")
    return str(name) if name else "unknown-model"
