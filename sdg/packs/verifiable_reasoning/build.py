import asyncio
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import diversity as common_diversity
from sdg.commons import eval as common_eval
from sdg.commons import model as common_model
from sdg.commons import publish as common_publish
from sdg.commons.run import load, run
from sdg.commons.run_log import log_event, write_snapshot
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json
from sdg.commons.work_queue import map_async_ordered, map_async_unordered
from sdg.packs.verifiable_reasoning import (
    blocked_star,
    countdownequal,
    cryptarithmetic,
    futoshiki,
    hitori,
    jugpuzzle,
    kakurasu,
    knightsandknaves,
    lightuppuzzle,
    lineup,
    numbrix,
    setsplitting,
    skyscraper,
    starbattle,
    zebra,
)

FAMILIES = {
    "blocked_star_logic": blocked_star,
    "countdownequal_logic": countdownequal,
    "cryptarithmetic_logic": cryptarithmetic,
    "futoshiki_logic": futoshiki,
    "hitori_logic": hitori,
    "jugpuzzle_logic": jugpuzzle,
    "knightsandknaves_logic": knightsandknaves,
    "kakurasu_logic": kakurasu,
    "lightuppuzzle_logic": lightuppuzzle,
    "lineup_logic": lineup,
    "numbrix_logic": numbrix,
    "setsplitting_logic": setsplitting,
    "skyscraper_logic": skyscraper,
    "starbattle_logic": starbattle,
    "zebra_logic": zebra,
}
DEFAULT_ALL_FAMILIES = (
    "zebra_logic",
    "lineup_logic",
    "countdownequal_logic",
    "cryptarithmetic_logic",
    "futoshiki_logic",
    "skyscraper_logic",
    "numbrix_logic",
    "setsplitting_logic",
    "hitori_logic",
    "jugpuzzle_logic",
    "knightsandknaves_logic",
    "kakurasu_logic",
    "blocked_star_logic",
    "starbattle_logic",
)
ANSWER_TEACHER_DISABLED_FAMILIES = {"lightuppuzzle_logic"}
ANSWER_TEACHER_TIMEOUT_SECONDS = 3000.0
ANSWER_TEACHER_MAX_TOKENS = 32000


def build(cfg: dict[str, Any]) -> BuildResult:
    return run(
        _build_run,
        pack="verifiable_reasoning",
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
    write_json(verification["dataset_checks"], outputs_dir / "dataset_checks.json")

    write_json(verification["metrics"], outputs_dir / "metrics.json")
    write_json(verification["failure_summary"], outputs_dir / "failure_summary.json")
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
    verified_rows = _verify_problem_rows(rows)
    response_checks_applied = _has_expected_responses(rows)
    if response_checks_applied:
        verified_rows = common_eval.verify(verified_rows, _response_reasoning_quality, name="response_reasoning_quality")
        verified_rows = common_eval.verify(verified_rows, _response_parseable, name="response_parseable")
        verified_rows = common_eval.verify(verified_rows, _response_correct, name="response_correct")

    failures = [row for row in verified_rows if not _row_passes(row)]
    dataset_checks = _dataset_checks(verified_rows, plan) if plan is not None else {}
    metrics = common_eval.aggregate_metrics(verified_rows)
    failure_summary = common_eval.summarize_failures(verified_rows)

    return {
        "rows": verified_rows,
        "failures": failures,
        "metrics": metrics,
        "failure_summary": failure_summary,
        "dataset_checks": dataset_checks,
        "response_checks_applied": response_checks_applied,
    }


def attach_targets(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    generation = cfg.get("generation", {})
    if not generation.get("attach_targets", False):
        return rows

    if not any(_answer_teacher_enabled(row["meta"]["family"]) for row in rows):
        return [_skip_answer_teacher_row(row) for row in rows]

    teacher, temperature, max_attempts, concurrency, max_tokens = _answer_teacher_settings(cfg)
    log_event(
        "verifiable_reasoning",
        "answer_teacher_started",
        rows=len(rows),
        concurrency=concurrency,
        max_attempts=max_attempts,
        max_tokens=max_tokens,
    )
    attached_rows = asyncio.run(
        _attach_targets_async(
            rows,
            teacher,
            temperature=temperature,
            max_attempts=max_attempts,
            concurrency=concurrency,
            max_tokens=max_tokens,
        )
    )
    log_event(
        "verifiable_reasoning",
        "answer_teacher_completed",
        rows=len(attached_rows),
        concurrency=concurrency,
        max_attempts=max_attempts,
        max_tokens=max_tokens,
    )
    return attached_rows


def summarize(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    dataset_checks_path = outputs_dir / "dataset_checks.json"
    rejections_path = outputs_dir / "rejections.jsonl"
    metrics = read_json(metrics_path) if metrics_path.exists() else common_eval.aggregate_metrics(rows)
    dataset_checks = read_json(dataset_checks_path) if dataset_checks_path.exists() else {}

    family_counts = Counter(row["meta"]["family"] for row in rows)
    language_counts = Counter(row["meta"]["prompt_language"] for row in rows)
    clue_counts = Counter(row["meta"]["clue_count"] for row in rows)
    difficulty_counts = Counter(row["meta"].get("difficulty") for row in rows if row["meta"].get("difficulty"))
    prompt_style_counts = Counter(row["meta"].get("prompt_style") for row in rows if row["meta"].get("prompt_style"))
    recipe_counts = Counter(row["meta"].get("recipe_id") for row in rows if row["meta"].get("recipe_id"))
    surface_keys = sorted(
        key
        for key in {item for row in rows for item in row["meta"]}
        if key.startswith("surface_")
    )
    surface_counts = {
        key: dict(Counter(row["meta"][key] for row in rows if row["meta"].get(key)))
        for key in surface_keys
    }

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "families": sorted(family_counts),
        "family_counts": dict(family_counts),
        "language_counts": dict(language_counts),
        "clue_counts": dict(clue_counts),
        "difficulty_counts": dict(difficulty_counts),
        "prompt_style_counts": dict(prompt_style_counts),
        "surface_counts": surface_counts,
        "recipe_counts": dict(recipe_counts),
        "artifacts": sorted(result.artifacts),
        "rejection_rows": store.jsonl_count(rejections_path),
        "metrics": metrics,
        "dataset_checks": dataset_checks,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    generation_model = _published_generation_model(result, cfg)
    stamped_rows = [_with_generation_model_source(row, generation_model) for row in rows]
    passing_rows = [row for row in stamped_rows if _row_passes(row)]
    failures = [_strip_hidden(row) for row in stamped_rows if not _row_passes(row)]
    export_rows = [_strip_hidden(row) for row in passing_rows]
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
        "rows": len(export_rows),
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
    plan_path, plan = _load_or_create_plan(cfg, outputs_dir, seed)
    dataset_path = _write_dataset(cfg, outputs_dir, seed, plan)
    preview_rows = store.jsonl_prefix(dataset_path, limit=20)
    common_publish.write_preview(preview_rows, outputs_dir / "sample_preview.jsonl", n=20)
    rejections_path = outputs_dir / "rejections.jsonl"
    if rejections_path.exists():
        rejection_preview = store.jsonl_prefix(rejections_path, limit=20)
        common_publish.write_preview(rejection_preview, outputs_dir / "rejections_preview.jsonl", n=20)

    families = _plan_families(plan)
    languages = _plan_languages(plan)
    row_count = store.jsonl_count(dataset_path)
    artifacts = {
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={
                "rows": row_count,
                "families": families,
                "languages": languages,
            },
        ),
        "plan": Artifact(
            name="plan",
            path=str(plan_path),
            kind="json",
            meta={
                "rows": _plan_target_rows(plan),
                "families": families,
                "languages": languages,
            },
        ),
    }
    if rejections_path.exists():
        artifacts["rejections"] = Artifact(
            name="rejections",
            path=str(rejections_path),
            kind="jsonl",
            meta={
                "rows": store.jsonl_count(rejections_path),
                "families": families,
                "languages": languages,
            },
        )
    return artifacts


def _load_or_create_plan(
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
) -> tuple[Path, dict[str, Any]]:
    plan_path = outputs_dir / "plan.json"
    if plan_path.exists():
        return plan_path, read_json(plan_path)

    plan = _build_plan(cfg, seed)
    write_json(plan, plan_path)
    return plan_path, plan


def _build_plan(cfg: dict[str, Any], seed: int | None) -> dict[str, Any]:
    generation = cfg["generation"]
    rng = Random(seed if seed is not None else 0)
    if generation.get("attach_targets", False):
        return _build_success_target_plan(generation, rng)
    planned_rows = _plan_rows(generation, rng)
    return {
        "mode": "fixed_rows",
        "rows": planned_rows,
        "family_counts": dict(Counter(item["family"] for item in planned_rows)),
        "language_counts": dict(Counter(item["language"] for item in planned_rows)),
    }


def _build_success_target_plan(generation: dict[str, Any], rng: Random) -> dict[str, Any]:
    count = int(generation["count"])
    families = _resolve_families(generation)
    languages = _resolve_languages(generation)
    max_candidate_attempts = int(
        generation.get(
            "max_candidate_attempts_per_variant",
            max(count * 4, count + 50),
        )
    )
    assert max_candidate_attempts >= count, "generation.max_candidate_attempts_per_variant must be at least generation.count"

    variant_groups: list[list[dict[str, Any]]] = []
    variants: list[dict[str, Any]] = []
    candidate_index = 0

    for family in families:
        for language in languages:
            family_module = _family_module(family)
            planned_recipes = common_diversity.plan_from_catalog(
                max_candidate_attempts,
                list(family_module.recipe_catalog(language)),
                rng,
            )
            candidate_rows: list[dict[str, Any]] = []
            for attempt_index, recipe in enumerate(planned_recipes):
                candidate_plan = {
                    "family": family,
                    "language": language,
                    "candidate_index": candidate_index,
                    "attempt_index": attempt_index,
                    **dict(recipe),
                }
                candidate_rows.append(candidate_plan)
                candidate_index += 1
            variant_groups.append(candidate_rows)
            variants.append(
                {
                    "family": family,
                    "language": language,
                    "target_rows": count,
                    "candidate_rows": candidate_rows,
                }
            )

    _apply_surface_plans([item for group in variant_groups for item in group], rng)

    family_counts = {family: count * len(languages) for family in families}
    language_counts = {language: count * len(families) for language in languages}
    return {
        "mode": "success_targets",
        "count_per_variant": count,
        "max_candidate_attempts_per_variant": max_candidate_attempts,
        "variants": variants,
        "family_counts": family_counts,
        "language_counts": language_counts,
    }


def _write_dataset(
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
    plan: dict[str, Any],
) -> Path:
    if plan.get("mode") == "success_targets":
        return _write_success_target_dataset(cfg, outputs_dir, seed, plan)

    dataset_path = outputs_dir / "dataset.jsonl"
    planned_rows = list(plan["rows"])
    total = len(planned_rows)
    completed = store.jsonl_count(dataset_path)

    assert completed <= total, "dataset.jsonl has more rows than the saved plan"
    if completed == total:
        return dataset_path

    generation = cfg["generation"]
    if generation.get("attach_targets", False):
        _stream_rows_with_target_overlap(
            planned_rows,
            dataset_path,
            seed,
            cfg,
            completed=completed,
        )
    else:
        _stream_rows_from_plan(
            planned_rows,
            dataset_path,
            seed,
            completed=completed,
        )

    final_count = store.jsonl_count(dataset_path)
    assert final_count == total, "dataset.jsonl row count does not match the saved plan"
    return dataset_path


def _write_success_target_dataset(
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
    plan: dict[str, Any],
) -> Path:
    dataset_path = outputs_dir / "dataset.jsonl"
    rejections_path = outputs_dir / "rejections.jsonl"
    targets = _success_targets_by_variant(plan)
    accepted = _variant_counts_from_rows_path(dataset_path)
    rejected = _variant_counts_from_rows_path(rejections_path)
    total_target = sum(targets.values())

    if _success_targets_met(accepted, targets):
        write_snapshot(
            "verifiable_reasoning_answer_teacher",
            {
                "stage": "completed",
                "completed": total_target,
                "total": total_target,
                "rejected": sum(rejected.values()),
            },
            force=True,
        )
        return dataset_path

    teacher, temperature, max_attempts, concurrency, max_tokens = _answer_teacher_settings(cfg)
    log_event(
        "verifiable_reasoning",
        "answer_teacher_started",
        rows=total_target,
        concurrency=concurrency,
        max_attempts=max_attempts,
        max_tokens=max_tokens,
        mode="success_targets",
    )
    asyncio.run(
        _stream_success_target_rows_async(
            plan,
            dataset_path,
            rejections_path,
            seed,
            teacher,
            temperature=temperature,
            max_attempts=max_attempts,
            concurrency=concurrency,
            max_tokens=max_tokens,
            accepted=accepted,
            rejected=rejected,
        )
    )

    accepted = _variant_counts_from_rows_path(dataset_path)
    if not _success_targets_met(accepted, targets):
        raise AssertionError(_unmet_success_targets_message(accepted, targets))

    log_event(
        "verifiable_reasoning",
        "answer_teacher_completed",
        rows=total_target,
        concurrency=concurrency,
        max_attempts=max_attempts,
        max_tokens=max_tokens,
        mode="success_targets",
    )
    write_snapshot(
        "verifiable_reasoning_answer_teacher",
        {
            "stage": "completed",
            "completed": total_target,
            "total": total_target,
            "rejected": sum(_variant_counts_from_rows_path(rejections_path).values()),
        },
        force=True,
    )
    return dataset_path


def _plan_rows(generation: dict[str, Any], rng: Random) -> list[dict[str, Any]]:
    count = int(generation["count"])
    families = _resolve_families(generation)
    languages = _resolve_languages(generation)
    variants = [(family, language) for family in families for language in languages]
    counts = _variant_counts(count, variants)

    grouped_plans: list[list[dict[str, Any]]] = []
    for family, language in variants:
        family_module = _family_module(family)
        planned_recipes = common_diversity.plan_from_catalog(
            counts[(family, language)],
            list(family_module.recipe_catalog(language)),
            rng,
        )
        grouped_plans.append(
            [
                {
                    "family": family,
                    "language": language,
                    **dict(recipe),
                }
                for recipe in planned_recipes
            ]
        )

    planned_rows = common_diversity.interleave_groups(grouped_plans)
    _apply_surface_plans(planned_rows, rng)
    return planned_rows


def _generate_rows_from_plan(
    planned_rows: list[dict[str, Any]],
    seed: int | None,
    *,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(start_index, len(planned_rows)):
        rows.append(_generate_row_from_plan(index, planned_rows[index], seed))
    return rows


def _generate_row_from_plan(
    index: int,
    planned: dict[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    rng = Random(_planned_row_seed(index, planned, seed))
    family = str(planned["family"])
    language = str(planned["language"])
    recipe = {
        key: value
        for key, value in planned.items()
        if key not in {"family", "language", "candidate_index", "attempt_index"} and not str(key).startswith("surface_")
    }
    surface_plan = {
        key: value
        for key, value in planned.items()
        if str(key).startswith("surface_")
    }
    row = _family_module(family).generate_row(
        index,
        rng,
        language=language,
        recipe=recipe,
        surface_plan=surface_plan,
    )
    return _with_response_envelope(row)


def _planned_row_seed(
    index: int,
    planned: dict[str, Any],
    seed: int | None,
) -> int:
    payload = json.dumps(
        {
            "seed": seed if seed is not None else 0,
            "index": index,
            "planned": planned,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _resolve_families(generation: dict[str, Any]) -> list[str]:
    if "families" in generation:
        raw_families = generation["families"]
        assert isinstance(raw_families, list) and raw_families, "generation.families must be a non-empty list"
        if len(raw_families) == 1 and str(raw_families[0]) == "all":
            families = list(DEFAULT_ALL_FAMILIES)
        else:
            families = [str(family) for family in raw_families]
    else:
        family = str(generation.get("family", "zebra_logic"))
        if family == "all":
            families = list(DEFAULT_ALL_FAMILIES)
        else:
            families = [family]

    for family in families:
        assert family in FAMILIES, f"Unsupported verifiable_reasoning family: {family}"
    return families


def _resolve_languages(generation: dict[str, Any]) -> list[str]:
    if "languages" in generation:
        raw_languages = generation["languages"]
        assert isinstance(raw_languages, list) and raw_languages, "generation.languages must be a non-empty list"
        languages = [str(language) for language in raw_languages]
    else:
        languages = [str(generation.get("language", "en"))]

    for language in languages:
        assert language in {"en", "da"}, "verifiable_reasoning supports only English (en) and Danish (da)"
    return languages


def _answer_teacher_settings(cfg: dict[str, Any]) -> tuple[object, float, int, int, int]:
    models = cfg.get("models", {})
    assert "answer_teacher" in models, "generation.attach_targets requires models.answer_teacher"
    teacher = common_model.load_clients({"answer_teacher": _answer_teacher_model_ref(models["answer_teacher"])})["answer_teacher"]
    generation = cfg.get("generation", {})
    temperature = float(generation.get("answer_temperature", 0.0))
    max_attempts = int(generation.get("max_answer_attempts", 3))
    assert max_attempts > 0, "generation.max_answer_attempts must be positive"
    concurrency = _teacher_concurrency(teacher)
    max_tokens = int(generation.get("answer_max_tokens", ANSWER_TEACHER_MAX_TOKENS))
    assert max_tokens > 0, "generation.answer_max_tokens must be positive"
    return teacher, temperature, max_attempts, concurrency, max_tokens


def _answer_teacher_model_ref(model_ref: object) -> object:
    if isinstance(model_ref, str):
        return {
            "endpoint": model_ref,
            "timeout_seconds": ANSWER_TEACHER_TIMEOUT_SECONDS,
        }

    if isinstance(model_ref, dict):
        updated = dict(model_ref)
        updated.setdefault("timeout_seconds", ANSWER_TEACHER_TIMEOUT_SECONDS)
        return updated

    return model_ref


def _stream_rows_from_plan(
    planned_rows: list[dict[str, Any]],
    dataset_path: Path,
    seed: int | None,
    *,
    completed: int,
) -> None:
    total = len(planned_rows)
    log_event(
        "verifiable_reasoning",
        "dataset_generation_started",
        rows=total,
        completed=completed,
    )
    with dataset_path.open("a") as handle:
        for index in range(completed, total):
            row = _generate_row_from_plan(index, planned_rows[index], seed)
            store.append_jsonl_line(handle, row)
            write_snapshot(
                "verifiable_reasoning_dataset",
                {
                    "stage": "writing_dataset",
                    "completed": index + 1,
                    "total": total,
                },
                force=index + 1 == total,
            )
    log_event(
        "verifiable_reasoning",
        "dataset_generation_completed",
        rows=total,
    )


def _stream_rows_with_target_overlap(
    planned_rows: list[dict[str, Any]],
    dataset_path: Path,
    seed: int | None,
    cfg: dict[str, Any],
    *,
    completed: int,
) -> None:
    total = len(planned_rows)
    if not any(_answer_teacher_enabled(item["family"]) for item in planned_rows):
        with dataset_path.open("a") as handle:
            for index in range(completed, total):
                row = _generate_row_from_plan(index, planned_rows[index], seed)
                store.append_jsonl_line(handle, _skip_answer_teacher_row(row))
                write_snapshot(
                    "verifiable_reasoning_dataset",
                    {
                        "stage": "writing_dataset",
                        "completed": index + 1,
                        "total": total,
                    },
                    force=index + 1 == total,
                )
        return

    teacher, temperature, max_attempts, concurrency, max_tokens = _answer_teacher_settings(cfg)
    log_event(
        "verifiable_reasoning",
        "answer_teacher_started",
        rows=total,
        concurrency=concurrency,
        max_attempts=max_attempts,
        max_tokens=max_tokens,
    )
    asyncio.run(
        _stream_rows_with_target_overlap_async(
            planned_rows,
            dataset_path,
            seed,
            teacher,
            temperature=temperature,
            max_attempts=max_attempts,
            concurrency=concurrency,
            max_tokens=max_tokens,
            completed=completed,
        )
    )
    log_event(
        "verifiable_reasoning",
        "answer_teacher_completed",
        rows=total,
        mode="streaming",
        max_attempts=max_attempts,
        max_tokens=max_tokens,
    )
    write_snapshot(
        "verifiable_reasoning_answer_teacher",
        {
            "stage": "completed",
            "completed": total,
            "total": total,
        },
        force=True,
    )


async def _stream_rows_with_target_overlap_async(
    planned_rows: list[dict[str, Any]],
    dataset_path: Path,
    seed: int | None,
    teacher,
    *,
    temperature: float,
    max_attempts: int,
    concurrency: int,
    max_tokens: int,
    completed: int,
) -> None:
    def row_stream():
        for index in range(completed, len(planned_rows)):
            yield _generate_row_from_plan(index, planned_rows[index], seed)

    total = len(planned_rows)

    async def worker(_index: int, row: dict[str, Any]) -> dict[str, Any]:
        return await _attach_target_async(
            row,
            teacher,
            temperature=temperature,
            max_attempts=max_attempts,
            max_tokens=max_tokens,
        )

    with dataset_path.open("a") as handle:
        async for attached in map_async_ordered(
            row_stream(),
            worker,
            concurrency=concurrency,
            progress=_target_attachment_progress(offset=completed, total=total),
            total=total - completed,
            producer_threaded=True,
        ):
            store.append_jsonl_line(handle, attached)


async def _stream_success_target_rows_async(
    plan: dict[str, Any],
    dataset_path: Path,
    rejections_path: Path,
    seed: int | None,
    teacher,
    *,
    temperature: float,
    max_attempts: int,
    concurrency: int,
    max_tokens: int,
    accepted: Counter[tuple[str, str]],
    rejected: Counter[tuple[str, str]],
) -> None:
    targets = _success_targets_by_variant(plan)
    total_target = sum(targets.values())
    variants = list(plan["variants"])
    next_variant_index = 0

    with dataset_path.open("a") as dataset_handle, rejections_path.open("a") as rejections_handle:
        while not _success_targets_met(accepted, targets):
            batch, next_variant_index = _next_success_target_batch(
                variants,
                accepted,
                rejected,
                batch_size=concurrency,
                start_index=next_variant_index,
            )
            if not batch:
                break

            async def worker(
                _index: int,
                item: dict[str, Any],
            ) -> tuple[tuple[str, str], dict[str, Any]]:
                family = str(item["family"])
                language = str(item["language"])
                row = _generate_row_from_plan(int(item["candidate_index"]), item, seed)
                attached = await _attach_target_async(
                    row,
                    teacher,
                    temperature=temperature,
                    max_attempts=max_attempts,
                    max_tokens=max_tokens,
                )
                return (family, language), attached

            async for variant_key, attached in map_async_unordered(
                batch,
                worker,
                concurrency=min(concurrency, len(batch)),
                total=len(batch),
            ):
                if _target_row_accepted(attached) and accepted[variant_key] < targets[variant_key]:
                    store.append_jsonl_line(dataset_handle, attached)
                    accepted[variant_key] += 1
                else:
                    store.append_jsonl_line(rejections_handle, attached)
                    rejected[variant_key] += 1

                write_snapshot(
                    "verifiable_reasoning_answer_teacher",
                    {
                        "stage": "attaching_targets",
                        "completed": sum(accepted.values()),
                        "total": total_target,
                        "rejected": sum(rejected.values()),
                    },
                )


def _apply_surface_plans(planned_rows: list[dict[str, Any]], rng: Random) -> None:
    grouped_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, planned in enumerate(planned_rows):
        grouped_indices[(str(planned["family"]), str(planned["language"]))].append(index)

    for (family, language), indices in grouped_indices.items():
        family_module = _family_module(family)
        axes_getter = getattr(family_module, "surface_axes", None)
        if axes_getter is None:
            continue

        surface_axes = dict(axes_getter(language))
        if not surface_axes:
            continue

        for key, values in surface_axes.items():
            planned_axis = common_diversity.plan_from_catalog(
                len(indices),
                [{key: value} for value in values],
                rng,
            )
            rng.shuffle(planned_axis)
            for row_index, surface_plan in zip(indices, planned_axis, strict=True):
                planned_rows[row_index].update(surface_plan)


def _verify_problem_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified_rows = common_eval.verify(rows, _clues_resolve_uniquely, name="clues_resolve_uniquely")
    return verified_rows


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
    plan = _load_plan(result)
    return verify_rows(_load_rows(result), plan=plan)["rows"]


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _response_parseable(row: dict[str, Any]) -> bool:
    return _parse_row_target(row) is not None


def _response_correct(row: dict[str, Any]) -> bool:
    family_module = _family_module(row["meta"]["family"])
    parsed = _parse_row_target(row)
    if parsed is None:
        return False
    return family_module.is_correct(parsed, row["hidden"])


def _clues_resolve_uniquely(row: dict[str, Any]) -> bool:
    family_module = _family_module(row["meta"]["family"])
    return family_module.clues_resolve_uniquely(row["hidden"])


def _dataset_checks(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("mode") == "success_targets":
        return _success_target_dataset_checks(rows, plan)

    planned_rows = plan["rows"]
    checks: dict[str, Any] = {
        "family_language_coverage": common_diversity.compare_planned_to_observed(
            planned_rows,
            rows,
            ("family", "language"),
            observed_getter=_meta_getter,
        ),
        "surface_response_envelope_coverage": common_diversity.compare_planned_to_observed(
            planned_rows,
            rows,
            ("surface_response_envelope",),
            observed_getter=_meta_getter,
        ),
    }

    family_results: dict[str, Any] = {}
    for family in sorted({row["meta"]["family"] for row in rows}):
        family_rows = [row for row in rows if row["meta"]["family"] == family]
        family_plan = [item for item in planned_rows if item["family"] == family]
        family_results[family] = _family_module(family).dataset_checks(family_rows, family_plan)
    checks["families"] = family_results
    checks["passed"] = _dataset_checks_passed(checks)
    return checks


def _success_target_dataset_checks(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    targets = _success_targets_by_variant(plan)
    observed = Counter(
        (str(row["meta"]["family"]), str(row["meta"]["prompt_language"]))
        for row in rows
    )
    family_language_coverage = {
        "planned": {f"{family}/{language}": target for (family, language), target in sorted(targets.items())},
        "observed": {f"{family}/{language}": observed[(family, language)] for (family, language) in sorted(targets.items())},
        "missing": {
            f"{family}/{language}": target - observed[(family, language)]
            for (family, language), target in sorted(targets.items())
            if observed[(family, language)] < target
        },
    }
    family_language_coverage["passed"] = not family_language_coverage["missing"]

    checks: dict[str, Any] = {
        "family_language_coverage": family_language_coverage,
        "unique_prompts": common_diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }
    checks["passed"] = _dataset_checks_passed(checks)
    return checks


def _dataset_checks_passed(checks: dict[str, Any]) -> bool:
    for key, value in checks.items():
        if key == "passed":
            continue
        if isinstance(value, dict) and "passed" in value:
            if not bool(value["passed"]):
                return False
            continue
        if isinstance(value, dict):
            if not _dataset_checks_passed(value):
                return False
    return True


def _family_module(name: str):
    assert name in FAMILIES, f"Unsupported verifiable_reasoning family: {name}"
    return FAMILIES[name]


def _row_passes(row: dict[str, Any]) -> bool:
    checks = row.get("checks", {})
    return all(bool(value) for value in checks.values())


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


def _variant_counts(total: int, variants: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    counts = {variant: total // len(variants) for variant in variants}
    for variant in variants[: total % len(variants)]:
        counts[variant] += 1
    return counts


def _plan_families(plan: dict[str, Any]) -> list[str]:
    if plan.get("mode") == "success_targets":
        return sorted({str(item["family"]) for item in plan["variants"]})
    return sorted({str(item["family"]) for item in plan["rows"]})


def _plan_languages(plan: dict[str, Any]) -> list[str]:
    if plan.get("mode") == "success_targets":
        return sorted({str(item["language"]) for item in plan["variants"]})
    return sorted({str(item["language"]) for item in plan["rows"]})


def _plan_target_rows(plan: dict[str, Any]) -> int:
    if plan.get("mode") == "success_targets":
        return sum(int(item["target_rows"]) for item in plan["variants"])
    return len(plan["rows"])


def _success_targets_by_variant(plan: dict[str, Any]) -> dict[tuple[str, str], int]:
    return {
        (str(item["family"]), str(item["language"])): int(item["target_rows"])
        for item in plan["variants"]
    }


def _variant_counts_from_rows_path(path: Path) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    if not path.exists():
        return counts

    for row in store.iter_jsonl(path):
        key = (str(row["meta"]["family"]), str(row["meta"]["prompt_language"]))
        counts[key] += 1
    return counts


def _success_targets_met(
    accepted: Counter[tuple[str, str]],
    targets: dict[tuple[str, str], int],
) -> bool:
    for key, target in targets.items():
        if accepted[key] < target:
            return False
    return True


def _next_success_target_batch(
    variants: list[dict[str, Any]],
    accepted: Counter[tuple[str, str]],
    rejected: Counter[tuple[str, str]],
    *,
    batch_size: int,
    start_index: int,
) -> tuple[list[dict[str, Any]], int]:
    if not variants or batch_size <= 0:
        return [], start_index

    batch: list[dict[str, Any]] = []
    variant_count = len(variants)
    cursor = start_index
    visited = 0

    while len(batch) < batch_size and visited < variant_count:
        variant = variants[cursor]
        key = (str(variant["family"]), str(variant["language"]))
        target_rows = int(variant["target_rows"])
        attempts_used = accepted[key] + rejected[key]
        candidate_rows = list(variant["candidate_rows"])

        if accepted[key] < target_rows and attempts_used < len(candidate_rows):
            batch.append(candidate_rows[attempts_used])

        cursor = (cursor + 1) % variant_count
        visited += 1

    return batch, cursor


def _target_row_accepted(row: dict[str, Any]) -> bool:
    return row["meta"].get("target_source") in {"answer_teacher", "answer_teacher_skipped"}


def _unmet_success_targets_message(
    accepted: Counter[tuple[str, str]],
    targets: dict[tuple[str, str], int],
) -> str:
    parts: list[str] = []
    for key, target in sorted(targets.items()):
        current = accepted[key]
        if current >= target:
            continue
        family, language = key
        parts.append(f"{family}/{language}: {current}/{target}")

    detail = ", ".join(parts)
    return f"verifiable_reasoning did not reach success quotas: {detail}"


def _meta_getter(row: dict[str, Any], key: str) -> object:
    if key == "language":
        return row["meta"]["prompt_language"]
    return row["meta"][key]


def _has_expected_responses(rows: list[dict[str, Any]]) -> bool:
    return any(_expects_response_check(row) for row in rows)


def _row_target(row: dict[str, Any]) -> str:
    return str(row.get("target", ""))


def _parse_row_target(row: dict[str, Any]) -> object | None:
    family_module = _family_module(row["meta"]["family"])
    target = _row_target(row).strip()
    if not target:
        return None

    if not row["hidden"].get("response_envelope"):
        parsed = family_module.parse_target(target, row["hidden"])
        if parsed is not None:
            return parsed

    _reasoning, unwrapped, found = _split_answer_response(
        target,
        family_module=family_module,
        hidden=row["hidden"],
    )
    if not found:
        return None
    return family_module.parse_target(unwrapped, row["hidden"])


def _expects_response_check(row: dict[str, Any]) -> bool:
    if _row_target(row).strip():
        return True
    target_source = str(row.get("meta", {}).get("target_source", ""))
    return target_source in {"answer_teacher", "answer_teacher_failed"}


async def _attach_targets_async(
    rows: list[dict[str, Any]],
    teacher,
    *,
    temperature: float,
    max_attempts: int,
    concurrency: int,
    max_tokens: int,
    progress_offset: int = 0,
    progress_total: int | None = None,
    finalize_progress: bool = True,
) -> list[dict[str, Any]]:
    attached_rows: list[dict[str, Any]] = []

    async def worker(_index: int, row: dict[str, Any]) -> dict[str, Any]:
        return await _attach_target_async(
            row,
            teacher,
            temperature=temperature,
            max_attempts=max_attempts,
            max_tokens=max_tokens,
        )

    async for attached in map_async_ordered(
        rows,
        worker,
        concurrency=concurrency,
        progress=_target_attachment_progress(
            offset=progress_offset,
            total=progress_total if progress_total is not None else len(rows),
        ),
        total=len(rows),
    ):
        attached_rows.append(attached)

    if finalize_progress:
        write_snapshot(
            "verifiable_reasoning_answer_teacher",
            {
                "stage": "completed",
                "completed": progress_offset + len(attached_rows),
                "total": progress_total if progress_total is not None else len(rows),
            },
            force=True,
        )
    return attached_rows


async def _attach_target_async(
    row: dict[str, Any],
    teacher,
    *,
    temperature: float,
    max_attempts: int,
    max_tokens: int,
) -> dict[str, Any]:
    row = _with_response_envelope(row)
    if not _answer_teacher_enabled(row["meta"]["family"]):
        return _skip_answer_teacher_row(row)

    family_module = _family_module(row["meta"]["family"])
    generation_model = _teacher_generation_model_name(teacher)
    last_raw_response = ""
    last_raw_target = ""
    last_reasoning = ""
    last_failure_reason = "missing_answer_block"

    for attempt in range(1, max_attempts + 1):
        try:
            raw_response = str(
                await teacher.achat(
                    _answer_messages(row),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            ).strip()
            reasoning, raw_target, found_answer_block = _split_answer_response(
                raw_response,
                family_module=family_module,
                hidden=row["hidden"],
            )
            raw_target = raw_target.strip()
            last_raw_response = raw_response
            last_raw_target = raw_target
            last_reasoning = reasoning
            parsed = family_module.parse_target(raw_target, row["hidden"])
            reasoning_failure = _reasoning_failure_reason(row["prompt"], reasoning)

            if reasoning_failure is None and parsed is not None and family_module.is_correct(parsed, row["hidden"]):
                inner_target = family_module.canonical_target(parsed, row["hidden"])
                target = _wrap_target_with_response_envelope(inner_target, row)
                updated = dict(row)
                updated["target"] = target
                if reasoning:
                    updated["reasoning"] = reasoning

                meta = dict(row["meta"])
                meta["target_source"] = "answer_teacher"
                meta["target_attempts"] = attempt
                updated["meta"] = meta
                updated = _with_generation_model_source(updated, generation_model)

                hidden = dict(row["hidden"])
                hidden["answer_teacher_raw_target"] = raw_target
                hidden["answer_teacher_raw_response"] = raw_response
                updated["hidden"] = hidden
                return updated

            reason = _answer_failure_reason(found_answer_block, raw_target, parsed, reasoning_failure)
        except Exception as error:
            reason = _answer_exception_reason(error)

        last_failure_reason = reason
        log_event(
            "verifiable_reasoning",
            "answer_teacher_retry",
            row_id=row["id"],
            family=row["meta"]["family"],
            attempt=attempt,
            max_attempts=max_attempts,
            reason=reason,
        )

    log_event(
        "verifiable_reasoning",
        "answer_teacher_exhausted",
        row_id=row["id"],
        family=row["meta"]["family"],
        max_attempts=max_attempts,
        reason=last_failure_reason,
    )

    updated = dict(row)
    if last_reasoning:
        updated["reasoning"] = last_reasoning

    meta = dict(row["meta"])
    meta["target_source"] = "answer_teacher_failed"
    meta["target_attempts"] = max_attempts
    updated["meta"] = meta
    updated = _with_generation_model_source(updated, generation_model)

    hidden = dict(row["hidden"])
    hidden["generation_error"] = f"answer_teacher exhausted retry budget: {last_failure_reason}"
    hidden["answer_teacher_raw_target"] = last_raw_target
    hidden["answer_teacher_raw_response"] = last_raw_response
    updated["hidden"] = hidden
    return updated


def _answer_exception_reason(error: Exception) -> str:
    return f"exception:{error.__class__.__name__}"


def _teacher_generation_model_name(teacher) -> str | None:
    model = getattr(teacher, "model", None)
    if not isinstance(model, str) or not model:
        return None
    return model


def _published_generation_model(result: BuildResult, cfg: dict[str, Any]) -> str | None:
    model_ref = cfg.get("models", {}).get("answer_teacher")
    if isinstance(model_ref, dict):
        model = model_ref.get("model")
        if isinstance(model, str) and model:
            return model

    metrics_path = Path(result.run_dir) / "outputs" / "model_metrics.json"
    if not metrics_path.exists():
        return None

    metrics = read_json(metrics_path)
    models = {
        str(target["model"])
        for target in metrics.get("targets", {}).values()
        if isinstance(target, dict) and isinstance(target.get("model"), str) and target.get("model")
    }
    if len(models) != 1:
        return None
    return next(iter(models))


def _with_generation_model_source(row: dict[str, Any], generation_model: str | None) -> dict[str, Any]:
    if generation_model is None:
        return row

    sources = list(row.get("sources", []))
    for source in sources:
        if source.get("kind") == "generation_model":
            return row

    updated = dict(row)
    updated["sources"] = [
        *sources,
        {
            "kind": "generation_model",
            "value": generation_model,
        },
    ]
    return updated


def _answer_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    row = _with_response_envelope(row)
    language = row["meta"].get("target_language", row["meta"]["prompt_language"])
    prompt = str(row["prompt"]).strip()
    system_text = _answer_prompt_text(language, str(row["meta"]["family"]))

    return [
        {
            "role": "system",
            "content": system_text,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def _answer_teacher_enabled(family: str) -> bool:
    return family not in ANSWER_TEACHER_DISABLED_FAMILIES


def _skip_answer_teacher_row(row: dict[str, Any]) -> dict[str, Any]:
    row = _with_response_envelope(row)
    updated = dict(row)
    meta = dict(row["meta"])
    meta["target_source"] = "answer_teacher_skipped"
    meta["target_attempts"] = 0
    updated["meta"] = meta

    hidden = dict(row["hidden"])
    hidden["generation_error"] = "answer_teacher disabled for family"
    updated["hidden"] = hidden
    return updated


def _with_response_envelope(row: dict[str, Any]) -> dict[str, Any]:
    if row["hidden"].get("response_envelope"):
        return row

    family_module = _family_module(row["meta"]["family"])
    language = str(row["meta"].get("target_language", row["meta"]["prompt_language"]))
    envelope = _response_envelope_variant(row)
    envelope_label = _response_envelope_label(envelope, language)
    answer_contract = family_module.answer_contract(row["hidden"], language)
    normalized_contract = _normalize_answer_contract(answer_contract, language)
    suffix = _response_envelope_prompt_text(envelope, normalized_contract, language)

    updated = dict(row)
    updated["prompt"] = f"{str(row['prompt']).rstrip()}\n\n{suffix}"

    hidden = dict(row["hidden"])
    hidden["response_envelope"] = envelope
    hidden["response_envelope_label"] = envelope_label
    updated["hidden"] = hidden

    meta = dict(row["meta"])
    meta["response_envelope"] = envelope
    meta["response_envelope_label"] = envelope_label
    updated["meta"] = meta
    return updated


def _response_envelope_variant(row: dict[str, Any]) -> str:
    return str(row["meta"].get("surface_response_envelope", "answer_block"))


def _response_envelope_label(envelope: str, language: str) -> str | None:
    if envelope == "answer_block":
        return "Svar" if language == "da" else "Answer"
    if envelope == "solution_block":
        return "Løsning" if language == "da" else "Solution"
    if envelope == "final_block":
        return "Endeligt svar" if language == "da" else "Final answer"
    return None


def _normalize_answer_contract(text: str, language: str) -> str:
    if language == "da":
        return text.replace("I din svarblok", "I dit endelige svar")

    return text.replace("In your answer block", "In your final response")


def _response_envelope_prompt_text(envelope: str, answer_contract: str, language: str) -> str:
    lines = [_response_envelope_heading(language)]
    lines.extend(_response_envelope_outer_lines(envelope, language))
    lines.append("")
    lines.append(answer_contract)
    return "\n".join(lines).strip()


def _response_envelope_heading(language: str) -> str:
    if language == "da":
        return "Endeligt svarformat:"
    return "Final response format:"


def _response_envelope_outer_lines(envelope: str, language: str) -> tuple[str, ...]:
    if envelope == "answer_block":
        if language == "da":
            return (
                "Afslut efter din begrundelse med præcis denne ydre struktur:",
                "Svar:",
                "<kun det endelige svar>",
            )
        return (
            "After your reasoning, finish with exactly this outer wrapper:",
            "Answer:",
            "<final answer only>",
        )

    if envelope == "solution_block":
        if language == "da":
            return (
                "Afslut efter din begrundelse med præcis denne ydre struktur:",
                "Løsning:",
                "<kun det endelige svar>",
            )
        return (
            "After your reasoning, finish with exactly this outer wrapper:",
            "Solution:",
            "<final answer only>",
        )

    if envelope == "final_block":
        if language == "da":
            return (
                "Afslut efter din begrundelse med præcis denne ydre struktur:",
                "Endeligt svar:",
                "<kun det endelige svar>",
            )
        return (
            "After your reasoning, finish with exactly this outer wrapper:",
            "Final answer:",
            "<final answer only>",
        )

    if envelope == "json":
        if language == "da":
            return (
                "Afslut efter din begrundelse med præcis dette JSON-objekt og intet andet:",
                '{"answer_lines": ["<linje 1>", "<linje 2>"]}',
                "Brug én streng i `answer_lines` per linje i det endelige svar.",
            )
        return (
            "After your reasoning, finish with exactly this JSON object and nothing else:",
            '{"answer_lines": ["<line 1>", "<line 2>"]}',
            "Use one string in `answer_lines` for each line of the final answer.",
        )

    if envelope == "xml":
        if language == "da":
            return (
                "Afslut efter din begrundelse med præcis denne XML-struktur og intet andet:",
                "<response>",
                "  <line><linje 1></line>",
                "  <line><linje 2></line>",
                "</response>",
                "Brug ét `<line>`-element per linje i det endelige svar.",
            )
        return (
            "After your reasoning, finish with exactly this XML structure and nothing else:",
            "<response>",
            "  <line><line 1></line>",
            "  <line><line 2></line>",
            "</response>",
            "Use one `<line>` element for each line of the final answer.",
        )

    assert envelope == "yaml", f"Unsupported response envelope: {envelope}"
    if language == "da":
        return (
            "Afslut efter din begrundelse med præcis denne YAML-struktur og intet andet:",
            "answer_lines:",
            '  - "<linje 1>"',
            '  - "<linje 2>"',
            "Brug ét punkt under `answer_lines` per linje i det endelige svar.",
        )
    return (
        "After your reasoning, finish with exactly this YAML structure and nothing else:",
        "answer_lines:",
        '  - "<line 1>"',
        '  - "<line 2>"',
        "Use one list item under `answer_lines` for each line of the final answer.",
    )


def _wrap_target_with_response_envelope(target: str, row: dict[str, Any]) -> str:
    envelope = str(row["hidden"].get("response_envelope", _response_envelope_variant(row)))
    lines = [line.rstrip() for line in target.splitlines()]
    label = row["hidden"].get("response_envelope_label")

    if envelope == "answer_block":
        assert label is not None
        return f"{label}:\n{target}".strip()

    if envelope == "solution_block":
        assert label is not None
        return f"{label}:\n{target}".strip()

    if envelope == "final_block":
        assert label is not None
        return f"{label}:\n{target}".strip()

    if envelope == "json":
        return json.dumps({"answer_lines": lines}, ensure_ascii=False, indent=2)

    if envelope == "xml":
        joined = "\n".join(f"  <line>{line}</line>" for line in lines)
        return f"<response>\n{joined}\n</response>"

    assert envelope == "yaml", f"Unsupported response envelope: {envelope}"
    joined = "\n".join(f"  - {json.dumps(line, ensure_ascii=False)}" for line in lines)
    return f"answer_lines:\n{joined}"


def _split_answer_response(
    text: str,
    *,
    family_module=None,
    hidden: dict[str, Any] | None = None,
) -> tuple[str, str, bool]:
    stripped = text.strip()
    if hidden is not None and hidden.get("response_envelope"):
        extracted = _extract_expected_envelope(stripped, hidden)
        if extracted is None:
            return stripped, _strip_wrapping_code_fence(stripped), False

        reasoning, cleaned_answer = extracted
        if family_module is not None:
            parsed = family_module.parse_target(cleaned_answer, hidden)
            if parsed is not None:
                return reasoning, cleaned_answer, True
        return reasoning, cleaned_answer, True

    extracted = _extract_enveloped_answer(stripped, None)
    if extracted is None:
        fallback = _extract_trailing_answer(stripped, family_module=family_module, hidden=hidden)
        if fallback is not None:
            return fallback[0], fallback[1], True
        return stripped, _strip_wrapping_code_fence(stripped), False

    reasoning, cleaned_answer = extracted
    if family_module is not None and hidden is not None:
        parsed = family_module.parse_target(cleaned_answer, hidden)
        if parsed is not None:
            return reasoning, cleaned_answer, True
        if not cleaned_answer or _looks_like_placeholder_answer(cleaned_answer):
            fallback = _extract_trailing_answer(reasoning, family_module=family_module, hidden=hidden)
            if fallback is not None:
                return fallback[0], fallback[1], True
    return reasoning, cleaned_answer, True


def _extract_expected_envelope(text: str, hidden: dict[str, Any]) -> tuple[str, str] | None:
    envelope = str(hidden["response_envelope"])
    label = hidden.get("response_envelope_label")
    return _extract_known_envelope(text, envelope, label=label)


def _extract_enveloped_answer(text: str, envelope: object | None) -> tuple[str, str] | None:
    if envelope is not None:
        return _extract_known_envelope(text, str(envelope))

    for candidate in ("answer_block", "solution_block", "final_block", "json", "xml", "yaml"):
        extracted = _extract_known_envelope(text, candidate)
        if extracted is not None:
            return extracted
    return None


def _extract_known_envelope(text: str, envelope: str, *, label: object | None = None) -> tuple[str, str] | None:
    if envelope == "answer_block":
        if isinstance(label, str):
            return _extract_labeled_block(text, (label,))
        return _extract_labeled_block(text, ("answer", "svar"))
    if envelope == "solution_block":
        if isinstance(label, str):
            return _extract_labeled_block(text, (label,))
        return _extract_labeled_block(text, ("solution", "løsning"))
    if envelope == "final_block":
        if isinstance(label, str):
            return _extract_labeled_block(text, (label,))
        return _extract_labeled_block(text, ("final answer", "endeligt svar"))
    if envelope == "json":
        return _extract_structured_answer(text, kind="json")
    if envelope == "xml":
        return _extract_structured_answer(text, kind="xml")
    if envelope == "yaml":
        return _extract_structured_answer(text, kind="yaml")
    return None


def _extract_labeled_block(text: str, labels: tuple[str, ...]) -> tuple[str, str] | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    matches = list(re.finditer(rf"(?im)^\s*(?:{label_pattern})\s*:\s*", text))
    if not matches:
        return None

    marker = matches[-1]
    reasoning = text[: marker.start()].strip()
    answer = text[marker.end() :].strip()
    return reasoning, _strip_wrapping_code_fence(answer)


def _extract_structured_answer(text: str, *, kind: str) -> tuple[str, str] | None:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None

    for start in range(len(lines) - 1, -1, -1):
        candidate = "\n".join(lines[start:]).strip()
        answer = _parse_structured_answer(candidate, kind=kind)
        if answer is None:
            continue
        reasoning = "\n".join(lines[:start]).strip()
        return reasoning, answer

    return None


def _parse_structured_answer(text: str, *, kind: str) -> str | None:
    stripped = _strip_wrapping_code_fence(text)

    if kind == "json":
        return _parse_json_answer(stripped)
    if kind == "xml":
        return _parse_xml_answer(stripped)

    assert kind == "yaml", f"Unsupported structured answer type: {kind}"
    return _parse_yaml_answer(stripped)


def _parse_json_answer(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return _answer_from_structured_payload(payload)


def _parse_xml_answer(text: str) -> str | None:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None

    if root.tag not in {"response", "answer", "solution"}:
        return None

    line_children = [child for child in root if child.tag == "line"]
    if line_children:
        lines = [(child.text or "").strip() for child in line_children]
        if not any(lines):
            return None
        return "\n".join(lines).strip()

    inner = "".join(root.itertext()).strip()
    if not inner:
        return None
    return inner


def _parse_yaml_answer(text: str) -> str | None:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0].strip() != "answer_lines:":
        return None

    answer_lines: list[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped.startswith("- "):
            return None
        value = stripped[2:].strip()
        answer_lines.append(_parse_yaml_scalar(value))

    if not answer_lines:
        return None
    return "\n".join(answer_lines).strip()


def _parse_yaml_scalar(value: str) -> str:
    if value[:1] not in {'"', "'"}:
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value.strip("\"'")


def _answer_from_structured_payload(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None

    for key in ("answer_lines", "solution_lines", "answer", "solution", "svar", "løsning"):
        if key not in payload:
            continue

        value = payload[key]
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None

        if isinstance(value, list):
            lines = [str(item).strip() for item in value]
            if not any(lines):
                return None
            return "\n".join(lines).strip()

    return None


def _extract_trailing_answer(
    text: str,
    *,
    family_module,
    hidden: dict[str, Any] | None,
) -> tuple[str, str] | None:
    if family_module is None or hidden is None:
        return None

    stripped = _strip_wrapping_code_fence(text)
    lines = [line.rstrip() for line in stripped.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return None

    for start in range(len(lines) - 1, -1, -1):
        candidate = "\n".join(lines[start:]).strip()
        if not candidate:
            continue
        parsed = family_module.parse_target(candidate, hidden)
        if parsed is None:
            continue
        reasoning = "\n".join(lines[:start]).strip()
        return reasoning, candidate

    return None


def _looks_like_placeholder_answer(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return True
    if "<" in lowered and ">" in lowered:
        return True
    if "det endelige svar" in lowered:
        return True
    if "final answer" in lowered:
        return True
    return False


def _answer_prompt_text(language: str, family: str) -> str:
    if family == "lightuppuzzle_logic":
        return _lightup_answer_prompt_text(language)
    if family == "knightsandknaves_logic":
        return _knightsandknaves_answer_prompt_text(language)

    if language == "da":
        return (
            "Løs opgaven omhyggeligt og tænk trin for trin. "
            "Skriv din begrundelse på dansk. "
            "Resonér i flere trin, før du giver den endelige løsning. "
            "Gentag ikke opsætningen, kategorierne eller ledetrådene. "
            "Kopiér ikke linjer fra opgaven ordret ind i begrundelsen. "
            "Citér ikke opgaven ordret. "
            "Følg det endelige svarformat i opgaven nøjagtigt. "
            "Skriv ikke noget efter det endelige svar."
        )

    return (
        "Solve the puzzle carefully and think step by step. "
        "Reason in steps before giving the final answer. "
        "Do not restate the setup, category list, or clue list. "
        "Do not copy lines from the puzzle prompt into your reasoning. "
        "Do not quote the puzzle verbatim. "
        "Follow the final response format in the puzzle prompt exactly. "
        "Do not add anything after the final response."
    )


def _knightsandknaves_answer_prompt_text(language: str) -> str:
    if language == "da":
        return (
            "Løs opgaven omhyggeligt og skriv din begrundelse på dansk. "
            "Hold fokus på hvilke udsagn der tvinger rollefordelingen, og undgå at gentage hele opgaven. "
            "Kopiér ikke linjer fra opgaven ordret ind i begrundelsen. "
            "Når du er færdig med begrundelsen, skal du straks give det endelige svar i formatet fra opgaven. "
            "Efter den ydre svarstruktur må der kun stå rå linjer i formatet `Navn: rolle`, én linje per person, i den viste rækkefølge. "
            "Brug ikke punkttegn, nummerering, prose eller pladsholdere som `<det endelige svar>`. "
            "Skriv ikke noget efter det endelige svar."
        )

    return (
        "Solve the puzzle carefully and reason step by step without repeating the full prompt. "
        "Do not copy lines from the puzzle prompt into your reasoning. "
        "Once the role assignment is fixed, immediately give the final response in the format from the prompt. "
        "Inside that outer structure there must only be raw lines in the format `Name: role`, one line per speaker, in the shown order. "
        "Do not use bullets, numbering, prose, or placeholders such as `<final answer>`. "
        "Do not add anything after the final response."
    )


def _lightup_answer_prompt_text(language: str) -> str:
    if language == "da":
        return (
            "Løs opgaven omhyggeligt og skriv din begrundelse på dansk. "
            "Hold begrundelsen kort og fokuser kun på de vigtigste trin, der tvinger lampens placeringer. "
            "Undgå lange casesplit og undgå at tjekke hele gitteret flere gange. "
            "Gentag ikke opsætningen eller regellisten. "
            "Kopiér ikke linjer fra opgaven ordret ind i begrundelsen. "
            "Når gitteret er fastlagt, skal du straks kopiere hele gitteret ind i svarformatet fra opgaven. "
            "Skriv ikke rækkeetiketter som `Række 0:` og brug ikke pladsholdere som `<det endelige svar>`. "
            "Inde i den ydre svarstruktur må der kun stå de rå gitterlinjer, én per linje. "
            "Skriv ikke noget efter det endelige svar."
        )

    return (
        "Solve the puzzle carefully and keep the reasoning concise. "
        "Only include the key deductions that force lamp placements. "
        "Avoid long case splits and do not re-check the whole grid repeatedly. "
        "Do not restate the setup or rule list. "
        "Do not copy lines from the puzzle prompt into your reasoning. "
        "Once the grid is fixed, immediately copy the full grid into the final response format from the prompt. "
        "Do not write row labels such as `Row 0:` and do not use placeholders such as `<final answer>`. "
        "Inside that outer structure there should only be the raw grid lines, one per line. "
        "Do not add anything after the final response."
    )


def _strip_wrapping_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 2:
        return stripped
    if not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _answer_failure_reason(
    found_answer_block: bool,
    raw_target: str,
    parsed: object,
    reasoning_failure: str | None,
) -> str:
    if reasoning_failure is not None:
        return reasoning_failure
    if not found_answer_block:
        return "missing_answer_block"
    if not raw_target:
        return "empty_answer"
    if parsed is None:
        return "unparseable_answer"
    return "incorrect_answer"


def _response_reasoning_quality(row: dict[str, Any]) -> bool:
    if not _expects_response_check(row):
        return True
    return _reasoning_failure_reason(str(row.get("prompt", "")), str(row.get("reasoning", ""))) is None


def _reasoning_failure_reason(prompt: str, reasoning: str) -> str | None:
    text = reasoning.strip()
    if not text:
        return "missing_reasoning"

    if _reasoning_step_count(text) < 3:
        return "reasoning_too_shallow"

    if _repeats_prompt_excessively(prompt, text):
        return "reasoning_copies_prompt"
    return None


def _repeats_prompt_excessively(prompt: str, reasoning: str) -> bool:
    prompt_lines = {_normalize_line(line) for line in prompt.splitlines() if _normalize_line(line)}
    reasoning_lines = [_normalize_line(line) for line in reasoning.splitlines() if _normalize_line(line)]
    if len(reasoning_lines) < 8:
        return False

    repeated = sum(1 for line in reasoning_lines if line in prompt_lines)
    if repeated < 6:
        return False

    return repeated * 2 >= len(reasoning_lines)


def _normalize_line(text: str) -> str:
    lowered = text.lower()
    compact = re.sub(r"\s+", " ", lowered).strip()
    return re.sub(r"^[0-9]+[.)]\s*", "", compact)


def _reasoning_step_count(text: str) -> int:
    parts = re.split(r"(?:\n+|(?<=[.!?])\s+)", text.strip())
    steps = [part.strip() for part in parts if len(part.strip()) >= 12]
    return len(steps)


def _teacher_concurrency(teacher) -> int:
    return max(1, int(getattr(getattr(teacher, "runtime", None), "max_concurrency", 1)))


def _target_attachment_progress(*, offset: int, total: int):
    def progress(completed: int, _total: int | None, elapsed: int) -> None:
        write_snapshot(
            "verifiable_reasoning_answer_teacher",
            {
                "stage": "attaching_targets",
                "completed": offset + completed,
                "total": total,
                "elapsed_seconds": elapsed,
            },
            force=offset + completed == total,
        )

    return progress
