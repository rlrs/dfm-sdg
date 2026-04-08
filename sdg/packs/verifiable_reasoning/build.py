import asyncio
import re
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
from sdg.commons.work_queue import map_async_ordered
from sdg.packs.verifiable_reasoning import (
    futoshiki,
    hitori,
    lineup,
    numbrix,
    skyscraper,
    zebra,
)

FAMILIES = {
    "futoshiki_logic": futoshiki,
    "hitori_logic": hitori,
    "lineup_logic": lineup,
    "numbrix_logic": numbrix,
    "skyscraper_logic": skyscraper,
    "zebra_logic": zebra,
}
ANSWER_TEACHER_TIMEOUT_SECONDS = 3000.0


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

    teacher, temperature, max_attempts, concurrency = _answer_teacher_settings(cfg)
    log_event(
        "verifiable_reasoning",
        "answer_teacher_started",
        rows=len(rows),
        concurrency=concurrency,
        max_attempts=max_attempts,
    )
    attached_rows = asyncio.run(
        _attach_targets_async(
            rows,
            teacher,
            temperature=temperature,
            max_attempts=max_attempts,
            concurrency=concurrency,
        )
    )
    log_event(
        "verifiable_reasoning",
        "answer_teacher_completed",
        rows=len(attached_rows),
        concurrency=concurrency,
        max_attempts=max_attempts,
    )
    return attached_rows


def summarize(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    dataset_checks_path = outputs_dir / "dataset_checks.json"
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
        "metrics": metrics,
        "dataset_checks": dataset_checks,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    export_rows = [_strip_hidden(row) for row in rows]
    failures = [row for row in export_rows if not _row_passes(row)]
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
    rows, plan = _make_rows(cfg, seed)
    dataset_path = store.write_jsonl(rows, outputs_dir / "dataset.jsonl")
    plan_path = write_json(plan, outputs_dir / "plan.json")
    common_publish.write_preview(rows, outputs_dir / "sample_preview.jsonl", n=20)

    families = sorted({row["meta"]["family"] for row in rows})
    languages = sorted({row["meta"]["prompt_language"] for row in rows})
    return {
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={
                "rows": len(rows),
                "families": families,
                "languages": languages,
            },
        ),
        "plan": Artifact(
            name="plan",
            path=str(plan_path),
            kind="json",
            meta={
                "rows": len(plan["rows"]),
                "families": families,
                "languages": languages,
            },
        ),
    }


def _make_rows(cfg: dict[str, Any], seed: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generation = cfg["generation"]
    rng = Random(seed if seed is not None else 0)
    planned_rows = _plan_rows(generation, rng)
    if generation.get("attach_targets", False):
        rows = _generate_rows_with_target_overlap(planned_rows, rng, cfg)
    else:
        rows = _generate_rows_from_plan(planned_rows, rng)

    plan = {
        "rows": planned_rows,
        "family_counts": dict(Counter(item["family"] for item in planned_rows)),
        "language_counts": dict(Counter(item["language"] for item in planned_rows)),
    }
    return rows, plan


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
    rng: Random,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, planned in enumerate(planned_rows):
        rows.append(_generate_row_from_plan(index, planned, rng))
    return rows


def _generate_row_from_plan(
    index: int,
    planned: dict[str, Any],
    rng: Random,
) -> dict[str, Any]:
    family = str(planned["family"])
    language = str(planned["language"])
    recipe = {
        key: value
        for key, value in planned.items()
        if key not in {"family", "language"} and not str(key).startswith("surface_")
    }
    surface_plan = {
        key: value
        for key, value in planned.items()
        if str(key).startswith("surface_")
    }
    return _family_module(family).generate_row(
        index,
        rng,
        language=language,
        recipe=recipe,
        surface_plan=surface_plan,
    )


def _resolve_families(generation: dict[str, Any]) -> list[str]:
    if "families" in generation:
        raw_families = generation["families"]
        assert isinstance(raw_families, list) and raw_families, "generation.families must be a non-empty list"
        families = [str(family) for family in raw_families]
    else:
        families = [str(generation.get("family", "zebra_logic"))]

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


def _answer_teacher_settings(cfg: dict[str, Any]) -> tuple[object, float, int, int]:
    models = cfg.get("models", {})
    assert "answer_teacher" in models, "generation.attach_targets requires models.answer_teacher"
    teacher = common_model.load_clients({"answer_teacher": _answer_teacher_model_ref(models["answer_teacher"])})["answer_teacher"]
    generation = cfg.get("generation", {})
    temperature = float(generation.get("answer_temperature", 0.0))
    max_attempts = int(generation.get("max_answer_attempts", 3))
    assert max_attempts > 0, "generation.max_answer_attempts must be positive"
    concurrency = _teacher_concurrency(teacher)
    return teacher, temperature, max_attempts, concurrency


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


def _generate_rows_with_target_overlap(
    planned_rows: list[dict[str, Any]],
    rng: Random,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    teacher, temperature, max_attempts, concurrency = _answer_teacher_settings(cfg)
    total = len(planned_rows)
    log_event(
        "verifiable_reasoning",
        "answer_teacher_started",
        rows=total,
        concurrency=concurrency,
        max_attempts=max_attempts,
    )
    rows = asyncio.run(
        _generate_rows_with_target_overlap_async(
            planned_rows,
            rng,
            teacher,
            temperature=temperature,
            max_attempts=max_attempts,
            concurrency=concurrency,
        )
    )
    log_event(
        "verifiable_reasoning",
        "answer_teacher_completed",
        rows=len(rows),
        mode="streaming",
        max_attempts=max_attempts,
    )
    write_snapshot(
        "verifiable_reasoning_answer_teacher",
        {
            "stage": "completed",
            "completed": len(rows),
            "total": total,
        },
        force=True,
    )
    return rows


async def _generate_rows_with_target_overlap_async(
    planned_rows: list[dict[str, Any]],
    rng: Random,
    teacher,
    *,
    temperature: float,
    max_attempts: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    def row_stream():
        for index, planned in enumerate(planned_rows):
            yield _generate_row_from_plan(index, planned, rng)

    rows: list[dict[str, Any]] = []
    total = len(planned_rows)

    async def worker(_index: int, row: dict[str, Any]) -> dict[str, Any]:
        return await _attach_target_async(
            row,
            teacher,
            temperature=temperature,
            max_attempts=max_attempts,
        )

    async for attached in map_async_ordered(
        row_stream(),
        worker,
        concurrency=concurrency,
        progress=_target_attachment_progress(offset=0, total=total),
        total=total,
        producer_threaded=True,
    ):
        rows.append(attached)

    return rows


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
    return _verify_problem_rows(_load_rows(result))


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _response_parseable(row: dict[str, Any]) -> bool:
    family_module = _family_module(row["meta"]["family"])
    parsed = family_module.parse_target(_row_target(row), row["hidden"])
    return parsed is not None


def _response_correct(row: dict[str, Any]) -> bool:
    family_module = _family_module(row["meta"]["family"])
    parsed = family_module.parse_target(_row_target(row), row["hidden"])
    if parsed is None:
        return False
    return family_module.is_correct(parsed, row["hidden"])


def _clues_resolve_uniquely(row: dict[str, Any]) -> bool:
    family_module = _family_module(row["meta"]["family"])
    return family_module.clues_resolve_uniquely(row["hidden"])


def _dataset_checks(rows: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    planned_rows = plan["rows"]
    checks: dict[str, Any] = {
        "family_language_coverage": common_diversity.compare_planned_to_observed(
            planned_rows,
            rows,
            ("family", "language"),
            observed_getter=_meta_getter,
        )
    }

    family_results: dict[str, Any] = {}
    for family in sorted({row["meta"]["family"] for row in rows}):
        family_rows = [row for row in rows if row["meta"]["family"] == family]
        family_plan = [item for item in planned_rows if item["family"] == family]
        family_results[family] = _family_module(family).dataset_checks(family_rows, family_plan)
    checks["families"] = family_results
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


def _meta_getter(row: dict[str, Any], key: str) -> object:
    if key == "language":
        return row["meta"]["prompt_language"]
    return row["meta"][key]


def _has_expected_responses(rows: list[dict[str, Any]]) -> bool:
    return any(_expects_response_check(row) for row in rows)


def _row_target(row: dict[str, Any]) -> str:
    return str(row.get("target", ""))


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
) -> dict[str, Any]:
    family_module = _family_module(row["meta"]["family"])
    last_raw_response = ""
    last_raw_target = ""
    last_reasoning = ""
    last_failure_reason = "missing_answer_block"

    for attempt in range(1, max_attempts + 1):
        raw_response = str(await teacher.achat(_answer_messages(row), temperature=temperature)).strip()
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
            target = family_module.canonical_target(parsed, row["hidden"])
            updated = dict(row)
            updated["target"] = target
            if reasoning:
                updated["reasoning"] = reasoning

            meta = dict(row["meta"])
            meta["target_source"] = "answer_teacher"
            meta["target_attempts"] = attempt
            updated["meta"] = meta

            hidden = dict(row["hidden"])
            hidden["answer_teacher_raw_target"] = raw_target
            hidden["answer_teacher_raw_response"] = raw_response
            updated["hidden"] = hidden
            return updated

        reason = _answer_failure_reason(found_answer_block, raw_target, parsed, reasoning_failure)
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

    hidden = dict(row["hidden"])
    hidden["generation_error"] = f"answer_teacher exhausted retry budget: {last_failure_reason}"
    hidden["answer_teacher_raw_target"] = last_raw_target
    hidden["answer_teacher_raw_response"] = last_raw_response
    updated["hidden"] = hidden
    return updated


def _answer_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    family_module = _family_module(row["meta"]["family"])
    language = row["meta"].get("target_language", row["meta"]["prompt_language"])
    answer_contract = family_module.answer_contract(row["hidden"], language)
    prompt = str(row["prompt"]).strip()
    system_text, user_suffix = _answer_prompt_text(language)

    return [
        {
            "role": "system",
            "content": system_text,
        },
        {
            "role": "user",
            "content": (
                f"{prompt}\n\n"
                f"{user_suffix}\n{answer_contract}\n\n"
            ),
        },
    ]


def _split_answer_response(
    text: str,
    *,
    family_module=None,
    hidden: dict[str, Any] | None = None,
) -> tuple[str, str, bool]:
    stripped = text.strip()
    matches = list(re.finditer(r"(?im)^\s*(?:(?:final|endeligt)\s+)?(?:answer|svar)\s*:\s*", stripped))
    if not matches:
        fallback = _extract_trailing_answer(stripped, family_module=family_module, hidden=hidden)
        if fallback is not None:
            return fallback[0], fallback[1], True
        return stripped, _strip_wrapping_code_fence(stripped), False

    marker = matches[-1]
    reasoning = stripped[: marker.start()].strip()
    answer = stripped[marker.end() :].strip()
    return reasoning, _strip_wrapping_code_fence(answer), True


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


def _answer_prompt_text(language: str) -> tuple[str, str]:
    if language == "da":
        return (
            "Løs opgaven omhyggeligt og tænk trin for trin. "
            "Skriv din begrundelse på dansk. "
            "Resonér i flere trin, før du giver den endelige løsning. "
            "Gentag ikke opsætningen, kategorierne eller ledetrådene. "
            "Citér ikke opgaven ordret. "
            "Afslut efter din begrundelse med en afsluttende svarblok i præcis dette format:\n"
            "Svar:\n"
            "<kun det endelige svar>\n"
            "Skriv ikke noget efter svarblokken.",
            "Svarformat:",
        )

    return (
        "Solve the puzzle carefully and think step by step. "
        "Reason in steps before giving the final answer. "
        "Do not restate the setup, category list, or clue list. "
        "Do not quote the puzzle verbatim. "
        "After your reasoning, finish with a final answer block in exactly this form:\n"
        "Answer:\n"
        "<final answer only>\n"
        "Do not add anything after the final answer block.",
        "Answer contract:",
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

    prompt_lines = {_normalize_line(line) for line in prompt.splitlines() if _normalize_line(line)}
    reasoning_lines = [_normalize_line(line) for line in text.splitlines() if _normalize_line(line)]
    repeated = sum(1 for line in reasoning_lines if line in prompt_lines)
    if repeated >= 2:
        return "reasoning_copies_prompt"
    return None


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
