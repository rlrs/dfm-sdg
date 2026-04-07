from collections import Counter
from pathlib import Path
from random import Random
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import diversity as common_diversity
from sdg.commons import eval as common_eval
from sdg.commons import publish as common_publish
from sdg.commons.run import load, run
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json
from sdg.packs.verifiable_reasoning import lineup, zebra

FAMILIES = {
    "lineup_logic": lineup,
    "zebra_logic": zebra,
}


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
    verified_rows = _verify_rows(rows)
    failures = [row for row in verified_rows if not _row_passes(row)]
    dataset_checks = _dataset_checks(verified_rows, plan)

    outputs_dir = Path(result.run_dir) / "outputs"
    store.write_jsonl(verified_rows, outputs_dir / "verified.jsonl")
    store.write_jsonl(failures, outputs_dir / "failures.jsonl")
    write_json(dataset_checks, outputs_dir / "dataset_checks.json")

    metrics = common_eval.aggregate_metrics(verified_rows)
    failure_summary = common_eval.summarize_failures(verified_rows)
    write_json(metrics, outputs_dir / "metrics.json")
    write_json(failure_summary, outputs_dir / "failure_summary.json")
    common_publish.write_preview(verified_rows, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "run_id": result.run_id,
        "verified_rows": len(verified_rows),
        "failed_rows": len(failures),
        "metrics": metrics,
        "failure_summary": failure_summary,
        "dataset_checks": dataset_checks,
    }


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

    rows: list[dict[str, Any]] = []
    for index, planned in enumerate(planned_rows):
        family = str(planned["family"])
        language = str(planned["language"])
        recipe = {key: value for key, value in planned.items() if key not in {"family", "language"}}
        rows.append(_family_module(family).generate_row(index, rng, language=language, recipe=recipe))

    plan = {
        "rows": planned_rows,
        "family_counts": dict(Counter(item["family"] for item in planned_rows)),
        "language_counts": dict(Counter(item["language"] for item in planned_rows)),
    }
    return rows, plan


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


def _verify_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    verified_rows = common_eval.verify(rows, _answer_parseable, name="answer_parseable")
    verified_rows = common_eval.verify(verified_rows, _answer_correct, name="answer_correct")
    verified_rows = common_eval.verify(verified_rows, _clues_resolve_uniquely, name="clues_resolve_uniquely")
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
    return _verify_rows(_load_rows(result))


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _answer_parseable(row: dict[str, Any]) -> bool:
    family_module = _family_module(row["meta"]["family"])
    parsed = family_module.parse_target(row["target"], row["hidden"])
    return parsed is not None


def _answer_correct(row: dict[str, Any]) -> bool:
    family_module = _family_module(row["meta"]["family"])
    parsed = family_module.parse_target(row["target"], row["hidden"])
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
