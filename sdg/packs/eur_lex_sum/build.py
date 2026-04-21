from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import concurrency as common_concurrency
from sdg.commons import eval as common_eval
from sdg.commons import progress as common_progress
from sdg.commons import publish as common_publish
from sdg.commons import sources as common_sources
from sdg.commons.model import LLM, load_clients
from sdg.commons.run import load, run
from sdg.commons.run_log import log_event
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json
from sdg.commons.work_queue import map_async_ordered

_STYLE_SEEDS = [
    "Stil: direkte kommando, anmodning først. Eks: 'Opsummer følgende tekst:\\n\\n{{DOKUMENT}}'",
    "Stil: høflig anmodning, anmodning først. Eks: 'Vil du hjælpe mig med at opsummere dette? {{DOKUMENT}}'",
    "Stil: kontekstuel indledning, anmodning først. Eks: 'Jeg har et EU-dokument jeg gerne vil have et overblik over.\\n\\n{{DOKUMENT}}\\n\\nKan du give mig et resumé?'",
    "Stil: konkret resumé-anmodning med længdebegrænsning, anmodning først. Eks: 'Kan du give mig et kort resumé af nedenstående på tre til fem sætninger?\\n\\n{{DOKUMENT}}'",
    "Stil: faglig/formel, anmodning først. Eks: 'Udarbejd venligst et resumé af følgende juridiske dokument:\\n\\n{{DOKUMENT}}'",
    "Stil: dokumentet først, kort afsluttende spørgsmål. Eks: '{{DOKUMENT}}\\n\\nHvad handler dette om?'",
    "Stil: dokumentet først, opsummering bedt om til sidst. Eks: '{{DOKUMENT}}\\n\\nKan du opsummere det ovenstående for mig?'",
    "Stil: dokumentet først, specifik resumé-anmodning til sidst. Eks: '{{DOKUMENT}}\\n\\nKan du skrive et kort resumé af ovenstående?'",
    "Stil: dokumentet først, uformel/nysgerrig tone. Eks: '{{DOKUMENT}}\\n\\nHvad er essensen her?'",
    "Stil: dokument midt i, omgivet af kontekst. Eks: 'Jeg sidder med dette EU-dokument:\\n\\n{{DOKUMENT}}\\n\\nKan du give mig et hurtigt overblik?'",
    "Stil: dokument midt i, med specifik opgaveformulering bagefter. Eks: 'Her er teksten:\\n\\n{{DOKUMENT}}\\n\\nSkriv et resumé på maksimalt 5 sætninger.'",
    "Stil: meget kort og direkte, ingen kontekst. Eks: 'Resumér:\\n\\n{{DOKUMENT}}'",
]

_SUMMARY_MARKER_RE = re.compile(r"^RESUMÉ(?: AF:)?$")
_SECTION_HEADER_RE = re.compile(r"^(HVAD\b|INTRODUKTION\b|HOVEDPUNKTER\b)")


def build(cfg: dict[str, Any]) -> BuildResult:
    return run(
        _build_run,
        pack="eur_lex_sum",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    cfg = _load_cfg(result)
    verified_rows = _verify_rows(_load_rows(result), cfg)
    failures = [row for row in verified_rows if not _row_passes(row)]

    outputs_dir = Path(result.run_dir) / "outputs"
    store.write_jsonl(verified_rows, outputs_dir / "verified.jsonl")
    store.write_jsonl(failures, outputs_dir / "failures.jsonl")

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
    }


def summarize(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    cfg = _load_cfg(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else common_eval.aggregate_metrics(rows)

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "source": {
            "dataset": cfg["source"].get("dataset"),
            "path": cfg["source"].get("path"),
            "split": cfg["source"].get("split", "train"),
        },
        "artifacts": sorted(result.artifacts),
        "metrics": metrics,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    failures = [row for row in rows if not _row_passes(row)]
    train_rows, eval_rows = _split_rows(rows, cfg["generation"]["train_fraction"])

    target_dir = _publish_dir(result, out_dir)
    store.ensure_dir(target_dir)
    store.write_parquet(train_rows, target_dir / "train.parquet")
    store.write_parquet(eval_rows, target_dir / "eval.parquet")
    store.write_parquet(failures, target_dir / "failures.parquet")
    common_publish.write_preview(rows, target_dir / "sample_preview.jsonl", n=20)

    outputs_dir = Path(result.run_dir) / "outputs"
    metrics = _load_or_compute(outputs_dir / "metrics.json", common_eval.aggregate_metrics(rows))
    failure_summary = _load_or_compute(outputs_dir / "failure_summary.json", common_eval.summarize_failures(rows))

    write_json(metrics, target_dir / "metrics.json")
    write_json(failure_summary, target_dir / "failure_summary.json")
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
    del seed

    dataset_path = outputs_dir / "dataset.jsonl"
    failures_path = outputs_dir / "generation_failures.jsonl"
    resume_state = _load_resume_state(dataset_path=dataset_path, failures_path=failures_path)
    source_stats = _count_documents(cfg, processed_source_ids=resume_state["processed_source_ids"])
    writer = _load_instruction_writer(cfg)
    _progress_log(
        f"loaded {source_stats['pending_rows']} pending documents after scanning "
        f"{source_stats['scanned_rows']} rows"
    )
    completed_rows, failed_rows = _make_rows(
        cfg=cfg,
        writer=writer,
        dataset_path=dataset_path,
        failures_path=failures_path,
        processed_source_ids=resume_state["processed_source_ids"],
        pending_rows=source_stats["pending_rows"],
        starting_completed_rows=resume_state["completed_rows"],
        starting_failed_rows=resume_state["failed_rows"],
    )

    common_publish.write_preview(
        store.jsonl_prefix(dataset_path, limit=20),
        outputs_dir / "sample_preview.jsonl",
        n=20,
    )

    artifacts = {
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={
                "rows": completed_rows,
                "source": common_sources.source_label(cfg["source"]),
                "generation_failures": failed_rows,
                **source_stats,
            },
        )
    }
    artifacts["generation_failures"] = Artifact(
        name="generation_failures",
        path=str(failures_path),
        kind="jsonl",
        meta={"rows": failed_rows},
    )
    return artifacts


def _count_documents(
    cfg: dict[str, Any],
    *,
    processed_source_ids: set[str],
) -> dict[str, Any]:
    source = cfg["source"]
    generation = cfg["generation"]
    min_document_chars = _min_document_chars(generation)
    max_documents = _max_documents(generation)
    scanned_rows = 0
    usable_rows = 0
    too_short_rows = 0
    resumed_rows = 0
    pending_rows = 0
    kept_rows = 0

    for record in common_sources.iter_source_records(source):
        scanned_rows += 1
        example = _record_to_example(record, source, scanned_rows - 1)
        if example is None:
            continue

        usable_rows += 1
        if len(example["text"]) < min_document_chars:
            too_short_rows += 1
            continue

        if max_documents is not None and kept_rows >= max_documents:
            break

        kept_rows += 1
        if example["source_id"] in processed_source_ids:
            resumed_rows += 1
            continue

        pending_rows += 1

    return {
        "scanned_rows": scanned_rows,
        "usable_rows": usable_rows,
        "too_short_rows": too_short_rows,
        "resumed_rows": resumed_rows,
        "pending_rows": pending_rows,
        "min_document_chars": min_document_chars,
        "max_documents": max_documents,
    }


def _iter_documents(
    cfg: dict[str, Any],
    *,
    processed_source_ids: set[str],
):
    source = cfg["source"]
    generation = cfg["generation"]
    min_document_chars = _min_document_chars(generation)
    max_documents = _max_documents(generation)
    kept_rows = 0

    for index, record in enumerate(common_sources.iter_source_records(source)):
        example = _record_to_example(record, source, index)
        if example is None:
            continue
        if len(example["text"]) < min_document_chars:
            continue
        if max_documents is not None and kept_rows >= max_documents:
            break
        kept_rows += 1
        if example["source_id"] in processed_source_ids:
            continue
        yield example


def _record_to_example(
    record: dict[str, Any],
    source: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    text = common_sources.read_record_value(record, source.get("text_field", "text"))
    if not text:
        return None

    summary = common_sources.read_record_value(record, _summary_field(source))
    if not summary:
        return None

    title = common_sources.read_record_value(record, source.get("title_field", "title"))
    url = common_sources.read_record_value(record, source.get("url_field", "url"))
    source_id = (
        common_sources.read_record_value(record, source.get("id_field"))
        or url
        or title
        or str(index)
    )

    return {
        "row_index": index,
        "source_id": source_id,
        "title": title,
        "text": text,
        "summary": summary,
        "url": url,
    }


def _summary_field(source: dict[str, Any]) -> str:
    field_name = source.get("summary_field")
    assert isinstance(field_name, str) and field_name.strip(), "eur_lex_sum requires source.summary_field"
    return field_name


def _load_instruction_writer(cfg: dict[str, Any]) -> LLM:
    models = load_clients({"instruction_writer": cfg["models"]["instruction_writer"]})
    writer = models["instruction_writer"]
    assert isinstance(writer, LLM)
    return writer


def _make_rows(
    *,
    cfg: dict[str, Any],
    writer: LLM,
    dataset_path: Path,
    failures_path: Path,
    processed_source_ids: set[str],
    pending_rows: int,
    starting_completed_rows: int,
    starting_failed_rows: int,
) -> tuple[int, int]:
    return asyncio.run(
        _make_rows_async(
            cfg=cfg,
            writer=writer,
            dataset_path=dataset_path,
            failures_path=failures_path,
            processed_source_ids=processed_source_ids,
            pending_rows=pending_rows,
            starting_completed_rows=starting_completed_rows,
            starting_failed_rows=starting_failed_rows,
        )
    )


async def _make_rows_async(
    *,
    cfg: dict[str, Any],
    writer: LLM,
    dataset_path: Path,
    failures_path: Path,
    processed_source_ids: set[str],
    pending_rows: int,
    starting_completed_rows: int,
    starting_failed_rows: int,
) -> tuple[int, int]:
    generation = cfg["generation"]
    temperature = float(generation.get("temperature", 0.2))
    source = cfg["source"]
    worker_concurrency = common_concurrency.runtime_concurrency(writer)
    completed_rows = 0
    failed_rows = 0
    starting_processed_rows = starting_completed_rows + starting_failed_rows
    total_rows = starting_processed_rows + pending_rows

    _progress_log(f"generating prompts with worker_concurrency={worker_concurrency}")
    log_event(
        "eur_lex_sum",
        "generation_started",
        rows=total_rows,
        pending_rows=pending_rows,
        worker_concurrency=worker_concurrency,
    )
    progress, progress_state = _progress_reporter(
        total=total_rows,
        starting_completed=starting_processed_rows,
        worker_concurrency=worker_concurrency,
    )

    dataset_mode = "a" if dataset_path.exists() else "w"
    failures_mode = "a" if failures_path.exists() else "w"
    with dataset_path.open(dataset_mode) as dataset_handle, failures_path.open(failures_mode) as failures_handle:
        async for row in map_async_ordered(
            _iter_documents(cfg, processed_source_ids=processed_source_ids),
            lambda _ignored_index, example: _generate_row_result(
                example=example,
                writer=writer,
                temperature=temperature,
                source=source,
            ),
            concurrency=worker_concurrency,
            progress=progress,
            total=pending_rows,
        ):
            if "row" in row:
                store.append_jsonl_line(dataset_handle, row["row"])
                completed_rows += 1
                continue
            store.append_jsonl_line(failures_handle, row["failure"])
            failed_rows += 1

    _finish_progress(
        total=total_rows,
        worker_concurrency=worker_concurrency,
        elapsed_seconds=progress_state["elapsed_seconds"],
    )
    log_event(
        "eur_lex_sum",
        "generation_completed",
        rows=starting_completed_rows + completed_rows,
        failures=starting_failed_rows + failed_rows,
        worker_concurrency=worker_concurrency,
    )
    _progress_log(
        f"generated {starting_completed_rows + completed_rows} rows, "
        f"recorded {starting_failed_rows + failed_rows} failures"
    )

    return starting_completed_rows + completed_rows, starting_failed_rows + failed_rows


async def _generate_row(
    *,
    example: dict[str, Any],
    writer: LLM,
    temperature: float,
    source: dict[str, Any],
) -> dict[str, Any]:
    framing = await _generate_prompt(writer, _framing_messages(example), temperature=temperature)
    cleaned_framing = _clean_generated_prompt(framing)
    prompt = _assemble_prompt(cleaned_framing, example["text"])
    target = _strip_summary_header(example["summary"])
    return {
        "id": f"eur_lex_sum-{example['row_index']:06d}",
        "prompt": prompt,
        "target": target,
        "sources": [
            _drop_none(
                {
                    "dataset": source.get("dataset"),
                    "path": source.get("path"),
                    "split": source.get("split", "train"),
                    "row_id": example["source_id"],
                    "title": example["title"],
                    "url": example["url"],
                }
            )
        ],
        "meta": _drop_none(
            {
                "title": example["title"],
                "source_id": example["source_id"],
                "source_dataset": source.get("dataset"),
                "source_path": source.get("path"),
                "source_split": source.get("split", "train"),
                "source_row_index": example["row_index"],
                "document_chars": len(example["text"]),
                "target_chars": len(target),
            }
        ),
    }


async def _generate_row_result(
    *,
    example: dict[str, Any],
    writer: LLM,
    temperature: float,
    source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        row = await _generate_row(
            example=example,
            writer=writer,
            temperature=temperature,
            source=source,
        )
    except Exception as error:
        log_event(
            "eur_lex_sum",
            "row_failed",
            row_index=example["row_index"],
            source_id=example["source_id"],
            error_type=error.__class__.__name__,
            error_message=str(error),
        )
        return {
            "failure": {
                "id": f"eur_lex_sum-failure-{example['row_index']:06d}",
                "source_id": example["source_id"],
                "title": example["title"],
                "url": example["url"],
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            }
        }
    return {"row": row}


async def _generate_prompt(
    writer: LLM,
    messages: list[dict[str, str]],
    *,
    temperature: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            achat = getattr(writer, "achat", None)
            if callable(achat):
                return await achat(messages, temperature=temperature)
            return await asyncio.to_thread(writer.chat, messages, temperature=temperature)
        except Exception as error:
            last_error = error
            if attempt == 2:
                break
            await asyncio.sleep(float(attempt + 1))

    assert last_error is not None
    raise last_error


def _min_document_chars(generation: dict[str, Any]) -> int:
    return int(generation.get("min_document_chars", generation.get("min_article_chars", 0)))


def _max_documents(generation: dict[str, Any]) -> int | None:
    value = generation.get("max_documents", generation.get("max_articles"))
    if value is None:
        return None

    max_documents = int(value)
    assert max_documents >= 0, "generation.max_documents must be non-negative"
    return max_documents


def _progress_log(message: str) -> None:
    print(f"[eur_lex_sum] {message}", flush=True)


def _progress_reporter(
    *,
    total: int,
    starting_completed: int,
    worker_concurrency: int,
):
    state = {"finished": False, "elapsed_seconds": 0}
    common_progress.write_progress_snapshot(
        "eur_lex_sum_progress",
        stage="generating_prompts",
        completed=starting_completed,
        total=total,
        elapsed_seconds=0,
        extra={
            "worker_concurrency": worker_concurrency,
            "rows_per_minute": common_progress.items_per_minute(starting_completed, 0),
        },
        force=True,
    )
    _print_progress_bar(starting_completed, total, worker_concurrency, 0)
    snapshot_progress = common_progress.snapshot_progress_reporter(
        "eur_lex_sum_progress",
        stage="generating_prompts",
        completed_offset=starting_completed,
        total=total,
        extra=lambda completed, _total, elapsed: {
            "worker_concurrency": worker_concurrency,
            "rows_per_minute": common_progress.items_per_minute(completed, elapsed),
        },
    )

    def report(completed: int, reported_total: int | None, elapsed: int) -> None:
        state["elapsed_seconds"] = elapsed
        current_completed = starting_completed + completed
        snapshot_progress(completed, reported_total, elapsed)
        _print_progress_bar(current_completed, total, worker_concurrency, elapsed)
        if current_completed >= total and not state["finished"]:
            sys.stdout.write("\n")
            sys.stdout.flush()
            state["finished"] = True

    return report, state


def _finish_progress(
    *,
    total: int,
    worker_concurrency: int,
    elapsed_seconds: int,
) -> None:
    common_progress.write_progress_snapshot(
        "eur_lex_sum_progress",
        stage="completed",
        completed=total,
        total=total,
        elapsed_seconds=elapsed_seconds,
        extra={
            "worker_concurrency": worker_concurrency,
            "rows_per_minute": common_progress.items_per_minute(total, elapsed_seconds),
        },
        force=True,
    )


def _print_progress_bar(
    completed: int,
    total: int,
    worker_concurrency: int,
    elapsed_seconds: int,
) -> None:
    width = 24
    filled = width if total == 0 else min(int(width * completed / total), width)
    bar = "#" * filled + "-" * (width - filled)
    total_text = str(total)
    message = (
        f"\r[eur_lex_sum] [{bar}] {completed}/{total_text} "
        f"elapsed={elapsed_seconds}s concurrency={worker_concurrency}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def _load_resume_state(
    *,
    dataset_path: Path,
    failures_path: Path,
) -> dict[str, Any]:
    completed_source_ids = store.jsonl_keys(dataset_path, key_for=_source_id_from_output_row)
    failed_source_ids = store.jsonl_keys(failures_path, key_for=_source_id_from_failure_row)
    processed_source_ids = set(completed_source_ids)
    processed_source_ids.update(failed_source_ids)

    return {
        "processed_source_ids": processed_source_ids,
        "completed_rows": len(completed_source_ids),
        "failed_rows": len(failed_source_ids - completed_source_ids),
    }


def _source_id_from_output_row(row: dict[str, Any]) -> str | None:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return None
    source_id = meta.get("source_id")
    if not isinstance(source_id, str):
        return None
    return source_id


def _source_id_from_failure_row(row: dict[str, Any]) -> str | None:
    source_id = row.get("source_id")
    if not isinstance(source_id, str):
        return None
    return source_id


def _framing_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    style_seed = _STYLE_SEEDS[example["row_index"] % len(_STYLE_SEEDS)]
    return [
        {
            "role": "system",
            "content": (
                "Du skriver brugerbesked-skabeloner til en dansk opsummeringsassistent. "
                "Brugeren har et langt EU-juridisk dokument på dansk og ønsker det opsummeret.\n\n"
                "Skriv én realistisk, naturlig besked en bruger kunne sende. "
                "Placer markøren {{DOKUMENT}} præcis det sted i beskeden, hvor dokumentteksten skal indsættes. "
                "Følg den angivne stil, men vær fri til at variere den præcise ordlyd. "
                "Skriv KUN beskedskabelonen med {{DOKUMENT}}-markøren, intet andet."
            ),
        },
        {
            "role": "user",
            "content": style_seed,
        },
    ]


def _assemble_prompt(framing: str, document_text: str) -> str:
    if "{{DOKUMENT}}" in framing:
        return framing.replace("{{DOKUMENT}}", document_text)
    return framing.rstrip() + "\n\n" + document_text


def _strip_summary_header(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)

    for index, line in enumerate(lines):
        if _SUMMARY_MARKER_RE.match(line.strip()):
            lines = lines[index + 1 :]
            break

    while lines and not lines[0].strip():
        lines.pop(0)

    if lines and _SECTION_HEADER_RE.match(lines[0].strip()):
        lines = lines[1:]

    while lines and not lines[0].strip():
        lines.pop(0)

    return "\n".join(lines).strip()


def _clean_generated_prompt(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    return text


def _load_rows(result: BuildResult) -> list[dict[str, Any]]:
    dataset_path = Path(result.artifacts["dataset"].path)
    return store.read_jsonl(dataset_path)


def _load_verified_rows(result: BuildResult) -> list[dict[str, Any]]:
    verified_path = Path(result.run_dir) / "outputs" / "verified.jsonl"
    if verified_path.exists():
        return store.read_jsonl(verified_path)
    return _verify_rows(_load_rows(result), _load_cfg(result))


def _verify_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    verified_rows = common_eval.verify(rows, _has_prompt, name="prompt_present")
    verified_rows = common_eval.verify(verified_rows, _has_target, name="target_present")
    min_summary_chars = int(cfg["generation"].get("min_summary_chars", 0))
    if min_summary_chars <= 0:
        return verified_rows
    return common_eval.verify(
        verified_rows,
        lambda row: _target_meets_min_chars(row, min_summary_chars),
        name="target_min_chars",
    )


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _has_prompt(row: dict[str, Any]) -> bool:
    return bool(str(row.get("prompt", "")).strip())


def _has_target(row: dict[str, Any]) -> bool:
    return bool(str(row.get("target", "")).strip())


def _target_meets_min_chars(row: dict[str, Any], min_target_chars: int) -> bool:
    return len(str(row.get("target", "")).strip()) >= min_target_chars


def _row_passes(row: dict[str, Any]) -> bool:
    checks = row.get("checks", {})
    assert isinstance(checks, dict), "row checks must be a mapping"
    return all(_check_passed(value) for value in checks.values())


def _check_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value

    assert isinstance(value, dict), "check value must be a bool or a status mapping"
    if "passed" in value:
        return bool(value["passed"])
    if "ok" in value:
        return bool(value["ok"])
    raise AssertionError("check status mapping must contain 'passed' or 'ok'")


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


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
