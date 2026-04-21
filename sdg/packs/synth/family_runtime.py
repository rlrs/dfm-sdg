from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypedDict

from sdg.commons import Artifact, store
from sdg.commons import concurrency as common_concurrency
from sdg.commons import progress as common_progress
from sdg.commons import publish as common_publish
from sdg.commons.model import LLM, load_clients
from sdg.commons.run_log import log_event, write_snapshot
from sdg.commons.work_queue import map_async_unordered
from sdg.packs.synth.languages import LanguagePlan, load_language_plan
from sdg.packs.synth.rows import materialize_row


class FamilyStats(TypedDict):
    rows: int
    candidate_rows: int
    rejected_rows: int


class ResumeState(TypedDict):
    candidate_ids: set[str]
    stats: FamilyStats
    reject_reasons: dict[str, int]


class IndexedQueryPlan(TypedDict):
    row_index: int
    plan: dict[str, Any]


class CommonFamilySettings(TypedDict):
    max_rows: int
    lead_sentences: int
    max_sentences_per_doc: int
    min_sentence_words: int
    max_sentence_words: int
    structured_facts: int
    language_plan: LanguagePlan


CandidateRowWorker = Callable[[IndexedQueryPlan], Awaitable[dict[str, Any]]]
AnnotateRow = Callable[[dict[str, Any], list[str]], dict[str, Any]]
FilterReasons = Callable[[dict[str, Any]], list[str]]


class FamilyProgressTracker:
    def __init__(
        self,
        *,
        snapshot_name: str,
        worker_concurrency: int,
        stats: FamilyStats,
        reject_reasons: dict[str, int],
    ):
        self.snapshot_name = snapshot_name
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
        write_snapshot(
            self.snapshot_name,
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
                "candidates_per_minute": common_progress.items_per_minute(self.candidate_rows, elapsed),
            },
            force=force,
            min_interval_seconds=1.0,
        )


def progress_log(message: str) -> None:
    print(f"[synth] {message}", flush=True)


def progress_reporter(
    label: str,
    tracker: FamilyProgressTracker,
) -> Callable[[int, int | None, int], None]:
    next_log = {"count": 1}

    def report(completed: int, total: int | None, elapsed: int) -> None:
        tracker.on_progress(completed, total, elapsed)
        total_text = "?" if total is None else str(total)
        if completed == 0:
            progress_log(f"{label}: 0/{total_text}")
            return

        if completed < next_log["count"]:
            return

        progress_log(f"{label}: {completed}/{total_text} ({elapsed}s)")
        if total is not None and total <= 10:
            next_log["count"] = completed + 1
            return
        if total is None:
            next_log["count"] = completed + 10
            return
        next_log["count"] = completed + max(1, total // 10)

    return report
def load_family_models(cfg: dict[str, Any], *, family: str) -> dict[str, LLM]:
    model_refs = cfg.get("models", {})
    required_roles = ["query_teacher", "answer_teacher", "judge"]
    missing = [role for role in required_roles if role not in model_refs]
    if missing:
        raise ValueError(f"Missing model roles for {family} LLM path: {', '.join(missing)}")

    requested_roles = dict(model_refs)
    if "reasoning_teacher" not in requested_roles:
        requested_roles["reasoning_teacher"] = requested_roles["answer_teacher"]
    if "task_planner" not in requested_roles:
        requested_roles["task_planner"] = requested_roles["query_teacher"]

    roles = ["task_planner", "query_teacher", "reasoning_teacher", "answer_teacher", "judge"]
    clients = load_clients({role: requested_roles[role] for role in roles})
    return {role: clients[role] for role in roles}


def generation_family_cfg(cfg: dict[str, Any], *, family: str) -> tuple[dict[str, Any], dict[str, Any]]:
    generation_cfg = cfg.get("generation", {})
    assert isinstance(generation_cfg, dict), "generation config must be a mapping"

    family_cfg = generation_cfg.get(family, {})
    assert isinstance(family_cfg, dict), f"{family} config must be a mapping"
    return generation_cfg, family_cfg


def positive_int(record: dict[str, Any], key: str, *, default: int) -> int:
    value = record.get(key)
    if value is None:
        return default
    assert isinstance(value, int) and value > 0, f"{key} must be a positive integer"
    return value


def common_family_settings(
    cfg: dict[str, Any],
    *,
    family: str,
    family_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    max_sentences_per_doc_default: int,
) -> CommonFamilySettings:
    return {
        "max_rows": positive_int(
            family_cfg,
            "max_rows",
            default=positive_int(generation_cfg, "max_rows_per_family", default=200),
        ),
        "lead_sentences": positive_int(family_cfg, "lead_sentences", default=8),
        "max_sentences_per_doc": positive_int(
            family_cfg,
            "max_sentences_per_doc",
            default=max_sentences_per_doc_default,
        ),
        "min_sentence_words": positive_int(family_cfg, "min_sentence_words", default=5),
        "max_sentence_words": positive_int(family_cfg, "max_sentence_words", default=90),
        "structured_facts": positive_int(family_cfg, "structured_facts", default=4),
        "language_plan": load_language_plan(cfg, family=family),
    }


def family_artifacts(
    family: str,
    outputs_dir: Path,
    stats: FamilyStats,
) -> dict[str, Artifact]:
    row_counts = {
        "rows": stats["rows"],
        "candidates": stats["candidate_rows"],
        "rejected": stats["rejected_rows"],
    }
    return {
        f"{family}_{name}": Artifact(
            name=f"{family}_{name}",
            path=str(outputs_dir / f"{family}_{name}.jsonl"),
            kind="jsonl",
            meta={
                "rows": row_counts[name],
                "family": family,
            },
        )
        for name in row_counts
    }


async def write_family_outputs_async(
    *,
    family: str,
    query_plans: Iterable[dict[str, Any]],
    outputs_dir: Path,
    models: dict[str, LLM],
    generate_candidate_row: CandidateRowWorker,
    row_filter_reasons: FilterReasons,
    annotate_filter_result: AnnotateRow,
) -> FamilyStats:
    rows_path = outputs_dir / f"{family}_rows.jsonl"
    candidate_path = outputs_dir / f"{family}_candidates.jsonl"
    rejected_path = outputs_dir / f"{family}_rejected.jsonl"
    current_worker_concurrency = common_concurrency.effective_concurrency(models.values())
    resume_state = load_resume_state(outputs_dir, family=family)
    stats = dict(resume_state["stats"])
    tracker = FamilyProgressTracker(
        snapshot_name=f"{family}_progress.json",
        worker_concurrency=current_worker_concurrency,
        stats=stats,
        reject_reasons=resume_state["reject_reasons"],
    )
    progress = progress_reporter(f"{family}.rows", tracker)
    indexed_plans = indexed_query_plans(query_plans)
    pending_plans = pending_query_plans(
        indexed_plans,
        resume_state["candidate_ids"],
        family=family,
    )

    resume_count = len(resume_state["candidate_ids"])
    if resume_count:
        progress_log(
            f"{family}: resuming with {resume_count} existing candidates and "
            f"worker_concurrency={current_worker_concurrency}"
        )
        log_event(
            "synth",
            f"{family}_resumed",
            existing_candidate_rows=resume_count,
            existing_rows=stats["rows"],
            existing_rejected_rows=stats["rejected_rows"],
            worker_concurrency=current_worker_concurrency,
        )
    else:
        progress_log(f"{family}: generating rows with worker_concurrency={current_worker_concurrency}")

    log_event(
        "synth",
        f"{family}_started",
        worker_concurrency=current_worker_concurrency,
        resumed=bool(resume_count),
    )
    tracker.start()
    file_mode = "a" if resume_count else "w"
    with rows_path.open(file_mode) as rows_handle, candidate_path.open(file_mode) as candidate_handle, rejected_path.open(file_mode) as rejected_handle:
        async for row in map_async_unordered(
            pending_plans,
            lambda _ignored_index, item: generate_candidate_row(item),
            concurrency=current_worker_concurrency,
            progress=progress,
        ):
            stats["candidate_rows"] += 1
            store.append_jsonl_line(candidate_handle, materialize_row(row))

            reasons = row_filter_reasons(row)
            annotated = annotate_filter_result(row, reasons)
            tracker.on_row(reasons)
            if reasons:
                stats["rejected_rows"] += 1
                store.append_jsonl_line(rejected_handle, materialize_row(annotated))
                continue

            stats["rows"] += 1
            store.append_jsonl_line(rows_handle, materialize_row(annotated))

    common_publish.write_preview(
        store.jsonl_prefix(rows_path, limit=50),
        outputs_dir / f"{family}_preview.jsonl",
        n=50,
    )
    common_publish.write_preview(
        store.jsonl_prefix(rejected_path, limit=50),
        outputs_dir / f"{family}_rejected_preview.jsonl",
        n=50,
    )
    tracker.finish()
    log_event(
        "synth",
        f"{family}_completed",
        worker_concurrency=current_worker_concurrency,
        candidate_rows=stats["candidate_rows"],
        rows=stats["rows"],
        rejected_rows=stats["rejected_rows"],
    )
    progress_log(f"{family}: kept {stats['rows']} rows, rejected {stats['rejected_rows']}")
    return stats


def load_resume_state(outputs_dir: Path, *, family: str) -> ResumeState:
    candidate_ids: set[str] = set()
    stats: FamilyStats = {"rows": 0, "candidate_rows": 0, "rejected_rows": 0}
    reject_reasons: dict[str, int] = {}

    candidate_path = outputs_dir / f"{family}_candidates.jsonl"
    candidate_ids = store.jsonl_keys(candidate_path, key_for=_candidate_row_id)
    stats["candidate_rows"] = len(candidate_ids)

    rows_path = outputs_dir / f"{family}_rows.jsonl"
    stats["rows"] = store.jsonl_count(rows_path)

    rejected_path = outputs_dir / f"{family}_rejected.jsonl"
    if rejected_path.exists():
        stats["rejected_rows"] = store.jsonl_count(rejected_path)
        for row in store.iter_jsonl(rejected_path):
            reasons = row.get("hidden", {}).get("generation_filter", {}).get("reasons", [])
            for reason in reasons:
                key = str(reason)
                reject_reasons[key] = reject_reasons.get(key, 0) + 1

    return {
        "candidate_ids": candidate_ids,
        "stats": stats,
        "reject_reasons": reject_reasons,
    }


def indexed_query_plans(query_plans: Iterable[dict[str, Any]]) -> Iterator[IndexedQueryPlan]:
    for row_index, plan in enumerate(query_plans):
        yield {
            "row_index": row_index,
            "plan": plan,
        }


def pending_query_plans(
    indexed_query_plans: Iterable[IndexedQueryPlan],
    existing_candidate_ids: set[str],
    *,
    family: str,
) -> Iterator[IndexedQueryPlan]:
    for item in indexed_query_plans:
        row_id = f"{family}-{item['row_index']:06d}"
        if row_id in existing_candidate_ids:
            continue
        yield item


def _candidate_row_id(row: dict[str, Any]) -> str | None:
    row_id = row.get("id")
    if row_id is None:
        return None
    return str(row_id)
