from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from sdg.commons.run_log import write_snapshot

ProgressFn = Callable[[int, int | None, int], None]
ExtraFields = Mapping[str, Any] | Callable[[int, int | None, int], Mapping[str, Any]]


def items_per_minute(completed: int, elapsed_seconds: int | None) -> float:
    if elapsed_seconds is None or elapsed_seconds <= 0:
        return 0.0
    return round(completed * 60 / elapsed_seconds, 2)


def write_progress_snapshot(
    snapshot_name: str,
    *,
    stage: str,
    completed: int,
    total: int | None,
    elapsed_seconds: int | None,
    extra: Mapping[str, Any] | None = None,
    force: bool = False,
) -> None:
    payload = {
        "stage": stage,
        "completed": completed,
        "total": total,
        "elapsed_seconds": elapsed_seconds,
    }
    if extra:
        payload.update(extra)
    write_snapshot(
        snapshot_name,
        payload,
        force=force,
        min_interval_seconds=1.0,
    )


def snapshot_progress_reporter(
    snapshot_name: str,
    *,
    stage: str,
    completed_offset: int = 0,
    total: int | None = None,
    extra: ExtraFields | None = None,
) -> ProgressFn:
    def report(completed: int, reported_total: int | None, elapsed: int) -> None:
        resolved_total = total if total is not None else reported_total
        resolved_extra: Mapping[str, Any] | None = None
        if callable(extra):
            resolved_extra = extra(completed + completed_offset, resolved_total, elapsed)
        elif extra is not None:
            resolved_extra = extra
        write_progress_snapshot(
            snapshot_name,
            stage=stage,
            completed=completed + completed_offset,
            total=resolved_total,
            elapsed_seconds=elapsed,
            extra=resolved_extra,
        )

    return report
