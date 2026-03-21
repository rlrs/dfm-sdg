from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sdg.commons.store import ensure_dir
from sdg.commons.utils import iso_timestamp, write_json


class StructuredRunLog:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.logs_dir = ensure_dir(self.run_dir / "logs")
        self.outputs_dir = ensure_dir(self.run_dir / "outputs")
        self.events_path = self.logs_dir / "events.jsonl"
        self._lock = threading.Lock()
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._next_snapshot_write: dict[str, float] = {}

    def event(self, component: str, event: str, **data: Any) -> None:
        row = {
            "ts": iso_timestamp(),
            "component": component,
            "event": event,
            "data": data,
        }
        payload = json.dumps(row, sort_keys=True)
        with self._lock:
            with self.events_path.open("a") as handle:
                handle.write(payload)
                handle.write("\n")

    def snapshot(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        force: bool = False,
        min_interval_seconds: float = 1.0,
    ) -> None:
        filename = name if name.endswith(".json") else f"{name}.json"
        should_write = force
        now = time.monotonic()
        with self._lock:
            self._snapshots[filename] = payload
            if not should_write:
                should_write = now >= self._next_snapshot_write.get(filename, 0.0)
            if not should_write:
                return
            write_json(payload, self.outputs_dir / filename)
            self._next_snapshot_write[filename] = now + min_interval_seconds

    def flush(self) -> None:
        with self._lock:
            snapshots = dict(self._snapshots)
        for filename, payload in snapshots.items():
            write_json(payload, self.outputs_dir / filename)


_RUN_LOG_STACK: list[StructuredRunLog] = []
_RUN_LOG_LOCK = threading.Lock()


@contextmanager
def activate_run_log(run_dir: str | Path) -> Iterator[StructuredRunLog]:
    logger = StructuredRunLog(run_dir)
    with _RUN_LOG_LOCK:
        _RUN_LOG_STACK.append(logger)
    try:
        yield logger
    finally:
        logger.flush()
        with _RUN_LOG_LOCK:
            assert _RUN_LOG_STACK and _RUN_LOG_STACK[-1] is logger
            _RUN_LOG_STACK.pop()


def current_run_log() -> StructuredRunLog | None:
    with _RUN_LOG_LOCK:
        if not _RUN_LOG_STACK:
            return None
        return _RUN_LOG_STACK[-1]


def log_event(component: str, event: str, **data: Any) -> None:
    logger = current_run_log()
    if logger is None:
        return
    logger.event(component, event, **data)


def write_snapshot(
    name: str,
    payload: dict[str, Any],
    *,
    force: bool = False,
    min_interval_seconds: float = 1.0,
) -> None:
    logger = current_run_log()
    if logger is None:
        return
    logger.snapshot(
        name,
        payload,
        force=force,
        min_interval_seconds=min_interval_seconds,
    )
