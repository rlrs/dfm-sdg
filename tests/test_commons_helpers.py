from __future__ import annotations

import sys
from types import SimpleNamespace

from sdg.commons.concurrency import effective_concurrency, runtime_concurrency
from sdg.commons.progress import items_per_minute, snapshot_progress_reporter
from sdg.commons.run_log import activate_run_log
from sdg.commons.sources import iter_source_records, read_record_value, source_label
from sdg.commons.store import read_blob, write_jsonl


def test_read_record_value_supports_nested_paths() -> None:
    record = {
        "title": "Example",
        "meta": {
            "nested": {
                "value": "  kept  ",
            }
        },
    }

    assert read_record_value(record, "title") == "Example"
    assert read_record_value(record, "meta.nested.value") == "kept"
    assert read_record_value(record, "meta.missing") is None
    assert read_record_value(record, None) is None


def test_iter_source_records_reads_jsonl_path(tmp_path) -> None:
    source_path = tmp_path / "source.jsonl"
    rows = [{"id": "a1"}, {"id": "a2"}]
    write_jsonl(rows, source_path)

    assert list(iter_source_records({"path": str(source_path)})) == rows


def test_iter_source_records_respects_streaming_default(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_load_dataset(**kwargs):
        calls.append(kwargs)
        return ["ok"]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    records = iter_source_records({"dataset": "demo", "split": "train"}, default_streaming=True)
    assert list(records) == ["ok"]
    assert calls == [
        {
            "path": "demo",
            "name": None,
            "split": "train",
            "streaming": True,
        }
    ]


def test_source_label_prefers_dataset() -> None:
    assert source_label({"dataset": "org/name", "path": "/tmp/source.jsonl"}) == "org/name"
    assert source_label({"path": "/tmp/source.jsonl"}) == "/tmp/source.jsonl"


def test_runtime_concurrency_helpers_use_model_runtime_limits() -> None:
    slow = SimpleNamespace(runtime=SimpleNamespace(max_concurrency=2))
    fast = SimpleNamespace(runtime=SimpleNamespace(max_concurrency=8))
    unknown = SimpleNamespace()

    assert runtime_concurrency(slow) == 2
    assert runtime_concurrency(unknown) == 1
    assert effective_concurrency([slow, fast, unknown]) == 8


def test_snapshot_progress_reporter_writes_snapshot(tmp_path) -> None:
    run_dir = tmp_path / "run"
    reporter = snapshot_progress_reporter(
        "demo_progress",
        stage="working",
        completed_offset=2,
        total=10,
        extra=lambda completed, _total, elapsed: {
            "rate": items_per_minute(completed, elapsed),
        },
    )

    with activate_run_log(run_dir):
        reporter(3, None, 15)

    assert read_blob(run_dir / "outputs" / "demo_progress.json") == {
        "stage": "working",
        "completed": 5,
        "total": 10,
        "elapsed_seconds": 15,
        "rate": round(5 * 60 / 15, 2),
    }
