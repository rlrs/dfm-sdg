from __future__ import annotations

from pathlib import Path

from sdg.commons import store
from sdg.commons.run import Artifact
from sdg.commons.viewer import (
    _build_jsonl_offsets,
    _default_artifact_name,
    _discover_live_artifacts,
    _resolve_artifact_view,
    _slice_jsonl_rows_with_offsets,
    _viewer_item,
)


def test_jsonl_offsets_support_direct_paging(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"id": f"row-{index}", "value": index} for index in range(6)]
    store.write_jsonl(rows, path)

    offsets = _build_jsonl_offsets(path)

    assert len(offsets) == 6
    assert _slice_jsonl_rows_with_offsets(path, offsets, offset=0, limit=2) == rows[:2]
    assert _slice_jsonl_rows_with_offsets(path, offsets, offset=2, limit=2) == rows[2:4]
    assert _slice_jsonl_rows_with_offsets(path, offsets, offset=5, limit=3) == rows[5:]


def test_discover_live_artifacts_finds_jsonl_outputs(tmp_path) -> None:
    run_dir = tmp_path / "run"
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    rows_path = outputs_dir / "grounded_qa_rows.jsonl"
    store.write_jsonl([{"id": "row-1"}], rows_path)

    artifacts = _discover_live_artifacts(Path(run_dir), {})

    assert "grounded_qa_rows" in artifacts
    assert artifacts["grounded_qa_rows"] == Artifact(
        name="grounded_qa_rows",
        path=str(rows_path),
        kind="jsonl",
        meta={},
    )


def test_viewer_item_has_stable_item_and_section_keys() -> None:
    row = {
        "id": "row-7",
        "prompt": "Question",
        "target": "Answer",
        "reasoning": "Reasoning",
    }

    view = _resolve_artifact_view([row], {})
    item = _viewer_item(row, view)

    assert item["key"] == "row-7"
    assert [section["key"] for section in item["sections"]] == ["prompt", "reasoning", "target"]


def test_default_artifact_name_prefers_pack_viewer_order() -> None:
    artifacts = {
        "dataset": Artifact(name="dataset", path="/tmp/dataset.jsonl", kind="jsonl", meta={}),
        "memory_chunks": Artifact(name="memory_chunks", path="/tmp/memory_chunks.jsonl", kind="jsonl", meta={}),
    }
    spec = {
        "artifacts": {
            "memory_chunks": {"label": "Chunks"},
            "dataset": {"label": "Rows"},
        }
    }

    assert _default_artifact_name(spec, artifacts) == "memory_chunks"
