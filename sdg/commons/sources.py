from __future__ import annotations

from pathlib import Path
from typing import Any

from sdg.commons import store


def iter_source_records(
    source: dict[str, Any],
    *,
    default_streaming: bool | None = None,
):
    path = source.get("path")
    if path:
        return store.iter_jsonl(Path(path).expanduser().resolve())

    from datasets import load_dataset

    load_kwargs: dict[str, Any] = {
        "path": source["dataset"],
        "name": source.get("config_name"),
        "split": source.get("split", "train"),
    }
    if default_streaming is not None or "streaming" in source:
        load_kwargs["streaming"] = bool(source.get("streaming", default_streaming))
    return load_dataset(**load_kwargs)


def read_record_value(record: dict[str, Any], field_name: str | None) -> str | None:
    if not field_name:
        return None

    value: Any = record
    for part in field_name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None

    text = str(value).strip()
    if not text:
        return None
    return text


def source_label(source: dict[str, Any]) -> str:
    if source.get("dataset"):
        return str(source["dataset"])
    return str(source["path"])
