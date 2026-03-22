from __future__ import annotations

import hashlib
import json
import threading
import webbrowser
from dataclasses import dataclass, field
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pyarrow as pa
import pyarrow.parquet as pq
from markdown import markdown

from sdg.commons.registry import load_pack
from sdg.commons.run import Artifact, load, progress as load_progress


@dataclass
class _ViewerContext:
    run: Any
    spec: dict[str, Any]
    artifacts: dict[str, Artifact]
    artifact_specs: dict[str, dict[str, Any]]
    default_artifact: str
    summary: Any
    default_page_size: int
    row_count_cache: dict[str, int] = field(default_factory=dict)
    view_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    filter_cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    jsonl_index_cache: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class RunningViewerServer:
    base_url: str
    server: ThreadingHTTPServer
    thread: threading.Thread

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def wait(self) -> None:
        self.thread.join()


def render_run_view(
    run_id_or_path: str,
    *,
    artifact_name: str | None = None,
    limit: int = 200,
    out_path: str | None = None,
) -> dict[str, Any]:
    assert limit > 0, "limit must be positive"

    context = _load_viewer_context(run_id_or_path, artifact_name=artifact_name)
    artifact_payloads: dict[str, dict[str, Any]] = {}
    for name, artifact in context.artifacts.items():
        selected = name == context.default_artifact
        artifact_spec = context.artifact_specs.get(name, {})
        artifact_payloads[name] = _artifact_payload(
            name,
            artifact,
            artifact_spec,
            limit=limit,
            selected=selected,
            default_page_size=context.default_page_size,
        )

    payload = {
        "title": context.spec.get("title") or f"{context.run.pack} viewer",
        "run": {
            "run_id": context.run.run_id,
            "pack": context.run.pack,
            "status": context.run.status,
            "run_dir": context.run.run_dir,
        },
        "summary": context.summary,
        "default_artifact": context.default_artifact,
        "artifacts": artifact_payloads,
    }

    target = _target_path(context.run.run_dir, out_path)
    target.write_text(_viewer_html(payload))
    return {
        "run_id": context.run.run_id,
        "pack": context.run.pack,
        "out_path": str(target),
        "default_artifact": context.default_artifact,
        "artifacts": {
            name: {
                "rows": artifact_payloads[name]["row_count"],
                "shown_rows": artifact_payloads[name]["shown_rows"],
                "page_size": artifact_payloads[name]["page_size"],
            }
            for name in artifact_payloads
        },
    }


def start_viewer_server(
    run_id_or_path: str,
    *,
    artifact_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    open_browser: bool = False,
) -> RunningViewerServer:
    context = _load_viewer_context(run_id_or_path, artifact_name=artifact_name)
    app = _ViewerApp(context)
    server = ThreadingHTTPServer((host, port), _viewer_handler(app))
    actual_host, actual_port = server.server_address[:2]
    base_url = f"http://{actual_host}:{actual_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if open_browser:
        webbrowser.open(base_url)
    return RunningViewerServer(base_url=base_url, server=server, thread=thread)


def _viewable_artifacts(artifacts: dict[str, Artifact]) -> dict[str, Artifact]:
    return {
        name: artifact
        for name, artifact in artifacts.items()
        if artifact.kind in {"jsonl", "parquet"}
    }


def _discover_live_artifacts(run_dir: Path, artifacts: dict[str, Artifact]) -> dict[str, Artifact]:
    outputs_dir = run_dir / "outputs"
    discovered = dict(artifacts)
    if not outputs_dir.exists():
        return discovered

    for path in sorted(outputs_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix not in {".jsonl", ".parquet"}:
            continue
        name = path.stem
        if name in discovered:
            continue
        kind = "jsonl" if path.suffix == ".jsonl" else "parquet"
        discovered[name] = Artifact(
            name=name,
            path=str(path),
            kind=kind,
            meta={},
        )
    return discovered


def _load_viewer_context(
    run_id_or_path: str,
    *,
    artifact_name: str | None,
) -> _ViewerContext:
    result = load(run_id_or_path)
    pack = load_pack(result.pack)
    spec = pack.viewer() if pack.viewer else {}
    artifacts = _viewable_artifacts(_discover_live_artifacts(Path(result.run_dir), result.artifacts))
    assert artifacts, "run has no viewable artifacts"
    default_artifact = artifact_name or _default_artifact_name(spec, artifacts)
    assert default_artifact in artifacts, f"unknown artifact: {default_artifact}"
    return _ViewerContext(
        run=result,
        spec=spec,
        artifacts=artifacts,
        artifact_specs=spec.get("artifacts", {}),
        default_artifact=default_artifact,
        summary=_compact_summary(pack.summarize(run_id_or_path)),
        default_page_size=int(spec.get("default_page_size", 40)),
    )


def _default_artifact_name(spec: dict[str, Any], artifacts: dict[str, Artifact]) -> str:
    preferred = str(spec.get("default_artifact", "")).strip()
    if preferred and preferred in artifacts:
        return preferred

    configured_artifacts = spec.get("artifacts", {})
    if isinstance(configured_artifacts, dict):
        for name in configured_artifacts:
            if name in artifacts:
                return name

    return next(iter(artifacts))


def _artifact_payload(
    name: str,
    artifact: Artifact,
    artifact_spec: dict[str, Any],
    *,
    limit: int,
    selected: bool,
    default_page_size: int,
) -> dict[str, Any]:
    row_limit = _artifact_limit(artifact_spec, limit=limit, selected=selected)
    rows, row_count = _read_rows(Path(artifact.path), artifact.kind, limit=row_limit, row_count_hint=artifact.meta.get("rows"))

    view = _resolve_artifact_view(rows, artifact_spec)
    items = [_viewer_item(row, view) for row in rows]
    filters = _filter_payloads(items, view["filters"])

    return {
        "name": name,
        "label": artifact_spec.get("label") or name.replace("_", " "),
        "kind": artifact.kind,
        "row_count": row_count,
        "shown_rows": len(rows),
        "truncated": row_count > len(rows),
        "page_size": int(artifact_spec.get("page_size", default_page_size)),
        "items": items,
        "filters": filters,
    }


def _artifact_summary(
    context: _ViewerContext,
    name: str,
    artifact: Artifact,
) -> dict[str, Any]:
    artifact_spec = context.artifact_specs.get(name, {})
    return {
        "name": name,
        "label": artifact_spec.get("label") or name.replace("_", " "),
        "kind": artifact.kind,
        "row_count": _artifact_row_count(context, name, artifact),
        "page_size": int(artifact_spec.get("page_size", context.default_page_size)),
    }


def _run_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text())


def _run_is_active(context: _ViewerContext) -> bool:
    status = _run_manifest(Path(context.run.run_dir)).get("status")
    return status == "running"


def _artifact_view_from_context(
    context: _ViewerContext,
    name: str,
) -> dict[str, Any]:
    cached = context.view_cache.get(name)
    if cached is not None:
        return cached

    artifact = context.artifacts[name]
    sample_rows, _row_count = _read_rows(Path(artifact.path), artifact.kind, limit=1, row_count_hint=artifact.meta.get("rows"))
    view = _resolve_artifact_view(sample_rows, context.artifact_specs.get(name, {}))
    context.view_cache[name] = view
    return view


def _artifact_row_count(
    context: _ViewerContext,
    name: str,
    artifact: Artifact,
) -> int:
    if not _run_is_active(context):
        cached = context.row_count_cache.get(name)
        if cached is not None:
            return cached

    hinted = artifact.meta.get("rows")
    if isinstance(hinted, int) and not _run_is_active(context):
        context.row_count_cache[name] = hinted
        return hinted

    path = Path(artifact.path)
    if artifact.kind == "parquet":
        count = pq.ParquetFile(path).metadata.num_rows
        context.row_count_cache[name] = count
        return count

    count = len(_jsonl_offsets(context, name, artifact))
    context.row_count_cache[name] = count
    return count


def _artifact_filter_payloads(
    context: _ViewerContext,
    name: str,
) -> list[dict[str, Any]]:
    if not _run_is_active(context):
        cached = context.filter_cache.get(name)
        if cached is not None:
            return cached

    artifact = context.artifacts[name]
    view = _artifact_view_from_context(context, name)
    option_sets = {filter_spec["key"]: set() for filter_spec in view["filters"]}
    for row in _iter_artifact_rows(Path(artifact.path), artifact.kind):
        item = _viewer_item(row, view)
        for filter_spec in view["filters"]:
            key = filter_spec["key"]
            for value in item["filters"].get(key, []):
                if value:
                    option_sets[key].add(value)

    payloads = [
        {
            "key": filter_spec["key"],
            "label": filter_spec["label"],
            "options": sorted(option_sets[filter_spec["key"]]),
        }
        for filter_spec in view["filters"]
    ]
    context.filter_cache[name] = payloads
    return payloads


def _artifact_page_payload(
    context: _ViewerContext,
    name: str,
    *,
    page: int,
    page_size: int,
    query: str,
    filters: dict[str, str],
) -> dict[str, Any]:
    artifact = context.artifacts[name]
    artifact_summary = _artifact_summary(context, name, artifact)
    view = _artifact_view_from_context(context, name)
    filter_payloads = _artifact_filter_payloads(context, name)
    normalized_query = query.strip().lower()
    active_filters = {key: value for key, value in filters.items() if value}
    page = max(page, 1)
    page_size = max(1, min(page_size, 200))
    offset = (page - 1) * page_size

    if not normalized_query and not active_filters:
        rows = _slice_artifact_rows(
            Path(artifact.path),
            artifact.kind,
            offset=offset,
            limit=page_size,
            jsonl_offsets=_jsonl_offsets(context, name, artifact) if artifact.kind == "jsonl" else None,
        )
        items = [_viewer_item(row, view) for row in rows]
        filtered_count = artifact_summary["row_count"]
    else:
        items = []
        filtered_count = 0
        for row in _iter_artifact_rows(Path(artifact.path), artifact.kind):
            item = _viewer_item(row, view)
            if not _item_matches(item, normalized_query, active_filters):
                continue
            if filtered_count >= offset and len(items) < page_size:
                items.append(item)
            filtered_count += 1

    total_pages = max(1, (filtered_count + page_size - 1) // page_size)
    return {
        "artifact": artifact_summary,
        "filters": filter_payloads,
        "page": min(page, total_pages),
        "page_size": page_size,
        "filtered_count": filtered_count,
        "total_pages": total_pages,
        "items": items,
    }


def _artifact_limit(artifact_spec: dict[str, Any], *, limit: int, selected: bool) -> int:
    if selected:
        return limit
    preview_limit = artifact_spec.get("preview_limit")
    if isinstance(preview_limit, int) and preview_limit > 0:
        return min(limit, preview_limit)
    return min(limit, 40)


def _read_rows(
    path: Path,
    kind: str,
    *,
    limit: int,
    row_count_hint: Any,
) -> tuple[list[dict[str, Any]], int]:
    if kind == "jsonl":
        rows: list[dict[str, Any]] = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if len(rows) >= limit:
                    break
        row_count = int(row_count_hint) if isinstance(row_count_hint, int) else len(rows)
        return rows, row_count

    if kind == "parquet":
        parquet = pq.ParquetFile(path)
        rows: list[dict[str, Any]] = []
        for batch in parquet.iter_batches(batch_size=min(limit, 512)):
            rows.extend(pa.Table.from_batches([batch]).to_pylist())
            if len(rows) >= limit:
                break
        return rows[:limit], parquet.metadata.num_rows

    raise AssertionError(f"unsupported viewer artifact kind: {kind}")


def _jsonl_offsets(
    context: _ViewerContext,
    name: str,
    artifact: Artifact,
) -> list[int]:
    if not _run_is_active(context):
        cached = context.jsonl_index_cache.get(name)
        if cached is not None:
            return cached

    assert artifact.kind == "jsonl", "jsonl offsets are only valid for jsonl artifacts"
    offsets = _build_jsonl_offsets(Path(artifact.path))
    context.jsonl_index_cache[name] = offsets
    context.row_count_cache[name] = len(offsets)
    return offsets


def _build_jsonl_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return offsets
            if line.strip():
                offsets.append(offset)


def _iter_artifact_rows(path: Path, kind: str):
    if kind == "jsonl":
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    if kind == "parquet":
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            for row in pa.Table.from_batches([batch]).to_pylist():
                yield row
        return

    raise AssertionError(f"unsupported viewer artifact kind: {kind}")


def _slice_artifact_rows(
    path: Path,
    kind: str,
    *,
    offset: int,
    limit: int,
    jsonl_offsets: list[int] | None = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    if kind == "jsonl" and jsonl_offsets is not None:
        return _slice_jsonl_rows_with_offsets(path, jsonl_offsets, offset=offset, limit=limit)

    rows: list[dict[str, Any]] = []
    seen = 0
    for row in _iter_artifact_rows(path, kind):
        if seen < offset:
            seen += 1
            continue
        rows.append(row)
        if len(rows) >= limit:
            return rows
    return rows


def _slice_jsonl_rows_with_offsets(
    path: Path,
    offsets: list[int],
    *,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    selected_offsets = offsets[offset : offset + limit]
    if not selected_offsets:
        return []

    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for row_offset in selected_offsets:
            handle.seek(row_offset)
            line = handle.readline()
            if not line.strip():
                continue
            rows.append(json.loads(line.decode("utf-8")))
    return rows


def _resolve_artifact_view(rows: list[dict[str, Any]], artifact_spec: dict[str, Any]) -> dict[str, Any]:
    title_path = str(artifact_spec.get("list_title") or _default_title_path(rows))
    subtitle_path = str(artifact_spec.get("list_subtitle") or _default_subtitle_path(rows, title_path))
    excerpt_path = str(artifact_spec.get("list_excerpt") or _default_excerpt_path(rows, title_path))

    badges = _normalize_badges(artifact_spec.get("badges"))
    if not badges:
        badges = _default_badges(rows)

    filters = _normalize_filters(artifact_spec.get("filters"))
    if not filters:
        filters = _default_filters(rows)

    detail_sections = _normalize_sections(artifact_spec.get("detail_sections"))
    if not detail_sections:
        detail_sections = _default_sections(rows)

    search_fields = _normalize_search_fields(artifact_spec.get("search_fields"))
    if not search_fields:
        search_fields = _default_search_fields(title_path, subtitle_path, excerpt_path, badges, detail_sections)

    return {
        "title_path": title_path,
        "subtitle_path": subtitle_path,
        "excerpt_path": excerpt_path,
        "badges": badges,
        "filters": filters,
        "detail_sections": detail_sections,
        "search_fields": search_fields,
    }


def _normalize_badges(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    badges: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            badges.append({"path": item, "label": _field_label(item), "tone": "muted"})
            continue
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            badges.append(
                {
                    "path": item["path"],
                    "label": str(item.get("label") or _field_label(item["path"])),
                    "tone": str(item.get("tone") or "muted"),
                }
            )
    return badges


def _normalize_filters(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    filters: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            filters.append({"key": item, "path": item, "label": _field_label(item)})
            continue
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            filters.append(
                {
                    "key": str(item.get("key") or item["path"]),
                    "path": item["path"],
                    "label": str(item.get("label") or _field_label(item["path"])),
                }
            )
    return filters


def _normalize_sections(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    sections: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            sections.append({"path": item, "label": _field_label(item), "format": "auto"})
            continue
        if isinstance(item, dict) and isinstance(item.get("path"), str):
            sections.append(
                {
                    "path": item["path"],
                    "label": str(item.get("label") or _field_label(item["path"])),
                    "format": str(item.get("format") or "auto"),
                }
            )
    return sections


def _normalize_search_fields(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [str(item) for item in items if isinstance(item, str)]


def _viewer_item(row: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    title = _single_line(_stringify(_value_at_path(row, view["title_path"])))
    subtitle = _single_line(_stringify(_value_at_path(row, view["subtitle_path"])))
    excerpt = _excerpt_text(_stringify(_value_at_path(row, view["excerpt_path"])))

    badges: list[dict[str, str]] = []
    for badge_spec in view["badges"]:
        for value in _expand_filter_values(_value_at_path(row, badge_spec["path"])):
            badges.append(
                {
                    "label": badge_spec["label"],
                    "value": value,
                    "tone": badge_spec["tone"],
                    "filter_key": badge_spec["path"],
                }
            )

    filter_values: dict[str, list[str]] = {}
    for filter_spec in view["filters"]:
        filter_values[filter_spec["key"]] = _expand_filter_values(_value_at_path(row, filter_spec["path"]))

    sections = []
    for section_spec in view["detail_sections"]:
        value = _value_at_path(row, section_spec["path"])
        if value in (None, "", [], {}):
            continue
        sections.append(
            {
                "key": section_spec["path"],
                "label": section_spec["label"],
                "text": _stringify(value, pretty=True),
                "format": _section_format(section_spec, value),
                "html": _section_html(section_spec, value),
            }
        )

    search_parts = []
    for path in view["search_fields"]:
        search_parts.append(_stringify(_value_at_path(row, path)))
    search_text = " ".join(part for part in search_parts if part).lower()

    return {
        "key": _viewer_item_key(row),
        "title": title or "(row)",
        "subtitle": subtitle,
        "excerpt": excerpt,
        "badges": badges,
        "filters": filter_values,
        "sections": sections,
        "raw_json": json.dumps(row, indent=2, ensure_ascii=False, sort_keys=True),
        "search": search_text,
    }


def _viewer_item_key(row: dict[str, Any]) -> str:
    row_id = row.get("id")
    if isinstance(row_id, (str, int, float)) and str(row_id):
        return str(row_id)
    digest = hashlib.sha1(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"row-{digest[:12]}"


def _item_matches(
    item: dict[str, Any],
    query: str,
    filters: dict[str, str],
) -> bool:
    if query and query not in item["search"]:
        return False

    for key, value in filters.items():
        if not value:
            continue
        if value not in item["filters"].get(key, []):
            return False

    return True


def _filter_payloads(items: list[dict[str, Any]], filter_specs: list[dict[str, str]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for filter_spec in filter_specs:
        values: set[str] = set()
        for item in items:
            for value in item["filters"].get(filter_spec["key"], []):
                if value:
                    values.add(value)
        payloads.append(
            {
                "key": filter_spec["key"],
                "label": filter_spec["label"],
                "options": sorted(values),
            }
        )
    return payloads


def _value_at_path(row: dict[str, Any], path: str) -> Any:
    current: Any = row
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _default_title_path(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "id"
    row = rows[0]
    for path in ("prompt", "title", "id"):
        if _value_at_path(row, path) is not None:
            return path
    return "id"


def _default_subtitle_path(rows: list[dict[str, Any]], title_path: str) -> str:
    if not rows:
        return "id"
    row = rows[0]
    for path in ("id", "meta.family", "meta.question_type"):
        if path != title_path and _value_at_path(row, path) is not None:
            return path
    return title_path


def _default_excerpt_path(rows: list[dict[str, Any]], title_path: str) -> str:
    if not rows:
        return title_path
    row = rows[0]
    for path in ("target", "reasoning", "text"):
        if path != title_path and _value_at_path(row, path) is not None:
            return path
    return title_path


def _default_badges(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not rows:
        return []
    row = rows[0]
    badges: list[dict[str, str]] = []
    for path, tone in (
        ("meta.family", "blue"),
        ("meta.question_type", "slate"),
        ("hidden.generation_filter.reasons", "rose"),
    ):
        if _value_at_path(row, path) is not None:
            badges.append({"path": path, "label": _field_label(path), "tone": tone})
    return badges


def _default_filters(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not rows:
        return []
    row = rows[0]
    filters: list[dict[str, str]] = []
    for path in ("meta.family", "meta.question_type", "hidden.generation_filter.reasons"):
        if _value_at_path(row, path) is not None:
            filters.append({"key": path, "path": path, "label": _field_label(path)})
    return filters


def _default_sections(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not rows:
        return [{"path": "id", "label": "Id", "format": "plain"}]
    row = rows[0]
    preferred = ["prompt", "reasoning", "target", "sources", "scores", "checks", "meta", "hidden"]
    sections = [{"path": path, "label": _field_label(path), "format": "auto"} for path in preferred if _value_at_path(row, path) is not None]
    if not sections:
        sections.append({"path": "id", "label": "Id", "format": "plain"})
    return sections


def _section_format(section_spec: dict[str, str], value: Any) -> str:
    format_name = section_spec.get("format", "auto")
    if format_name != "auto":
        return format_name
    if isinstance(value, (dict, list)):
        return "code"
    return "plain"


def _section_html(section_spec: dict[str, str], value: Any) -> str:
    if _section_format(section_spec, value) != "markdown":
        return ""
    text = _stringify(value, pretty=True)
    return markdown(escape(text), extensions=["extra", "sane_lists", "nl2br"])


def _default_search_fields(
    title_path: str,
    subtitle_path: str,
    excerpt_path: str,
    badges: list[dict[str, str]],
    sections: list[dict[str, str]],
) -> list[str]:
    paths = [title_path, subtitle_path, excerpt_path]
    paths.extend(badge["path"] for badge in badges)
    paths.extend(section["path"] for section in sections[:4])
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _expand_filter_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (_single_line(_stringify(item)) for item in value) if item]
    if isinstance(value, dict):
        return [_single_line(_stringify(value))]
    text = _single_line(_stringify(value))
    return [text] if text else []


def _stringify(value: Any, *, pretty: bool = False) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if pretty:
        return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _excerpt_text(text: str, *, limit: int = 220) -> str:
    text = _single_line(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _compact_summary(value: Any) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_preview") or key in {"kept_preview", "rejected_preview"}:
                continue
            compact[key] = _compact_summary(item)
        return compact

    if isinstance(value, list):
        if len(value) <= 10:
            return [_compact_summary(item) for item in value]
        trimmed = [_compact_summary(item) for item in value[:10]]
        trimmed.append(f"... {len(value) - 10} more")
        return trimmed

    return value


def _field_label(path: str) -> str:
    return path.split(".")[-1].replace("_", " ").title()


def _target_path(run_dir: str, out_path: str | None) -> Path:
    if out_path:
        target = Path(out_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    target = Path(run_dir) / "outputs" / "viewer.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _server_manifest(context: _ViewerContext) -> dict[str, Any]:
    manifest = _run_manifest(Path(context.run.run_dir))
    return {
        "title": context.spec.get("title") or f"{context.run.pack} viewer",
        "run": {
            "run_id": manifest["run_id"],
            "pack": manifest["pack"],
            "status": manifest["status"],
            "run_dir": context.run.run_dir,
        },
        "summary": context.summary,
        "default_artifact": context.default_artifact,
        "artifacts": {
            name: _artifact_summary(context, name, artifact)
            for name, artifact in context.artifacts.items()
        },
    }


def _viewer_progress_payload(context: _ViewerContext) -> dict[str, Any]:
    payload = load_progress(context.run.run_id, include_model_events=False, limit=10)
    payload["artifacts"] = {
        name: _artifact_summary(context, name, artifact)
        for name, artifact in context.artifacts.items()
    }
    return payload


class _ViewerApp:
    def __init__(self, context: _ViewerContext):
        self.context = context

    def handle(self, handler: BaseHTTPRequestHandler) -> None:
        parsed = urlparse(handler.path)
        if parsed.path in {"", "/", "/index.html"}:
            self._send_html(handler, _server_viewer_html(_server_manifest(self.context)))
            return

        if parsed.path == "/api/run":
            self._send_json(handler, _server_manifest(self.context))
            return

        if parsed.path == "/api/progress":
            self._send_json(handler, _viewer_progress_payload(self.context))
            return

        if parsed.path == "/api/artifact":
            params = parse_qs(parsed.query)
            self._handle_artifact(handler, params)
            return

        self._send_text(handler, 404, "Not found")

    def _handle_artifact(self, handler: BaseHTTPRequestHandler, params: dict[str, list[str]]) -> None:
        name = str(params.get("name", [""])[0]).strip()
        if name not in self.context.artifacts:
            self._send_text(handler, 404, "Unknown artifact")
            return

        page = _parse_positive_int(params.get("page", ["1"])[0], default=1)
        page_size = _parse_positive_int(
            params.get("page_size", [str(_artifact_summary(self.context, name, self.context.artifacts[name])["page_size"])])[0],
            default=_artifact_summary(self.context, name, self.context.artifacts[name])["page_size"],
        )
        query = str(params.get("q", [""])[0])
        filters = _parse_filters(params.get("filters", ["{}"])[0])
        payload = _artifact_page_payload(
            self.context,
            name,
            page=page,
            page_size=page_size,
            query=query,
            filters=filters,
        )
        self._send_json(handler, payload)

    def _send_html(self, handler: BaseHTTPRequestHandler, body: str) -> None:
        encoded = body.encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "text/html; charset=utf-8")
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)

    def _send_json(self, handler: BaseHTTPRequestHandler, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)

    def _send_text(self, handler: BaseHTTPRequestHandler, status: int, message: str) -> None:
        encoded = message.encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "text/plain; charset=utf-8")
        handler.send_header("Content-Length", str(len(encoded)))
        handler.end_headers()
        handler.wfile.write(encoded)


def _viewer_handler(app: _ViewerApp):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            app.handle(self)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def _parse_positive_int(text: str, *, default: int) -> int:
    try:
        value = int(text)
    except ValueError:
        return default
    if value <= 0:
        return default
    return value


def _parse_filters(text: str) -> dict[str, str]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    filters: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            continue
        if not isinstance(value, str):
            continue
        if not value:
            continue
        filters[key] = value
    return filters


def _server_viewer_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --line: #dfe3ec;
      --line-strong: #ccd4e2;
      --text: #18202b;
      --muted: #667085;
      --accent: #1769e0;
      --accent-soft: #e8f0ff;
      --chip: #eef2f7;
      --chip-blue: #e6f0ff;
      --chip-rose: #fdecef;
      --chip-amber: #fff3dd;
      --chip-slate: #eef1f6;
      --shadow: 0 18px 48px rgba(24, 32, 43, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(23, 105, 224, 0.08), transparent 26rem),
        linear-gradient(180deg, #f9fafc 0%, var(--bg) 100%);
    }}
    .app {{
      display: grid;
      grid-template-columns: 21rem minmax(0, 1fr);
      gap: 0.85rem;
      min-height: 100vh;
      padding: 0.85rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      overflow: hidden;
      min-width: 0;
    }}
    .sidebar {{
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .head {{
      padding: 0.85rem 0.95rem;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,255,0.92));
    }}
    .title {{
      margin: 0;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.85rem;
      line-height: 1.45;
    }}
    .live-progress {{
      margin-top: 0.35rem;
      font-size: 0.78rem;
      color: var(--muted);
    }}
    .toolbar {{
      display: grid;
      gap: 0.7rem;
      padding: 0.85rem 0.95rem;
      border-bottom: 1px solid var(--line);
      background: #fbfcff;
    }}
    .field-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.7rem;
    }}
    label {{
      display: grid;
      gap: 0.3rem;
      min-width: 0;
    }}
    .label {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    input, select, button {{
      font: inherit;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.58rem 0.68rem;
      background: white;
      color: var(--text);
      min-width: 0;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .filters label {{
      min-width: 9rem;
      flex: 1 1 10rem;
    }}
    .row-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.75rem;
      padding: 0.65rem 0.95rem;
      border-bottom: 1px solid var(--line);
      background: #fbfcff;
    }}
    .list {{
      overflow: auto;
      padding: 0.55rem;
      display: grid;
      gap: 0.45rem;
      min-height: 0;
    }}
    .row-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 0.68rem 0.75rem;
      background: #fff;
      cursor: pointer;
      display: grid;
      gap: 0.35rem;
    }}
    .row-top {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 0.6rem;
    }}
    .row-card:hover {{ border-color: var(--line-strong); }}
    .row-card.active {{
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
      background: var(--accent-soft);
    }}
    .row-title {{ font-size: 0.91rem; font-weight: 600; line-height: 1.38; }}
    .row-subtitle {{ color: var(--muted); font-size: 0.77rem; line-height: 1.25; flex: 0 0 auto; }}
    .row-excerpt {{
      color: #3d4756;
      font-size: 0.82rem;
      line-height: 1.4;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 0.4rem; }}
    .badge {{
      border: 0;
      border-radius: 999px;
      padding: 0.16rem 0.46rem;
      color: #334155;
      background: var(--chip);
      font-size: 0.69rem;
      cursor: pointer;
    }}
    .badge[data-tone="blue"] {{ background: var(--chip-blue); }}
    .badge[data-tone="rose"] {{ background: var(--chip-rose); }}
    .badge[data-tone="amber"] {{ background: var(--chip-amber); }}
    .badge[data-tone="slate"] {{ background: var(--chip-slate); }}
    .detail {{
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    .detail-head {{
      padding: 0.85rem 0.95rem;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 0.5rem;
      background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,250,255,0.92));
    }}
    .detail-title {{ font-size: 1.02rem; font-weight: 700; line-height: 1.3; }}
    .detail-subtitle {{ color: var(--muted); font-size: 0.84rem; }}
    .detail-actions {{
      display: flex;
      justify-content: space-between;
      gap: 0.7rem;
      align-items: center;
      flex-wrap: wrap;
    }}
    .summary-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      padding: 0.7rem 0.95rem 0;
    }}
    .summary-card {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.34rem 0.6rem;
      background: #fbfcff;
    }}
    .summary-label {{
      color: var(--muted);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .summary-value {{ margin-top: 0.08rem; font-size: 0.8rem; font-weight: 600; }}
    .detail-body {{
      padding: 0.8rem 0.95rem 0.95rem;
      display: grid;
      gap: 0.7rem;
      overflow: auto;
    }}
    .section {{
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
    }}
    .section-head {{
      padding: 0.58rem 0.72rem;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 0.7rem;
      align-items: center;
      background: #fbfcff;
    }}
    .section-title {{ font-size: 0.84rem; font-weight: 700; }}
    .section-actions {{
      display: flex;
      gap: 0.35rem;
      align-items: center;
      flex-wrap: wrap;
    }}
    .button {{
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      color: var(--text);
      padding: 0.3rem 0.56rem;
      font-size: 0.76rem;
      cursor: pointer;
    }}
    .button.subtle {{
      background: #fbfcff;
      color: var(--muted);
    }}
    .button.primary {{
      border-color: rgba(23, 105, 224, 0.22);
      background: var(--accent-soft);
      color: var(--accent);
    }}
    pre {{
      margin: 0;
      padding: 0.75rem;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.84rem;
      line-height: 1.52;
    }}
    .section-text {{
      padding: 0.75rem;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font-size: 0.92rem;
      line-height: 1.65;
    }}
    .section-markdown {{
      padding: 0.75rem;
      font-size: 0.92rem;
      line-height: 1.7;
    }}
    .section-markdown > :first-child {{ margin-top: 0; }}
    .section-markdown > :last-child {{ margin-bottom: 0; }}
    .section-markdown p,
    .section-markdown ul,
    .section-markdown ol,
    .section-markdown blockquote {{
      margin: 0.6rem 0;
    }}
    .section-markdown ul,
    .section-markdown ol {{
      padding-left: 1.3rem;
    }}
    .section-markdown code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85em;
      background: #f4f7fb;
      padding: 0.08rem 0.3rem;
      border-radius: 6px;
    }}
    .section-markdown pre {{
      margin: 0.8rem 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fbfcff;
    }}
    .section-body[hidden] {{ display: none; }}
    .section-collapsed .section-head {{
      border-bottom: 0;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }}
    summary {{
      padding: 0.65rem 0.75rem;
      cursor: pointer;
      color: var(--muted);
      font-weight: 600;
    }}
    .empty {{
      color: var(--muted);
      padding: 1rem;
      border: 1px dashed var(--line-strong);
      border-radius: 16px;
      background: rgba(255,255,255,0.75);
    }}
    @media (max-width: 1100px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ max-height: 48vh; }}
      .field-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel sidebar">
      <div class="head">
        <h1 class="title">{title}</h1>
        <div class="meta" id="run-meta"></div>
        <div class="live-progress" id="live-progress"></div>
      </div>
      <div class="toolbar">
        <div class="field-grid">
          <label>
            <span class="label">Artifact</span>
            <select id="artifact-select"></select>
          </label>
          <label>
            <span class="label">Rows per page</span>
            <select id="page-size-select">
              <option value="20">20</option>
              <option value="40">40</option>
              <option value="80">80</option>
              <option value="160">160</option>
            </select>
          </label>
        </div>
        <label>
          <span class="label">Search</span>
          <input id="search-input" type="search" placeholder="Search prompt, target, reasoning, badges">
        </label>
        <div class="filters" id="filters"></div>
      </div>
      <div class="row-meta">
        <div class="meta" id="results-meta"></div>
        <div style="display:flex; gap:0.45rem;">
          <button class="button" id="prev-page">Prev</button>
          <button class="button" id="next-page">Next</button>
        </div>
      </div>
      <div class="list" id="rows"></div>
    </aside>
    <main class="panel detail">
      <div class="detail-head">
        <div class="detail-title" id="detail-title"></div>
        <div class="detail-subtitle" id="detail-subtitle"></div>
        <div class="badges" id="detail-badges"></div>
        <div class="detail-actions">
          <div class="meta" id="artifact-meta"></div>
          <button class="button primary" id="copy-json">Copy Row JSON</button>
        </div>
      </div>
      <div class="summary-strip" id="summary-strip"></div>
      <div class="detail-body">
        <details>
          <summary>Run summary</summary>
          <pre id="summary-json"></pre>
        </details>
        <div id="detail-sections"></div>
      </div>
    </main>
  </div>
  <script>
    const manifest = {data};
    const artifactSelect = document.getElementById("artifact-select");
    const pageSizeSelect = document.getElementById("page-size-select");
    const searchInput = document.getElementById("search-input");
    const filtersEl = document.getElementById("filters");
    const rowsEl = document.getElementById("rows");
    const resultsMeta = document.getElementById("results-meta");
    const prevPage = document.getElementById("prev-page");
    const nextPage = document.getElementById("next-page");
    const runMeta = document.getElementById("run-meta");
    const liveProgress = document.getElementById("live-progress");
    const artifactMeta = document.getElementById("artifact-meta");
    const detailTitle = document.getElementById("detail-title");
    const detailSubtitle = document.getElementById("detail-subtitle");
    const detailBadges = document.getElementById("detail-badges");
    const detailSections = document.getElementById("detail-sections");
    const summaryStrip = document.getElementById("summary-strip");
    const summaryJson = document.getElementById("summary-json");
    const copyJson = document.getElementById("copy-json");

    let currentArtifact = manifest.default_artifact;
    let currentPage = 1;
    let currentIndex = 0;
    let filterState = {{}};
    let currentPayload = null;
    let requestVersion = 0;
    let refreshInFlight = false;
    const pageCache = new Map();
    const sectionState = new Map();

    runMeta.textContent = `${{manifest.run.pack}} · ${{manifest.run.run_id}} · ${{manifest.run.status}}`;
    summaryJson.textContent = JSON.stringify(manifest.summary, null, 2);

    artifactSelect.addEventListener("change", async () => {{
      currentArtifact = artifactSelect.value;
      currentPage = 1;
      currentIndex = 0;
      filterState = {{}};
      pageSizeSelect.value = String(artifactMetaInfo().page_size);
      await render();
    }});

    pageSizeSelect.addEventListener("change", async () => {{
      currentPage = 1;
      currentIndex = 0;
      await render();
    }});

    searchInput.addEventListener("input", async () => {{
      currentPage = 1;
      currentIndex = 0;
      await render();
    }});

    prevPage.addEventListener("click", async () => {{
      currentPage = Math.max(1, currentPage - 1);
      currentIndex = 0;
      await render();
    }});

    nextPage.addEventListener("click", async () => {{
      if (!currentPayload) {{
        return;
      }}
      currentPage = Math.min(currentPayload.total_pages, currentPage + 1);
      currentIndex = 0;
      await render();
    }});

    copyJson.addEventListener("click", async () => {{
      const item = currentItem();
      if (!item) {{
        return;
      }}
      await navigator.clipboard.writeText(item.raw_json);
    }});

    document.addEventListener("keydown", async event => {{
      if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) {{
        return;
      }}
      const items = currentPayload ? currentPayload.items : [];
      if (!items.length) {{
        return;
      }}
      if (event.key === "j" || event.key === "ArrowDown") {{
        currentIndex = Math.min(items.length - 1, currentIndex + 1);
        renderRows();
        renderDetails();
      }}
      if (event.key === "k" || event.key === "ArrowUp") {{
        currentIndex = Math.max(0, currentIndex - 1);
        renderRows();
        renderDetails();
      }}
    }});

    pageSizeSelect.value = String(artifactMetaInfo().page_size);
    updateArtifactOptions();
    updateProgressBanner(null);

    function artifactMetaInfo() {{
      return manifest.artifacts[currentArtifact];
    }}

    function updateArtifactOptions() {{
      const seen = new Set();
      for (const [name, artifact] of Object.entries(manifest.artifacts)) {{
        seen.add(name);
        let option = artifactSelect.querySelector(`option[value="${{cssEscape(name)}}"]`);
        if (!option) {{
          option = document.createElement("option");
          option.value = name;
          artifactSelect.appendChild(option);
        }}
        option.textContent = `${{artifact.label}} (${{artifact.row_count}})`;
        option.selected = name === currentArtifact;
      }}

      for (const option of Array.from(artifactSelect.options)) {{
        if (!seen.has(option.value)) {{
          option.remove();
        }}
      }}
    }}

    function familyProgressPayload(progressPayload) {{
      if (!progressPayload || !progressPayload.snapshots) {{
        return null;
      }}
      for (const [key, value] of Object.entries(progressPayload.snapshots)) {{
        if (key.endsWith("_progress")) {{
          return value;
        }}
      }}
      return null;
    }}

    function updateProgressBanner(progressPayload) {{
      if (!progressPayload) {{
        liveProgress.textContent = manifest.run.status === "running" ? "Refreshing while the run is active." : "";
        return;
      }}

      const family = familyProgressPayload(progressPayload);
      if (!family) {{
        liveProgress.textContent = `${{progressPayload.status}}`;
        return;
      }}

      const parts = [
        `status: ${{progressPayload.status}}`,
        `stage: ${{family.stage}}`,
        `kept: ${{family.rows}}`,
        `rejected: ${{family.rejected_rows}}`,
        `candidates: ${{family.candidate_rows}}`,
      ];
      liveProgress.textContent = parts.join(" · ");
    }}

    async function refreshLiveProgress() {{
      if (refreshInFlight) {{
        return;
      }}
      if (manifest.run.status !== "running") {{
        return;
      }}

      refreshInFlight = true;
      try {{
        const response = await fetch("/api/progress");
        const progressPayload = await response.json();
        manifest.run.status = progressPayload.status;
        manifest.artifacts = progressPayload.artifacts;
        runMeta.textContent = `${{manifest.run.pack}} · ${{manifest.run.run_id}} · ${{progressPayload.status}}`;
        updateProgressBanner(progressPayload);
        updateArtifactOptions();

        const previousPayload = currentPayload;
        const previousItemKey = currentItem() ? currentItem().key : null;
        const previousRowCount = previousPayload ? previousPayload.artifact.row_count : null;
        const nextRowCount = artifactMetaInfo().row_count;
        if (previousRowCount === nextRowCount && progressPayload.status === "running") {{
          renderPageMeta();
          return;
        }}

        pageCache.clear();
        const nextPayload = await fetchPayload();
        applyPayload(nextPayload, previousPayload, previousItemKey);
      }} finally {{
        refreshInFlight = false;
      }}
    }}

    function pageSize() {{
      return Number(pageSizeSelect.value || artifactMetaInfo().page_size || 40);
    }}

    function currentItem() {{
      if (!currentPayload || !currentPayload.items.length) {{
        return null;
      }}
      currentIndex = Math.min(currentPayload.items.length - 1, currentIndex);
      return currentPayload.items[currentIndex];
    }}

    function currentItemKey() {{
      const item = currentItem();
      return item ? item.key : null;
    }}

    function cacheKey() {{
      return JSON.stringify([
        currentArtifact,
        currentPage,
        pageSize(),
        searchInput.value.trim(),
        filterState,
      ]);
    }}

    async function fetchPayload() {{
      const key = cacheKey();
      if (pageCache.has(key)) {{
        return pageCache.get(key);
      }}

      const params = new URLSearchParams();
      params.set("name", currentArtifact);
      params.set("page", String(currentPage));
      params.set("page_size", String(pageSize()));
      params.set("q", searchInput.value.trim());
      params.set("filters", JSON.stringify(filterState));
      const response = await fetch(`/api/artifact?${{params.toString()}}`);
      const payload = await response.json();
      pageCache.set(key, payload);
      return payload;
    }}

    function sameFiltersPayload(left, right) {{
      if (!left || !right) {{
        return false;
      }}
      return JSON.stringify(left) === JSON.stringify(right);
    }}

    function sameItems(left, right) {{
      if (!left || !right) {{
        return false;
      }}
      if (left.page !== right.page || left.total_pages !== right.total_pages || left.filtered_count !== right.filtered_count) {{
        return false;
      }}
      if (left.items.length !== right.items.length) {{
        return false;
      }}
      for (let index = 0; index < left.items.length; index += 1) {{
        const previous = left.items[index];
        const next = right.items[index];
        if (previous.key !== next.key || previous.raw_json !== next.raw_json) {{
          return false;
        }}
      }}
      return true;
    }}

    function applyPayload(nextPayload, previousPayload, previousItemKey) {{
      const rowsChanged = !sameItems(previousPayload, nextPayload);
      const filtersChanged = !sameFiltersPayload(previousPayload ? previousPayload.filters : null, nextPayload.filters);

      currentPayload = nextPayload;
      currentPage = nextPayload.page;

      if (previousItemKey) {{
        const matchedIndex = nextPayload.items.findIndex(item => item.key === previousItemKey);
        if (matchedIndex >= 0) {{
          currentIndex = matchedIndex;
        }} else {{
          currentIndex = Math.min(currentIndex, Math.max(0, nextPayload.items.length - 1));
        }}
      }} else {{
        currentIndex = Math.min(currentIndex, Math.max(0, nextPayload.items.length - 1));
      }}

      if (filtersChanged) {{
        renderFilters(nextPayload.filters);
      }}
      if (rowsChanged) {{
        renderRows();
        renderDetails();
        return;
      }}
      renderPageMeta();
    }}

    function renderFilters(filters) {{
      filtersEl.innerHTML = "";
      for (const filter of filters) {{
        const label = document.createElement("label");
        const title = document.createElement("span");
        title.className = "label";
        title.textContent = filter.label;
        const select = document.createElement("select");
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "All";
        select.appendChild(blank);
        for (const optionValue of filter.options) {{
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = optionValue;
          if (filterState[filter.key] === optionValue) {{
            option.selected = true;
          }}
          select.appendChild(option);
        }}
        select.addEventListener("change", async () => {{
          filterState[filter.key] = select.value;
          currentPage = 1;
          currentIndex = 0;
          await render();
        }});
        label.appendChild(title);
        label.appendChild(select);
        filtersEl.appendChild(label);
      }}
    }}

    function renderRows() {{
      rowsEl.innerHTML = "";
      if (!currentPayload || !currentPayload.items.length) {{
        rowsEl.innerHTML = '<div class="empty">No rows match the current filters.</div>';
        return;
      }}

      currentPayload.items.forEach((item, index) => {{
        const card = document.createElement("div");
        card.className = "row-card" + (index === currentIndex ? " active" : "");
        card.innerHTML = `
          <div class="row-top">
            <div class="row-title">${{escapeHtml(item.title)}}</div>
            <div class="row-subtitle">${{escapeHtml(item.subtitle || "")}}</div>
          </div>
          <div class="badges">
            ${{item.badges.map(badge => `<button class="badge" data-tone="${{badge.tone}}">${{escapeHtml(badge.label)}}: ${{escapeHtml(badge.value)}}</button>`).join("")}}
          </div>
          <div class="row-excerpt">${{escapeHtml(item.excerpt || "")}}</div>
        `;
        card.addEventListener("click", () => {{
          currentIndex = index;
          renderRows();
          renderDetails();
        }});
        rowsEl.appendChild(card);
      }});

      rowsEl.querySelectorAll(".badge").forEach((node, index) => {{
        const item = currentPayload.items[Math.floor(index / 1)];
      }});
    }}

    function defaultSectionCollapsed(section, index) {{
      if (section.label === "Target") {{
        return false;
      }}
      const heavyLabels = new Set(["Messages", "Sources", "Scores", "Checks", "Meta", "Generation Filter", "Hidden"]);
      if (heavyLabels.has(section.label)) {{
        return true;
      }}
      return section.format === "code" && index >= 2;
    }}

    function sectionStateKey(itemKey, sectionKey) {{
      return `${{itemKey}}::${{sectionKey}}`;
    }}

    function appendSectionBody(block, section) {{
      if (section.format === "markdown" && section.html) {{
        const body = document.createElement("div");
        body.className = "section-markdown";
        body.innerHTML = section.html;
        block.appendChild(body);
        return;
      }}

      if (section.format === "plain") {{
        const body = document.createElement("div");
        body.className = "section-text";
        body.textContent = section.text;
        block.appendChild(body);
        return;
      }}

      const pre = document.createElement("pre");
      pre.textContent = section.text;
      block.appendChild(pre);
    }}

    function appendSection(block, itemKey, section, index) {{
      const stateKey = sectionStateKey(itemKey, section.key);
      const defaultCollapsed = defaultSectionCollapsed(section, index);
      const expanded = sectionState.has(stateKey) ? sectionState.get(stateKey) : !defaultCollapsed;
      if (!expanded) {{
        block.classList.add("section-collapsed");
      }}

      const head = document.createElement("div");
      head.className = "section-head";
      const title = document.createElement("div");
      title.className = "section-title";
      title.textContent = section.label;
      const actions = document.createElement("div");
      actions.className = "section-actions";
      const copyButton = document.createElement("button");
      copyButton.className = "button";
      copyButton.textContent = "Copy";
      copyButton.addEventListener("click", async () => {{
        await navigator.clipboard.writeText(section.text);
      }});

      const body = document.createElement("div");
      body.className = "section-body";
      body.hidden = !expanded;
      appendSectionBody(body, section);

      if (defaultCollapsed) {{
        const toggleButton = document.createElement("button");
        toggleButton.className = "button subtle";
        toggleButton.textContent = body.hidden ? "Show" : "Hide";
        toggleButton.addEventListener("click", () => {{
          body.hidden = !body.hidden;
          block.classList.toggle("section-collapsed", body.hidden);
          sectionState.set(stateKey, !body.hidden);
          toggleButton.textContent = body.hidden ? "Show" : "Hide";
        }});
        actions.appendChild(toggleButton);
      }}

      actions.appendChild(copyButton);
      head.appendChild(title);
      head.appendChild(actions);
      block.appendChild(head);
      block.appendChild(body);
    }}

    function renderPageMeta() {{
      const item = currentItem();
      const payload = currentPayload;
      if (!payload) {{
        resultsMeta.textContent = "Loading…";
        artifactMeta.textContent = artifactMetaInfo().label;
        prevPage.disabled = true;
        nextPage.disabled = true;
        summaryStrip.innerHTML = "";
        return;
      }}

      let rangeText = "showing 0";
      if (payload.filtered_count > 0) {{
        const start = (payload.page - 1) * payload.page_size + 1;
        const end = Math.min(payload.filtered_count, start + payload.page_size - 1);
        rangeText = `showing ${{start}}-${{end}}`;
      }}
      resultsMeta.textContent = `${{payload.filtered_count}} matching rows · page ${{payload.page}} of ${{payload.total_pages}} · ${{rangeText}}`;
      artifactMeta.textContent = `${{payload.artifact.label}} · ${{payload.artifact.row_count}} rows`;
      prevPage.disabled = payload.page <= 1;
      nextPage.disabled = payload.page >= payload.total_pages;

      summaryStrip.innerHTML = "";
      for (const [label, value] of [
        ["Artifact", payload.artifact.label],
        ["Matches", String(payload.filtered_count)],
        ["Page", `${{payload.page}}/${{payload.total_pages}}`],
        ["Visible", String(payload.items.length)],
        ["Rows", String(payload.artifact.row_count)],
      ]) {{
        const card = document.createElement("div");
        card.className = "summary-card";
        card.innerHTML = `<div class="summary-label">${{label}}</div><div class="summary-value">${{escapeHtml(value)}}</div>`;
        summaryStrip.appendChild(card);
      }}
    }}

    function renderDetailContent() {{
      const item = currentItem();

      if (!currentPayload) {{
        detailTitle.textContent = "Loading…";
        detailSubtitle.textContent = "";
        detailBadges.innerHTML = "";
        detailSections.innerHTML = "";
        return;
      }}

      detailSections.innerHTML = "";
      detailBadges.innerHTML = "";
      if (!item) {{
        detailTitle.textContent = "No rows";
        detailSubtitle.textContent = "Adjust filters or choose another artifact.";
        return;
      }}

      detailTitle.textContent = item.title;
      detailSubtitle.textContent = item.subtitle || "";
      for (const badge of item.badges) {{
        const chip = document.createElement("button");
        chip.className = "badge";
        chip.dataset.tone = badge.tone;
        chip.textContent = `${{badge.label}}: ${{badge.value}}`;
        chip.addEventListener("click", async () => {{
          filterState[badge.filter_key] = badge.value;
          currentPage = 1;
          currentIndex = 0;
          await render();
        }});
        detailBadges.appendChild(chip);
      }}

      for (const section of item.sections) {{
        const block = document.createElement("section");
        block.className = "section";
        appendSection(block, item.key, section, detailSections.childElementCount);
        detailSections.appendChild(block);
      }}

      const raw = document.createElement("section");
      raw.className = "section section-collapsed";
      appendSection(
        raw,
        item.key,
        {{
          key: "__raw_json__",
          label: "Raw JSON",
          text: item.raw_json,
          format: "code",
          html: "",
        }},
        999,
      );
      detailSections.appendChild(raw);
    }}

    function renderDetails() {{
      renderPageMeta();
      renderDetailContent();
    }}

    async function render() {{
      const version = ++requestVersion;
      const previousPayload = currentPayload;
      const previousItemKey = currentItemKey();
      const payload = await fetchPayload();
      if (version !== requestVersion) {{
        return;
      }}
      applyPayload(payload, previousPayload, previousItemKey);
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function cssEscape(value) {{
      if (window.CSS && typeof window.CSS.escape === "function") {{
        return window.CSS.escape(String(value));
      }}
      return String(value).replaceAll('"', '\\"');
    }}

    render();
    if (manifest.run.status === "running") {{
      setInterval(refreshLiveProgress, 5000);
    }}
  </script>
</body>
</html>
"""


def _viewer_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    title = escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --line: #dfe3ec;
      --line-strong: #ccd4e2;
      --text: #18202b;
      --muted: #667085;
      --accent: #1769e0;
      --accent-soft: #e8f0ff;
      --chip: #eef2f7;
      --chip-blue: #e6f0ff;
      --chip-rose: #fdecef;
      --chip-amber: #fff3dd;
      --chip-slate: #eef1f6;
      --shadow: 0 18px 48px rgba(24, 32, 43, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(23, 105, 224, 0.08), transparent 26rem),
        linear-gradient(180deg, #f9fafc 0%, var(--bg) 100%);
    }}
    .app {{
      display: grid;
      grid-template-columns: 21rem minmax(0, 1fr);
      gap: 0.85rem;
      min-height: 100vh;
      padding: 0.85rem;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      overflow: hidden;
      min-width: 0;
    }}
    .sidebar {{
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .head {{
      padding: 0.85rem 0.95rem;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,255,0.92));
    }}
    .title {{
      margin: 0;
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.01em;
    }}
    .meta {{
      color: var(--muted);
      font-size: 0.85rem;
      line-height: 1.45;
    }}
    .toolbar {{
      display: grid;
      gap: 0.7rem;
      padding: 0.85rem 0.95rem;
      border-bottom: 1px solid var(--line);
      background: #fbfcff;
    }}
    .field-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.7rem;
    }}
    label {{
      display: grid;
      gap: 0.3rem;
      min-width: 0;
    }}
    .label {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    input, select, button {{
      font: inherit;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0.58rem 0.68rem;
      background: white;
      color: var(--text);
      min-width: 0;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .filters label {{
      min-width: 9rem;
      flex: 1 1 10rem;
    }}
    .row-meta {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.75rem;
      padding: 0.65rem 0.95rem;
      border-bottom: 1px solid var(--line);
      background: #fbfcff;
    }}
    .list {{
      overflow: auto;
      padding: 0.55rem;
      display: grid;
      gap: 0.45rem;
      min-height: 0;
    }}
    .row-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 0.68rem 0.75rem;
      background: #fff;
      cursor: pointer;
      display: grid;
      gap: 0.35rem;
    }}
    .row-top {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 0.6rem;
    }}
    .row-card:hover {{
      border-color: var(--line-strong);
    }}
    .row-card.active {{
      border-color: var(--accent);
      box-shadow: inset 0 0 0 1px var(--accent);
      background: var(--accent-soft);
    }}
    .row-title {{
      font-size: 0.91rem;
      font-weight: 600;
      line-height: 1.38;
    }}
    .row-subtitle {{
      color: var(--muted);
      font-size: 0.77rem;
      line-height: 1.25;
      flex: 0 0 auto;
    }}
    .row-excerpt {{
      color: #3d4756;
      font-size: 0.82rem;
      line-height: 1.4;
      display: -webkit-box;
      -webkit-line-clamp: 3;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
    }}
    .badge {{
      border: 0;
      border-radius: 999px;
      padding: 0.16rem 0.46rem;
      color: #334155;
      background: var(--chip);
      font-size: 0.69rem;
      cursor: pointer;
    }}
    .badge[data-tone="blue"] {{ background: var(--chip-blue); }}
    .badge[data-tone="rose"] {{ background: var(--chip-rose); }}
    .badge[data-tone="amber"] {{ background: var(--chip-amber); }}
    .badge[data-tone="slate"] {{ background: var(--chip-slate); }}
    .detail {{
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    .detail-head {{
      padding: 0.85rem 0.95rem;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 0.5rem;
      background: linear-gradient(180deg, rgba(255,255,255,0.97), rgba(248,250,255,0.92));
    }}
    .detail-title {{
      font-size: 1.02rem;
      font-weight: 700;
      line-height: 1.3;
    }}
    .detail-subtitle {{
      color: var(--muted);
      font-size: 0.84rem;
    }}
    .detail-actions {{
      display: flex;
      justify-content: space-between;
      gap: 0.7rem;
      align-items: center;
      flex-wrap: wrap;
    }}
    .summary-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
      padding: 0.7rem 0.95rem 0;
    }}
    .summary-card {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 0.34rem 0.6rem;
      background: #fbfcff;
    }}
    .summary-label {{
      color: var(--muted);
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .summary-value {{
      margin-top: 0.08rem;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .detail-body {{
      padding: 0.8rem 0.95rem 0.95rem;
      display: grid;
      gap: 0.7rem;
      overflow: auto;
    }}
    .section {{
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
    }}
    .section-head {{
      padding: 0.58rem 0.72rem;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 0.7rem;
      align-items: center;
      background: #fbfcff;
    }}
    .section-title {{
      font-size: 0.84rem;
      font-weight: 700;
    }}
    .section-actions {{
      display: flex;
      gap: 0.35rem;
      align-items: center;
      flex-wrap: wrap;
    }}
    .button {{
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
      color: var(--text);
      padding: 0.3rem 0.56rem;
      font-size: 0.76rem;
      cursor: pointer;
    }}
    .button.subtle {{
      background: #fbfcff;
      color: var(--muted);
    }}
    .button.primary {{
      border-color: rgba(23, 105, 224, 0.22);
      background: var(--accent-soft);
      color: var(--accent);
    }}
    pre {{
      margin: 0;
      padding: 0.75rem;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.84rem;
      line-height: 1.52;
    }}
    .section-text {{
      padding: 0.75rem;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
      font-size: 0.92rem;
      line-height: 1.65;
    }}
    .section-markdown {{
      padding: 0.75rem;
      font-size: 0.92rem;
      line-height: 1.7;
    }}
    .section-markdown > :first-child {{ margin-top: 0; }}
    .section-markdown > :last-child {{ margin-bottom: 0; }}
    .section-markdown p,
    .section-markdown ul,
    .section-markdown ol,
    .section-markdown blockquote {{
      margin: 0.6rem 0;
    }}
    .section-markdown ul,
    .section-markdown ol {{
      padding-left: 1.3rem;
    }}
    .section-markdown code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85em;
      background: #f4f7fb;
      padding: 0.08rem 0.3rem;
      border-radius: 6px;
    }}
    .section-markdown pre {{
      margin: 0.8rem 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fbfcff;
    }}
    .section-body[hidden] {{ display: none; }}
    .section-collapsed .section-head {{
      border-bottom: 0;
    }}
    details {{
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }}
    summary {{
      padding: 0.65rem 0.75rem;
      cursor: pointer;
      color: var(--muted);
      font-weight: 600;
    }}
    .empty {{
      color: var(--muted);
      padding: 1rem;
      border: 1px dashed var(--line-strong);
      border-radius: 16px;
      background: rgba(255,255,255,0.75);
    }}
    @media (max-width: 1100px) {{
      .app {{ grid-template-columns: 1fr; }}
      .sidebar {{ max-height: 48vh; }}
      .field-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel sidebar">
      <div class="head">
        <h1 class="title">{title}</h1>
        <div class="meta" id="run-meta"></div>
      </div>
      <div class="toolbar">
        <div class="field-grid">
          <label>
            <span class="label">Artifact</span>
            <select id="artifact-select"></select>
          </label>
          <label>
            <span class="label">Rows per page</span>
            <select id="page-size-select">
              <option value="20">20</option>
              <option value="40">40</option>
              <option value="80">80</option>
              <option value="160">160</option>
            </select>
          </label>
        </div>
        <label>
          <span class="label">Search</span>
          <input id="search-input" type="search" placeholder="Search prompt, target, reasoning, badges">
        </label>
        <div class="filters" id="filters"></div>
      </div>
      <div class="row-meta">
        <div class="meta" id="results-meta"></div>
        <div style="display:flex; gap:0.45rem;">
          <button class="button" id="prev-page">Prev</button>
          <button class="button" id="next-page">Next</button>
        </div>
      </div>
      <div class="list" id="rows"></div>
    </aside>
    <main class="panel detail">
      <div class="detail-head">
        <div class="detail-title" id="detail-title"></div>
        <div class="detail-subtitle" id="detail-subtitle"></div>
        <div class="badges" id="detail-badges"></div>
        <div class="detail-actions">
          <div class="meta" id="artifact-meta"></div>
          <button class="button primary" id="copy-json">Copy Row JSON</button>
        </div>
      </div>
      <div class="summary-strip" id="summary-strip"></div>
      <div class="detail-body">
        <details>
          <summary>Run summary</summary>
          <pre id="summary-json"></pre>
        </details>
        <div id="detail-sections"></div>
      </div>
    </main>
  </div>
  <script>
    const payload = {data};

    const artifactSelect = document.getElementById("artifact-select");
    const pageSizeSelect = document.getElementById("page-size-select");
    const searchInput = document.getElementById("search-input");
    const filtersEl = document.getElementById("filters");
    const rowsEl = document.getElementById("rows");
    const resultsMeta = document.getElementById("results-meta");
    const prevPage = document.getElementById("prev-page");
    const nextPage = document.getElementById("next-page");
    const runMeta = document.getElementById("run-meta");
    const artifactMeta = document.getElementById("artifact-meta");
    const detailTitle = document.getElementById("detail-title");
    const detailSubtitle = document.getElementById("detail-subtitle");
    const detailBadges = document.getElementById("detail-badges");
    const detailSections = document.getElementById("detail-sections");
    const summaryStrip = document.getElementById("summary-strip");
    const summaryJson = document.getElementById("summary-json");
    const copyJson = document.getElementById("copy-json");

    let currentArtifact = payload.default_artifact;
    let currentPage = 1;
    let currentIndex = 0;
    let filterState = {{}};

    runMeta.textContent = `${{payload.run.pack}} · ${{payload.run.run_id}} · ${{payload.run.status}}`;
    summaryJson.textContent = JSON.stringify(payload.summary, null, 2);

    for (const [name, artifact] of Object.entries(payload.artifacts)) {{
      const option = document.createElement("option");
      option.value = name;
      option.textContent = `${{artifact.label}} (${{artifact.row_count}})`;
      if (name === currentArtifact) {{
        option.selected = true;
      }}
      artifactSelect.appendChild(option);
    }}

    artifactSelect.addEventListener("change", () => {{
      currentArtifact = artifactSelect.value;
      currentPage = 1;
      currentIndex = 0;
      filterState = {{}};
      pageSizeSelect.value = String(artifact().page_size);
      render();
    }});

    pageSizeSelect.addEventListener("change", () => {{
      currentPage = 1;
      currentIndex = 0;
      render();
    }});

    searchInput.addEventListener("input", () => {{
      currentPage = 1;
      currentIndex = 0;
      render();
    }});

    prevPage.addEventListener("click", () => {{
      currentPage = Math.max(1, currentPage - 1);
      currentIndex = 0;
      render();
    }});

    nextPage.addEventListener("click", () => {{
      const totalPages = Math.max(1, Math.ceil(filteredItems().length / pageSize()));
      currentPage = Math.min(totalPages, currentPage + 1);
      currentIndex = 0;
      render();
    }});

    copyJson.addEventListener("click", async () => {{
      const item = currentItem();
      if (!item) {{
        return;
      }}
      await navigator.clipboard.writeText(item.raw_json);
    }});

    document.addEventListener("keydown", event => {{
      if (event.target && ["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) {{
        return;
      }}
      const items = pagedItems();
      if (!items.length) {{
        return;
      }}
      if (event.key === "j" || event.key === "ArrowDown") {{
        currentIndex = Math.min(items.length - 1, currentIndex + 1);
        render();
      }}
      if (event.key === "k" || event.key === "ArrowUp") {{
        currentIndex = Math.max(0, currentIndex - 1);
        render();
      }}
    }});

    pageSizeSelect.value = String(artifact().page_size);

    function artifact() {{
      return payload.artifacts[currentArtifact];
    }}

    function pageSize() {{
      return Number(pageSizeSelect.value || artifact().page_size || 40);
    }}

    function filteredItems() {{
      const query = searchInput.value.trim().toLowerCase();
      return artifact().items.filter(item => {{
        if (query && !item.search.includes(query)) {{
          return false;
        }}
        for (const filter of artifact().filters) {{
          const active = filterState[filter.key];
          if (!active) {{
            continue;
          }}
          const values = item.filters[filter.key] || [];
          if (!values.includes(active)) {{
            return false;
          }}
        }}
        return true;
      }});
    }}

    function pagedItems() {{
      const items = filteredItems();
      const size = pageSize();
      const totalPages = Math.max(1, Math.ceil(items.length / size));
      currentPage = Math.min(totalPages, currentPage);
      const start = (currentPage - 1) * size;
      return items.slice(start, start + size);
    }}

    function currentItem() {{
      const items = pagedItems();
      if (!items.length) {{
        return null;
      }}
      currentIndex = Math.min(items.length - 1, currentIndex);
      return items[currentIndex];
    }}

    function renderFilters() {{
      filtersEl.innerHTML = "";
      for (const filter of artifact().filters) {{
        const label = document.createElement("label");
        const title = document.createElement("span");
        title.className = "label";
        title.textContent = filter.label;
        const select = document.createElement("select");
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "All";
        select.appendChild(blank);
        for (const optionValue of filter.options) {{
          const option = document.createElement("option");
          option.value = optionValue;
          option.textContent = optionValue;
          if (filterState[filter.key] === optionValue) {{
            option.selected = true;
          }}
          select.appendChild(option);
        }}
        select.addEventListener("change", () => {{
          filterState[filter.key] = select.value;
          currentPage = 1;
          currentIndex = 0;
          render();
        }});
        label.appendChild(title);
        label.appendChild(select);
        filtersEl.appendChild(label);
      }}
    }}

    function renderRows() {{
      const items = pagedItems();
      rowsEl.innerHTML = "";
      if (!items.length) {{
        rowsEl.innerHTML = '<div class="empty">No rows match the current filters.</div>';
        return;
      }}

      items.forEach((item, index) => {{
        const card = document.createElement("div");
        card.className = "row-card" + (index === currentIndex ? " active" : "");
        card.innerHTML = `
          <div class="row-top">
            <div class="row-title">${{escapeHtml(item.title)}}</div>
            <div class="row-subtitle">${{escapeHtml(item.subtitle || "")}}</div>
          </div>
          <div class="badges">
            ${{item.badges.map(badge => `<button class="badge" data-tone="${{badge.tone}}" data-filter-key="${{badge.filter_key}}" data-filter-value="${{escapeAttribute(badge.value)}}">${{escapeHtml(badge.label)}}: ${{escapeHtml(badge.value)}}</button>`).join("")}}
          </div>
          <div class="row-excerpt">${{escapeHtml(item.excerpt || "")}}</div>
        `;
        card.addEventListener("click", () => {{
          currentIndex = index;
          renderDetails();
          renderRows();
        }});
        rowsEl.appendChild(card);
      }});

      rowsEl.querySelectorAll(".badge").forEach(node => {{
        node.addEventListener("click", event => {{
          event.stopPropagation();
          const key = node.dataset.filterKey;
          const value = node.dataset.filterValue;
          if (!key || !value) {{
            return;
          }}
          filterState[key] = value;
          currentPage = 1;
          currentIndex = 0;
          render();
        }});
      }});
    }}

    function isCollapsedSection(section, index) {{
      if (section.label === "Target") {{
        return false;
      }}
      const heavyLabels = new Set(["Messages", "Sources", "Scores", "Checks", "Meta", "Generation Filter", "Hidden"]);
      if (heavyLabels.has(section.label)) {{
        return true;
      }}
      return section.format === "code" && index >= 2;
    }}

    function appendSectionBody(block, section) {{
      if (section.format === "markdown" && section.html) {{
        const body = document.createElement("div");
        body.className = "section-markdown";
        body.innerHTML = section.html;
        block.appendChild(body);
        return;
      }}

      if (section.format === "plain") {{
        const body = document.createElement("div");
        body.className = "section-text";
        body.textContent = section.text;
        block.appendChild(body);
        return;
      }}

      const pre = document.createElement("pre");
      pre.textContent = section.text;
      block.appendChild(pre);
    }}

    function appendSection(block, section, index) {{
      const collapsed = isCollapsedSection(section, index);
      if (collapsed) {{
        block.classList.add("section-collapsed");
      }}

      const head = document.createElement("div");
      head.className = "section-head";
      const title = document.createElement("div");
      title.className = "section-title";
      title.textContent = section.label;
      const actions = document.createElement("div");
      actions.className = "section-actions";
      const copyButton = document.createElement("button");
      copyButton.className = "button";
      copyButton.textContent = "Copy";
      copyButton.addEventListener("click", async () => {{
        await navigator.clipboard.writeText(section.text);
      }});

      const body = document.createElement("div");
      body.className = "section-body";
      body.hidden = collapsed;
      appendSectionBody(body, section);

      if (collapsed) {{
        const toggleButton = document.createElement("button");
        toggleButton.className = "button subtle";
        toggleButton.textContent = "Show";
        toggleButton.addEventListener("click", () => {{
          body.hidden = !body.hidden;
          block.classList.toggle("section-collapsed", body.hidden);
          toggleButton.textContent = body.hidden ? "Show" : "Hide";
        }});
        actions.appendChild(toggleButton);
      }}

      actions.appendChild(copyButton);
      head.appendChild(title);
      head.appendChild(actions);
      block.appendChild(head);
      block.appendChild(body);
    }}

    function renderDetails() {{
      const item = currentItem();
      const filtered = filteredItems();
      const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize()));
      let rangeText = "showing 0";
      if (filtered.length > 0) {{
        const start = (currentPage - 1) * pageSize() + 1;
        const end = Math.min(filtered.length, start + pageSize() - 1);
        rangeText = `showing ${{start}}-${{end}}`;
      }}
      resultsMeta.textContent = `${{filtered.length}} matching rows · page ${{currentPage}} of ${{totalPages}} · ${{rangeText}}`;
      artifactMeta.textContent = `${{artifact().label}} · ${{artifact().shown_rows}} loaded of ${{artifact().row_count}} rows`;
      prevPage.disabled = currentPage <= 1;
      nextPage.disabled = currentPage >= totalPages;

      summaryStrip.innerHTML = "";
      for (const [label, value] of [
        ["Artifact", artifact().label],
        ["Matches", String(filtered.length)],
        ["Page", `${{currentPage}}/${{totalPages}}`],
        ["Visible", String(pagedItems().length)],
        ["Rows", String(artifact().row_count)],
      ]) {{
        const card = document.createElement("div");
        card.className = "summary-card";
        card.innerHTML = `<div class="summary-label">${{label}}</div><div class="summary-value">${{escapeHtml(value)}}</div>`;
        summaryStrip.appendChild(card);
      }}

      detailSections.innerHTML = "";
      detailBadges.innerHTML = "";
      if (!item) {{
        detailTitle.textContent = "No rows";
        detailSubtitle.textContent = "Adjust filters or choose another artifact.";
        return;
      }}

      detailTitle.textContent = item.title;
      detailSubtitle.textContent = item.subtitle || "";
      for (const badge of item.badges) {{
        const chip = document.createElement("button");
        chip.className = "badge";
        chip.dataset.tone = badge.tone;
        chip.textContent = `${{badge.label}}: ${{badge.value}}`;
        chip.addEventListener("click", () => {{
          filterState[badge.filter_key] = badge.value;
          currentPage = 1;
          currentIndex = 0;
          render();
        }});
        detailBadges.appendChild(chip);
      }}

      for (const section of item.sections) {{
        const block = document.createElement("section");
        block.className = "section";
        appendSection(block, section, detailSections.childElementCount);
        detailSections.appendChild(block);
      }}

      const raw = document.createElement("section");
      raw.className = "section section-collapsed";
      appendSection(
        raw,
        {{
          label: "Raw JSON",
          text: item.raw_json,
          format: "code",
          html: "",
        }},
        999,
      );
      detailSections.appendChild(raw);
    }}

    function render() {{
      renderFilters();
      renderRows();
      renderDetails();
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function escapeAttribute(value) {{
      return escapeHtml(value).replaceAll('"', "&quot;");
    }}

    render();
  </script>
</body>
</html>
"""
