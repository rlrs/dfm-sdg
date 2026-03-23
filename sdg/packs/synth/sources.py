from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sdg.commons import store
from sdg.commons.utils import artifacts_root, iso_timestamp, read_json, write_json
from sdg.packs.synth.languages import LanguageCode, source_language_from_memory_cfg
from sdg.packs.synth.record_loader import as_records
from sdg.packs.synth.types import Record

USER_AGENT = "dfm-sdg/0.1 (synthetic-data-research)"
WIKIPEDIA_LICENSE = "CC BY-SA 4.0"
WIKIPEDIA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
WikipediaExpander = Literal["structured_wikipedia", "wikidata"]


class WikipediaSourceConfig(TypedDict):
    kind: Literal["wikipedia_vital_articles"]
    language: str
    vital_level: int
    max_articles: int | None
    refresh: bool
    batch_size: int
    request_pause: float
    expand_with: frozenset[WikipediaExpander]


class InlineDocsSourceConfig(TypedDict):
    kind: Literal["inline_docs"]
    docs: list[Record]
    default_license: str
    source_language: LanguageCode


class PathDocsSourceConfig(TypedDict):
    kind: Literal["path"]
    path: Path
    default_license: str
    source_language: LanguageCode


MemorySourceConfig = WikipediaSourceConfig | InlineDocsSourceConfig | PathDocsSourceConfig


def load_sources(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source_config = _memory_source_config(cfg)
    match source_config["kind"]:
        case "wikipedia_vital_articles":
            return load_wikipedia_vital_articles(source_config)
        case "inline_docs":
            return [
                _normalize_doc(
                    doc,
                    index,
                    default_source="inline",
                    default_license=source_config["default_license"],
                    default_dataset="inline",
                    default_language=source_config["source_language"],
                )
                for index, doc in enumerate(source_config["docs"])
            ]
        case "path":
            return _load_path_docs(source_config)
        case _:
            raise AssertionError(f"Unsupported memory source kind: {source_config['kind']}")


def load_wikipedia_vital_articles(source_config: WikipediaSourceConfig) -> list[dict[str, Any]]:
    language = source_config["language"]
    level = source_config["vital_level"]
    cache_dir = _cache_dir(language, level)

    title_entries = load_vital_title_entries(
        language=language,
        level=level,
        cache_dir=cache_dir,
        refresh=source_config["refresh"],
        request_pause=source_config["request_pause"],
    )
    if source_config["max_articles"] is not None:
        title_entries = title_entries[: source_config["max_articles"]]

    docs = load_wikipedia_docs(
        language=language,
        level=level,
        title_entries=title_entries,
        cache_dir=cache_dir,
        refresh=source_config["refresh"],
        batch_size=source_config["batch_size"],
        request_pause=source_config["request_pause"],
    )

    if "structured_wikipedia" in source_config["expand_with"]:
        docs = attach_structured_wikipedia(
            docs,
            language=language,
            cache_dir=cache_dir,
            refresh=source_config["refresh"],
            batch_size=source_config["batch_size"],
            request_pause=source_config["request_pause"],
        )

    if "wikidata" in source_config["expand_with"]:
        docs = attach_wikidata(
            docs,
            cache_dir=cache_dir,
            refresh=source_config["refresh"],
            batch_size=min(source_config["batch_size"], 25),
            request_pause=source_config["request_pause"],
        )

    return docs


def attach_structured_wikipedia(
    docs: list[dict[str, Any]],
    *,
    language: str,
    cache_dir: Path,
    refresh: bool,
    batch_size: int,
    request_pause: float,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / "structured_wikipedia.json"
    existing_by_title: dict[str, Any] = {}

    if cache_path.exists() and not refresh:
        existing_by_title = read_json(cache_path)

    missing_titles = sorted({doc["title"] for doc in docs if doc["title"] not in existing_by_title})
    if missing_titles:
        existing_by_title.update(
            fetch_structured_wikipedia(
                language=language,
                titles=missing_titles,
                batch_size=batch_size,
                request_pause=request_pause,
            )
        )
        write_json(existing_by_title, cache_path)

    enriched_docs: list[dict[str, Any]] = []
    for doc in docs:
        meta = dict(doc.get("meta") or {})
        meta["structured_wikipedia"] = existing_by_title.get(doc["title"])
        enriched_docs.append({**doc, "meta": meta})

    return enriched_docs


def load_vital_title_entries(
    *,
    language: str,
    level: int,
    cache_dir: Path,
    refresh: bool,
    request_pause: float,
) -> list[dict[str, str]]:
    cache_path = cache_dir / "titles.json"
    if cache_path.exists() and not refresh:
        return read_json(cache_path)["titles"]

    root_page = f"Wikipedia:Vital articles/Level/{level}"
    titles = discover_vital_title_entries(language, root_page, request_pause=request_pause)
    write_json(
        {
            "language": language,
            "vital_level": level,
            "retrieved_at": iso_timestamp(),
            "titles": titles,
        },
        cache_path,
    )
    return titles


def discover_vital_title_entries(language: str, root_page: str, *, request_pause: float) -> list[dict[str, str]]:
    pages_to_visit = [root_page]
    visited_pages: set[str] = set()
    seen_titles: set[str] = set()
    title_entries: list[dict[str, str]] = []

    while pages_to_visit:
        page = pages_to_visit.pop(0)
        if page in visited_pages:
            continue
        visited_pages.add(page)

        parsed = wikipedia_api_json(
            language,
            {
                "action": "parse",
                "format": "json",
                "formatversion": "2",
                "page": page,
                "prop": "links",
            },
        )["parse"]

        for link in parsed.get("links", []):
            title = link["title"]
            if link.get("ns") == 0 and link.get("exists") and title not in seen_titles:
                seen_titles.add(title)
                title_entries.append({"title": title, "listing_page": page})

            if (
                link.get("ns") == 4
                and link.get("exists")
                and title.startswith(f"{root_page}/")
                and title not in visited_pages
            ):
                pages_to_visit.append(title)

        _maybe_sleep(request_pause)

    return title_entries


def load_wikipedia_docs(
    *,
    language: str,
    level: int,
    title_entries: list[dict[str, str]],
    cache_dir: Path,
    refresh: bool,
    batch_size: int,
    request_pause: float,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / "pages.jsonl"
    existing_by_title: dict[str, dict[str, Any]] = {}

    if cache_path.exists() and not refresh:
        existing_by_title = {
            doc["title"]: doc
            for doc in store.read_jsonl(cache_path)
            if _has_non_empty_text(doc)
        }

    missing_titles = [entry["title"] for entry in title_entries if entry["title"] not in existing_by_title]
    if missing_titles:
        new_docs = fetch_wikipedia_docs(
            language=language,
            level=level,
            titles=missing_titles,
            batch_size=batch_size,
            request_pause=request_pause,
        )
        existing_by_title.update({doc["title"]: doc for doc in new_docs})

    docs: list[dict[str, Any]] = []
    missing_after_fetch: list[str] = []

    for entry in title_entries:
        title = entry["title"]
        doc = existing_by_title.get(title)
        if doc is None:
            missing_after_fetch.append(title)
            continue
        meta = dict(doc.get("meta") or {})
        meta["dataset"] = "wikipedia_vital_articles"
        meta["language"] = language
        meta["vital_level"] = level
        meta["listing_page"] = entry["listing_page"]
        docs.append({**doc, "meta": meta})

    if missing_after_fetch:
        raise ValueError(f"Missing fetched documents for {len(missing_after_fetch)} titles")

    store.write_jsonl(docs, cache_path)
    return docs


def fetch_wikipedia_docs(
    *,
    language: str,
    level: int,
    titles: list[str],
    batch_size: int,
    request_pause: float,
) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    for batch in _batched(titles, batch_size):
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            docs.extend(
                executor.map(
                    lambda title: fetch_wikipedia_doc(language=language, level=level, title=title),
                    batch,
                )
            )
        _maybe_sleep(request_pause)

    return docs


def fetch_wikipedia_doc(*, language: str, level: int, title: str) -> dict[str, Any]:
    response = wikipedia_api_json(
        language,
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|pageprops|info",
            "inprop": "url",
            "titles": title,
            "redirects": "1",
            "explaintext": "1",
            "exsectionformat": "plain",
        },
    )

    pages = [page for page in response["query"]["pages"] if "missing" not in page]
    assert len(pages) == 1, f"Expected exactly one Wikipedia page for {title}"
    page = pages[0]

    extract = page.get("extract")
    assert isinstance(extract, str), f"Wikipedia extract must be a string for {title}"
    extract = extract.strip()
    assert extract, f"Wikipedia extract is empty for {title}"

    pageprops = page.get("pageprops", {})
    url = page.get("canonicalurl") or page.get("fullurl")
    return {
        "id": _slug(page["title"]),
        "title": page["title"],
        "text": extract,
        "source": url,
        "url": url,
        "license": WIKIPEDIA_LICENSE,
        "meta": {
            "dataset": "wikipedia_vital_articles",
            "language": language,
            "vital_level": level,
            "pageid": page.get("pageid"),
            "lastrevid": page.get("lastrevid"),
            "length": page.get("length"),
            "touched": page.get("touched"),
            "retrieved_at": iso_timestamp(),
            "wikidata_id": pageprops.get("wikibase_item"),
            "wikibase_shortdesc": pageprops.get("wikibase-shortdesc"),
            "license_url": WIKIPEDIA_LICENSE_URL,
        },
    }


def attach_wikidata(
    docs: list[dict[str, Any]],
    *,
    cache_dir: Path,
    refresh: bool,
    batch_size: int,
    request_pause: float,
) -> list[dict[str, Any]]:
    cache_path = cache_dir / "wikidata.json"
    existing_entities: dict[str, Any] = {}

    if cache_path.exists() and not refresh:
        existing_entities = read_json(cache_path)

    missing_ids = sorted(
        {
            doc["meta"]["wikidata_id"]
            for doc in docs
            if doc.get("meta", {}).get("wikidata_id") and doc["meta"]["wikidata_id"] not in existing_entities
        }
    )

    if missing_ids:
        existing_entities.update(
            fetch_wikidata_entities(
                entity_ids=missing_ids,
                batch_size=batch_size,
                request_pause=request_pause,
            )
        )
        write_json(existing_entities, cache_path)

    enriched_docs: list[dict[str, Any]] = []
    for doc in docs:
        meta = dict(doc.get("meta") or {})
        wikidata_id = meta.get("wikidata_id")
        if wikidata_id:
            meta["wikidata"] = existing_entities.get(wikidata_id)
        enriched_docs.append({**doc, "meta": meta})

    return enriched_docs


def fetch_structured_wikipedia(
    *,
    language: str,
    titles: list[str],
    batch_size: int,
    request_pause: float,
) -> dict[str, Any]:
    structured_by_title: dict[str, Any] = {}

    for batch in _batched(titles, batch_size):
        batch_structured: dict[str, dict[str, Any]] = {}
        continue_args: dict[str, Any] = {}

        while True:
            response = wikipedia_api_json(
                language,
                {
                    "action": "query",
                    "format": "json",
                    "formatversion": "2",
                    "prop": "categories|links|templates",
                    "titles": "|".join(batch),
                    "redirects": "1",
                    "cllimit": "max",
                    "clshow": "!hidden",
                    "pllimit": "max",
                    "tllimit": "max",
                    **continue_args,
                },
            )

            for page in response["query"]["pages"]:
                if "missing" in page:
                    continue

                entry = batch_structured.setdefault(
                    page["title"],
                    {
                        "categories": [],
                        "outgoing_links": [],
                        "templates": [],
                        "infobox_templates": [],
                    },
                )

                categories = page.get("categories", [])
                category_titles = [row["title"].removeprefix("Category:") for row in categories]
                entry["categories"] = _merge_unique(entry["categories"], category_titles)

                links = page.get("links", [])
                article_links = [
                    row["title"]
                    for row in links
                    if row.get("ns") == 0 and "missing" not in row
                ]
                entry["outgoing_links"] = _merge_unique(entry["outgoing_links"], article_links)

                templates = page.get("templates", [])
                template_titles = [row["title"] for row in templates if "missing" not in row]
                entry["templates"] = _merge_unique(entry["templates"], template_titles)
                infobox_templates = [
                    title
                    for title in template_titles
                    if title.startswith("Template:Infobox")
                ]
                entry["infobox_templates"] = _merge_unique(entry["infobox_templates"], infobox_templates)

            if "continue" not in response:
                break

            continue_args = response["continue"]
            _maybe_sleep(request_pause)

        structured_by_title.update(batch_structured)
        _maybe_sleep(request_pause)

    return structured_by_title


def fetch_wikidata_entities(
    *,
    entity_ids: list[str],
    batch_size: int,
    request_pause: float,
) -> dict[str, Any]:
    entities: dict[str, Any] = {}

    for batch in _batched(entity_ids, batch_size):
        response = wikidata_api_json(
            {
                "action": "wbgetentities",
                "format": "json",
                "ids": "|".join(batch),
                "props": "labels|descriptions|aliases|sitelinks",
                "languages": "en",
            }
        )

        for entity_id in batch:
            raw = response["entities"].get(entity_id)
            if raw is None or "missing" in raw:
                continue

            entities[entity_id] = {
                "id": entity_id,
                "label": _lang_value(raw.get("labels", {})),
                "description": _lang_value(raw.get("descriptions", {})),
                "aliases": [alias["value"] for alias in raw.get("aliases", {}).get("en", [])],
                "sitelinks": {
                    name: value.get("title")
                    for name, value in raw.get("sitelinks", {}).items()
                    if name.endswith("wiki")
                },
            }

        _maybe_sleep(request_pause)

    return entities


def wikipedia_api_json(language: str, params: dict[str, Any]) -> dict[str, Any]:
    base_url = f"https://{language}.wikipedia.org/w/api.php"
    return _api_json(base_url, params)


def wikidata_api_json(params: dict[str, Any]) -> dict[str, Any]:
    return _api_json("https://www.wikidata.org/w/api.php", params)


def _api_json(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({key: value for key, value in params.items() if value is not None})
    request_obj = Request(f"{base_url}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request_obj) as response:
        return json.loads(response.read().decode("utf-8"))


def _cache_dir(language: str, level: int) -> Path:
    target = artifacts_root() / "external" / "wikipedia" / language / f"vital_level_{level}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _memory_source_config(cfg: dict[str, Any]) -> MemorySourceConfig:
    memory_cfg = cfg.get("memory_core")
    assert isinstance(memory_cfg, dict), "memory_core config must be a mapping"

    source = memory_cfg.get("source")
    inline_docs = memory_cfg.get("inline_docs")
    source_path = memory_cfg.get("source_path")
    selected = sum(value is not None for value in (source, inline_docs, source_path))
    assert selected == 1, "memory_core must define exactly one of source, inline_docs, or source_path"

    if source is not None:
        assert source == "wikipedia_vital_articles", f"Unsupported memory_core source: {source}"
        return _wikipedia_source_config(memory_cfg)

    if inline_docs is not None:
        return {
            "kind": "inline_docs",
            "docs": as_records(inline_docs, label="memory_core inline_docs"),
            "default_license": _optional_str(memory_cfg, "default_license", default="unknown"),
            "source_language": source_language_from_memory_cfg(memory_cfg),
        }

    assert source_path is not None, "memory_core source_path is required"
    assert isinstance(source_path, str) and source_path, "memory_core source_path must be a non-empty string"
    path = Path(source_path).expanduser().resolve()
    assert path.exists(), f"memory_core source_path does not exist: {path}"
    return {
        "kind": "path",
        "path": path,
        "default_license": _optional_str(memory_cfg, "default_license", default="unknown"),
        "source_language": source_language_from_memory_cfg(memory_cfg),
    }


def _wikipedia_source_config(memory_cfg: Record) -> WikipediaSourceConfig:
    return {
        "kind": "wikipedia_vital_articles",
        "language": source_language_from_memory_cfg(memory_cfg),
        "vital_level": _positive_int(memory_cfg, "vital_level", default=4),
        "max_articles": _optional_positive_int(memory_cfg, "max_articles"),
        "refresh": _bool_value(memory_cfg, "refresh", default=False),
        "batch_size": _positive_int(memory_cfg, "fetch_batch_size", default=20),
        "request_pause": _non_negative_float(memory_cfg, "request_pause_seconds", default=0.0),
        "expand_with": _expand_with(memory_cfg),
    }


def _expand_with(record: Record) -> frozenset[WikipediaExpander]:
    raw_values = _string_list(record, "expand_with")
    allowed_values = {"structured_wikipedia", "wikidata"}
    unknown_values = sorted(set(raw_values) - allowed_values)
    assert not unknown_values, f"Unsupported memory_core expand_with values: {', '.join(unknown_values)}"
    return frozenset(cast(WikipediaExpander, value) for value in raw_values)


def _load_path_docs(source_config: PathDocsSourceConfig) -> list[dict[str, Any]]:
    path = source_config["path"]
    if path.suffix == ".jsonl":
        docs = store.read_jsonl(path)
        return [
            _normalize_doc(
                doc,
                index,
                default_source=str(path),
                default_license=source_config["default_license"],
                default_dataset="path",
                default_language=source_config["source_language"],
            )
            for index, doc in enumerate(docs)
        ]
    if path.suffix == ".json":
        docs = as_records(read_json(path), label=f"memory_core source_path at {path}")
        return [
            _normalize_doc(
                doc,
                index,
                default_source=str(path),
                default_license=source_config["default_license"],
                default_dataset="path",
                default_language=source_config["source_language"],
            )
            for index, doc in enumerate(docs)
        ]

    return [
        _normalize_doc(
            {
                "id": path.stem,
                "title": path.stem.replace("_", " ").title(),
                "text": path.read_text(),
                "source": str(path),
                "url": str(path),
                "license": source_config["default_license"],
                "meta": {"dataset": "local_file"},
            },
            0,
            default_source=str(path),
            default_license=source_config["default_license"],
            default_dataset="local_file",
            default_language=source_config["source_language"],
        )
    ]


def _normalize_doc(
    doc: Record,
    index: int,
    *,
    default_source: str,
    default_license: str,
    default_dataset: str,
    default_language: LanguageCode,
) -> dict[str, Any]:
    doc_id = _optional_str(doc, "id", default=f"doc-{index:05d}")
    meta = _meta(doc)
    if "dataset" not in meta:
        meta["dataset"] = default_dataset
    if "language" in meta:
        assert meta["language"] == default_language, "document meta.language must match memory_core source language"
    else:
        meta["language"] = default_language

    return {
        "id": doc_id,
        "title": _optional_str(doc, "title", default=doc_id),
        "text": _optional_str(doc, "text", default=""),
        "source": _optional_str(doc, "source", default=default_source),
        "url": _optional_str(doc, "url", default=_optional_str(doc, "source", default=default_source)),
        "license": _optional_str(doc, "license", default=default_license),
        "meta": meta,
    }


def _lang_value(values: dict[str, Any], language: str = "en") -> str | None:
    if language in values:
        return values[language]["value"]
    return None


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _maybe_sleep(request_pause: float) -> None:
    if request_pause > 0:
        time.sleep(request_pause)


def _slug(title: str) -> str:
    return title.lower().replace(" ", "_").replace("/", "_")


def _merge_unique(existing: list[str], incoming: list[str]) -> list[str]:
    merged = list(existing)
    seen = set(existing)
    for value in incoming:
        if value in seen:
            continue
        seen.add(value)
        merged.append(value)
    return merged


def _has_non_empty_text(doc: dict[str, Any]) -> bool:
    text = doc.get("text")
    return isinstance(text, str) and bool(text.strip())


def _optional_str(record: Record, key: str, *, default: str) -> str:
    value = record.get(key)
    if value is None:
        return default
    assert isinstance(value, str) and value, f"{key} must be a non-empty string"
    return value


def _positive_int(record: Record, key: str, *, default: int) -> int:
    value = record.get(key)
    if value is None:
        return default
    assert isinstance(value, int) and value > 0, f"{key} must be a positive integer"
    return value


def _optional_positive_int(record: Record, key: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    assert isinstance(value, int) and value > 0, f"{key} must be a positive integer"
    return value


def _bool_value(record: Record, key: str, *, default: bool) -> bool:
    value = record.get(key)
    if value is None:
        return default
    assert isinstance(value, bool), f"{key} must be a boolean"
    return value


def _non_negative_float(record: Record, key: str, *, default: float) -> float:
    value = record.get(key)
    if value is None:
        return default
    assert isinstance(value, int | float) and value >= 0, f"{key} must be a non-negative number"
    return float(value)


def _string_list(record: Record, key: str) -> list[str]:
    value = record.get(key, [])
    assert isinstance(value, list), f"{key} must be a list"
    return [str(item) for item in value if str(item).strip()]


def _meta(record: Record) -> Record:
    value = record.get("meta", {})
    assert isinstance(value, dict), "meta must be a mapping"
    return dict(value)
