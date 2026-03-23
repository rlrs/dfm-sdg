from __future__ import annotations

from pathlib import Path

import pytest

from sdg.packs.synth import sources


def test_load_wikipedia_docs_refetches_cached_empty_extract(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "pages.jsonl").write_text("[]")

    cached_docs = [
        {
            "id": "accordion",
            "title": "Accordion",
            "text": "",
            "source": "https://en.wikipedia.org/wiki/Accordion",
            "url": "https://en.wikipedia.org/wiki/Accordion",
            "license": sources.WIKIPEDIA_LICENSE,
            "meta": {},
        }
    ]
    fetched_docs = [
        {
            "id": "accordion",
            "title": "Accordion",
            "text": "Accordions are bellows-driven free-reed instruments.",
            "source": "https://en.wikipedia.org/wiki/Accordion",
            "url": "https://en.wikipedia.org/wiki/Accordion",
            "license": sources.WIKIPEDIA_LICENSE,
            "meta": {},
        }
    ]
    fetch_calls: list[list[str]] = []
    written_docs: list[dict[str, object]] = []

    monkeypatch.setattr(sources.store, "read_jsonl", lambda path: cached_docs)

    def fake_fetch_wikipedia_docs(**kwargs):
        fetch_calls.append(list(kwargs["titles"]))
        return fetched_docs

    monkeypatch.setattr(sources, "fetch_wikipedia_docs", fake_fetch_wikipedia_docs)
    monkeypatch.setattr(sources.store, "write_jsonl", lambda docs, path: written_docs.extend(docs) or path)

    docs = sources.load_wikipedia_docs(
        language="en",
        level=4,
        title_entries=[{"title": "Accordion", "listing_page": "Wikipedia:Vital articles/Level/4/Arts"}],
        cache_dir=cache_dir,
        refresh=False,
        batch_size=20,
        request_pause=0.0,
    )

    assert fetch_calls == [["Accordion"]]
    assert docs[0]["text"] == fetched_docs[0]["text"]
    assert written_docs[0]["text"] == fetched_docs[0]["text"]


def test_fetch_wikipedia_docs_rejects_empty_extract(monkeypatch) -> None:
    monkeypatch.setattr(
        sources,
        "wikipedia_api_json",
        lambda language, params: {
            "query": {
                "pages": [
                    {
                        "title": "Accordion",
                        "pageid": 1162,
                        "extract": "",
                        "pageprops": {},
                        "canonicalurl": "https://en.wikipedia.org/wiki/Accordion",
                        "fullurl": "https://en.wikipedia.org/wiki/Accordion",
                        "length": 80871,
                        "lastrevid": 1,
                        "touched": "2026-03-19T00:00:00Z",
                    }
                ]
            }
        },
    )

    with pytest.raises(AssertionError, match="Wikipedia extract is empty for Accordion"):
        sources.fetch_wikipedia_docs(
            language="en",
            level=4,
            titles=["Accordion"],
            batch_size=20,
            request_pause=0.0,
        )


def test_fetch_wikipedia_docs_fetches_each_title(monkeypatch) -> None:
    requested_titles: list[str] = []

    def fake_wikipedia_api_json(language, params):
        title = params["titles"]
        requested_titles.append(title)
        return {
            "query": {
                "pages": [
                    {
                        "title": title,
                        "pageid": len(requested_titles),
                        "extract": f"{title} text",
                        "pageprops": {},
                        "canonicalurl": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "fullurl": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                        "length": 100,
                        "lastrevid": 1,
                        "touched": "2026-03-19T00:00:00Z",
                    }
                ]
            }
        }

    monkeypatch.setattr(sources, "wikipedia_api_json", fake_wikipedia_api_json)

    docs = sources.fetch_wikipedia_docs(
        language="en",
        level=4,
        titles=["Accordion", "Academy Awards"],
        batch_size=20,
        request_pause=0.0,
    )

    assert requested_titles == ["Accordion", "Academy Awards"]
    assert [doc["title"] for doc in docs] == ["Accordion", "Academy Awards"]
    assert [doc["text"] for doc in docs] == ["Accordion text", "Academy Awards text"]


def test_load_sources_assigns_source_language_to_path_docs(tmp_path) -> None:
    path = tmp_path / "docs.jsonl"
    path.write_text('{"id":"doc-1","title":"Hej","text":"Hej verden"}\n')

    docs = sources.load_sources(
        {
            "memory_core": {
                "source_path": str(path),
                "source_language": "da",
            }
        }
    )

    assert docs[0]["meta"]["language"] == "da"


def test_wikipedia_source_config_keeps_expand_with_as_data() -> None:
    config = sources._wikipedia_source_config(
        {
            "source_language": "en",
            "expand_with": ["structured_wikipedia", "wikidata"],
        }
    )

    assert config["expand_with"] == frozenset({"structured_wikipedia", "wikidata"})
    assert "with_structured_wikipedia" not in config
    assert "with_wikidata" not in config
