from __future__ import annotations

from pathlib import Path

import httpx

from sdg.commons import store
from sdg.commons.run import load, progress, read_events
from sdg.commons.viewer import render_run_view, start_viewer_server
from sdg.packs.backtranslation_passages_dynaword.build import (
    _iter_chunks,
    _load_source_contexts,
    build,
    publish,
    summarize,
    verify,
)


class FakeInstructionWriter:
    def chat(self, messages, temperature=0.0):
        del messages, temperature
        return "Kan du skrive en kort og klar tekst om emnet?"


def test_backtranslation_passages_dynaword_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        "sdg.packs.backtranslation_passages_dynaword.build._load_instruction_writer",
        lambda cfg: FakeInstructionWriter(),
    )

    source_path = tmp_path / "articles.jsonl"
    store.write_jsonl(
        [
            {
                "id": "t1",
                "title": "Regional opdatering",
                "text": (
                    "Første afsnit med relevant indhold om lokale forhold og udvikling. "
                    "Andet afsnit med flere detaljer om konsekvenser og planlægning.\n\n"
                    "Tredje afsnit med afsluttende perspektiv og konkrete næste skridt."
                ),
            },
            {
                "id": "t2",
                "title": "Kort notits",
                "text": "For kort.",
            },
        ],
        source_path,
    )

    cfg = {
        "pack": "backtranslation_passages_dynaword",
        "reuse_completed": True,
        "models": {"instruction_writer": "openai"},
        "generation": {
            "min_article_chars": 80,
            "temperature": 0.0,
            "train_fraction": 0.5,
            "chunk_mode": "hybrid",
            "full_text_max_chars": 600,
            "max_chars": 1000,
        },
        "sources": [
            {
                "name": "tv2r",
                "source": {
                    "path": str(source_path),
                    "text_field": "text",
                    "title_field": "title",
                    "id_field": "id",
                },
                "generation": {
                    "source_type": "news",
                    "max_articles": 10,
                },
            }
        ],
    }

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id

    loaded = load(first.run_id)
    assert loaded.pack == "backtranslation_passages_dynaword"
    assert "dataset" in loaded.artifacts

    dataset_rows = store.read_jsonl(Path(loaded.artifacts["dataset"].path))
    assert len(dataset_rows) == 1
    assert dataset_rows[0]["prompt"].endswith("?")
    assert dataset_rows[0]["meta"]["source_name"] == "tv2r"
    assert dataset_rows[0]["meta"]["source_type"] == "news"

    run_events = read_events(first.run_id, component="run")
    assert [event["event"] for event in run_events] == ["started", "completed"]
    run_progress = progress(first.run_id)
    assert run_progress["status"] == "completed"

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0

    summary = summarize(first.run_id)
    assert summary["rows"] == 1
    assert summary["metrics"]["checks"]["target_min_chars"]["failed"] == 0

    view = render_run_view(first.run_id, limit=10)
    viewer_path = Path(view["out_path"])
    assert view["default_artifact"] == "dataset"
    assert viewer_path.exists()
    viewer_html = viewer_path.read_text()
    assert "Kan du skrive en kort og klar tekst om emnet?" in viewer_html
    assert "Rows per page" in viewer_html

    running = start_viewer_server(first.run_id, host="127.0.0.1", port=0)
    try:
        with httpx.Client(base_url=running.base_url, timeout=5.0) as client:
            run_payload = client.get("/api/run").json()
            assert run_payload["default_artifact"] == "dataset"
            progress_payload = client.get("/api/progress").json()
            assert progress_payload["status"] == "completed"
            page = client.get(
                "/api/artifact",
                params={"name": "dataset", "page": 1, "page_size": 5},
            ).json()
            assert page["artifact"]["name"] == "dataset"
            assert page["filtered_count"] == 1
            assert len(page["items"]) == 1
            html = client.get("/").text
            assert "Copy Row JSON" in html
    finally:
        running.close()

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()


def test_backtranslation_passages_dynaword_hybrid_chunking() -> None:
    article = {
        "source_id": "row-1",
        "text": "Kort men lang nok tekst til at blive beholdt som et enkelt stykke.",
        "source_entry_key": "tv2r",
    }
    generation = {
        "chunk_mode": "hybrid",
        "full_text_max_chars": 200,
        "max_chars": 500,
    }

    chunks = list(_iter_chunks(article, {}, generation, min_chars=20))

    assert len(chunks) == 1
    assert chunks[0]["text"] == article["text"]
    assert chunks[0]["source_id"].endswith("::p000")


def test_backtranslation_passages_dynaword_source_contexts_cover_selected_sources() -> None:
    source_contexts = _load_source_contexts()
    expected = {
        "danske-taler",
        "ft",
        "nordjyllandnews",
        "tv2r",
        "skat",
        "miljoeportalen",
        "ai-aktindsigt",
    }

    assert expected.issubset(source_contexts.keys())
