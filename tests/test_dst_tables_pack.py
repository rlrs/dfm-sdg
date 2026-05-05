from __future__ import annotations

from pathlib import Path

from sdg.commons import store
from sdg.commons.run import load, progress, read_events
from sdg.packs.dst_tables.build import (
    build,
    publish,
    summarize,
    verify,
)

_FAKE_PERSONAS = [
    "A data journalist at a national newspaper.",
    "A policy analyst at a government ministry.",
    "A university student writing an economics thesis.",
]

_TABLES_MD = """\
| SEKTOR | 2024 | 2025 |
| --- | --- | --- |
| Privat | 2.100.000 | 2.145.000 |
| Offentlig | 800.000 | 800.000 |\
"""


class FakeInstructionWriter:
    def chat(self, messages, temperature=0.0):
        del messages, temperature
        return "Baseret på følgende statistik fra Danmarks Statistik, skriv en kort artikel der beskriver de vigtigste tendenser:\n\n[table]"


def test_dst_tables_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        "sdg.packs.dst_tables.build._load_instruction_writer",
        lambda cfg: FakeInstructionWriter(),
    )
    monkeypatch.setattr(
        "sdg.packs.dst_tables.build._load_personas",
        lambda n: _FAKE_PERSONAS,
    )

    source_path = tmp_path / "articles.jsonl"
    store.write_jsonl(
        [
            {
                "article_id": "a1",
                "title": "Beskæftigelsen stiger i Danmark",
                "url": "https://www.dst.dk/nyt/12345",
                "text": (
                    "1. januar 2025\n\n"
                    "Beskæftigelsen i Danmark er steget markant i 2025 sammenlignet med "
                    "2024. Ifølge de nyeste tal fra Danmarks Statistik er der kommet "
                    "45.000 flere i arbejde, svarende til en stigning på 1,5 procent."
                ),
                "inline_tables_md": _TABLES_MD,
            },
            {
                "article_id": "a2",
                "title": "For kort",
                "url": "https://www.dst.dk/nyt/99999",
                "text": "For kort.",
                "inline_tables_md": _TABLES_MD,
            },
            {
                "article_id": "a3",
                "title": "Ingen tabeller",
                "url": "https://www.dst.dk/nyt/88888",
                "text": "Denne artikel har ingen tabeller og bør blive filtreret fra.",
                "inline_tables_md": "",
            },
        ],
        source_path,
    )

    cfg = {
        "pack": "dst_tables",
        "reuse_completed": True,
        "models": {"instruction_writer": "openai"},
        "source": {
            "path": str(source_path),
            "text_field": "text",
            "title_field": "title",
            "url_field": "url",
            "id_field": "article_id",
            "tables_md_field": "inline_tables_md",
        },
        "generation": {
            "min_article_chars": 50,
            "temperature": 0.0,
            "train_fraction": 0.5,
            "max_articles": None,
        },
    }

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id

    loaded = load(first.run_id)
    assert loaded.pack == "dst_tables"
    assert "dataset" in loaded.artifacts

    dataset_rows = store.read_jsonl(Path(loaded.artifacts["dataset"].path))
    assert len(dataset_rows) == 1

    row = dataset_rows[0]
    assert "2.100.000" in row["prompt"]                  # table values injected
    assert "[table]" not in row["prompt"]                 # placeholder replaced
    assert not row["target"].startswith("1. januar")      # date prefix stripped
    assert row["meta"]["url"] == "https://www.dst.dk/nyt/12345"
    assert row["meta"]["table_chars"] == len(_TABLES_MD)
    assert "style_seed_index" in row["meta"]
    assert row["sources"][0]["row_id"] == "a1"

    run_events = read_events(first.run_id, component="run")
    assert [e["event"] for e in run_events] == ["started", "completed"]
    run_progress = progress(first.run_id)
    assert run_progress["status"] == "completed"

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0

    summary = summarize(first.run_id)
    assert summary["rows"] == 1

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()
