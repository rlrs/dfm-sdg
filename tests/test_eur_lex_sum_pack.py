from __future__ import annotations

from pathlib import Path

from sdg.commons import store
from sdg.commons.run import load, progress, read_events
from sdg.packs.eur_lex_sum.build import (
    _count_documents,
    _iter_documents,
    _load_resume_state,
    build,
    publish,
    summarize,
    verify,
)


class FakeInstructionWriter:
    def chat(self, messages, temperature=0.0):
        del messages
        del temperature
        return "Opsummer kort:\n\n{{DOKUMENT}}"


def test_eur_lex_sum_pack_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        "sdg.packs.eur_lex_sum.build._load_instruction_writer",
        lambda cfg: FakeInstructionWriter(),
    )

    source_path = tmp_path / "summaries.jsonl"
    store.write_jsonl(
        [
            {
                "id": "e1",
                "title": "Lissabontraktaten",
                "document": "A" * 120,
                "summary": (
                    "Lissabontraktaten\n"
                    "RESUMÉ\n"
                    "Lissabontraktaten om ændring af traktaten om Den Europæiske Union trådte i kraft."
                ),
            },
            {
                "id": "e2",
                "title": "Kort",
                "document": "Kort tekst.",
                "summary": "RESUMÉ\nKort resumé.",
            },
            {
                "id": "e3",
                "title": "Primærretten",
                "document": "B" * 140,
                "summary": (
                    "Primærretten i Den Europæiske Union\n"
                    "RESUMÉ AF:\n"
                    "HVAD ER PRIMÆRRETTEN?\n"
                    "Det er det samlede regelsæt, som danner grundlaget for EU."
                ),
            },
        ],
        source_path,
    )

    cfg = {
        "pack": "eur_lex_sum",
        "reuse_completed": True,
        "models": {"instruction_writer": "openai"},
        "source": {
            "path": str(source_path),
            "text_field": "document",
            "summary_field": "summary",
            "title_field": "title",
            "id_field": "id",
        },
        "generation": {
            "min_document_chars": 80,
            "temperature": 0.0,
            "train_fraction": 0.5,
        },
    }

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id

    loaded = load(first.run_id)
    dataset_rows = store.read_jsonl(Path(loaded.artifacts["dataset"].path))

    assert len(dataset_rows) == 2
    assert dataset_rows[0]["prompt"] == f"Opsummer kort:\n\n{'A' * 120}"
    assert dataset_rows[0]["target"].startswith("Lissabontraktaten om ændring")
    assert not dataset_rows[0]["target"].startswith("Lissabontraktaten\nRESUMÉ")
    assert dataset_rows[1]["target"] == "Det er det samlede regelsæt, som danner grundlaget for EU."
    assert dataset_rows[1]["meta"]["document_chars"] == 140

    run_events = read_events(first.run_id, component="run")
    assert [event["event"] for event in run_events] == ["started", "completed"]
    run_progress = progress(first.run_id)
    assert run_progress["status"] == "completed"

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0

    summary = summarize(first.run_id)
    assert summary["rows"] == 2

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()


def test_eur_lex_sum_resume_state_respects_max_documents(tmp_path) -> None:
    source_path = tmp_path / "documents.jsonl"
    store.write_jsonl(
        [
            {"id": "e1", "title": "One", "document": "A" * 100, "summary": "S" * 40},
            {"id": "e2", "title": "Two", "document": "B" * 100, "summary": "T" * 40},
            {"id": "e3", "title": "Three", "document": "C" * 100, "summary": "U" * 40},
            {"id": "e4", "title": "Four", "document": "D" * 100, "summary": "V" * 40},
        ],
        source_path,
    )

    dataset_path = tmp_path / "dataset.jsonl"
    failures_path = tmp_path / "generation_failures.jsonl"
    store.write_jsonl(
        [
            {
                "id": "eur_lex_sum-000000",
                "prompt": "Prompt 1",
                "target": "S" * 40,
                "meta": {"source_id": "e1"},
            }
        ],
        dataset_path,
    )
    store.write_jsonl(
        [
            {
                "id": "eur_lex_sum-failure-000001",
                "source_id": "e2",
                "error_type": "RuntimeError",
                "error_message": "boom",
            }
        ],
        failures_path,
    )

    cfg = {
        "source": {
            "path": str(source_path),
            "text_field": "document",
            "summary_field": "summary",
            "title_field": "title",
            "id_field": "id",
        },
        "generation": {
            "min_document_chars": 80,
            "max_documents": 3,
        },
    }

    resume_state = _load_resume_state(
        dataset_path=dataset_path,
        failures_path=failures_path,
    )

    assert resume_state["completed_rows"] == 1
    assert resume_state["failed_rows"] == 1
    assert resume_state["processed_source_ids"] == {"e1", "e2"}

    stats = _count_documents(cfg, processed_source_ids=resume_state["processed_source_ids"])
    assert stats["resumed_rows"] == 2
    assert stats["pending_rows"] == 1
    assert stats["max_documents"] == 3

    pending_documents = list(
        _iter_documents(cfg, processed_source_ids=resume_state["processed_source_ids"])
    )
    assert [document["source_id"] for document in pending_documents] == ["e3"]
    assert [document["row_index"] for document in pending_documents] == [2]
