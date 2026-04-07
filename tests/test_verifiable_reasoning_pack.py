from pathlib import Path

import pytest

from sdg.commons import store
from sdg.commons.run import load
from sdg.commons.utils import read_json, read_yaml
from sdg.packs.verifiable_reasoning import lineup, zebra
from sdg.packs.verifiable_reasoning.build import build, publish, summarize, verify


@pytest.mark.parametrize(
    ("config_name", "language", "family", "expected_output_format", "catalog_size"),
    [
        ("base.yaml", "en", "zebra_logic", "house_table", len(zebra.recipe_catalog("en"))),
        ("base_da.yaml", "da", "zebra_logic", "house_table", len(zebra.recipe_catalog("da"))),
        ("lineup.yaml", "en", "lineup_logic", "ordered_names", len(lineup.recipe_catalog("en"))),
        ("lineup_da.yaml", "da", "lineup_logic", "ordered_names", len(lineup.recipe_catalog("da"))),
    ],
)
def test_verifiable_reasoning_pack_end_to_end(
    config_name: str,
    language: str,
    family: str,
    expected_output_format: str,
    catalog_size: int,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "sdg"
        / "packs"
        / "verifiable_reasoning"
        / "configs"
        / config_name
    )
    cfg = read_yaml(cfg_path)

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id

    loaded = load(first.run_id)
    rows = store.read_jsonl(Path(loaded.artifacts["dataset"].path))
    plan = read_json(Path(loaded.artifacts["plan"].path))

    assert loaded.pack == "verifiable_reasoning"
    assert rows
    assert all(row["meta"]["family"] == family for row in rows)
    assert all(row["meta"]["prompt_language"] == language for row in rows)
    assert all(row["meta"]["output_format"] == expected_output_format for row in rows)
    assert all("recipe_id" in row["meta"] for row in rows)
    assert len({row["meta"]["recipe_id"] for row in rows}) == catalog_size
    assert len(plan["rows"]) == cfg["generation"]["count"]

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0
    assert verification["metrics"]["checks"]["answer_correct"]["failed"] == 0
    assert verification["metrics"]["checks"]["clues_resolve_uniquely"]["failed"] == 0
    assert verification["dataset_checks"]["passed"] is True

    summary = summarize(first.run_id)
    assert summary["rows"] == cfg["generation"]["count"]
    assert summary["family_counts"][family] == cfg["generation"]["count"]
    assert summary["language_counts"][language] == cfg["generation"]["count"]
    assert summary["dataset_checks"]["passed"] is True
    assert len(summary["difficulty_counts"]) == 3

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "dataset_checks.json").exists()

    train_rows = store.read_parquet(out_dir / "train.parquet")
    assert train_rows
    assert "hidden" not in train_rows[0]


def test_verifiable_reasoning_pack_supports_mixed_families_and_languages(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "sdg"
        / "packs"
        / "verifiable_reasoning"
        / "configs"
        / "mixed.yaml"
    )
    cfg = read_yaml(cfg_path)

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    families = {row["meta"]["family"] for row in rows}
    languages = {row["meta"]["prompt_language"] for row in rows}

    assert families == {"zebra_logic", "lineup_logic"}
    assert languages == {"en", "da"}

    verification = verify(result.run_id)
    assert verification["dataset_checks"]["passed"] is True

    summary = summarize(result.run_id)
    assert summary["family_counts"]["zebra_logic"] == 12
    assert summary["family_counts"]["lineup_logic"] == 12
    assert summary["language_counts"]["en"] == 12
    assert summary["language_counts"]["da"] == 12
    assert summary["dataset_checks"]["passed"] is True
