import importlib
from pathlib import Path
from random import Random

import pytest

from sdg.commons import store
from sdg.commons.run import load
from sdg.commons.utils import read_yaml
from sdg.packs.verifiable_reasoning import futoshiki, lineup, ordering, zebra
from sdg.packs.verifiable_reasoning.build import (
    attach_targets,
    build,
    publish,
    summarize,
    verify,
    verify_rows,
)

verifiable_reasoning_build_module = importlib.import_module("sdg.packs.verifiable_reasoning.build")


def _config_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "sdg"
        / "packs"
        / "verifiable_reasoning"
        / "configs"
        / name
    )


@pytest.mark.parametrize(
    ("config_name", "language", "family", "expected_output_format"),
    [
        ("base.yaml", "en", "zebra_logic", "house_table"),
        ("futoshiki.yaml", "en", "futoshiki_logic", "number_grid"),
        ("hitori.yaml", "en", "hitori_logic", "mask_grid"),
        ("numbrix.yaml", "en", "numbrix_logic", "number_grid"),
        ("skyscraper.yaml", "en", "skyscraper_logic", "number_grid"),
        ("lineup_da.yaml", "da", "lineup_logic", "ordered_names"),
    ],
)
def test_verifiable_reasoning_pack_end_to_end(
    config_name: str,
    language: str,
    family: str,
    expected_output_format: str,
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    cfg = read_yaml(_config_path(config_name))
    cfg["generation"]["count"] = 4

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id

    loaded = load(first.run_id)
    rows = store.read_jsonl(Path(loaded.artifacts["dataset"].path))

    assert loaded.pack == "verifiable_reasoning"
    assert len(rows) == cfg["generation"]["count"]
    assert all(row["meta"]["family"] == family for row in rows)
    assert all(row["meta"]["prompt_language"] == language for row in rows)
    assert all(row["meta"]["output_format"] == expected_output_format for row in rows)
    assert all("target" not in row for row in rows)

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0
    assert verification["response_checks_applied"] is False
    assert verification["dataset_checks"]["passed"] is True

    summary = summarize(first.run_id)
    assert summary["rows"] == cfg["generation"]["count"]
    assert summary["family_counts"][family] == cfg["generation"]["count"]
    assert summary["language_counts"][language] == cfg["generation"]["count"]
    assert summary["dataset_checks"]["passed"] is True

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()

    train_rows = store.read_parquet(out_dir / "train.parquet")
    assert train_rows
    assert "hidden" not in train_rows[0]
    assert "target" not in train_rows[0]


def test_verifiable_reasoning_pack_supports_mixed_families_and_languages(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    cfg = read_yaml(_config_path("mixed.yaml"))
    cfg["generation"]["count"] = 6

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert len(rows) == cfg["generation"]["count"]
    assert {row["meta"]["family"] for row in rows} == {"zebra_logic", "lineup_logic"}
    assert {row["meta"]["prompt_language"] for row in rows} == {"en", "da"}

    verification = verify(result.run_id)
    assert verification["dataset_checks"]["passed"] is True


def test_verifiable_reasoning_can_verify_attached_responses() -> None:
    zebra_row = zebra.generate_row(
        0,
        Random(7),
        language="en",
        recipe=dict(zebra.recipe_catalog("en")[0]),
    )
    zebra_row["target"] = zebra.format_target(
        zebra_row["hidden"]["solution_rows"],
        tuple(zebra_row["hidden"]["axes"]),
    )
    zebra_row["reasoning"] = "I use the strongest fixed clues first, then fill the remaining houses consistently."

    lineup_row = lineup.generate_row(
        1,
        Random(11),
        language="en",
        recipe=dict(lineup.recipe_catalog("en")[0]),
    )
    lineup_row["target"] = ordering.format_target(tuple(lineup_row["hidden"]["solution"]))
    lineup_row["reasoning"] = "I place the direct positions first and resolve the remaining order from the relative clues."

    futoshiki_row = futoshiki.generate_row(
        2,
        Random(13),
        language="en",
        recipe=dict(futoshiki.recipe_catalog("en")[0]),
    )
    futoshiki_row["target"] = futoshiki.format_target(tuple(tuple(row) for row in futoshiki_row["hidden"]["solution_grid"]))
    futoshiki_row["reasoning"] = "I fill the fixed digits first and use the inequalities to narrow the remaining cells."

    verification = verify_rows([zebra_row, lineup_row, futoshiki_row])

    assert verification["response_checks_applied"] is True
    assert verification["metrics"]["checks"]["response_parseable"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_correct"]["failed"] == 0


def test_verifiable_reasoning_attach_targets_keeps_failed_example(monkeypatch) -> None:
    row = lineup.generate_row(0, Random(5), language="en", recipe=dict(lineup.recipe_catalog("en")[0]))

    class FakeLLM:
        def __init__(self) -> None:
            self.model = "fake-answer-teacher"

        async def achat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
            return "I will summarize the puzzle first.\n\nAnswer:\nnot a valid ordering"

    monkeypatch.setattr(
        verifiable_reasoning_build_module.common_model,
        "load_clients",
        lambda refs: {"answer_teacher": FakeLLM()},
    )

    cfg = {
        "models": {"answer_teacher": "openai"},
        "generation": {
            "attach_targets": True,
            "answer_temperature": 0.0,
            "max_answer_attempts": 2,
        },
    }
    attached = attach_targets([row], cfg)

    assert "target" not in attached[0]
    assert attached[0]["meta"]["target_source"] == "answer_teacher_failed"
    assert attached[0]["meta"]["target_attempts"] == 2

    verification = verify_rows(attached)
    assert verification["response_checks_applied"] is True
    assert verification["failures"] == verification["rows"]


def test_verifiable_reasoning_build_keeps_failed_target_attachment_examples(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    class FakeLLM:
        def __init__(self) -> None:
            self.model = "fake-answer-teacher"

        async def achat(self, messages: list[dict[str, str]], *, temperature: float) -> str:
            return "I will summarize the puzzle.\n\nAnswer:\nnot a valid ordering"

    monkeypatch.setattr(
        verifiable_reasoning_build_module.common_model,
        "load_clients",
        lambda refs: {"answer_teacher": FakeLLM()},
    )

    cfg = {
        "pack": "verifiable_reasoning",
        "seed": 7,
        "reuse_completed": True,
        "models": {"answer_teacher": "openai"},
        "generation": {
            "family": "lineup_logic",
            "count": 2,
            "languages": ["en"],
            "train_fraction": 0.8,
            "attach_targets": True,
            "answer_temperature": 0.0,
            "max_answer_attempts": 1,
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert len(rows) == 2
    assert all("target" not in row for row in rows)
    assert all(row["meta"]["target_source"] == "answer_teacher_failed" for row in rows)


def test_verifiable_reasoning_danish_answer_prompt_supports_localized_answers() -> None:
    row = zebra.generate_row(0, Random(7), language="da", recipe=dict(zebra.recipe_catalog("da")[0]))

    messages = verifiable_reasoning_build_module._answer_messages(row)
    system_text = messages[0]["content"]
    user_text = messages[1]["content"]

    assert "Skriv din begrundelse på dansk." in system_text
    assert "Svar:" in system_text
    assert "Svarformat:" in user_text

    _reasoning, answer, found = verifiable_reasoning_build_module._split_answer_response(
        "Jeg bruger de stærkeste ledetråde først.\n\nSvar:\n1: Anna | kaffe | kat"
    )
    assert found is True
    assert answer == "1: Anna | kaffe | kat"


def test_verifiable_reasoning_accepts_trailing_valid_answer_without_marker() -> None:
    row = futoshiki.generate_row(0, Random(23), language="da", recipe=dict(futoshiki.recipe_catalog("da")[0]))
    response = (
        "Jeg bruger først de faste felter og derefter ulighederne.\n\n"
        "Endeligt gitter\n"
        "2 4 3 1\n"
        "3 2 1 4\n"
        "4 1 2 3\n"
        "1 3 4 2"
    )

    reasoning, answer, found = verifiable_reasoning_build_module._split_answer_response(
        response,
        family_module=futoshiki,
        hidden=row["hidden"],
    )

    assert found is True
    assert reasoning.startswith("Jeg bruger først")
    assert answer == "2 4 3 1\n3 2 1 4\n4 1 2 3\n1 3 4 2"
