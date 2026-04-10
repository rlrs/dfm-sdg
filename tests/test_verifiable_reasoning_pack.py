import importlib
import json
from pathlib import Path
from random import Random

import pytest

from sdg.commons import store
from sdg.commons.run import load
from sdg.commons.utils import read_yaml
from sdg.packs.verifiable_reasoning import (
    countdownequal,
    cryptarithmetic,
    futoshiki,
    jugpuzzle,
    knightsandknaves,
    lineup,
    ordering,
    setsplitting,
    zebra,
)
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


def _response_envelope_marker(row: dict[str, object]) -> str:
    envelope = str(row["meta"]["surface_response_envelope"])
    language = str(row["meta"]["prompt_language"])

    if envelope == "answer_block":
        return "Svar:" if language == "da" else "Answer:"
    if envelope == "solution_block":
        return "Løsning:" if language == "da" else "Solution:"
    if envelope == "final_block":
        return "Endeligt svar:" if language == "da" else "Final answer:"
    if envelope == "json":
        return '"answer_lines"'
    if envelope == "xml":
        return "<response>"
    if envelope == "yaml":
        return "answer_lines:"
    raise AssertionError(f"Unsupported envelope: {envelope}")


@pytest.mark.parametrize(
    ("config_name", "language", "family", "expected_output_format"),
    [
        ("base.yaml", "en", "zebra_logic", "house_table"),
        ("knightsandknaves.yaml", "en", "knightsandknaves_logic", "role_assignment"),
        ("countdownequal.yaml", "en", "countdownequal_logic", "expression_string"),
        ("cryptarithmetic.yaml", "en", "cryptarithmetic_logic", "digit_sequence"),
        ("jugpuzzle.yaml", "en", "jugpuzzle_logic", "action_sequence"),
        ("futoshiki.yaml", "en", "futoshiki_logic", "number_grid"),
        ("hitori.yaml", "en", "hitori_logic", "mask_grid"),
        ("kakurasu.yaml", "en", "kakurasu_logic", "binary_grid"),
        ("lightuppuzzle.yaml", "en", "lightuppuzzle_logic", "annotated_grid"),
        ("numbrix.yaml", "en", "numbrix_logic", "number_grid"),
        ("setsplitting.yaml", "en", "setsplitting_logic", "group_assignment"),
        ("skyscraper.yaml", "en", "skyscraper_logic", "number_grid"),
        ("starbattle.yaml", "en", "starbattle_logic", "mask_grid"),
        ("blocked_star.yaml", "en", "blocked_star_logic", "annotated_grid"),
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
    assert all("surface_response_envelope" in row["meta"] for row in rows)
    assert all("response_envelope" in row["hidden"] for row in rows)
    assert all("Final response format:" in row["prompt"] or "Endeligt svarformat:" in row["prompt"] for row in rows)
    assert all(_response_envelope_marker(row) in row["prompt"] for row in rows)

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


def test_verifiable_reasoning_dataset_streaming_can_resume_from_partial_jsonl(tmp_path) -> None:
    cfg = read_yaml(_config_path("base.yaml"))
    cfg["generation"]["count"] = 4

    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    plan_path, plan = verifiable_reasoning_build_module._load_or_create_plan(cfg, outputs_dir, seed=7)
    assert plan_path.exists()

    dataset_path = outputs_dir / "dataset.jsonl"
    partial_rows = [
        verifiable_reasoning_build_module._generate_row_from_plan(index, plan["rows"][index], 7)
        for index in range(2)
    ]
    with dataset_path.open("a") as handle:
        for row in partial_rows:
            store.append_jsonl_line(handle, row)

    written_path = verifiable_reasoning_build_module._write_dataset(cfg, outputs_dir, 7, plan)
    rows = store.read_jsonl(written_path)

    assert written_path == dataset_path
    assert len(rows) == cfg["generation"]["count"]
    assert rows[:2] == partial_rows
    assert [row["id"] for row in rows] == [f"verifiable-reasoning-{index:05d}" for index in range(4)]


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


def test_knightsandknaves_can_verify_attached_response() -> None:
    row = knightsandknaves.generate_row(
        0,
        Random(17),
        language="en",
        recipe=dict(knightsandknaves.recipe_catalog("en")[0]),
    )
    row["target"] = knightsandknaves.format_target(
        tuple(str(name) for name in row["hidden"]["speakers"]),
        tuple(bool(role) for role in row["hidden"]["solution_roles"]),
        "en",
    )
    row["reasoning"] = "I compare each speaker's statement against every possible role assignment until only one remains."

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["metrics"]["checks"]["response_parseable"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_correct"]["failed"] == 0


def test_knightsandknaves_hard_quintet_can_verify_attached_response() -> None:
    row = knightsandknaves.generate_row(
        0,
        Random(41),
        language="da",
        recipe=dict(knightsandknaves.recipe_catalog("da")[-1]),
    )
    row["target"] = knightsandknaves.format_target(
        tuple(str(name) for name in row["hidden"]["speakers"]),
        tuple(bool(role) for role in row["hidden"]["solution_roles"]),
        "da",
    )
    row["reasoning"] = "Jeg sammenholder udsagnene med alle mulige rollefordelinger, indtil kun én fordeling passer."

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["metrics"]["checks"]["response_parseable"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_correct"]["failed"] == 0


def test_cryptarithmetic_can_verify_attached_response() -> None:
    row = cryptarithmetic.generate_row(
        0,
        Random(29),
        language="en",
        recipe=dict(cryptarithmetic.recipe_catalog("en")[0]),
    )
    row["target"] = cryptarithmetic.format_target(
        tuple(int(digit) for digit in row["hidden"]["solution"])
    )
    row["reasoning"] = "I compare the column arithmetic under the unknown base until only one digit assignment fits all columns."

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["metrics"]["checks"]["response_parseable"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_correct"]["failed"] == 0


def test_countdownequal_can_verify_attached_response() -> None:
    row = countdownequal.generate_row(
        0,
        Random(37),
        language="en",
        recipe=dict(countdownequal.recipe_catalog("en")[0]),
    )
    row["target"] = countdownequal.format_target(str(row["hidden"]["solution_expr"]))
    row["reasoning"] = (
        "I compare short arithmetic combinations first, then keep only the expression "
        "that reaches the target with the fewest operations."
    )

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["metrics"]["checks"]["response_parseable"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_correct"]["failed"] == 0


def test_jugpuzzle_can_verify_attached_response() -> None:
    row = jugpuzzle.generate_row(
        0,
        Random(31),
        language="en",
        recipe=dict(jugpuzzle.recipe_catalog("en")[0]),
    )
    actions = tuple(
        jugpuzzle.JugAction(
            kind=str(action["kind"]),
            source=int(action["source"]),
            target=None if action["target"] is None else int(action["target"]),
        )
        for action in row["hidden"]["solution"]
    )
    row["target"] = jugpuzzle.format_target(actions)
    row["reasoning"] = "I track the jug states and keep only the shortest valid action sequence that reaches the target amount."

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["metrics"]["checks"]["response_parseable"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_correct"]["failed"] == 0


def test_setsplitting_can_verify_attached_response() -> None:
    row = setsplitting.generate_row(
        0,
        Random(43),
        language="en",
        recipe=dict(setsplitting.recipe_catalog("en")[0]),
    )
    row["target"] = setsplitting.format_target(tuple(str(group) for group in row["hidden"]["solution"]))
    row["reasoning"] = (
        "I start with element 1 fixed in A. "
        "Then I place the remaining elements so every listed subset hits both groups. "
        "Only one full split satisfies all constraints."
    )

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["metrics"]["checks"]["response_parseable"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_correct"]["failed"] == 0


def test_verifiable_reasoning_attach_targets_keeps_failed_example(monkeypatch) -> None:
    row = lineup.generate_row(0, Random(5), language="en", recipe=dict(lineup.recipe_catalog("en")[0]))

    class FakeLLM:
        def __init__(self) -> None:
            self.model = "fake-answer-teacher"

        async def achat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
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
    assert attached[0]["sources"][-1] == {"kind": "generation_model", "value": "fake-answer-teacher"}

    verification = verify_rows(attached)
    assert verification["response_checks_applied"] is True
    assert verification["failures"] == verification["rows"]


def test_verifiable_reasoning_attach_targets_keeps_teacher_exception_as_failed_row(monkeypatch) -> None:
    row = lineup.generate_row(0, Random(5), language="en", recipe=dict(lineup.recipe_catalog("en")[0]))

    class FakeLLM:
        def __init__(self) -> None:
            self.model = "fake-answer-teacher"

        async def achat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
            raise TimeoutError("simulated timeout")

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

    assert len(attached) == 1
    assert "target" not in attached[0]
    assert attached[0]["meta"]["target_source"] == "answer_teacher_failed"
    assert attached[0]["meta"]["target_attempts"] == 2
    assert attached[0]["sources"][-1] == {"kind": "generation_model", "value": "fake-answer-teacher"}
    assert attached[0]["hidden"]["generation_error"] == "answer_teacher exhausted retry budget: exception:TimeoutError"


def test_verifiable_reasoning_build_writes_rejections_without_counting_them_as_dataset_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    calls = {"count": 0}

    async def fake_attach_target_async(row, teacher, *, temperature: float, max_attempts: int, max_tokens: int):
        calls["count"] += 1
        updated = dict(row)
        meta = dict(row["meta"])
        hidden = dict(row["hidden"])
        if calls["count"] == 1:
            meta["target_source"] = "answer_teacher_failed"
            meta["target_attempts"] = 1
            hidden["generation_error"] = "answer_teacher exhausted retry budget: incorrect_answer"
            updated["meta"] = meta
            updated["hidden"] = hidden
            return updated

        meta["target_source"] = "answer_teacher"
        meta["target_attempts"] = 1
        updated["meta"] = meta
        updated["hidden"] = hidden
        updated["target"] = "synthetic-target"
        return updated

    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_attach_target_async",
        fake_attach_target_async,
    )
    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_answer_teacher_settings",
        lambda cfg: (object(), 0.0, 1, 1, 32000),
    )
    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_published_generation_model",
        lambda result, cfg: "qwen-235b",
    )
    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_published_generation_model",
        lambda result, cfg: "qwen-235b",
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
            "max_candidate_attempts_per_variant": 4,
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))
    rejections = store.read_jsonl(Path(load(result.run_id).artifacts["rejections"].path))

    assert len(rows) == 2
    assert all(row["meta"]["target_source"] == "answer_teacher" for row in rows)
    assert len(rejections) == 1
    assert rejections[0]["meta"]["target_source"] == "answer_teacher_failed"


def test_verifiable_reasoning_build_raises_when_success_quota_is_unmet(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    async def fake_attach_target_async(row, teacher, *, temperature: float, max_attempts: int, max_tokens: int):
        updated = dict(row)
        meta = dict(row["meta"])
        hidden = dict(row["hidden"])
        meta["target_source"] = "answer_teacher_failed"
        meta["target_attempts"] = 1
        hidden["generation_error"] = "answer_teacher exhausted retry budget: incorrect_answer"
        updated["meta"] = meta
        updated["hidden"] = hidden
        return updated

    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_attach_target_async",
        fake_attach_target_async,
    )
    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_answer_teacher_settings",
        lambda cfg: (object(), 0.0, 1, 1, 32000),
    )
    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_published_generation_model",
        lambda result, cfg: "qwen-235b",
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
            "max_candidate_attempts_per_variant": 2,
        },
    }

    with pytest.raises(AssertionError, match="did not reach success quotas"):
        build(cfg)


def test_verifiable_reasoning_build_repairs_stale_manifest_from_output_marker(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    cfg = read_yaml(_config_path("base.yaml"))
    cfg["generation"]["count"] = 2

    result = build(cfg)
    run_dir = Path(result.run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["finished_at"] = None
    manifest["output_artifacts"] = {}
    manifest["error"] = None
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    repaired = build(cfg)
    repaired_manifest = json.loads(manifest_path.read_text())

    assert repaired.run_id == result.run_id
    assert repaired.status == "completed"
    assert repaired_manifest["finished_at"] is not None
    assert repaired_manifest["output_artifacts"]


def test_verifiable_reasoning_publish_exports_only_passing_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    calls = {"count": 0}

    async def fake_attach_target_async(row, teacher, *, temperature: float, max_attempts: int, max_tokens: int):
        calls["count"] += 1
        updated = dict(row)
        meta = dict(row["meta"])
        hidden = dict(row["hidden"])
        if calls["count"] == 1:
            meta["target_source"] = "answer_teacher_failed"
            meta["target_attempts"] = 1
            hidden["generation_error"] = "answer_teacher exhausted retry budget: incorrect_answer"
            updated["meta"] = meta
            updated["hidden"] = hidden
            return updated

        meta["target_source"] = "answer_teacher"
        meta["target_attempts"] = 1
        updated["meta"] = meta
        updated["hidden"] = hidden
        inner_target = ordering.format_target(tuple(updated["hidden"]["solution"]))
        updated["target"] = verifiable_reasoning_build_module._wrap_target_with_response_envelope(inner_target, updated)
        updated["reasoning"] = (
            "I place the fixed positions first.\n"
            "Then I apply the direct relative clue.\n"
            "That leaves only one consistent ordering."
        )
        return updated

    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_attach_target_async",
        fake_attach_target_async,
    )
    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_answer_teacher_settings",
        lambda cfg: (object(), 0.0, 1, 1, 32000),
    )
    monkeypatch.setattr(
        verifiable_reasoning_build_module,
        "_published_generation_model",
        lambda result, cfg: "qwen-235b",
    )

    cfg = {
        "pack": "verifiable_reasoning",
        "seed": 7,
        "reuse_completed": True,
        "models": {"answer_teacher": "openai"},
        "generation": {
            "family": "lineup_logic",
            "count": 1,
            "languages": ["en"],
            "train_fraction": 1.0,
            "attach_targets": True,
            "max_candidate_attempts_per_variant": 3,
        },
    }

    result = build(cfg)
    published = publish(result.run_id, out_dir=str(tmp_path / "published"))

    train_rows = store.read_parquet(Path(published["out_dir"]) / "train.parquet")
    failure_rows = store.read_parquet(Path(published["out_dir"]) / "failures.parquet")

    assert published["rows"] == 1
    assert published["train_rows"] == 1
    assert published["failure_rows"] == 0
    assert len(train_rows) == 1
    assert train_rows[0]["sources"][-1] == {"kind": "generation_model", "value": "qwen-235b"}
    assert not failure_rows


def test_verifiable_reasoning_attach_targets_passes_default_and_override_max_tokens(monkeypatch) -> None:
    row = lineup.generate_row(0, Random(5), language="en", recipe=dict(lineup.recipe_catalog("en")[0]))
    observed_max_tokens: list[int] = []

    class FakeLLM:
        def __init__(self) -> None:
            self.model = "fake-answer-teacher"

        async def achat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
            observed_max_tokens.append(max_tokens)
            return "I will summarize the puzzle first.\n\nAnswer:\nnot a valid ordering"

    monkeypatch.setattr(
        verifiable_reasoning_build_module.common_model,
        "load_clients",
        lambda refs: {"answer_teacher": FakeLLM()},
    )

    default_cfg = {
        "models": {"answer_teacher": "openai"},
        "generation": {
            "attach_targets": True,
            "answer_temperature": 0.0,
            "max_answer_attempts": 1,
        },
    }
    attach_targets([row], default_cfg)
    assert observed_max_tokens == [32000]

    override_cfg = {
        "models": {"answer_teacher": "openai"},
        "generation": {
            "attach_targets": True,
            "answer_temperature": 0.0,
            "max_answer_attempts": 1,
            "answer_max_tokens": 4096,
        },
    }
    attach_targets([row], override_cfg)
    assert observed_max_tokens == [32000, 4096]


def test_verifiable_reasoning_danish_answer_prompt_supports_localized_answers() -> None:
    row = zebra.generate_row(0, Random(7), language="da", recipe=dict(zebra.recipe_catalog("da")[0]))
    row["meta"]["surface_response_envelope"] = "answer_block"

    messages = verifiable_reasoning_build_module._answer_messages(row)
    system_text = messages[0]["content"]
    user_text = messages[1]["content"]

    assert "Skriv din begrundelse på dansk." in system_text
    assert "Følg det endelige svarformat i opgaven nøjagtigt." in system_text
    assert "Endeligt svarformat:" in user_text
    assert "Svar:" in user_text

    _reasoning, answer, found = verifiable_reasoning_build_module._split_answer_response(
        "Jeg bruger de stærkeste ledetråde først.\n\nSvar:\n1: Anna | kaffe | kat"
    )
    assert found is True
    assert answer == "1: Anna | kaffe | kat"


def test_knightsandknaves_answer_prompt_requires_raw_role_lines() -> None:
    row = knightsandknaves.generate_row(
        0,
        Random(17),
        language="da",
        recipe=dict(knightsandknaves.recipe_catalog("da")[-1]),
    )
    row["meta"]["surface_response_envelope"] = "solution_block"

    messages = verifiable_reasoning_build_module._answer_messages(row)
    system_text = messages[0]["content"]
    user_text = messages[1]["content"]

    assert "der kun stå rå linjer i formatet `Navn: rolle`" in system_text
    assert "Skriv ikke noget efter det endelige svar." in system_text
    assert "Endeligt svarformat:" in user_text
    assert "Løsning:" in user_text


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


@pytest.mark.parametrize(
    ("envelope", "response"),
    [
        (
            "solution_block",
            "Reasoning.\n\nSolution:\n2 4 3 1\n3 2 1 4\n4 1 2 3\n1 3 4 2",
        ),
        (
            "json",
            'Reasoning.\n\n{"answer_lines": ["2 4 3 1", "3 2 1 4", "4 1 2 3", "1 3 4 2"]}',
        ),
        (
            "xml",
            "Reasoning.\n\n<response>\n  <line>2 4 3 1</line>\n  <line>3 2 1 4</line>\n  <line>4 1 2 3</line>\n  <line>1 3 4 2</line>\n</response>",
        ),
        (
            "yaml",
            'Reasoning.\n\nanswer_lines:\n  - "2 4 3 1"\n  - "3 2 1 4"\n  - "4 1 2 3"\n  - "1 3 4 2"',
        ),
    ],
)
def test_verifiable_reasoning_extracts_multiple_response_envelopes(envelope: str, response: str) -> None:
    row = futoshiki.generate_row(0, Random(23), language="en", recipe=dict(futoshiki.recipe_catalog("en")[0]))
    row["meta"]["surface_response_envelope"] = envelope
    row = verifiable_reasoning_build_module._with_response_envelope(row)

    reasoning, answer, found = verifiable_reasoning_build_module._split_answer_response(
        response,
        family_module=futoshiki,
        hidden=row["hidden"],
    )

    assert found is True
    assert reasoning == "Reasoning."
    assert answer == "2 4 3 1\n3 2 1 4\n4 1 2 3\n1 3 4 2"


def test_verifiable_reasoning_verifies_wrapped_targets() -> None:
    row = futoshiki.generate_row(
        0,
        Random(13),
        language="en",
        recipe=dict(futoshiki.recipe_catalog("en")[0]),
    )
    row["meta"]["surface_response_envelope"] = "json"
    row = verifiable_reasoning_build_module._with_response_envelope(row)
    inner_target = futoshiki.format_target(tuple(tuple(r) for r in row["hidden"]["solution_grid"]))
    row["target"] = verifiable_reasoning_build_module._wrap_target_with_response_envelope(inner_target, row)
    row["reasoning"] = "I use the givens first and then resolve the remaining cells from the inequalities."

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["metrics"]["checks"]["response_parseable"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_correct"]["failed"] == 0


def test_verifiable_reasoning_rejects_wrong_response_envelope() -> None:
    row = futoshiki.generate_row(0, Random(23), language="en", recipe=dict(futoshiki.recipe_catalog("en")[0]))
    row["meta"]["surface_response_envelope"] = "solution_block"
    row = verifiable_reasoning_build_module._with_response_envelope(row)

    reasoning, answer, found = verifiable_reasoning_build_module._split_answer_response(
        "Reasoning.\n\nAnswer:\n2 4 3 1\n3 2 1 4\n4 1 2 3\n1 3 4 2",
        family_module=futoshiki,
        hidden=row["hidden"],
    )

    assert found is False
    assert reasoning.startswith("Reasoning.")
    assert "Answer:" in answer


def test_verifiable_reasoning_rejects_raw_trailing_answer_when_envelope_is_required() -> None:
    row = futoshiki.generate_row(0, Random(23), language="en", recipe=dict(futoshiki.recipe_catalog("en")[0]))
    row["meta"]["surface_response_envelope"] = "json"
    row = verifiable_reasoning_build_module._with_response_envelope(row)

    reasoning, answer, found = verifiable_reasoning_build_module._split_answer_response(
        "Reasoning.\n\n2 4 3 1\n3 2 1 4\n4 1 2 3\n1 3 4 2",
        family_module=futoshiki,
        hidden=row["hidden"],
    )

    assert found is False
    assert reasoning.startswith("Reasoning.")
    assert answer.endswith("1 3 4 2")
