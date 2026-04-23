from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from random import Random

from sdg.commons import store
from sdg.commons.run import load
from sdg.commons.utils import read_yaml
from sdg.packs.instruction_following.build import (
    build,
    publish,
    summarize,
    verify,
    verify_rows,
)
from sdg.packs.instruction_following.constraints import render_constraint_lines
from sdg.packs.instruction_following.generator import (
    instruction_surface_keys,
    render_instruction_block,
    render_messages,
)

instruction_following_build_module = importlib.import_module("sdg.packs.instruction_following.build")
instruction_following_constraints_module = importlib.import_module("sdg.packs.instruction_following.constraints")


def _config_path(name: str) -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "sdg"
        / "packs"
        / "instruction_following"
        / "configs"
        / name
    )


class FakeScenarioWriter:
    model = "fake-scenario-writer"

    def chat(self, messages, *, temperature: float) -> str:
        del temperature
        if len(messages) == 2 and (
            "Allowed task_type values:" in messages[1]["content"]
            or "Tilladte task_type værdier:" in messages[1]["content"]
        ):
            content = messages[1]["content"]
            profile = {
                "task_type": "general_generation",
                "semantic_rigidity": "open",
                "requested_item_count": None,
                "requested_sentence_count": None,
                "requested_line_count": None,
                "numeric_task": False,
                "preserve_literal_source_text": False,
                "semantic_keywords": ["emnet", "note", "kort"],
                "safe_response_shapes": ["plain_text", "json_object", "xml_object"],
                "naturalness_confidence": "high",
            }
            if "ét sætning" in content or "one sentence" in content:
                profile.update(
                    {
                        "task_type": "explanation",
                        "semantic_rigidity": "rigid",
                        "requested_sentence_count": 1,
                        "semantic_keywords": ["vejr", "klima"],
                        "safe_response_shapes": ["plain_text"],
                    }
                )
            elif "5 originale idéer" in content or "5 original ideas" in content:
                profile.update(
                    {
                        "task_type": "listing",
                        "semantic_rigidity": "medium",
                        "requested_item_count": 5,
                        "semantic_keywords": ["idéer", "stumme film"],
                        "safe_response_shapes": [
                            "bullet_list",
                            "numbered_list",
                            "plain_text",
                            "separated_responses",
                        ],
                    }
                )
            elif "solsortolie er sundt" in content:
                profile.update(
                    {
                        "task_type": "translation",
                        "semantic_rigidity": "rigid",
                        "preserve_literal_source_text": True,
                        "semantic_keywords": ["solsortolie", "sundt", "latin"],
                        "safe_response_shapes": ["plain_text"],
                        "naturalness_confidence": "low",
                    }
                )
            elif "elbiler og hybridbiler" in content:
                profile.update(
                    {
                        "task_type": "comparison",
                        "semantic_rigidity": "medium",
                        "semantic_keywords": ["elbiler", "hybridbiler", "bilkøbere"],
                        "safe_response_shapes": ["plain_text", "json_object"],
                    }
                )
            elif "løbesko" in content:
                profile.update(
                    {
                        "task_type": "recommendation",
                        "semantic_rigidity": "medium",
                        "semantic_keywords": ["løbesko", "begyndere", "guide"],
                        "safe_response_shapes": ["plain_text", "bullet_list", "numbered_list"],
                    }
                )
            elif "Oversæt denne sætning" in content:
                profile.update(
                    {
                        "task_type": "translation",
                        "semantic_rigidity": "rigid",
                        "preserve_literal_source_text": True,
                        "semantic_keywords": ["sætning", "tysk", "bog"],
                        "safe_response_shapes": ["plain_text"],
                    }
                )
            elif "Klassificér disse dyr" in content:
                profile.update(
                    {
                        "task_type": "classification",
                        "semantic_rigidity": "rigid",
                        "semantic_keywords": ["dyr", "planteædere", "rovdyr"],
                        "safe_response_shapes": ["plain_text", "json_object", "xml_object"],
                    }
                )
            elif "Forklar kort hvad fotosyntese er" in content:
                profile.update(
                    {
                        "task_type": "explanation",
                        "semantic_rigidity": "medium",
                        "semantic_keywords": ["fotosyntese"],
                        "safe_response_shapes": ["plain_text", "json_object"],
                    }
                )
            elif "Sammenlign fordele og ulemper" in content:
                profile.update(
                    {
                        "task_type": "comparison",
                        "semantic_rigidity": "medium",
                        "semantic_keywords": ["fordele", "ulemper", "fjernarbejde", "kontorarbejde"],
                        "safe_response_shapes": ["plain_text", "json_object"],
                    }
                )
            elif "Oplist fem ideer" in content:
                profile.update(
                    {
                        "task_type": "listing",
                        "semantic_rigidity": "medium",
                        "requested_item_count": 5,
                        "semantic_keywords": ["ideer", "skolehave"],
                        "safe_response_shapes": ["bullet_list", "numbered_list", "plain_text"],
                    }
                )
            return json.dumps(profile, ensure_ascii=False)

        if len(messages) == 2 and (
            "Candidate keywords:" in messages[1]["content"]
            or "Kandidatord:" in messages[1]["content"]
        ):
            keywords = re.findall(r'"([^"]+)"', messages[1]["content"])
            return '{"keywords":[' + ",".join(f'"{keyword}"' for keyword in keywords[:3]) + "]}"

        if len(messages) == 1:
            prompt = messages[0]["content"]
            if "elbiler og hybridbiler" in prompt:
                return "Elbiler kører kun på strøm, mens hybridbiler kombinerer en elmotor med en forbrændingsmotor."
            if "løbesko" in prompt:
                return "Vælg sko med god pasform, passende støtte og komfort til dit løbemønster."
            return "Her er et første svar på forespørgslen."

        prompt = messages[1]["content"]
        if "Interaktionsstil: single_turn" in prompt:
            return '{"title":"Kort note","user_prefix":"Skriv en kort note om emnet."}'
        if "Sprog: Dansk" in prompt:
            return (
                '{"title":"Kort note","base_user":"Skriv et første udkast om emnet.",'
                '"assistant_reply":"Her er et første udkast om emnet.",'
                '"final_user_prefix":"Skriv det om og følg disse præcise krav."}'
            )
        if "Interaction style: single_turn" in prompt:
            return '{"title":"Short note","user_prefix":"Write a short note about the topic."}'
        return (
            '{"title":"Short note","base_user":"Write a first draft about the topic.",'
            '"assistant_reply":"Here is a first draft about the topic.",'
            '"final_user_prefix":"Rewrite it and follow these exact requirements."}'
        )

    async def achat(self, messages, *, temperature: float) -> str:
        return self.chat(messages, temperature=temperature)


class FakeAnswerTeacher:
    model = "fake-answer-teacher"

    def chat(self, messages, *, temperature: float) -> str:
        del messages, temperature
        return "amber closes the note, and that is the full answer."

    async def achat(self, messages, *, temperature: float) -> str:
        return self.chat(messages, temperature=temperature)


class FlakyAnswerTeacher:
    model = "flaky-answer-teacher"

    def chat(self, messages, *, temperature: float) -> str:
        del temperature
        prompt = messages[0]["content"]
        if "festival" in prompt:
            raise TimeoutError("teacher timed out")
        return "amber closes the note, and that is the full answer."

    async def achat(self, messages, *, temperature: float) -> str:
        return self.chat(messages, temperature=temperature)


class EmptyAnswerTeacher:
    model = "empty-answer-teacher"

    def chat(self, messages, *, temperature: float) -> str:
        del messages, temperature
        return "   "

    async def achat(self, messages, *, temperature: float) -> str:
        return self.chat(messages, temperature=temperature)


def _make_row(*, target: str | None) -> dict[str, object]:
    constraints = [
        {"id": "start_end:first_word", "params": {"word": "amber"}},
        {"id": "start_end:end_phrase", "params": {"phrase": "that is the full answer."}},
        {"id": "format:no_digits", "params": {}},
    ]
    instruction_lines = render_constraint_lines(constraints, language="en")
    instruction_block = render_instruction_block("en", instruction_lines)
    messages = [
        {"role": "user", "content": "Write a short answer about a local event."},
        {"role": "assistant", "content": "Here is a short draft."},
        {"role": "user", "content": f"Rewrite it and follow these exact requirements.\n\n{instruction_block}"},
    ]
    row: dict[str, object] = {
        "id": "instruction-following-00000",
        "prompt": render_messages(messages, language="en"),
        "messages": messages,
        "sources": [],
        "meta": {
            "family": "instruction_following",
            "benchmark": "ifbench",
            "language": "en",
            "interaction_style": "multi_turn_isolation",
            "response_shape": "plain_text",
            "instruction_count": len(constraints),
            "constraint_categories": ["position", "format"],
            "scenario_kind": "announcement",
        },
        "hidden": {
            "constraints": constraints,
            "instruction_lines": instruction_lines,
            "instruction_block": instruction_block,
            "scenario_bundle": {
                "title": "Short note",
                "base_user": "Write a short answer about a local event.",
                "assistant_reply": "Here is a short draft.",
                "final_user_prefix": "Rewrite it and follow these exact requirements.",
            },
            "topic": "a community festival",
        },
    }
    if target is not None:
        row["target"] = target
    return row


def test_instruction_following_pack_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )

    cfg = read_yaml(_config_path("ifbench_mixed.yaml"))
    cfg["generation"]["count"] = 4
    cfg["generation"]["response_shapes"] = ["plain_text"]

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id

    loaded = load(first.run_id)
    rows = store.read_jsonl(Path(loaded.artifacts["dataset"].path))

    assert loaded.pack == "instruction_following"
    assert len(rows) == 4
    assert {row["meta"]["language"] for row in rows} == {"en", "da"}
    assert all(row["meta"]["interaction_style"] == "multi_turn_isolation" for row in rows)
    assert all(len(row["messages"]) == 3 for row in rows)

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0
    assert verification["response_checks_applied"] is False
    assert verification["dataset_checks"]["passed"] is True

    summary = summarize(first.run_id)
    assert summary["rows"] == 4
    assert summary["language_counts"] == {"en": 2, "da": 2}

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()


def test_instruction_following_verify_rows_accepts_valid_target() -> None:
    row = _make_row(target="amber closes the note, and that is the full answer.")

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["failures"] == []
    assert verification["metrics"]["checks"]["response_follows_strict"]["failed"] == 0
    assert verification["metrics"]["checks"]["response_follows_loose"]["failed"] == 0


def test_instruction_following_verify_rows_rejects_invalid_target() -> None:
    row = _make_row(target="the note ends without the required phrase")

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert len(verification["failures"]) == 1
    assert verification["metrics"]["checks"]["response_follows_strict"]["failed"] == 1


def test_instruction_following_build_can_attach_targets(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: FakeAnswerTeacher(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_create_plan",
        lambda cfg, seed: {
            "mode": "planned_rows",
            "rows": [
                    {
                        "language": "en",
                        "interaction_style": "multi_turn_isolation",
                        "response_shape": "plain_text",
                        "scenario_kind": "announcement",
                        "topic": "a community festival",
                    "constraints": [
                        {"id": "start_end:first_word", "params": {"word": "amber"}},
                        {"id": "start_end:end_phrase", "params": {"phrase": "that is the full answer."}},
                    ],
                }
            ],
            "benchmark": "ifbench",
            "language_counts": {"en": 1},
            "interaction_style_counts": {"multi_turn_isolation": 1},
            "response_shape_counts": {"plain_text": 1},
            "constraint_counts": {
                "start_end:first_word": 1,
                "start_end:end_phrase": 1,
            },
        },
    )

    cfg = {
        "pack": "instruction_following",
        "models": {
            "scenario_writer": "openai",
            "answer_teacher": "openai",
        },
        "generation": {
            "count": 1,
            "languages": ["en"],
            "interaction_style": "multi_turn_isolation",
            "attach_targets": True,
            "train_fraction": 0.8,
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert "target" not in rows[0]
    assert rows[0]["messages"][-1] == {
        "role": "assistant",
        "content": "amber closes the note, and that is the full answer.",
    }

    verification = verify(result.run_id)
    assert verification["response_checks_applied"] is True
    assert verification["failed_rows"] == 0


def test_instruction_following_verify_rows_accepts_valid_xml_target() -> None:
    instruction_line = "Return valid XML with root tag <report> and exactly these direct child tags: <status> and <detail>."
    instruction_block = f"Additional requirements:\n- {instruction_line}"
    row = {
        "id": "instruction-following-xml-00000",
        "prompt": f"User:\nReturn XML about a local event.\n\n{instruction_block}",
        "messages": [{"role": "user", "content": f"Return XML about a local event.\n\n{instruction_block}"}],
        "sources": [],
        "meta": {
            "family": "instruction_following",
            "benchmark": "ifbench",
            "language": "en",
            "interaction_style": "single_turn",
            "response_shape": "xml_object",
            "instruction_count": 1,
            "constraint_categories": ["format"],
            "scenario_kind": "announcement",
        },
        "hidden": {
            "constraints": [
                {
                    "id": "format:xml_tags",
                    "params": {"root_tag": "report", "child_tags": ["status", "detail"]},
                }
            ],
            "instruction_lines": [instruction_line],
            "instruction_block": instruction_block,
            "scenario_bundle": {
                "title": "XML note",
                "user_prefix": "Return XML about a local event.",
            },
            "topic": "a community festival",
        },
        "target": "<report><status>open</status><detail>today</detail></report>",
    }

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["failures"] == []


def test_instruction_following_verify_rows_accepts_full_conversation_with_target() -> None:
    row = _make_row(target="amber closes the note, and that is the full answer.")
    row["messages"] = [
        *row["messages"],
        {"role": "assistant", "content": "amber closes the note, and that is the full answer."},
    ]

    verification = verify_rows([row])

    assert verification["response_checks_applied"] is True
    assert verification["failures"] == []


def test_instruction_following_build_uses_seeded_single_turn_prompt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    source_path = tmp_path / "laerebogen_like.jsonl"
    store.write_jsonl(
        [
            {
                "messages": [
                    {"role": "user", "content": "Skriv en praktisk guide til at vælge løbesko til begyndere."},
                    {"role": "assistant", "content": "Det gamle svar skal ikke bruges."},
                ]
            }
        ],
        source_path,
    )

    cfg = {
        "pack": "instruction_following",
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 1,
            "languages": ["da"],
            "interaction_style": "single_turn",
            "response_shapes": ["plain_text"],
            "min_constraints": 1,
            "max_constraints": 1,
            "train_fraction": 0.8,
            "prompt_sources": {
                "da": {
                    "path": str(source_path),
                    "messages_field": "messages",
                    "selection": "first_user",
                    "min_chars": 20,
                    "max_chars": 200,
                }
            },
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert len(rows) == 1
    seed_text = "Skriv en praktisk guide til at vælge løbesko til begyndere."
    assert rows[0]["messages"][0]["content"].startswith(seed_text)
    assert rows[0]["meta"]["prompt_source"] == str(source_path)
    assert rows[0]["hidden"]["prompt_seed"]["text"] == seed_text
    assert not any(source["kind"] == "scenario_model" for source in rows[0]["sources"])


def test_instruction_following_build_uses_first_user_turn_for_seeded_multiturn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    source_path = tmp_path / "laerebogen_like.jsonl"
    store.write_jsonl(
        [
            {
                "messages": [
                    {"role": "user", "content": "Forklar forskellen på elbiler og hybridbiler for nye bilkøbere."},
                    {"role": "assistant", "content": "Det gamle assistentsvar må ikke genbruges."},
                    {"role": "user", "content": "Kan du også give tre eksempler?"},
                ]
            }
        ],
        source_path,
    )

    cfg = {
        "pack": "instruction_following",
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 1,
            "languages": ["da"],
            "interaction_style": "multi_turn_isolation",
            "response_shapes": ["plain_text"],
            "min_constraints": 1,
            "max_constraints": 1,
            "train_fraction": 0.8,
            "prompt_sources": {
                "da": {
                    "path": str(source_path),
                    "messages_field": "messages",
                    "selection": "first_user",
                    "min_chars": 20,
                    "max_chars": 200,
                }
            },
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"] == "Forklar forskellen på elbiler og hybridbiler for nye bilkøbere."
    assert (
        rows[0]["messages"][1]["content"]
        == "Elbiler kører kun på strøm, mens hybridbiler kombinerer en elmotor med en forbrændingsmotor."
    )
    assert rows[0]["messages"][2]["content"].startswith(
        (
            "Skriv svaret om",
            "Svar igen",
            "Skriv en ny version",
            "Lav en revideret version",
        )
    )
    assert rows[0]["hidden"]["instruction_block"] in rows[0]["messages"][2]["content"]
    assert rows[0]["meta"]["prompt_source"] == str(source_path)
    assert any(source["kind"] == "scenario_model" for source in rows[0]["sources"])


def test_instruction_following_weighted_interaction_styles_bias_single_turn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )

    cfg = {
        "pack": "instruction_following",
        "seed": 7,
        "reuse_completed": False,
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 5,
            "languages": ["en"],
            "interaction_styles": ["single_turn", "multi_turn_isolation"],
            "interaction_style_weights": {
                "single_turn": 4,
                "multi_turn_isolation": 1,
            },
            "response_shapes": ["plain_text"],
            "min_constraints": 1,
            "max_constraints": 1,
            "train_fraction": 0.8,
        },
    }

    result = build(cfg)
    summary = summarize(result.run_id)

    assert summary["interaction_style_counts"] == {
        "single_turn": 4,
        "multi_turn_isolation": 1,
    }


def test_instruction_following_varies_instruction_surfaces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )

    cfg = {
        "pack": "instruction_following",
        "seed": 13,
        "reuse_completed": False,
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": len(instruction_surface_keys("da")),
            "languages": ["da"],
            "interaction_style": "single_turn",
            "response_shapes": ["plain_text"],
            "min_constraints": 1,
            "max_constraints": 1,
            "train_fraction": 0.8,
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))
    blocks = [row["hidden"]["instruction_block"] for row in rows if "\n- " in row["hidden"]["instruction_block"]]
    inlines = [row["hidden"]["instruction_block"] for row in rows if "\n- " not in row["hidden"]["instruction_block"]]

    assert blocks
    assert inlines
    assert any(block.splitlines()[0] == "Yderligere krav:" for block in blocks)
    assert any("Svaret skal også" in inline or "Sørg også" in inline for inline in inlines)


def test_instruction_following_prompt_balancing_spreads_task_types(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )

    source_path = tmp_path / "task_balanced_prompts.jsonl"
    store.write_jsonl(
        [
            {"instruction": "Oversæt denne sætning til tysk: 'Jeg læser en bog.'"},
            {"instruction": "Klassificér disse dyr som planteædere eller rovdyr: ko, ulv, rådyr."},
            {"instruction": "Forklar kort hvad fotosyntese er."},
            {"instruction": "Sammenlign fordele og ulemper ved fjernarbejde og kontorarbejde."},
            {"instruction": "Oplist fem ideer til en bæredygtig skolehave."},
        ],
        source_path,
    )

    cfg = {
        "pack": "instruction_following",
        "seed": 11,
        "reuse_completed": False,
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 5,
            "languages": ["da"],
            "interaction_style": "single_turn",
            "response_shapes": ["plain_text"],
            "min_constraints": 1,
            "max_constraints": 1,
            "train_fraction": 0.8,
            "prompt_sources": {
                "da": {
                    "path": str(source_path),
                    "prompt_field": "instruction",
                    "min_chars": 20,
                    "max_chars": 200,
                }
            },
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))
    task_types = {row["meta"]["prompt_task_type"] for row in rows}

    assert task_types == {
        "translation",
        "classification",
        "explanation",
        "comparison",
        "listing",
    }

    summary = summarize(result.run_id)
    assert summary["prompt_task_type_counts"] == {
        "translation": 1,
        "classification": 1,
        "explanation": 1,
        "comparison": 1,
        "listing": 1,
    }


def test_instruction_following_sentence_limited_prompt_avoids_structured_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )

    source_path = tmp_path / "sentence_prompt.jsonl"
    store.write_jsonl(
        [{"instruction": "Forklar forskellen mellem vejr og klima i ét sætning."}],
        source_path,
    )

    cfg = {
        "pack": "instruction_following",
        "seed": 3,
        "reuse_completed": False,
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 1,
            "languages": ["da"],
            "interaction_style": "single_turn",
            "response_shapes": ["json_object"],
            "min_constraints": 1,
            "max_constraints": 1,
            "train_fraction": 0.8,
            "prompt_sources": {
                "da": {
                    "path": str(source_path),
                    "prompt_field": "instruction",
                    "min_chars": 20,
                    "max_chars": 200,
                }
            },
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert rows[0]["meta"]["response_shape"] == "plain_text"
    assert rows[0]["meta"]["prompt_semantic_rigidity"] == "rigid"


def test_instruction_following_counted_list_prompt_avoids_xml_shape(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )

    source_path = tmp_path / "list_prompt.jsonl"
    store.write_jsonl(
        [{"instruction": "Generer en liste med 5 originale idéer til korte stumme film."}],
        source_path,
    )

    cfg = {
        "pack": "instruction_following",
        "seed": 5,
        "reuse_completed": False,
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 1,
            "languages": ["da"],
            "interaction_style": "single_turn",
            "response_shapes": ["xml_object"],
            "min_constraints": 1,
            "max_constraints": 1,
            "train_fraction": 0.8,
            "prompt_sources": {
                "da": {
                    "path": str(source_path),
                    "prompt_field": "instruction",
                    "min_chars": 20,
                    "max_chars": 200,
                }
            },
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert rows[0]["meta"]["response_shape"] in {
        "bullet_list",
        "numbered_list",
        "plain_text",
        "separated_responses",
    }
    assert rows[0]["hidden"]["prompt_seed"]["requested_item_count"] == 5


def test_instruction_following_translation_prompt_profiles_to_plain_text(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )

    source_path = tmp_path / "translation_prompt.jsonl"
    store.write_jsonl(
        [{"instruction": "Oversæt 'solsortolie er sundt' til latin."}],
        source_path,
    )

    cfg = {
        "pack": "instruction_following",
        "seed": 17,
        "reuse_completed": False,
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 1,
            "languages": ["da"],
            "interaction_style": "single_turn",
            "response_shapes": ["bullet_list"],
            "min_constraints": 1,
            "max_constraints": 2,
            "train_fraction": 0.8,
            "prompt_sources": {
                "da": {
                    "path": str(source_path),
                    "prompt_field": "instruction",
                    "min_chars": 20,
                    "max_chars": 200,
                }
            },
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert rows[0]["meta"]["response_shape"] == "plain_text"
    assert rows[0]["hidden"]["prompt_seed"]["preserve_literal_source_text"] is True


def test_instruction_following_rigid_numeric_prompt_filters_high_conflict_constraints() -> None:
    sample_constraints = instruction_following_constraints_module.sample_constraints

    prompt_seed = {
        "semantic_rigidity": "rigid",
        "numeric_task": True,
        "semantic_keywords": ["samantha", "venner", "fødselsdagsfest"],
    }
    seen_ids: set[str] = set()
    for seed in range(40):
        constraints = sample_constraints(
            Random(seed),
            language="da",
            shape="plain_text",
            min_count=2,
            max_count=3,
            prompt_seed=prompt_seed,
        )
        seen_ids.update(str(constraint["id"]) for constraint in constraints)

    assert "format:no_digits" not in seen_ids
    assert "words:word_positions" not in seen_ids
    assert "count:word_count_range" not in seen_ids
    assert "count:keywords_multiple" not in seen_ids
    assert "words:ordered_keywords" not in seen_ids


def test_instruction_following_literal_source_prompt_filters_lexical_constraints() -> None:
    sample_constraints = instruction_following_constraints_module.sample_constraints

    prompt_seed = {
        "semantic_rigidity": "rigid",
        "numeric_task": False,
        "preserve_literal_source_text": True,
        "semantic_keywords": ["solsortolie", "sundt", "latin"],
    }
    seen_ids: set[str] = set()
    for seed in range(40):
        constraints = sample_constraints(
            Random(seed),
            language="da",
            shape="plain_text",
            min_count=2,
            max_count=3,
            prompt_seed=prompt_seed,
        )
        seen_ids.update(str(constraint["id"]) for constraint in constraints)

    assert "count:keywords_multiple" not in seen_ids
    assert "start_end:first_word" not in seen_ids
    assert "start_end:last_word" not in seen_ids
    assert "words:forbidden_words" not in seen_ids
    assert "words:ordered_keywords" not in seen_ids
    assert "words:word_positions" not in seen_ids


def test_instruction_following_uses_model_selected_prompt_keywords(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_create_plan",
        lambda cfg, seed: {
            "mode": "planned_rows",
            "rows": [
                {
                    "language": "da",
                    "interaction_style": "single_turn",
                    "response_shape": "plain_text",
                    "scenario_kind": "source_prompt",
                    "topic": "",
                    "prompt_seed": {
                        "text": "Skriv en kort note om bibliotek, festival og byliv.",
                        "task_type": "general_generation",
                        "length_bucket": "medium",
                        "semantic_rigidity": "open",
                        "requested_item_count": None,
                        "requested_sentence_count": None,
                        "requested_line_count": None,
                        "numeric_task": False,
                        "semantic_keywords": ["bibliotek", "festival", "byliv"],
                        "source_label": "test-source",
                    },
                    "constraints": [
                        {
                            "id": "count:keywords_multiple",
                            "params": {
                                "keywords": [
                                    {"word": "havn", "count": 1},
                                    {"word": "sti", "count": 2},
                                ]
                            },
                        },
                        {
                            "id": "start_end:first_word",
                            "params": {"word": "sky"},
                        },
                    ],
                }
            ],
            "benchmark": "ifbench",
            "language_counts": {"da": 1},
            "interaction_style_counts": {"single_turn": 1},
            "response_shape_counts": {"plain_text": 1},
            "constraint_counts": {
                "count:keywords_multiple": 1,
                "start_end:first_word": 1,
            },
        },
    )

    cfg = {
        "pack": "instruction_following",
        "reuse_completed": False,
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 1,
            "languages": ["da"],
            "interaction_style": "single_turn",
            "response_shapes": ["plain_text"],
            "attach_targets": False,
            "train_fraction": 0.8,
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert rows[0]["hidden"]["selected_prompt_keywords"] == [
        "bibliotek",
        "festival",
        "byliv",
    ]
    keyword_constraint = rows[0]["hidden"]["constraints"][0]
    assert keyword_constraint["params"]["keywords"] == [
        {"word": "bibliotek", "count": 1},
        {"word": "festival", "count": 2},
    ]
    first_word_constraint = rows[0]["hidden"]["constraints"][1]
    assert first_word_constraint["params"]["word"] == "byliv"


def test_instruction_following_build_keeps_other_rows_when_answer_teacher_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: FlakyAnswerTeacher(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_create_plan",
        lambda cfg, seed: {
            "mode": "planned_rows",
            "rows": [
                {
                    "language": "en",
                    "interaction_style": "single_turn",
                    "response_shape": "plain_text",
                    "scenario_kind": "announcement",
                    "topic": "a community festival",
                    "prompt_seed": {
                        "text": "Write a short answer about the community festival.",
                    },
                    "constraints": [
                        {"id": "start_end:first_word", "params": {"word": "amber"}},
                        {"id": "start_end:end_phrase", "params": {"phrase": "that is the full answer."}},
                    ],
                },
                {
                    "language": "en",
                    "interaction_style": "single_turn",
                    "response_shape": "plain_text",
                    "scenario_kind": "announcement",
                    "topic": "a public library",
                    "prompt_seed": {
                        "text": "Write a short answer about the public library.",
                    },
                    "constraints": [
                        {"id": "start_end:first_word", "params": {"word": "amber"}},
                        {"id": "start_end:end_phrase", "params": {"phrase": "that is the full answer."}},
                    ],
                },
            ],
            "benchmark": "ifbench",
            "language_counts": {"en": 2},
            "interaction_style_counts": {"single_turn": 2},
            "response_shape_counts": {"plain_text": 2},
            "constraint_counts": {
                "start_end:first_word": 2,
                "start_end:end_phrase": 2,
            },
        },
    )

    cfg = {
        "pack": "instruction_following",
        "reuse_completed": False,
        "models": {
            "scenario_writer": "openai",
            "answer_teacher": "openai",
        },
        "generation": {
            "count": 2,
            "languages": ["en"],
            "interaction_style": "single_turn",
            "response_shapes": ["plain_text"],
            "attach_targets": True,
            "train_fraction": 0.8,
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert len(rows) == 2
    assert rows[0]["meta"]["target_source"] == "answer_teacher_failed"
    assert "generation_error" in rows[0]["hidden"]
    assert rows[0]["messages"][-1]["role"] == "user"
    assert rows[1]["meta"]["target_source"] == "answer_teacher"
    assert rows[1]["messages"][-1]["role"] == "assistant"

    verification = verify(result.run_id)
    assert verification["response_checks_applied"] is True
    assert verification["failed_rows"] == 1


def test_instruction_following_build_keeps_other_rows_when_one_row_crashes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: None,
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_create_plan",
        lambda cfg, seed: {
            "mode": "planned_rows",
            "rows": [
                {
                    "language": "en",
                    "interaction_style": "single_turn",
                    "response_shape": "plain_text",
                    "scenario_kind": "announcement",
                    "topic": "a broken festival row",
                    "constraints": [
                        {"id": "format:no_digits", "params": {}},
                    ],
                },
                {
                    "language": "en",
                    "interaction_style": "single_turn",
                    "response_shape": "plain_text",
                    "scenario_kind": "announcement",
                    "topic": "a healthy library row",
                    "constraints": [
                        {"id": "format:no_digits", "params": {}},
                    ],
                },
            ],
            "benchmark": "ifbench",
            "language_counts": {"en": 2},
            "interaction_style_counts": {"single_turn": 2},
            "response_shape_counts": {"plain_text": 2},
            "constraint_counts": {
                "format:no_digits": 2,
            },
        },
    )

    original_materialize = instruction_following_build_module.materialize_messages

    def flaky_materialize(bundle, row_plan, *, instruction_block: str):
        if row_plan["topic"] == "a broken festival row":
            raise RuntimeError("row exploded")
        return original_materialize(bundle, row_plan, instruction_block=instruction_block)

    monkeypatch.setattr(
        instruction_following_build_module,
        "materialize_messages",
        flaky_materialize,
    )

    cfg = {
        "pack": "instruction_following",
        "reuse_completed": False,
        "models": {"scenario_writer": "openai"},
        "generation": {
            "count": 2,
            "languages": ["en"],
            "interaction_style": "single_turn",
            "response_shapes": ["plain_text"],
            "attach_targets": False,
            "train_fraction": 0.8,
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert len(rows) == 2
    assert rows[0]["meta"]["row_status"] == "generation_failed"
    assert rows[0]["hidden"]["generation_error"] == "RuntimeError: row exploded"
    assert rows[0]["messages"][0]["content"]
    assert "row_status" not in rows[1]["meta"]

    verification = verify(result.run_id)
    assert verification["response_checks_applied"] is False
    assert verification["failed_rows"] == 1


def test_instruction_following_build_treats_empty_target_as_failed_generation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_scenario_writer",
        lambda cfg: FakeScenarioWriter(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_load_answer_teacher",
        lambda cfg: EmptyAnswerTeacher(),
    )
    monkeypatch.setattr(
        instruction_following_build_module,
        "_create_plan",
        lambda cfg, seed: {
            "mode": "planned_rows",
            "rows": [
                {
                    "language": "en",
                    "interaction_style": "single_turn",
                    "response_shape": "plain_text",
                    "scenario_kind": "announcement",
                    "topic": "a public library",
                    "prompt_seed": {
                        "text": "Write a short answer about the public library.",
                    },
                    "constraints": [
                        {"id": "start_end:first_word", "params": {"word": "amber"}},
                        {"id": "start_end:end_phrase", "params": {"phrase": "that is the full answer."}},
                    ],
                },
            ],
            "benchmark": "ifbench",
            "language_counts": {"en": 1},
            "interaction_style_counts": {"single_turn": 1},
            "response_shape_counts": {"plain_text": 1},
            "constraint_counts": {
                "start_end:first_word": 1,
                "start_end:end_phrase": 1,
            },
        },
    )

    cfg = {
        "pack": "instruction_following",
        "reuse_completed": False,
        "models": {
            "scenario_writer": "openai",
            "answer_teacher": "openai",
        },
        "generation": {
            "count": 1,
            "languages": ["en"],
            "interaction_style": "single_turn",
            "response_shapes": ["plain_text"],
            "attach_targets": True,
            "train_fraction": 0.8,
        },
    }

    result = build(cfg)
    rows = store.read_jsonl(Path(load(result.run_id).artifacts["dataset"].path))

    assert rows[0]["meta"]["target_source"] == "answer_teacher_failed"
    assert rows[0]["hidden"]["generation_error"] == "ValueError: answer_teacher returned an empty response"
    assert rows[0]["messages"][-1]["role"] == "user"

    verification = verify(result.run_id)
    assert verification["response_checks_applied"] is True
    assert verification["failed_rows"] == 1
