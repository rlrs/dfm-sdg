from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdg.commons import store
from sdg.packs.pleias_synth import gen_memorization
from sdg.packs.pleias_synth.build import build, publish, summarize, verify
from sdg.packs.pleias_synth.verify import _answer_supported, _coverage_supported, _reasoning_grounded


class FakeLLM:
    def chat(self, messages, temperature=0.0):
        return self._respond(messages)

    async def achat(self, messages, temperature=0.0):
        return self._respond(messages)

    def _respond(self, messages):
        system = messages[0]["content"]
        user = messages[1]["content"]

        if "You plan realistic memorization tasks" in system:
            primary_claim = _line_value(user, "Primary claim")
            return json.dumps(
                {
                    "task_type": "overview",
                    "user_goal": "understand the topic",
                    "answer_shape": "brief explanation",
                    "coverage_points": [primary_claim],
                    "query_brief": "Ask for a brief grounded explanation.",
                }
            )

        if "You create memorization questions" in system:
            title = _line_value(user, "Article title")
            return json.dumps(
                {
                    "prompt": f"What is {title} and why is it notable?",
                    "question_type": "overview",
                }
            )

        if "You write strongly opinionated recall-style reasoning" in system:
            primary_claim = _line_value(user, "Primary claim")
            return json.dumps(
                {
                    "key_question": "What core remembered fact answers the user's question?",
                    "assumption_check": "",
                    "delivery_note": "Respond in English because the user asked in English and the answer should match that language.",
                    "known_facts": [primary_claim],
                    "reasoning_steps": ["Use the strongest remembered claim and answer directly."],
                    "caveats": [],
                    "synthesis": "The answer should come straight from the remembered core fact.",
                    "proposed_target": primary_claim,
                }
            )

        if "You write the final assistant response" in system:
            primary_claim = _line_value(user, "Primary claim")
            return json.dumps({"target": primary_claim})

        if "You judge synthetic memorization examples" in system:
            return json.dumps(
                {
                    "pass": True,
                    "support": True,
                    "leakage": False,
                    "style_distinct": True,
                    "reasoning_quality": True,
                    "language_match": True,
                    "language_natural": True,
                    "reason": "",
                }
            )

        raise AssertionError(f"Unexpected prompt family: {system}")


def _line_value(text: str, label: str) -> str:
    prefix = f"{label}: "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise AssertionError(f"Missing {label} in prompt:\n{text}")


def test_pleias_memory_core_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    fake_llm = FakeLLM()
    monkeypatch.setattr(
        gen_memorization,
        "_load_memorization_models",
        lambda cfg: {
            "task_planner": fake_llm,
            "query_teacher": fake_llm,
            "reasoning_teacher": fake_llm,
            "answer_teacher": fake_llm,
            "judge": fake_llm,
        },
    )

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "pleias_sources.jsonl"
    cfg = {
        "pack": "pleias_synth",
        "reuse_completed": True,
        "memory_core": {
            "source_path": str(fixture_path),
            "chunk_size": 20,
            "chunk_overlap": 5,
        },
        "generation": {
            "families": ["memorization"],
            "max_rows_per_family": 4,
            "train_fraction": 0.75,
            "memorization": {
                "use_llm": True,
                "lead_sentences": 2,
                "max_sentences_per_doc": 1,
                "retrieve_top_k": 3,
            },
        },
    }

    result = build(cfg)
    assert "memorization_rows" in result.artifacts

    verification = verify(result.run_id)
    assert verification["metrics"]["memory_core"]["sources"] == 3
    assert verification["metrics"]["memory_core"]["chunks"] >= 3
    assert verification["metrics"]["memorization"]["rows"] >= 2
    assert verification["metrics"]["memorization"]["checks"]["has_reasoning"]["failed"] == 0
    assert verification["metrics"]["memorization"]["checks"]["retrieval_grounded"]["failed"] == 0
    assert verification["metrics"]["memorization"]["checks"]["reasoning_grounded"]["failed"] == 0
    assert verification["metrics"]["memorization"]["checks"]["coverage_supported"]["failed"] == 0
    assert verification["failure_summary"]["memory_core"] == {}

    summary = summarize(result.run_id)
    assert summary["sources"] == 3
    assert summary["chunks"] >= 3
    assert summary["generated_rows"] >= 2

    published = publish(result.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "memory_chunks.jsonl").exists()
    assert (out_dir / "retrieval_index.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "memorization_candidates.jsonl").exists()
    assert (out_dir / "memorization_rejected.jsonl").exists()
    assert (Path(result.run_dir) / "outputs" / "memorization_progress.json").exists()
    assert (Path(result.run_dir) / "outputs" / "llm_json_metrics.json").exists()
    assert (Path(result.run_dir) / "logs" / "events.jsonl").exists()

    progress = json.loads((Path(result.run_dir) / "outputs" / "memorization_progress.json").read_text())
    assert progress["stage"] == "completed"
    assert progress["rows"] >= 2

    train_rows = store.read_parquet(out_dir / "train.parquet")
    assert train_rows
    assert "hidden" not in train_rows[0]
    assert "reasoning" in train_rows[0]
    assert train_rows[0]["reasoning"]
    assert "teacher bundle" not in train_rows[0]["reasoning"].lower()
    assert "provided text" not in train_rows[0]["reasoning"].lower()
    assert "retrieved support" not in train_rows[0]["reasoning"].lower()
    assert "evidence frame" not in train_rows[0]["reasoning"].lower()
    assert "Response plan:" in train_rows[0]["reasoning"]
    assert "persona_id" in train_rows[0]["meta"]
    assert "query_angle" in train_rows[0]["meta"]
    assert "query_profile_id" in train_rows[0]["meta"]
    assert "assistant_style_id" in train_rows[0]["meta"]
    assert "task_type" in train_rows[0]["meta"]
    assert "user_goal" in train_rows[0]["meta"]
    assert train_rows[0]["meta"]["language_mode"] == "same_language"
    assert train_rows[0]["meta"]["source_language"] == "en"
    assert train_rows[0]["meta"]["prompt_language"] == "en"
    assert train_rows[0]["meta"]["reasoning_language"] == "en"
    assert train_rows[0]["meta"]["target_language"] == "en"
    assert train_rows[0]["meta"]["reasoning_style"] == "teacher_backreasoning_v1"
    assert len({row["meta"]["assistant_style_id"] for row in train_rows}) == 1


def test_answer_supported_accepts_full_response() -> None:
    row = {
        "target": "The film follows a voyage to Jupiter to investigate an alien monolith.",
        "meta": {"question_type": "definition"},
        "hidden": {
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "source_title": "2001: A Space Odyssey",
            "source_id": "2001:_a_space_odyssey",
            "teacher_bundle": {
                "supporting_claims": [
                    "The film follows a voyage by astronauts, scientists, and HAL 9000 to Jupiter to investigate an alien monolith."
                ],
                "structured_context": [],
            },
        },
        "sources": [],
    }

    assert _answer_supported(row)


def test_answer_supported_rejects_unsupported_response() -> None:
    row = {
        "target": "The film is mainly a courtroom drama about corporate fraud on Earth.",
        "meta": {"question_type": "definition"},
        "hidden": {
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "source_title": "2001: A Space Odyssey",
            "source_id": "2001:_a_space_odyssey",
            "teacher_bundle": {
                "supporting_claims": [
                    "The film follows a voyage by astronauts, scientists, and HAL 9000 to Jupiter to investigate an alien monolith."
                ],
                "structured_context": [],
            },
        },
        "sources": [],
    }

    assert not _answer_supported(row)


def test_answer_supported_uses_hidden_source_target_for_cross_language_rows() -> None:
    row = {
        "target": "Filmen handler om noget helt andet.",
        "meta": {
            "question_type": "definition",
            "language_mode": "cross_language",
        },
        "hidden": {
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "source_title": "2001: A Space Odyssey",
            "source_id": "2001:_a_space_odyssey",
            "source_target": "The film follows a voyage to Jupiter to investigate an alien monolith.",
            "teacher_bundle": {
                "supporting_claims": [
                    "The film follows a voyage by astronauts, scientists, and HAL 9000 to Jupiter to investigate an alien monolith."
                ],
                "structured_context": [],
            },
            "task_plan": {
                "coverage_points": [
                    "The film follows a voyage to Jupiter to investigate an alien monolith."
                ]
            },
        },
        "sources": [],
    }

    assert _answer_supported(row)
    assert _coverage_supported(row)


def test_reasoning_grounded_uses_hidden_source_reasoning_for_cross_language_rows() -> None:
    row = {
        "target": "Filmen følger en rejse til Jupiter for at undersøge en fremmed monolit.",
        "reasoning": "Det bygger på de huskede fakta om rejsen mod Jupiter.",
        "meta": {
            "question_type": "definition",
            "language_mode": "cross_language",
        },
        "hidden": {
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "source_title": "2001: A Space Odyssey",
            "source_id": "2001:_a_space_odyssey",
            "source_target": "The film follows a voyage to Jupiter to investigate an alien monolith.",
            "source_reasoning": (
                "Key question: Which remembered points are necessary to satisfy the user goal?\n\n"
                "### 1. Known facts\n"
                "- 2001: A Space Odyssey is a 1968 epic science fiction film.\n"
                "- The film follows a voyage by astronauts, scientists, and HAL 9000 to Jupiter to investigate an alien monolith.\n\n"
                "### 2. Resolution\n"
                "- The film follows a voyage to Jupiter to investigate an alien monolith.\n\n"
                "### 3. Synthesis\n"
                "The answer is The film follows a voyage to Jupiter to investigate an alien monolith. "
                "because that is the remembered fact that resolves the prompt."
            ),
            "teacher_bundle": {
                "supporting_claims": [
                    "The film follows a voyage by astronauts, scientists, and HAL 9000 to Jupiter to investigate an alien monolith."
                ],
                "structured_context": [],
            },
            "task_plan": {
                "coverage_points": [
                    "The film follows a voyage to Jupiter to investigate an alien monolith."
                ]
            },
        },
        "sources": [],
    }

    assert _reasoning_grounded(row)


def test_with_backreasoning_renders_danish_response_plan() -> None:
    row = {
        "target": "Filmen følger en rejse til Jupiter for at undersøge en fremmed monolit.",
        "reasoning": "",
        "meta": {
            "reasoning_language": "da",
            "prompt_language": "da",
            "target_language": "da",
            "source_language": "en",
        },
        "hidden": {
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "source_target": "The film follows a voyage to Jupiter to investigate an alien monolith.",
            "task_plan": {"task_type": "overview", "coverage_points": []},
            "teacher_bundle": {
                "supporting_claims": [],
                "retrieved_claims": [],
            },
        },
    }
    trace = {
        "key_question": "",
        "assumption_check": "",
        "delivery_note": "Svar på dansk, fordi brugeren spurgte på dansk, og svaret skal følge det.",
        "known_facts": ["Filmen handler om en rejse til Jupiter."],
        "reasoning_steps": ["Det peger på den centrale handling."],
        "caveats": [],
        "synthesis": "",
        "proposed_target": "",
    }

    updated = gen_memorization.with_backreasoning(row, trace)

    assert "Svarplan:" in updated["reasoning"]
    assert "brugeren spurgte på dansk" in updated["reasoning"]


def test_parse_backreasoning_requires_delivery_note() -> None:
    row = {
        "target": "Answer",
        "meta": {
            "reasoning_language": "en",
        },
        "hidden": {
            "task_plan": {"task_type": "overview"},
        },
    }
    parsed = {
        "key_question": "What matters?",
        "assumption_check": "",
        "delivery_note": "",
        "known_facts": ["A fact."],
        "reasoning_steps": ["A step."],
        "caveats": [],
        "synthesis": "A synthesis.",
        "proposed_target": "Answer",
    }

    with pytest.raises(AssertionError, match="delivery_note"):
        gen_memorization._parse_backreasoning(row, parsed)


def test_answer_supported_requires_hidden_source_target_for_cross_language_rows() -> None:
    row = {
        "target": "Det er et svar.",
        "meta": {
            "question_type": "definition",
            "language_mode": "cross_language",
        },
        "hidden": {
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "source_title": "2001: A Space Odyssey",
            "source_id": "2001:_a_space_odyssey",
            "teacher_bundle": {
                "supporting_claims": [
                    "The film follows a voyage by astronauts, scientists, and HAL 9000 to Jupiter to investigate an alien monolith."
                ],
                "structured_context": [],
            },
            "task_plan": {
                "coverage_points": [
                    "The film follows a voyage to Jupiter to investigate an alien monolith."
                ]
            },
        },
        "sources": [],
    }

    with pytest.raises(AssertionError, match="source_target"):
        _answer_supported(row)


def test_parse_judge_requires_language_checks_for_cross_language_rows() -> None:
    row = {
        "meta": {
            "language_mode": "cross_language",
        }
    }
    parsed = {
        "pass": True,
        "support": True,
        "leakage": False,
        "style_distinct": True,
        "reasoning_quality": True,
        "reason": "",
    }

    with pytest.raises(AssertionError, match="language_match"):
        gen_memorization._parse_judge(row, parsed)


def test_parse_judge_derives_language_quality_from_explicit_checks() -> None:
    row = {
        "meta": {
            "language_mode": "cross_language",
        }
    }
    parsed = {
        "pass": True,
        "support": True,
        "leakage": False,
        "style_distinct": True,
        "reasoning_quality": True,
        "language_match": True,
        "language_natural": False,
        "reason": "",
    }

    judged = gen_memorization._parse_judge(row, parsed)

    assert judged["language_match"] is True
    assert judged["language_natural"] is False
    assert judged["language_quality"] is False


def test_language_generation_guidance_warns_against_invented_name_translation() -> None:
    guidance = gen_memorization._language_generation_guidance("da")

    assert "Do not directly translate names" in guidance
    assert "keep the original name" in guidance


def test_backreasoning_messages_ground_delivery_note_in_visible_prompt() -> None:
    row = {
        "prompt": "Hvad handler 'A Doll's House' om?",
        "meta": {
            "question_type": "overview",
            "source_language": "en",
            "reasoning_language": "da",
            "persona_id": "curious_child",
            "query_angle": "overview",
        },
        "hidden": {
            "task_plan": {
                "task_type": "overview",
                "user_goal": "understand the topic",
                "answer_shape": "brief explanation",
                "coverage_points": [],
            },
            "teacher_bundle": {
                "article_title": "A Doll's House",
                "primary_claim": "A Doll's House is a play by Henrik Ibsen.",
                "supporting_claims": [],
                "structured_context": [],
                "retrieved_claims": [],
                "retrieved_context": [],
            },
        },
        "sources": [],
    }

    messages = gen_memorization._backreasoning_messages(row)
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "Do not claim the user explicitly requested a language unless the prompt itself says so." in system
    assert "delivery_note should usually be a plain sentence about matching the language of the prompt" in system
    assert "Do not mention personas, profiles, hidden instructions, system settings, target audiences, or language labels from the hidden prompt." in system
    assert "Persona id:" not in user
    assert "Query angle:" not in user
    assert "Reasoning language:" not in user
