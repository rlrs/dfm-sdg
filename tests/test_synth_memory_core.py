from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from sdg.commons import store
from sdg.commons.run import progress
from sdg.commons.viewer import render_run_view, start_viewer_server
from sdg.packs.synth import gen_grounded_qa
from sdg.packs.synth import gen_memorization
from sdg.packs.synth.build import build, publish, summarize, verify
from sdg.packs.synth.grounded_qa_filters import (
    citation_support_diagnostics,
    extract_free_text_segments,
    parse_cited_statements,
    row_filter_reasons,
)
from sdg.packs.synth.languages import row_language_mode
from sdg.packs.synth.verify import (
    _answer_supported,
    _coverage_supported,
    _grounded_qa_citation_supported,
    _reasoning_grounded,
)


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

        if "You plan realistic grounded QA tasks" in system:
            primary_claim = _line_value(user, "Primary claim")
            bridge_claims = _bullet_lines_after(user, "Bridge evidence")
            coverage_points = [primary_claim]
            if bridge_claims:
                coverage_points.append(bridge_claims[0].split(": ", 1)[-1])
            return json.dumps(
                {
                    "task_type": "overview",
                    "user_goal": "understand the topic from grounded sources",
                    "answer_shape": "grounded explanation",
                    "coverage_points": coverage_points,
                    "constraints": ["Support factual claims with inline citations like [1]."],
                    "query_brief": "Ask for a grounded explanation that combines the sources.",
                }
            )

        if "You create user-facing grounded QA prompts" in system:
            seed_title = _line_value(user, "Seed article")
            return json.dumps(
                {
                    "prompt": f"What do the sources show about {seed_title}?",
                    "question_type": "overview",
                }
            )

        if "You write grounded reasoning notes for synthetic QA data" in system:
            coverage_points = _bullet_lines_after(user, "Coverage points")
            known_facts = coverage_points[:2] or ["Use the packed sources."]
            return json.dumps(
                {
                    "key_question": "Which source-backed points matter most for the answer?",
                    "delivery_note": "Respond in English because the prompt is in English and the answer should match it.",
                    "source_plan": [
                        "Source 1 carries the main answer.",
                        "Source 2 only adds supporting detail.",
                    ],
                    "known_facts": known_facts,
                    "reasoning_steps": ["Combine the strongest cited points into one grounded answer."],
                    "caveats": [],
                    "synthesis": "The answer should stay close to the cited sources.",
                    "proposed_target": "",
                }
            )

        if "You write the final assistant response for synthetic grounded QA data" in system:
            coverage_points = _bullet_lines_after(user, "Coverage points")
            source_target = " ".join(coverage_points[:2])
            return json.dumps(
                {
                    "target": f'<statement cites="1 2">{source_target}</statement>',
                    "source_target": f'<statement cites="1 2">{source_target}</statement>',
                }
            )

        if "You judge synthetic grounded QA examples" in system:
            return json.dumps(
                {
                    "pass": True,
                    "support": True,
                    "citations": True,
                    "completeness": True,
                    "language_match": True,
                    "language_natural": True,
                    "reason": "",
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


class SequenceLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    async def achat(self, messages, temperature=0.0):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _line_value(text: str, label: str) -> str:
    prefix = f"{label}: "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise AssertionError(f"Missing {label} in prompt:\n{text}")


def _bullet_lines_after(text: str, label: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line != f"{label}:":
            continue
        values: list[str] = []
        for item in lines[index + 1 :]:
            if not item.startswith("- "):
                break
            values.append(item[2:].strip())
        return values
    return []


def test_synth_memory_core_flow(tmp_path, monkeypatch) -> None:
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

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "synth_sources.jsonl"
    cfg = {
        "pack": "synth",
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
    assert summary["memorization"]["rows"] >= 2
    assert summary["memorization"]["candidate_rows"] >= summary["memorization"]["rows"]
    assert summary["memorization"]["kept_preview"]
    assert summary["progress"]["stage"] == "completed"
    assert summary["llm_json_metrics"]["parse_successes"] >= 1
    run_progress = progress(result.run_id)
    assert "memorization_progress" in run_progress["snapshots"]
    assert "llm_json_metrics" in run_progress["snapshots"]
    assert all(event["component"] != "model" for event in run_progress["recent_events"])
    run_progress_with_models = progress(result.run_id, include_model_events=True, limit=200)
    if "model_metrics" in run_progress_with_models["snapshots"]:
        assert any(event["component"] == "model" for event in run_progress_with_models["recent_events"])

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

    progress_snapshot = json.loads((Path(result.run_dir) / "outputs" / "memorization_progress.json").read_text())
    assert progress_snapshot["stage"] == "completed"
    assert progress_snapshot["rows"] >= 2

    train_rows = store.read_parquet(out_dir / "train.parquet")
    assert train_rows
    assert "hidden" not in train_rows[0]
    assert "reasoning" in train_rows[0]
    assert train_rows[0]["reasoning"]
    assert train_rows[0]["messages"][0]["role"] == "system"
    assert train_rows[0]["messages"][1]["role"] == "user"
    assert train_rows[0]["messages"][2]["role"] == "assistant"
    assert train_rows[0]["messages"][1]["content"] == train_rows[0]["prompt"]
    assert train_rows[0]["messages"][2]["content"].startswith("<think>\n")
    assert train_rows[0]["messages"][2]["content"].endswith(train_rows[0]["target"])
    assert train_rows[0]["reasoning"] in train_rows[0]["messages"][2]["content"]
    assert "Answer in" not in train_rows[0]["messages"][0]["content"]
    assert "English" not in train_rows[0]["messages"][0]["content"]
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
    assert "language_mode" not in train_rows[0]["meta"]
    assert row_language_mode(train_rows[0]) == "same_language"
    assert train_rows[0]["meta"]["source_language"] == "en"
    assert train_rows[0]["meta"]["prompt_language"] == "en"
    assert train_rows[0]["meta"]["reasoning_language"] == "en"
    assert train_rows[0]["meta"]["target_language"] == "en"
    assert train_rows[0]["meta"]["reasoning_style"] == "teacher_backreasoning_v1"
    assert len({row["meta"]["assistant_style_id"] for row in train_rows}) == 1


def test_synth_grounded_qa_flow(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    fake_llm = FakeLLM()
    monkeypatch.setattr(
        gen_grounded_qa,
        "_load_grounded_qa_models",
        lambda cfg: {
            "task_planner": fake_llm,
            "query_teacher": fake_llm,
            "reasoning_teacher": fake_llm,
            "answer_teacher": fake_llm,
            "judge": fake_llm,
        },
    )

    source_path = tmp_path / "grounded_docs.jsonl"
    store.write_jsonl(
        [
            {
                "id": "solar-system",
                "title": "Solar System",
                "text": (
                    "The Solar System consists of the Sun and the objects that orbit it. "
                    "Earth is the third planet from the Sun."
                ),
                "source": "fixture",
                "url": "https://example.invalid/solar-system",
                "license": "CC BY-SA 4.0",
                "meta": {"dataset": "fixture"},
            },
            {
                "id": "earth",
                "title": "Earth",
                "text": (
                    "Earth is the third planet from the Sun. "
                    "Earth is part of the Solar System."
                ),
                "source": "fixture",
                "url": "https://example.invalid/earth",
                "license": "CC BY-SA 4.0",
                "meta": {"dataset": "fixture"},
            },
            {
                "id": "jupiter",
                "title": "Jupiter",
                "text": (
                    "Jupiter is the largest planet in the Solar System. "
                    "Jupiter orbits the Sun."
                ),
                "source": "fixture",
                "url": "https://example.invalid/jupiter",
                "license": "CC BY-SA 4.0",
                "meta": {"dataset": "fixture"},
            },
        ],
        source_path,
    )

    cfg = {
        "pack": "synth",
        "reuse_completed": True,
        "memory_core": {
            "source_path": str(source_path),
            "chunk_size": 20,
            "chunk_overlap": 5,
        },
        "generation": {
            "families": ["grounded_qa"],
            "max_rows_per_family": 4,
            "train_fraction": 0.75,
            "grounded_qa": {
                "lead_sentences": 2,
                "max_sentences_per_doc": 1,
                "min_sources": 2,
                "bridge_sources": 2,
                "retrieve_top_k": 4,
            },
        },
    }

    result = build(cfg)
    assert "grounded_qa_rows" in result.artifacts

    verification = verify(result.run_id)
    assert verification["metrics"]["grounded_qa"]["rows"] >= 1
    assert verification["metrics"]["grounded_qa"]["checks"]["has_citations"]["failed"] == 0
    assert verification["metrics"]["grounded_qa"]["checks"]["citation_supported"]["failed"] == 0
    assert "citation_supported" in verification["metrics"]["grounded_qa"]["heuristics"]
    assert verification["metrics"]["grounded_qa"]["checks"]["retrieval_grounded"]["failed"] == 0

    summary = summarize(result.run_id)
    assert summary["grounded_qa"]["rows"] >= 1
    assert summary["grounded_qa"]["source_counts"]
    assert summary["family_progress"]["grounded_qa"]["stage"] == "completed"

    published = publish(result.run_id)
    out_dir = Path(published["out_dir"])
    train_rows = store.read_parquet(out_dir / "train.parquet")
    grounded_rows = [row for row in train_rows if row["meta"]["family"] == "grounded_qa"]
    assert grounded_rows
    assert '<statement cites="' in grounded_rows[0]["target"]
    assert grounded_rows[0]["query_seed_url"]
    assert grounded_rows[0]["additional_seed_urls"]
    assert grounded_rows[0]["constraints"]
    assert "citation_check" in grounded_rows[0]["scores"]
    assert grounded_rows[0]["messages"][0]["role"] == "system"
    assert grounded_rows[0]["messages"][1]["role"] == "user"
    assert grounded_rows[0]["messages"][2]["role"] == "assistant"
    assert grounded_rows[0]["messages"][1]["content"] == grounded_rows[0]["prompt"]
    assert grounded_rows[0]["messages"][2]["content"].startswith("<think>\n")
    assert grounded_rows[0]["messages"][2]["content"].endswith(grounded_rows[0]["target"])
    assert grounded_rows[0]["reasoning"] in grounded_rows[0]["messages"][2]["content"]
    assert "provided sources" in grounded_rows[0]["messages"][0]["content"]
    assert "only facts grounded" in grounded_rows[0]["messages"][0]["content"]
    assert "Synth Assistant" in grounded_rows[0]["messages"][0]["content"]
    assert "Answer in" not in grounded_rows[0]["messages"][0]["content"]
    assert "English" not in grounded_rows[0]["messages"][0]["content"]

    view = render_run_view(result.run_id, artifact_name="grounded_qa_rows", limit=10)
    viewer_path = Path(view["out_path"])
    assert view["default_artifact"] == "grounded_qa_rows"
    assert viewer_path.exists()
    viewer_html = viewer_path.read_text()
    assert "Synth Viewer" in viewer_html
    assert "Grounded QA kept" in viewer_html
    assert "Reasoning" in viewer_html
    assert "Messages" in viewer_html
    assert "Rows per page" in viewer_html
    assert "Type" in viewer_html
    assert "section-markdown" in viewer_html

    running = start_viewer_server(result.run_id, artifact_name="grounded_qa_rows", host="127.0.0.1", port=0)
    try:
        with httpx.Client(base_url=running.base_url, timeout=5.0) as client:
            progress_payload = client.get("/api/progress").json()
            assert progress_payload["status"] == "completed"
            assert "grounded_qa_rows" in progress_payload["artifacts"]
            payload = client.get(
                "/api/artifact",
                params={"name": "grounded_qa_rows", "page": 1, "page_size": 5},
            ).json()
            filter_keys = {item["key"] for item in payload["filters"]}
            assert "meta.question_type" in filter_keys
            assert payload["artifact"]["name"] == "grounded_qa_rows"
            assert payload["items"]
            assert "reasoning" in payload["items"][0]["raw_json"].lower()
            section_formats = {section["format"] for section in payload["items"][0]["sections"]}
            assert "markdown" in section_formats
            assert any(section["label"] == "Messages" for section in payload["items"][0]["sections"])
            reasoning_section = next(section for section in payload["items"][0]["sections"] if section["label"] == "Reasoning")
            assert reasoning_section["html"]
    finally:
        running.close()


def test_parse_cited_statements_allows_plain_text_between_statements() -> None:
    parsed = parse_cited_statements(
        "Kort svar: "
        '<statement cites="1">Første pointe.</statement> '
        "Mere kontekst. "
        '<statement cites="2 3">Anden pointe.</statement>'
    )

    assert parsed == [
        {"text": "Første pointe.", "citations": ["1"]},
        {"text": "Anden pointe.", "citations": ["2", "3"]},
    ]


def test_extract_free_text_segments_ignores_punctuation_only_tails() -> None:
    segments = extract_free_text_segments(
        'Kort svar: <statement cites="1">Første pointe.</statement>. '
        '<statement cites="2">Anden pointe.</statement>'
    )

    assert segments == ["Kort svar:"]


def test_citation_support_diagnostics_accepts_short_identification_statement() -> None:
    row = {
        "target": '<statement cites="3">Stanley Kubrick</statement>',
        "sources": [
            {
                "citation_id": "3",
                "snippet": "Stanley Kubrick directed, produced, and co-wrote 2001: A Space Odyssey.",
            }
        ],
        "hidden": {
            "source_target": '<statement cites="3">Stanley Kubrick</statement>',
        },
        "meta": {
            "source_language": "en",
        },
    }

    diagnostics = citation_support_diagnostics(row)

    assert diagnostics["ok"] is True
    assert diagnostics["supported_statements"] == 1


def test_answer_messages_push_for_small_supported_statements() -> None:
    row = {
        "prompt": "Hvem skrev manuskriptet?",
        "reasoning": "Svarplan: Svar på dansk.",
        "sources": [
            {
                "citation_id": "3",
                "title": "2001: A Space Odyssey",
                "snippet": "Arthur C. Clarke co-wrote the screenplay for 2001: A Space Odyssey with Stanley Kubrick.",
            }
        ],
        "meta": {
            "source_language": "en",
            "target_language": "da",
            "question_type": "recall",
        },
        "hidden": {
            "persona": {
                "name": "Test",
                "knowledge_level": "novice",
            },
            "query_profile": {
                "channel": "chat",
                "fluency": "fluent",
            },
            "assistant_style": {
                "style_id": "plain",
                "instructions": "Be direct.",
                "exemplars": [],
            },
            "task_plan": {
                "task_type": "identification",
                "user_goal": "find the name",
                "answer_shape": "short answer",
                "coverage_points": ["Arthur C. Clarke co-wrote the screenplay with Stanley Kubrick."],
                "constraints": [],
            },
        },
    }

    system = gen_grounded_qa._answer_messages(row)[0]["content"]

    assert "Prefer several short statements over one long statement." in system
    assert "If part of a sentence is less directly supported, split it into a separate statement or drop it." in system
    assert "If a sentence contains a factual claim, put that sentence inside a statement tag." in system
    assert "Do not leave names, titles, dates, attributions, or other factual details outside statement tags." in system
    assert "Use the source plan in the recall notes" in system
    assert "For identification tasks, a single short statement is enough." in system


def test_answer_messages_require_multi_source_synthesis_for_broad_rows() -> None:
    row = {
        "prompt": "Hvordan hænger de her ting sammen?",
        "reasoning": "Svarplan: Svar på dansk.",
        "sources": [
            {
                "citation_id": "1",
                "title": "One",
                "snippet": "First supporting source.",
            },
            {
                "citation_id": "2",
                "title": "Two",
                "snippet": "Second supporting source.",
            },
        ],
        "meta": {
            "source_language": "en",
            "target_language": "da",
            "question_type": "overview",
        },
        "hidden": {
            "required_cited_sources": 2,
            "persona": {
                "name": "Test",
                "knowledge_level": "novice",
            },
            "query_profile": {
                "channel": "chat",
                "fluency": "fluent",
            },
            "assistant_style": {
                "style_id": "plain",
                "instructions": "Be direct.",
                "exemplars": [],
            },
            "task_plan": {
                "task_type": "overview",
                "user_goal": "connect the sources",
                "answer_shape": "grounded explanation",
                "coverage_points": ["First point", "Second point"],
                "constraints": [],
            },
        },
    }

    system = gen_grounded_qa._answer_messages(row)[0]["content"]

    assert "synthesize at least 2 different cited sources" in system


@pytest.mark.anyio
async def test_generate_answer_retries_when_model_omits_statement_tags() -> None:
    row = {
        "id": "grounded_qa-000001",
        "prompt": "Hvordan hænger det sammen?",
        "reasoning": "Svarplan: Svar på dansk.",
        "sources": [
            {
                "citation_id": "1",
                "title": "One",
                "snippet": "First supporting source.",
            }
        ],
        "meta": {
            "source_language": "en",
            "target_language": "da",
            "question_type": "overview",
        },
        "hidden": {
            "required_cited_sources": 1,
            "persona": {
                "name": "Test",
                "knowledge_level": "novice",
            },
            "query_profile": {
                "channel": "chat",
                "fluency": "fluent",
            },
            "assistant_style": {
                "style_id": "plain",
                "instructions": "Be direct.",
                "exemplars": [],
            },
            "task_plan": {
                "task_type": "overview",
                "user_goal": "understand the topic",
                "answer_shape": "grounded explanation",
                "coverage_points": ["First point"],
                "constraints": [],
            },
        },
    }
    llm = SequenceLLM(
        [
            json.dumps({"target": "Kort svar.", "source_target": "Short answer."}),
            json.dumps(
                {
                    "target": '<statement cites="1">Kort svar.</statement>',
                    "source_target": '<statement cites="1">Short answer.</statement>',
                }
            ),
        ]
    )

    answer = await gen_grounded_qa._generate_answer_async(row, llm)

    assert answer["target"] == '<statement cites="1">Kort svar.</statement>'
    assert llm.calls == 2


def test_generation_error_short_circuits_grounded_qa_filter_reasons() -> None:
    row = {
        "prompt": "Prompt",
        "target": "",
        "reasoning": "",
        "sources": [],
        "meta": {},
        "hidden": {
            "generation_error": "grounded_qa target retry budget exhausted",
        },
        "scores": {},
    }

    assert row_filter_reasons(row) == ["generation_failed"]


def test_grounded_qa_judge_messages_treat_source_target_as_reference_only() -> None:
    row = {
        "prompt": "Hvad går 2001: A Space Odyssey ud på?",
        "target": '<statement cites="3">2001: A Space Odyssey er en science fiction-film</statement>',
        "reasoning": "Svarplan: Svar på dansk.",
        "sources": [
            {
                "citation_id": "3",
                "title": "2001: A Space Odyssey",
                "snippet": "2001: A Space Odyssey is a 1968 epic science fiction film produced and directed by Stanley Kubrick.",
            }
        ],
        "meta": {
            "source_language": "en",
            "prompt_language": "da",
            "reasoning_language": "da",
            "target_language": "da",
        },
        "hidden": {
            "source_target": '<statement cites="3">2001: A Space Odyssey is a science fiction film</statement>',
            "task_plan": {
                "task_type": "overview",
                "user_goal": "understand the topic",
                "coverage_points": ["Explain what the film is."],
            },
        },
    }

    messages = gen_grounded_qa._judge_messages(
        row,
        {
            "has_citations": True,
            "ok": True,
            "supported_statements": 1,
            "total_statements": 1,
            "issues": [],
        },
    )

    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "Use the packed sources as the evidence base" in system
    assert "Do not treat the source-language reference answer as evidence." in system
    assert "Do not use prior knowledge, common knowledge, or plausible inference" in system
    assert "Source-language reference answer (meaning only, not evidence):" in user


def test_retrieve_support_row_keeps_seed_and_bridge_sources_first() -> None:
    seed_chunk = {
        "id": "seed",
        "source_id": "seed_doc",
        "title": "Seed",
        "text": "Seed document fact about the main topic.",
        "url": "https://example.invalid/seed",
        "meta": {"word_count": 6},
    }
    bridge_chunk = {
        "id": "bridge",
        "source_id": "bridge_doc",
        "title": "Bridge",
        "text": "Bridge source adds related context for the same topic.",
        "url": "https://example.invalid/bridge",
        "meta": {"word_count": 9},
    }
    extra_chunk = {
        "id": "extra",
        "source_id": "extra_doc",
        "title": "Extra",
        "text": "Extra source also mentions the topic and related context.",
        "url": "https://example.invalid/extra",
        "meta": {"word_count": 9},
    }
    row = {
        "prompt": "What do the sources show about the topic?",
        "sources": [],
        "hidden": {
            "source_id": "seed_doc",
            "teacher_bundle": {
                "seed_source": gen_grounded_qa._source_spec(seed_chunk, score=5),
                "bridge_sources": [gen_grounded_qa._source_spec(bridge_chunk, score=4)],
            },
            "task_plan": {
                "coverage_points": ["Seed fact", "Bridge fact"],
            },
        },
        "meta": {
            "source_language": "en",
            "prompt_language": "en",
            "reasoning_language": "en",
            "target_language": "en",
        },
    }
    index = {
        "chunks": {
            "seed": {"tokens": set(gen_grounded_qa.tokenize(seed_chunk["text"]))},
            "bridge": {"tokens": set(gen_grounded_qa.tokenize(bridge_chunk["text"]))},
            "extra": {"tokens": set(gen_grounded_qa.tokenize(extra_chunk["text"]))},
        }
    }
    chunk_lookup = {
        "seed": seed_chunk,
        "bridge": bridge_chunk,
        "extra": extra_chunk,
    }

    updated = gen_grounded_qa.retrieve_support_row(row, index, chunk_lookup, {"retrieve_top_k": 3})

    assert [source["source_id"] for source in updated["sources"][:2]] == ["seed_doc", "bridge_doc"]
    assert updated["meta"]["source_count"] == 3


def test_backreasoning_messages_require_source_triage() -> None:
    row = {
        "prompt": "What do the sources show?",
        "sources": [
            {
                "citation_id": "1",
                "title": "One",
                "snippet": "Main source snippet.",
            },
            {
                "citation_id": "2",
                "title": "Two",
                "snippet": "Secondary source snippet.",
            },
        ],
        "meta": {
            "source_language": "en",
            "reasoning_language": "en",
            "question_type": "overview",
        },
        "hidden": {
            "task_plan": {
                "task_type": "overview",
                "user_goal": "understand the topic",
                "answer_shape": "grounded explanation",
                "coverage_points": ["Main point", "Secondary point"],
            },
        },
    }

    system = gen_grounded_qa._backreasoning_messages(row)[0]["content"]

    assert "keys key_question, delivery_note, source_plan, known_facts, reasoning_steps, caveats, synthesis, and proposed_target" in system
    assert "Use source_plan to sort which packed sources are most useful" in system


def test_parse_backreasoning_requires_source_plan() -> None:
    row = {
        "meta": {
            "reasoning_language": "en",
        },
        "hidden": {
            "task_plan": {"task_type": "overview"},
        },
    }
    parsed = {
        "key_question": "What matters?",
        "delivery_note": "Respond in English because the prompt is in English.",
        "source_plan": [],
        "known_facts": ["A fact."],
        "reasoning_steps": ["A step."],
        "caveats": [],
        "synthesis": "A synthesis.",
        "proposed_target": "Answer",
    }

    with pytest.raises(AssertionError, match="source_plan"):
        gen_grounded_qa._parse_backreasoning(row, parsed)


def test_answer_supported_accepts_full_response() -> None:
    row = {
        "target": "The film follows a voyage to Jupiter to investigate an alien monolith.",
        "meta": {
            "question_type": "definition",
            "source_language": "en",
            "prompt_language": "en",
            "reasoning_language": "en",
            "target_language": "en",
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
        },
        "sources": [],
    }

    assert _answer_supported(row)


def test_answer_supported_rejects_unsupported_response() -> None:
    row = {
        "target": "The film is mainly a courtroom drama about corporate fraud on Earth.",
        "meta": {
            "question_type": "definition",
            "source_language": "en",
            "prompt_language": "en",
            "reasoning_language": "en",
            "target_language": "en",
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
        },
        "sources": [],
    }

    assert not _answer_supported(row)


def test_answer_supported_uses_hidden_source_target_for_cross_language_rows() -> None:
    row = {
        "target": "Filmen handler om noget helt andet.",
        "meta": {
            "question_type": "definition",
            "source_language": "en",
            "prompt_language": "da",
            "reasoning_language": "da",
            "target_language": "da",
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
            "source_language": "en",
            "prompt_language": "da",
            "reasoning_language": "da",
            "target_language": "da",
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
        "prompt": "Hvad handler filmen om?",
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
            "assistant_style": {
                "name": "Synth Assistant",
                "instructions": "Answer clearly and directly.",
                "tone": "calm",
                "detail_level": "moderate",
                "structure": "clear",
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
            "source_language": "en",
            "prompt_language": "da",
            "reasoning_language": "da",
            "target_language": "da",
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
            "source_language": "en",
            "prompt_language": "da",
            "reasoning_language": "da",
            "target_language": "da",
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


def test_grounded_qa_verify_uses_judge_for_citation_supported() -> None:
    row = {
        "scores": {
            "judge": {
                "citations": True,
            }
        },
        "target": '<statement cites="3">Den anses som et landmark inden for science fiction-cinema</statement>',
        "sources": [
            {
                "citation_id": "3",
                "snippet": "The film is noted for its pioneering special effects and ambiguous themes.",
            }
        ],
        "hidden": {
            "source_target": '<statement cites="3">It is regarded as a landmark in science fiction cinema</statement>',
        },
        "meta": {
            "source_language": "en",
        },
    }

    assert citation_support_diagnostics(row)["ok"] is False
    assert _grounded_qa_citation_supported(row) is True


def test_parse_judge_derives_language_quality_from_explicit_checks() -> None:
    row = {
        "meta": {
            "source_language": "en",
            "prompt_language": "da",
            "reasoning_language": "da",
            "target_language": "da",
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


def test_build_resumes_failed_memorization_run(tmp_path, monkeypatch) -> None:
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

    original_judge = gen_memorization._judge_row_async
    state = {"failed": False}

    async def flaky_judge(row: dict[str, object], llm: object) -> dict[str, object]:
        if row["id"] == "memorization-000001" and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("planned stop")
        return await original_judge(row, llm)  # type: ignore[arg-type]

    monkeypatch.setattr(gen_memorization, "_judge_row_async", flaky_judge)

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "synth_sources.jsonl"
    cfg = {
        "pack": "synth",
        "reuse_completed": False,
        "resume_incomplete": True,
        "memory_core": {
            "source_path": str(fixture_path),
            "chunk_size": 20,
            "chunk_overlap": 5,
        },
        "generation": {
            "families": ["memorization"],
            "max_rows_per_family": 3,
            "train_fraction": 0.75,
            "memorization": {
                "lead_sentences": 2,
                "max_sentences_per_doc": 1,
                "retrieve_top_k": 3,
            },
        },
    }

    with pytest.raises(RuntimeError, match="planned stop"):
        build(cfg)

    run_dirs = sorted((tmp_path / "artifacts" / "runs" / "synth").glob("*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    partial_candidates = store.read_jsonl(run_dir / "outputs" / "memorization_candidates.jsonl")
    assert len(partial_candidates) == 1
    assert partial_candidates[0]["id"] == "memorization-000000"

    monkeypatch.setattr(gen_memorization, "_judge_row_async", original_judge)
    result = build(cfg)

    assert result.run_id == run_dir.name
    final_candidates = store.read_jsonl(run_dir / "outputs" / "memorization_candidates.jsonl")
    assert [row["id"] for row in final_candidates].count("memorization-000000") == 1
    assert len(final_candidates) >= 2
