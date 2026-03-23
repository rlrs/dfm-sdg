from __future__ import annotations

from sdg.packs.synth.gen_memorization import retrieve_support_row
from sdg.packs.synth.languages import load_language_plan, row_language_mode


def test_load_language_plan_defaults_to_source_language() -> None:
    cfg = {
        "memory_core": {
            "language": "en",
        },
        "generation": {
            "memorization": {},
        },
    }

    assert load_language_plan(cfg) == {
        "kind": "same_language",
        "source": "en",
        "prompt": "en",
        "reasoning": "en",
        "target": "en",
    }


def test_load_language_plan_supports_cross_language_rows() -> None:
    cfg = {
        "memory_core": {
            "source_language": "en",
        },
        "generation": {
            "memorization": {
                "language_plan": {
                    "prompt": "da",
                    "reasoning": "en",
                    "target": "da",
                }
            },
        },
    }

    assert load_language_plan(cfg) == {
        "kind": "cross_language",
        "source": "en",
        "prompt": "da",
        "reasoning": "en",
        "target": "da",
    }


def test_load_language_plan_reads_family_specific_config() -> None:
    cfg = {
        "memory_core": {
            "source_language": "en",
        },
        "generation": {
            "grounded_qa": {
                "language_plan": {
                    "prompt": "da",
                    "reasoning": "da",
                    "target": "da",
                }
            },
        },
    }

    assert load_language_plan(cfg, family="grounded_qa") == {
        "kind": "cross_language",
        "source": "en",
        "prompt": "da",
        "reasoning": "da",
        "target": "da",
    }


def test_row_language_mode_derives_from_language_fields() -> None:
    row = {
        "meta": {
            "source_language": "en",
            "prompt_language": "da",
            "reasoning_language": "en",
            "target_language": "da",
        }
    }

    assert row_language_mode(row) == "cross_language"


def test_retrieve_support_row_uses_source_side_query_for_cross_language_rows() -> None:
    row = {
        "prompt": "Hvad handler filmen om?",
        "meta": {
            "source_language": "en",
            "prompt_language": "da",
            "reasoning_language": "en",
            "target_language": "da",
        },
        "hidden": {
            "source_id": "2001",
            "source_title": "2001: A Space Odyssey",
            "sentence": "2001: A Space Odyssey is a 1968 epic science fiction film.",
            "task_plan": {
                "coverage_points": [
                    "The film follows a voyage to Jupiter to investigate an alien monolith.",
                ]
            },
            "teacher_bundle": {
                "supporting_claims": [
                    "The film follows a voyage to Jupiter to investigate an alien monolith.",
                ],
                "retrieved_claims": [],
            },
        },
    }
    index = {
        "chunks": {
            "chunk-1": {
                "tokens": ["film", "voyage", "jupiter", "investigate", "alien", "monolith"],
            },
            "chunk-2": {
                "tokens": ["garden", "flowers"],
            },
        }
    }
    chunk_lookup = {
        "chunk-1": {
            "id": "chunk-1",
            "doc_id": "2001",
            "source_id": "2001",
            "title": "2001: A Space Odyssey",
            "text": "The film follows a voyage to Jupiter to investigate an alien monolith.",
            "url": "https://example.com/2001",
            "meta": {"word_count": 11},
        },
        "chunk-2": {
            "id": "chunk-2",
            "doc_id": "flowers",
            "source_id": "flowers",
            "title": "Flowers",
            "text": "Flowers grow in a garden.",
            "url": "https://example.com/flowers",
            "meta": {"word_count": 5},
        },
    }

    updated = retrieve_support_row(row, index, chunk_lookup, settings={"retrieve_top_k": 1})

    assert updated["sources"][0]["source_id"] == "2001"
