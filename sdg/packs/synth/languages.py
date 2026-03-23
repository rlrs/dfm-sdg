from __future__ import annotations

from typing import Literal, TypedDict, cast

from sdg.packs.synth.types import Record

LanguageCode = Literal["en", "da"]
LanguageMode = Literal["same_language", "cross_language"]

LANGUAGE_NAMES: dict[LanguageCode, str] = {
    "en": "English",
    "da": "Danish",
}


class SameLanguagePlan(TypedDict):
    kind: Literal["same_language"]
    source: LanguageCode
    prompt: LanguageCode
    reasoning: LanguageCode
    target: LanguageCode


class CrossLanguagePlan(TypedDict):
    kind: Literal["cross_language"]
    source: LanguageCode
    prompt: LanguageCode
    reasoning: LanguageCode
    target: LanguageCode


LanguagePlan = SameLanguagePlan | CrossLanguagePlan


def language_name(code: LanguageCode) -> str:
    return LANGUAGE_NAMES[code]


def source_language_from_memory_cfg(memory_cfg: Record) -> LanguageCode:
    source_language = memory_cfg.get("source_language")
    legacy_language = memory_cfg.get("language")
    if source_language is not None and legacy_language is not None:
        assert source_language == legacy_language, "memory_core source_language and language must match"

    value = source_language if source_language is not None else legacy_language
    if value is None:
        return "en"
    return _language_code(value, label="memory_core source language")


def load_language_plan(cfg: Record, *, family: str = "memorization") -> LanguagePlan:
    memory_cfg = cfg.get("memory_core", {})
    assert isinstance(memory_cfg, dict), "memory_core config must be a mapping"
    source = source_language_from_memory_cfg(memory_cfg)

    generation_cfg = cfg.get("generation", {})
    assert isinstance(generation_cfg, dict), "generation config must be a mapping"

    family_cfg = generation_cfg.get(family, {})
    assert isinstance(family_cfg, dict), f"{family} config must be a mapping"

    raw_plan = family_cfg.get("language_plan")
    if raw_plan is None:
        return {
            "kind": "same_language",
            "source": source,
            "prompt": source,
            "reasoning": source,
            "target": source,
        }

    assert isinstance(raw_plan, dict), f"{family} language_plan must be a mapping"
    prompt = _required_language(raw_plan, "prompt", label=f"{family} language_plan prompt")
    reasoning = _required_language(raw_plan, "reasoning", label=f"{family} language_plan reasoning")
    target = _required_language(raw_plan, "target", label=f"{family} language_plan target")

    return {
        "kind": language_mode(source, prompt, reasoning, target),
        "source": source,
        "prompt": prompt,
        "reasoning": reasoning,
        "target": target,
    }


def language_mode(
    source: LanguageCode,
    prompt: LanguageCode,
    reasoning: LanguageCode,
    target: LanguageCode,
) -> LanguageMode:
    if prompt == source and reasoning == source and target == source:
        return "same_language"
    return "cross_language"


def row_language_mode(row: Record) -> LanguageMode:
    meta = row.get("meta")
    assert isinstance(meta, dict), "row meta must be a mapping"

    source = _row_language_code(meta, "source_language")
    prompt = _row_language_code(meta, "prompt_language")
    reasoning = _row_language_code(meta, "reasoning_language")
    target = _row_language_code(meta, "target_language")
    if source is not None and prompt is not None and reasoning is not None and target is not None:
        return language_mode(source, prompt, reasoning, target)

    legacy_mode = meta.get("language_mode")
    if legacy_mode is not None:
        assert legacy_mode in {"same_language", "cross_language"}, "row meta language_mode must be supported"
        return cast(LanguageMode, legacy_mode)

    raise AssertionError(
        "row meta must define source_language, prompt_language, reasoning_language, and target_language"
    )


def row_uses_cross_language(row: Record) -> bool:
    return row_language_mode(row) == "cross_language"


def _required_language(record: Record, key: str, *, label: str) -> LanguageCode:
    value = record.get(key)
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    return _language_code(value, label=label)


def _language_code(value: object, *, label: str) -> LanguageCode:
    assert isinstance(value, str) and value, f"{label} must be a non-empty string"
    assert value in LANGUAGE_NAMES, f"Unsupported {label}: {value}"
    return cast(LanguageCode, value)


def _row_language_code(meta: Record, key: str) -> LanguageCode | None:
    value = meta.get(key)
    if value is None:
        return None
    return _language_code(value, label=f"row meta {key}")
