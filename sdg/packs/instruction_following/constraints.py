from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from typing import Any, Literal

LanguageCode = Literal["en", "da"]
ResponseShape = Literal[
    "plain_text",
    "numbered_list",
    "bullet_list",
    "json_object",
    "xml_object",
    "separated_responses",
    "indented_lines",
]

WORD_RE = re.compile(r"[0-9A-Za-zÆØÅæøå]+(?:[-'][0-9A-Za-zÆØÅæøå]+)?", flags=re.UNICODE)
TEXT_SHAPES: tuple[ResponseShape, ...] = (
    "plain_text",
    "numbered_list",
    "bullet_list",
    "separated_responses",
    "indented_lines",
)
ALL_SHAPES: tuple[ResponseShape, ...] = (
    *TEXT_SHAPES,
    "json_object",
    "xml_object",
)

TOPIC_POOL: dict[LanguageCode, tuple[str, ...]] = {
    "en": (
        "a community festival",
        "a new museum exhibit",
        "a local bakery launch",
        "a neighborhood clean-up day",
        "a student research project",
        "a family travel plan",
        "a software team retrospective",
        "a weekly volunteer update",
        "a city library event",
        "a charity fundraiser",
        "a school newsletter item",
        "a product launch announcement",
        "a travel safety reminder",
        "a workshop summary",
        "a hiring process update",
        "a restaurant recommendation",
        "a park renovation plan",
        "a cycling campaign",
        "a town hall meeting",
        "a digital privacy guide",
    ),
    "da": (
        "en lokal festival",
        "en ny museumsudstilling",
        "et bageri, der åbner",
        "en oprydningsdag i kvarteret",
        "et studieprojekt",
        "en familieudflugt",
        "et retrospektiv for et softwareteam",
        "en ugentlig frivilligopdatering",
        "et biblioteksarrangement",
        "en velgørenhedsindsamling",
        "et punkt til skolebladet",
        "en produktlancering",
        "en rejsevejledning",
        "en workshopopsummering",
        "en opdatering om ansættelsesprocessen",
        "en restaurantanbefaling",
        "en plan for renovering af en park",
        "en cykelkampagne",
        "et borgermøde",
        "en guide til digitalt privatliv",
    ),
}

SCENARIO_KIND_POOL: dict[LanguageCode, tuple[str, ...]] = {
    "en": (
        "email",
        "summary",
        "plan",
        "recommendation",
        "announcement",
        "story",
        "briefing note",
        "status update",
        "meeting follow-up",
        "travel note",
    ),
    "da": (
        "e-mail",
        "opsummering",
        "plan",
        "anbefaling",
        "meddelelse",
        "historie",
        "briefingnote",
        "statusopdatering",
        "mødeopfølgning",
        "rejsebesked",
    ),
}

KEYWORD_POOL: dict[LanguageCode, tuple[str, ...]] = {
    "en": (
        "amber",
        "bridge",
        "candle",
        "cloud",
        "forest",
        "garden",
        "harbor",
        "lantern",
        "meadow",
        "pocket",
        "river",
        "signal",
        "silver",
        "window",
        "market",
        "sunrise",
        "library",
        "harvest",
    ),
    "da": (
        "bro",
        "eng",
        "flod",
        "have",
        "havn",
        "høst",
        "lanterne",
        "lomme",
        "signal",
        "sky",
        "skov",
        "sølv",
        "vindue",
        "marked",
        "solopgang",
        "bibliotek",
        "lys",
        "sti",
    ),
}

FORBIDDEN_WORD_POOL: dict[LanguageCode, tuple[str, ...]] = {
    "en": ("banana", "violet", "rocket", "mirror", "thunder", "copper", "velvet", "desert"),
    "da": ("banan", "violet", "raket", "spejl", "torden", "kobber", "fløjl", "ørken"),
}

END_PHRASE_POOL: dict[LanguageCode, tuple[str, ...]] = {
    "en": (
        "that is the full answer.",
        "this is the final version.",
        "nothing else needs to be added.",
        "that completes the response.",
    ),
    "da": (
        "det er hele svaret.",
        "dette er den endelige version.",
        "der skal ikke tilføjes mere.",
        "det afslutter svaret.",
    ),
}

JSON_KEY_POOL: dict[LanguageCode, tuple[str, ...]] = {
    "en": ("answer", "reason", "title", "summary", "status", "next_step"),
    "da": ("svar", "grund", "titel", "opsummering", "status", "næste_skridt"),
}

XML_TAG_POOL: dict[LanguageCode, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "en": (
        ("response", ("title", "summary")),
        ("note", ("topic", "answer")),
        ("report", ("status", "detail")),
        ("update", ("headline", "body")),
    ),
    "da": (
        ("svar", ("titel", "opsummering")),
        ("note", ("emne", "svar")),
        ("rapport", ("status", "detalje")),
        ("opdatering", ("overskrift", "tekst")),
    ),
}

SampleFn = Callable[[Random, LanguageCode], dict[str, Any]]
RenderFn = Callable[[dict[str, Any], LanguageCode], str]
CheckFn = Callable[[str, dict[str, Any], LanguageCode], bool]


@dataclass(frozen=True)
class ConstraintDefinition:
    constraint_id: str
    category: str
    supported_languages: tuple[LanguageCode, ...]
    compatible_shapes: tuple[ResponseShape, ...]
    required_shape: ResponseShape | None
    exclusive_groups: tuple[str, ...]
    sample_params: SampleFn
    render: RenderFn
    check: CheckFn


def available_languages() -> tuple[LanguageCode, ...]:
    return ("en", "da")


def available_shapes() -> tuple[ResponseShape, ...]:
    return ALL_SHAPES


def topic_pool(language: LanguageCode) -> tuple[str, ...]:
    return TOPIC_POOL[language]


def scenario_kind_pool(language: LanguageCode) -> tuple[str, ...]:
    return SCENARIO_KIND_POOL[language]


def available_constraints(
    *,
    language: LanguageCode,
    shape: ResponseShape,
) -> tuple[ConstraintDefinition, ...]:
    return tuple(
        definition
        for definition in CONSTRAINTS.values()
        if language in definition.supported_languages
        and shape in definition.compatible_shapes
    )


def sample_constraints(
    rng: Random,
    *,
    language: LanguageCode,
    shape: ResponseShape,
    min_count: int,
    max_count: int,
    prompt_seed: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    definitions = list(available_constraints(language=language, shape=shape))
    required = [definition for definition in definitions if definition.required_shape == shape]
    extras = [
        definition
        for definition in definitions
        if definition.required_shape is None
        and _prompt_allows_constraint(definition, prompt_seed)
    ]

    assert len(required) <= 1, f"expected at most one structural constraint for shape {shape}"

    minimum = max(min_count, len(required))
    maximum = max(max_count, minimum)
    desired = rng.randint(minimum, maximum)

    chosen = list(required)
    rng.shuffle(extras)
    for definition in extras:
        if len(chosen) >= desired:
            break
        if not _can_add_constraint(definition, chosen):
            continue
        chosen.append(definition)

    return [
        {
            "id": definition.constraint_id,
            "params": _sample_constraint_params(
                definition,
                rng,
                language=language,
                prompt_seed=prompt_seed,
            ),
        }
        for definition in chosen
    ]


def render_constraint_line(constraint: dict[str, Any], *, language: LanguageCode) -> str:
    definition = constraint_definition(str(constraint["id"]))
    return definition.render(dict(constraint.get("params", {})), language)


def render_constraint_lines(constraints: list[dict[str, Any]], *, language: LanguageCode) -> list[str]:
    return [render_constraint_line(constraint, language=language) for constraint in constraints]


def check_constraints_strict(
    response: str,
    constraints: list[dict[str, Any]],
    *,
    language: LanguageCode,
) -> list[bool]:
    return [
        constraint_definition(str(constraint["id"])).check(response, dict(constraint.get("params", {})), language)
        for constraint in constraints
    ]


def check_constraints_loose(
    response: str,
    constraints: list[dict[str, Any]],
    *,
    language: LanguageCode,
) -> list[bool]:
    variants = loose_response_variants(response)
    results: list[bool] = []
    for constraint in constraints:
        definition = constraint_definition(str(constraint["id"]))
        params = dict(constraint.get("params", {}))
        passed = any(
            variant.strip() and definition.check(variant, params, language)
            for variant in variants
        )
        results.append(passed)
    return results


def constraint_categories(constraints: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for constraint in constraints:
        category = constraint_definition(str(constraint["id"])).category
        if category not in seen:
            seen.append(category)
    return seen


def constraint_definition(constraint_id: str) -> ConstraintDefinition:
    assert constraint_id in CONSTRAINTS, f"unknown constraint: {constraint_id}"
    return CONSTRAINTS[constraint_id]


def loose_response_variants(response: str) -> list[str]:
    if response is None:
        return [""]

    lines = str(response).splitlines()
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()

    variants = [
        str(response),
        str(response).replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]

    deduped: list[str] = []
    for variant in variants:
        if variant in deduped:
            continue
        deduped.append(variant)
    return deduped


def _can_add_constraint(definition: ConstraintDefinition, chosen: list[ConstraintDefinition]) -> bool:
    candidate_groups = set(definition.exclusive_groups)
    for current in chosen:
        if candidate_groups.intersection(current.exclusive_groups):
            return False
    return True


def _prompt_allows_constraint(
    definition: ConstraintDefinition,
    prompt_seed: dict[str, Any] | None,
) -> bool:
    if not isinstance(prompt_seed, dict):
        return True

    constraint_id = definition.constraint_id
    rigidity = str(prompt_seed.get("semantic_rigidity", "medium"))
    numeric_task = bool(prompt_seed.get("numeric_task"))
    preserve_literal_source_text = bool(prompt_seed.get("preserve_literal_source_text"))
    sentence_count = _prompt_seed_count(prompt_seed, "requested_sentence_count")
    line_count = _prompt_seed_count(prompt_seed, "requested_line_count")
    item_count = _prompt_seed_count(prompt_seed, "requested_item_count")
    keyword_count = len(_prompt_keyword_candidates(prompt_seed))

    if preserve_literal_source_text and constraint_id in {
        "count:keywords_multiple",
        "start_end:first_word",
        "start_end:last_word",
        "words:forbidden_words",
        "words:ordered_keywords",
        "words:word_positions",
    }:
        return False

    if constraint_id == "words:word_positions":
        return rigidity == "open" and keyword_count >= 2
    if constraint_id == "count:word_count_range":
        return rigidity == "open" and sentence_count is None and line_count is None and item_count is None
    if constraint_id == "count:keywords_multiple":
        return rigidity != "rigid" and keyword_count >= 2
    if constraint_id == "words:ordered_keywords":
        return keyword_count >= 2 and not (rigidity == "rigid" and numeric_task)
    if constraint_id in {"start_end:first_word", "start_end:last_word"}:
        return keyword_count >= 1 and not numeric_task
    if constraint_id == "start_end:end_phrase":
        return not (numeric_task and rigidity == "rigid")
    if constraint_id == "format:no_digits":
        return not numeric_task
    if constraint_id == "count:sentences":
        return sentence_count is None
    if constraint_id == "count:paragraphs":
        return sentence_count is None and line_count is None
    if constraint_id == "count:line_count":
        return line_count is None and item_count is None
    if constraint_id == "format:quoted_spans":
        return rigidity != "rigid"
    if constraint_id == "punctuation:semicolon_count":
        return rigidity == "open"
    return True


def _sample_constraint_params(
    definition: ConstraintDefinition,
    rng: Random,
    *,
    language: LanguageCode,
    prompt_seed: dict[str, Any] | None,
) -> dict[str, Any]:
    constraint_id = definition.constraint_id
    keywords = _prompt_keyword_candidates(prompt_seed)

    if constraint_id == "count:keywords_multiple" and len(keywords) >= 2:
        return _sample_keywords_multiple_from_prompt(rng, keywords)
    if constraint_id == "words:ordered_keywords" and len(keywords) >= 2:
        return _sample_ordered_keywords_from_prompt(rng, keywords)
    if constraint_id == "start_end:first_word" and keywords:
        return {"word": rng.choice(keywords)}
    if constraint_id == "start_end:last_word" and keywords:
        return {"word": rng.choice(keywords)}
    if constraint_id == "words:word_positions" and len(keywords) >= 2:
        return _sample_word_positions_from_prompt(rng, keywords)
    if constraint_id == "words:forbidden_words":
        return _sample_forbidden_words_with_prompt(rng, language, prompt_seed)
    if constraint_id == "format:bullet_list":
        return _sample_bullet_list_with_prompt(rng, prompt_seed)
    if constraint_id == "format:numbered_list":
        return _sample_numbered_list_with_prompt(rng, prompt_seed)
    return definition.sample_params(rng, language)


def _prompt_seed_count(prompt_seed: dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(prompt_seed, dict):
        return None
    value = prompt_seed.get(key)
    if value is None:
        return None
    assert isinstance(value, int) and value > 0, f"prompt seed {key} must be a positive integer"
    return value


def _prompt_keyword_candidates(prompt_seed: dict[str, Any] | None) -> list[str]:
    if not isinstance(prompt_seed, dict):
        return []
    raw = prompt_seed.get("semantic_keywords")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str) and item]


def _sample_keywords_multiple(rng: Random, language: LanguageCode) -> dict[str, Any]:
    count = rng.choice((2, 3))
    keywords = rng.sample(KEYWORD_POOL[language], count)
    frequencies = list(range(1, count + 1))
    if count == 3 and rng.random() < 0.5:
        frequencies[-1] = rng.choice((3, 4))
    return {"keywords": [{"word": word, "count": freq} for word, freq in zip(keywords, frequencies, strict=True)]}


def _sample_keywords_multiple_from_prompt(rng: Random, keywords: list[str]) -> dict[str, Any]:
    count = 2 if len(keywords) == 2 else rng.choice((2, 3))
    chosen = rng.sample(keywords, min(count, len(keywords)))
    frequencies = list(range(1, len(chosen) + 1))
    return {"keywords": [{"word": word, "count": freq} for word, freq in zip(chosen, frequencies, strict=True)]}


def _render_keywords_multiple(params: dict[str, Any], language: LanguageCode) -> str:
    pairs = list(params["keywords"])
    parts = [
        (
            f'use the word "{pair["word"]}" exactly {pair["count"]} times'
            if language == "en"
            else f'brug ordet "{pair["word"]}" præcis {pair["count"]} gange'
        )
        for pair in pairs
    ]
    if language == "en":
        return _join_parts(parts, final_joiner="and").capitalize() + "."
    return _join_parts(parts, final_joiner="og").capitalize() + "."


def _check_keywords_multiple(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    tokens = [token.casefold() for token in _word_tokens(response)]
    return all(tokens.count(str(pair["word"]).casefold()) == int(pair["count"]) for pair in params["keywords"])


def _sample_forbidden_words(rng: Random, language: LanguageCode) -> dict[str, Any]:
    count = rng.choice((1, 2))
    return {"words": sorted(rng.sample(FORBIDDEN_WORD_POOL[language], count))}


def _sample_forbidden_words_with_prompt(
    rng: Random,
    language: LanguageCode,
    prompt_seed: dict[str, Any] | None,
) -> dict[str, Any]:
    prompt_keywords = set(_prompt_keyword_candidates(prompt_seed))
    candidates = [word for word in FORBIDDEN_WORD_POOL[language] if word not in prompt_keywords]
    if len(candidates) < 2:
        candidates = list(FORBIDDEN_WORD_POOL[language])
    count = rng.choice((1, 2))
    return {"words": sorted(rng.sample(candidates, count))}


def _render_forbidden_words(params: dict[str, Any], language: LanguageCode) -> str:
    words = [f'"{word}"' for word in params["words"]]
    joined = _join_parts(words, final_joiner="and" if language == "en" else "og")
    if language == "en":
        return f"Do not use the word {joined}."
    return f"Brug ikke ordet {joined}."


def _check_forbidden_words(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    tokens = {token.casefold() for token in _word_tokens(response)}
    forbidden = {str(word).casefold() for word in params["words"]}
    return tokens.isdisjoint(forbidden)


def _sample_no_digits(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del rng, language
    return {}


def _render_no_digits(params: dict[str, Any], language: LanguageCode) -> str:
    del params
    if language == "en":
        return "Do not use any Arabic numerals."
    return "Brug ikke arabiske tal."


def _check_no_digits(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del params, language
    return re.search(r"\d", response) is None


def _sample_all_lowercase(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del rng, language
    return {}


def _render_all_lowercase(params: dict[str, Any], language: LanguageCode) -> str:
    del params
    if language == "en":
        return "Write the entire response in lowercase."
    return "Skriv hele svaret med små bogstaver."


def _check_all_lowercase(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del params, language
    letters = [char for char in response if char.isalpha()]
    return all(char == char.lower() for char in letters)


def _sample_word_count_range(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    minimum = rng.choice((18, 20, 24))
    maximum = minimum + rng.choice((4, 6, 8))
    return {"min_words": minimum, "max_words": maximum}


def _render_word_count_range(params: dict[str, Any], language: LanguageCode) -> str:
    minimum = int(params["min_words"])
    maximum = int(params["max_words"])
    if language == "en":
        return f"Use between {minimum} and {maximum} words."
    return f"Brug mellem {minimum} og {maximum} ord."


def _check_word_count_range(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    count = len(_word_tokens(response))
    return int(params["min_words"]) <= count <= int(params["max_words"])


def _sample_sentence_count(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    return {"count": rng.choice((2, 3, 4))}


def _render_sentence_count(params: dict[str, Any], language: LanguageCode) -> str:
    count = int(params["count"])
    if language == "en":
        return f"Write exactly {count} sentences."
    return f"Skriv præcis {count} sætninger."


def _check_sentence_count(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    return len(_sentence_segments(response)) == int(params["count"])


def _sample_paragraph_count(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    return {"count": rng.choice((2, 3))}


def _render_paragraph_count(params: dict[str, Any], language: LanguageCode) -> str:
    count = int(params["count"])
    if language == "en":
        return f"Use exactly {count} paragraphs."
    return f"Brug præcis {count} afsnit."


def _check_paragraph_count(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    return len(_paragraphs(response)) == int(params["count"])


def _sample_line_count(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    return {"count": rng.choice((3, 4, 5))}


def _render_line_count(params: dict[str, Any], language: LanguageCode) -> str:
    count = int(params["count"])
    if language == "en":
        return f"Use exactly {count} non-empty lines."
    return f"Brug præcis {count} ikke-tomme linjer."


def _check_line_count(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    return len(_non_empty_lines(response)) == int(params["count"])


def _sample_first_word(rng: Random, language: LanguageCode) -> dict[str, Any]:
    return {"word": rng.choice(KEYWORD_POOL[language])}


def _render_first_word(params: dict[str, Any], language: LanguageCode) -> str:
    word = str(params["word"])
    if language == "en":
        return f'Start the response with the word "{word}".'
    return f'Begynd svaret med ordet "{word}".'


def _check_first_word(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    tokens = _word_tokens(response)
    if not tokens:
        return False
    return tokens[0].casefold() == str(params["word"]).casefold()


def _sample_end_phrase(rng: Random, language: LanguageCode) -> dict[str, Any]:
    return {"phrase": rng.choice(END_PHRASE_POOL[language])}


def _render_end_phrase(params: dict[str, Any], language: LanguageCode) -> str:
    phrase = str(params["phrase"])
    if language == "en":
        return f'End the response with "{phrase}".'
    return f'Afslut svaret med "{phrase}".'


def _check_end_phrase(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    return response.strip().endswith(str(params["phrase"]))


def _sample_last_word(rng: Random, language: LanguageCode) -> dict[str, Any]:
    return {"word": rng.choice(KEYWORD_POOL[language])}


def _render_last_word(params: dict[str, Any], language: LanguageCode) -> str:
    word = str(params["word"])
    if language == "en":
        return f'End the response with the word "{word}".'
    return f'Afslut svaret med ordet "{word}".'


def _check_last_word(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    tokens = _word_tokens(response)
    if not tokens:
        return False
    return tokens[-1].casefold() == str(params["word"]).casefold()


def _sample_numbered_list(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    return {"count": rng.choice((3, 4))}


def _sample_numbered_list_with_prompt(rng: Random, prompt_seed: dict[str, Any] | None) -> dict[str, Any]:
    requested_count = _prompt_seed_count(prompt_seed, "requested_item_count")
    if requested_count is not None and 2 <= requested_count <= 5:
        return {"count": requested_count}
    return {"count": rng.choice((3, 4))}


def _render_numbered_list(params: dict[str, Any], language: LanguageCode) -> str:
    count = int(params["count"])
    if language == "en":
        return f"Return a numbered list with exactly {count} items."
    return f"Returner en nummereret liste med præcis {count} punkter."


def _check_numbered_list(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    lines = [line.strip() for line in _non_empty_lines(response)]
    count = int(params["count"])
    if len(lines) != count:
        return False
    expected_prefixes = [f"{index}." for index in range(1, count + 1)]
    return all(line.startswith(prefix) for line, prefix in zip(lines, expected_prefixes, strict=True))


def _sample_bullet_list(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    return {"count": rng.choice((3, 4, 5))}


def _sample_bullet_list_with_prompt(rng: Random, prompt_seed: dict[str, Any] | None) -> dict[str, Any]:
    requested_count = _prompt_seed_count(prompt_seed, "requested_item_count")
    if requested_count is not None and 2 <= requested_count <= 5:
        return {"count": requested_count}
    return {"count": rng.choice((3, 4, 5))}


def _render_bullet_list(params: dict[str, Any], language: LanguageCode) -> str:
    count = int(params["count"])
    if language == "en":
        return f"Return a bullet list with exactly {count} items using '-' markers."
    return f"Returner en punktopstilling med præcis {count} punkter med '-' som markør."


def _check_bullet_list(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    lines = [line.rstrip() for line in _non_empty_lines(response)]
    count = int(params["count"])
    return len(lines) == count and all(line.lstrip().startswith("- ") for line in lines)


def _sample_json_keys(rng: Random, language: LanguageCode) -> dict[str, Any]:
    count = rng.choice((2, 3))
    return {"keys": list(rng.sample(JSON_KEY_POOL[language], count))}


def _render_json_keys(params: dict[str, Any], language: LanguageCode) -> str:
    keys = [f'"{key}"' for key in params["keys"]]
    joined = _join_parts(keys, final_joiner="and" if language == "en" else "og")
    if language == "en":
        return f"Return valid JSON with exactly these top-level keys: {joined}."
    return f"Returner gyldig JSON med præcis disse nøgler på topniveau: {joined}."


def _check_json_keys(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return False
    if not isinstance(parsed, dict):
        return False
    return set(parsed) == set(params["keys"])


def _sample_xml_tags(rng: Random, language: LanguageCode) -> dict[str, Any]:
    root_tag, child_tags = rng.choice(XML_TAG_POOL[language])
    return {"root_tag": root_tag, "child_tags": list(child_tags)}


def _render_xml_tags(params: dict[str, Any], language: LanguageCode) -> str:
    root_tag = str(params["root_tag"])
    child_tags = [f"<{tag}>" for tag in params["child_tags"]]
    joined = _join_parts(child_tags, final_joiner="and" if language == "en" else "og")
    if language == "en":
        return f"Return valid XML with root tag <{root_tag}> and exactly these direct child tags: {joined}."
    return f"Returner gyldig XML med rodtag <{root_tag}> og præcis disse direkte child-tags: {joined}."


def _check_xml_tags(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    try:
        root = ET.fromstring(response)
    except ET.ParseError:
        return False
    child_tags = [child.tag for child in list(root)]
    return root.tag == str(params["root_tag"]) and child_tags == list(params["child_tags"])


def _sample_two_responses(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del rng, language
    return {"separator": "******"}


def _render_two_responses(params: dict[str, Any], language: LanguageCode) -> str:
    separator = str(params["separator"])
    if language == "en":
        return f'Return exactly two non-empty sections separated by "{separator}".'
    return f'Returner præcis to ikke-tomme sektioner adskilt af "{separator}".'


def _check_two_responses(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    separator = str(params["separator"])
    if response.count(separator) != 1:
        return False
    left, right = response.split(separator)
    return bool(left.strip() and right.strip())


def _sample_word_positions(rng: Random, language: LanguageCode) -> dict[str, Any]:
    words = rng.sample(KEYWORD_POOL[language], 2)
    positions = (3, 8)
    return {
        "pairs": [
            {"position": positions[0], "word": words[0]},
            {"position": positions[1], "word": words[1]},
        ]
    }


def _sample_word_positions_from_prompt(rng: Random, keywords: list[str]) -> dict[str, Any]:
    words = rng.sample(keywords, 2)
    return {
        "pairs": [
            {"position": 3, "word": words[0]},
            {"position": 8, "word": words[1]},
        ]
    }


def _render_word_positions(params: dict[str, Any], language: LanguageCode) -> str:
    pairs = list(params["pairs"])
    fragments = [
        (
            f'make the {pair["position"]}{_ordinal_suffix(int(pair["position"]))} word "{pair["word"]}"'
            if language == "en"
            else f'lad ord nummer {pair["position"]} være "{pair["word"]}"'
        )
        for pair in pairs
    ]
    if language == "en":
        return _join_parts(fragments, final_joiner="and").capitalize() + "."
    return _join_parts(fragments, final_joiner="og").capitalize() + "."


def _check_word_positions(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    tokens = _word_tokens(response)
    for pair in params["pairs"]:
        index = int(pair["position"]) - 1
        if index >= len(tokens):
            return False
        if tokens[index].casefold() != str(pair["word"]).casefold():
            return False
    return True


def _sample_line_indent(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    line_count = rng.choice((3, 4))
    return {"line_count": line_count, "step": 2}


def _render_line_indent(params: dict[str, Any], language: LanguageCode) -> str:
    line_count = int(params["line_count"])
    if language == "en":
        return (
            f"Use exactly {line_count} non-empty lines, with each new line indented two spaces more than the previous line."
        )
    return (
        f"Brug præcis {line_count} ikke-tomme linjer, hvor hver ny linje har to ekstra indryk i forhold til den forrige."
    )


def _check_line_indent(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    lines = _non_empty_lines(response)
    line_count = int(params["line_count"])
    step = int(params["step"])
    if len(lines) != line_count:
        return False

    expected_indent = 0
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        if indent != expected_indent:
            return False
        expected_indent += step
    return True


def _sample_no_commas(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del rng, language
    return {}


def _render_no_commas(params: dict[str, Any], language: LanguageCode) -> str:
    del params
    if language == "en":
        return "Do not use any commas."
    return "Brug ingen kommaer."


def _check_no_commas(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del params, language
    return "," not in response


def _sample_semicolon_count(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    return {"count": rng.choice((1, 2))}


def _render_semicolon_count(params: dict[str, Any], language: LanguageCode) -> str:
    count = int(params["count"])
    if language == "en":
        return f"Use exactly {count} semicolons."
    return f"Brug præcis {count} semikoloner."


def _check_semicolon_count(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    return response.count(";") == int(params["count"])


def _sample_ordered_keywords(rng: Random, language: LanguageCode) -> dict[str, Any]:
    first, second = rng.sample(KEYWORD_POOL[language], 2)
    return {"first": first, "second": second}


def _sample_ordered_keywords_from_prompt(rng: Random, keywords: list[str]) -> dict[str, Any]:
    first, second = rng.sample(keywords, 2)
    return {"first": first, "second": second}


def _render_ordered_keywords(params: dict[str, Any], language: LanguageCode) -> str:
    first = str(params["first"])
    second = str(params["second"])
    if language == "en":
        return f'Include both "{first}" and "{second}", and mention "{first}" before "{second}".'
    return f'Brug både "{first}" og "{second}", og nævn "{first}" før "{second}".'


def _check_ordered_keywords(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    tokens = [token.casefold() for token in _word_tokens(response)]
    first = str(params["first"]).casefold()
    second = str(params["second"]).casefold()
    if first not in tokens or second not in tokens:
        return False
    return tokens.index(first) < tokens.index(second)


def _sample_quoted_spans(rng: Random, language: LanguageCode) -> dict[str, Any]:
    del language
    return {"count": rng.choice((1, 2))}


def _render_quoted_spans(params: dict[str, Any], language: LanguageCode) -> str:
    count = int(params["count"])
    if language == "en":
        return f"Include exactly {count} double-quoted spans."
    return f"Brug præcis {count} dobbelte citationer."


def _check_quoted_spans(response: str, params: dict[str, Any], language: LanguageCode) -> bool:
    del language
    spans = re.findall(r'"[^"\n]+"', response)
    return len(spans) == int(params["count"])


def _join_parts(parts: list[str], *, final_joiner: str) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} {final_joiner} {parts[1]}"
    return f'{", ".join(parts[:-1])}, {final_joiner} {parts[-1]}'


def _word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]


def _sentence_segments(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text.strip()) if segment.strip()]


def _non_empty_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def _ordinal_suffix(number: int) -> str:
    if 10 <= number % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")


CONSTRAINTS: dict[str, ConstraintDefinition] = {
    "count:keywords_multiple": ConstraintDefinition(
        constraint_id="count:keywords_multiple",
        category="keywords",
        supported_languages=("en", "da"),
        compatible_shapes=TEXT_SHAPES,
        required_shape=None,
        exclusive_groups=(),
        sample_params=_sample_keywords_multiple,
        render=_render_keywords_multiple,
        check=_check_keywords_multiple,
    ),
    "words:forbidden_words": ConstraintDefinition(
        constraint_id="words:forbidden_words",
        category="keywords",
        supported_languages=("en", "da"),
        compatible_shapes=TEXT_SHAPES,
        required_shape=None,
        exclusive_groups=(),
        sample_params=_sample_forbidden_words,
        render=_render_forbidden_words,
        check=_check_forbidden_words,
    ),
    "format:no_digits": ConstraintDefinition(
        constraint_id="format:no_digits",
        category="format",
        supported_languages=("en", "da"),
        compatible_shapes=("plain_text", "bullet_list", "json_object", "xml_object", "separated_responses", "indented_lines"),
        required_shape=None,
        exclusive_groups=(),
        sample_params=_sample_no_digits,
        render=_render_no_digits,
        check=_check_no_digits,
    ),
    "change_case:all_lowercase": ConstraintDefinition(
        constraint_id="change_case:all_lowercase",
        category="case",
        supported_languages=("en", "da"),
        compatible_shapes=ALL_SHAPES,
        required_shape=None,
        exclusive_groups=("case",),
        sample_params=_sample_all_lowercase,
        render=_render_all_lowercase,
        check=_check_all_lowercase,
    ),
    "count:word_count_range": ConstraintDefinition(
        constraint_id="count:word_count_range",
        category="length",
        supported_languages=("en", "da"),
        compatible_shapes=TEXT_SHAPES,
        required_shape=None,
        exclusive_groups=("length",),
        sample_params=_sample_word_count_range,
        render=_render_word_count_range,
        check=_check_word_count_range,
    ),
    "count:sentences": ConstraintDefinition(
        constraint_id="count:sentences",
        category="length",
        supported_languages=("en", "da"),
        compatible_shapes=("plain_text",),
        required_shape=None,
        exclusive_groups=("length",),
        sample_params=_sample_sentence_count,
        render=_render_sentence_count,
        check=_check_sentence_count,
    ),
    "count:paragraphs": ConstraintDefinition(
        constraint_id="count:paragraphs",
        category="length",
        supported_languages=("en", "da"),
        compatible_shapes=("plain_text",),
        required_shape=None,
        exclusive_groups=("length",),
        sample_params=_sample_paragraph_count,
        render=_render_paragraph_count,
        check=_check_paragraph_count,
    ),
    "count:line_count": ConstraintDefinition(
        constraint_id="count:line_count",
        category="length",
        supported_languages=("en", "da"),
        compatible_shapes=("numbered_list", "bullet_list", "indented_lines"),
        required_shape=None,
        exclusive_groups=("line_measure",),
        sample_params=_sample_line_count,
        render=_render_line_count,
        check=_check_line_count,
    ),
    "start_end:first_word": ConstraintDefinition(
        constraint_id="start_end:first_word",
        category="position",
        supported_languages=("en", "da"),
        compatible_shapes=TEXT_SHAPES,
        required_shape=None,
        exclusive_groups=(),
        sample_params=_sample_first_word,
        render=_render_first_word,
        check=_check_first_word,
    ),
    "start_end:end_phrase": ConstraintDefinition(
        constraint_id="start_end:end_phrase",
        category="position",
        supported_languages=("en", "da"),
        compatible_shapes=TEXT_SHAPES,
        required_shape=None,
        exclusive_groups=("ending",),
        sample_params=_sample_end_phrase,
        render=_render_end_phrase,
        check=_check_end_phrase,
    ),
    "start_end:last_word": ConstraintDefinition(
        constraint_id="start_end:last_word",
        category="position",
        supported_languages=("en", "da"),
        compatible_shapes=TEXT_SHAPES,
        required_shape=None,
        exclusive_groups=("ending",),
        sample_params=_sample_last_word,
        render=_render_last_word,
        check=_check_last_word,
    ),
    "format:numbered_list": ConstraintDefinition(
        constraint_id="format:numbered_list",
        category="format",
        supported_languages=("en", "da"),
        compatible_shapes=("numbered_list",),
        required_shape="numbered_list",
        exclusive_groups=("line_measure",),
        sample_params=_sample_numbered_list,
        render=_render_numbered_list,
        check=_check_numbered_list,
    ),
    "format:bullet_list": ConstraintDefinition(
        constraint_id="format:bullet_list",
        category="format",
        supported_languages=("en", "da"),
        compatible_shapes=("bullet_list",),
        required_shape="bullet_list",
        exclusive_groups=("line_measure",),
        sample_params=_sample_bullet_list,
        render=_render_bullet_list,
        check=_check_bullet_list,
    ),
    "format:json_keys": ConstraintDefinition(
        constraint_id="format:json_keys",
        category="format",
        supported_languages=("en", "da"),
        compatible_shapes=("json_object",),
        required_shape="json_object",
        exclusive_groups=(),
        sample_params=_sample_json_keys,
        render=_render_json_keys,
        check=_check_json_keys,
    ),
    "format:xml_tags": ConstraintDefinition(
        constraint_id="format:xml_tags",
        category="format",
        supported_languages=("en", "da"),
        compatible_shapes=("xml_object",),
        required_shape="xml_object",
        exclusive_groups=(),
        sample_params=_sample_xml_tags,
        render=_render_xml_tags,
        check=_check_xml_tags,
    ),
    "format:two_responses": ConstraintDefinition(
        constraint_id="format:two_responses",
        category="format",
        supported_languages=("en", "da"),
        compatible_shapes=("separated_responses",),
        required_shape="separated_responses",
        exclusive_groups=(),
        sample_params=_sample_two_responses,
        render=_render_two_responses,
        check=_check_two_responses,
    ),
    "words:word_positions": ConstraintDefinition(
        constraint_id="words:word_positions",
        category="position",
        supported_languages=("en", "da"),
        compatible_shapes=("plain_text",),
        required_shape=None,
        exclusive_groups=("position_exact",),
        sample_params=_sample_word_positions,
        render=_render_word_positions,
        check=_check_word_positions,
    ),
    "format:line_indent": ConstraintDefinition(
        constraint_id="format:line_indent",
        category="format",
        supported_languages=("en", "da"),
        compatible_shapes=("indented_lines",),
        required_shape="indented_lines",
        exclusive_groups=("line_measure",),
        sample_params=_sample_line_indent,
        render=_render_line_indent,
        check=_check_line_indent,
    ),
    "punctuation:no_commas": ConstraintDefinition(
        constraint_id="punctuation:no_commas",
        category="punctuation",
        supported_languages=("en", "da"),
        compatible_shapes=TEXT_SHAPES,
        required_shape=None,
        exclusive_groups=(),
        sample_params=_sample_no_commas,
        render=_render_no_commas,
        check=_check_no_commas,
    ),
    "punctuation:semicolon_count": ConstraintDefinition(
        constraint_id="punctuation:semicolon_count",
        category="punctuation",
        supported_languages=("en", "da"),
        compatible_shapes=("plain_text", "separated_responses"),
        required_shape=None,
        exclusive_groups=(),
        sample_params=_sample_semicolon_count,
        render=_render_semicolon_count,
        check=_check_semicolon_count,
    ),
    "words:ordered_keywords": ConstraintDefinition(
        constraint_id="words:ordered_keywords",
        category="keywords",
        supported_languages=("en", "da"),
        compatible_shapes=TEXT_SHAPES,
        required_shape=None,
        exclusive_groups=(),
        sample_params=_sample_ordered_keywords,
        render=_render_ordered_keywords,
        check=_check_ordered_keywords,
    ),
    "format:quoted_spans": ConstraintDefinition(
        constraint_id="format:quoted_spans",
        category="format",
        supported_languages=("en", "da"),
        compatible_shapes=("plain_text", "separated_responses"),
        required_shape=None,
        exclusive_groups=(),
        sample_params=_sample_quoted_spans,
        render=_render_quoted_spans,
        check=_check_quoted_spans,
    ),
}
