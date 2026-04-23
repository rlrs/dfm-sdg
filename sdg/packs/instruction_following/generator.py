from __future__ import annotations

from typing import Any

from sdg.commons.model import LLM
from sdg.packs.synth.llm_json import achat_json, chat_json

from .constraints import LanguageCode, ResponseShape

TASK_TYPES: tuple[str, ...] = (
    "analysis",
    "classification",
    "comparison",
    "creative",
    "explanation",
    "general_generation",
    "listing",
    "question_answering",
    "recommendation",
    "rewrite",
    "summarization",
    "translation",
)
RIGIDITY_LEVELS = {"open", "medium", "rigid"}
NATURALNESS_LEVELS = {"low", "medium", "high"}

BLOCK_INSTRUCTION_SURFACES = {
    "en": {
        "additional_requirements": "Additional requirements:",
        "follow_these_requirements": "Follow these requirements:",
        "answer_must_satisfy": "Your answer must satisfy the following:",
        "keep_in_mind": "Keep the following in mind:",
        "requirements_for_answer": "Requirements for the answer:",
    },
    "da": {
        "yderligere_krav": "Yderligere krav:",
        "foelg_disse_krav": "Følg disse krav:",
        "dit_svar_skal_opfylde": "Dit svar skal opfylde følgende:",
        "husk_foelgende": "Husk følgende krav:",
        "krav_til_svaret": "Krav til svaret:",
    },
}

INLINE_INSTRUCTION_SURFACES = {
    "en": {
        "answer_should_also": "The answer should also satisfy the following requirements:",
        "make_sure_answer": "Make sure the answer follows these requirements:",
        "keep_answer_within": "Keep the answer within these constraints:",
        "answer_needs_to": "The answer also needs to satisfy the following:",
    },
    "da": {
        "svaret_skal_ogsaa": "Svaret skal også opfylde følgende:",
        "soerg_for_svaret": "Sørg også for, at svaret opfylder følgende:",
        "hold_svaret_indenfor": "Hold også svaret inden for disse krav:",
        "svaret_skal_samtidig": "Svaret skal samtidig følge disse krav:",
    },
}

FOLLOW_UP_SURFACES = {
    "en": {
        "rewrite_requirements": "Rewrite your response and follow these requirements.",
        "answer_again_requirements": "Answer again, but make sure your response satisfies these requirements.",
        "new_version_constraints": "Write a new version of your response so that it meets these requirements.",
        "revise_response": "Revise your response so it follows these requirements.",
    },
    "da": {
        "rewrite_requirements": "Skriv svaret om, og følg disse krav.",
        "answer_again_requirements": "Svar igen, men sørg for, at svaret opfylder disse krav.",
        "new_version_constraints": "Skriv en ny version af svaret, så det opfylder disse krav.",
        "revise_response": "Lav en revideret version af svaret, der følger disse krav.",
    },
}


def instruction_surface_keys(language: LanguageCode) -> list[str]:
    block = [f"block:{key}" for key in BLOCK_INSTRUCTION_SURFACES[language]]
    inline = [f"inline:{key}" for key in INLINE_INSTRUCTION_SURFACES[language]]
    return [*block, *inline]


def follow_up_surface_keys(language: LanguageCode) -> list[str]:
    return list(FOLLOW_UP_SURFACES[language])


def render_instruction_block(
    language: LanguageCode,
    lines: list[str],
    *,
    surface_key: str | None = None,
) -> str:
    layout, key = _instruction_surface(language, surface_key)
    if layout == "inline":
        intro = INLINE_INSTRUCTION_SURFACES[language][key]
        body = " ".join(_instruction_sentence(line) for line in lines)
        return f"{intro} {body}"

    heading = BLOCK_INSTRUCTION_SURFACES[language][key]
    bullets = "\n".join(f"- {line}" for line in lines)
    return f"{heading}\n{bullets}"


def render_messages(messages: list[dict[str, str]], *, language: LanguageCode) -> str:
    labels = {
        "en": {"user": "User", "assistant": "Assistant", "system": "System"},
        "da": {"user": "Bruger", "assistant": "Assistent", "system": "System"},
    }[language]

    rendered: list[str] = []
    for message in messages:
        role = str(message["role"])
        content = str(message["content"]).strip()
        label = labels.get(role, role.capitalize())
        rendered.append(f"{label}:\n{content}")
    return "\n\n".join(rendered)


async def generate_scenario_bundle(
    writer: LLM,
    row_plan: dict[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    prompt_seed = _prompt_seed_text(row_plan)
    interaction_style = str(row_plan["interaction_style"])
    if prompt_seed and interaction_style == "single_turn":
        return {"title": _seed_title(prompt_seed), "user_prefix": prompt_seed}
    if prompt_seed and interaction_style == "multi_turn_isolation":
        assistant_reply = await writer.achat(
            [{"role": "user", "content": prompt_seed}],
            temperature=temperature,
        )
        return _seeded_multiturn_bundle(
            row_plan,
            prompt_seed=prompt_seed,
            assistant_reply=str(assistant_reply).strip(),
        )

    payload = await achat_json(writer, _scenario_messages(row_plan), temperature=temperature)
    _validate_bundle(payload, interaction_style=interaction_style)
    return payload


async def select_prompt_keywords(
    writer: LLM,
    row_plan: dict[str, Any],
    *,
    candidate_keywords: list[str],
    target_count: int,
) -> list[str]:
    prompt_seed = _prompt_seed_text(row_plan)
    if not prompt_seed or target_count <= 0:
        return []
    if len(candidate_keywords) <= target_count:
        return list(candidate_keywords)

    payload = await achat_json(
        writer,
        _keyword_selection_messages(
            row_plan,
            candidate_keywords=candidate_keywords,
            target_count=target_count,
        ),
        temperature=0.0,
    )
    return _validate_keyword_selection(
        payload,
        candidate_keywords=candidate_keywords,
        target_count=target_count,
    )


async def profile_prompt_seed(
    writer: LLM,
    prompt_seed: dict[str, Any],
    *,
    allowed_shapes: tuple[ResponseShape, ...],
) -> dict[str, Any]:
    payload = await achat_json(
        writer,
        _prompt_profile_messages(
            prompt_seed,
            allowed_shapes=allowed_shapes,
        ),
        temperature=0.0,
    )
    return _validate_prompt_profile(
        payload,
        prompt_seed=prompt_seed,
        allowed_shapes=allowed_shapes,
    )


def fallback_scenario_bundle(row_plan: dict[str, Any]) -> dict[str, Any]:
    prompt_seed = _prompt_seed_text(row_plan)
    interaction_style = str(row_plan["interaction_style"])
    language = str(row_plan["language"])
    if prompt_seed and interaction_style == "single_turn":
        return {"title": _seed_title(prompt_seed), "user_prefix": prompt_seed}
    if prompt_seed and interaction_style == "multi_turn_isolation":
        return _seeded_multiturn_bundle(
            row_plan,
            prompt_seed=prompt_seed,
            assistant_reply=_default_assistant_reply(language),
        )

    request = _fallback_user_request(row_plan)
    if interaction_style == "single_turn":
        return {"title": _seed_title(request), "user_prefix": request}

    follow_up_surface = str(row_plan.get("follow_up_surface", "")).strip() or None
    return {
        "title": _seed_title(request),
        "base_user": request,
        "assistant_reply": _default_assistant_reply(language),
        "final_user_prefix": _default_follow_up(language, surface_key=follow_up_surface),
    }


def generate_scenario_bundle_sync(
    writer: LLM,
    row_plan: dict[str, Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    prompt_seed = _prompt_seed_text(row_plan)
    interaction_style = str(row_plan["interaction_style"])
    if prompt_seed and interaction_style == "single_turn":
        return {"title": _seed_title(prompt_seed), "user_prefix": prompt_seed}
    if prompt_seed and interaction_style == "multi_turn_isolation":
        assistant_reply = writer.chat(
            [{"role": "user", "content": prompt_seed}],
            temperature=temperature,
        )
        return _seeded_multiturn_bundle(
            row_plan,
            prompt_seed=prompt_seed,
            assistant_reply=str(assistant_reply).strip(),
        )

    payload = chat_json(writer, _scenario_messages(row_plan), temperature=temperature)
    _validate_bundle(payload, interaction_style=interaction_style)
    return payload


def materialize_messages(
    bundle: dict[str, Any],
    row_plan: dict[str, Any],
    *,
    instruction_block: str,
) -> list[dict[str, str]]:
    interaction_style = str(row_plan["interaction_style"])
    if interaction_style == "single_turn":
        user_prefix = str(bundle["user_prefix"]).strip()
        final_user = _merge_prefix_and_block(user_prefix, instruction_block)
        return [{"role": "user", "content": final_user}]

    base_user = str(bundle["base_user"]).strip()
    assistant_reply = str(bundle["assistant_reply"]).strip()
    final_user_prefix = str(bundle["final_user_prefix"]).strip()
    final_user = _merge_prefix_and_block(final_user_prefix, instruction_block)
    return [
        {"role": "user", "content": base_user},
        {"role": "assistant", "content": assistant_reply},
        {"role": "user", "content": final_user},
    ]


def _merge_prefix_and_block(prefix: str, instruction_block: str) -> str:
    if not prefix:
        return instruction_block
    return f"{prefix}\n\n{instruction_block}"


def _scenario_messages(row_plan: dict[str, Any]) -> list[dict[str, str]]:
    language = str(row_plan["language"])
    interaction_style = str(row_plan["interaction_style"])
    response_shape = str(row_plan["response_shape"])
    scenario_kind = str(row_plan["scenario_kind"])
    topic = str(row_plan["topic"])
    constraint_lines = "\n".join(f"- {line}" for line in row_plan["constraint_lines"])

    if language == "en":
        system = (
            "You write natural conversation scaffolds for a verifiable instruction-following benchmark. "
            "Return JSON only. Keep the task realistic and keep the constraint text out of the earlier turns."
        )
        user = (
            f"Language: English\n"
            f"Interaction style: {interaction_style}\n"
            f"Scenario kind: {scenario_kind}\n"
            f"Topic: {topic}\n"
            f"Response shape to anticipate: {response_shape}\n"
            f"Visible constraint lines that will be appended later:\n{constraint_lines}\n\n"
            "If the interaction style is single_turn, return:\n"
            '{"title":"short title","user_prefix":"natural user request"}\n\n'
            "If the interaction style is multi_turn_isolation, return:\n"
            '{"title":"short title","base_user":"first user turn","assistant_reply":"brief assistant answer to the first turn","final_user_prefix":"follow-up user turn that asks for a revised answer and naturally introduces the exact requirements that will be appended later"}'
        )
    else:
        system = (
            "Du skriver naturlige samtaleskabeloner til et benchmark for verificerbar instruktionsefterlevelse. "
            "Returner kun JSON. Gør opgaven realistisk, og hold selve kravteksten ude af de tidligere ture."
        )
        user = (
            f"Sprog: Dansk\n"
            f"Interaktionsstil: {interaction_style}\n"
            f"Scenarietype: {scenario_kind}\n"
            f"Emne: {topic}\n"
            f"Forventet svarform: {response_shape}\n"
            f"Synlige kravlinjer, som bliver tilføjet senere:\n{constraint_lines}\n\n"
            "Hvis interaktionsstilen er single_turn, så returner:\n"
            '{"title":"kort titel","user_prefix":"naturlig brugerforespørgsel"}\n\n'
            "Hvis interaktionsstilen er multi_turn_isolation, så returner:\n"
            '{"title":"kort titel","base_user":"første brugertur","assistant_reply":"kort assistentsvar på den første tur","final_user_prefix":"opfølgende brugertur, der beder om en ny version og naturligt introducerer de præcise krav, som senere bliver tilføjet"}'
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _keyword_selection_messages(
    row_plan: dict[str, Any],
    *,
    candidate_keywords: list[str],
    target_count: int,
) -> list[dict[str, str]]:
    language = str(row_plan["language"])
    prompt_seed = _prompt_seed_text(row_plan)
    candidates = ", ".join(f'"{keyword}"' for keyword in candidate_keywords)

    if language == "en":
        system = (
            "You select semantically central keywords for constrained rewrites. "
            "Return JSON only, and choose only from the provided candidate keywords."
        )
        user = (
            f"Prompt:\n{prompt_seed}\n\n"
            f"Candidate keywords: [{candidates}]\n"
            f"Return exactly {target_count} unique keywords that are most important to preserve in a faithful rewrite.\n"
            'Return {"keywords":["keyword one","keyword two"]}.'
        )
    else:
        system = (
            "Du udvælger semantisk centrale nøgleord til begrænsede omskrivninger. "
            "Returner kun JSON, og vælg kun fra de givne kandidatord."
        )
        user = (
            f"Prompt:\n{prompt_seed}\n\n"
            f"Kandidatord: [{candidates}]\n"
            f"Returner præcis {target_count} unikke nøgleord, som er vigtigst at bevare i en trofast omskrivning.\n"
            'Returner {"keywords":["ord et","ord to"]}.'
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _prompt_profile_messages(
    prompt_seed: dict[str, Any],
    *,
    allowed_shapes: tuple[ResponseShape, ...],
) -> list[dict[str, str]]:
    language = str(prompt_seed["language"])
    prompt_text = str(prompt_seed["text"]).strip()
    task_types = ", ".join(f'"{task_type}"' for task_type in TASK_TYPES)
    shapes = ", ".join(f'"{shape}"' for shape in allowed_shapes)

    if language == "da":
        system = (
            "Du profilerer kildeprompter til en generator af verificerbare instruction-following eksempler. "
            "Returner kun JSON. Vurder, hvilke svarformer der bevarer promptens semantik, og vær konservativ ved "
            "oversættelse, regneopgaver eller mærkelige prompter."
        )
        user = (
            f"Kildeprompt:\n{prompt_text}\n\n"
            f"Tilladte task_type værdier: [{task_types}]\n"
            'Tilladte semantic_rigidity værdier: ["open", "medium", "rigid"]\n'
            'Tilladte naturalness_confidence værdier: ["low", "medium", "high"]\n'
            f"Tilladte response shapes: [{shapes}]\n\n"
            "Returner et JSON-objekt med disse felter:\n"
            '- "task_type": én værdi fra listen\n'
            '- "semantic_rigidity": én værdi fra listen\n'
            '- "requested_item_count": positivt heltal eller null\n'
            '- "requested_sentence_count": positivt heltal eller null\n'
            '- "requested_line_count": positivt heltal eller null\n'
            '- "numeric_task": true eller false\n'
            '- "preserve_literal_source_text": true eller false\n'
            '- "semantic_keywords": to til seks korte nøgleord eller nøglefraser fra prompten\n'
            '- "safe_response_shapes": en ikke-tom liste af response shapes, som bevarer promptens mening\n'
            '- "naturalness_confidence": én værdi fra listen\n\n'
            "Hvis prompten kræver trofast gengivelse eller tæt bevaring af kildetekst, så sæt "
            '"preserve_literal_source_text" til true og vælg konservative svarformer.'
        )
    else:
        system = (
            "You profile source prompts for a generator of verifiable instruction-following examples. "
            "Return JSON only. Judge which response shapes preserve the prompt semantics, and be conservative for "
            "translation, calculation, or odd prompts."
        )
        user = (
            f"Source prompt:\n{prompt_text}\n\n"
            f"Allowed task_type values: [{task_types}]\n"
            'Allowed semantic_rigidity values: ["open", "medium", "rigid"]\n'
            'Allowed naturalness_confidence values: ["low", "medium", "high"]\n'
            f"Allowed response shapes: [{shapes}]\n\n"
            "Return a JSON object with these fields:\n"
            '- "task_type": one value from the list\n'
            '- "semantic_rigidity": one value from the list\n'
            '- "requested_item_count": positive integer or null\n'
            '- "requested_sentence_count": positive integer or null\n'
            '- "requested_line_count": positive integer or null\n'
            '- "numeric_task": true or false\n'
            '- "preserve_literal_source_text": true or false\n'
            '- "semantic_keywords": two to six short keywords or key phrases from the prompt\n'
            '- "safe_response_shapes": a non-empty list of response shapes that preserve the prompt meaning\n'
            '- "naturalness_confidence": one value from the list\n\n'
            'If the prompt requires faithful reproduction or close preservation of source text, set '
            '"preserve_literal_source_text" to true and choose conservative response shapes.'
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _validate_bundle(bundle: dict[str, Any], *, interaction_style: str) -> None:
    title = str(bundle.get("title", "")).strip()
    assert title, "scenario bundle must include title"

    if interaction_style == "single_turn":
        user_prefix = str(bundle.get("user_prefix", "")).strip()
        assert user_prefix, "single_turn scenario bundle must include user_prefix"
        return

    base_user = str(bundle.get("base_user", "")).strip()
    assistant_reply = str(bundle.get("assistant_reply", "")).strip()
    final_user_prefix = str(bundle.get("final_user_prefix", "")).strip()
    assert base_user, "multi_turn scenario bundle must include base_user"
    assert assistant_reply, "multi_turn scenario bundle must include assistant_reply"
    assert final_user_prefix, "multi_turn scenario bundle must include final_user_prefix"


def _validate_keyword_selection(
    payload: dict[str, Any],
    *,
    candidate_keywords: list[str],
    target_count: int,
) -> list[str]:
    raw_keywords = payload.get("keywords")
    assert isinstance(raw_keywords, list), "keyword selection must include a keywords list"

    allowed = {keyword.casefold(): keyword for keyword in candidate_keywords}
    selected: list[str] = []
    seen: set[str] = set()
    for item in raw_keywords:
        key = str(item).strip().casefold()
        if key not in allowed or key in seen:
            continue
        selected.append(allowed[key])
        seen.add(key)
        if len(selected) >= target_count:
            break

    assert selected, "keyword selection must include at least one allowed keyword"
    return selected


def _validate_prompt_profile(
    payload: dict[str, Any],
    *,
    prompt_seed: dict[str, Any],
    allowed_shapes: tuple[ResponseShape, ...],
) -> dict[str, Any]:
    allowed_shape_map = {shape: shape for shape in allowed_shapes}
    task_type = str(payload.get("task_type", prompt_seed.get("task_type", "general_generation"))).strip()
    if task_type not in TASK_TYPES:
        task_type = str(prompt_seed.get("task_type", "general_generation"))

    semantic_rigidity = str(payload.get("semantic_rigidity", prompt_seed.get("semantic_rigidity", "medium"))).strip()
    if semantic_rigidity not in RIGIDITY_LEVELS:
        semantic_rigidity = str(prompt_seed.get("semantic_rigidity", "medium"))

    naturalness_confidence = str(
        payload.get("naturalness_confidence", prompt_seed.get("naturalness_confidence", "medium"))
    ).strip()
    if naturalness_confidence not in NATURALNESS_LEVELS:
        naturalness_confidence = "medium"

    semantic_keywords = _profile_keywords(payload.get("semantic_keywords"), prompt_seed)
    safe_response_shapes = _profile_safe_shapes(
        payload.get("safe_response_shapes"),
        prompt_seed=prompt_seed,
        allowed_shape_map=allowed_shape_map,
    )

    return {
        "task_type": task_type,
        "semantic_rigidity": semantic_rigidity,
        "requested_item_count": _profile_count(payload.get("requested_item_count"), prompt_seed, "requested_item_count"),
        "requested_sentence_count": _profile_count(
            payload.get("requested_sentence_count"),
            prompt_seed,
            "requested_sentence_count",
        ),
        "requested_line_count": _profile_count(payload.get("requested_line_count"), prompt_seed, "requested_line_count"),
        "numeric_task": _profile_bool(payload, prompt_seed, "numeric_task"),
        "preserve_literal_source_text": _profile_bool(payload, prompt_seed, "preserve_literal_source_text"),
        "semantic_keywords": semantic_keywords,
        "safe_response_shapes": safe_response_shapes,
        "naturalness_confidence": naturalness_confidence,
        "profile_source": "model",
    }


def _profile_count(value: Any, prompt_seed: dict[str, Any], key: str) -> int | None:
    if value is None:
        fallback = prompt_seed.get(key)
        return int(fallback) if isinstance(fallback, int) and fallback > 0 else None
    if isinstance(value, int) and value > 0:
        return value
    return None


def _profile_bool(payload: dict[str, Any], prompt_seed: dict[str, Any], key: str) -> bool:
    if key in payload:
        return bool(payload[key])
    return bool(prompt_seed.get(key, False))


def _profile_keywords(value: Any, prompt_seed: dict[str, Any]) -> list[str]:
    fallback = [
        str(item).strip()
        for item in prompt_seed.get("semantic_keywords", [])
        if isinstance(item, str) and str(item).strip()
    ]
    if not isinstance(value, list):
        return fallback

    selected: list[str] = []
    seen: set[str] = set()
    for item in value:
        keyword = str(item).strip()
        if not keyword:
            continue
        lowered = keyword.casefold()
        if lowered in seen:
            continue
        selected.append(keyword)
        seen.add(lowered)
        if len(selected) >= 6:
            break

    if len(selected) >= 2:
        return selected
    return fallback


def _profile_safe_shapes(
    value: Any,
    *,
    prompt_seed: dict[str, Any],
    allowed_shape_map: dict[ResponseShape, ResponseShape],
) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    if isinstance(value, list):
        for item in value:
            shape = str(item).strip()
            if shape not in allowed_shape_map or shape in seen:
                continue
            selected.append(allowed_shape_map[shape])
            seen.add(shape)

    if selected:
        return selected

    fallback = prompt_seed.get("safe_response_shapes")
    if isinstance(fallback, list):
        return [
            str(item)
            for item in fallback
            if isinstance(item, str) and str(item) in allowed_shape_map
        ]
    return list(allowed_shape_map.values())


def _prompt_seed_text(row_plan: dict[str, Any]) -> str:
    prompt_seed = row_plan.get("prompt_seed")
    if not isinstance(prompt_seed, dict):
        return ""
    return str(prompt_seed.get("text", "")).strip()


def _seed_title(prompt_seed: str) -> str:
    words = prompt_seed.split()
    if not words:
        return "Seed prompt"
    title = " ".join(words[:6]).strip()
    if len(words) > 6:
        return f"{title}..."
    return title


def _seeded_multiturn_bundle(
    row_plan: dict[str, Any],
    *,
    prompt_seed: str,
    assistant_reply: str,
) -> dict[str, Any]:
    language = str(row_plan["language"])
    reply = assistant_reply or _default_assistant_reply(language)
    follow_up_surface = str(row_plan.get("follow_up_surface", "")).strip() or None
    return {
        "title": _seed_title(prompt_seed),
        "base_user": prompt_seed,
        "assistant_reply": reply,
        "final_user_prefix": _default_follow_up(language, surface_key=follow_up_surface),
    }


def _default_follow_up(language: str, *, surface_key: str | None = None) -> str:
    surfaces = FOLLOW_UP_SURFACES[language]
    key = surface_key or next(iter(surfaces))
    return surfaces[key]


def _default_assistant_reply(language: str) -> str:
    if language == "da":
        return "Her er et første svar."
    return "Here is an initial answer."


def _fallback_user_request(row_plan: dict[str, Any]) -> str:
    language = str(row_plan["language"])
    scenario_kind = str(row_plan["scenario_kind"])
    topic = str(row_plan["topic"])
    if language == "da":
        return f"Skriv en kort {scenario_kind} om {topic}."
    return f"Write a short {scenario_kind} about {topic}."


def _instruction_surface(language: LanguageCode, surface_key: str | None) -> tuple[str, str]:
    if not surface_key:
        key = next(iter(BLOCK_INSTRUCTION_SURFACES[language]))
        return "block", key

    layout, _, raw_key = surface_key.partition(":")
    assert raw_key, "instruction surface key must include a layout prefix"
    if layout == "block":
        assert raw_key in BLOCK_INSTRUCTION_SURFACES[language], f"unknown block instruction surface: {surface_key}"
        return layout, raw_key
    assert layout == "inline", f"unsupported instruction layout: {layout}"
    assert raw_key in INLINE_INSTRUCTION_SURFACES[language], f"unknown inline instruction surface: {surface_key}"
    return layout, raw_key


def _instruction_sentence(line: str) -> str:
    text = line.strip()
    if text.endswith((".", "!", "?")):
        return text
    return f"{text}."
