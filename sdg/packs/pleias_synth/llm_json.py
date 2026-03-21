from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from sdg.commons.model import LLM

MAX_JSON_REPAIR_ATTEMPTS = 3


def parse_json_response(text: Any) -> dict[str, Any]:
    payload = _extract_json_object(str(text).strip())
    parsed = json.loads(payload)
    assert isinstance(parsed, dict), "LLM response must decode to a JSON object"
    return parsed


def chat_json(llm: LLM, messages: list[dict[str, str]], *, temperature: float) -> dict[str, Any]:
    response = llm.chat(messages, temperature=temperature)
    return _parse_json_with_repair(
        response,
        lambda broken: llm.chat(_repair_json_messages(broken), temperature=0.0),
    )


async def achat_json(llm: LLM, messages: list[dict[str, str]], *, temperature: float) -> dict[str, Any]:
    response = await llm.achat(messages, temperature=temperature)
    return await _parse_json_with_async_repair(
        response,
        lambda broken: llm.achat(_repair_json_messages(broken), temperature=0.0),
    )


def _parse_json_with_repair(
    response: Any,
    repair: Callable[[Any], Any],
) -> dict[str, Any]:
    current = response
    last_error: Exception | None = None
    for _ in range(MAX_JSON_REPAIR_ATTEMPTS):
        try:
            return parse_json_response(current)
        except (AssertionError, json.JSONDecodeError) as error:
            last_error = error
            current = repair(current)
    assert last_error is not None
    raise last_error


async def _parse_json_with_async_repair(
    response: Any,
    repair: Callable[[Any], Awaitable[Any]],
) -> dict[str, Any]:
    current = response
    last_error: Exception | None = None
    for _ in range(MAX_JSON_REPAIR_ATTEMPTS):
        try:
            return parse_json_response(current)
        except (AssertionError, json.JSONDecodeError) as error:
            last_error = error
            current = await repair(current)
    assert last_error is not None
    raise last_error


def _repair_json_messages(response: Any) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair malformed JSON. "
                "Return only valid JSON with the same top-level object structure. "
                "Do not add commentary."
            ),
        },
        {
            "role": "user",
            "content": str(response),
        },
    ]


def _extract_json_object(content: str) -> str:
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    if content.startswith("{") and content.endswith("}"):
        return content

    start = content.find("{")
    if start == -1:
        return content

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char != "}":
            continue

        depth -= 1
        if depth == 0:
            return content[start : index + 1]

    return content
