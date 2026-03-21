from __future__ import annotations

import json
import re
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from sdg.commons.model import LLM
from sdg.commons.run_log import current_run_log, log_event, write_snapshot

MAX_JSON_REPAIR_ATTEMPTS = 3
_JSON_METRICS_LOCK = threading.Lock()
_JSON_METRICS_SESSION: int | None = None
_JSON_METRICS = {
    "parse_attempts": 0,
    "parse_successes": 0,
    "parse_failures": 0,
    "repair_attempts": 0,
    "repair_successes": 0,
    "repair_failures": 0,
}


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
        _record_json_parse_attempt()
        try:
            parsed = parse_json_response(current)
            _record_json_parse_success(repaired=last_error is not None)
            return parsed
        except (AssertionError, json.JSONDecodeError) as error:
            last_error = error
            _record_json_parse_failure()
            _record_json_repair_attempt()
            log_event(
                "llm_json",
                "repair_requested",
                error_type=error.__class__.__name__,
                error_message=str(error),
            )
            current = repair(current)
    assert last_error is not None
    _record_json_repair_failure()
    log_event(
        "llm_json",
        "repair_failed",
        error_type=last_error.__class__.__name__,
        error_message=str(last_error),
    )
    raise last_error


async def _parse_json_with_async_repair(
    response: Any,
    repair: Callable[[Any], Awaitable[Any]],
) -> dict[str, Any]:
    current = response
    last_error: Exception | None = None
    for _ in range(MAX_JSON_REPAIR_ATTEMPTS):
        _record_json_parse_attempt()
        try:
            parsed = parse_json_response(current)
            _record_json_parse_success(repaired=last_error is not None)
            return parsed
        except (AssertionError, json.JSONDecodeError) as error:
            last_error = error
            _record_json_parse_failure()
            _record_json_repair_attempt()
            log_event(
                "llm_json",
                "repair_requested",
                error_type=error.__class__.__name__,
                error_message=str(error),
            )
            current = await repair(current)
    assert last_error is not None
    _record_json_repair_failure()
    log_event(
        "llm_json",
        "repair_failed",
        error_type=last_error.__class__.__name__,
        error_message=str(last_error),
    )
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


def _record_json_parse_attempt() -> None:
    with _JSON_METRICS_LOCK:
        if not _ensure_json_metrics_session_locked():
            return
        _JSON_METRICS["parse_attempts"] += 1
        _write_json_metrics_snapshot_locked()


def _record_json_parse_success(*, repaired: bool) -> None:
    with _JSON_METRICS_LOCK:
        if not _ensure_json_metrics_session_locked():
            return
        _JSON_METRICS["parse_successes"] += 1
        if repaired:
            _JSON_METRICS["repair_successes"] += 1
        _write_json_metrics_snapshot_locked()


def _record_json_parse_failure() -> None:
    with _JSON_METRICS_LOCK:
        if not _ensure_json_metrics_session_locked():
            return
        _JSON_METRICS["parse_failures"] += 1
        _write_json_metrics_snapshot_locked()


def _record_json_repair_attempt() -> None:
    with _JSON_METRICS_LOCK:
        if not _ensure_json_metrics_session_locked():
            return
        _JSON_METRICS["repair_attempts"] += 1
        _write_json_metrics_snapshot_locked()


def _record_json_repair_failure() -> None:
    with _JSON_METRICS_LOCK:
        if not _ensure_json_metrics_session_locked():
            return
        _JSON_METRICS["repair_failures"] += 1
        _write_json_metrics_snapshot_locked()


def _ensure_json_metrics_session_locked() -> bool:
    global _JSON_METRICS, _JSON_METRICS_SESSION
    logger = current_run_log()
    if logger is None:
        return False
    session = id(logger)
    if _JSON_METRICS_SESSION == session:
        return True
    _JSON_METRICS_SESSION = session
    _JSON_METRICS = {
        "parse_attempts": 0,
        "parse_successes": 0,
        "parse_failures": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "repair_failures": 0,
    }
    return True


def _write_json_metrics_snapshot_locked() -> None:
    write_snapshot("llm_json_metrics.json", dict(_JSON_METRICS), min_interval_seconds=1.0)
