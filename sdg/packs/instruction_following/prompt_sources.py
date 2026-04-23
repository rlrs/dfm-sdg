from __future__ import annotations

import re
import unicodedata
from random import Random
from typing import Any

from sdg.commons.sources import iter_source_records, source_label

from .constraints import LanguageCode

DEFAULT_EXCLUDED_SUBSTRINGS: tuple[str, ...] = (
    "additional requirements",
    "yderligere krav",
    "return valid json",
    "returner gyldig json",
    "return valid xml",
    "returner gyldig xml",
    "numbered list",
    "nummereret liste",
    "bullet list",
    "punktopstilling",
    "start the response",
    "begynd svaret",
    "end the response",
    "afslut svaret",
    "do not use",
    "brug ikke",
    "exactly ",
    "præcis ",
)

STRUCTURED_LINE_RE = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+")
WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)?", flags=re.UNICODE)
COUNT_TOKEN_RE = re.compile(r"\d+|[^\W\d_]+(?:[-'][^\W\d_]+)?", flags=re.UNICODE)

TASK_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("translation", ("oversæt", "translate", "translate into", "oversaet")),
    ("rewrite", ("omskriv", "rephrase", "rewrite", "redigér", "rediger", "edit", "omformuler")),
    ("classification", ("klassificér", "klassificer", "classify", "categorize", "kategoriser")),
    ("summarization", ("opsummér", "opsummer", "summarize", "resumer", "reduce following text", "reducer følgende tekst")),
    ("comparison", ("sammenlign", "compare", "forklar forskellen", "what is the difference", "hvad er forskellen")),
    ("listing", ("oplist", "list", "angiv", "nævn", "naevn", "give me a list", "generer en liste")),
    ("analysis", ("analyser", "analyze", "analyse", "fortolk", "interpret", "vurder", "evaluate")),
    ("recommendation", ("foreslå", "anbefal", "recommend", "suggest", "giv et råd", "give advice")),
    ("explanation", ("forklar", "explain", "beskriv", "describe", "hvad er", "what is", "hvordan", "how does", "why does", "hvorfor")),
    ("creative", ("skriv", "write", "fortæl", "tell", "opfind", "invent", "lav et", "create a")),
)

SEMANTIC_STOPWORDS: dict[LanguageCode, set[str]] = {
    "en": {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "that",
        "to",
        "using",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "write",
        "list",
        "give",
        "generate",
        "create",
        "describe",
        "explain",
        "compare",
        "summarize",
        "translate",
        "classify",
        "recommend",
        "suggest",
        "answer",
        "response",
    },
    "da": {
        "af",
        "at",
        "de",
        "den",
        "der",
        "det",
        "en",
        "et",
        "for",
        "fra",
        "hvad",
        "hvem",
        "hvilke",
        "hvilken",
        "hvilket",
        "hvor",
        "hvordan",
        "hvorfor",
        "i",
        "med",
        "og",
        "om",
        "på",
        "som",
        "til",
        "ud",
        "ved",
        "skriv",
        "oplist",
        "angiv",
        "nævn",
        "forklar",
        "sammenlign",
        "opsummer",
        "oversæt",
        "klassificer",
        "anbefal",
        "foreslå",
        "svar",
        "svaret",
    },
}

NUMBER_WORDS: dict[LanguageCode, dict[str, int]] = {
    "en": {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    },
    "da": {
        "en": 1,
        "et": 1,
        "ét": 1,
        "to": 2,
        "tre": 3,
        "fire": 4,
        "fem": 5,
        "seks": 6,
        "syv": 7,
        "otte": 8,
        "ni": 9,
        "ti": 10,
        "elleve": 11,
        "tolv": 12,
    },
}

LIST_HINTS: dict[LanguageCode, tuple[str, ...]] = {
    "en": ("list", "ideas", "examples", "steps", "ways", "facts", "reasons", "items", "bullet", "points"),
    "da": ("liste", "idéer", "ideer", "eksempler", "trin", "måder", "punkter", "fakta", "grunde"),
}

MATH_HINTS: dict[LanguageCode, tuple[str, ...]] = {
    "en": ("calculate", "cost", "price", "watt", "kwh", "percent", "plus", "minus", "total", "expected"),
    "da": ("beregn", "koste", "pris", "watt", "kwh", "procent", "plus", "minus", "i alt", "forventer"),
}


def load_prompt_reservoirs(
    generation: dict[str, Any],
    *,
    languages: tuple[LanguageCode, ...],
) -> dict[LanguageCode, list[dict[str, Any]]]:
    raw_sources = generation.get("prompt_sources", {})
    if raw_sources is None:
        return {}

    assert isinstance(raw_sources, dict), "generation prompt_sources must be a mapping"
    reservoirs: dict[LanguageCode, list[dict[str, Any]]] = {}
    for language in languages:
        source = raw_sources.get(language)
        if source is None:
            continue
        assert isinstance(source, dict) and source, f"generation prompt_sources.{language} must be a mapping"
        reservoirs[language] = _load_prompt_reservoir(language, source)
    return reservoirs


def prompt_source_label_for_row(row_plan: dict[str, Any]) -> str:
    prompt_seed = row_plan.get("prompt_seed")
    if isinstance(prompt_seed, dict):
        label = prompt_seed.get("source_label")
        if label:
            return str(label)
    return "synthetic"


def build_prompt_sampler(
    prompts: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    rng = Random(seed)
    buckets: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        task_type = str(prompt["task_type"])
        length_bucket = str(prompt["length_bucket"])
        task_bucket = buckets.setdefault(task_type, {"used": 0, "subbuckets": {}})
        subbucket = task_bucket["subbuckets"].setdefault(
            length_bucket,
            {"used": 0, "cursor": 0, "prompts": []},
        )
        subbucket["prompts"].append(dict(prompt))

    for task_bucket in buckets.values():
        for subbucket in task_bucket["subbuckets"].values():
            rng.shuffle(subbucket["prompts"])

    return {"buckets": buckets}


def sample_balanced_prompt_seed(
    sampler: dict[str, Any],
    *,
    rng: Random,
) -> dict[str, Any]:
    buckets = dict(sampler["buckets"])
    task_type = rng.choice(_least_used_labels({label: int(bucket["used"]) for label, bucket in buckets.items()}))
    task_bucket = buckets[task_type]
    subbuckets = dict(task_bucket["subbuckets"])
    length_bucket = rng.choice(_least_used_labels({label: int(bucket["used"]) for label, bucket in subbuckets.items()}))
    subbucket = subbuckets[length_bucket]
    prompts = list(subbucket["prompts"])
    prompt = dict(prompts[int(subbucket["cursor"]) % len(prompts)])
    subbucket["cursor"] = int(subbucket["cursor"]) + 1
    subbucket["used"] = int(subbucket["used"]) + 1
    task_bucket["used"] = int(task_bucket["used"]) + 1
    return prompt


def _load_prompt_reservoir(
    language: LanguageCode,
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    max_records = _positive_int(source, "max_records", default=200_000)
    max_prompts = _positive_int(source, "max_prompts", default=50_000)
    min_chars = _positive_int(source, "min_chars", default=40)
    max_chars = _positive_int(source, "max_chars", default=700)
    max_lines = _positive_int(source, "max_lines", default=12)
    selection = _selection(source)
    prompt_field = _optional_field_name(source, "prompt_field")
    messages_field = str(source.get("messages_field", "messages"))
    role_field = str(source.get("role_field", "role"))
    content_field = str(source.get("content_field", "content"))
    prompt_role = str(source.get("prompt_role", "user"))
    excluded_substrings = _excluded_substrings(source)
    label = source_label(source)

    prompts: list[dict[str, Any]] = []
    seen: set[str] = set()
    scanned = 0
    for record in iter_source_records(source, default_streaming=True):
        scanned += 1
        if scanned > max_records or len(prompts) >= max_prompts:
            break

        candidates = _extract_prompt_candidates(
            record,
            prompt_field=prompt_field,
            messages_field=messages_field,
            role_field=role_field,
            content_field=content_field,
            prompt_role=prompt_role,
            selection=selection,
        )
        for candidate in candidates:
            if not _keep_prompt(
                candidate,
                min_chars=min_chars,
                max_chars=max_chars,
                max_lines=max_lines,
                excluded_substrings=excluded_substrings,
            ):
                continue

            normalized = _normalize_prompt(candidate)
            if normalized in seen:
                continue

            task_type = _task_type(candidate)
            numeric_task = _numeric_task(
                candidate,
                language=language,
                task_type=task_type,
            )
            requested_item_count = _requested_item_count(
                candidate,
                language=language,
                task_type=task_type,
            )
            requested_sentence_count = _requested_sentence_count(
                candidate,
                language=language,
            )
            requested_line_count = _requested_line_count(
                candidate,
                language=language,
            )

            prompts.append(
                {
                    "text": candidate,
                    "source_label": label,
                    "source_record": scanned - 1,
                    "language": language,
                    "task_type": task_type,
                    "length_bucket": _length_bucket(candidate),
                    "semantic_rigidity": _semantic_rigidity(
                        candidate,
                        language=language,
                        task_type=task_type,
                    ),
                    "requested_item_count": requested_item_count,
                    "requested_sentence_count": requested_sentence_count,
                    "requested_line_count": requested_line_count,
                    "numeric_task": numeric_task,
                    "semantic_keywords": _semantic_keywords(
                        candidate,
                        language=language,
                    ),
                }
            )
            seen.add(normalized)
            if len(prompts) >= max_prompts:
                break

    assert prompts, f"no usable prompts found for {language} prompt source {label}"
    return prompts


def _extract_prompt_candidates(
    record: dict[str, Any],
    *,
    prompt_field: str | None,
    messages_field: str,
    role_field: str,
    content_field: str,
    prompt_role: str,
    selection: str,
) -> list[str]:
    if prompt_field:
        value = _field_value(record, prompt_field)
        if value is None:
            return []
        text = str(value).strip()
        return [text] if text else []

    messages = _field_value(record, messages_field)
    if not isinstance(messages, list):
        return []

    user_turns: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _field_value(message, role_field)
        content = _field_value(message, content_field)
        if str(role).strip() != prompt_role:
            continue
        text = str(content).strip()
        if not text:
            continue
        user_turns.append(text)

    if selection == "all_user":
        return user_turns
    if not user_turns:
        return []
    return [user_turns[0]]


def _keep_prompt(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    max_lines: int,
    excluded_substrings: tuple[str, ...],
) -> bool:
    if len(text) < min_chars or len(text) > max_chars:
        return False

    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if len(non_empty_lines) > max_lines:
        return False

    if "```" in text:
        return False

    structured_lines = sum(1 for line in non_empty_lines if STRUCTURED_LINE_RE.match(line))
    if structured_lines >= 3:
        return False

    lowered = text.casefold()
    if any(fragment in lowered for fragment in excluded_substrings):
        return False

    return True


def _normalize_prompt(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.casefold())
    stripped = "".join(char for char in folded if not unicodedata.combining(char))
    return " ".join(stripped.split())


def _task_type(text: str) -> str:
    normalized = _normalize_prompt(text)
    lead = normalized.split(":", 1)[0]
    for task_type, patterns in TASK_TYPE_PATTERNS:
        if any(lead.startswith(pattern) or f" {pattern} " in normalized for pattern in patterns):
            return task_type
    if "?" in text:
        return "question_answering"
    return "general_generation"


def _length_bucket(text: str) -> str:
    word_count = len(WORD_RE.findall(text))
    if word_count <= 10:
        return "short"
    if word_count <= 22:
        return "medium"
    return "long"


def _semantic_rigidity(
    text: str,
    *,
    language: LanguageCode,
    task_type: str,
) -> str:
    if _numeric_task(text, language=language, task_type=task_type):
        return "rigid"
    if _requested_sentence_count(text, language=language) is not None:
        return "rigid"
    if _requested_line_count(text, language=language) is not None:
        return "rigid"
    if task_type in {"translation", "classification"}:
        return "rigid"
    if _requested_item_count(text, language=language, task_type=task_type) is not None:
        return "medium"
    if task_type in {"analysis", "comparison", "explanation", "question_answering", "rewrite", "summarization"}:
        return "medium"
    return "open"


def _requested_item_count(
    text: str,
    *,
    language: LanguageCode,
    task_type: str,
) -> int | None:
    if task_type not in {"classification", "creative", "listing", "recommendation", "translation"}:
        return None

    lowered = _normalize_prompt(text)
    if not any(hint in lowered for hint in LIST_HINTS[language]):
        if task_type != "translation":
            return None

    for value in _small_numbers(text, language=language):
        if 2 <= value <= 10:
            return value
    return None


def _requested_sentence_count(text: str, *, language: LanguageCode) -> int | None:
    sentence_words = ("sentence", "sentences") if language == "en" else ("sætning", "sætninger")
    return _count_near_units(text, language=language, units=sentence_words)


def _requested_line_count(text: str, *, language: LanguageCode) -> int | None:
    line_words = ("line", "lines") if language == "en" else ("linje", "linjer")
    return _count_near_units(text, language=language, units=line_words)


def _numeric_task(
    text: str,
    *,
    language: LanguageCode,
    task_type: str,
) -> bool:
    if task_type not in {"analysis", "comparison", "explanation", "question_answering"}:
        return False

    lowered = _normalize_prompt(text)
    if any(char.isdigit() for char in text):
        return True
    return any(hint in lowered for hint in MATH_HINTS[language])


def _semantic_keywords(text: str, *, language: LanguageCode) -> list[str]:
    stopwords = SEMANTIC_STOPWORDS[language]
    number_words = set(NUMBER_WORDS[language])
    keywords: list[str] = []
    seen: set[str] = set()
    for token in WORD_RE.findall(text):
        normalized = token.casefold()
        if normalized in seen:
            continue
        if normalized in stopwords or normalized in number_words:
            continue
        if normalized.isdigit():
            continue
        if len(normalized) <= 2:
            continue
        keywords.append(normalized)
        seen.add(normalized)
        if len(keywords) >= 12:
            break
    return keywords


def _count_near_units(
    text: str,
    *,
    language: LanguageCode,
    units: tuple[str, ...],
) -> int | None:
    lowered = _normalize_prompt(text)
    tokens = COUNT_TOKEN_RE.findall(lowered)
    number_words = NUMBER_WORDS[language]
    unit_set = set(units)
    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        if next_token not in unit_set:
            continue
        if token.isdigit():
            value = int(token)
        else:
            value = number_words.get(token)
        if value is None:
            continue
        if 1 <= value <= 12:
            return value
    return None


def _small_numbers(text: str, *, language: LanguageCode) -> list[int]:
    values: list[int] = []
    number_words = NUMBER_WORDS[language]
    for token in COUNT_TOKEN_RE.findall(_normalize_prompt(text)):
        if token.isdigit():
            value = int(token)
        else:
            value = number_words.get(token)
        if value is None:
            continue
        if 1 <= value <= 12:
            values.append(value)
    return values


def _excluded_substrings(source: dict[str, Any]) -> tuple[str, ...]:
    raw = source.get("excluded_substrings")
    if raw is None:
        return DEFAULT_EXCLUDED_SUBSTRINGS
    assert isinstance(raw, list), "prompt source excluded_substrings must be a list"
    return tuple(str(item).casefold() for item in raw)


def _selection(source: dict[str, Any]) -> str:
    value = str(source.get("selection", "first_user"))
    assert value in {"first_user", "all_user"}, "prompt source selection must be first_user or all_user"
    return value


def _optional_field_name(source: dict[str, Any], key: str) -> str | None:
    value = source.get(key)
    if value is None:
        return None
    assert isinstance(value, str) and value.strip(), f"prompt source {key} must be a non-empty string"
    return value


def _least_used_labels(counts: dict[str, int]) -> list[str]:
    minimum = min(counts.values())
    return [label for label, count in counts.items() if count == minimum]


def _field_value(record: dict[str, Any], field_name: str) -> Any:
    value: Any = record
    for part in field_name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None
    return value


def _positive_int(record: dict[str, Any], key: str, *, default: int) -> int:
    value = record.get(key)
    if value is None:
        return default
    assert isinstance(value, int) and value > 0, f"{key} must be a positive integer"
    return value
