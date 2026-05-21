from __future__ import annotations

import contextlib
import html as _html
import re
from typing import Any, Iterator

from sdg.commons import sources as common_sources
from sdg.packs.backtranslation_passages_dsk import build as dsk_build

PACK = "backtranslation_passages_eurlex"
_ORIGINAL_LOOKS_LIKE_USER_REQUEST = None
_ORIGINAL_CLEAN_GENERATED_PROMPT = None


def build(cfg: dict[str, Any]):
    with _patched_dsk_module():
        return dsk_build.build(cfg)


def verify(run_id_or_path: str) -> dict[str, Any]:
    with _patched_dsk_module():
        return dsk_build.verify(run_id_or_path)


def summarize(run_id_or_path: str) -> dict[str, Any]:
    with _patched_dsk_module():
        return dsk_build.summarize(run_id_or_path)


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    with _patched_dsk_module():
        return dsk_build.publish(run_id_or_path, out_dir=out_dir)


@contextlib.contextmanager
def _patched_dsk_module() -> Iterator[None]:
    global _ORIGINAL_LOOKS_LIKE_USER_REQUEST
    global _ORIGINAL_CLEAN_GENERATED_PROMPT

    original_pack = dsk_build.PACK
    original_record_to_article = dsk_build._record_to_article
    original_instruction_news = dsk_build._instruction_messages_news
    original_instruction_promo = dsk_build._instruction_messages_promo
    original_score_paragraph = dsk_build._score_paragraph_web
    original_contains_shouting = dsk_build._contains_shouting_token
    original_looks_like_user_request = dsk_build._looks_like_user_request
    original_clean_generated_prompt = dsk_build._clean_generated_prompt
    original_generate_row = dsk_build._generate_row

    dsk_build.PACK = PACK
    dsk_build._record_to_article = _record_to_article
    dsk_build._instruction_messages_news = _instruction_messages_legal
    dsk_build._instruction_messages_promo = _instruction_messages_legal
    dsk_build._score_paragraph_web = _score_paragraph_legal
    dsk_build._contains_shouting_token = _contains_shouting_token_legal
    _ORIGINAL_LOOKS_LIKE_USER_REQUEST = original_looks_like_user_request
    _ORIGINAL_CLEAN_GENERATED_PROMPT = original_clean_generated_prompt
    dsk_build._looks_like_user_request = _looks_like_user_request_legal
    dsk_build._clean_generated_prompt = _clean_generated_prompt_legal
    dsk_build._generate_row = _wrap_generate_row(original_generate_row)

    try:
        yield
    finally:
        dsk_build.PACK = original_pack
        dsk_build._record_to_article = original_record_to_article
        dsk_build._instruction_messages_news = original_instruction_news
        dsk_build._instruction_messages_promo = original_instruction_promo
        dsk_build._score_paragraph_web = original_score_paragraph
        dsk_build._contains_shouting_token = original_contains_shouting
        dsk_build._looks_like_user_request = original_looks_like_user_request
        dsk_build._clean_generated_prompt = original_clean_generated_prompt
        dsk_build._generate_row = original_generate_row
        _ORIGINAL_LOOKS_LIKE_USER_REQUEST = None
        _ORIGINAL_CLEAN_GENERATED_PROMPT = None


def _instruction_messages_legal(
    article: dict[str, Any],
    *,
    persona: str | None = None,
    prompt_length: str = "medium",
) -> list[dict[str, str]]:
    del persona

    user_lines: list[str] = []
    if article.get("resource_type"):
        user_lines.append(f"Dokumenttype: {article['resource_type']}")
    if article.get("title"):
        user_lines.append(f"Titel: {article['title']}")
    user_lines.append(article["text"])

    length_rule = dsk_build._LENGTH_RULES.get(prompt_length, dsk_build._LENGTH_RULES["medium"])
    system_lines = [
        "Du er en assistent der laver træningsdata til sprogmodeller.",
        "Du svarer KUN med selve prompten — ingen forklaringer, overskrifter eller meta-kommentarer.",
        length_rule,
        "Prompten skal ligne en realistisk dansk brugerbesked til en moderne chatbot.",
        "Start gerne med fx 'Kan du', 'Skriv', 'Udarbejd' eller 'Forklar'.",
        "Prompten skal bede om en juridisk, forvaltningsmæssig eller regulatorisk tekst i neutral formel stil.",
        "Prompten skal typisk efterspørge en fuld tekst i afsnit (notat, redegørelse, vurdering eller meddelelse) — ikke en kort opsummering.",
        "Prompten skal ligne en normal bestilling til en assistent, ikke en intern dokumentklassifikation.",
        "Foretræk ord som notat, redegørelse, vurdering eller juridisk tekst.",
        "Undgå ord som retsakt, retsaktstykke, retsaktstekst, retsstatslig og domsafgørelse i prompten.",
        "Prompten må gerne nævne EU-retlig kontekst, men må IKKE henvise til den vedlagte tekst som kilde.",
        "Undgå at nævne konkrete sagsnumre, CELEX-numre eller navngivne parter.",
        "Prompten skal beskrive opgaven klart: fx redegørelse, notat, afgørelsesresume, vejledende tekst eller officiel meddelelse.",
        "Undgå abstrakte modeord og sjældne formuleringer.",
        "Undgå rollespil, slogans og billedsprog.",
    ]

    user_content = (
        "\n".join(user_lines)
        + "\n\n---\nSkriv en prompt der ville få en AI til at skrive en tilsvarende juridisk tekst på dansk."
    )

    return [
        {"role": "system", "content": "\n".join(system_lines)},
        {"role": "user", "content": user_content},
    ]


def _record_to_article(
    record: dict[str, Any],
    source: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    start_row = int(source.get("start_row", 0))
    if index < start_row:
        return None

    text = _html.unescape(
        common_sources.read_record_value(record, source.get("text_field", "text_da")) or ""
    )
    if not text:
        return None

    text_source_field = source.get("text_source_field")
    if text_source_field:
        text_source = common_sources.read_record_value(record, text_source_field)
        allowed_text_sources = source.get("allowed_text_sources")
        if allowed_text_sources:
            allowed = {str(value).strip().lower() for value in allowed_text_sources}
            if str(text_source).strip().lower() not in allowed:
                return None

    title = common_sources.read_record_value(record, source.get("title_field", "title_da"))
    url = common_sources.read_record_value(record, source.get("url_field", "url"))
    resource_type = common_sources.read_record_value(record, source.get("resource_type_field", "resource_type"))
    source_id = (
        common_sources.read_record_value(record, source.get("id_field"))
        or url
        or title
        or str(index)
    )

    return {
        "record_index": index,
        "source_id": source_id,
        "title": title,
        "text": text,
        "url": url,
        "resource_type": resource_type,
        "text_source": common_sources.read_record_value(record, text_source_field)
        if text_source_field
        else None,
    }


def _score_paragraph_legal(paragraph: str) -> float:
    if len(paragraph) < 120:
        return 0.0
    if paragraph.lstrip().startswith("<"):
        return 0.0

    alpha_ratio = sum(char.isalpha() for char in paragraph) / len(paragraph)
    if alpha_ratio < 0.5:
        return 0.0

    markup_chars = len(re.findall(r"[#*`\[\]]", paragraph))
    if markup_chars / len(paragraph) > 0.08:
        return 0.0

    return alpha_ratio * min(len(paragraph) / 1200, 1.0)


def _contains_shouting_token_legal(text: str) -> bool:
    del text
    return False


def _looks_like_user_request_legal(text: str) -> bool:
    if _ORIGINAL_LOOKS_LIKE_USER_REQUEST and _ORIGINAL_LOOKS_LIKE_USER_REQUEST(text):
        return True
    return bool(
        re.search(
            r"\b(redegør|vurder|analyser|afklar|sammenfat|udarbejd|beskriv|forklar)\b",
            text,
            re.IGNORECASE,
        )
    )


def _clean_generated_prompt_legal(value: str) -> str:
    if not _ORIGINAL_CLEAN_GENERATED_PROMPT:
        raise RuntimeError("original prompt cleaner unavailable")
    text = _ORIGINAL_CLEAN_GENERATED_PROMPT(value)

    if re.search(r"\b(sag|sagen)\s+[CTF]-?\d+/\d+\b", text, re.IGNORECASE):
        raise ValueError(f"case-id style reference rejected: {text[:120]!r}")
    if re.search(r"\bcelex\b", text, re.IGNORECASE):
        raise ValueError(f"celex reference rejected: {text[:120]!r}")
    if re.search(r"\bretsaktuel\b", text, re.IGNORECASE):
        raise ValueError(f"unnatural jargon rejected: {text[:120]!r}")
    if re.search(r"\bretsakt\w*\b", text, re.IGNORECASE):
        raise ValueError(f"document-classification jargon rejected: {text[:120]!r}")
    if re.search(r"\b(retsafgørelse|domsbeskrivelse|retsstatslig)\b", text, re.IGNORECASE):
        raise ValueError(f"document-classification jargon rejected: {text[:120]!r}")
    _assert_prompt_style_legal(text)
    return text


def _wrap_generate_row(original_generate_row):
    async def _generate_row_with_legal_meta(
        *,
        article: dict[str, Any],
        writer,
        temperature: float,
        personas: list[str],
    ) -> dict[str, Any]:
        row = await original_generate_row(
            article=article,
            writer=writer,
            temperature=temperature,
            personas=personas,
        )
        _assert_prompt_style_legal(str(row.get("prompt", "")))

        meta = row.get("meta")
        if not isinstance(meta, dict):
            return row

        if article.get("resource_type") is not None:
            meta["resource_type"] = article.get("resource_type")
        if article.get("text_source") is not None:
            meta["text_source"] = article.get("text_source")
        return row

    return _generate_row_with_legal_meta


def _assert_prompt_style_legal(text: str) -> None:
    lowered = text.lower()
    blocked_tokens = [
        "retsakt",
        "retsstatslig",
        "retsaktuel",
        "domsbeskrivelse",
        "retsafgørelse",
    ]
    for token in blocked_tokens:
        if token in lowered:
            raise ValueError(f"document-classification jargon rejected: {text[:120]!r}")
