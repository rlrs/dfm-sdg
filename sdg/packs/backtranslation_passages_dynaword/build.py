from __future__ import annotations

import asyncio
import hashlib
import html as _html
import json
import re
import sys
from pathlib import Path
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import concurrency as common_concurrency
from sdg.commons import eval as common_eval
from sdg.commons import progress as common_progress
from sdg.commons import publish as common_publish
from sdg.commons import sources as common_sources
from sdg.commons.model import LLM, load_clients
from sdg.commons.run import load, run
from sdg.commons.run_log import log_event
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json
from sdg.commons.work_queue import map_async_ordered

PACK = "backtranslation_passages_dynaword"
_SOURCE_CONTEXTS_PATH = Path(__file__).resolve().parent / "source_contexts.json"

_DA_STOP = frozenset(
    [
        "og",
        "i",
        "at",
        "er",
        "en",
        "det",
        "til",
        "af",
        "som",
        "på",
        "med",
        "de",
        "den",
        "for",
        "ikke",
        "om",
        "vi",
        "da",
        "her",
        "men",
        "har",
        "fra",
        "var",
        "kan",
        "sig",
        "et",
        "der",
        "så",
        "også",
    ]
)
_EN_STOP = frozenset(
    [
        "the",
        "and",
        "of",
        "in",
        "is",
        "are",
        "this",
        "that",
        "be",
        "have",
        "it",
        "by",
        "an",
        "which",
        "were",
        "been",
        "their",
    ]
)
_CONTENT_STOP = frozenset(
    {
        "og",
        "eller",
        "som",
        "med",
        "uden",
        "over",
        "under",
        "after",
        "before",
        "fordi",
        "hvor",
        "hvorfor",
        "hvorvidt",
        "samt",
        "kan",
        "skal",
        "ville",
        "kunne",
        "about",
        "into",
        "from",
    }
)

_LENGTH_BUCKETS: list[str] = ["kort", "kort", "medium", "medium", "medium", "medium"]
_LENGTH_RULES: dict[str, str] = {
    "kort": "Prompten skal være MEGET KORT: 1 sætning og højst 220 tegn.",
    "medium": "Prompten skal være kort: højst 2 sætninger og højst 380 tegn.",
    "lang": "Prompten må gerne være lang og detaljeret — 5 sætninger eller mere.",
}
_SENSITIVE_TERMS = frozenset(
    [
        "tyske landsmænd",
        "tyske landsmaend",
        "führer",
        "furer",
        "weimartiden",
        "det demokratiske tyskland",
        "heil hitler",
        "naz",
        "nationalsozial",
        "furer",
        "führer",
        "racehygiejne",
        "arisk",
        "jøde",
        "jøder",
        "weimar",
    ]
)


def build(cfg: dict[str, Any]) -> BuildResult:
    return run(
        _build_run,
        pack=PACK,
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    cfg = _load_cfg(result)
    min_article_chars = _verify_min_article_chars(cfg)

    verified_rows = common_eval.verify(rows, _has_prompt, name="prompt_present")
    verified_rows = common_eval.verify(verified_rows, _has_target, name="target_present")
    verified_rows = common_eval.verify(
        verified_rows,
        lambda row: _target_meets_min_chars(row, min_article_chars),
        name="target_min_chars",
    )
    failures = [row for row in verified_rows if not _row_passes(row)]

    outputs_dir = Path(result.run_dir) / "outputs"
    store.write_jsonl(verified_rows, outputs_dir / "verified.jsonl")
    store.write_jsonl(failures, outputs_dir / "failures.jsonl")

    metrics = common_eval.aggregate_metrics(verified_rows)
    failure_summary = common_eval.summarize_failures(verified_rows)
    write_json(metrics, outputs_dir / "metrics.json")
    write_json(failure_summary, outputs_dir / "failure_summary.json")
    common_publish.write_preview(verified_rows, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "run_id": result.run_id,
        "verified_rows": len(verified_rows),
        "failed_rows": len(failures),
        "metrics": metrics,
        "failure_summary": failure_summary,
    }


def summarize(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    cfg = _load_cfg(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else common_eval.aggregate_metrics(rows)

    entries = _source_entries(cfg)
    source_summary: dict[str, Any]
    if len(entries) > 1:
        source_summary = {
            "sources": [
                {
                    "name": entry["name"],
                    "dataset": entry["source"].get("dataset"),
                    "path": entry["source"].get("path"),
                    "split": entry["source"].get("split", "train"),
                    "config_name": entry["source"].get("config_name"),
                    "source_type": entry["generation"].get("source_type", "news"),
                    "chunk_mode": entry["generation"].get("chunk_mode", "hybrid"),
                }
                for entry in entries
            ]
        }
    else:
        source_summary = _source_summary(entries[0]["source"])

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "source": source_summary,
        "artifacts": sorted(result.artifacts),
        "metrics": metrics,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    failures = [row for row in rows if not _row_passes(row)]
    train_rows, eval_rows = _split_rows(rows, _train_fraction(cfg))

    target_dir = _publish_dir(result, out_dir)
    store.ensure_dir(target_dir)
    store.write_parquet(train_rows, target_dir / "train.parquet")
    store.write_parquet(eval_rows, target_dir / "eval.parquet")
    store.write_parquet(failures, target_dir / "failures.parquet")
    common_publish.write_preview(rows, target_dir / "sample_preview.jsonl", n=20)

    outputs_dir = Path(result.run_dir) / "outputs"
    metrics = _load_or_compute(outputs_dir / "metrics.json", common_eval.aggregate_metrics(rows))
    failure_summary = _load_or_compute(outputs_dir / "failure_summary.json", common_eval.summarize_failures(rows))

    write_json(metrics, target_dir / "metrics.json")
    write_json(failure_summary, target_dir / "failure_summary.json")
    common_publish.write_report(metrics, failure_summary, target_dir / "report.json")
    common_publish.write_manifest(
        {
            "pack": result.pack,
            "run_id": result.run_id,
            "source_run_dir": result.run_dir,
            "source_artifacts": sorted(result.artifacts),
            "published_artifacts": [
                "train.parquet",
                "eval.parquet",
                "failures.parquet",
                "sample_preview.jsonl",
                "manifest.json",
                "metrics.json",
                "failure_summary.json",
                "report.json",
            ],
        },
        target_dir / "manifest.json",
    )

    return {
        "run_id": result.run_id,
        "out_dir": str(target_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "failure_rows": len(failures),
    }


def _build_run(
    *,
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
) -> dict[str, Artifact]:
    del seed

    source_contexts = _load_source_contexts()
    _validate_source_contexts(_source_entries(cfg), source_contexts)
    dataset_path = outputs_dir / "dataset.jsonl"
    failures_path = outputs_dir / "generation_failures.jsonl"
    resume_state = _load_resume_state(dataset_path=dataset_path, failures_path=failures_path)
    source_stats = _count_passages(cfg, processed_source_ids=resume_state["processed_source_ids"])
    writer = _load_instruction_writer(cfg)
    _progress_log(
        f"loaded {source_stats['pending_rows']} pending passages after scanning "
        f"{source_stats['scanned_rows']} rows"
    )
    completed_rows, failed_rows = _make_rows(
        cfg=cfg,
        writer=writer,
        source_contexts=source_contexts,
        dataset_path=dataset_path,
        failures_path=failures_path,
        processed_source_ids=resume_state["processed_source_ids"],
        pending_rows=source_stats["pending_rows"],
        starting_completed_rows=resume_state["completed_rows"],
        starting_failed_rows=resume_state["failed_rows"],
    )

    common_publish.write_preview(
        store.jsonl_prefix(dataset_path, limit=20),
        outputs_dir / "sample_preview.jsonl",
        n=20,
    )

    artifacts = {
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={
                "rows": completed_rows,
                "source": _source_artifact_label(cfg),
                "generation_failures": failed_rows,
                **source_stats,
            },
        )
    }
    artifacts["generation_failures"] = Artifact(
        name="generation_failures",
        path=str(failures_path),
        kind="jsonl",
        meta={"rows": failed_rows},
    )
    return artifacts


def _count_passages(
    cfg: dict[str, Any],
    *,
    processed_source_ids: set[str],
) -> dict[str, Any]:
    entries = _source_entries(cfg)
    if not entries:
        return {
            "scanned_rows": 0,
            "usable_rows": 0,
            "too_short_rows": 0,
            "resumed_rows": 0,
            "pending_rows": 0,
            "min_article_chars": 0,
            "max_articles": None,
            "sources": [],
        }

    scanned_rows = 0
    usable_rows = 0
    too_short_rows = 0
    resumed_rows = 0
    pending_rows = 0
    source_summaries: list[dict[str, Any]] = []

    for source_index, entry in enumerate(entries):
        source = entry["source"]
        generation = entry["generation"]
        min_article_chars = int(generation.get("min_article_chars", 0))
        max_articles = _max_articles(generation)
        source_scanned_rows = 0
        source_usable_rows = 0
        source_too_short_rows = 0
        source_resumed_rows = 0
        source_pending_rows = 0
        kept_articles = 0

        for record in common_sources.iter_source_records(source):
            source_scanned_rows += 1
            record_index = source_scanned_rows - 1
            article = _record_to_article(record, source, record_index)
            if article is None:
                continue

            usable_rows += 1
            source_usable_rows += 1
            if len(article["text"]) < min_article_chars:
                too_short_rows += 1
                source_too_short_rows += 1
                continue

            if max_articles is not None and kept_articles >= max_articles:
                break

            kept_articles += 1
            article = _with_source_entry(
                article,
                source=source,
                generation=generation,
                source_index=source_index,
                source_key=entry["key"],
                source_name=entry["name"],
            )
            for passage in _iter_chunks(article, source, generation, min_chars=min_article_chars):
                if passage["source_id"] in processed_source_ids:
                    resumed_rows += 1
                    source_resumed_rows += 1
                else:
                    pending_rows += 1
                    source_pending_rows += 1

        scanned_rows += source_scanned_rows
        source_summaries.append(
            {
                "key": entry["key"],
                "name": entry["name"],
                "source_type": generation.get("source_type", "news"),
                "chunk_mode": generation.get("chunk_mode", "hybrid"),
                "scanned_rows": source_scanned_rows,
                "usable_rows": source_usable_rows,
                "too_short_rows": source_too_short_rows,
                "resumed_rows": source_resumed_rows,
                "pending_rows": source_pending_rows,
                "min_article_chars": min_article_chars,
                "max_articles": max_articles,
                "source": _source_summary(source),
            }
        )

    return {
        "scanned_rows": scanned_rows,
        "usable_rows": usable_rows,
        "too_short_rows": too_short_rows,
        "resumed_rows": resumed_rows,
        "pending_rows": pending_rows,
        "min_article_chars": source_summaries[0]["min_article_chars"],
        "max_articles": source_summaries[0]["max_articles"],
        "sources": source_summaries,
    }


def _iter_passages(
    cfg: dict[str, Any],
    *,
    processed_source_ids: set[str],
):
    for source_index, entry in enumerate(_source_entries(cfg)):
        source = entry["source"]
        generation = entry["generation"]
        min_article_chars = int(generation.get("min_article_chars", 0))
        max_articles = _max_articles(generation)
        kept_rows = 0

        for index, record in enumerate(common_sources.iter_source_records(source)):
            article = _record_to_article(record, source, index)
            if article is None:
                continue
            if len(article["text"]) < min_article_chars:
                continue
            if max_articles is not None and kept_rows >= max_articles:
                break
            kept_rows += 1
            article = _with_source_entry(
                article,
                source=source,
                generation=generation,
                source_index=source_index,
                source_key=entry["key"],
                source_name=entry["name"],
            )
            for passage in _iter_chunks(article, source, generation, min_chars=min_article_chars):
                if passage["source_id"] in processed_source_ids:
                    continue
                yield passage


def _iter_chunks(
    article: dict[str, Any],
    source: dict[str, Any],
    generation: dict[str, Any],
    *,
    min_chars: int,
):
    del source

    chunk_mode = str(generation.get("chunk_mode", "hybrid")).strip().lower()
    full_text_max_chars = int(generation.get("full_text_max_chars", 1400))
    max_chars = int(generation.get("max_chars", 2500))
    max_passages = int(generation.get("max_passages_per_article", 0))
    text = str(article["text"])

    if chunk_mode in {"full", "single"}:
        passages = [text]
    elif chunk_mode == "hybrid" and len(text) <= full_text_max_chars:
        passages = [text]
    else:
        passages = _extract_passages(text, min_chars=min_chars, max_chars=max_chars)
        if not passages and len(text) >= min_chars:
            passages = [text[:max_chars].strip()]

    filtered_passages = [
        passage_text
        for passage_text in passages
        if _passage_is_usable_for_instruction_tuning(passage_text, min_chars=min_chars)
    ]
    if max_passages > 0:
        filtered_passages = filtered_passages[:max_passages]

    for i, passage_text in enumerate(filtered_passages):
        source_entry_key = str(article.get("source_entry_key", "single"))
        row_prefix = f"{source_entry_key}::{article['source_id']}"
        yield {
            **article,
            "source_id": f"{row_prefix}::p{i:03d}",
            "passage_idx": i,
            "text": passage_text,
        }


def _extract_passages(
    text: str,
    *,
    min_chars: int = 200,
    max_chars: int = 2500,
) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    passages: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def _seal() -> None:
        if current_parts:
            block = "\n\n".join(current_parts).strip()
            if len(block) >= min_chars:
                passages.append(block)
        current_parts.clear()

    for p in paragraphs:
        score = _score_paragraph_web(p)
        if score == 0.0:
            _seal()
            current_len = 0
            continue

        added = len(p) + (2 if current_parts else 0)
        if current_len + added > max_chars:
            _seal()
            current_len = 0
            p = p[:max_chars]

        current_parts.append(p)
        current_len += len(p) + (2 if len(current_parts) > 1 else 0)

    _seal()
    return passages


def _score_paragraph_web(p: str) -> float:
    if len(p) < 80:
        return 0.0
    if p.lstrip().startswith("<"):
        return 0.0

    stripped = p.lstrip()
    if stripped and stripped[0].islower():
        return 0.0

    if re.match(r"^\s*(foto|figur\s*\d|kilde)\s*:", p, re.IGNORECASE):
        return 0.0
    if re.search(r"\bFoto:\s+[A-ZÆØÅ]\w", p):
        return 0.0

    alpha_ratio = sum(c.isalpha() for c in p) / len(p)
    if alpha_ratio < 0.55:
        return 0.0

    lines = [ln for ln in p.splitlines() if ln.strip()]
    if lines and sum(ln.lstrip().startswith("|") for ln in lines) / len(lines) > 0.5:
        return 0.0

    if lines and sum(len(ln) < 60 for ln in lines) / len(lines) > 0.7:
        return 0.0

    markup_chars = len(re.findall(r"[#*`\[\]]", p))
    if markup_chars / len(p) > 0.05:
        return 0.0

    words_lower = re.findall(r"\b[a-z]{2,}\b", p[:600].lower())
    if words_lower:
        da_hits = sum(1 for w in words_lower if w in _DA_STOP)
        en_hits = sum(1 for w in words_lower if w in _EN_STOP)
        if en_hits > da_hits and en_hits >= 3:
            return 0.0

    length_score = min(len(p) / 1000, 1.0)
    return alpha_ratio * length_score


def _passage_is_usable_for_instruction_tuning(text: str, *, min_chars: int) -> bool:
    candidate = text.strip()
    if len(candidate) < min_chars:
        return False
    if _contains_sensitive_content(candidate):
        return False
    if _has_ocr_noise(candidate):
        return False
    if _is_too_repetitive(candidate):
        return False
    return True


def _contains_sensitive_content(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in _SENSITIVE_TERMS)


def _has_ocr_noise(text: str) -> bool:
    if not text:
        return True

    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.6:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        repeated_line_count = max(lines.count(line) for line in set(lines))
        if repeated_line_count >= 3:
            return True

    if re.search(r"\b([A-Za-zÆØÅæøå]{1,4})\s+\1\b", text):
        return True

    if re.search(r"\b[a-zæøå]{3,}\s+(fi|fl)\s+[a-zæøå]{1,4}\b", text.lower()):
        return True

    tokens = re.findall(r"\S+", text)
    if not tokens:
        return True

    mixed_tokens = [token for token in tokens if re.search(r"[A-Za-zÆØÅæøå]", token) and re.search(r"\d", token)]
    if len(mixed_tokens) / len(tokens) > 0.08:
        return True

    upper_tokens = [token for token in tokens if len(token) >= 4 and token.isupper()]
    if len(upper_tokens) / len(tokens) > 0.2:
        return True

    return False


def _is_too_repetitive(text: str) -> bool:
    words = re.findall(r"\b[A-Za-zÆØÅæøå]{3,}\b", text.lower())
    if len(words) < 30:
        return False

    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.28:
        return True

    max_run = 1
    current_run = 1
    for prev, curr in zip(words, words[1:]):
        if prev == curr:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
            continue
        current_run = 1

    return max_run >= 6


def _instruction_messages(
    article: dict[str, Any],
    source_context: dict[str, Any],
    *,
    prompt_length: str,
) -> list[dict[str, str]]:
    source_type = str(article.get("source_type", source_context.get("source_type", "news"))).strip().lower()
    if source_type == "tax_guidance":
        return _instruction_messages_tax_guidance(article, source_context, prompt_length=prompt_length)
    if source_type == "government":
        return _instruction_messages_government(article, source_context, prompt_length=prompt_length)
    if source_type == "speech":
        return _instruction_messages_speech(article, source_context, prompt_length=prompt_length)
    return _instruction_messages_news(article, source_context, prompt_length=prompt_length)


def _instruction_messages_news(
    article: dict[str, Any],
    source_context: dict[str, Any],
    *,
    prompt_length: str,
) -> list[dict[str, str]]:
    length_rule = _LENGTH_RULES.get(prompt_length, _LENGTH_RULES["medium"])
    system_lines = [
        "Du er en assistent der laver træningsdata til sprogmodeller.",
        "Du svarer KUN med selve prompten — ingen forklaringer, overskrifter eller meta-kommentarer.",
        length_rule,
        "Prompten skal ligne en realistisk brugerbesked til en moderne chatbot: konkret, handlingsorienteret og uden rollespil.",
        "Prompten skal være selvstændig og må IKKE referere til specifikke artikler, dokumenter eller tekster der ikke er vedlagt.",
        "Prompten må ikke genbruge konkrete navne, datoer eller unikke hændelser fra teksten.",
        "Undgå kreative metaforer, slogans og reklamesprog i prompten.",
    ]
    return _build_messages(article, source_context, system_lines, "Skriv en prompt der ville få en AI til at skrive ovenstående nyhedstekst.")


def _instruction_messages_speech(
    article: dict[str, Any],
    source_context: dict[str, Any],
    *,
    prompt_length: str,
) -> list[dict[str, str]]:
    length_rule = _LENGTH_RULES.get(prompt_length, _LENGTH_RULES["medium"])
    system_lines = [
        "Du er en assistent der laver træningsdata til sprogmodeller.",
        "Du svarer KUN med selve prompten — ingen forklaringer, overskrifter eller meta-kommentarer.",
        length_rule,
        "Prompten skal bede om en tale, debatindlæg, parlamentarisk indlæg eller offentlig redegørelse.",
        "Prompten skal fokusere på emne, målgruppe og tone, ikke på at reproducere en specifik historisk tale.",
        "Prompten må ikke genbruge konkrete navne, datoer eller unikke hændelser fra teksten.",
        "Prompten skal være selvstændig og uden henvisning til den vedlagte tekst.",
        "Undgå vinkler der opfordrer til had, diskrimination eller ekstremisme.",
    ]
    return _build_messages(article, source_context, system_lines, "Skriv en prompt der ville få en AI til at skrive ovenstående tale- eller debattekst.")


def _instruction_messages_government(
    article: dict[str, Any],
    source_context: dict[str, Any],
    *,
    prompt_length: str,
) -> list[dict[str, str]]:
    length_rule = _LENGTH_RULES.get(prompt_length, _LENGTH_RULES["medium"])
    system_lines = [
        "Du er en assistent der laver træningsdata til sprogmodeller.",
        "Du svarer KUN med selve prompten — ingen forklaringer, overskrifter eller meta-kommentarer.",
        length_rule,
        "Prompten skal bede om en administrativ, offentlig eller myndighedsnær tekst på klart dansk.",
        "Prompten må gerne fokusere på regler, processer, vejledning eller offentlig information.",
        "Prompten må ikke genbruge konkrete navne, datoer eller unikke hændelser fra teksten.",
        "Prompten skal være selvstændig og uden henvisning til den vedlagte tekst.",
    ]
    return _build_messages(article, source_context, system_lines, "Skriv en prompt der ville få en AI til at skrive ovenstående offentlige informations- eller myndighedstekst.")


def _instruction_messages_tax_guidance(
    article: dict[str, Any],
    source_context: dict[str, Any],
    *,
    prompt_length: str,
) -> list[dict[str, str]]:
    length_rule = _LENGTH_RULES.get(prompt_length, _LENGTH_RULES["medium"])
    system_lines = [
        "Du er en assistent der laver træningsdata til sprogmodeller.",
        "Du svarer KUN med selve prompten — ingen forklaringer, overskrifter eller meta-kommentarer.",
        length_rule,
        "Prompten skal ligne en praktisk borger- eller virksomhedsforespørgsel om skat.",
        "Prompten skal typisk bede om trinvis vejledning, frister, dokumentkrav eller konkrete eksempler.",
        "Prompten må IKKE være juridisk redegørelse eller anmode om detaljeret lovfortolkning.",
        "Prompten må ikke genbruge konkrete navne, datoer eller unikke hændelser fra teksten.",
        "Prompten skal være selvstændig og uden henvisning til den vedlagte tekst.",
    ]
    return _build_messages(article, source_context, system_lines, "Skriv en prompt der ville få en AI til at skrive ovenstående skattevejledning i et praktisk format.")


def _build_messages(
    article: dict[str, Any],
    source_context: dict[str, Any],
    system_lines: list[str],
    instruction: str,
) -> list[dict[str, str]]:
    user_lines: list[str] = []
    if article.get("title"):
        user_lines.append(f"Titel: {article['title']}")
    user_lines.append(article["text"])

    context_payload = json.dumps(source_context, ensure_ascii=False, sort_keys=True)
    user_content = (
        f"Kildekontekst (til stilforståelse):\n{context_payload}\n\n"
        + "\n".join(user_lines)
        + f"\n\n---\n{instruction}"
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
    text = _html.unescape(
        common_sources.read_record_value(record, source.get("text_field", "text")) or ""
    )
    if not text:
        return None

    title = common_sources.read_record_value(record, source.get("title_field", "title"))
    url = common_sources.read_record_value(record, source.get("url_field", "url"))
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
    }


def _load_instruction_writer(cfg: dict[str, Any]) -> LLM:
    models = load_clients({"instruction_writer": cfg["models"]["instruction_writer"]})
    writer = models["instruction_writer"]
    assert isinstance(writer, LLM)
    return writer


def _make_rows(
    *,
    cfg: dict[str, Any],
    writer: LLM,
    source_contexts: dict[str, dict[str, Any]],
    dataset_path: Path,
    failures_path: Path,
    processed_source_ids: set[str],
    pending_rows: int,
    starting_completed_rows: int,
    starting_failed_rows: int,
) -> tuple[int, int]:
    return asyncio.run(
        _make_rows_async(
            cfg=cfg,
            writer=writer,
            source_contexts=source_contexts,
            dataset_path=dataset_path,
            failures_path=failures_path,
            processed_source_ids=processed_source_ids,
            pending_rows=pending_rows,
            starting_completed_rows=starting_completed_rows,
            starting_failed_rows=starting_failed_rows,
        )
    )


async def _make_rows_async(
    *,
    cfg: dict[str, Any],
    writer: LLM,
    source_contexts: dict[str, dict[str, Any]],
    dataset_path: Path,
    failures_path: Path,
    processed_source_ids: set[str],
    pending_rows: int,
    starting_completed_rows: int,
    starting_failed_rows: int,
) -> tuple[int, int]:
    temperature = _generation_temperature(cfg)
    worker_concurrency = common_concurrency.runtime_concurrency(writer)
    completed_rows = 0
    failed_rows = 0
    starting_processed_rows = starting_completed_rows + starting_failed_rows
    total_rows = starting_processed_rows + pending_rows

    _progress_log(f"generating prompts with worker_concurrency={worker_concurrency}")
    log_event(
        PACK,
        "generation_started",
        rows=total_rows,
        pending_rows=pending_rows,
        worker_concurrency=worker_concurrency,
    )
    progress, progress_state = _progress_reporter(
        total=total_rows,
        starting_completed=starting_processed_rows,
        worker_concurrency=worker_concurrency,
    )

    dataset_mode = "a" if dataset_path.exists() else "w"
    failures_mode = "a" if failures_path.exists() else "w"
    with dataset_path.open(dataset_mode) as dataset_handle, failures_path.open(failures_mode) as failures_handle:
        async for row in map_async_ordered(
            _iter_passages(cfg, processed_source_ids=processed_source_ids),
            lambda _ignored_index, passage: _generate_row_result(
                article=passage,
                writer=writer,
                temperature=temperature,
                source_contexts=source_contexts,
            ),
            concurrency=worker_concurrency,
            progress=progress,
            total=pending_rows,
        ):
            if "row" in row:
                store.append_jsonl_line(dataset_handle, row["row"])
                completed_rows += 1
                continue
            store.append_jsonl_line(failures_handle, row["failure"])
            failed_rows += 1

    _finish_progress(
        total=total_rows,
        worker_concurrency=worker_concurrency,
        elapsed_seconds=progress_state["elapsed_seconds"],
    )
    log_event(
        PACK,
        "generation_completed",
        rows=starting_completed_rows + completed_rows,
        failures=starting_failed_rows + failed_rows,
        worker_concurrency=worker_concurrency,
    )
    _progress_log(
        f"generated {starting_completed_rows + completed_rows} rows, "
        f"recorded {starting_failed_rows + failed_rows} failures"
    )

    return starting_completed_rows + completed_rows, starting_failed_rows + failed_rows


async def _generate_row(
    *,
    article: dict[str, Any],
    writer: LLM,
    temperature: float,
    source_contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    length_key = (article.get("record_index", 0) + article.get("passage_idx", 0)) % len(_LENGTH_BUCKETS)
    prompt_length = _LENGTH_BUCKETS[length_key]
    source_name = str(article.get("source_name", ""))
    source_context = source_contexts.get(source_name, {})
    messages = _instruction_messages(article, source_context, prompt_length=prompt_length)

    cleaned_prompt: str | None = None
    last_error: Exception | None = None
    for _ in range(3):
        prompt = await _generate_prompt(writer, messages, temperature=temperature)
        try:
            cleaned_prompt = _clean_generated_prompt(prompt)
            if _prompt_leaks_target_details(cleaned_prompt, article["text"]):
                raise ValueError("prompt appears to leak source-specific details")
            break
        except ValueError as error:
            last_error = error

    if cleaned_prompt is None:
        assert last_error is not None
        raise last_error

    passage_idx = article.get("passage_idx", 0)
    source_key = str(article.get("source_entry_key", "single"))
    row_hash = hashlib.sha1(f"{source_key}:{article['record_index']}:{passage_idx}".encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"{PACK}-{row_hash}",
        "prompt": cleaned_prompt,
        "target": article["text"],
        "sources": [
            _drop_none(
                {
                    "dataset": article.get("source_dataset"),
                    "path": article.get("source_path"),
                    "split": article.get("source_split", "train"),
                    "config_name": article.get("source_config_name"),
                    "row_id": article["source_id"],
                    "title": article["title"],
                    "url": article["url"],
                }
            )
        ],
        "meta": _drop_none(
            {
                "title": article["title"],
                "source_id": article["source_id"],
                "source_key": article.get("source_entry_key"),
                "source_name": article.get("source_name"),
                "source_type": article.get("source_type"),
                "source_dataset": article.get("source_dataset"),
                "source_path": article.get("source_path"),
                "source_split": article.get("source_split", "train"),
                "source_config_name": article.get("source_config_name"),
                "source_record_index": article["record_index"],
                "passage_idx": passage_idx,
                "target_chars": len(article["text"]),
            }
        ),
    }


async def _generate_row_result(
    *,
    article: dict[str, Any],
    writer: LLM,
    temperature: float,
    source_contexts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        row = await _generate_row(
            article=article,
            writer=writer,
            temperature=temperature,
            source_contexts=source_contexts,
        )
    except Exception as error:
        source_key = str(article.get("source_entry_key", "single"))
        failure_hash = hashlib.sha1(
            f"{source_key}:{article['record_index']}:{article.get('passage_idx', 0)}:failure".encode("utf-8")
        ).hexdigest()[:12]
        log_event(
            PACK,
            "row_failed",
            record_index=article["record_index"],
            passage_idx=article.get("passage_idx", 0),
            source_id=article["source_id"],
            error_type=error.__class__.__name__,
            error_message=str(error),
        )
        return {
            "failure": {
                "id": f"{PACK}-failure-{failure_hash}",
                "source_id": article["source_id"],
                "title": article["title"],
                "url": article["url"],
                "error_type": error.__class__.__name__,
                "error_message": str(error),
            }
        }
    return {"row": row}


async def _generate_prompt(
    writer: LLM,
    messages: list[dict[str, str]],
    *,
    temperature: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            achat = getattr(writer, "achat", None)
            if callable(achat):
                return await achat(messages, temperature=temperature)
            return await asyncio.to_thread(writer.chat, messages, temperature=temperature)
        except Exception as error:
            last_error = error
            if attempt == 2:
                break
            await asyncio.sleep(float(attempt + 1))

    assert last_error is not None
    raise last_error


def _max_articles(generation: dict[str, Any]) -> int | None:
    value = generation.get("max_articles")
    if value is None:
        return None

    max_articles = int(value)
    assert max_articles >= 0, "generation.max_articles must be non-negative"
    return max_articles


def _progress_log(message: str) -> None:
    print(f"[{PACK}] {message}", flush=True)


def _progress_reporter(
    *,
    total: int,
    starting_completed: int,
    worker_concurrency: int,
):
    state = {"finished": False, "elapsed_seconds": 0}
    common_progress.write_progress_snapshot(
        f"{PACK}_progress",
        stage="generating_prompts",
        completed=starting_completed,
        total=total,
        elapsed_seconds=0,
        extra={
            "worker_concurrency": worker_concurrency,
            "rows_per_minute": common_progress.items_per_minute(starting_completed, 0),
        },
        force=True,
    )
    _print_progress_bar(starting_completed, total, worker_concurrency, 0)
    snapshot_progress = common_progress.snapshot_progress_reporter(
        f"{PACK}_progress",
        stage="generating_prompts",
        completed_offset=starting_completed,
        total=total,
        extra=lambda completed, _total, elapsed: {
            "worker_concurrency": worker_concurrency,
            "rows_per_minute": common_progress.items_per_minute(completed, elapsed),
        },
    )

    def report(completed: int, _reported_total: int | None, elapsed: int) -> None:
        state["elapsed_seconds"] = elapsed
        current_completed = starting_completed + completed
        snapshot_progress(completed, _reported_total, elapsed)
        _print_progress_bar(current_completed, total, worker_concurrency, elapsed)
        if current_completed >= total and not state["finished"]:
            sys.stdout.write("\n")
            sys.stdout.flush()
            state["finished"] = True

    return report, state


def _finish_progress(
    *,
    total: int,
    worker_concurrency: int,
    elapsed_seconds: int,
) -> None:
    common_progress.write_progress_snapshot(
        f"{PACK}_progress",
        stage="completed",
        completed=total,
        total=total,
        elapsed_seconds=elapsed_seconds,
        extra={
            "worker_concurrency": worker_concurrency,
            "rows_per_minute": common_progress.items_per_minute(total, elapsed_seconds),
        },
        force=True,
    )


def _print_progress_bar(
    completed: int,
    total: int,
    worker_concurrency: int,
    elapsed_seconds: int,
) -> None:
    width = 24
    filled = width if total == 0 else min(int(width * completed / total), width)
    bar = "#" * filled + "-" * (width - filled)
    total_text = str(total)
    message = (
        f"\r[{PACK}] [{bar}] {completed}/{total_text} "
        f"elapsed={elapsed_seconds}s concurrency={worker_concurrency}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def _load_resume_state(
    *,
    dataset_path: Path,
    failures_path: Path,
) -> dict[str, Any]:
    completed_source_ids = store.jsonl_keys(dataset_path, key_for=_source_id_from_output_row)
    failed_source_ids = store.jsonl_keys(failures_path, key_for=_source_id_from_failure_row)
    processed_source_ids = set(completed_source_ids)
    processed_source_ids.update(failed_source_ids)

    return {
        "processed_source_ids": processed_source_ids,
        "completed_rows": len(completed_source_ids),
        "failed_rows": len(failed_source_ids - completed_source_ids),
    }


def _source_id_from_output_row(row: dict[str, Any]) -> str | None:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return None
    source_id = meta.get("source_id")
    if not isinstance(source_id, str):
        return None
    return source_id


def _source_id_from_failure_row(row: dict[str, Any]) -> str | None:
    source_id = row.get("source_id")
    if not isinstance(source_id, str):
        return None
    return source_id


def _clean_generated_prompt(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"^\*\*[^*]{1,40}\*\*\s*:?\s*\n+", "", text).strip()
    if text.startswith('**"'):
        match = re.match(r'^\*\*"(.+?)"?\*\*', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            text = text[3:].strip()
    text = re.sub(r"\s+", " ", text).strip()
    if not text or text[-1] not in {"?", "!", "."}:
        raise ValueError(f"generated prompt has no terminal punctuation: {text[:120]!r}")
    if len(text) > 600:
        raise ValueError(f"prompt too long for chatbot usage: {text[:120]!r}")
    if re.search(
        r"(formuler en prompt|skriv en prompt der|få en (AI|sprogmodel) til at (skrive|generere))",
        text,
        re.IGNORECASE,
    ):
        raise ValueError(f"meta-prompt detected: {text[:120]!r}")
    if re.search(r"\b(figur\s+\d|tabel\s+\d)\b", text, re.IGNORECASE):
        raise ValueError(f"document-referential prompt: {text[:120]!r}")
    words = re.findall(r"\b[a-z]{2,}\b", text[:300].lower())
    if words:
        en_hits = sum(1 for w in words if w in _EN_STOP)
        da_hits = sum(1 for w in words if w in _DA_STOP)
        if en_hits >= 4 and en_hits > da_hits:
            raise ValueError(f"prompt appears to be in English: {text[:120]!r}")
    if _contains_shouting_token(text):
        raise ValueError(f"prompt contains shouting token: {text[:120]!r}")
    if not _looks_like_user_request(text):
        raise ValueError(f"prompt does not look like user request: {text[:120]!r}")
    if re.search(r"\bdu\s+er\b", text, re.IGNORECASE):
        raise ValueError(f"roleplay-style prompt rejected: {text[:120]!r}")
    if re.search(r"\b(som\s+en|ligesom|som\s+om|et\s+strejf\s+af)\b", text, re.IGNORECASE):
        raise ValueError(f"metaphor-style prompt rejected: {text[:120]!r}")
    return text


def _looks_like_user_request(text: str) -> bool:
    if "?" in text:
        return True
    lowered = text.lower().strip()
    if lowered.startswith(("kan du", "kan i", "skriv", "lav", "hjælp mig med", "jeg skal bruge")):
        return True
    return bool(
        re.search(
            r"\b(kan\s+du|kan\s+vi|skriv|lav|udarbejd|hjælp|giv|forklar|opsummer|formuler|foreslå|generer|beskriv|oversæt|fremlæg|redegør|gennemgå)\b",
            text,
            re.IGNORECASE,
        )
    )


def _contains_shouting_token(text: str) -> bool:
    allowed = {"AI", "USA", "EU", "SKAT", "NIS2", "CITES", "IDA", "HOFOR"}
    for token in re.findall(r"\b[A-ZÆØÅ]{4,}\b", text):
        if token not in allowed:
            return True
    return False


def _prompt_leaks_target_details(prompt: str, target: str) -> bool:
    prompt_words = set(_content_words(prompt))
    target_words = set(_content_words(target))
    if not prompt_words or not target_words:
        return False

    overlap = prompt_words & target_words
    if len(overlap) >= 5:
        return True

    if prompt_has_dates_or_numbers(prompt) and shared_numeric_tokens(prompt, target) >= 2:
        return True

    return False


def _content_words(text: str) -> list[str]:
    words = re.findall(r"\b[A-Za-zÆØÅæøå]{5,}\b", text.lower())
    return [word for word in words if word not in _CONTENT_STOP and word not in _DA_STOP and word not in _EN_STOP]


def prompt_has_dates_or_numbers(text: str) -> bool:
    return bool(re.search(r"\b\d{2,4}\b", text))


def shared_numeric_tokens(prompt: str, target: str) -> int:
    prompt_nums = set(re.findall(r"\b\d{2,4}\b", prompt))
    if not prompt_nums:
        return 0
    target_nums = set(re.findall(r"\b\d{2,4}\b", target))
    return len(prompt_nums & target_nums)


def _source_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if "sources" in cfg:
        entries = list(cfg["sources"])
        assert entries, "sources must not be empty"
        normalized = []
        seen_keys: set[str] = set()
        for index, item in enumerate(entries):
            assert isinstance(item, dict), "sources entries must be mappings"
            assert "source" in item, "sources entries must include source"
            source = dict(item["source"])
            generation = _entry_generation(cfg, item)
            raw_name = item.get("name") or source.get("config_name") or source.get("dataset") or source.get("path")
            name = str(raw_name).strip()
            assert name, "sources entries require a non-empty name"
            key = _normalize_source_key(name, index=index)
            assert key not in seen_keys, f"Duplicate source key: {key}"
            seen_keys.add(key)
            normalized.append({"key": key, "name": name, "source": source, "generation": generation})
        return normalized

    assert "source" in cfg, "config must include source or sources"
    assert "generation" in cfg, "config must include generation or sources entries"
    source = dict(cfg["source"])
    generation = dict(cfg["generation"])
    raw_name = source.get("config_name") or source.get("name") or source.get("dataset") or source.get("path")
    name = str(raw_name).strip()
    assert name, "source requires config_name or name"
    return [{"key": _normalize_source_key(name, index=0), "name": name, "source": source, "generation": generation}]


def _entry_generation(cfg: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    base_generation = dict(cfg.get("generation", {}))
    if "generation" in entry:
        base_generation.update(dict(entry["generation"]))
    return base_generation


def _normalize_source_key(value: Any, *, index: int) -> str:
    raw = "" if value is None else str(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    if normalized:
        return normalized
    return f"source-{index + 1}"


def _source_summary(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": source.get("dataset"),
        "path": source.get("path"),
        "split": source.get("split", "train"),
        "config_name": source.get("config_name"),
    }


def _with_source_entry(
    article: dict[str, Any],
    *,
    source: dict[str, Any],
    generation: dict[str, Any],
    source_index: int,
    source_key: str,
    source_name: str,
) -> dict[str, Any]:
    source_entry = _drop_none(
        {
            "source_entry_index": source_index,
            "source_entry_key": source_key,
            "source_name": source_name,
            "source_type": generation.get("source_type", "news"),
            "source_dataset": source.get("dataset"),
            "source_path": source.get("path"),
            "source_split": source.get("split", "train"),
            "source_config_name": source.get("config_name"),
            "chunk_mode": generation.get("chunk_mode", "hybrid"),
        }
    )
    return {**article, **source_entry}


def _source_artifact_label(cfg: dict[str, Any]) -> str:
    entries = _source_entries(cfg)
    if len(entries) == 1:
        return common_sources.source_label(entries[0]["source"])
    return f"multi-source:{len(entries)}"


def _generation_temperature(cfg: dict[str, Any]) -> float:
    entries = _source_entries(cfg)
    assert entries, "sources must not be empty"
    temperatures = [float(entry["generation"].get("temperature", 0.2)) for entry in entries]
    if len(set(temperatures)) != 1:
        raise ValueError("build requires consistent generation.temperature across sources")
    return temperatures[0]


def _verify_min_article_chars(cfg: dict[str, Any]) -> int:
    if "generation" in cfg:
        return int(cfg["generation"].get("min_article_chars", 0))
    entries = _source_entries(cfg)
    if not entries:
        return 0
    mins = [int(entry["generation"].get("min_article_chars", 0)) for entry in entries]
    if len(set(mins)) != 1:
        raise ValueError("verify requires consistent generation.min_article_chars across sources")
    return mins[0]


def _train_fraction(cfg: dict[str, Any]) -> float:
    if "generation" in cfg:
        return float(cfg["generation"]["train_fraction"])
    entries = _source_entries(cfg)
    assert entries, "sources must not be empty"
    fractions = [float(entry["generation"]["train_fraction"]) for entry in entries]
    if len(set(fractions)) != 1:
        raise ValueError("publish requires consistent generation.train_fraction across sources")
    return fractions[0]


def _load_rows(result: BuildResult) -> list[dict[str, Any]]:
    dataset_path = Path(result.artifacts["dataset"].path)
    return store.read_jsonl(dataset_path)


def _load_verified_rows(result: BuildResult) -> list[dict[str, Any]]:
    verified_path = Path(result.run_dir) / "outputs" / "verified.jsonl"
    if verified_path.exists():
        return store.read_jsonl(verified_path)

    rows = _load_rows(result)
    cfg = _load_cfg(result)
    min_article_chars = _verify_min_article_chars(cfg)
    verified_rows = common_eval.verify(rows, _has_prompt, name="prompt_present")
    verified_rows = common_eval.verify(verified_rows, _has_target, name="target_present")
    return common_eval.verify(
        verified_rows,
        lambda row: _target_meets_min_chars(row, min_article_chars),
        name="target_min_chars",
    )


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _has_prompt(row: dict[str, Any]) -> bool:
    return bool(str(row.get("prompt", "")).strip())


def _has_target(row: dict[str, Any]) -> bool:
    return bool(str(row.get("target", "")).strip())


def _target_meets_min_chars(row: dict[str, Any], min_article_chars: int) -> bool:
    return len(str(row.get("target", "")).strip()) >= min_article_chars


def _row_passes(row: dict[str, Any]) -> bool:
    checks = row.get("checks", {})
    assert isinstance(checks, dict), "row checks must be a mapping"
    return all(_check_passed(value) for value in checks.values())


def _check_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value

    assert isinstance(value, dict), "check value must be a bool or a status mapping"
    if "passed" in value:
        return bool(value["passed"])
    if "ok" in value:
        return bool(value["ok"])
    raise AssertionError("check status mapping must contain 'passed' or 'ok'")


def _split_rows(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_at = int(len(rows) * train_fraction)
    return rows[:split_at], rows[split_at:]


def _publish_dir(result: BuildResult, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    return reports_root() / result.pack / result.run_id


def _load_or_compute(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return fallback


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _load_source_contexts() -> dict[str, dict[str, Any]]:
    payload = read_json(_SOURCE_CONTEXTS_PATH)
    assert isinstance(payload, dict), "source_contexts.json must be a JSON object"
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        assert isinstance(key, str) and key.strip(), "source context keys must be non-empty strings"
        assert isinstance(value, dict), "source context entries must be mappings"
        normalized[key] = value
    return normalized


def _validate_source_contexts(entries: list[dict[str, Any]], source_contexts: dict[str, dict[str, Any]]) -> None:
    missing = sorted(entry["name"] for entry in entries if entry["name"] not in source_contexts)
    assert not missing, f"Missing source contexts for: {', '.join(missing)}"
