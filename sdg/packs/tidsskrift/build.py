from __future__ import annotations

import asyncio
import html as _html
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

_JOURNALS_PATH = Path(__file__).parent / "data" / "journals.json"


def _load_journals() -> dict[str, dict[str, Any]]:
    return read_json(_JOURNALS_PATH)


_JOURNALS: dict[str, dict[str, Any]] = _load_journals()

# Stop-word sets for language detection in _score_paragraph.
_DA_STOP = frozenset(
    ["og", "i", "at", "er", "en", "det", "til", "af", "som", "på", "med", "de", "den",
     "for", "ikke", "om", "vi", "da", "her", "men", "har", "fra", "var", "kan", "sig",
     "et", "der", "så", "også"]
)
_EN_STOP = frozenset(
    ["the", "and", "of", "in", "is", "are", "this", "that", "be", "have", "it", "by",
     "an", "which", "were", "been", "their"]
)

# Deterministic length buckets: 50% short, 33% medium, 17% long.
_LENGTH_BUCKETS: list[str] = ["kort", "kort", "kort", "medium", "medium", "lang"]

_LENGTH_RULES: dict[str, str] = {
    "kort": "Prompten skal være MEGET KORT: maksimalt 1-2 sætninger. Kortere er bedre.",
    "medium": "Prompten skal være 2-4 sætninger lang.",
    "lang": "Prompten må gerne være lang og detaljeret — 5 sætninger eller mere.",
}


def build(cfg: dict[str, Any]) -> BuildResult:
    return run(
        _build_run,
        pack="tidsskrift",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    cfg = _load_cfg(result)
    min_article_chars = int(cfg["generation"].get("min_article_chars", 0))

    verified_rows = common_eval.verify(rows, _has_prompt, name="prompt_present")
    verified_rows = common_eval.verify(verified_rows, _has_target, name="target_present")
    verified_rows = common_eval.verify(
        verified_rows,
        lambda row: _target_meets_min_chars(row, min_article_chars),
        name="target_min_chars",
    )
    verified_rows = common_eval.verify(
        verified_rows,
        _target_is_not_references,
        name="target_not_references",
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

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "source": {
            "dataset": cfg["source"].get("dataset"),
            "path": cfg["source"].get("path"),
            "split": cfg["source"].get("split", "train"),
        },
        "artifacts": sorted(result.artifacts),
        "metrics": metrics,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    verified_rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    failures = [row for row in verified_rows if not _row_passes(row)]
    rows = [row for row in verified_rows if _row_passes(row)]
    train_rows, eval_rows = _split_rows(rows, cfg["generation"]["train_fraction"])

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


def _load_personas(n: int) -> list[str]:
    """Stream the first n personas from nvidia/Nemotron-Personas-USA."""
    import itertools

    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train", streaming=True)
    return [str(row["persona"]) for row in itertools.islice(ds, n)]  # type: ignore[arg-type]


def _build_run(
    *,
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
) -> dict[str, Artifact]:
    del seed

    dataset_path = outputs_dir / "dataset.jsonl"
    failures_path = outputs_dir / "generation_failures.jsonl"
    resume_state = _load_resume_state(dataset_path=dataset_path, failures_path=failures_path)
    source_stats = _count_passages(cfg, processed_source_ids=resume_state["processed_source_ids"])
    writer = _load_instruction_writer(cfg)
    _progress_log(
        f"loaded {source_stats['pending_rows']} pending passages from "
        f"{source_stats['scanned_docs']} documents"
    )
    _progress_log("loading personas from nvidia/Nemotron-Personas-USA...")
    personas = _load_personas(max(source_stats["pending_rows"], 1))
    _progress_log(f"loaded {len(personas)} personas")
    completed_rows, failed_rows = _make_rows(
        cfg=cfg,
        writer=writer,
        personas=personas,
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

    return {
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={
                "rows": completed_rows,
                "source": common_sources.source_label(cfg["source"]),
                "generation_failures": failed_rows,
                **source_stats,
            },
        ),
        "generation_failures": Artifact(
            name="generation_failures",
            path=str(failures_path),
            kind="jsonl",
            meta={"rows": failed_rows},
        ),
    }


def _count_passages(
    cfg: dict[str, Any],
    *,
    processed_source_ids: set[str],
) -> dict[str, Any]:
    source = cfg["source"]
    generation = cfg["generation"]
    max_articles = _max_articles(generation)
    max_per_journal = _max_articles_per_journal(generation)
    scanned_docs = 0
    usable_docs = 0
    total_passages = 0
    resumed_passages = 0
    pending_passages = 0
    kept_docs = 0
    kept_per_journal: dict[str, int] = {}

    for record in common_sources.iter_source_records(source):
        scanned_docs += 1
        article = _record_to_article(record, source, scanned_docs - 1)
        if article is None:
            continue

        journal = article["journal"] or ""
        if max_per_journal is not None and kept_per_journal.get(journal, 0) >= max_per_journal:
            continue
        if max_articles is not None and kept_docs >= max_articles:
            break

        passages = _extract_passages(article["text"])
        if not passages:
            continue

        usable_docs += 1
        kept_docs += 1
        kept_per_journal[journal] = kept_per_journal.get(journal, 0) + 1

        for p_idx, _ in enumerate(passages):
            source_id = f"{article['source_id']}::p{p_idx}"
            total_passages += 1
            if source_id in processed_source_ids:
                resumed_passages += 1
            else:
                pending_passages += 1

    return {
        "scanned_docs": scanned_docs,
        "usable_docs": usable_docs,
        "total_passages": total_passages,
        "resumed_passages": resumed_passages,
        "pending_rows": pending_passages,
    }


def _iter_passages(
    cfg: dict[str, Any],
    *,
    processed_source_ids: set[str],
):
    """Yield one entry per (article, passage) pair, skipping already-processed ones."""
    source = cfg["source"]
    generation = cfg["generation"]
    max_articles = _max_articles(generation)
    max_per_journal = _max_articles_per_journal(generation)
    kept_docs = 0
    kept_per_journal: dict[str, int] = {}

    for record_index, record in enumerate(common_sources.iter_source_records(source)):
        article = _record_to_article(record, source, record_index)
        if article is None:
            continue

        journal = article["journal"] or ""
        if max_per_journal is not None and kept_per_journal.get(journal, 0) >= max_per_journal:
            continue
        if max_articles is not None and kept_docs >= max_articles:
            break

        min_chars = max(int(generation.get("min_article_chars", 0)), 200)
        passages = _extract_passages(article["text"], min_chars=min_chars)
        if not passages:
            continue

        kept_docs += 1
        kept_per_journal[journal] = kept_per_journal.get(journal, 0) + 1

        for p_idx, passage in enumerate(passages):
            source_id = f"{article['source_id']}::p{p_idx}"
            if source_id in processed_source_ids:
                continue
            yield {
                **article,
                "passage": passage,
                "source_id": source_id,
                "passage_idx": p_idx,
                "row_index": p_idx,
            }


def _record_to_article(
    record: dict[str, Any],
    source: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    text = common_sources.read_record_value(record, source.get("text_field", "text"))
    if not text:
        return None
    if _looks_garbled(text):
        return None

    # Decode HTML entities (e.g. &amp; → &, &lt; → <) before any other processing.
    text = _html.unescape(text)

    # Clean image artifacts upfront so both passage extraction and LLM see clean text.
    text = re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)
    # Normalize repeated spaces (PDF justified-text artifact) but preserve newlines.
    text = re.sub(r" {2,}", " ", text)
    # Fix PDF soft-hyphen word-breaks: "ideolo -gier" → "ideologier", "mar -ked" → "marked"
    text = re.sub(r"([a-zæøåA-ZÆØÅ]) -([a-zæøåA-ZÆØÅ])", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    title = common_sources.read_record_value(record, source.get("title_field", "title"))
    url = common_sources.read_record_value(record, source.get("url_field", "url"))
    journal = common_sources.read_record_value(record, source.get("journal_field", "journal"))
    journal_description = common_sources.read_record_value(
        record, source.get("journal_description_field", "journal_description")
    )
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
        "journal": journal,
        "journal_description": journal_description,
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
    personas: list[str],
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
            personas=personas,
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
    personas: list[str],
    dataset_path: Path,
    failures_path: Path,
    processed_source_ids: set[str],
    pending_rows: int,
    starting_completed_rows: int,
    starting_failed_rows: int,
) -> tuple[int, int]:
    generation = cfg["generation"]
    temperature = float(generation.get("temperature", 0.7))
    source = cfg["source"]
    worker_concurrency = common_concurrency.runtime_concurrency(writer)
    completed_rows = 0
    failed_rows = 0
    starting_processed_rows = starting_completed_rows + starting_failed_rows
    total_rows = starting_processed_rows + pending_rows

    _progress_log(f"generating prompts with worker_concurrency={worker_concurrency}")
    log_event(
        "tidsskrift",
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
            lambda _, article: _generate_row_result(
                article=article,
                writer=writer,
                personas=personas,
                temperature=temperature,
                source=source,
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
        "tidsskrift",
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
    personas: list[str],
    temperature: float,
    source: dict[str, Any],
) -> dict[str, Any]:
    # Sample a persona deterministically. Multiply record_index to spread passages
    # within the same article across distant parts of the persona list.
    persona_idx = (article["record_index"] * 7 + article["passage_idx"]) % len(personas)
    persona = personas[persona_idx]

    length_key = (article.get("record_index", 0) + article.get("passage_idx", 0)) % len(_LENGTH_BUCKETS)
    prompt_length = _LENGTH_BUCKETS[length_key]

    messages = _instruction_messages(article, persona=persona, prompt_length=prompt_length)
    prompt = await _generate_prompt(writer, messages, temperature=temperature)
    cleaned_prompt = _clean_generated_prompt(prompt)
    return {
        "id": f"tidsskrift-{article['record_index']:06d}-p{article['passage_idx']:02d}",
        "prompt": cleaned_prompt,
        "target": article["passage"],
        "sources": [
            _drop_none(
                {
                    "dataset": source.get("dataset"),
                    "path": source.get("path"),
                    "split": source.get("split", "train"),
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
                "source_dataset": source.get("dataset"),
                "source_path": source.get("path"),
                "source_split": source.get("split", "train"),
                "source_row_index": article["record_index"],
                "passage_idx": article["passage_idx"],
                "journal": article["journal"],
                "target_chars": len(article["passage"]),
            }
        ),
    }


async def _generate_row_result(
    *,
    article: dict[str, Any],
    writer: LLM,
    personas: list[str],
    temperature: float,
    source: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    try:
        row = await _generate_row(
            article=article,
            writer=writer,
            personas=personas,
            temperature=temperature,
            source=source,
        )
    except Exception as error:
        log_event(
            "tidsskrift",
            "row_failed",
            row_index=article["row_index"],
            source_id=article["source_id"],
            journal=article.get("journal"),
            error_type=error.__class__.__name__,
            error_message=str(error),
        )
        return {
            "failure": {
                "id": f"tidsskrift-failure-{article['record_index']:06d}-p{article['passage_idx']:02d}",
                "source_id": article["source_id"],
                "journal": article.get("journal"),
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
                return await achat(messages, temperature=temperature)  # type: ignore[misc]
            return await asyncio.to_thread(writer.chat, messages, temperature=temperature)
        except Exception as error:
            last_error = error
            if attempt == 2:
                break
            await asyncio.sleep(float(attempt + 1))

    assert last_error is not None
    raise last_error


def _instruction_messages(
    article: dict[str, Any],
    *,
    persona: str | None = None,
    prompt_length: str = "medium",
) -> list[dict[str, str]]:
    journal = article.get("journal") or ""
    journal_meta = _JOURNALS.get(journal, {})

    # Prefer description from the dataset record; fall back to journals.json.
    journal_description = (
        article.get("journal_description")
        or journal_meta.get("description")
        or ""
    )

    user_lines: list[str] = []
    if article.get("title"):
        user_lines.append(f"Titel: {article['title']}")
    user_lines.append(article["passage"])

    length_rule = _LENGTH_RULES.get(prompt_length, _LENGTH_RULES["medium"])

    # System message sets context, output rules, and length constraint.
    system_lines = [
        "Du er en assistent der laver træningsdata til sprogmodeller.",
        "Du svarer KUN med selve prompten — ingen forklaringer, overskrifter eller meta-kommentarer.",
        length_rule,
        "Prompten skal være selvstændig og må IKKE referere til specifikke artikler, dokumenter "
        "eller tekster der ikke er vedlagt — ingen 'disse artikler', 'ovenstående tekst', "
        "'nedenstående dokument', 'de tre artikler' eller lignende. "
        "Prompten skal beskrive hvad brugeren ønsker skrevet, ikke hvad brugeren ønsker præsenteret.",
        "Prompten må IKKE bede om referencer til bestemte navngivne akademiske artikler, "
        "studier eller forskere der ikke naturligt indgår i emnet.",
    ]
    if journal_description:
        system_lines.append(f"\nTidsskriftets beskrivelse: {journal_description}")

    system_content = "\n".join(system_lines)

    style_block = (
        f"Persona (in English):\n{persona}\n\n"
        "Write the prompt as this type of person would — in Danish, adapted to a Danish context."
        if persona
        else ""
    )

    user_content = "\n".join(user_lines) + "\n\n---\nSkriv en prompt der ville føre en AI til at skrive ovenstående tekst."
    if style_block:
        user_content += f"\n\n{style_block}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _clean_generated_prompt(value: str) -> str:
    text = value.strip()
    # Strip fenced code blocks
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    # Strip leading markdown bold labels like "**Prompt:**" that some models emit
    text = re.sub(r"^\*\*[^*]{1,30}\*\*\s*:?\s*\n+", "", text).strip()
    # Strip surrounding quotes
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()
    # Reject meta-prompts: the model wrote a prompt *about* generating prompts instead
    # of a user-facing prompt. These slip through when the passage is an editorial intro.
    if re.search(
        r"(formuler en prompt|skriv en prompt der|få en (AI|sprogmodel) til at (skrive|generere))",
        text,
        re.IGNORECASE,
    ):
        raise ValueError(f"meta-prompt detected: {text[:120]!r}")
    # Reject self-referential prompts that reference non-existent attached documents.
    # "disse artikler / disse to bidrag / disse tekster" etc. indicate the model invented
    # a framing where multiple articles are being presented — they don't exist in the data.
    if re.search(
        r"\bdisse\s+(?:(?:to|tre|fire|fem|\d+)\s+)?(?:artik(?:el|ler)|bidrag|tekster?|essays?|afsnit)\b",
        text,
        re.IGNORECASE,
    ):
        raise ValueError(f"self-referential prompt detected: {text[:120]!r}")
    return text


def _extract_passages(
    text: str,
    *,
    min_chars: int = 200,
    max_chars: int = 2500,
) -> list[str]:
    """Extract all natural prose passage blocks from an article.

    Walks through paragraphs in document order, accumulating consecutive
    well-scoring paragraphs into blocks. A block is sealed when a bad paragraph
    breaks the run, or when the block reaches max_chars. Each sealed block that
    meets min_chars is returned as a passage.

    Returns a variable number of passages (0 for garbage documents, many for
    long articles). All text is verbatim — no synthesis.
    """
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
        score = _score_paragraph(p)

        if score == 0.0:
            _seal()
            current_len = 0
            continue

        added = len(p) + (2 if current_parts else 0)
        if current_len + added > max_chars:
            _seal()
            current_len = 0
            # Start a fresh block with this paragraph (truncate if it alone exceeds max)
            p = p[:max_chars]

        current_parts.append(p)
        current_len += len(p) + (2 if len(current_parts) > 1 else 0)

    _seal()
    return passages


def _score_paragraph(p: str) -> float:
    """Score a paragraph for suitability as a training target. Returns 0 to reject."""
    # Too short to be meaningful prose
    if len(p) < 80:
        return 0.0

    # Markdown headers — skip
    if p.startswith("#"):
        return 0.0

    # Mid-sentence fragments — paragraphs starting with a lowercase letter indicate
    # extraction hit max_chars in the middle of a sentence in the original document.
    stripped = p.lstrip()
    if stripped and stripped[0].islower():
        return 0.0

    # Image tags / captions left over from HTML or markdown (e.g. "<figure>", "< image >")
    if p.lstrip().startswith("<"):
        return 0.0

    # Photo credits, figure captions ("Foto: ...", "Figur 1.", "Kilde: ...")
    if re.match(r"^\s*(foto|figur\s*\d|kilde)\s*:", p, re.IGNORECASE):
        return 0.0
    # Inline photo credits inside an otherwise short figure-metadata paragraph.
    # e.g. "Modellen på ARKEN: ..., ARKEN Museum for Moderne Kunst. Foto: Sophie Amalie..."
    if re.search(r"\bFoto:\s+[A-ZÆØÅ]\w", p):
        return 0.0

    # Metadata label lines ("DANSK RESUMÉ: ...", "ENGLISH ABSTRACT:", "NOTER:", etc.)
    # Match both standalone headers (whole paragraph is the label) and label-prefixed content.
    if re.match(
        r"^\s*(dansk\s+resum[eé]|english\s+abstract|noter|note\s+\d|appendix)\s*[:\.]",
        p,
        re.IGNORECASE,
    ):
        return 0.0

    # Editorial meta-passages that reference other articles in the same journal issue.
    # These only make sense in that specific publishing context and are poor training targets.
    if re.search(
        r"\b(dette\s+temanummer|det\s+følgende\s+nummer|fra\s+redaktionens\s+side|"
        r"temanummerets\s+(?:\w+\s+)?artik(?:el|ler)|"
        r"(?:første|anden|tredje)\s+artikel\s+i\s+(?:dette|temanummeret)|"
        r"artiklerne\s+i\s+dette\s+nummer|"
        r"dette\s+nummer\s+af\s+[A-ZÆØÅ]|"
        r"temaredaktionen\b)",
        p,
        re.IGNORECASE,
    ):
        return 0.0
    # "Temanummeret om..." at start of paragraph — editorial overview opening
    if re.match(r"^\s*[Tt]emanummeret\b", p):
        return 0.0

    # Cross-reference to a named article within a journal issue.
    # Catches patterns like:
    #   "[Author]s artikel '[Title]'"        — possessive author reference
    #   "i artiklen '[Title]'"               — in the article (singular)
    #   "i deres artikler '[A]' og '[B]'"    — in their articles (plural, editorial)
    #   "emnet for [Author]s artikel"        — editorial topic framing
    #   "Artiklen '[Title]' af [Author]"     — article named by title
    #   "[Author] har skrevet artiklen '[Title]'"  — attribution sentence
    if re.search(
        r"\b\w[\w\-']*s\s+artikel\s+['\u2018\u2019\u201c\u201d\"]"
        r"|"
        r"\bi\s+(?:sin|deres)\s+artik(?:el|ler)\s+['\u2018\u2019\u201c\u201d\"]"
        r"|"
        r"\bi\s+artiklen,?\s+['\u2018\u2019\u201c\u201d\"]"
        r"|"
        # "i artiklen [CapTitle]" — article name (unquoted) follows immediately after "artiklen"
        r"\bi\s+artiklen\s+[A-ZÆØÅ]"
        r"|"
        r"\bemnet\s+for\s+\w[\w\-']*s\s+artikel\b"
        r"|"
        r"^\s*[Aa]rtiklen\s+['\u2018\u2019\u201c\u201d\"]"
        r"|"
        r"\bhar\s+skrevet\s+artiklen\s+['\u2018\u2019\u201c\u201d\"]"
        r"|"
        # Editorial summaries: "[Article Title] tilbyder/udforsker [Author Name, Author Name]"
        # The pattern: verb immediately followed by a capitalised name then comma (multiple authors)
        r"\b(?:tilbyder|udforsker|analyserer)\s+[A-ZÆØÅ]\w+(?:\s+[A-ZÆØÅ]\w+)?,\s+[A-ZÆØÅ]"
        r"|"
        # "[Author] præsenterer i [Article Title]" — author summarising their own co-published piece
        r"\b[A-ZÆØÅ]\w+\s+præsenterer\s+i\s+[A-ZÆØÅ]",
        p,
        re.IGNORECASE,
    ):
        return 0.0

    # Low alphabetic ratio — likely a table, list of references, or garbage
    alpha_ratio = sum(c.isalpha() for c in p) / len(p)
    if alpha_ratio < 0.55:
        return 0.0

    # Markdown tables: majority of lines start with "|"
    lines = [ln for ln in p.splitlines() if ln.strip()]
    if lines and sum(ln.lstrip().startswith("|") for ln in lines) / len(lines) > 0.5:
        return 0.0

    # Metadata patterns: author lines, emails, DOIs, URLs, affiliations, abstract/keyword headers
    if re.search(
        r"@|\bdoi:\b|https?://|lektor\b|professor\b|ph\.d\b|cand\.|mail:|aarhus\s+uni|syddansk",
        p,
        re.IGNORECASE,
    ):
        return 0.0
    # Publication metadata blocks (journal submission/acceptance lines, publisher credits)
    if re.search(
        r"\baccept[e-]ret\s+til\s+publikation\b"
        r"|\bmodtaget\s+til\s+(?:review|vurdering)\b"
        r"|PUBLICERET\s+AF\s+DET\s+KGL\b"
        r"|\bpubliceret\s+af\s+det\s+(?:kgl|kongelige)\b",
        p,
        re.IGNORECASE,
    ):
        return 0.0
    if re.match(r"^\s*(keywords?|abstract|english\s+abstract|resumé)\s*:", p, re.IGNORECASE):
        return 0.0

    # Very short lines dominate — probably a reference list or structured metadata
    if lines and sum(len(ln) < 60 for ln in lines) / len(lines) > 0.7:
        return 0.0

    # Single-entry bibliography references can be one long paragraph.
    if _looks_like_bibliographic_entry(p):
        return 0.0

    ref_lines = sum(1 for ln in lines if _is_reference_line(ln))
    if lines and ref_lines / len(lines) > 0.3:
        return 0.0

    # Skip predominantly English passages — we target Danish text.
    words_lower = re.findall(r"\b[a-z]{2,}\b", p[:600].lower())
    if words_lower:
        da_hits = sum(1 for w in words_lower if w in _DA_STOP)
        en_hits = sum(1 for w in words_lower if w in _EN_STOP)
        if en_hits > da_hits and en_hits >= 3:
            return 0.0

    # Penalise heavy markdown formatting (lots of ## or ** or ``` etc.)
    markup_chars = len(re.findall(r"[#*`\[\]]", p))
    markup_ratio = markup_chars / len(p)
    if markup_ratio > 0.05:
        return 0.0

    # Reward length up to ~1000 chars, then plateau
    length_score = min(len(p) / 1000, 1.0)
    return alpha_ratio * length_score


def _looks_garbled(text: str) -> bool:
    """Return True if the text appears to be corrupted OCR / bad PDF extraction.

    Heuristics:
    - High ratio of single-character tokens separated by spaces (spaced-out text like
      "K æ r l i g h e d" from PDF layout boxes).
    - Presence of random-looking byte-noise strings common in bad PDF headers.
    """
    # Spaced-character ratio: runs of "X " where X is a single letter/digit
    spaced = re.findall(r"(?<!\w)\S (?!\w)", text)
    words = text.split()
    if words and len(spaced) / max(len(words), 1) > 0.4:
        return True

    # Noise strings: sequences of 3+ non-ASCII or mixed-case garbage at start
    noise_pattern = re.compile(r"^[^\s]{3,}\s[^\s]{3,}\s[^\s]{3,}")
    first_line = text.splitlines()[0] if text else ""
    if noise_pattern.match(first_line) and not first_line[0].isalpha():
        return True

    return False


def _max_articles(generation: dict[str, Any]) -> int | None:
    value = generation.get("max_articles")
    if value is None:
        return None
    max_articles = int(value)
    assert max_articles >= 0, "generation.max_articles must be non-negative"
    return max_articles


def _max_articles_per_journal(generation: dict[str, Any]) -> int | None:
    value = generation.get("max_articles_per_journal")
    if value is None:
        return None
    n = int(value)
    assert n >= 0, "generation.max_articles_per_journal must be non-negative"
    return n


def _progress_log(message: str) -> None:
    print(f"[tidsskrift] {message}", flush=True)


def _progress_reporter(
    *,
    total: int,
    starting_completed: int,
    worker_concurrency: int,
):
    state = {"finished": False, "elapsed_seconds": 0}
    common_progress.write_progress_snapshot(
        "tidsskrift_progress",
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
        "tidsskrift_progress",
        stage="generating_prompts",
        completed_offset=starting_completed,
        total=total,
        extra=lambda completed, _, elapsed: {
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
        "tidsskrift_progress",
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
        f"\r[tidsskrift] [{bar}] {completed}/{total_text} "
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


def _load_rows(result: BuildResult) -> list[dict[str, Any]]:
    dataset_path = Path(result.artifacts["dataset"].path)
    return store.read_jsonl(dataset_path)


def _load_verified_rows(result: BuildResult) -> list[dict[str, Any]]:
    verified_path = Path(result.run_dir) / "outputs" / "verified.jsonl"
    if verified_path.exists():
        return store.read_jsonl(verified_path)

    rows = _load_rows(result)
    cfg = _load_cfg(result)
    min_article_chars = int(cfg["generation"].get("min_article_chars", 0))
    verified_rows = common_eval.verify(rows, _has_prompt, name="prompt_present")
    verified_rows = common_eval.verify(verified_rows, _has_target, name="target_present")
    verified_rows = common_eval.verify(
        verified_rows,
        lambda row: _target_meets_min_chars(row, min_article_chars),
        name="target_min_chars",
    )
    return common_eval.verify(
        verified_rows,
        _target_is_not_references,
        name="target_not_references",
    )


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


_REF_LINE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*[-–]\s+\w.{0,40}\(\d{4}\)"),  # noqa: RUF001
    re.compile(r"^\s*[-–]\s+\d{1,3}\s+\w"),  # noqa: RUF001
    re.compile(r"^\s*[-–]\s+[A-ZÆØÅ]\w.{0,80}\d{4}"),  # noqa: RUF001
    re.compile(r"^\s*\[\d+\]"),
    re.compile(r"^\s*\w[\w\s,\-\.]{5,40}\.\s+\d{4}"),
    re.compile(r"^\s*\d{1,3}\s+[A-ZÆØÅ]\w"),
    re.compile(r"^\s*\d{1,3}\.\s+[A-ZÆØÅ\w]"),
    re.compile(r"^\s*[A-ZÆØÅ]\w+,\s+\w[\w\.\-]{0,10}\s*\(\d"),
    re.compile(r"^\s*[A-ZÆØÅ]\w+,\s+\w[\w\.\s,&\-]*\(\d{4}\)"),  # multi-author APA
]


def _is_reference_line(line: str) -> bool:
    return any(p.match(line) for p in _REF_LINE_PATTERNS)


def _looks_like_bibliographic_entry(text: str) -> bool:
    compact = " ".join(text.split())
    if len(compact) < 70:
        return False

    authorish_start = bool(re.match(r"^[A-ZÆØÅ][\w'\-]+,\s+[A-ZÆØÅ]", compact))
    year_colon_marker = bool(re.search(r"\b\d{4}[a-z]?\s*:", compact, re.IGNORECASE))
    year_paren_marker = bool(re.search(r"\(\s*\d{4}[a-z]?\s*\)", compact, re.IGNORECASE))
    in_press_marker = bool(re.search(r"\(\s*in\s+press\s*\)", compact, re.IGNORECASE))
    in_marker = bool(re.search(r"\bIn:\b", compact, re.IGNORECASE))
    i_marker = bool(re.search(r"\bI:\b", compact, re.IGNORECASE))
    pages_marker = bool(re.search(r"\(pp?\.\s*\d+\s*[-–]\s*\d+\)", compact, re.IGNORECASE))  # noqa: RUF001
    danish_pages_marker = bool(re.search(r"\(s\.\s*\d+\s*[-–]\s*\d+\)", compact, re.IGNORECASE))  # noqa: RUF001
    editor_marker = bool(re.search(r"\((?:red\.|eds?\.)\)", compact, re.IGNORECASE))
    issue_pages_marker = bool(re.search(r"\bnr\.\s*\d+\b.*\b\d+\s*[-–]\s*\d+\b", compact, re.IGNORECASE))  # noqa: RUF001
    place_publisher_marker = bool(re.search(r"\b[A-ZÆØÅ][a-zæøå]+:\s+[A-ZÆØÅ][\w&\-\s]+", compact))
    institutional_publisher_tail = bool(
        re.search(
            r"\b(?:center|forlag|universitetsforlag|samfundslitteratur)\b[\w\s\-]*\.?$",
            compact,
            re.IGNORECASE,
        )
    )

    temporal_marker = year_colon_marker or year_paren_marker or in_press_marker
    source_marker = in_marker or i_marker

    if authorish_start and in_press_marker:
        return True
    if authorish_start and temporal_marker and (source_marker or pages_marker or danish_pages_marker or editor_marker):
        return True
    if source_marker and temporal_marker and (pages_marker or danish_pages_marker or editor_marker or place_publisher_marker):
        return True
    if authorish_start and temporal_marker and issue_pages_marker:
        return True
    if authorish_start and temporal_marker and institutional_publisher_tail:
        return True

    return False


def _has_prompt(row: dict[str, Any]) -> bool:
    return bool(str(row.get("prompt", "")).strip())


def _has_target(row: dict[str, Any]) -> bool:
    return bool(str(row.get("target", "")).strip())


def _target_meets_min_chars(row: dict[str, Any], min_article_chars: int) -> bool:
    return len(str(row.get("target", "")).strip()) >= min_article_chars


def _target_is_not_references(row: dict[str, Any]) -> bool:
    target = str(row.get("target", ""))
    if _looks_like_bibliographic_entry(target):
        return False

    lines = [ln for ln in target.splitlines() if ln.strip()]
    if not lines:
        return True
    ref_count = sum(1 for ln in lines if _is_reference_line(ln))
    return ref_count / len(lines) <= 0.3


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


def _split_rows(
    rows: list[dict[str, Any]], train_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
