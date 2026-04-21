from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import yaml

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import eval as common_eval
from sdg.commons import publish as common_publish
from sdg.commons.run import load, run
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json

SUPPORTED_LANGUAGES = {"da", "en"}
SUPPORTED_DIRECTIONS = {"da-en", "en-da"}
SUPPORTED_INPUT_MODES = {"plain", "marked_text", "json_field"}
SUPPORTED_OUTPUT_MODES = {
    "plain",
    "json_object",
    "json_object_meta",
    "json_object_source",
    "xml_tag",
    "xml_tag_meta",
    "yaml_field",
    "yaml_field_meta",
    "yaml_field_source",
    "json_string",
}
STYLE_SPECS = {
    "translate_into": {"input_mode": "plain", "output_mode": "plain"},
    "from_to": {"input_mode": "plain", "output_mode": "plain"},
    "translator_role": {"input_mode": "plain", "output_mode": "plain"},
    "strict_instructions": {"input_mode": "plain", "output_mode": "plain"},
    "minimal_output": {"input_mode": "plain", "output_mode": "plain"},
    "marked_text_only": {"input_mode": "marked_text", "output_mode": "plain"},
    "json_field_only": {"input_mode": "json_field", "output_mode": "plain"},
    "json_object_output": {"input_mode": "plain", "output_mode": "json_object"},
    "json_object_meta_output": {"input_mode": "plain", "output_mode": "json_object_meta"},
    "json_object_source_output": {"input_mode": "plain", "output_mode": "json_object_source"},
    "xml_output": {"input_mode": "plain", "output_mode": "xml_tag"},
    "xml_meta_output": {"input_mode": "plain", "output_mode": "xml_tag_meta"},
    "yaml_output": {"input_mode": "plain", "output_mode": "yaml_field"},
    "yaml_meta_output": {"input_mode": "plain", "output_mode": "yaml_field_meta"},
    "yaml_source_output": {"input_mode": "plain", "output_mode": "yaml_field_source"},
    "json_string_output": {"input_mode": "plain", "output_mode": "json_string"},
}
SUPPORTED_TEMPLATE_STYLES = set(STYLE_SPECS)

TEXT = {
    "da": {
        "names": {"da": "dansk", "en": "engelsk"},
        "field_name": "tekst",
        "text_label": "Tekst",
        "language_suffix": "tekst",
        "reply_only": "Svar kun med oversættelsen.",
        "output_only": "Svar kun med den oversatte tekst.",
        "translate_into": "Oversæt denne tekst til {target_language}.",
        "from_to": "Fra {source_language} til {target_language}. Returner kun oversættelsen.",
        "translator_role": "Du oversætter tekst på {source_language} til {target_language}.",
        "translator_role_rule": (
            "Returner kun oversættelsen. Medtag ikke en introduktion, forklaring, noter, "
            "alternativer, markdown-formatering eller citationstegn omkring svaret, "
            "medmindre citationstegnene hører til selve oversættelsen."
        ),
        "strict_intro": "Oversæt følgende tekst fra {source_language} til {target_language}.",
        "strict_rule": (
            "Returner kun den oversatte tekst. Ingen forklaring, noter, alternativer, "
            "markdown eller citationstegn omkring svaret, medmindre oversættelsen kræver det."
        ),
        "marked_text_only": (
            "Oversæt kun teksten inde i <translate_me>-taggene fra {source_language} til {target_language}. "
            "Ignorer al tekst uden for taggene. Returner kun oversættelsen."
        ),
        "json_field_only": (
            'Oversæt kun værdien i feltet "{field_name}" fra {source_language} til {target_language}. '
            "Ignorer de andre felter. Returner kun den oversatte værdi."
        ),
        "marked_notes": (
            "Dokumentoversigt: kategori=oversættelse | prioritet=normal | reference=DA-2048",
            "Intern journalnote: behold kun indholdet mellem taggene i svaret.",
        ),
        "json_payload": {
            "document_id": "da-2048",
            "domain": "forvaltning",
            "review_status": "klar til behandling",
            "comment": "bevar betydningen præcist",
        },
        "output_rules": {
            "json_object_output": 'Returner et JSON-objekt med præcis én nøgle, "translation", og ingen anden tekst.',
            "json_object_meta_output": (
                'Returner et JSON-objekt med præcis nøglerne "source_language", '
                '"target_language" og "translation", og ingen anden tekst. '
                "Brug ISO-sprogkoder i language-felterne."
            ),
            "json_object_source_output": (
                'Returner et JSON-objekt med præcis nøglerne "source_language", '
                '"target_language", "source_text" og "translation", og ingen anden tekst. '
                "Brug ISO-sprogkoder i language-felterne."
            ),
            "xml_output": "Returner XML i formen <translation>...</translation> og intet andet.",
            "xml_meta_output": (
                "Returner XML i formen "
                '<translation source_language="..." target_language="...">...</translation> '
                "og intet andet. Brug ISO-sprogkoder i attributterne."
            ),
            "yaml_output": "Returner YAML med præcis ét felt, translation: ..., og ingen anden tekst.",
            "yaml_meta_output": (
                "Returner YAML med præcis felterne source_language, target_language og "
                "translation, og ingen anden tekst. Brug ISO-sprogkoder i language-felterne."
            ),
            "yaml_source_output": (
                "Returner YAML med præcis felterne source_language, target_language, "
                "source_text og translation, og ingen anden tekst. Brug ISO-sprogkoder i "
                "language-felterne."
            ),
            "json_string_output": "Returner oversættelsen som en JSON-strengliteral og intet andet.",
        },
    },
    "en": {
        "names": {"da": "Danish", "en": "English"},
        "field_name": "text",
        "text_label": "Text",
        "language_suffix": "text",
        "reply_only": "Reply with the translation and nothing else.",
        "output_only": "Output only the translated text.",
        "translate_into": "Translate this text into {target_language}.",
        "from_to": "From {source_language} to {target_language}. Return only the translation.",
        "translator_role": "You are translating {source_language} text into {target_language}.",
        "translator_role_rule": (
            "Return only the translation. Do not include an introduction, explanation, notes, "
            "alternatives, markdown formatting, or a quoted wrapper unless quotes are part of "
            "the translation."
        ),
        "strict_intro": "Translate the following {source_language} text into {target_language}.",
        "strict_rule": (
            "Only output the translated text. No explanation, notes, alternatives, markdown, "
            "or surrounding quotes unless the translation requires them."
        ),
        "marked_text_only": (
            "Translate only the text inside the <translate_me> tags from {source_language} to {target_language}. "
            "Ignore any text outside the tags. Return only the translation."
        ),
        "json_field_only": (
            'Translate only the value of the "{field_name}" field from {source_language} to {target_language}. '
            "Ignore the other fields. Return only the translated value."
        ),
        "marked_notes": (
            "Document summary: category=translation | priority=normal | reference=EN-2048",
            "Internal processing note: the answer should only use the tagged segment.",
        ),
        "json_payload": {
            "document_id": "en-2048",
            "domain": "public-administration",
            "review_status": "ready",
            "comment": "preserve meaning exactly",
        },
        "output_rules": {
            "json_object_output": 'Return a JSON object with exactly one key, "translation", and no other text.',
            "json_object_meta_output": (
                'Return a JSON object with exactly the keys "source_language", '
                '"target_language", and "translation", and no other text. '
                "Use ISO language codes in the language fields."
            ),
            "json_object_source_output": (
                'Return a JSON object with exactly the keys "source_language", '
                '"target_language", "source_text", and "translation", and no other text. '
                "Use ISO language codes in the language fields."
            ),
            "xml_output": "Return XML in the form <translation>...</translation> and nothing else.",
            "xml_meta_output": (
                "Return XML in the form "
                '<translation source_language="..." target_language="...">...</translation> '
                "and nothing else. Use ISO language codes in the attributes."
            ),
            "yaml_output": "Return YAML with exactly one field, translation: ..., and nothing else.",
            "yaml_meta_output": (
                "Return YAML with exactly the fields source_language, target_language, and "
                "translation, and nothing else. Use ISO language codes in the language fields."
            ),
            "yaml_source_output": (
                "Return YAML with exactly the fields source_language, target_language, "
                "source_text, and translation, and nothing else. Use ISO language codes in the "
                "language fields."
            ),
            "json_string_output": "Return the translation as a JSON string literal and nothing else.",
        },
    },
}


def build(cfg: dict[str, Any]) -> BuildResult:
    return run(
        _build_run,
        pack="translation",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    verified_rows = common_eval.verify(rows, _has_prompt, name="prompt_present")
    verified_rows = common_eval.verify(verified_rows, _has_target, name="target_present")
    verified_rows = common_eval.verify(verified_rows, _has_sources, name="source_present")
    verified_rows = common_eval.verify(
        verified_rows,
        _has_translation_metadata,
        name="translation_metadata_present",
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
    source_entries = _source_entries(cfg)
    dataset_artifact = result.artifacts.get("dataset")
    artifact_meta = dataset_artifact.meta if dataset_artifact else {}
    source_pairs_by_config = artifact_meta.get("source_pairs_by_config", {})
    template_pool_counts_by_config = artifact_meta.get("template_pool_counts_by_config", {})
    variant_count_by_config = artifact_meta.get("variant_count_by_config", {})
    long_text_variant_count_by_config = artifact_meta.get("long_text_variant_count_by_config", {})

    sources = [
        {
            "key": entry["key"],
            "dataset": entry["source"].get("dataset"),
            "path": entry["source"].get("path"),
            "split": entry["source"].get("split", "train"),
            "rows": source_pairs_by_config.get(entry["key"]),
            "generation": _generation_summary(
                entry["generation"],
                variant_count=variant_count_by_config.get(entry["key"]),
                long_text_variant_count=long_text_variant_count_by_config.get(entry["key"]),
                template_pool_counts=template_pool_counts_by_config.get(entry["key"]),
            ),
        }
        for entry in source_entries
    ]

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "source_pairs": artifact_meta.get("source_pairs"),
        "source": _source_summary(source_entries[0]["source"]) if len(source_entries) == 1 else None,
        "generation": (
            _generation_summary(
                source_entries[0]["generation"],
                variant_count=artifact_meta.get("variant_count"),
                long_text_variant_count=artifact_meta.get("long_text_variant_count"),
                template_pool_counts=artifact_meta.get("template_pool_counts"),
            )
            if len(source_entries) == 1
            else None
        ),
        "sources": sources,
        "artifacts": sorted(result.artifacts),
        "metrics": metrics,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    failures = [row for row in rows if not _row_passes(row)]
    train_rows, eval_rows = _split_rows(rows, cfg)

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

    dataset_path = outputs_dir / "dataset.jsonl"
    rows_written = 0
    source_pairs = 0
    source_entries = _source_entries(cfg)
    template_pool_counts: dict[str, int] = {}
    source_pairs_by_config: dict[str, int] = {}
    template_pool_counts_by_config: dict[str, dict[str, int]] = {}
    variant_count_by_config: dict[str, int] = {}
    long_text_variant_count_by_config: dict[str, int | None] = {}

    with dataset_path.open("w") as handle:
        for entry in source_entries:
            generation = entry["generation"]
            variant_specs = _variant_specs(generation)
            default_template_styles = tuple(_selected_template_styles(generation))
            variant_specs_by_pool: dict[tuple[str, ...], list[dict[str, str]]] = {
                default_template_styles: variant_specs,
            }
            emitted_rows_by_pool: dict[tuple[str, ...], int] = {}

            variant_count_by_config[entry["key"]] = len(variant_specs)
            long_text_variant_count_by_config[entry["key"]] = _long_text_variant_count(generation)

            for pair in _iter_translation_pairs(entry):
                template_styles, template_style_pool = _template_styles_for_pair(generation, pair)
                pool_key = tuple(template_styles)
                pool_variant_specs = variant_specs_by_pool.get(pool_key)
                if pool_variant_specs is None:
                    pool_variant_specs = _variant_specs(generation, template_styles=template_styles)
                    variant_specs_by_pool[pool_key] = pool_variant_specs

                pool_row_index = emitted_rows_by_pool.get(pool_key, 0)
                row = _row_for_pair(
                    pair,
                    entry,
                    variant=pool_variant_specs[pool_row_index % len(pool_variant_specs)],
                    template_style_pool=template_style_pool,
                )
                store.append_jsonl_line(handle, row)
                source_pairs += 1
                rows_written += 1
                emitted_rows_by_pool[pool_key] = pool_row_index + 1
                source_pairs_by_config[entry["key"]] = source_pairs_by_config.get(entry["key"], 0) + 1
                template_pool_counts[template_style_pool] = (
                    template_pool_counts.get(template_style_pool, 0) + 1
                )
                pool_counts = template_pool_counts_by_config.setdefault(entry["key"], {})
                pool_counts[template_style_pool] = pool_counts.get(template_style_pool, 0) + 1

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
                "rows": rows_written,
                "source_pairs": source_pairs,
                "variant_count": (
                    variant_count_by_config[source_entries[0]["key"]]
                    if len(source_entries) == 1
                    else None
                ),
                "long_text_variant_count": (
                    long_text_variant_count_by_config[source_entries[0]["key"]]
                    if len(source_entries) == 1
                    else None
                ),
                "template_pool_counts": template_pool_counts,
                "source_pairs_by_config": source_pairs_by_config,
                "variant_count_by_config": variant_count_by_config,
                "long_text_variant_count_by_config": long_text_variant_count_by_config,
                "template_pool_counts_by_config": template_pool_counts_by_config,
                "source": (
                    _source_label(source_entries[0]["source"])
                    if len(source_entries) == 1
                    else None
                ),
            },
        )
    }


def _iter_translation_pairs(entry: dict[str, Any]):
    source = entry["source"]
    generation = entry["generation"]
    max_pairs = _max_pairs(generation)
    max_pairs_per_source = _max_pairs_per_source(generation)
    max_pair_chars = _max_pair_chars(generation)
    kept_pairs = 0
    kept_pairs_by_source: dict[str, int] = {}

    for index, record in enumerate(_iter_source_records(source)):
        pair = _record_to_pair(record, source, index)
        if pair is None:
            continue
        if max_pair_chars is not None and pair["pair_max_chars"] > max_pair_chars:
            continue

        source_label = pair["origin"]
        if max_pairs_per_source is not None and source_label:
            current_source_pairs = kept_pairs_by_source.get(source_label, 0)
            if current_source_pairs >= max_pairs_per_source:
                continue

        yield pair
        kept_pairs += 1
        if max_pairs_per_source is not None and source_label:
            kept_pairs_by_source[source_label] = current_source_pairs + 1
        if max_pairs is not None and kept_pairs >= max_pairs:
            break


def _iter_source_records(source: dict[str, Any]):
    path = source.get("path")
    if path:
        return store.iter_jsonl(Path(path).expanduser().resolve())

    from datasets import load_dataset

    return load_dataset(
        path=source["dataset"],
        name=source.get("config_name"),
        split=source.get("split", "train"),
        streaming=bool(source.get("streaming", True)),
    )


def _record_to_pair(
    record: dict[str, Any],
    source: dict[str, Any],
    index: int,
) -> dict[str, Any] | None:
    danish = _read_record_value(record, source.get("danish_field", "danish"))
    english = _read_record_value(record, source.get("english_field", "english"))
    if not danish or not english:
        return None

    source_id = _read_record_value(record, source.get("id_field")) or str(index)
    origin = _read_record_value(record, source.get("origin_field", "source"))
    danish_chars = len(danish)
    english_chars = len(english)

    return {
        "row_index": index,
        "source_id": source_id,
        "danish": danish,
        "english": english,
        "origin": origin,
        "danish_chars": danish_chars,
        "english_chars": english_chars,
        "pair_max_chars": max(danish_chars, english_chars),
    }


def _read_record_value(record: dict[str, Any], field_name: str | None) -> str | None:
    if not field_name:
        return None

    value: Any = record
    for part in field_name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
        if value is None:
            return None

    text = str(value).strip()
    if not text:
        return None
    return text


def _row_for_pair(
    pair: dict[str, Any],
    entry: dict[str, Any],
    *,
    variant: dict[str, str],
    template_style_pool: str,
):
    source = entry["source"]
    source_key = entry["key"]
    source_language = variant["source_language"]
    target_language = variant["target_language"]
    prompt_language = variant["prompt_language"]
    template_style = variant["template_style"]
    source_text = _pair_text(pair, source_language)
    target_text = _pair_text(pair, target_language)
    input_mode = _input_mode_for_template_style(template_style)
    output_mode = _output_mode_for_template_style(template_style)
    rendered_target = _render_target(
        target_text,
        output_mode=output_mode,
        source_language=source_language,
        target_language=target_language,
        source_text=source_text,
    )

    return {
        "id": (
            f"translation-{source_key}-{pair['row_index']:08d}-"
            f"{source_language}-{target_language}-"
            f"{prompt_language}-{template_style}"
        ),
        "prompt": _render_prompt(
            prompt_language=prompt_language,
            template_style=template_style,
            source_language=source_language,
            target_language=target_language,
            source_text=source_text,
        ),
        "target": rendered_target,
        "sources": [
            _drop_none(
                {
                    "dataset": source.get("dataset"),
                    "path": source.get("path"),
                    "split": source.get("split", "train"),
                    "row_id": pair["source_id"],
                    "source_config": source_key,
                    "source": pair["origin"],
                    "source_language": source_language,
                    "target_language": target_language,
                }
            )
        ],
        "meta": _drop_none(
            {
                "source_id": pair["source_id"],
                "source_row_index": pair["row_index"],
                "source_config": source_key,
                "source_dataset": source.get("dataset"),
                "source_path": source.get("path"),
                "source_split": source.get("split", "train"),
                "source_origin": pair["origin"],
                "source_language": source_language,
                "target_language": target_language,
                "prompt_language": prompt_language,
                "template_style": template_style,
                "template_style_pool": template_style_pool,
                "input_mode": input_mode,
                "output_mode": output_mode,
                "pair_max_chars": pair["pair_max_chars"],
                "source_chars": len(source_text),
                "target_chars": len(rendered_target),
            }
        ),
    }


def _pair_text(pair: dict[str, Any], language: str) -> str:
    if language == "da":
        return str(pair["danish"])
    if language == "en":
        return str(pair["english"])
    raise AssertionError(f"Unsupported language: {language}")


def _render_prompt(
    *,
    prompt_language: str,
    template_style: str,
    source_language: str,
    target_language: str,
    source_text: str,
) -> str:
    text = TEXT[prompt_language]
    source_name = text["names"][source_language]
    target_name = text["names"][target_language]
    source_label = f"{_language_label(source_name)} {text['language_suffix']}"
    params = {
        "source_language": source_name,
        "target_language": target_name,
        "field_name": text["field_name"],
    }
    if template_style == "translate_into":
        return _join_sections(
            text["translate_into"].format(**params),
            text["reply_only"],
            _text_block(text["text_label"], source_text),
        )
    if template_style == "from_to":
        return _join_sections(text["from_to"].format(**params), _text_block(text["text_label"], source_text))
    if template_style == "translator_role":
        return _join_sections(
            text["translator_role"].format(**params),
            text["translator_role_rule"],
            _text_block(source_label, source_text),
        )
    if template_style == "strict_instructions":
        return _join_sections(
            text["strict_intro"].format(**params),
            text["strict_rule"],
            _text_block(source_label, source_text),
        )
    if template_style == "minimal_output":
        return _join_sections(
            f"{_language_label(source_name)} til {_language_label(target_name)}."
            if prompt_language == "da"
            else f"{_language_label(source_name)} to {_language_label(target_name)}.",
            text["output_only"],
            source_text,
        )
    if template_style == "marked_text_only":
        return _join_sections(
            text["marked_text_only"].format(**params),
            _marked_source_block(text["marked_notes"], source_text),
        )
    if template_style == "json_field_only":
        return _join_sections(
            text["json_field_only"].format(**params),
            _text_block(
                "Payload",
                _json_payload(text["json_payload"], text["field_name"], source_text),
            ),
        )
    return _join_sections(
        text["strict_intro"].format(**params),
        text["output_rules"][template_style],
        _text_block(source_label, source_text),
    )


def _language_label(value: str) -> str:
    if not value:
        return value
    return value[0].upper() + value[1:]
def _text_block(label: str, text: str) -> str:
    return f"{label}:\n{text}"
def _join_sections(*parts: str) -> str:
    return "\n\n".join(part for part in parts if part)
def _json_payload(payload: dict[str, str], field_name: str, source_text: str) -> str:
    return json.dumps({**payload, field_name: source_text}, ensure_ascii=False, sort_keys=True)
def _marked_source_block(notes: tuple[str, str], source_text: str) -> str:
    return "\n".join([notes[0], "<translate_me>", source_text, "</translate_me>", notes[1]])


def _render_target(
    target_text: str,
    *,
    output_mode: str,
    source_language: str,
    target_language: str,
    source_text: str,
) -> str:
    if output_mode == "plain":
        return target_text
    if output_mode == "json_object":
        return json.dumps({"translation": target_text}, ensure_ascii=False, sort_keys=True)
    if output_mode == "json_object_meta":
        return json.dumps(
            {
                "source_language": source_language,
                "target_language": target_language,
                "translation": target_text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if output_mode == "json_object_source":
        return json.dumps(
            {
                "source_language": source_language,
                "target_language": target_language,
                "source_text": source_text,
                "translation": target_text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    if output_mode == "xml_tag":
        return f"<translation>{xml_escape(target_text)}</translation>"
    if output_mode == "xml_tag_meta":
        return (
            f'<translation source_language="{xml_escape(source_language)}" '
            f'target_language="{xml_escape(target_language)}">'
            f"{xml_escape(target_text)}</translation>"
        )
    if output_mode == "yaml_field":
        return yaml.safe_dump(
            {"translation": target_text},
            allow_unicode=True,
            sort_keys=True,
        ).strip()
    if output_mode == "yaml_field_meta":
        return yaml.safe_dump(
            {
                "source_language": source_language,
                "target_language": target_language,
                "translation": target_text,
            },
            allow_unicode=True,
            sort_keys=True,
        ).strip()
    if output_mode == "yaml_field_source":
        return yaml.safe_dump(
            {
                "source_language": source_language,
                "target_language": target_language,
                "source_text": source_text,
                "translation": target_text,
            },
            allow_unicode=True,
            sort_keys=True,
        ).strip()
    if output_mode == "json_string":
        return json.dumps(target_text, ensure_ascii=False)
    raise AssertionError(f"Unsupported output mode: {output_mode}")


def _input_mode_for_template_style(template_style: str) -> str:
    return str(STYLE_SPECS[template_style]["input_mode"])


def _output_mode_for_template_style(template_style: str) -> str:
    return str(STYLE_SPECS[template_style]["output_mode"])


def _selected_prompt_languages(generation: dict[str, Any]) -> list[str]:
    return _normalized_choices(
        generation.get("prompt_languages", ["da", "en"]),
        supported=SUPPORTED_LANGUAGES,
        label="generation.prompt_languages",
    )


def _selected_directions(generation: dict[str, Any]) -> list[str]:
    return _normalized_choices(
        generation.get("directions", ["da-en", "en-da"]),
        supported=SUPPORTED_DIRECTIONS,
        label="generation.directions",
    )


def _selected_template_styles(generation: dict[str, Any]) -> list[str]:
    return _normalized_choices(
        generation.get("template_styles", ["translate_into", "from_to"]),
        supported=SUPPORTED_TEMPLATE_STYLES,
        label="generation.template_styles",
    )


def _normalized_choices(
    values: str | list[str],
    *,
    supported: set[str],
    label: str,
) -> list[str]:
    raw_values = [values] if isinstance(values, str) else list(values)
    assert raw_values, f"{label} must not be empty"

    normalized: list[str] = []
    for value in raw_values:
        item = str(value)
        assert item in supported, f"{label} contains unsupported value: {item}"
        if item not in normalized:
            normalized.append(item)
    return normalized


def _template_styles_for_pair(
    generation: dict[str, Any],
    pair: dict[str, Any],
) -> tuple[list[str], str]:
    default_template_styles = _selected_template_styles(generation)
    long_text_threshold_chars = _long_text_threshold_chars(generation)
    if long_text_threshold_chars is None:
        return default_template_styles, "default"
    if pair["pair_max_chars"] <= long_text_threshold_chars:
        return default_template_styles, "default"

    long_text_template_styles = _selected_long_text_template_styles(generation)
    if not long_text_template_styles or long_text_template_styles == default_template_styles:
        return default_template_styles, "default"
    return long_text_template_styles, "long_text"


def _variant_specs(
    generation: dict[str, Any],
    *,
    template_styles: list[str] | None = None,
) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    selected_template_styles = template_styles or _selected_template_styles(generation)
    for direction in _selected_directions(generation):
        source_language, target_language = direction.split("-")
        for prompt_language in _selected_prompt_languages(generation):
            for template_style in selected_template_styles:
                variants.append(
                    {
                        "source_language": source_language,
                        "target_language": target_language,
                        "prompt_language": prompt_language,
                        "template_style": template_style,
                    }
                )

    assert variants, "generation must include at least one prompt variant"
    return variants


def _max_pairs(generation: dict[str, Any]) -> int | None:
    value = generation.get("max_pairs")
    if value is None:
        return None

    max_pairs = int(value)
    assert max_pairs >= 0, "generation.max_pairs must be non-negative"
    return max_pairs
def _max_pairs_per_source(generation: dict[str, Any]) -> int | None:
    value = generation.get("max_pairs_per_source")
    if value is None:
        return None

    max_pairs_per_source = int(value)
    assert max_pairs_per_source >= 0, "generation.max_pairs_per_source must be non-negative"
    return max_pairs_per_source
def _max_pair_chars(generation: dict[str, Any]) -> int | None:
    value = generation.get("max_pair_chars")
    if value is None:
        return None

    max_pair_chars = int(value)
    assert max_pair_chars >= 0, "generation.max_pair_chars must be non-negative"
    return max_pair_chars
def _long_text_threshold_chars(generation: dict[str, Any]) -> int | None:
    value = generation.get("long_text_threshold_chars")
    if value is None:
        return None

    long_text_threshold_chars = int(value)
    assert (
        long_text_threshold_chars >= 0
    ), "generation.long_text_threshold_chars must be non-negative"
    return long_text_threshold_chars
def _selected_long_text_template_styles(generation: dict[str, Any]) -> list[str] | None:
    values = generation.get("long_text_template_styles")
    if values is None:
        return None

    long_text_template_styles = _normalized_choices(
        values,
        supported=SUPPORTED_TEMPLATE_STYLES,
        label="generation.long_text_template_styles",
    )
    selected_template_styles = _selected_template_styles(generation)
    unsupported_styles = [
        style for style in long_text_template_styles if style not in selected_template_styles
    ]
    assert not unsupported_styles, (
        "generation.long_text_template_styles must be a subset of "
        f"generation.template_styles: {unsupported_styles}"
    )
    return long_text_template_styles


def _long_text_variant_count(generation: dict[str, Any]) -> int | None:
    long_text_template_styles = _selected_long_text_template_styles(generation)
    if not long_text_template_styles:
        return None
    return len(_variant_specs(generation, template_styles=long_text_template_styles))


def _source_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    if "sources" in cfg:
        entries = list(cfg["sources"])
        assert entries, "sources must not be empty"
        normalized = []
        seen_keys: set[str] = set()
        for index, item in enumerate(entries):
            assert isinstance(item, dict), "sources entries must be mappings"
            assert "source" in item, "sources entries must include source"
            assert "generation" in item, "sources entries must include generation"
            source = dict(item["source"])
            generation = dict(item["generation"])
            raw_key = item.get("name") or source.get("name") or source.get("dataset") or source.get("path")
            key = _normalize_source_key(raw_key, index=index)
            assert key not in seen_keys, f"Duplicate source key: {key}"
            seen_keys.add(key)
            normalized.append({"key": key, "source": source, "generation": generation})
        return normalized

    assert "source" in cfg, "config must include source or sources"
    assert "generation" in cfg, "config must include generation or sources entries"
    raw_key = cfg["source"].get("name") or cfg["source"].get("dataset") or cfg["source"].get("path")
    return [
        {
            "key": _normalize_source_key(raw_key, index=0),
            "source": dict(cfg["source"]),
            "generation": dict(cfg["generation"]),
        }
    ]


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
    }


def _generation_summary(
    generation: dict[str, Any],
    *,
    variant_count: int | None,
    long_text_variant_count: int | None,
    template_pool_counts: dict[str, int] | None,
) -> dict[str, Any]:
    return {
        "max_pairs": _max_pairs(generation),
        "max_pairs_per_source": _max_pairs_per_source(generation),
        "max_pair_chars": _max_pair_chars(generation),
        "long_text_threshold_chars": _long_text_threshold_chars(generation),
        "prompt_languages": _selected_prompt_languages(generation),
        "directions": _selected_directions(generation),
        "template_styles": _selected_template_styles(generation),
        "long_text_template_styles": _selected_long_text_template_styles(generation),
        "variant_count": variant_count if variant_count is not None else len(_variant_specs(generation)),
        "long_text_variant_count": (
            long_text_variant_count
            if long_text_variant_count is not None
            else _long_text_variant_count(generation)
        ),
        "template_pool_counts": template_pool_counts or {},
        "train_fraction": float(generation["train_fraction"]),
    }


def _load_rows(result: BuildResult) -> list[dict[str, Any]]:
    dataset_path = Path(result.artifacts["dataset"].path)
    return store.read_jsonl(dataset_path)


def _load_verified_rows(result: BuildResult) -> list[dict[str, Any]]:
    verified_path = Path(result.run_dir) / "outputs" / "verified.jsonl"
    if verified_path.exists():
        return store.read_jsonl(verified_path)

    rows = _load_rows(result)
    verified_rows = common_eval.verify(rows, _has_prompt, name="prompt_present")
    verified_rows = common_eval.verify(verified_rows, _has_target, name="target_present")
    verified_rows = common_eval.verify(verified_rows, _has_sources, name="source_present")
    return common_eval.verify(
        verified_rows,
        _has_translation_metadata,
        name="translation_metadata_present",
    )


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")
def _has_prompt(row: dict[str, Any]) -> bool:
    return bool(str(row.get("prompt", "")).strip())
def _has_target(row: dict[str, Any]) -> bool:
    return bool(str(row.get("target", "")).strip())
def _has_sources(row: dict[str, Any]) -> bool:
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources:
        return False
    return all(isinstance(source, dict) and source for source in sources)


def _has_translation_metadata(row: dict[str, Any]) -> bool:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return False

    source_language = meta.get("source_language")
    target_language = meta.get("target_language")
    prompt_language = meta.get("prompt_language")
    source_config = meta.get("source_config")
    template_style = meta.get("template_style")
    input_mode = meta.get("input_mode")
    output_mode = meta.get("output_mode")
    pair_max_chars = meta.get("pair_max_chars")
    source_chars = meta.get("source_chars")
    target_chars = meta.get("target_chars")

    if source_language not in SUPPORTED_LANGUAGES:
        return False
    if target_language not in SUPPORTED_LANGUAGES:
        return False
    if source_language == target_language:
        return False
    if prompt_language not in SUPPORTED_LANGUAGES:
        return False
    if not isinstance(source_config, str) or not source_config.strip():
        return False
    if template_style not in SUPPORTED_TEMPLATE_STYLES:
        return False
    if input_mode not in SUPPORTED_INPUT_MODES:
        return False
    if output_mode not in SUPPORTED_OUTPUT_MODES:
        return False
    if not isinstance(pair_max_chars, int) or pair_max_chars <= 0:
        return False
    if not isinstance(source_chars, int) or source_chars <= 0:
        return False
    if not isinstance(target_chars, int) or target_chars <= 0:
        return False
    return target_chars == len(str(row.get("target", "")).strip())


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


def _split_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_entries = _source_entries(cfg)
    if len(source_entries) == 1:
        train_fraction = float(source_entries[0]["generation"]["train_fraction"])
        split_at = int(len(rows) * train_fraction)
        return rows[:split_at], rows[split_at:]

    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_key = str(row.get("meta", {}).get("source_config", "")).strip()
        assert source_key, "rows must include meta.source_config for multi-source publish"
        rows_by_source.setdefault(source_key, []).append(row)

    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for entry in source_entries:
        source_rows = rows_by_source.get(entry["key"], [])
        split_at = int(len(source_rows) * float(entry["generation"]["train_fraction"]))
        train_rows.extend(source_rows[:split_at])
        eval_rows.extend(source_rows[split_at:])
    return train_rows, eval_rows


def _publish_dir(result: BuildResult, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    return reports_root() / result.pack / result.run_id


def _load_or_compute(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return fallback


def _source_label(source: dict[str, Any]) -> str:
    if source.get("dataset"):
        return str(source["dataset"])
    return str(source["path"])


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}
