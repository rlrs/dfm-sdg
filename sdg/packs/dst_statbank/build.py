from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import eval as common_eval
from sdg.commons import publish as common_publish
from sdg.commons.run import load, run
from sdg.commons.utils import read_json, reports_root, write_json

PACK = "dst_statbank"

_DATE_LINE_RE = re.compile(r"^\d{1,2}\.\s+\w+ \d{4}\s*$", re.MULTILINE)
_TRAILING_RE = re.compile(
    r"\n+(Vis hele teksten|Hent som PDF|Næste udgivelse|Del sidens indhold"
    r"|Statistik\xaddokumentation|« Minimer teksten).*$",
    re.DOTALL,
)
# Strips trailing DST footer: "<series title>\n\n<DD. måned YYYY - Nr. NNN>"
_TRAILING_FOOTER_RE = re.compile(
    r"\n+[^\n]+\n+\d{1,2}\.\s+\w+\s+\d{4}[^\n]*\s*$"
)


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

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
    min_chars = int(cfg.get("generation", {}).get("min_target_chars", 200))

    verified = common_eval.verify(rows, lambda r: bool(r.get("prompt")), name="prompt_present")
    verified = common_eval.verify(verified, lambda r: bool(r.get("target")), name="target_present")
    verified = common_eval.verify(
        verified,
        lambda r: len(r.get("target", "")) >= min_chars,
        name="target_min_chars",
    )
    verified = common_eval.verify(
        verified,
        lambda r: any(c.isdigit() for c in r.get("prompt", "")),
        name="prompt_has_numbers",
    )

    failures = [r for r in verified if not _row_passes(r)]
    outputs_dir = Path(result.run_dir) / "outputs"
    store.write_jsonl(verified, outputs_dir / "verified.jsonl")
    store.write_jsonl(failures, outputs_dir / "failures.jsonl")

    metrics = common_eval.aggregate_metrics(verified)
    failure_summary = common_eval.summarize_failures(verified)
    write_json(metrics, outputs_dir / "metrics.json")
    write_json(failure_summary, outputs_dir / "failure_summary.json")
    common_publish.write_preview(verified, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "run_id": result.run_id,
        "verified_rows": len(verified),
        "failed_rows": len(failures),
        "metrics": metrics,
        "failure_summary": failure_summary,
    }


def summarize(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "artifacts": sorted(result.artifacts),
        "metrics": metrics,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    train_frac = float(cfg.get("generation", {}).get("train_fraction", 0.9))

    failures = [r for r in rows if not _row_passes(r)]
    passing = [r for r in rows if _row_passes(r)]
    split = int(len(passing) * train_frac)
    train_rows, eval_rows = passing[:split], passing[split:]

    target_dir = _publish_dir(result, out_dir)
    store.ensure_dir(target_dir)
    store.write_parquet(train_rows, target_dir / "train.parquet")
    store.write_parquet(eval_rows, target_dir / "eval.parquet")
    store.write_parquet(failures, target_dir / "failures.parquet")
    common_publish.write_preview(rows, target_dir / "sample_preview.jsonl", n=20)

    outputs_dir = Path(result.run_dir) / "outputs"
    metrics = read_json(outputs_dir / "metrics.json") if (outputs_dir / "metrics.json").exists() else {}
    failure_summary = read_json(outputs_dir / "failure_summary.json") if (outputs_dir / "failure_summary.json").exists() else {}

    write_json(metrics, target_dir / "metrics.json")
    write_json(failure_summary, target_dir / "failure_summary.json")
    common_publish.write_report(metrics, failure_summary, target_dir / "report.json")
    common_publish.write_manifest(
        {
            "pack": result.pack,
            "run_id": result.run_id,
            "source_run_dir": result.run_dir,
            "published_artifacts": [
                "train.parquet", "eval.parquet", "failures.parquet",
                "sample_preview.jsonl", "manifest.json",
                "metrics.json", "failure_summary.json", "report.json",
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


# ---------------------------------------------------------------------------
# Core build logic
# ---------------------------------------------------------------------------

def _build_run(
    *,
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
) -> dict[str, Artifact]:
    del seed

    progress_file = Path(cfg["source"]["progress_file"])
    cache_file = Path(cfg["source"]["cache_file"])
    gen = cfg.get("generation", {})
    max_tables = int(gen.get("max_tables_per_article", 5))
    max_rows = int(gen.get("max_rows_per_table", 150))
    min_target_chars = int(gen.get("min_target_chars", 200))
    max_articles = gen.get("max_articles")  # None = no limit

    print(f"Loading StatBank cache from {cache_file}...")
    cache = _load_cache(cache_file)
    print(f"  {len(cache)} cached articles, {sum(1 for v in cache.values() if v.get('statbank_data'))} with data")

    print(f"Loading articles from {progress_file}...")
    articles = _load_articles(progress_file, content_types=cfg["source"].get("content_types", ["nyt"]))
    print(f"  {len(articles)} articles loaded")

    rows = []
    skipped_no_cache = skipped_no_data = skipped_short = 0

    for article in articles:
        if max_articles is not None and len(rows) >= max_articles:
            break

        url = article["url"]
        cached = cache.get(url)

        if cached is None:
            skipped_no_cache += 1
            continue
        if not cached.get("statbank_data"):
            skipped_no_data += 1
            continue

        target = _clean_target(article.get("text", ""))
        if len(target) < min_target_chars:
            skipped_short += 1
            continue

        prompt = _build_prompt(cached["statbank_data"], max_tables=max_tables, max_rows=max_rows)
        if not prompt:
            skipped_no_data += 1
            continue

        row = {
            "id": f"{PACK}-{len(rows):06d}",
            "prompt": prompt,
            "target": target,
            "sources": [{
                "dataset": "oliverkinch/danmarks-statistik",
                "url": url,
                "table_codes": cached.get("statbank_tables", []),
            }],
            "meta": {
                "title": article.get("title", ""),
                "date": article.get("date", ""),
                "series": article.get("series", ""),
                "content_type": article.get("content_type", ""),
                "table_codes": cached.get("statbank_tables", []),
                "tables_used": list(cached["statbank_data"].keys())[:max_tables],
                "target_chars": len(target),
            },
        }
        rows.append(row)

    print(f"\nBuilt {len(rows)} rows")
    print(f"  skipped (not in cache): {skipped_no_cache}")
    print(f"  skipped (no table data): {skipped_no_data}")
    print(f"  skipped (target too short): {skipped_short}")

    dataset_path = outputs_dir / "dataset.jsonl"
    store.write_jsonl(rows, dataset_path)
    common_publish.write_preview(rows, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={
                "rows": len(rows),
                "skipped_no_cache": skipped_no_cache,
                "skipped_no_data": skipped_no_data,
                "skipped_short": skipped_short,
            },
        )
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cache(path: Path) -> dict[str, dict]:
    """Load statbank_cache.jsonl keyed by URL. On duplicates, last entry wins."""
    if not path.exists():
        raise FileNotFoundError(f"StatBank cache not found: {path}. Run enrich_statbank.py first.")
    cache: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                cache[r["url"]] = r
            except Exception:
                pass
    return cache


def _load_articles(path: Path, content_types: list[str]) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Progress file not found: {path}. Run scraper.py first.")
    articles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") == "ok" and r.get("content_type") in content_types:
            articles.append(r)
    return articles


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def _clean_target(text: str) -> str:
    """Strip leading date line and trailing navigation boilerplate."""
    # Remove leading date line (e.g. "22. april 2026\n\n")
    text = _DATE_LINE_RE.sub("", text, count=1).lstrip("\n")
    # Remove trailing navigation text
    text = _TRAILING_RE.sub("", text)
    # Remove trailing DST footer: "<series title>\n\n<DD. måned YYYY - Nr. NNN>"
    text = _TRAILING_FOOTER_RE.sub("", text)
    return text.strip()


def _build_prompt(
    statbank_data: dict[str, dict],
    max_tables: int,
    max_rows: int,
) -> str:
    """Format StatBank table data as a Danish instruction prompt."""
    tables = list(statbank_data.items())[:max_tables]
    if not tables:
        return ""

    sections: list[str] = []
    for code, data in tables:
        rows = data.get("rows", [])
        if not rows:
            continue
        section = _format_table(
            code=code,
            title=data.get("table_title", ""),
            unit=data.get("unit", ""),
            rows=rows[:max_rows],
            total_rows=len(rows),
        )
        sections.append(section)

    if not sections:
        return ""

    data_block = "\n\n".join(sections)
    return (
        "Jeg har trukket følgende data fra Danmarks Statistiks statistikbank. "
        "Hjælp mig med at skrive en kort statistisk nyhedsartikel på dansk, "
        "der præsenterer de vigtigste resultater – i stil med artiklerne i "
        "\"Nyt fra Danmarks Statistik\".\n\n"
        + data_block
    )


def _format_table(
    code: str,
    title: str,
    unit: str,
    rows: list[dict],
    total_rows: int,
) -> str:
    if not rows:
        return ""

    cols = list(rows[0].keys())
    header = " | ".join(cols)
    separator = " | ".join("---" for _ in cols)
    data_lines = [" | ".join(str(row.get(c, "")) for c in cols) for row in rows]

    truncation_note = f"\n*(viser {len(rows)} af {total_rows} rækker)*" if total_rows > len(rows) else ""

    unit_str = f", enhed: {unit}" if unit and unit != "-" else ""
    return (
        f"**{title} (tabel: {code}{unit_str})**\n\n"
        f"| {header} |\n"
        f"| {separator} |\n"
        + "\n".join(f"| {line} |" for line in data_lines)
        + truncation_note
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_passes(row: dict) -> bool:
    checks = row.get("checks", {})
    return all(checks.get(k, True) for k in ("prompt_present", "target_present", "target_min_chars", "prompt_has_numbers"))


def _load_rows(result: BuildResult) -> list[dict]:
    outputs_dir = Path(result.run_dir) / "outputs"
    path = outputs_dir / "dataset.jsonl"
    return store.read_jsonl(path) if path.exists() else []


def _load_verified_rows(result: BuildResult) -> list[dict]:
    outputs_dir = Path(result.run_dir) / "outputs"
    path = outputs_dir / "verified.jsonl"
    if path.exists():
        return store.read_jsonl(path)
    return _load_rows(result)


def _load_cfg(result: BuildResult) -> dict:
    cfg_path = Path(result.run_dir) / "cfg.json"
    return read_json(cfg_path) if cfg_path.exists() else {}


def _publish_dir(result: BuildResult, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir)
    return reports_root() / PACK / result.run_id
