# dfm-sdg

`dfm-sdg` is a small framework for synthetic data generation.

It keeps the shared layer thin and puts method-specific logic in packs.

## What is here

- `sdg.commons.*` handles runs, artifacts, model endpoints, verification helpers, publishing helpers, and pack discovery.
- `sdg.packs.demo` is a tiny arithmetic pack that exercises the full build and verify flow.
- `sdg.packs.synth` is the main research pack. It currently supports memory-core building from Wikipedia-style sources, memorization generation, grounded QA generation, verification, publication, and run viewing.
- `sdg` is the CLI entrypoint.

## Current CLI

The CLI currently exposes:

- `build`
- `verify`
- `summarize`
- `publish`
- `compare`
- `events`
- `progress`
- `view`
- `serve`
- `list-packs`

## Quick start

This project targets Python 3.13 and uses `uv`.

```bash
uv sync --dev
uv run sdg list-packs
uv run sdg build sdg/packs/demo/configs/base.yaml
uv run sdg build sdg/packs/synth/configs/smoke.yaml
uv run sdg summarize <run-id>
uv run sdg verify <run-id>
uv run sdg serve <run-id> --open
```

## Models

Model-backed steps load named endpoints from `.env`.

Use pack configs to bind model roles to those named endpoints.

## Development

```bash
uv run ruff check .
uv run pytest
```
