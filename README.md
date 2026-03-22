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

## Adding a pack

Packs live under `sdg/packs/<name>`.

Start by copying the shape of `sdg/packs/demo`:

- add `pack.yaml` with `build`, `verify`, `summarize`, and `publish` entrypoints
- add one config under `configs/` so the pack is runnable from the CLI
- keep pack-specific logic inside `sdg.packs.<name>.*` modules and call into `sdg.commons.*` only for shared concerns like runs, storage, publishing, model clients, and viewer helpers
- add a `README.md` that explains the pack's scope and current status
- add a `viewer` entrypoint only if the default viewer needs pack-specific labels, sections, filters, or a default artifact

Minimal `pack.yaml`:

```yaml
name: my_pack
description: Short pack description.
entrypoints:
  build: sdg.packs.my_pack.build:build
  verify: sdg.packs.my_pack.build:verify
  summarize: sdg.packs.my_pack.build:summarize
  publish: sdg.packs.my_pack.build:publish
```

Then add a test like `tests/test_demo_pack.py` that exercises build, verify, summarize, publish, and view for the new pack.

## Models

Model-backed steps load named endpoints from `.env`.

Use pack configs to bind model roles to those named endpoints.

## Development

```bash
uv run ruff check .
uv run pytest
```
