# dfm-sdg

`dfm-sdg` is a small framework for synthetic data generation.

It keeps the shared layer thin and puts method-specific logic in packs.

## Repo shape

- `sdg/commons/`: shared runtime code for runs, artifacts, model clients, publishing, viewer support, and pack discovery
- `sdg/packs/demo/`: tiny reference pack showing the full flow end to end
- `sdg/packs/backtranslation/`: backtranslation-style data generation from finished texts
- `sdg/packs/translation/`: translation prompt-surface generation from Danish-English parallel corpora
- `sdg/packs/synth/`: the main synthesis pack for memory-core building, memorization, grounded QA, and related workflows
- `sdg/packs/verifiable_reasoning/`: starter scaffold for exactly checkable reasoning families, starting with logic puzzles
- `sdg/packs/instruction_following/`: expanded IFBench-style instruction-following rows with deterministic verification
- `sdg/packs/tool_use/`: starter scaffold for structured tool-calling rows and validators
- `sdg/packs/python_algorithms/`: starter scaffold for code-generation tasks with executable checks
- `tests/`: pack tests and shared runtime tests
- `artifacts/runs/<pack>/<run-id>/`: local run outputs, logs, progress snapshots, and generated datasets
- `reports/<pack>/<run-id>/`: published outputs such as parquet exports and reports

## Quick start

This project targets Python 3.13 and uses `uv`.

```bash
uv sync --dev
uv run sdg list-packs
uv run sdg build sdg/packs/demo/configs/base.yaml
uv run sdg build sdg/packs/backtranslation/configs/base.yaml
uv run sdg build sdg/packs/translation/configs/base.yaml
uv run sdg build sdg/packs/synth/configs/smoke.yaml
uv run sdg build sdg/packs/verifiable_reasoning/configs/base.yaml
uv run sdg build sdg/packs/instruction_following/configs/ifbench.yaml
uv run sdg build sdg/packs/tool_use/configs/base.yaml
uv run sdg build sdg/packs/python_algorithms/configs/base.yaml
```

## Typical workflow

Run a config:

```bash
uv run sdg build sdg/packs/backtranslation/configs/base.yaml
```

Inspect a run while it is running or after it completes:

```bash
uv run sdg progress <run-id>
uv run sdg summarize <run-id>
uv run sdg view <run-id>
uv run sdg serve <run-id> --open
```

Verify and publish:

```bash
uv run sdg verify <run-id>
uv run sdg publish <run-id>
```

Upload a file artifact from a run to Hugging Face:

```bash
uv run sdg upload-hf <run-id> --artifact dataset --repo <org/name> --private
```

## Working in the repo

In practice there are three common ways to add something:

- add a new config when the task fits an existing pack and existing code path
- add a new task/profile inside a pack when the runtime is shared but the source normalization or prompting changes
- add a new pack when the workflow, artifacts, or verification logic is genuinely different

As a rule:

- keep pack-specific logic inside `sdg.packs.<name>`
- keep shared runtime and generic helpers in `sdg.commons`
- prefer small configs over branching one config for many unrelated tasks

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

If you want to upload to Hugging Face, make sure the machine is logged in with a token that has write access to the target repo or org.

## Development

```bash
uv run ruff check .
uv run pytest
```
