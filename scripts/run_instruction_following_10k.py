from __future__ import annotations

from pathlib import Path

from sdg.commons.run import run
from sdg.commons.utils import read_yaml
from sdg.packs.instruction_following.build import _build_run, verify


def main() -> None:
    config_path = Path("sdg/packs/instruction_following/configs/ifbench_laerebogen_da_10k.yaml")
    cfg = read_yaml(config_path)

    print(f"starting build from {config_path}", flush=True)
    result = run(
        _build_run,
        pack="instruction_following",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
        resume_incomplete=False,
    )
    print(f"build_complete run_id={result.run_id} run_dir={result.run_dir}", flush=True)

    verification = verify(result.run_id)
    print(
        "verify_complete "
        f"run_id={result.run_id} "
        f"verified_rows={verification['verified_rows']} "
        f"failed_rows={verification['failed_rows']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
