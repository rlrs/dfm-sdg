from __future__ import annotations

from collections.abc import Iterable


def runtime_concurrency(model: object) -> int:
    runtime = getattr(model, "runtime", None)
    return max(int(getattr(runtime, "max_concurrency", 1)), 1)


def effective_concurrency(models: Iterable[object]) -> int:
    return max((runtime_concurrency(model) for model in models), default=1)
