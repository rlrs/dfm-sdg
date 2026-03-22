from __future__ import annotations

import asyncio

import pytest

from sdg.commons.work_queue import map_async_ordered, map_async_unordered


def test_map_async_ordered_preserves_order_and_reports_total() -> None:
    async def run() -> tuple[list[int], list[tuple[int, int | None, int]], int]:
        current = 0
        max_seen = 0
        lock = asyncio.Lock()
        progress_events: list[tuple[int, int | None, int]] = []

        async def worker(index: int, item: int) -> int:
            nonlocal current, max_seen
            async with lock:
                current += 1
                max_seen = max(max_seen, current)

            await asyncio.sleep(0.03 if item % 2 == 0 else 0.01)

            async with lock:
                current -= 1

            return item * 10

        rows: list[int] = []
        async for row in map_async_ordered(
            (item for item in range(6)),
            worker,
            concurrency=2,
            progress=lambda completed, total, elapsed: progress_events.append((completed, total, elapsed)),
        ):
            rows.append(row)

        return rows, progress_events, max_seen

    rows, progress_events, max_seen = asyncio.run(run())

    assert rows == [0, 10, 20, 30, 40, 50]
    assert max_seen == 2
    assert progress_events[0] == (0, None, 0)
    assert progress_events[-1][0] == 6
    assert progress_events[-1][1] == 6


def test_map_async_ordered_raises_worker_errors() -> None:
    async def run() -> None:
        async def worker(index: int, item: int) -> int:
            if item == 2:
                raise ValueError("bad item")
            return item

        async for _ in map_async_ordered([0, 1, 2, 3], worker, concurrency=2):
            pass

    with pytest.raises(ValueError, match="bad item"):
        asyncio.run(run())


def test_map_async_unordered_yields_completed_results_immediately() -> None:
    async def run() -> list[int]:
        async def worker(index: int, item: int) -> int:
            await asyncio.sleep(0.03 if item == 0 else 0.01)
            return item * 10

        rows: list[int] = []
        async for row in map_async_unordered([0, 1], worker, concurrency=2):
            rows.append(row)
        return rows

    assert asyncio.run(run()) == [10, 0]


def test_map_async_unordered_raises_when_worker_is_cancelled() -> None:
    async def run() -> None:
        async def worker(index: int, item: int) -> int:
            if item == 1:
                raise asyncio.CancelledError()
            return item

        async for _ in map_async_unordered([0, 1, 2], worker, concurrency=2):
            pass

    with pytest.raises(RuntimeError, match="cancelled unexpectedly"):
        asyncio.run(run())
