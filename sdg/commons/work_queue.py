from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

Item = TypeVar("Item")
Result = TypeVar("Result")
ProgressFn = Callable[[int, int | None, int], None]
WorkerFn = Callable[[int, Item], Awaitable[Result]]


@dataclass(frozen=True)
class _WorkItem(Generic[Item]):
    index: int
    value: Item


@dataclass(frozen=True)
class _StopWork:
    pass


@dataclass(frozen=True)
class _WorkResult(Generic[Result]):
    index: int
    value: Result


@dataclass(frozen=True)
class _WorkError:
    error: Exception


@dataclass(frozen=True)
class _WorkerFinished:
    pass


InputMessage = _WorkItem[Item] | _StopWork
OutputMessage = _WorkResult[Result] | _WorkError | _WorkerFinished


async def map_async_ordered(
    items: Iterable[Item],
    worker: WorkerFn[Item, Result],
    *,
    concurrency: int,
    progress: ProgressFn | None = None,
    total: int | None = None,
) -> AsyncIterator[Result]:
    worker_count = max(concurrency, 1)
    queue_size = max(worker_count * 2, 1)
    known_total = total if total is not None else _known_total(items)
    started_at = time.monotonic()

    if progress is not None:
        progress(0, known_total, 0)

    pending: asyncio.Queue[InputMessage[Item]] = asyncio.Queue(maxsize=queue_size)
    completed: asyncio.Queue[OutputMessage[Result]] = asyncio.Queue(maxsize=queue_size)
    producer = asyncio.create_task(_produce_work(items, pending, completed, worker_count))
    workers = [
        asyncio.create_task(_run_worker(worker, pending, completed))
        for _ in range(worker_count)
    ]

    next_index = 0
    completed_count = 0
    finished_workers = 0
    buffered: dict[int, Result] = {}

    try:
        while finished_workers < worker_count:
            message = await completed.get()
            match message:
                case _WorkResult(index=index, value=value):
                    buffered[index] = value
                    while next_index in buffered:
                        yield buffered.pop(next_index)
                        next_index += 1
                        completed_count += 1
                        if progress is not None:
                            progress(
                                completed_count,
                                _producer_total(producer, known_total),
                                int(time.monotonic() - started_at),
                            )
                case _WorkError(error=error):
                    raise error
                case _WorkerFinished():
                    finished_workers += 1
                case _:
                    raise AssertionError(f"Unsupported work queue message: {message}")

        produced_count = await producer
        assert completed_count == produced_count, "Work queue lost results"
    finally:
        producer.cancel()
        for task in workers:
            task.cancel()
        await asyncio.gather(producer, *workers, return_exceptions=True)


async def _produce_work(
    items: Iterable[Item],
    pending: asyncio.Queue[InputMessage[Item]],
    completed: asyncio.Queue[OutputMessage[Result]],
    worker_count: int,
) -> int:
    produced_count = 0
    try:
        for index, item in enumerate(items):
            await pending.put(_WorkItem(index=index, value=item))
            produced_count = index + 1
    except Exception as error:
        await completed.put(_WorkError(error))
    finally:
        for _ in range(worker_count):
            await pending.put(_StopWork())
    return produced_count


async def _run_worker(
    worker: WorkerFn[Item, Result],
    pending: asyncio.Queue[InputMessage[Item]],
    completed: asyncio.Queue[OutputMessage[Result]],
) -> None:
    while True:
        message = await pending.get()
        match message:
            case _StopWork():
                await completed.put(_WorkerFinished())
                return
            case _WorkItem(index=index, value=value):
                try:
                    result = await worker(index, value)
                except Exception as error:
                    await completed.put(_WorkError(error))
                    return
                await completed.put(_WorkResult(index=index, value=result))
            case _:
                raise AssertionError(f"Unsupported work item: {message}")


def _known_total(items: Iterable[Item]) -> int | None:
    try:
        return len(items)  # type: ignore[arg-type]
    except TypeError:
        return None


def _producer_total(producer: asyncio.Task[int], known_total: int | None) -> int | None:
    if known_total is not None:
        return known_total
    if not producer.done() or producer.cancelled():
        return None
    return producer.result()
