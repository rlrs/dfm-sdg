import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass

ProgressFn = Callable[[int, int | None, int], None]
type WorkerFn[Item, Result] = Callable[[int, Item], Awaitable[Result]]


@dataclass(frozen=True)
class _WorkItem[Item]:
    index: int
    value: Item


@dataclass(frozen=True)
class _StopWork:
    pass


@dataclass(frozen=True)
class _WorkResult[Result]:
    index: int
    value: Result


@dataclass(frozen=True)
class _WorkError:
    error: Exception


@dataclass(frozen=True)
class _WorkerFinished:
    pass


type InputMessage[Item] = _WorkItem[Item] | _StopWork
type OutputMessage[Result] = _WorkResult[Result] | _WorkError | _WorkerFinished


async def map_async_ordered[Item, Result](
    items: Iterable[Item],
    worker: WorkerFn[Item, Result],
    *,
    concurrency: int,
    progress: ProgressFn | None = None,
    total: int | None = None,
    producer_threaded: bool = False,
) -> AsyncIterator[Result]:
    async for value in _map_async(
        items,
        worker,
        concurrency=concurrency,
        progress=progress,
        total=total,
        ordered=True,
        producer_threaded=producer_threaded,
    ):
        yield value


async def map_async_unordered[Item, Result](
    items: Iterable[Item],
    worker: WorkerFn[Item, Result],
    *,
    concurrency: int,
    progress: ProgressFn | None = None,
    total: int | None = None,
    producer_threaded: bool = False,
) -> AsyncIterator[Result]:
    async for value in _map_async(
        items,
        worker,
        concurrency=concurrency,
        progress=progress,
        total=total,
        ordered=False,
        producer_threaded=producer_threaded,
    ):
        yield value


async def _map_async[Item, Result](
    items: Iterable[Item],
    worker: WorkerFn[Item, Result],
    *,
    concurrency: int,
    progress: ProgressFn | None,
    total: int | None,
    ordered: bool,
    producer_threaded: bool,
) -> AsyncIterator[Result]:
    worker_count = max(concurrency, 1)
    queue_size = max(worker_count * 2, 1)
    known_total = total if total is not None else _known_total(items)
    started_at = time.monotonic()

    if progress is not None:
        progress(0, known_total, 0)

    pending: asyncio.Queue[InputMessage[Item]] = asyncio.Queue(maxsize=queue_size)
    completed: asyncio.Queue[OutputMessage[Result]] = asyncio.Queue(maxsize=queue_size)
    producer = asyncio.create_task(
        _produce_work(
            items,
            pending,
            completed,
            worker_count,
            producer_threaded=producer_threaded,
        )
    )
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
            wait_for_completed = asyncio.create_task(completed.get())
            done, _ = await asyncio.wait(
                {wait_for_completed, *workers},
                return_when=asyncio.FIRST_COMPLETED,
            )

            worker_error = _unexpected_worker_error(workers, done)
            if worker_error is not None:
                wait_for_completed.cancel()
                await asyncio.gather(wait_for_completed, return_exceptions=True)
                raise worker_error

            if wait_for_completed not in done:
                wait_for_completed.cancel()
                await asyncio.gather(wait_for_completed, return_exceptions=True)
                continue

            message = wait_for_completed.result()
            match message:
                case _WorkResult(index=index, value=value):
                    if not ordered:
                        yield value
                        completed_count += 1
                        if progress is not None:
                            progress(
                                completed_count,
                                _producer_total(producer, known_total),
                                int(time.monotonic() - started_at),
                            )
                        continue

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


async def _produce_work[Item, Result](
    items: Iterable[Item],
    pending: asyncio.Queue[InputMessage[Item]],
    completed: asyncio.Queue[OutputMessage[Result]],
    worker_count: int,
    *,
    producer_threaded: bool,
) -> int:
    produced_count = 0
    try:
        item_iterator = iter(items)
        async for index, item in _enumerate_items(item_iterator, producer_threaded=producer_threaded):
            await pending.put(_WorkItem(index=index, value=item))
            produced_count = index + 1
    except Exception as error:
        await completed.put(_WorkError(error))
    finally:
        for _ in range(worker_count):
            await pending.put(_StopWork())
    return produced_count


async def _enumerate_items[Item](
    item_iterator,
    *,
    producer_threaded: bool,
) -> AsyncIterator[tuple[int, Item]]:
    sentinel = object()
    index = 0

    while True:
        if producer_threaded:
            item = await asyncio.to_thread(next, item_iterator, sentinel)
        else:
            item = next(item_iterator, sentinel)

        if item is sentinel:
            return

        yield index, item
        index += 1


async def _run_worker[Item, Result](
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


def _known_total[Item](items: Iterable[Item]) -> int | None:
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


def _unexpected_worker_error(
    workers: list[asyncio.Task[None]],
    done: set[asyncio.Task[object]],
) -> Exception | None:
    for worker in workers:
        if worker not in done:
            continue
        if worker.cancelled():
            return RuntimeError("Work queue worker was cancelled unexpectedly")
        error = worker.exception()
        if error is not None:
            return error
    return None
