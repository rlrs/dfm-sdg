from collections import Counter
from collections.abc import Callable
from random import Random
from typing import Any

Item = dict[str, Any]
Bucket = tuple[object, ...]


def plan_from_catalog(count: int, catalog: list[Item] | tuple[Item, ...], rng: Random) -> list[Item]:
    assert catalog, "recipe catalog must not be empty"
    ordered = [dict(item) for item in catalog]
    rng.shuffle(ordered)

    planned: list[Item] = []
    for index in range(count):
        planned.append(dict(ordered[index % len(ordered)]))
    return planned


def interleave_groups(groups: list[list[Item]]) -> list[Item]:
    pending = [list(group) for group in groups]
    merged: list[Item] = []

    while pending:
        next_round: list[list[Item]] = []
        for group in pending:
            if not group:
                continue
            merged.append(group.pop(0))
            if group:
                next_round.append(group)
        pending = next_round

    return merged


def bucket_counts(
    items: list[Item],
    keys: tuple[str, ...],
    *,
    getter: Callable[[Item, str], object] | None = None,
) -> Counter[Bucket]:
    fetch = getter or _default_getter
    counts: Counter[Bucket] = Counter()
    for item in items:
        bucket = tuple(fetch(item, key) for key in keys)
        counts[bucket] += 1
    return counts


def compare_planned_to_observed(
    planned: list[Item],
    observed: list[Item],
    keys: tuple[str, ...],
    *,
    planned_getter: Callable[[Item, str], object] | None = None,
    observed_getter: Callable[[Item, str], object] | None = None,
) -> dict[str, object]:
    planned_counts = bucket_counts(planned, keys, getter=planned_getter)
    observed_counts = bucket_counts(observed, keys, getter=observed_getter)
    buckets = sorted(set(planned_counts) | set(observed_counts), key=_bucket_sort_key)

    missing: dict[str, int] = {}
    mismatched: dict[str, dict[str, int]] = {}
    unexpected: dict[str, int] = {}

    for bucket in buckets:
        planned_value = planned_counts.get(bucket, 0)
        observed_value = observed_counts.get(bucket, 0)
        label = _bucket_label(bucket)

        if planned_value == 0:
            unexpected[label] = observed_value
            continue
        if observed_value == 0:
            missing[label] = planned_value
            continue
        if planned_value != observed_value:
            mismatched[label] = {
                "planned": planned_value,
                "observed": observed_value,
            }

    return {
        "passed": not missing and not mismatched and not unexpected,
        "keys": list(keys),
        "planned": _stringify_counter(planned_counts),
        "observed": _stringify_counter(observed_counts),
        "missing": missing,
        "mismatched": mismatched,
        "unexpected": unexpected,
    }


def counter_minimum_check(counter: Counter[str], minimums: dict[str, int]) -> dict[str, object]:
    missing: dict[str, dict[str, int]] = {}
    for key, minimum in minimums.items():
        observed = counter.get(key, 0)
        if observed < minimum:
            missing[key] = {
                "minimum": minimum,
                "observed": observed,
            }

    return {
        "passed": not missing,
        "minimums": dict(minimums),
        "observed": dict(counter),
        "missing": missing,
    }


def unique_count_check(values: list[str], minimum: int) -> dict[str, object]:
    unique_count = len(set(values))
    return {
        "passed": unique_count >= minimum,
        "minimum": minimum,
        "observed": unique_count,
    }


def _default_getter(item: Item, key: str) -> object:
    return item[key]


def _bucket_label(bucket: Bucket) -> str:
    return " | ".join(str(value) for value in bucket)


def _bucket_sort_key(bucket: Bucket) -> tuple[str, ...]:
    return tuple(str(value) for value in bucket)


def _stringify_counter(counter: Counter[Bucket]) -> dict[str, int]:
    return {
        _bucket_label(bucket): count
        for bucket, count in sorted(counter.items(), key=lambda item: _bucket_sort_key(item[0]))
    }
