from dataclasses import dataclass
from itertools import permutations
from random import Random
from typing import Any


@dataclass(frozen=True)
class Clue:
    kind: str
    first: str
    second: str | None = None
    third: str | None = None
    slot: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "kind": self.kind,
            "first": self.first,
            "second": self.second,
            "third": self.third,
            "slot": self.slot,
        }


@dataclass(frozen=True)
class OrderPuzzle:
    family: str
    language: str
    participants: tuple[str, ...]
    solution: tuple[str, ...]
    clues: tuple[Clue, ...]
    prompt: str
    source_subset: str


def select_clues(solution: tuple[str, ...], rng: Random, recipe: dict[str, Any]) -> tuple[Clue, ...]:
    participants = tuple(sorted(solution))
    candidates = _build_candidates(solution, recipe["allowed_kinds"])
    rng.shuffle(candidates)

    selected: list[Clue] = []
    remaining_solutions = solve_puzzle(participants, selected)

    for kind in recipe["required_kinds"]:
        choice = _best_clue_of_kind(
            kind,
            participants,
            candidates,
            selected,
            remaining_solutions,
            recipe["max_kind_counts"],
        )
        assert choice is not None, f"missing required clue kind: {kind}"
        selected.append(choice[0])
        remaining_solutions = choice[1]

    while len(remaining_solutions) > 1:
        choice = _best_clue(
            participants,
            candidates,
            selected,
            remaining_solutions,
            recipe["max_kind_counts"],
            recipe["kind_priority"],
        )
        assert choice is not None, "ordering puzzle must become uniquely solvable"
        selected.append(choice[0])
        remaining_solutions = choice[1]

    while len(selected) < recipe["target_clue_count"]:
        extra = _best_extra_clue(
            participants,
            candidates,
            selected,
            remaining_solutions,
            recipe["max_kind_counts"],
            recipe["kind_priority"],
        )
        if extra is None:
            break
        selected.append(extra)

    assert remaining_solutions == [solution], "generated clue set must resolve to exactly one order"
    return tuple(selected)


def clue_from_dict(payload: dict[str, str | int | None]) -> Clue:
    return Clue(
        kind=str(payload["kind"]),
        first=str(payload["first"]),
        second=_optional_text(payload.get("second")),
        third=_optional_text(payload.get("third")),
        slot=_optional_int(payload.get("slot")),
    )


def format_target(solution: tuple[str, ...]) -> str:
    return ", ".join(solution)


def parse_target(text: str, participants: list[str] | tuple[str, ...]) -> tuple[str, ...] | None:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != len(participants):
        return None
    if set(parts) != set(participants):
        return None
    return tuple(parts)


def solve_puzzle(participants: tuple[str, ...], clues: list[Clue] | tuple[Clue, ...]) -> list[tuple[str, ...]]:
    valid_orders: list[tuple[str, ...]] = []
    for order in permutations(participants):
        if all(_clue_holds(order, clue) for clue in clues):
            valid_orders.append(tuple(order))
    return valid_orders


def _optional_text(value: str | int | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _build_candidates(solution: tuple[str, ...], allowed_kinds: tuple[str, ...]) -> list[Clue]:
    candidates: list[Clue] = []
    allowed = set(allowed_kinds)

    if "slot" in allowed:
        candidates.extend(Clue("slot", name, slot=index + 1) for index, name in enumerate(solution))

    if "immediately_before" in allowed or "adjacent" in allowed or "one_between" in allowed:
        for index in range(len(solution) - 1):
            left = solution[index]
            right = solution[index + 1]
            if "immediately_before" in allowed:
                candidates.append(Clue("immediately_before", left, right))
            if "adjacent" in allowed:
                candidates.append(Clue("adjacent", left, right))

        if "one_between" in allowed:
            for index in range(len(solution) - 2):
                left = solution[index]
                right = solution[index + 2]
                candidates.append(Clue("one_between", left, right))

    if "before" in allowed:
        for left_index, left in enumerate(solution):
            for right in solution[left_index + 1 :]:
                candidates.append(Clue("before", left, right))

    if "not_adjacent" in allowed:
        for left_index, left in enumerate(solution):
            for right in solution[left_index + 1 :]:
                if abs(left_index - solution.index(right)) > 1:
                    candidates.append(Clue("not_adjacent", left, right))

    if "between" in allowed:
        for middle_index in range(1, len(solution) - 1):
            middle = solution[middle_index]
            for left_index in range(middle_index):
                for right_index in range(middle_index + 1, len(solution)):
                    candidates.append(Clue("between", middle, solution[left_index], solution[right_index]))

    unique: list[Clue] = []
    seen: set[tuple[str, str, str | None, str | None, int | None]] = set()
    for clue in candidates:
        key = (clue.kind, clue.first, clue.second, clue.third, clue.slot)
        if key in seen:
            continue
        seen.add(key)
        unique.append(clue)
    return unique


def _best_clue_of_kind(
    kind: str,
    participants: tuple[str, ...],
    candidates: list[Clue],
    selected: list[Clue],
    remaining_solutions: list[tuple[str, ...]],
    max_kind_counts: dict[str, int],
) -> tuple[Clue, list[tuple[str, ...]]] | None:
    best_choice: tuple[Clue, list[tuple[str, ...]]] | None = None
    best_score: int | None = None

    for clue in candidates:
        if clue.kind != kind or clue in selected:
            continue
        if _exceeds_kind_limit(selected, clue, max_kind_counts):
            continue

        narrowed = solve_puzzle(participants, [*selected, clue])
        if len(narrowed) == len(remaining_solutions):
            continue
        if best_score is None or len(narrowed) < best_score:
            best_score = len(narrowed)
            best_choice = (clue, narrowed)

    return best_choice


def _best_clue(
    participants: tuple[str, ...],
    candidates: list[Clue],
    selected: list[Clue],
    remaining_solutions: list[tuple[str, ...]],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> tuple[Clue, list[tuple[str, ...]]] | None:
    best_choice: tuple[Clue, list[tuple[str, ...]]] | None = None
    best_score: tuple[int, int] | None = None

    for clue in candidates:
        if clue in selected:
            continue
        if _exceeds_kind_limit(selected, clue, max_kind_counts):
            continue

        narrowed = solve_puzzle(participants, [*selected, clue])
        if len(narrowed) == len(remaining_solutions):
            continue

        score = (len(narrowed), kind_priority[clue.kind])
        if best_score is None or score < best_score:
            best_score = score
            best_choice = (clue, narrowed)

    return best_choice


def _best_extra_clue(
    participants: tuple[str, ...],
    candidates: list[Clue],
    selected: list[Clue],
    remaining_solutions: list[tuple[str, ...]],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> Clue | None:
    best_clue: Clue | None = None
    best_score: tuple[int, int] | None = None

    for clue in candidates:
        if clue in selected:
            continue
        if _exceeds_kind_limit(selected, clue, max_kind_counts):
            continue

        narrowed = solve_puzzle(participants, [*selected, clue])
        if not narrowed:
            continue

        score = (kind_priority[clue.kind], len(narrowed))
        if best_score is None or score < best_score:
            best_score = score
            best_clue = clue

    return best_clue


def _exceeds_kind_limit(selected: list[Clue], clue: Clue, max_kind_counts: dict[str, int]) -> bool:
    if clue.kind not in max_kind_counts:
        return False
    count = sum(1 for item in selected if item.kind == clue.kind)
    return count >= max_kind_counts[clue.kind]


def _clue_holds(order: tuple[str, ...], clue: Clue) -> bool:
    positions = {name: index for index, name in enumerate(order)}
    first_position = positions[clue.first]

    if clue.kind == "slot":
        assert clue.slot is not None, "slot clue requires slot"
        return first_position == clue.slot - 1

    assert clue.second is not None, f"{clue.kind} clue requires second"
    second_position = positions[clue.second]

    if clue.kind == "before":
        return first_position < second_position
    if clue.kind == "immediately_before":
        return first_position + 1 == second_position
    if clue.kind == "adjacent":
        return abs(first_position - second_position) == 1
    if clue.kind == "not_adjacent":
        return abs(first_position - second_position) > 1
    if clue.kind == "one_between":
        return abs(first_position - second_position) == 2
    if clue.kind == "between":
        assert clue.third is not None, "between clue requires third"
        third_position = positions[clue.third]
        return (second_position < first_position < third_position) or (third_position < first_position < second_position)

    raise AssertionError(f"Unsupported clue kind: {clue.kind}")
