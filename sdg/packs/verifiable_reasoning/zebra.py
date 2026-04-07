from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from random import Random
from typing import Any

from sdg.commons import diversity

HOUSE_COUNT = 4


@dataclass(frozen=True)
class ZebraClue:
    kind: str
    left: str
    right: str | None = None
    house: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "kind": self.kind,
            "left": self.left,
            "right": self.right,
            "house": self.house,
        }


@dataclass(frozen=True)
class AxisSpec:
    key: str
    labels: dict[str, str]
    singular_labels: dict[str, str]
    values: dict[str, tuple[str, ...]]


AXIS_SPECS = {
    "name": AxisSpec(
        key="name",
        labels={"en": "Residents", "da": "Beboere"},
        singular_labels={"en": "Name", "da": "Navn"},
        values={
            "en": ("Alice", "Ben", "Chloe", "Daniel", "Emma", "Grace", "Henry", "Lucy"),
            "da": ("Anna", "Bo", "Clara", "Emil", "Ida", "Karl", "Maja", "Sofie"),
        },
    ),
    "drink": AxisSpec(
        key="drink",
        labels={"en": "Drinks", "da": "Drikke"},
        singular_labels={"en": "Drink", "da": "Drik"},
        values={
            "en": ("coffee", "juice", "lemonade", "milk", "tea", "water", "cocoa", "cider"),
            "da": ("kaffe", "juice", "limonade", "mælk", "te", "vand", "kakao", "cider"),
        },
    ),
    "pet": AxisSpec(
        key="pet",
        labels={"en": "Pets", "da": "Kæledyr"},
        singular_labels={"en": "Pet", "da": "Kæledyr"},
        values={
            "en": ("bird", "cat", "dog", "fish", "hamster", "rabbit", "turtle", "parrot"),
            "da": ("fisk", "fugl", "hund", "hamster", "kanin", "kat", "papegøje", "skildpadde"),
        },
    ),
    "color": AxisSpec(
        key="color",
        labels={"en": "Colors", "da": "Farver"},
        singular_labels={"en": "Color", "da": "Farve"},
        values={
            "en": ("blue", "green", "orange", "purple", "red", "white", "yellow", "black"),
            "da": ("blå", "grøn", "gul", "hvid", "lilla", "orange", "rød", "sort"),
        },
    ),
    "lunch": AxisSpec(
        key="lunch",
        labels={"en": "Lunches", "da": "Frokoster"},
        singular_labels={"en": "Lunch", "da": "Frokost"},
        values={
            "en": ("curry", "omelette", "pasta", "pizza", "salad", "sandwich", "soup", "tacos"),
            "da": ("curry", "omelet", "pasta", "pizza", "salat", "sandwich", "suppe", "tacos"),
        },
    ),
    "flower": AxisSpec(
        key="flower",
        labels={"en": "Flowers", "da": "Blomster"},
        singular_labels={"en": "Flower", "da": "Blomst"},
        values={
            "en": ("carnations", "daisies", "irises", "lilies", "orchids", "roses", "tulips", "violets"),
            "da": ("nelliker", "orkidéer", "roser", "stedmoderblomster", "tulipaner", "violer", "iris", "liljer"),
        },
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_grid_household",
        "difficulty": "easy",
        "prompt_style": "grid",
        "axis_profile": "drink_pet",
        "attribute_axes": ("drink", "pet"),
        "clue_profile": "absolute",
        "target_clue_count": 6,
        "required_kinds": ("same_house", "house", "adjacent"),
        "allowed_kinds": ("same_house", "left_of", "immediately_left_of", "adjacent", "one_between", "house"),
        "max_kind_counts": {"house": 2, "same_house": 3},
        "kind_priority": {"same_house": 0, "immediately_left_of": 1, "adjacent": 2, "one_between": 3, "left_of": 4, "house": 5},
    },
    {
        "recipe_id": "easy_ledger_color_pet",
        "difficulty": "easy",
        "prompt_style": "ledger",
        "axis_profile": "color_pet",
        "attribute_axes": ("color", "pet"),
        "clue_profile": "absolute",
        "target_clue_count": 6,
        "required_kinds": ("same_house", "house", "immediately_left_of"),
        "allowed_kinds": ("same_house", "left_of", "immediately_left_of", "adjacent", "one_between", "house"),
        "max_kind_counts": {"house": 2, "same_house": 3},
        "kind_priority": {"same_house": 0, "immediately_left_of": 1, "adjacent": 2, "one_between": 3, "left_of": 4, "house": 5},
    },
    {
        "recipe_id": "easy_deduce_meal_flower",
        "difficulty": "easy",
        "prompt_style": "deduce",
        "axis_profile": "lunch_flower",
        "attribute_axes": ("lunch", "flower"),
        "clue_profile": "spacing",
        "target_clue_count": 6,
        "required_kinds": ("same_house", "house", "one_between"),
        "allowed_kinds": ("same_house", "left_of", "immediately_left_of", "adjacent", "one_between", "house"),
        "max_kind_counts": {"house": 2, "same_house": 3},
        "kind_priority": {"same_house": 0, "one_between": 1, "immediately_left_of": 2, "adjacent": 3, "left_of": 4, "house": 5},
    },
    {
        "recipe_id": "medium_grid_drink_color",
        "difficulty": "medium",
        "prompt_style": "grid",
        "axis_profile": "drink_color",
        "attribute_axes": ("drink", "color"),
        "clue_profile": "relational",
        "target_clue_count": 7,
        "required_kinds": ("same_house", "left_of", "adjacent"),
        "allowed_kinds": ("same_house", "left_of", "immediately_left_of", "adjacent", "one_between", "house"),
        "max_kind_counts": {"house": 1, "same_house": 3},
        "kind_priority": {"same_house": 0, "immediately_left_of": 1, "one_between": 2, "adjacent": 3, "left_of": 4, "house": 5},
    },
    {
        "recipe_id": "medium_ledger_lunch_pet",
        "difficulty": "medium",
        "prompt_style": "ledger",
        "axis_profile": "lunch_pet",
        "attribute_axes": ("lunch", "pet"),
        "clue_profile": "spacing",
        "target_clue_count": 7,
        "required_kinds": ("same_house", "immediately_left_of", "one_between"),
        "allowed_kinds": ("same_house", "left_of", "immediately_left_of", "adjacent", "one_between", "house"),
        "max_kind_counts": {"house": 1, "same_house": 3},
        "kind_priority": {"same_house": 0, "one_between": 1, "immediately_left_of": 2, "adjacent": 3, "left_of": 4, "house": 5},
    },
    {
        "recipe_id": "medium_deduce_flower_color",
        "difficulty": "medium",
        "prompt_style": "deduce",
        "axis_profile": "flower_color",
        "attribute_axes": ("flower", "color"),
        "clue_profile": "mixed",
        "target_clue_count": 7,
        "required_kinds": ("same_house", "left_of", "house"),
        "allowed_kinds": ("same_house", "left_of", "immediately_left_of", "adjacent", "one_between", "house"),
        "max_kind_counts": {"house": 1, "same_house": 3},
        "kind_priority": {"same_house": 0, "immediately_left_of": 1, "adjacent": 2, "one_between": 3, "left_of": 4, "house": 5},
    },
    {
        "recipe_id": "hard_grid_negative_household",
        "difficulty": "hard",
        "prompt_style": "grid",
        "axis_profile": "drink_pet",
        "attribute_axes": ("drink", "pet"),
        "clue_profile": "negative",
        "target_clue_count": 8,
        "required_kinds": ("same_house", "not_same_house", "one_between", "left_of"),
        "allowed_kinds": ("same_house", "not_same_house", "left_of", "immediately_left_of", "adjacent", "one_between"),
        "max_kind_counts": {"same_house": 3},
        "kind_priority": {"same_house": 0, "one_between": 1, "not_same_house": 2, "immediately_left_of": 3, "adjacent": 4, "left_of": 5},
    },
    {
        "recipe_id": "hard_ledger_negative_meal",
        "difficulty": "hard",
        "prompt_style": "ledger",
        "axis_profile": "lunch_flower",
        "attribute_axes": ("lunch", "flower"),
        "clue_profile": "negative",
        "target_clue_count": 8,
        "required_kinds": ("same_house", "not_same_house", "immediately_left_of", "left_of"),
        "allowed_kinds": ("same_house", "not_same_house", "left_of", "immediately_left_of", "adjacent", "one_between"),
        "max_kind_counts": {"same_house": 3},
        "kind_priority": {"same_house": 0, "not_same_house": 1, "one_between": 2, "immediately_left_of": 3, "adjacent": 4, "left_of": 5},
    },
    {
        "recipe_id": "hard_deduce_negative_color",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "axis_profile": "color_pet",
        "attribute_axes": ("color", "pet"),
        "clue_profile": "negative",
        "target_clue_count": 8,
        "required_kinds": ("same_house", "not_same_house", "adjacent", "one_between"),
        "allowed_kinds": ("same_house", "not_same_house", "left_of", "immediately_left_of", "adjacent", "one_between"),
        "max_kind_counts": {"same_house": 3},
        "kind_priority": {"same_house": 0, "not_same_house": 1, "one_between": 2, "immediately_left_of": 3, "adjacent": 4, "left_of": 5},
    },
)


def recipe_catalog(language: str) -> tuple[dict[str, Any], ...]:
    return RECIPES


def generate_row(index: int, rng: Random, *, language: str, recipe: dict[str, Any]) -> dict[str, object]:
    axes = ("name", *recipe["attribute_axes"])

    for _ in range(200):
        axis_values = _sample_axis_values(axes, language=language, rng=rng)
        solution_state = {
            axis: tuple(rng.sample(axis_values[axis], k=HOUSE_COUNT))
            for axis in axes
        }
        clues = _select_clues(solution_state, axis_values, axes, recipe, rng)
        if clues is None:
            continue

        solution_rows = _rows_from_state(solution_state, axes)
        return {
            "id": f"verifiable-reasoning-{index:05d}",
            "prompt": _format_prompt(axis_values, axes, clues, language=language, recipe=recipe),
            "target": format_target(solution_rows, axes),
            "hidden": {
                "axes": list(axes),
                "axis_values": {axis: list(values) for axis, values in axis_values.items()},
                "solution_rows": solution_rows,
                "clues": [clue.to_dict() for clue in clues],
            },
            "sources": [{"kind": "dolci_subset", "value": "zebralogics"}],
            "meta": {
                "family": "zebra_logic",
                "domain": "logic_puzzles",
                "prompt_language": language,
                "target_language": language,
                "clue_count": len(clues),
                "house_count": HOUSE_COUNT,
                "output_format": "house_table",
                "recipe_id": recipe["recipe_id"],
                "difficulty": recipe["difficulty"],
                "prompt_style": recipe["prompt_style"],
                "axis_profile": recipe["axis_profile"],
                "axis_count": len(axes),
                "clue_profile": recipe["clue_profile"],
            },
        }

    raise AssertionError("failed to generate zebra puzzle")


def parse_target(text: str, hidden: dict[str, object]) -> list[dict[str, object]] | None:
    axes = tuple(hidden["axes"])
    axis_values = {axis: set(values) for axis, values in hidden["axis_values"].items()}
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(raw_lines) != HOUSE_COUNT:
        return None

    rows: list[dict[str, object]] = []
    seen_houses: set[int] = set()
    seen_values = {axis: set() for axis in axes}

    for line in raw_lines:
        if ":" not in line:
            return None
        house_text, values_text = line.split(":", 1)
        if not house_text.strip().isdigit():
            return None

        house = int(house_text.strip())
        if house < 1 or house > HOUSE_COUNT or house in seen_houses:
            return None

        parts = [part.strip() for part in values_text.split("|")]
        if len(parts) != len(axes):
            return None

        row = {"house": house}
        for axis, value in zip(axes, parts, strict=True):
            if value not in axis_values[axis]:
                return None
            if value in seen_values[axis]:
                return None
            row[axis] = value
            seen_values[axis].add(value)

        seen_houses.add(house)
        rows.append(row)

    rows.sort(key=lambda row: int(row["house"]))
    return rows


def format_target(rows: list[dict[str, object]], axes: tuple[str, ...]) -> str:
    lines = []
    for row in rows:
        values = " | ".join(str(row[axis]) for axis in axes)
        lines.append(f"{row['house']}: {values}")
    return "\n".join(lines)


def is_correct(parsed: list[dict[str, object]], hidden: dict[str, object]) -> bool:
    return parsed == hidden["solution_rows"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    axes = tuple(hidden["axes"])
    axis_values = {axis: list(values) for axis, values in hidden["axis_values"].items()}
    clues = [clue_from_dict(payload) for payload in hidden["clues"]]
    solutions = solve_puzzle(axis_values, axes, clues)
    if len(solutions) != 1:
        return False
    return solutions[0] == hidden["solution_rows"]


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    clue_kind_counts: Counter[str] = Counter()
    required_minimums: Counter[str] = Counter()

    for row in rows:
        for clue in row["hidden"]["clues"]:
            clue_kind_counts[clue["kind"]] += 1

    for item in planned:
        recipe = _recipe_by_id(item["recipe_id"])
        for kind in recipe["required_kinds"]:
            required_minimums[kind] += 1

    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "prompt_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("prompt_style",), observed_getter=_meta_getter),
        "axis_profile_coverage": diversity.compare_planned_to_observed(planned, rows, ("axis_profile",), observed_getter=_meta_getter),
        "clue_profile_coverage": diversity.compare_planned_to_observed(planned, rows, ("clue_profile",), observed_getter=_meta_getter),
        "clue_kind_minimums": diversity.counter_minimum_check(clue_kind_counts, dict(required_minimums)),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def clue_from_dict(payload: dict[str, str | int | None]) -> ZebraClue:
    return ZebraClue(
        kind=str(payload["kind"]),
        left=str(payload["left"]),
        right=_optional_text(payload.get("right")),
        house=_optional_int(payload.get("house")),
    )


def solve_puzzle(
    axis_values: dict[str, list[str]] | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    clues: list[ZebraClue],
) -> list[list[dict[str, object]]]:
    states = _enumerate_states(axis_values, axes)
    valid_states = [state for state in states if _state_satisfies(state, clues, axes)]
    return [_rows_from_state(state, axes) for state in valid_states]


def _sample_axis_values(axes: tuple[str, ...], *, language: str, rng: Random) -> dict[str, tuple[str, ...]]:
    sampled: dict[str, tuple[str, ...]] = {}
    for axis in axes:
        values = rng.sample(AXIS_SPECS[axis].values[language], k=HOUSE_COUNT)
        sampled[axis] = tuple(sorted(values))
    return sampled


def _select_clues(
    solution_state: dict[str, tuple[str, ...]],
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    recipe: dict[str, Any],
    rng: Random,
) -> tuple[ZebraClue, ...] | None:
    remaining_states = _enumerate_states(axis_values, axes)
    candidates = _build_candidates(solution_state, axes, recipe["allowed_kinds"])
    rng.shuffle(candidates)

    selected: list[ZebraClue] = []

    for kind in recipe["required_kinds"]:
        choice = _best_clue_of_kind(kind, candidates, selected, remaining_states, recipe["max_kind_counts"], recipe["kind_priority"])
        if choice is None:
            return None
        selected.append(choice[0])
        remaining_states = choice[1]

    while len(remaining_states) > 1:
        choice = _best_clue(candidates, selected, remaining_states, recipe["max_kind_counts"], recipe["kind_priority"])
        if choice is None:
            return None
        selected.append(choice[0])
        remaining_states = choice[1]

    while len(selected) < recipe["target_clue_count"]:
        extra = _best_extra_clue(candidates, selected, remaining_states, recipe["max_kind_counts"], recipe["kind_priority"])
        if extra is None:
            break
        selected.append(extra)

    if len(remaining_states) != 1:
        return None
    return tuple(selected)


def _build_candidates(
    solution_state: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    allowed_kinds: tuple[str, ...],
) -> list[ZebraClue]:
    allowed = set(allowed_kinds)
    positions = _positions(solution_state, axes)
    candidates: list[ZebraClue] = []
    entities = [entity for axis in axes for entity in solution_state[axis]]

    if "same_house" in allowed or "not_same_house" in allowed:
        for left_axis_index, left_axis in enumerate(axes):
            for right_axis in axes[left_axis_index + 1 :]:
                for left in solution_state[left_axis]:
                    for right in solution_state[right_axis]:
                        if positions[left] == positions[right]:
                            if "same_house" in allowed:
                                candidates.append(ZebraClue("same_house", left, right))
                        elif "not_same_house" in allowed:
                            candidates.append(ZebraClue("not_same_house", left, right))

    if "house" in allowed:
        for axis in axes:
            for house_index, entity in enumerate(solution_state[axis], start=1):
                candidates.append(ZebraClue("house", entity, house=house_index))

    for left in entities:
        for right in entities:
            if left == right:
                continue
            if positions[left] >= positions[right]:
                continue

            if "left_of" in allowed:
                candidates.append(ZebraClue("left_of", left, right))
            if "immediately_left_of" in allowed and positions[right] - positions[left] == 1:
                candidates.append(ZebraClue("immediately_left_of", left, right))
            if "adjacent" in allowed and abs(positions[right] - positions[left]) == 1:
                candidates.append(ZebraClue("adjacent", left, right))
            if "one_between" in allowed and abs(positions[right] - positions[left]) == 2:
                candidates.append(ZebraClue("one_between", left, right))

    unique: list[ZebraClue] = []
    seen: set[tuple[str, str, str | None, int | None]] = set()
    for clue in candidates:
        key = (clue.kind, clue.left, clue.right, clue.house)
        if key in seen:
            continue
        seen.add(key)
        unique.append(clue)
    return unique


def _best_clue_of_kind(
    kind: str,
    candidates: list[ZebraClue],
    selected: list[ZebraClue],
    remaining_states: list[dict[str, tuple[str, ...]]],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> tuple[ZebraClue, list[dict[str, tuple[str, ...]]]] | None:
    best_choice: tuple[ZebraClue, list[dict[str, tuple[str, ...]]]] | None = None
    best_score: tuple[int, int] | None = None

    for clue in candidates:
        if clue.kind != kind or clue in selected:
            continue
        if _exceeds_kind_limit(selected, clue, max_kind_counts):
            continue

        narrowed = _filter_states(remaining_states, [*selected, clue])
        if len(narrowed) == len(remaining_states):
            continue

        score = (len(narrowed), kind_priority[clue.kind])
        if best_score is None or score < best_score:
            best_score = score
            best_choice = (clue, narrowed)

    return best_choice


def _best_clue(
    candidates: list[ZebraClue],
    selected: list[ZebraClue],
    remaining_states: list[dict[str, tuple[str, ...]]],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> tuple[ZebraClue, list[dict[str, tuple[str, ...]]]] | None:
    best_choice: tuple[ZebraClue, list[dict[str, tuple[str, ...]]]] | None = None
    best_score: tuple[int, int] | None = None

    for clue in candidates:
        if clue in selected:
            continue
        if _exceeds_kind_limit(selected, clue, max_kind_counts):
            continue

        narrowed = _filter_states(remaining_states, [*selected, clue])
        if len(narrowed) == len(remaining_states):
            continue

        score = (len(narrowed), kind_priority[clue.kind])
        if best_score is None or score < best_score:
            best_score = score
            best_choice = (clue, narrowed)

    return best_choice


def _best_extra_clue(
    candidates: list[ZebraClue],
    selected: list[ZebraClue],
    remaining_states: list[dict[str, tuple[str, ...]]],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> ZebraClue | None:
    best_clue: ZebraClue | None = None
    best_score: tuple[int, int] | None = None

    for clue in candidates:
        if clue in selected:
            continue
        if _exceeds_kind_limit(selected, clue, max_kind_counts):
            continue

        narrowed = _filter_states(remaining_states, [*selected, clue])
        if not narrowed:
            continue

        score = (kind_priority[clue.kind], len(narrowed))
        if best_score is None or score < best_score:
            best_score = score
            best_clue = clue

    return best_clue


def _enumerate_states(
    axis_values: dict[str, list[str]] | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
) -> list[dict[str, tuple[str, ...]]]:
    states = [dict()]
    for axis in axes:
        next_states: list[dict[str, tuple[str, ...]]] = []
        for order in permutations(axis_values[axis]):
            ordered = tuple(order)
            for state in states:
                next_states.append({**state, axis: ordered})
        states = next_states
    return states


def _rows_from_state(state: dict[str, tuple[str, ...]], axes: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for house_index in range(HOUSE_COUNT):
        row = {"house": house_index + 1}
        for axis in axes:
            row[axis] = state[axis][house_index]
        rows.append(row)
    return rows


def _state_satisfies(state: dict[str, tuple[str, ...]], clues: list[ZebraClue], axes: tuple[str, ...]) -> bool:
    positions = _positions(state, axes)
    return all(_clue_holds(positions, clue) for clue in clues)


def _filter_states(
    states: list[dict[str, tuple[str, ...]]],
    clues: list[ZebraClue],
) -> list[dict[str, tuple[str, ...]]]:
    filtered: list[dict[str, tuple[str, ...]]] = []
    for state in states:
        if _state_satisfies(state, clues, tuple(state.keys())):
            filtered.append(state)
    return filtered


def _positions(state: dict[str, tuple[str, ...]], axes: tuple[str, ...]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for axis in axes:
        for house_index, entity in enumerate(state[axis]):
            positions[entity] = house_index
    return positions


def _clue_holds(positions: dict[str, int], clue: ZebraClue) -> bool:
    left = positions[clue.left]

    if clue.kind == "house":
        assert clue.house is not None
        return left == clue.house - 1

    assert clue.right is not None
    right = positions[clue.right]

    if clue.kind == "same_house":
        return left == right
    if clue.kind == "not_same_house":
        return left != right
    if clue.kind == "left_of":
        return left < right
    if clue.kind == "immediately_left_of":
        return left + 1 == right
    if clue.kind == "adjacent":
        return abs(left - right) == 1
    if clue.kind == "one_between":
        return abs(left - right) == 2

    raise AssertionError(f"unsupported zebra clue kind: {clue.kind}")


def _format_prompt(
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    clues: tuple[ZebraClue, ...],
    *,
    language: str,
    recipe: dict[str, Any],
) -> str:
    clue_lines = "\n".join(
        f"{index}. {_format_clue(clue, axis_values, axes, language=language)}"
        for index, clue in enumerate(clues, start=1)
    )
    category_lines = "\n".join(
        f"{AXIS_SPECS[axis].labels[language]}: {', '.join(axis_values[axis])}"
        for axis in axes
    )
    header = _prompt_header(language=language, style=recipe["prompt_style"])
    answer_format = _answer_format(language=language, axes=axes)
    clue_label = "Ledetråde" if language == "da" else "Clues"

    return (
        f"{header}\n\n"
        f"{category_lines}\n\n"
        f"{_instruction_text(language=language, style=recipe['prompt_style'])}\n"
        f"{answer_format}\n\n"
        f"{clue_label}:\n{clue_lines}"
    )


def _prompt_header(*, language: str, style: str) -> str:
    if language == "da":
        headers = {
            "grid": "Dette er en zebra-logikopgave med fire huse på række, nummereret 1 til 4 fra venstre mod højre.",
            "ledger": "Fire nabo-huse står på række som hus 1 til 4. Hvert hus matcher præcis én værdi fra hver kategori.",
            "deduce": "Find hele husoversigten for fire huse på række, markeret 1 til 4 fra venstre mod højre.",
        }
        return headers[style]

    headers = {
        "grid": "This is a zebra-style logic puzzle with four houses in a row, numbered 1 to 4 from left to right.",
        "ledger": "Four neighboring houses stand in a row as houses 1 to 4. Each house matches exactly one value from each category.",
        "deduce": "Work out the full house ledger for four houses in a row, labeled 1 to 4 from left to right.",
    }
    return headers[style]


def _instruction_text(*, language: str, style: str) -> str:
    if language == "da":
        texts = {
            "grid": "Brug ledetrådene til at bestemme den fulde fordeling. Der er præcis én gyldig løsning.",
            "ledger": "Ledetrådene bestemmer én og kun én komplet fordeling af personer og egenskaber.",
            "deduce": "Udled hele oversigten. Alle kategorier skal passe sammen i præcis én løsning.",
        }
        return texts[style]

    texts = {
        "grid": "Use the clues to determine the full assignment. There is exactly one valid solution.",
        "ledger": "The clues determine one and only one complete assignment of people and properties.",
        "deduce": "Deduce the entire ledger. Every category must fit together in exactly one solution.",
    }
    return texts[style]


def _answer_format(*, language: str, axes: tuple[str, ...]) -> str:
    labels = [AXIS_SPECS[axis].singular_labels[language] for axis in axes]
    line = " | ".join(labels)

    if language == "da":
        return (
            "Svar med én linje per hus i dette format:\n"
            f"1: {line}\n"
            f"2: {line}\n"
            f"3: {line}\n"
            f"4: {line}"
        )

    return (
        "Respond with one line per house in this format:\n"
        f"1: {line}\n"
        f"2: {line}\n"
        f"3: {line}\n"
        f"4: {line}"
    )


def _format_clue(
    clue: ZebraClue,
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    *,
    language: str,
) -> str:
    left = _describe_entity(clue.left, axis_values, axes, language=language, sentence_start=True)
    left_inline = _describe_entity(clue.left, axis_values, axes, language=language, sentence_start=False)

    if clue.kind == "house":
        assert clue.house is not None
        if language == "da":
            return f"{left} er i hus {clue.house}."
        return f"{left} is in house {clue.house}."

    assert clue.right is not None
    right = _describe_entity(clue.right, axis_values, axes, language=language, sentence_start=False)

    if clue.kind == "same_house":
        if language == "da":
            return f"{left} hører til samme hus som {right}."
        return f"{left} is in the same house as {right}."

    if clue.kind == "not_same_house":
        if language == "da":
            return f"{left} er ikke i samme hus som {right}."
        return f"{left} is not in the same house as {right}."

    if clue.kind == "left_of":
        if language == "da":
            return f"{left} er et sted til venstre for {right}."
        return f"{left} is somewhere to the left of {right}."

    if clue.kind == "immediately_left_of":
        if language == "da":
            return f"{left} er umiddelbart til venstre for {right}."
        return f"{left} is immediately left of {right}."

    if clue.kind == "adjacent":
        if language == "da":
            return f"{left} bor ved siden af {right}."
        return f"{left} lives next to {right}."

    if language == "da":
        return f"Der er ét hus mellem {left_inline} og {right}."
    return f"There is one house between {left_inline} and {right}."


def _describe_entity(
    entity: str,
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    *,
    language: str,
    sentence_start: bool,
) -> str:
    axis = _entity_axis(axis_values, axes, entity)

    if axis == "name":
        return entity

    if language == "da":
        if axis == "drink":
            text = f"personen, der drikker {entity}"
        elif axis == "pet":
            text = f"personen med {entity}"
        elif axis == "color":
            text = f"personen, der foretrækker {entity}"
        elif axis == "lunch":
            text = f"personen, der spiser {entity}"
        else:
            text = f"personen, der kan lide {entity}"
    else:
        if axis == "drink":
            text = f"the {entity} drinker"
        elif axis == "pet":
            text = f"the {entity} owner"
        elif axis == "color":
            text = f"the person who likes {entity}"
        elif axis == "lunch":
            text = f"the person who eats {entity}"
        else:
            text = f"the person who likes {entity}"

    if sentence_start:
        return text[:1].upper() + text[1:]
    return text


def _entity_axis(
    axis_values: dict[str, tuple[str, ...]] | dict[str, list[str]],
    axes: tuple[str, ...],
    entity: str,
) -> str:
    for axis in axes:
        if entity in axis_values[axis]:
            return axis
    raise AssertionError(f"unknown zebra entity: {entity}")


def _recipe_by_id(recipe_id: str) -> dict[str, Any]:
    for recipe in RECIPES:
        if recipe["recipe_id"] == recipe_id:
            return recipe
    raise AssertionError(f"unknown zebra recipe: {recipe_id}")


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]


def _exceeds_kind_limit(selected: list[ZebraClue], clue: ZebraClue, max_kind_counts: dict[str, int]) -> bool:
    if clue.kind not in max_kind_counts:
        return False
    count = sum(1 for item in selected if item.kind == clue.kind)
    return count >= max_kind_counts[clue.kind]


def _optional_text(value: str | int | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    return int(value)
