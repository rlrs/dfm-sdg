from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any

import z3

from sdg.commons import diversity, prompt_surface, z3_solver


@dataclass(frozen=True)
class ZebraClue:
    kind: str
    left: str
    right: str | None = None
    third: str | None = None
    house: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "kind": self.kind,
            "left": self.left,
            "right": self.right,
            "third": self.third,
            "house": self.house,
        }


@dataclass(frozen=True)
class AxisSpec:
    key: str
    labels: dict[str, str]
    singular_labels: dict[str, str]
    values: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ZebraSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class ZebraState:
    axis_orders: dict[str, tuple[str, ...]]
    positions: dict[str, int]


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
            "da": ("kaffe", "kakao", "limonade", "mælk", "most", "saft", "te", "vand"),
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
            "da": ("frikadeller", "lasagne", "omelet", "pastasalat", "salat", "smørrebrød", "suppe", "tærte"),
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

SURFACE_SPECS = {
    "grid": ZebraSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="grid",
            block_order=("intro", "categories", "instruction", "answer", "clues"),
        ),
    ),
    "ledger": ZebraSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="ledger",
            block_order=("intro", "instruction", "categories", "answer", "clues"),
        ),
    ),
    "deduce": ZebraSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "categories", "answer", "instruction", "clues"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_grid_household",
        "difficulty": "easy",
        "prompt_style": "grid",
        "house_count": 4,
        "axis_profile": "drink_pet",
        "attribute_axes": ("drink", "pet"),
        "clue_profile": "absolute",
        "target_clue_count": 6,
        "max_clue_count": 8,
        "required_kinds": ("same_house", "house", "adjacent"),
        "allowed_kinds": ("same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "house", "not_house"),
        "max_kind_counts": {"house": 2, "same_house": 1, "not_house": 1},
        "kind_priority": {"adjacent": 0, "house": 1, "right_of": 2, "left_of": 3, "immediately_left_of": 4, "same_house": 5, "one_between": 6, "not_house": 7},
    },
    {
        "recipe_id": "easy_ledger_color_pet",
        "difficulty": "easy",
        "prompt_style": "ledger",
        "house_count": 4,
        "axis_profile": "color_pet",
        "attribute_axes": ("color", "pet"),
        "clue_profile": "absolute",
        "target_clue_count": 6,
        "max_clue_count": 8,
        "required_kinds": ("same_house", "not_house", "immediately_left_of"),
        "allowed_kinds": ("same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "house", "not_house"),
        "max_kind_counts": {"house": 1, "same_house": 1, "not_house": 2},
        "kind_priority": {"not_house": 0, "immediately_left_of": 1, "adjacent": 2, "right_of": 3, "left_of": 4, "house": 5, "same_house": 6, "one_between": 7},
    },
    {
        "recipe_id": "easy_deduce_meal_flower",
        "difficulty": "easy",
        "prompt_style": "deduce",
        "house_count": 4,
        "axis_profile": "lunch_flower",
        "attribute_axes": ("lunch", "flower"),
        "clue_profile": "spacing",
        "target_clue_count": 6,
        "max_clue_count": 8,
        "required_kinds": ("same_house", "house", "right_of"),
        "allowed_kinds": ("same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "house", "not_house"),
        "max_kind_counts": {"house": 2, "same_house": 1, "right_of": 1},
        "kind_priority": {"right_of": 0, "house": 1, "one_between": 2, "adjacent": 3, "left_of": 4, "immediately_left_of": 5, "same_house": 6, "not_house": 7},
    },
    {
        "recipe_id": "medium_grid_drink_color",
        "difficulty": "medium",
        "prompt_style": "grid",
        "house_count": 4,
        "axis_profile": "drink_color_pet",
        "attribute_axes": ("drink", "color", "pet"),
        "clue_profile": "relational",
        "target_clue_count": 8,
        "max_clue_count": 10,
        "required_kinds": ("same_house", "left_of", "adjacent", "not_house"),
        "allowed_kinds": ("same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "between", "house", "not_house"),
        "max_kind_counts": {"house": 1, "same_house": 1, "not_house": 2, "between": 1, "adjacent": 1},
        "kind_priority": {"not_house": 0, "adjacent": 1, "left_of": 2, "between": 3, "right_of": 4, "immediately_left_of": 5, "same_house": 6, "one_between": 7, "house": 8},
    },
    {
        "recipe_id": "medium_ledger_lunch_pet",
        "difficulty": "medium",
        "prompt_style": "ledger",
        "house_count": 4,
        "axis_profile": "lunch_pet_flower",
        "attribute_axes": ("lunch", "pet", "flower"),
        "clue_profile": "spacing",
        "target_clue_count": 8,
        "max_clue_count": 10,
        "required_kinds": ("same_house", "immediately_left_of", "one_between", "between"),
        "allowed_kinds": ("same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "between", "house", "not_house"),
        "max_kind_counts": {"house": 1, "same_house": 1, "not_house": 1, "between": 2, "one_between": 2},
        "kind_priority": {"between": 0, "one_between": 1, "immediately_left_of": 2, "not_house": 3, "adjacent": 4, "right_of": 5, "left_of": 6, "same_house": 7, "house": 8},
    },
    {
        "recipe_id": "medium_deduce_flower_color",
        "difficulty": "medium",
        "prompt_style": "deduce",
        "house_count": 4,
        "axis_profile": "flower_color",
        "attribute_axes": ("flower", "color"),
        "clue_profile": "mixed",
        "target_clue_count": 7,
        "max_clue_count": 9,
        "required_kinds": ("same_house", "house", "right_of", "not_house"),
        "allowed_kinds": ("same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "between", "house", "not_house"),
        "max_kind_counts": {"house": 1, "same_house": 1, "not_house": 2, "right_of": 1},
        "kind_priority": {"right_of": 0, "not_house": 1, "house": 2, "between": 3, "adjacent": 4, "left_of": 5, "one_between": 6, "immediately_left_of": 7, "same_house": 8},
    },
    {
        "recipe_id": "hard_grid_negative_household",
        "difficulty": "hard",
        "prompt_style": "grid",
        "house_count": 5,
        "axis_profile": "drink_color",
        "attribute_axes": ("drink", "color"),
        "clue_profile": "negative",
        "target_clue_count": 9,
        "max_clue_count": 12,
        "required_kinds": ("same_house", "not_same_house", "one_between", "left_of", "not_house"),
        "allowed_kinds": ("same_house", "not_same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "between", "not_house"),
        "max_kind_counts": {"same_house": 1, "not_same_house": 2, "not_house": 2, "between": 2, "right_of": 1},
        "kind_priority": {"not_same_house": 0, "not_house": 1, "between": 2, "one_between": 3, "left_of": 4, "right_of": 5, "adjacent": 6, "immediately_left_of": 7, "same_house": 8},
    },
    {
        "recipe_id": "hard_ledger_negative_meal",
        "difficulty": "hard",
        "prompt_style": "ledger",
        "house_count": 4,
        "axis_profile": "lunch_flower",
        "attribute_axes": ("lunch", "flower"),
        "clue_profile": "negative",
        "target_clue_count": 8,
        "max_clue_count": 11,
        "required_kinds": ("same_house", "not_same_house", "immediately_left_of", "right_of", "between"),
        "allowed_kinds": ("same_house", "not_same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "between", "not_house"),
        "max_kind_counts": {"same_house": 1, "not_same_house": 2, "not_house": 1, "between": 2, "right_of": 1},
        "kind_priority": {"not_same_house": 0, "between": 1, "right_of": 2, "not_house": 3, "one_between": 4, "adjacent": 5, "immediately_left_of": 6, "left_of": 7, "same_house": 8},
    },
    {
        "recipe_id": "hard_deduce_negative_color",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "house_count": 4,
        "axis_profile": "color_pet",
        "attribute_axes": ("color", "pet"),
        "clue_profile": "negative",
        "target_clue_count": 8,
        "max_clue_count": 11,
        "required_kinds": ("same_house", "not_same_house", "adjacent", "one_between", "right_of"),
        "allowed_kinds": ("same_house", "not_same_house", "left_of", "right_of", "immediately_left_of", "adjacent", "one_between", "between", "not_house"),
        "max_kind_counts": {"same_house": 1, "not_same_house": 2, "not_house": 2, "adjacent": 1, "right_of": 1},
        "kind_priority": {"not_same_house": 0, "right_of": 1, "adjacent": 2, "not_house": 3, "between": 4, "one_between": 5, "left_of": 6, "immediately_left_of": 7, "same_house": 8},
    },
)


def recipe_catalog(language: str) -> tuple[dict[str, Any], ...]:
    return RECIPES


def surface_axes(language: str) -> dict[str, tuple[str, ...]]:
    return prompt_surface.default_surface_axes()


def generate_row(
    index: int,
    rng: Random,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object] | None = None,
) -> dict[str, object]:
    axes = ("name", *recipe["attribute_axes"])
    house_count = int(recipe["house_count"])
    chosen_surface = dict(surface_plan or prompt_surface.sample_surface_plan(rng, surface_axes(language)))

    for _ in range(200):
        axis_values = _sample_axis_values(axes, house_count=house_count, language=language, rng=rng)
        solution_state = {
            axis: tuple(rng.sample(axis_values[axis], k=house_count))
            for axis in axes
        }
        clues = _select_clues(solution_state, axis_values, axes, house_count, recipe, rng)
        if clues is None:
            continue

        solution_rows = _rows_from_state(solution_state, axes, house_count)
        return {
            "id": f"verifiable-reasoning-{index:05d}",
            "prompt": _format_prompt(
                axis_values,
                axes,
                clues,
                house_count=house_count,
                language=language,
                recipe=recipe,
                surface_plan=chosen_surface,
            ),
            "hidden": {
                "axes": list(axes),
                "house_count": house_count,
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
                "house_count": house_count,
                "output_format": "house_table",
                "recipe_id": recipe["recipe_id"],
                "difficulty": recipe["difficulty"],
                "prompt_style": recipe["prompt_style"],
                "axis_profile": recipe["axis_profile"],
                "axis_count": len(axes),
                "clue_profile": recipe["clue_profile"],
                **chosen_surface,
            },
        }

    raise AssertionError("failed to generate zebra puzzle")


def parse_target(text: str, hidden: dict[str, object]) -> list[dict[str, object]] | None:
    axes = tuple(hidden["axes"])
    house_count = int(hidden["house_count"])
    axis_values = {axis: set(values) for axis, values in hidden["axis_values"].items()}
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(raw_lines) != house_count:
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
        if house < 1 or house > house_count or house in seen_houses:
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


def canonical_target(parsed: list[dict[str, object]], hidden: dict[str, object]) -> str:
    return format_target(parsed, tuple(hidden["axes"]))


def answer_contract(hidden: dict[str, object], language: str) -> str:
    axes = tuple(hidden["axes"])
    house_count = int(hidden["house_count"])
    labels = " | ".join(AXIS_SPECS[axis].singular_labels[language] for axis in axes)
    lines = [f"1: <{labels}>"]
    if house_count >= 2:
        lines.append(f"2: <{labels}>")
    if house_count > 2:
        lines.append("...")
    if house_count >= 3:
        lines.append(f"{house_count}: <{labels}>")

    if language == "da":
        return (
            "I din svarblok skal den endelige løsning være en husoversigt med én linje pr. hus.\n"
            "Brug dette format:\n"
            + "\n".join(lines)
        )

    return (
        "In your answer block, the final solution should be a house ledger with one line per house.\n"
        "Use this format:\n"
        + "\n".join(lines)
    )


def is_correct(parsed: list[dict[str, object]], hidden: dict[str, object]) -> bool:
    return parsed == hidden["solution_rows"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    axes = tuple(hidden["axes"])
    house_count = int(hidden["house_count"])
    axis_values = {axis: list(values) for axis, values in hidden["axis_values"].items()}
    clues = [clue_from_dict(payload) for payload in hidden["clues"]]
    solutions = _solve_states(axis_values, axes, house_count, clues, limit=2)
    if len(solutions) != 1:
        return False
    return _rows_from_state(solutions[0], axes, house_count) == hidden["solution_rows"]


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
        "surface_intro_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_intro",), observed_getter=_meta_getter),
        "surface_instruction_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_instruction",), observed_getter=_meta_getter),
        "surface_answer_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_answer",), observed_getter=_meta_getter),
        "surface_clue_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_clue",), observed_getter=_meta_getter),
        "surface_combo_coverage": diversity.compare_planned_to_observed(
            planned,
            rows,
            ("surface_intro", "surface_instruction", "surface_answer", "surface_clue"),
            observed_getter=_meta_getter,
        ),
        "axis_profile_coverage": diversity.compare_planned_to_observed(planned, rows, ("axis_profile",), observed_getter=_meta_getter),
        "axis_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("axis_count",), planned_getter=_planned_axis_count_getter, observed_getter=_meta_getter),
        "clue_profile_coverage": diversity.compare_planned_to_observed(planned, rows, ("clue_profile",), observed_getter=_meta_getter),
        "clue_kind_minimums": diversity.counter_minimum_check(clue_kind_counts, dict(required_minimums)),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def clue_from_dict(payload: dict[str, str | int | None]) -> ZebraClue:
    return ZebraClue(
        kind=str(payload["kind"]),
        left=str(payload["left"]),
        right=_optional_text(payload.get("right")),
        third=_optional_text(payload.get("third")),
        house=_optional_int(payload.get("house")),
    )


def solve_puzzle(
    axis_values: dict[str, list[str]] | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    clues: list[ZebraClue],
) -> list[list[dict[str, object]]]:
    house_count = len(next(iter(axis_values.values())))
    valid_states = _solve_states(axis_values, axes, house_count, clues)
    return [_rows_from_state(state, axes, house_count) for state in valid_states]


def _sample_axis_values(
    axes: tuple[str, ...],
    *,
    house_count: int,
    language: str,
    rng: Random,
) -> dict[str, tuple[str, ...]]:
    sampled: dict[str, tuple[str, ...]] = {}
    for axis in axes:
        values = rng.sample(AXIS_SPECS[axis].values[language], k=house_count)
        sampled[axis] = tuple(sorted(values))
    return sampled


def _select_clues(
    solution_state: dict[str, tuple[str, ...]],
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    house_count: int,
    recipe: dict[str, Any],
    rng: Random,
) -> tuple[ZebraClue, ...] | None:
    candidates = _build_candidates(solution_state, axes, recipe["allowed_kinds"])
    ordered_candidates = _ordered_candidates(candidates, recipe["kind_priority"], rng)
    max_clue_count = int(recipe["max_clue_count"])

    selected: list[ZebraClue] = []

    for kind in recipe["required_kinds"]:
        choice = _pick_narrowing_clue_of_kind(
            kind,
            axis_values,
            axes,
            house_count,
            ordered_candidates,
            selected,
            recipe["max_kind_counts"],
            recipe["kind_priority"],
        )
        if choice is None:
            return None
        clue, _, _ = choice
        selected.append(clue)

    is_unique = _has_unique_solution(axis_values, axes, house_count, selected)
    while not is_unique:
        if len(selected) >= max_clue_count:
            return None
        choice = _pick_narrowing_clue(
            axis_values,
            axes,
            house_count,
            ordered_candidates,
            selected,
            recipe["max_kind_counts"],
            recipe["kind_priority"],
        )
        if choice is None:
            return None
        clue, count, complete = choice
        selected.append(clue)
        is_unique = count == 1 and complete

    while len(selected) < recipe["target_clue_count"]:
        extra = _pick_extra_clue(
            ordered_candidates,
            selected,
            recipe["max_kind_counts"],
            recipe["kind_priority"],
        )
        if extra is None:
            break
        selected.append(extra)

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

    if "house" in allowed or "not_house" in allowed:
        house_count = len(solution_state[axes[0]])
        for axis in axes:
            for house_index, entity in enumerate(solution_state[axis], start=1):
                if "house" in allowed:
                    candidates.append(ZebraClue("house", entity, house=house_index))
                if "not_house" in allowed:
                    for other_house in range(1, house_count + 1):
                        if other_house == house_index:
                            continue
                        candidates.append(ZebraClue("not_house", entity, house=other_house))

    for left in entities:
        for right in entities:
            if left == right:
                continue
            if positions[left] < positions[right] and "left_of" in allowed:
                candidates.append(ZebraClue("left_of", left, right))
            if positions[left] > positions[right] and "right_of" in allowed:
                candidates.append(ZebraClue("right_of", left, right))
            if "immediately_left_of" in allowed and positions[right] - positions[left] == 1:
                candidates.append(ZebraClue("immediately_left_of", left, right))
            if "adjacent" in allowed and abs(positions[right] - positions[left]) == 1:
                candidates.append(ZebraClue("adjacent", left, right))
            if "one_between" in allowed and abs(positions[right] - positions[left]) == 2:
                candidates.append(ZebraClue("one_between", left, right))

    if "between" in allowed:
        for middle in entities:
            middle_position = positions[middle]
            for left in entities:
                if left == middle:
                    continue
                left_position = positions[left]
                for right in entities:
                    if right in {middle, left}:
                        continue
                    right_position = positions[right]
                    if left_position < middle_position < right_position:
                        candidates.append(ZebraClue("between", middle, left, third=right))

    unique: list[ZebraClue] = []
    seen: set[tuple[str, str, str | None, str | None, int | None]] = set()
    for clue in candidates:
        key = (clue.kind, clue.left, clue.right, clue.third, clue.house)
        if key in seen:
            continue
        seen.add(key)
        unique.append(clue)
    return unique


def _pick_narrowing_clue_of_kind(
    kind: str,
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    house_count: int,
    candidates: list[ZebraClue],
    selected: list[ZebraClue],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> tuple[ZebraClue, int, bool] | None:
    base_solver, variables, entity_vars = _build_solver(axis_values, axes, house_count, selected)
    best_choice: tuple[ZebraClue, int, bool] | None = None
    best_score: tuple[int, bool, int, int, int, int] | None = None

    for clue in _candidate_window(
        candidates,
        selected,
        max_kind_counts,
        kind=kind,
        per_kind_limit=16,
        total_limit=16,
    ):
        constraint = _clue_constraint(entity_vars, clue)
        if not z3_solver.has_model_with(base_solver, z3.Not(constraint)):
            continue
        count, complete = z3_solver.count_int_models_in_place(
            base_solver,
            variables,
            extra_constraints=(constraint,),
            limit=8,
        )
        score = _narrowing_score(clue, count, complete, selected, kind_priority)
        if best_score is None or score < best_score:
            best_score = score
            best_choice = (clue, count, complete)

    return best_choice


def _pick_narrowing_clue(
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    house_count: int,
    candidates: list[ZebraClue],
    selected: list[ZebraClue],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> tuple[ZebraClue, int, bool] | None:
    base_solver, variables, entity_vars = _build_solver(axis_values, axes, house_count, selected)
    best_choice: tuple[ZebraClue, int, bool] | None = None
    best_score: tuple[int, bool, int, int, int, int] | None = None

    for clue in _candidate_window(candidates, selected, max_kind_counts):
        constraint = _clue_constraint(entity_vars, clue)
        if not z3_solver.has_model_with(base_solver, z3.Not(constraint)):
            continue
        count, complete = z3_solver.count_int_models_in_place(
            base_solver,
            variables,
            extra_constraints=(constraint,),
            limit=8,
        )
        score = _narrowing_score(clue, count, complete, selected, kind_priority)
        if best_score is None or score < best_score:
            best_score = score
            best_choice = (clue, count, complete)

    return best_choice


def _pick_extra_clue(
    candidates: list[ZebraClue],
    selected: list[ZebraClue],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> ZebraClue | None:
    kind_counts = Counter(clue.kind for clue in selected)
    group_counts = Counter(_clue_group(clue.kind) for clue in selected)
    best_clue: ZebraClue | None = None
    best_score: tuple[int, int, int, int] | None = None

    for clue in _candidate_window(candidates, selected, max_kind_counts):
        overlap = _entity_overlap_count(clue, selected)
        score = (
            kind_counts[clue.kind],
            group_counts[_clue_group(clue.kind)],
            overlap,
            kind_priority[clue.kind],
        )

        if best_score is None or score < best_score:
            best_score = score
            best_clue = clue

    return best_clue


def _candidate_window(
    candidates: list[ZebraClue],
    selected: list[ZebraClue],
    max_kind_counts: dict[str, int],
    *,
    kind: str | None = None,
    per_kind_limit: int = 3,
    total_limit: int = 18,
) -> list[ZebraClue]:
    window: list[ZebraClue] = []
    kind_counts: Counter[str] = Counter()

    for clue in candidates:
        if kind is not None and clue.kind != kind:
            continue
        if clue in selected:
            continue
        if _exceeds_kind_limit(selected, clue, max_kind_counts):
            continue
        if kind_counts[clue.kind] >= per_kind_limit:
            continue
        window.append(clue)
        kind_counts[clue.kind] += 1
        if len(window) >= total_limit:
            break

    return window


def _narrowing_score(
    clue: ZebraClue,
    count: int,
    complete: bool,
    selected: list[ZebraClue],
    kind_priority: dict[str, int],
) -> tuple[int, bool, int, int, int, int]:
    kind_count = sum(item.kind == clue.kind for item in selected)
    group_count = sum(_clue_group(item.kind) == _clue_group(clue.kind) for item in selected)
    entity_overlap = _entity_overlap_count(clue, selected)
    return (
        count,
        not complete,
        kind_count,
        group_count,
        entity_overlap,
        kind_priority[clue.kind],
    )


def _entity_overlap_count(clue: ZebraClue, selected: list[ZebraClue]) -> int:
    clue_entities = set(_clue_entities(clue))
    return sum(bool(clue_entities & set(_clue_entities(item))) for item in selected)


def _clue_entities(clue: ZebraClue) -> tuple[str, ...]:
    entities = [clue.left]
    if clue.right is not None:
        entities.append(clue.right)
    if clue.third is not None:
        entities.append(clue.third)
    return tuple(entities)


def _clue_group(kind: str) -> str:
    if kind in {"house", "not_house"}:
        return "house"
    if kind in {"same_house", "not_same_house"}:
        return "cohouse"
    if kind in {"left_of", "right_of", "immediately_left_of"}:
        return "direction"
    return "distance"


def _solve_states(
    axis_values: dict[str, list[str]] | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    house_count: int,
    clues: list[ZebraClue],
    *,
    limit: int | None = None,
) -> list[ZebraState]:
    solver, variables, _ = _build_solver(axis_values, axes, house_count, clues)
    assignments = z3_solver.enumerate_int_models(solver, variables, limit=limit)
    return [
        _state_from_assignment(axis_values, axes, assignment)
        for assignment in assignments
    ]


def _solve_state_count(
    axis_values: dict[str, list[str]] | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    house_count: int,
    clues: list[ZebraClue],
    *,
    limit: int | None = None,
) -> tuple[int, bool]:
    solver, variables, _ = _build_solver(axis_values, axes, house_count, clues)
    return z3_solver.count_int_models(solver, variables, limit=limit)


def _has_unique_solution(
    axis_values: dict[str, list[str]] | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    house_count: int,
    clues: list[ZebraClue],
) -> bool:
    count, complete = _solve_state_count(axis_values, axes, house_count, clues, limit=2)
    return count == 1 and complete


def _ordered_candidates(
    candidates: list[ZebraClue],
    kind_priority: dict[str, int],
    rng: Random,
) -> list[ZebraClue]:
    ordered = list(candidates)
    rng.shuffle(ordered)
    ordered.sort(key=lambda clue: kind_priority[clue.kind])
    return ordered


def _build_solver(
    axis_values: dict[str, list[str]] | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    house_count: int,
    clues: list[ZebraClue],
) -> tuple[z3.Solver, dict[str, z3.ArithRef], dict[str, z3.ArithRef]]:
    solver = z3.Solver()
    variables: dict[str, z3.ArithRef] = {}
    entity_vars: dict[str, z3.ArithRef] = {}

    for axis in axes:
        axis_variables: list[z3.ArithRef] = []
        for entity in axis_values[axis]:
            key = _var_key(axis, entity)
            variable = z3.Int(key)
            variables[key] = variable
            assert entity not in entity_vars, f"duplicate zebra entity across axes: {entity}"
            entity_vars[entity] = variable
            axis_variables.append(variable)
            solver.add(variable >= 0, variable < house_count)
        solver.add(z3.Distinct(axis_variables))

    for clue in clues:
        solver.add(_clue_constraint(entity_vars, clue))

    return solver, variables, entity_vars


def _state_from_assignment(
    axis_values: dict[str, list[str]] | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    assignment: dict[str, int],
) -> ZebraState:
    axis_orders: dict[str, tuple[str, ...]] = {}
    positions: dict[str, int] = {}

    for axis in axes:
        ordered = tuple(
            sorted(
                axis_values[axis],
                key=lambda entity: assignment[_var_key(axis, entity)],
            )
        )
        axis_orders[axis] = ordered
        for house_index, entity in enumerate(ordered):
            positions[entity] = house_index

    return ZebraState(axis_orders=axis_orders, positions=positions)


def _rows_from_state(
    state: ZebraState | dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    house_count: int,
) -> list[dict[str, object]]:
    axis_orders = state.axis_orders if isinstance(state, ZebraState) else state
    rows: list[dict[str, object]] = []
    for house_index in range(house_count):
        row = {"house": house_index + 1}
        for axis in axes:
            row[axis] = axis_orders[axis][house_index]
        rows.append(row)
    return rows


def _positions(state: dict[str, tuple[str, ...]], axes: tuple[str, ...]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for axis in axes:
        for house_index, entity in enumerate(state[axis]):
            positions[entity] = house_index
    return positions


def _clue_constraint(entity_vars: dict[str, z3.ArithRef], clue: ZebraClue) -> z3.BoolRef:
    left = entity_vars[clue.left]

    if clue.kind == "house":
        assert clue.house is not None
        return left == clue.house - 1
    if clue.kind == "not_house":
        assert clue.house is not None
        return left != clue.house - 1

    assert clue.right is not None
    right = entity_vars[clue.right]

    if clue.kind == "same_house":
        return left == right
    if clue.kind == "not_same_house":
        return left != right
    if clue.kind == "left_of":
        return left < right
    if clue.kind == "right_of":
        return left > right
    if clue.kind == "immediately_left_of":
        return left + 1 == right
    if clue.kind == "adjacent":
        return z3.Abs(left - right) == 1
    if clue.kind == "one_between":
        return z3.Abs(left - right) == 2
    if clue.kind == "between":
        assert clue.third is not None
        third = entity_vars[clue.third]
        return z3.Or(
            z3.And(right < left, left < third),
            z3.And(third < left, left < right),
        )

    raise AssertionError(f"unsupported zebra clue kind: {clue.kind}")


def _var_key(axis: str, entity: str) -> str:
    return f"{axis}_{entity}"


def _format_prompt(
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    clues: tuple[ZebraClue, ...],
    *,
    house_count: int,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    surface_spec = SURFACE_SPECS[recipe["prompt_style"]]
    clue_style = str(surface_plan["surface_clue"])
    clue_lines = tuple(
        _format_clue(
            clue,
            axis_values,
            axes,
            language=language,
            clue_style=clue_style,
        )
        for clue in clues
    )
    category_lines = tuple(
        f"{AXIS_SPECS[axis].labels[language]}: {', '.join(axis_values[axis])}"
        for axis in axes
    )
    clue_label = "Ledetråde" if language == "da" else "Clues"
    blocks = {
        "intro": prompt_surface.PromptBlock(
            "intro",
            (
                _intro_line(
                    language=language,
                    house_count=house_count,
                    intro_variant=str(surface_plan["surface_intro"]),
                ),
            ),
        ),
        "categories": prompt_surface.PromptBlock("categories", category_lines),
        "instruction": prompt_surface.PromptBlock(
            "instruction",
            (_instruction_line(language=language, instruction_variant=str(surface_plan["surface_instruction"])),),
        ),
        "answer": prompt_surface.PromptBlock(
            "answer",
            _answer_lines(
                language=language,
                axes=axes,
                house_count=house_count,
                answer_variant=str(surface_plan["surface_answer"]),
            ),
        ),
        "clues": prompt_surface.PromptBlock("clues", clue_lines, heading=clue_label, numbered=True),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_line(*, language: str, house_count: int, intro_variant: str) -> str:
    if language == "da":
        if intro_variant == "context":
            return (
                f"Dette er en zebra-logikopgave med {house_count} huse på række, "
                f"nummereret 1 til {house_count} fra venstre mod højre."
            )
        if intro_variant == "assignment":
            return (
                f"{house_count} nabohuse står på række som hus 1 til {house_count}. "
                "Hvert hus matcher præcis én værdi fra hver kategori."
            )
        return (
            f"Find hele husoversigten for {house_count} huse på række, "
            f"markeret 1 til {house_count} fra venstre mod højre."
        )

    if intro_variant == "context":
        return (
            f"This is a zebra-style logic puzzle with {house_count} houses in a row, "
            f"numbered 1 to {house_count} from left to right."
        )
    if intro_variant == "assignment":
        return (
            f"{house_count} neighboring houses stand in a row as houses 1 to {house_count}. "
            "Each house matches exactly one value from each category."
        )
    return (
        f"Work out the full house ledger for {house_count} houses in a row, "
        f"labeled 1 to {house_count} from left to right."
    )


def _instruction_line(*, language: str, instruction_variant: str) -> str:
    if language == "da":
        if instruction_variant == "solve":
            return "Brug ledetrådene til at bestemme den fulde fordeling. Der er præcis én gyldig løsning."
        if instruction_variant == "unique":
            return "Ledetrådene bestemmer én og kun én komplet fordeling af personer og egenskaber."
        return "Udled hele oversigten. Alle kategorier skal passe sammen i præcis én løsning."

    if instruction_variant == "solve":
        return "Use the clues to determine the full assignment. There is exactly one valid solution."
    if instruction_variant == "unique":
        return "The clues determine one and only one complete assignment of people and properties."
    return "Deduce the entire ledger. Every category must fit together in exactly one solution."


def _answer_lines(
    *,
    language: str,
    axes: tuple[str, ...],
    house_count: int,
    answer_variant: str,
) -> tuple[str, ...]:
    labels = [AXIS_SPECS[axis].singular_labels[language] for axis in axes]
    line = " | ".join(labels)
    numbered_lines = tuple(
        f"{house}: {line}"
        for house in range(1, house_count + 1)
    )

    if language == "da":
        first_line = {
            "respond": f"Inde i det endelige svar skal husoversigten have præcis {house_count} linjer i dette format:",
            "write": f"Inde i det endelige svar skal du bruge præcis {house_count} linjer i dette format:",
            "complete": f"Selve svarindholdet skal være en husoversigt med præcis {house_count} linjer i dette format:",
        }[answer_variant]
        return (first_line, *numbered_lines)

    first_line = {
        "respond": f"Inside the final response, the house ledger should use exactly {house_count} lines in this format:",
        "write": f"Inside the final response, use exactly {house_count} lines in this format:",
        "complete": f"The answer content should be a house ledger with exactly {house_count} lines in this format:",
    }[answer_variant]
    return (first_line, *numbered_lines)


def _format_clue(
    clue: ZebraClue,
    axis_values: dict[str, tuple[str, ...]],
    axes: tuple[str, ...],
    *,
    language: str,
    clue_style: str,
) -> str:
    left = _describe_entity(clue.left, axis_values, axes, language=language, sentence_start=True)
    left_inline = _describe_entity(clue.left, axis_values, axes, language=language, sentence_start=False)

    if clue.kind == "house":
        assert clue.house is not None
        if language == "da":
            if clue_style == "compact":
                return f"I hus {clue.house} bor {left_inline}."
            return f"{left} bor i hus {clue.house}."
        if clue_style == "compact":
            return f"House {clue.house} is the house with {left_inline}."
        return f"{left} is in house {clue.house}."
    if clue.kind == "not_house":
        assert clue.house is not None
        if language == "da":
            if clue_style == "deductive":
                return f"{left} kan ikke bo i hus {clue.house}."
            return f"{left} bor ikke i hus {clue.house}."
        if clue_style == "deductive":
            return f"The house with {left_inline} is not house {clue.house}."
        return f"{left} is not in house {clue.house}."

    assert clue.right is not None
    right = _describe_entity(clue.right, axis_values, axes, language=language, sentence_start=False)

    if clue.kind == "same_house":
        if language == "da":
            if clue_style == "compact":
                return f"{left} og {right} bor i samme hus."
            return f"{left} bor i samme hus som {right}."
        if clue_style == "compact":
            return f"{left} and {right} belong to the same house."
        return f"{left} is in the same house as {right}."

    if clue.kind == "not_same_house":
        if language == "da":
            if clue_style == "deductive":
                return f"{left} og {right} bor i forskellige huse."
            return f"{left} bor ikke i samme hus som {right}."
        if clue_style == "deductive":
            return f"{left} and {right} are in different houses."
        return f"{left} is not in the same house as {right}."

    if clue.kind == "left_of":
        if language == "da":
            if clue_style == "compact":
                return f"{left} bor i et hus til venstre for {right}."
            return f"{left} bor et sted til venstre for {right}."
        if clue_style == "compact":
            return f"{left} is in a house to the left of {right}."
        return f"{left} is somewhere to the left of {right}."
    if clue.kind == "right_of":
        if language == "da":
            if clue_style == "compact":
                return f"{left} bor i et hus til højre for {right}."
            return f"{left} bor et sted til højre for {right}."
        if clue_style == "compact":
            return f"{left} is in a house to the right of {right}."
        return f"{left} is somewhere to the right of {right}."

    if clue.kind == "immediately_left_of":
        if language == "da":
            if clue_style == "deductive":
                return f"{left} bor direkte til venstre for {right}."
            return f"{left} bor umiddelbart til venstre for {right}."
        if clue_style == "deductive":
            return f"{left} is directly to the left of {right}."
        return f"{left} is immediately left of {right}."

    if clue.kind == "adjacent":
        if language == "da":
            if clue_style == "compact":
                return f"{left} er nabo til {right}."
            return f"{left} bor ved siden af {right}."
        if clue_style == "compact":
            return f"{left} is in a neighboring house to {right}."
        return f"{left} lives next to {right}."
    if clue.kind == "between":
        assert clue.third is not None
        third = _describe_entity(clue.third, axis_values, axes, language=language, sentence_start=False)
        if language == "da":
            if clue_style == "deductive":
                return f"Huset med {left_inline} ligger mellem {right} og {third}."
            return f"{left} bor et sted mellem {right} og {third}."
        if clue_style == "deductive":
            return f"The house with {left_inline} lies between {right} and {third}."
        return f"{left} is somewhere between {right} and {third}."

    if language == "da":
        if clue_style == "compact":
            return f"{left} er to huse væk fra {right}."
        return f"Der er ét hus mellem {left_inline} og {right}."
    if clue_style == "compact":
        return f"{left} is two houses away from {right}."
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


def _planned_axis_count_getter(item: dict[str, object], key: str) -> object:
    assert key == "axis_count"
    return len(item["attribute_axes"]) + 1


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
