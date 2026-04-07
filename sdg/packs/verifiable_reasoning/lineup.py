from collections import Counter
from random import Random
from typing import Any

from sdg.commons import diversity
from sdg.packs.verifiable_reasoning import ordering
from sdg.packs.verifiable_reasoning.ordering import Clue, OrderPuzzle

NAME_POOLS = {
    "en": ("Alice", "Ben", "Chloe", "Daniel", "Emma", "Grace", "Henry", "Lucy", "Noah", "Zoe"),
    "da": ("Anna", "Bo", "Clara", "Emil", "Freja", "Ida", "Karl", "Maja", "Niels", "Sofie"),
}

RECIPES = (
    {
        "recipe_id": "easy_chairs",
        "difficulty": "easy",
        "scenario": "chairs",
        "participant_count": 4,
        "target_clue_count": 4,
        "required_kinds": ("slot", "immediately_before"),
        "allowed_kinds": ("slot", "before", "immediately_before", "adjacent"),
        "max_kind_counts": {"slot": 2, "adjacent": 1},
        "kind_priority": {"immediately_before": 0, "adjacent": 1, "before": 2, "slot": 3},
    },
    {
        "recipe_id": "easy_queue",
        "difficulty": "easy",
        "scenario": "queue",
        "participant_count": 4,
        "target_clue_count": 4,
        "required_kinds": ("slot", "adjacent"),
        "allowed_kinds": ("slot", "before", "immediately_before", "adjacent"),
        "max_kind_counts": {"slot": 2, "adjacent": 1},
        "kind_priority": {"adjacent": 0, "immediately_before": 1, "before": 2, "slot": 3},
    },
    {
        "recipe_id": "easy_ranking",
        "difficulty": "easy",
        "scenario": "ranking",
        "participant_count": 4,
        "target_clue_count": 4,
        "required_kinds": ("slot", "before"),
        "allowed_kinds": ("slot", "before", "immediately_before", "one_between"),
        "max_kind_counts": {"slot": 2},
        "kind_priority": {"immediately_before": 0, "one_between": 1, "before": 2, "slot": 3},
    },
    {
        "recipe_id": "medium_chairs",
        "difficulty": "medium",
        "scenario": "chairs",
        "participant_count": 5,
        "target_clue_count": 5,
        "required_kinds": ("immediately_before", "adjacent", "before"),
        "allowed_kinds": ("slot", "before", "immediately_before", "adjacent", "one_between"),
        "max_kind_counts": {"slot": 1, "adjacent": 2},
        "kind_priority": {"immediately_before": 0, "one_between": 1, "adjacent": 2, "before": 3, "slot": 4},
    },
    {
        "recipe_id": "medium_queue",
        "difficulty": "medium",
        "scenario": "queue",
        "participant_count": 5,
        "target_clue_count": 5,
        "required_kinds": ("one_between", "before", "not_adjacent"),
        "allowed_kinds": ("slot", "before", "immediately_before", "adjacent", "not_adjacent", "one_between"),
        "max_kind_counts": {"slot": 1},
        "kind_priority": {"one_between": 0, "not_adjacent": 1, "immediately_before": 2, "before": 3, "adjacent": 4, "slot": 5},
    },
    {
        "recipe_id": "medium_ranking",
        "difficulty": "medium",
        "scenario": "ranking",
        "participant_count": 5,
        "target_clue_count": 5,
        "required_kinds": ("between", "before", "slot"),
        "allowed_kinds": ("slot", "before", "immediately_before", "one_between", "between"),
        "max_kind_counts": {"slot": 1},
        "kind_priority": {"between": 0, "one_between": 1, "immediately_before": 2, "before": 3, "slot": 4},
    },
    {
        "recipe_id": "hard_chairs",
        "difficulty": "hard",
        "scenario": "chairs",
        "participant_count": 5,
        "target_clue_count": 6,
        "required_kinds": ("between", "not_adjacent", "one_between"),
        "allowed_kinds": ("before", "immediately_before", "adjacent", "not_adjacent", "one_between", "between"),
        "max_kind_counts": {"adjacent": 1},
        "kind_priority": {"between": 0, "one_between": 1, "not_adjacent": 2, "immediately_before": 3, "adjacent": 4, "before": 5},
    },
    {
        "recipe_id": "hard_queue",
        "difficulty": "hard",
        "scenario": "queue",
        "participant_count": 5,
        "target_clue_count": 6,
        "required_kinds": ("between", "not_adjacent", "before"),
        "allowed_kinds": ("before", "immediately_before", "adjacent", "not_adjacent", "one_between", "between"),
        "max_kind_counts": {"adjacent": 1},
        "kind_priority": {"between": 0, "not_adjacent": 1, "one_between": 2, "immediately_before": 3, "before": 4, "adjacent": 5},
    },
    {
        "recipe_id": "hard_ranking",
        "difficulty": "hard",
        "scenario": "ranking",
        "participant_count": 5,
        "target_clue_count": 6,
        "required_kinds": ("between", "adjacent", "not_adjacent"),
        "allowed_kinds": ("before", "immediately_before", "adjacent", "not_adjacent", "one_between", "between"),
        "max_kind_counts": {"adjacent": 1},
        "kind_priority": {"between": 0, "not_adjacent": 1, "one_between": 2, "immediately_before": 3, "adjacent": 4, "before": 5},
    },
)


def recipe_catalog(language: str) -> tuple[dict[str, Any], ...]:
    return RECIPES


def generate_row(index: int, rng: Random, *, language: str, recipe: dict[str, Any]) -> dict[str, object]:
    puzzle = _generate_puzzle(rng, language=language, recipe=recipe)
    return {
        "id": f"verifiable-reasoning-{index:05d}",
        "prompt": puzzle.prompt,
        "target": ordering.format_target(puzzle.solution),
        "hidden": {
            "participants": list(puzzle.participants),
            "solution": list(puzzle.solution),
            "clues": [clue.to_dict() for clue in puzzle.clues],
        },
        "sources": [{"kind": "dolci_subset", "value": puzzle.source_subset}],
        "meta": {
            "family": puzzle.family,
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": len(puzzle.clues),
            "output_format": "ordered_names",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "scenario": recipe["scenario"],
            "participant_count": recipe["participant_count"],
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[str, ...] | None:
    return ordering.parse_target(text, hidden["participants"])


def is_correct(parsed: tuple[str, ...], hidden: dict[str, object]) -> bool:
    return list(parsed) == hidden["solution"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    participants = tuple(hidden["participants"])
    clues = [ordering.clue_from_dict(item) for item in hidden["clues"]]
    solutions = ordering.solve_puzzle(participants, clues)
    if len(solutions) != 1:
        return False
    return list(solutions[0]) == hidden["solution"]


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    clue_kind_counts: Counter[str] = Counter()
    for row in rows:
        for clue in row["hidden"]["clues"]:
            clue_kind_counts[clue["kind"]] += 1

    required_minimums: Counter[str] = Counter()
    for item in planned:
        recipe = _recipe_by_id(item["recipe_id"])
        for kind in recipe["required_kinds"]:
            required_minimums[kind] += 1

    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "scenario_coverage": diversity.compare_planned_to_observed(planned, rows, ("scenario",), observed_getter=_meta_getter),
        "participant_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("participant_count",), observed_getter=_meta_getter),
        "clue_kind_minimums": diversity.counter_minimum_check(clue_kind_counts, dict(required_minimums)),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def _generate_puzzle(rng: Random, *, language: str, recipe: dict[str, Any]) -> OrderPuzzle:
    participants = tuple(sorted(rng.sample(NAME_POOLS[language], k=recipe["participant_count"])))

    for _ in range(200):
        solution = tuple(rng.sample(participants, k=len(participants)))
        clues = ordering.select_clues(solution, rng, recipe)
        prompt = _format_prompt(participants, clues, language=language, recipe=recipe)
        return OrderPuzzle(
            family="lineup_logic",
            language=language,
            participants=participants,
            solution=solution,
            clues=clues,
            prompt=prompt,
            source_subset="zebralogics",
        )

    raise AssertionError("failed to generate lineup puzzle")


def _format_prompt(
    participants: tuple[str, ...],
    clues: tuple[Clue, ...],
    *,
    language: str,
    recipe: dict[str, Any],
) -> str:
    names = ", ".join(participants)
    clue_lines = "\n".join(
        f"{index}. {_format_clue(clue, language=language, scenario=recipe['scenario'], participant_count=recipe['participant_count'])}"
        for index, clue in enumerate(clues, start=1)
    )

    if language == "da":
        if recipe["scenario"] == "chairs":
            return (
                f"{recipe['participant_count']} personer skal placeres på en række stole, nummereret 1 til {recipe['participant_count']} fra venstre mod højre: {names}.\n\n"
                "Brug ledetrådene til at bestemme hele rækkefølgen. Der er præcis en gyldig løsning.\n"
                "Svar kun med navnene i rækkefølge, adskilt af kommaer.\n\n"
                f"Ledetråde:\n{clue_lines}"
            )
        if recipe["scenario"] == "queue":
            return (
                f"{recipe['participant_count']} personer står i en kø. Positionerne går fra 1 foran til {recipe['participant_count']} bagerst: {names}.\n\n"
                "Brug ledetrådene til at bestemme hele rækkefølgen. Der er præcis en gyldig løsning.\n"
                "Svar kun med navnene fra forrest til bagerst, adskilt af kommaer.\n\n"
                f"Ledetråde:\n{clue_lines}"
            )
        return (
            f"{recipe['participant_count']} deltagere ender i en rangliste med pladser fra 1 til {recipe['participant_count']}: {names}.\n\n"
            "Brug ledetrådene til at bestemme hele rangordenen. Der er præcis en gyldig løsning.\n"
            "Svar kun med navnene fra bedste til dårligste placering, adskilt af kommaer.\n\n"
            f"Ledetråde:\n{clue_lines}"
        )

    if recipe["scenario"] == "chairs":
        return (
            f"{recipe['participant_count']} people are arranged in a row of chairs, numbered 1 to {recipe['participant_count']} from left to right: {names}.\n\n"
            "Use the clues to determine the full order. There is exactly one valid arrangement.\n"
            "Respond with only the names in order, separated by commas.\n\n"
            f"Clues:\n{clue_lines}"
        )
    if recipe["scenario"] == "queue":
        return (
            f"{recipe['participant_count']} people are standing in a queue. The positions run from 1 at the front to {recipe['participant_count']} at the back: {names}.\n\n"
            "Use the clues to determine the full order. There is exactly one valid arrangement.\n"
            "Respond with only the names from front to back, separated by commas.\n\n"
            f"Clues:\n{clue_lines}"
        )
    return (
        f"{recipe['participant_count']} participants finish in a ranking with places from 1 to {recipe['participant_count']}: {names}.\n\n"
        "Use the clues to determine the full ranking. There is exactly one valid result.\n"
        "Respond with only the names from best place to worst place, separated by commas.\n\n"
        f"Clues:\n{clue_lines}"
    )


def _format_clue(clue: Clue, *, language: str, scenario: str, participant_count: int) -> str:
    if scenario == "chairs":
        return _format_chair_clue(clue, language=language)
    if scenario == "queue":
        return _format_queue_clue(clue, language=language)
    return _format_ranking_clue(clue, language=language, participant_count=participant_count)


def _format_chair_clue(clue: Clue, *, language: str) -> str:
    if language == "da":
        if clue.kind == "slot":
            assert clue.slot is not None
            return f"{clue.first} sidder på stol {clue.slot}."
        if clue.kind == "before":
            assert clue.second is not None
            return f"{clue.first} sidder et sted til venstre for {clue.second}."
        if clue.kind == "immediately_before":
            assert clue.second is not None
            return f"{clue.first} sidder umiddelbart til venstre for {clue.second}."
        if clue.kind == "adjacent":
            assert clue.second is not None
            return f"{clue.first} sidder ved siden af {clue.second}."
        if clue.kind == "not_adjacent":
            assert clue.second is not None
            return f"{clue.first} sidder ikke ved siden af {clue.second}."
        if clue.kind == "one_between":
            assert clue.second is not None
            return f"Der er én stol mellem {clue.first} og {clue.second}."
        assert clue.second is not None and clue.third is not None
        return f"{clue.first} sidder et sted mellem {clue.second} og {clue.third}."

    if clue.kind == "slot":
        assert clue.slot is not None
        return f"{clue.first} sits in seat {clue.slot}."
    if clue.kind == "before":
        assert clue.second is not None
        return f"{clue.first} sits somewhere to the left of {clue.second}."
    if clue.kind == "immediately_before":
        assert clue.second is not None
        return f"{clue.first} sits immediately to the left of {clue.second}."
    if clue.kind == "adjacent":
        assert clue.second is not None
        return f"{clue.first} sits next to {clue.second}."
    if clue.kind == "not_adjacent":
        assert clue.second is not None
        return f"{clue.first} does not sit next to {clue.second}."
    if clue.kind == "one_between":
        assert clue.second is not None
        return f"There is one seat between {clue.first} and {clue.second}."
    assert clue.second is not None and clue.third is not None
    return f"{clue.first} sits somewhere between {clue.second} and {clue.third}."


def _format_queue_clue(clue: Clue, *, language: str) -> str:
    if language == "da":
        if clue.kind == "slot":
            assert clue.slot is not None
            return f"{clue.first} står som nummer {clue.slot} i køen."
        if clue.kind == "before":
            assert clue.second is not None
            return f"{clue.first} står foran {clue.second} i køen."
        if clue.kind == "immediately_before":
            assert clue.second is not None
            return f"{clue.first} står direkte foran {clue.second} i køen."
        if clue.kind == "adjacent":
            assert clue.second is not None
            return f"{clue.first} og {clue.second} står ved siden af hinanden i køen."
        if clue.kind == "not_adjacent":
            assert clue.second is not None
            return f"{clue.first} og {clue.second} står ikke ved siden af hinanden i køen."
        if clue.kind == "one_between":
            assert clue.second is not None
            return f"Der er præcis én person mellem {clue.first} og {clue.second} i køen."
        assert clue.second is not None and clue.third is not None
        return f"{clue.first} står et sted mellem {clue.second} og {clue.third} i køen."

    if clue.kind == "slot":
        assert clue.slot is not None
        return f"{clue.first} is in position {clue.slot} in the queue."
    if clue.kind == "before":
        assert clue.second is not None
        return f"{clue.first} stands ahead of {clue.second} in the queue."
    if clue.kind == "immediately_before":
        assert clue.second is not None
        return f"{clue.first} stands directly ahead of {clue.second} in the queue."
    if clue.kind == "adjacent":
        assert clue.second is not None
        return f"{clue.first} and {clue.second} stand next to each other in the queue."
    if clue.kind == "not_adjacent":
        assert clue.second is not None
        return f"{clue.first} and {clue.second} do not stand next to each other in the queue."
    if clue.kind == "one_between":
        assert clue.second is not None
        return f"There is exactly one person between {clue.first} and {clue.second} in the queue."
    assert clue.second is not None and clue.third is not None
    return f"{clue.first} stands somewhere between {clue.second} and {clue.third} in the queue."


def _format_ranking_clue(clue: Clue, *, language: str, participant_count: int) -> str:
    if language == "da":
        if clue.kind == "slot":
            assert clue.slot is not None
            return f"{clue.first} ender på plads {clue.slot}."
        if clue.kind == "before":
            assert clue.second is not None
            return f"{clue.first} ender foran {clue.second}."
        if clue.kind == "immediately_before":
            assert clue.second is not None
            return f"{clue.first} ender lige foran {clue.second}."
        if clue.kind == "adjacent":
            assert clue.second is not None
            return f"{clue.first} og {clue.second} ender på nabo-pladser."
        if clue.kind == "not_adjacent":
            assert clue.second is not None
            return f"{clue.first} og {clue.second} ender ikke på nabo-pladser."
        if clue.kind == "one_between":
            assert clue.second is not None
            return f"Der er præcis én plads mellem {clue.first} og {clue.second}."
        assert clue.second is not None and clue.third is not None
        return f"{clue.first} ender et sted mellem {clue.second} og {clue.third}."

    if clue.kind == "slot":
        assert clue.slot is not None
        return f"{clue.first} finishes in place {clue.slot}."
    if clue.kind == "before":
        assert clue.second is not None
        return f"{clue.first} finishes ahead of {clue.second}."
    if clue.kind == "immediately_before":
        assert clue.second is not None
        return f"{clue.first} finishes immediately ahead of {clue.second}."
    if clue.kind == "adjacent":
        assert clue.second is not None
        return f"{clue.first} and {clue.second} finish in adjacent places."
    if clue.kind == "not_adjacent":
        assert clue.second is not None
        return f"{clue.first} and {clue.second} do not finish in adjacent places."
    if clue.kind == "one_between":
        assert clue.second is not None
        return f"There is exactly one place between {clue.first} and {clue.second}."
    assert clue.second is not None and clue.third is not None
    return f"{clue.first} finishes somewhere between {clue.second} and {clue.third}."


def _recipe_by_id(recipe_id: str) -> dict[str, Any]:
    for recipe in RECIPES:
        if recipe["recipe_id"] == recipe_id:
            return recipe
    raise AssertionError(f"unknown lineup recipe: {recipe_id}")


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]
