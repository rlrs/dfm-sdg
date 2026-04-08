from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any

from sdg.commons import diversity, prompt_surface
from sdg.packs.verifiable_reasoning import ordering
from sdg.packs.verifiable_reasoning.ordering import Clue, OrderPuzzle

NAME_POOLS = {
    "en": ("Alice", "Ben", "Chloe", "Daniel", "Emma", "Grace", "Henry", "Lucy", "Noah", "Zoe"),
    "da": ("Anna", "Bo", "Clara", "Emil", "Freja", "Ida", "Karl", "Maja", "Niels", "Sofie"),
}


@dataclass(frozen=True)
class LineupSurfaceSpec:
    plan: prompt_surface.SurfacePlan

SURFACE_SPECS = {
    "standard": LineupSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="standard",
            block_order=("intro", "instruction", "answer", "clues"),
        ),
    ),
    "briefing": LineupSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "roster", "instruction", "answer", "clues"),
        ),
    ),
    "deduce": LineupSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "roster", "answer", "instruction", "clues"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_chairs",
        "difficulty": "easy",
        "scenario": "chairs",
        "prompt_style": "standard",
        "participant_count": 4,
        "target_clue_count": 5,
        "required_kinds": ("slot", "immediately_before", "at_end"),
        "allowed_kinds": ("slot", "not_slot", "at_end", "before", "immediately_before", "adjacent"),
        "max_kind_counts": {"slot": 1, "at_end": 1, "adjacent": 1},
        "kind_priority": {
            "immediately_before": 0,
            "slot": 1,
            "at_end": 2,
            "adjacent": 3,
            "not_slot": 4,
            "before": 5,
        },
    },
    {
        "recipe_id": "easy_queue",
        "difficulty": "easy",
        "scenario": "queue",
        "prompt_style": "briefing",
        "participant_count": 4,
        "target_clue_count": 5,
        "required_kinds": ("slot", "adjacent", "not_slot"),
        "allowed_kinds": ("slot", "not_slot", "at_end", "before", "immediately_before", "adjacent"),
        "max_kind_counts": {"slot": 1, "adjacent": 1},
        "kind_priority": {
            "adjacent": 0,
            "immediately_before": 1,
            "slot": 2,
            "not_slot": 3,
            "at_end": 4,
            "before": 5,
        },
    },
    {
        "recipe_id": "easy_ranking",
        "difficulty": "easy",
        "scenario": "ranking",
        "prompt_style": "deduce",
        "participant_count": 4,
        "target_clue_count": 5,
        "required_kinds": ("slot", "before", "not_at_end"),
        "allowed_kinds": (
            "slot",
            "not_slot",
            "at_end",
            "not_at_end",
            "before",
            "immediately_before",
            "one_between",
        ),
        "max_kind_counts": {"slot": 1, "at_end": 1, "not_at_end": 1},
        "kind_priority": {
            "immediately_before": 0,
            "one_between": 1,
            "slot": 2,
            "not_at_end": 3,
            "not_slot": 4,
            "before": 5,
            "at_end": 6,
        },
    },
    {
        "recipe_id": "medium_chairs",
        "difficulty": "medium",
        "scenario": "chairs",
        "prompt_style": "briefing",
        "participant_count": 5,
        "target_clue_count": 6,
        "required_kinds": ("middle", "adjacent", "one_between"),
        "allowed_kinds": (
            "slot",
            "not_slot",
            "at_end",
            "not_at_end",
            "middle",
            "before",
            "immediately_before",
            "adjacent",
            "not_adjacent",
            "one_between",
        ),
        "max_kind_counts": {"slot": 1, "middle": 1, "adjacent": 1, "at_end": 1},
        "kind_priority": {
            "middle": 0,
            "one_between": 1,
            "immediately_before": 2,
            "adjacent": 3,
            "not_slot": 4,
            "not_at_end": 5,
            "not_adjacent": 6,
            "before": 7,
            "slot": 8,
            "at_end": 9,
        },
    },
    {
        "recipe_id": "medium_queue",
        "difficulty": "medium",
        "scenario": "queue",
        "prompt_style": "deduce",
        "participant_count": 5,
        "target_clue_count": 6,
        "required_kinds": ("not_slot", "not_adjacent", "before"),
        "allowed_kinds": (
            "slot",
            "not_slot",
            "at_end",
            "not_at_end",
            "middle",
            "before",
            "immediately_before",
            "adjacent",
            "not_adjacent",
            "one_between",
            "between",
        ),
        "max_kind_counts": {"slot": 1, "middle": 1, "at_end": 1},
        "kind_priority": {
            "not_adjacent": 0,
            "one_between": 1,
            "not_slot": 2,
            "immediately_before": 3,
            "before": 4,
            "middle": 5,
            "between": 6,
            "not_at_end": 7,
            "adjacent": 8,
            "slot": 9,
            "at_end": 10,
        },
    },
    {
        "recipe_id": "medium_ranking",
        "difficulty": "medium",
        "scenario": "ranking",
        "prompt_style": "standard",
        "participant_count": 5,
        "target_clue_count": 6,
        "required_kinds": ("between", "middle", "not_at_end"),
        "allowed_kinds": (
            "slot",
            "not_slot",
            "at_end",
            "not_at_end",
            "middle",
            "before",
            "immediately_before",
            "one_between",
            "between",
        ),
        "max_kind_counts": {"slot": 1, "middle": 1, "at_end": 1, "not_at_end": 1},
        "kind_priority": {
            "between": 0,
            "middle": 1,
            "one_between": 2,
            "immediately_before": 3,
            "not_at_end": 4,
            "before": 5,
            "not_slot": 6,
            "slot": 7,
            "at_end": 8,
        },
    },
    {
        "recipe_id": "hard_chairs",
        "difficulty": "hard",
        "scenario": "chairs",
        "prompt_style": "deduce",
        "participant_count": 6,
        "target_clue_count": 7,
        "required_kinds": ("at_end", "not_slot", "between", "not_adjacent"),
        "allowed_kinds": (
            "not_slot",
            "at_end",
            "not_at_end",
            "before",
            "immediately_before",
            "adjacent",
            "not_adjacent",
            "one_between",
            "between",
        ),
        "max_kind_counts": {"at_end": 1, "adjacent": 1, "not_at_end": 1},
        "kind_priority": {
            "between": 0,
            "one_between": 1,
            "not_adjacent": 2,
            "immediately_before": 3,
            "not_slot": 4,
            "at_end": 5,
            "not_at_end": 6,
            "before": 7,
            "adjacent": 8,
        },
    },
    {
        "recipe_id": "hard_queue",
        "difficulty": "hard",
        "scenario": "queue",
        "prompt_style": "standard",
        "participant_count": 6,
        "target_clue_count": 7,
        "required_kinds": ("at_end", "not_at_end", "not_adjacent", "one_between"),
        "allowed_kinds": (
            "not_slot",
            "at_end",
            "not_at_end",
            "before",
            "immediately_before",
            "adjacent",
            "not_adjacent",
            "one_between",
            "between",
        ),
        "max_kind_counts": {"at_end": 1, "not_at_end": 1, "adjacent": 1},
        "kind_priority": {
            "one_between": 0,
            "between": 1,
            "not_adjacent": 2,
            "immediately_before": 3,
            "not_at_end": 4,
            "at_end": 5,
            "not_slot": 6,
            "before": 7,
            "adjacent": 8,
        },
    },
    {
        "recipe_id": "hard_ranking",
        "difficulty": "hard",
        "scenario": "ranking",
        "prompt_style": "briefing",
        "participant_count": 6,
        "target_clue_count": 7,
        "required_kinds": ("at_end", "not_slot", "between", "before"),
        "allowed_kinds": (
            "not_slot",
            "at_end",
            "not_at_end",
            "before",
            "immediately_before",
            "adjacent",
            "not_adjacent",
            "one_between",
            "between",
        ),
        "max_kind_counts": {"at_end": 1, "not_at_end": 1, "adjacent": 1},
        "kind_priority": {
            "between": 0,
            "one_between": 1,
            "immediately_before": 2,
            "not_slot": 3,
            "at_end": 4,
            "before": 5,
            "not_at_end": 6,
            "not_adjacent": 7,
            "adjacent": 8,
        },
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
    chosen_surface = dict(surface_plan or prompt_surface.sample_surface_plan(rng, surface_axes(language)))
    puzzle = _generate_puzzle(rng, language=language, recipe=recipe, surface_plan=chosen_surface)
    return {
        "id": f"verifiable-reasoning-{index:05d}",
        "prompt": puzzle.prompt,
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
            "prompt_style": recipe["prompt_style"],
            "participant_count": recipe["participant_count"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[str, ...] | None:
    return ordering.parse_target(text, hidden["participants"])


def canonical_target(parsed: tuple[str, ...], hidden: dict[str, object]) -> str:
    return ordering.format_target(parsed)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    names = ", ".join(hidden["participants"])
    placeholder_format = _name_placeholder_list(language, len(hidden["participants"]))
    if language == "da":
        return (
            "I din svarblok skal den endelige løsning være rækkefølgen som en kommasepareret liste i korrekt orden.\n"
            f"Brug kun disse navne: {names}\n"
            f"Format: {placeholder_format}"
        )

    return (
        "In your answer block, the final solution should be the ordered names as a comma-separated list in the correct order.\n"
        f"Use only these names: {names}\n"
        f"Format: {placeholder_format}"
    )


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
        "participant_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("participant_count",), observed_getter=_meta_getter),
        "clue_kind_minimums": diversity.counter_minimum_check(clue_kind_counts, dict(required_minimums)),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def _generate_puzzle(
    rng: Random,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> OrderPuzzle:
    participants = tuple(sorted(rng.sample(NAME_POOLS[language], k=recipe["participant_count"])))

    for _ in range(200):
        solution = tuple(rng.sample(participants, k=len(participants)))
        clues = ordering.select_clues(solution, rng, recipe)
        prompt = _format_prompt(
            participants,
            clues,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
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
    surface_plan: dict[str, object],
) -> str:
    names = ", ".join(participants)
    surface_spec = SURFACE_SPECS[recipe["prompt_style"]]
    intro_lines = _intro_lines(
        language=language,
        scenario=recipe["scenario"],
        participant_count=recipe["participant_count"],
        names=names,
        intro_variant=str(surface_plan["surface_intro"]),
        prompt_style=recipe["prompt_style"],
    )
    roster_line = _roster_line(
        language=language,
        scenario=recipe["scenario"],
        names=names,
        intro_variant=str(surface_plan["surface_intro"]),
    )
    clue_lines = tuple(
        _format_clue(
            clue,
            language=language,
            scenario=recipe["scenario"],
            participant_count=recipe["participant_count"],
            clue_style=str(surface_plan["surface_clue"]),
        )
        for clue in clues
    )
    clue_heading = "Ledetråde" if language == "da" else "Clues"

    blocks = {
        "intro": prompt_surface.PromptBlock("intro", intro_lines),
        "roster": prompt_surface.PromptBlock("roster", (roster_line,)),
        "instruction": prompt_surface.PromptBlock(
            "instruction",
            (_instruction_line(language=language, instruction_variant=str(surface_plan["surface_instruction"])),),
        ),
        "answer": prompt_surface.PromptBlock(
            "answer",
            _answer_lines(
                language=language,
                scenario=recipe["scenario"],
                participant_count=recipe["participant_count"],
                answer_variant=str(surface_plan["surface_answer"]),
            ),
        ),
        "clues": prompt_surface.PromptBlock("clues", clue_lines, heading=clue_heading, numbered=True),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_lines(
    *,
    language: str,
    scenario: str,
    participant_count: int,
    names: str,
    intro_variant: str,
    prompt_style: str,
) -> tuple[str, ...]:
    scenario_line = _scenario_line(
        language=language,
        scenario=scenario,
        participant_count=participant_count,
        intro_variant=intro_variant,
    )
    roster_line = _roster_line(
        language=language,
        scenario=scenario,
        names=names,
        intro_variant=intro_variant,
    )

    if prompt_style == "standard":
        return (f"{scenario_line} {roster_line}",)
    if prompt_style == "briefing":
        return (scenario_line,)
    if language == "da":
        return ("Find hele rækkefølgen ud fra ledetrådene.", scenario_line)
    return ("Work out the complete order from the clues.", scenario_line)


def _scenario_line(
    *,
    language: str,
    scenario: str,
    participant_count: int,
    intro_variant: str,
) -> str:
    if scenario == "chairs":
        if language == "da":
            if intro_variant == "context":
                return f"{participant_count} personer skal placeres på en række stole, nummereret 1 til {participant_count} fra venstre mod højre."
            if intro_variant == "assignment":
                return f"En række med {participant_count} stole skal besættes, og pladserne går fra 1 til {participant_count} fra venstre mod højre."
            return f"Bestem hvem der sidder på stol 1 til {participant_count} i en række fra venstre mod højre."
        if intro_variant == "context":
            return f"{participant_count} people are arranged in a row of chairs, numbered 1 to {participant_count} from left to right."
        if intro_variant == "assignment":
            return f"A row of {participant_count} chairs has to be filled, with seats numbered 1 to {participant_count} from left to right."
        return f"Determine who sits in seats 1 to {participant_count} in a single left-to-right row."

    if scenario == "queue":
        if language == "da":
            if intro_variant == "context":
                return f"{participant_count} personer står i en kø. Positionerne går fra 1 foran til {participant_count} bagerst."
            if intro_variant == "assignment":
                return f"En kø med {participant_count} personer skal ordnes fra position 1 forrest til {participant_count} bagerst."
            return f"Bestem hele køen fra den forreste plads 1 til den bagerste plads {participant_count}."
        if intro_variant == "context":
            return f"{participant_count} people are standing in a queue. The positions run from 1 at the front to {participant_count} at the back."
        if intro_variant == "assignment":
            return f"A queue of {participant_count} people must be ordered from position 1 at the front to {participant_count} at the back."
        return f"Work out the full queue from front position 1 to back position {participant_count}."

    if language == "da":
        if intro_variant == "context":
            return f"{participant_count} deltagere ender i en rangliste med pladser fra 1 til {participant_count}."
        if intro_variant == "assignment":
            return f"En rangliste med {participant_count} pladser skal udfyldes fra plads 1 til plads {participant_count}."
        return f"Bestem hele ranglisten fra plads 1 til plads {participant_count}."
    if intro_variant == "context":
        return f"{participant_count} participants finish in a ranking with places from 1 to {participant_count}."
    if intro_variant == "assignment":
        return f"A ranking with {participant_count} places has to be filled from place 1 to place {participant_count}."
    return f"Determine the full ranking from place 1 to place {participant_count}."


def _roster_line(*, language: str, scenario: str, names: str, intro_variant: str) -> str:
    if scenario == "queue":
        if language == "da":
            if intro_variant == "task_first":
                return f"Brug disse navne: {names}."
            if intro_variant == "assignment":
                return f"Følgende personer indgår i køen: {names}."
            return f"Personerne i køen er: {names}."
        if intro_variant == "task_first":
            return f"Use these names: {names}."
        if intro_variant == "assignment":
            return f"The queue includes these people: {names}."
        return f"The people in the queue are: {names}."

    if scenario == "ranking":
        if language == "da":
            if intro_variant == "task_first":
                return f"Brug disse deltagere: {names}."
            if intro_variant == "assignment":
                return f"Ranglisten består af disse deltagere: {names}."
            return f"Deltagerne på ranglisten er: {names}."
        if intro_variant == "task_first":
            return f"Use these participants: {names}."
        if intro_variant == "assignment":
            return f"The ranking includes these participants: {names}."
        return f"The participants are: {names}."

    if language == "da":
        if intro_variant == "task_first":
            return f"Brug disse navne: {names}."
        if intro_variant == "assignment":
            return f"Følgende personer skal placeres: {names}."
        return f"Personerne er: {names}."
    if intro_variant == "task_first":
        return f"Use these names: {names}."
    if intro_variant == "assignment":
        return f"The following people must be placed: {names}."
    return f"The people are: {names}."


def _instruction_line(*, language: str, instruction_variant: str) -> str:
    if language == "da":
        if instruction_variant == "solve":
            return "Brug ledetrådene til at bestemme hele rækkefølgen. Der er præcis én gyldig løsning."
        if instruction_variant == "unique":
            return "Ledetrådene fastlægger én og kun én fuld rækkefølge."
        return "Alle ledetråde skal passe til én entydig løsning."

    if instruction_variant == "solve":
        return "Use the clues to determine the full order. There is exactly one valid arrangement."
    if instruction_variant == "unique":
        return "The clues pin down one and only one complete order."
    return "Every clue must fit a single consistent solution."


def _answer_lines(
    *,
    language: str,
    scenario: str,
    participant_count: int,
    answer_variant: str,
) -> tuple[str, ...]:
    placeholder_format = _name_placeholder_list(language, participant_count)

    if scenario == "queue":
        if language == "da":
            options = {
                "respond": "Når du giver dit endelige svar, skal navnene stå fra forrest til bagerst, adskilt af kommaer.",
                "write": "Når du skriver det endelige svar, skal navnene stå fra første til sidste plads i køen, adskilt af kommaer.",
                "complete": "Giv den endelige kø fra forrest til bagerst som en kommasepareret liste.",
            }
            return (
                options[answer_variant],
                f"Brug præcis {participant_count} navne. Format: {placeholder_format}",
            )
        options = {
            "respond": "When you give your final answer, list the names from front to back, separated by commas.",
            "write": "When you write the final answer, list the names from first to last position in the queue, separated by commas.",
            "complete": "Give the final queue from front to back as a comma-separated list.",
        }
        return (
            options[answer_variant],
            f"Use exactly {participant_count} names. Format: {placeholder_format}",
        )

    if scenario == "ranking":
        if language == "da":
            options = {
                "respond": "Når du giver dit endelige svar, skal navnene stå fra bedste til dårligste placering, adskilt af kommaer.",
                "write": "Når du skriver det endelige svar, skal navnene stå fra første til sidste plads, adskilt af kommaer.",
                "complete": "Giv den endelige rangliste fra bedst til dårligst som en kommasepareret liste.",
            }
            return (
                options[answer_variant],
                f"Brug præcis {participant_count} navne. Format: {placeholder_format}",
            )
        options = {
            "respond": "When you give your final answer, list the names from best place to worst place, separated by commas.",
            "write": "When you write the final answer, list the names from first place to last place, separated by commas.",
            "complete": "Give the final ranking from best to worst as a comma-separated list.",
        }
        return (
            options[answer_variant],
            f"Use exactly {participant_count} names. Format: {placeholder_format}",
        )

    if language == "da":
        options = {
            "respond": "Når du giver dit endelige svar, skal navnene stå i rækkefølge, adskilt af kommaer.",
            "write": "Når du skriver det endelige svar, skal navnene stå i den korrekte rækkefølge, adskilt af kommaer.",
            "complete": "Giv den endelige rækkefølge som en kommasepareret liste.",
        }
        return (
            options[answer_variant],
            f"Brug præcis {participant_count} navne. Format: {placeholder_format}",
        )
    options = {
        "respond": "When you give your final answer, list the names in order, separated by commas.",
        "write": "When you write the final answer, list the names in the correct order, separated by commas.",
        "complete": "Give the final order as a comma-separated list.",
    }
    return (
        options[answer_variant],
        f"Use exactly {participant_count} names. Format: {placeholder_format}",
    )


def _name_placeholder_list(language: str, participant_count: int) -> str:
    token = "Navn" if language == "da" else "Name"
    return ", ".join(token for _ in range(participant_count))


def _format_clue(
    clue: Clue,
    *,
    language: str,
    scenario: str,
    participant_count: int,
    clue_style: str,
) -> str:
    if scenario == "chairs":
        return _format_chair_clue(clue, language=language, clue_style=clue_style)
    if scenario == "queue":
        return _format_queue_clue(clue, language=language, clue_style=clue_style)
    return _format_ranking_clue(
        clue,
        language=language,
        participant_count=participant_count,
        clue_style=clue_style,
    )


def _format_chair_clue(clue: Clue, *, language: str, clue_style: str) -> str:
    if language == "da":
        if clue.kind == "slot":
            assert clue.slot is not None
            if clue_style == "compact":
                return f"Stol {clue.slot} tilhører {clue.first}."
            if clue_style == "deductive":
                return f"{clue.first} må sidde på stol {clue.slot}."
            return f"{clue.first} sidder på stol {clue.slot}."
        if clue.kind == "not_slot":
            assert clue.slot is not None
            if clue_style == "compact":
                return f"{clue.first} sidder ikke på stol {clue.slot}."
            if clue_style == "deductive":
                return f"Stol {clue.slot} kan ikke tilhøre {clue.first}."
            return f"{clue.first} sidder ikke på stol {clue.slot}."
        if clue.kind == "at_end":
            if clue_style == "compact":
                return f"{clue.first} sidder yderst."
            if clue_style == "deductive":
                return f"{clue.first} må sidde enten yderst til venstre eller yderst til højre."
            return f"{clue.first} sidder på en af endestolene."
        if clue.kind == "not_at_end":
            if clue_style == "compact":
                return f"{clue.first} sidder ikke yderst."
            if clue_style == "deductive":
                return f"{clue.first} må sidde et sted mellem de to ender."
            return f"{clue.first} sidder ikke yderst."
        if clue.kind == "middle":
            if clue_style == "compact":
                return f"Midterstolen tilhører {clue.first}."
            if clue_style == "deductive":
                return f"{clue.first} må sidde på den midterste stol."
            return f"{clue.first} sidder på den midterste stol."
        if clue.kind == "before":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} sidder til venstre for {clue.second}."
            if clue_style == "deductive":
                return f"{clue.second} sidder et sted til højre for {clue.first}."
            return f"{clue.first} sidder et sted til venstre for {clue.second}."
        if clue.kind == "immediately_before":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} sidder lige før {clue.second}."
            if clue_style == "deductive":
                return f"{clue.second} sidder umiddelbart til højre for {clue.first}."
            return f"{clue.first} sidder umiddelbart til venstre for {clue.second}."
        if clue.kind == "adjacent":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} sidder side om side."
            if clue_style == "deductive":
                return f"Der står to nabostole med {clue.first} og {clue.second}."
            return f"{clue.first} sidder ved siden af {clue.second}."
        if clue.kind == "not_adjacent":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} sidder ikke side om side."
            if clue_style == "deductive":
                return f"Der er mindst én stol mellem {clue.first} og {clue.second}."
            return f"{clue.first} sidder ikke ved siden af {clue.second}."
        if clue.kind == "one_between":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} har én stol imellem sig."
            if clue_style == "deductive":
                return f"Præcis én stol skiller {clue.first} fra {clue.second}."
            return f"Der er én stol mellem {clue.first} og {clue.second}."
        assert clue.second is not None and clue.third is not None
        if clue_style == "compact":
            return f"{clue.first} sidder mellem {clue.second} og {clue.third}."
        if clue_style == "deductive":
            return f"{clue.second} og {clue.third} ligger på hver sin side af {clue.first}."
        return f"{clue.first} sidder et sted mellem {clue.second} og {clue.third}."

    if clue.kind == "slot":
        assert clue.slot is not None
        if clue_style == "compact":
            return f"Seat {clue.slot} belongs to {clue.first}."
        if clue_style == "deductive":
            return f"{clue.first} has to be in seat {clue.slot}."
        return f"{clue.first} sits in seat {clue.slot}."
    if clue.kind == "not_slot":
        assert clue.slot is not None
        if clue_style == "compact":
            return f"{clue.first} is not in seat {clue.slot}."
        if clue_style == "deductive":
            return f"Seat {clue.slot} cannot belong to {clue.first}."
        return f"{clue.first} does not sit in seat {clue.slot}."
    if clue.kind == "at_end":
        if clue_style == "compact":
            return f"{clue.first} sits in one of the end seats."
        if clue_style == "deductive":
            return f"{clue.first} has to sit either at the far left or the far right."
        return f"{clue.first} sits at one end of the row."
    if clue.kind == "not_at_end":
        if clue_style == "compact":
            return f"{clue.first} is not in an end seat."
        if clue_style == "deductive":
            return f"{clue.first} has to sit somewhere between the two ends."
        return f"{clue.first} does not sit at either end."
    if clue.kind == "middle":
        if clue_style == "compact":
            return f"The middle seat belongs to {clue.first}."
        if clue_style == "deductive":
            return f"{clue.first} has to sit in the middle seat."
        return f"{clue.first} sits in the middle seat."
    if clue.kind == "before":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} sits to the left of {clue.second}."
        if clue_style == "deductive":
            return f"{clue.second} sits somewhere to the right of {clue.first}."
        return f"{clue.first} sits somewhere to the left of {clue.second}."
    if clue.kind == "immediately_before":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} sits right before {clue.second}."
        if clue_style == "deductive":
            return f"{clue.second} sits immediately to the right of {clue.first}."
        return f"{clue.first} sits immediately to the left of {clue.second}."
    if clue.kind == "adjacent":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} sit side by side."
        if clue_style == "deductive":
            return f"There is a neighboring pair with {clue.first} and {clue.second}."
        return f"{clue.first} sits next to {clue.second}."
    if clue.kind == "not_adjacent":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} do not sit side by side."
        if clue_style == "deductive":
            return f"There is at least one seat between {clue.first} and {clue.second}."
        return f"{clue.first} does not sit next to {clue.second}."
    if clue.kind == "one_between":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} have one seat between them."
        if clue_style == "deductive":
            return f"Exactly one seat separates {clue.first} and {clue.second}."
        return f"There is one seat between {clue.first} and {clue.second}."
    assert clue.second is not None and clue.third is not None
    if clue_style == "compact":
        return f"{clue.first} sits between {clue.second} and {clue.third}."
    if clue_style == "deductive":
        return f"{clue.second} and {clue.third} sit on opposite sides of {clue.first}."
    return f"{clue.first} sits somewhere between {clue.second} and {clue.third}."


def _format_queue_clue(clue: Clue, *, language: str, clue_style: str) -> str:
    if language == "da":
        if clue.kind == "slot":
            assert clue.slot is not None
            if clue_style == "compact":
                return f"Plads {clue.slot} i køen tilhører {clue.first}."
            if clue_style == "deductive":
                return f"{clue.first} må stå som nummer {clue.slot} i køen."
            return f"{clue.first} står som nummer {clue.slot} i køen."
        if clue.kind == "not_slot":
            assert clue.slot is not None
            if clue_style == "compact":
                return f"{clue.first} står ikke som nummer {clue.slot} i køen."
            if clue_style == "deductive":
                return f"Plads {clue.slot} i køen kan ikke tilhøre {clue.first}."
            return f"{clue.first} står ikke på plads {clue.slot} i køen."
        if clue.kind == "at_end":
            if clue_style == "compact":
                return f"{clue.first} står enten helt forrest eller helt bagerst."
            if clue_style == "deductive":
                return f"{clue.first} må stå enten helt forrest eller helt bagerst."
            return f"{clue.first} står enten helt forrest eller helt bagerst i køen."
        if clue.kind == "not_at_end":
            if clue_style == "compact":
                return f"{clue.first} står hverken forrest eller bagerst."
            if clue_style == "deductive":
                return f"{clue.first} må stå et sted mellem forrest og bagerst."
            return f"{clue.first} står hverken forrest eller bagerst i køen."
        if clue.kind == "middle":
            if clue_style == "compact":
                return f"Midterpladsen i køen tilhører {clue.first}."
            if clue_style == "deductive":
                return f"{clue.first} må stå i midten af køen."
            return f"{clue.first} står på den midterste plads i køen."
        if clue.kind == "before":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} står foran {clue.second}."
            if clue_style == "deductive":
                return f"{clue.second} står bag {clue.first} i køen."
            return f"{clue.first} står foran {clue.second} i køen."
        if clue.kind == "immediately_before":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} står lige foran {clue.second}."
            if clue_style == "deductive":
                return f"{clue.second} står direkte bag {clue.first} i køen."
            return f"{clue.first} står direkte foran {clue.second} i køen."
        if clue.kind == "adjacent":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} står side om side i køen."
            if clue_style == "deductive":
                return f"Der står ingen mellem {clue.first} og {clue.second} i køen."
            return f"{clue.first} og {clue.second} står ved siden af hinanden i køen."
        if clue.kind == "not_adjacent":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} står ikke side om side i køen."
            if clue_style == "deductive":
                return f"Der står mindst én person mellem {clue.first} og {clue.second} i køen."
            return f"{clue.first} og {clue.second} står ikke ved siden af hinanden i køen."
        if clue.kind == "one_between":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} har præcis én person mellem sig i køen."
            if clue_style == "deductive":
                return f"Præcis én person skiller {clue.first} og {clue.second} i køen."
            return f"Der er præcis én person mellem {clue.first} og {clue.second} i køen."
        assert clue.second is not None and clue.third is not None
        if clue_style == "compact":
            return f"{clue.first} står mellem {clue.second} og {clue.third} i køen."
        if clue_style == "deductive":
            return f"{clue.second} og {clue.third} står på hver sin side af {clue.first} i køen."
        return f"{clue.first} står et sted mellem {clue.second} og {clue.third} i køen."

    if clue.kind == "slot":
        assert clue.slot is not None
        if clue_style == "compact":
            return f"Queue position {clue.slot} belongs to {clue.first}."
        if clue_style == "deductive":
            return f"{clue.first} has to stand in position {clue.slot}."
        return f"{clue.first} is in position {clue.slot} in the queue."
    if clue.kind == "not_slot":
        assert clue.slot is not None
        if clue_style == "compact":
            return f"{clue.first} is not in queue position {clue.slot}."
        if clue_style == "deductive":
            return f"Queue position {clue.slot} cannot belong to {clue.first}."
        return f"{clue.first} is not in position {clue.slot} in the queue."
    if clue.kind == "at_end":
        if clue_style == "compact":
            return f"{clue.first} stands at one end of the queue."
        if clue_style == "deductive":
            return f"{clue.first} has to stand either at the very front or the very back."
        return f"{clue.first} stands either at the very front or at the very back of the queue."
    if clue.kind == "not_at_end":
        if clue_style == "compact":
            return f"{clue.first} is not at either end of the queue."
        if clue_style == "deductive":
            return f"{clue.first} has to stand somewhere between the front and the back."
        return f"{clue.first} stands neither at the front nor at the back of the queue."
    if clue.kind == "middle":
        if clue_style == "compact":
            return f"The middle queue position belongs to {clue.first}."
        if clue_style == "deductive":
            return f"{clue.first} has to stand in the middle of the queue."
        return f"{clue.first} stands in the middle position of the queue."
    if clue.kind == "before":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} stands before {clue.second}."
        if clue_style == "deductive":
            return f"{clue.second} stands behind {clue.first} in the queue."
        return f"{clue.first} stands ahead of {clue.second} in the queue."
    if clue.kind == "immediately_before":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} stands right before {clue.second}."
        if clue_style == "deductive":
            return f"{clue.second} stands directly behind {clue.first} in the queue."
        return f"{clue.first} stands directly ahead of {clue.second} in the queue."
    if clue.kind == "adjacent":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} stand side by side in the queue."
        if clue_style == "deductive":
            return f"No one stands between {clue.first} and {clue.second} in the queue."
        return f"{clue.first} and {clue.second} stand next to each other in the queue."
    if clue.kind == "not_adjacent":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} do not stand side by side in the queue."
        if clue_style == "deductive":
            return f"At least one person stands between {clue.first} and {clue.second} in the queue."
        return f"{clue.first} and {clue.second} do not stand next to each other in the queue."
    if clue.kind == "one_between":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} have exactly one person between them in the queue."
        if clue_style == "deductive":
            return f"Exactly one person separates {clue.first} and {clue.second} in the queue."
        return f"There is exactly one person between {clue.first} and {clue.second} in the queue."
    assert clue.second is not None and clue.third is not None
    if clue_style == "compact":
        return f"{clue.first} stands between {clue.second} and {clue.third} in the queue."
    if clue_style == "deductive":
        return f"{clue.second} and {clue.third} stand on opposite sides of {clue.first} in the queue."
    return f"{clue.first} stands somewhere between {clue.second} and {clue.third} in the queue."


def _format_ranking_clue(
    clue: Clue,
    *,
    language: str,
    participant_count: int,
    clue_style: str,
) -> str:
    if language == "da":
        if clue.kind == "slot":
            assert clue.slot is not None
            if clue_style == "compact":
                return f"Plads {clue.slot} går til {clue.first}."
            if clue_style == "deductive":
                return f"{clue.first} må ende på plads {clue.slot}."
            return f"{clue.first} ender på plads {clue.slot}."
        if clue.kind == "not_slot":
            assert clue.slot is not None
            if clue_style == "compact":
                return f"{clue.first} ender ikke på plads {clue.slot}."
            if clue_style == "deductive":
                return f"Plads {clue.slot} kan ikke gå til {clue.first}."
            return f"{clue.first} ender ikke på plads {clue.slot}."
        if clue.kind == "at_end":
            if clue_style == "compact":
                return f"{clue.first} ender enten først eller sidst."
            if clue_style == "deductive":
                return f"{clue.first} må ende enten som nummer et eller sidst."
            return f"{clue.first} ender enten først eller sidst."
        if clue.kind == "not_at_end":
            if clue_style == "compact":
                return f"{clue.first} ender hverken først eller sidst."
            if clue_style == "deductive":
                return f"{clue.first} må ende et sted mellem første og sidste plads."
            return f"{clue.first} ender hverken først eller sidst."
        if clue.kind == "middle":
            middle_place = participant_count // 2 + 1
            if clue_style == "compact":
                return f"Midterpladsen går til {clue.first}."
            if clue_style == "deductive":
                return f"{clue.first} må ende på den midterste plads, altså plads {middle_place}."
            return f"{clue.first} ender på den midterste plads."
        if clue.kind == "before":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} ender foran {clue.second}."
            if clue_style == "deductive":
                return f"{clue.second} ender bag {clue.first}."
            return f"{clue.first} ender foran {clue.second}."
        if clue.kind == "immediately_before":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} ender lige før {clue.second}."
            if clue_style == "deductive":
                return f"{clue.second} ender lige bag {clue.first}."
            return f"{clue.first} ender lige foran {clue.second}."
        if clue.kind == "adjacent":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} ender side om side på ranglisten."
            if clue_style == "deductive":
                return f"Der er ingen plads mellem {clue.first} og {clue.second} i ranglisten."
            return f"{clue.first} og {clue.second} ender på nabo-pladser."
        if clue.kind == "not_adjacent":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} ender ikke side om side."
            if clue_style == "deductive":
                return f"Mindst én placering skiller {clue.first} fra {clue.second}."
            return f"{clue.first} og {clue.second} ender ikke på nabo-pladser."
        if clue.kind == "one_between":
            assert clue.second is not None
            if clue_style == "compact":
                return f"{clue.first} og {clue.second} har præcis én plads mellem sig."
            if clue_style == "deductive":
                return f"Præcis én placering skiller {clue.first} fra {clue.second}."
            return f"Der er præcis én plads mellem {clue.first} og {clue.second}."
        assert clue.second is not None and clue.third is not None
        if clue_style == "compact":
            return f"{clue.first} ender mellem {clue.second} og {clue.third}."
        if clue_style == "deductive":
            return f"{clue.second} og {clue.third} ender på hver sin side af {clue.first}."
        return f"{clue.first} ender et sted mellem {clue.second} og {clue.third}."

    if clue.kind == "slot":
        assert clue.slot is not None
        if clue_style == "compact":
            return f"Place {clue.slot} goes to {clue.first}."
        if clue_style == "deductive":
            return f"{clue.first} has to finish in place {clue.slot}."
        return f"{clue.first} finishes in place {clue.slot}."
    if clue.kind == "not_slot":
        assert clue.slot is not None
        if clue_style == "compact":
            return f"{clue.first} does not finish in place {clue.slot}."
        if clue_style == "deductive":
            return f"Place {clue.slot} cannot go to {clue.first}."
        return f"{clue.first} does not finish in place {clue.slot}."
    if clue.kind == "at_end":
        if clue_style == "compact":
            return f"{clue.first} finishes at one end of the ranking."
        if clue_style == "deductive":
            return f"{clue.first} has to finish either first or last."
        return f"{clue.first} finishes either first or last."
    if clue.kind == "not_at_end":
        if clue_style == "compact":
            return f"{clue.first} does not finish at either end."
        if clue_style == "deductive":
            return f"{clue.first} has to finish somewhere between first and last."
        return f"{clue.first} finishes neither first nor last."
    if clue.kind == "middle":
        middle_place = participant_count // 2 + 1
        if clue_style == "compact":
            return f"The middle place goes to {clue.first}."
        if clue_style == "deductive":
            return f"{clue.first} has to finish in the middle place, namely place {middle_place}."
        return f"{clue.first} finishes in the middle place."
    if clue.kind == "before":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} finishes before {clue.second}."
        if clue_style == "deductive":
            return f"{clue.second} finishes behind {clue.first}."
        return f"{clue.first} finishes ahead of {clue.second}."
    if clue.kind == "immediately_before":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} finishes right before {clue.second}."
        if clue_style == "deductive":
            return f"{clue.second} finishes immediately behind {clue.first}."
        return f"{clue.first} finishes immediately ahead of {clue.second}."
    if clue.kind == "adjacent":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} finish side by side in the ranking."
        if clue_style == "deductive":
            return f"There is no place between {clue.first} and {clue.second} in the ranking."
        return f"{clue.first} and {clue.second} finish in adjacent places."
    if clue.kind == "not_adjacent":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} do not finish side by side."
        if clue_style == "deductive":
            return f"At least one place separates {clue.first} and {clue.second}."
        return f"{clue.first} and {clue.second} do not finish in adjacent places."
    if clue.kind == "one_between":
        assert clue.second is not None
        if clue_style == "compact":
            return f"{clue.first} and {clue.second} have exactly one place between them."
        if clue_style == "deductive":
            return f"Exactly one place separates {clue.first} and {clue.second}."
        return f"There is exactly one place between {clue.first} and {clue.second}."
    assert clue.second is not None and clue.third is not None
    if clue_style == "compact":
        return f"{clue.first} finishes between {clue.second} and {clue.third}."
    if clue_style == "deductive":
        return f"{clue.second} and {clue.third} finish on opposite sides of {clue.first}."
    return f"{clue.first} finishes somewhere between {clue.second} and {clue.third}."


def _recipe_by_id(recipe_id: str) -> dict[str, Any]:
    for recipe in RECIPES:
        if recipe["recipe_id"] == recipe_id:
            return recipe
    raise AssertionError(f"unknown lineup recipe: {recipe_id}")


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]
