from collections import deque
from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Any

from sdg.commons import diversity, prompt_surface


@dataclass(frozen=True)
class JugAction:
    kind: str
    source: int
    target: int | None = None

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "kind": self.kind,
            "source": self.source,
            "target": self.target,
        }


@dataclass(frozen=True)
class JugPuzzle:
    capacities: tuple[int, ...]
    target: int
    solution: tuple[JugAction, ...]
    prompt: str


@dataclass(frozen=True)
class JugSurfaceSpec:
    plan: prompt_surface.SurfacePlan


SURFACE_SPECS = {
    "capacities": JugSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="capacities",
            block_order=("intro", "capacities", "rules", "instruction", "answer"),
        ),
    ),
    "goal_first": JugSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="goal_first",
            block_order=("intro", "instruction", "capacities", "rules", "answer"),
        ),
    ),
    "briefing": JugSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "rules", "capacities", "instruction", "answer"),
        ),
    ),
}


RECIPES = (
    {
        "recipe_id": "easy_goal_first_3",
        "difficulty": "easy",
        "prompt_style": "goal_first",
        "jug_count": 3,
        "capacity_min": 2,
        "capacity_max": 8,
        "min_steps": 1,
        "max_steps": 3,
        "max_state_space": 900,
    },
    {
        "recipe_id": "easy_capacities_4",
        "difficulty": "easy",
        "prompt_style": "capacities",
        "jug_count": 4,
        "capacity_min": 2,
        "capacity_max": 8,
        "min_steps": 2,
        "max_steps": 4,
        "max_state_space": 2200,
    },
    {
        "recipe_id": "medium_briefing_4",
        "difficulty": "medium",
        "prompt_style": "briefing",
        "jug_count": 4,
        "capacity_min": 3,
        "capacity_max": 9,
        "min_steps": 4,
        "max_steps": 6,
        "max_state_space": 4500,
    },
    {
        "recipe_id": "medium_goal_first_4",
        "difficulty": "medium",
        "prompt_style": "goal_first",
        "jug_count": 4,
        "capacity_min": 3,
        "capacity_max": 10,
        "min_steps": 4,
        "max_steps": 7,
        "max_state_space": 7000,
    },
    {
        "recipe_id": "hard_capacities_4",
        "difficulty": "hard",
        "prompt_style": "capacities",
        "jug_count": 4,
        "capacity_min": 4,
        "capacity_max": 12,
        "min_steps": 4,
        "max_steps": 8,
        "max_state_space": 12000,
    },
    {
        "recipe_id": "hard_briefing_4",
        "difficulty": "hard",
        "prompt_style": "briefing",
        "jug_count": 4,
        "capacity_min": 4,
        "capacity_max": 12,
        "min_steps": 5,
        "max_steps": 9,
        "max_state_space": 12000,
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
            "language": language,
            "capacities": list(puzzle.capacities),
            "target": puzzle.target,
            "solution": [action.to_dict() for action in puzzle.solution],
            "minimal_steps": len(puzzle.solution),
        },
        "sources": [{"kind": "dolci_subset", "value": "jugpuzzle"}],
        "meta": {
            "family": "jugpuzzle_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": len(puzzle.capacities),
            "given_count": len(puzzle.capacities),
            "jug_count": len(puzzle.capacities),
            "target_amount": puzzle.target,
            "minimal_steps": len(puzzle.solution),
            "output_format": "action_sequence",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[JugAction, ...] | None:
    capacities = tuple(int(value) for value in hidden["capacities"])
    jug_count = len(capacities)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    parsed: list[JugAction] = []
    for line in lines:
        normalized = line.strip().lstrip("*•- ").strip()
        parts = normalized.split()
        if not parts:
            return None
        command = parts[0].lower()

        if command == "fill" and len(parts) == 2 and parts[1].isdigit():
            source = int(parts[1])
            if not 0 <= source < jug_count:
                return None
            parsed.append(JugAction("fill", source))
            continue

        if command == "empty" and len(parts) == 2 and parts[1].isdigit():
            source = int(parts[1])
            if not 0 <= source < jug_count:
                return None
            parsed.append(JugAction("empty", source))
            continue

        if command == "pour" and len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            source = int(parts[1])
            target = int(parts[2])
            if not 0 <= source < jug_count:
                return None
            if not 0 <= target < jug_count:
                return None
            if source == target:
                return None
            parsed.append(JugAction("pour", source, target))
            continue

        return None

    return tuple(parsed)


def canonical_target(parsed: tuple[JugAction, ...], hidden: dict[str, object]) -> str:
    return format_target(parsed)


def format_target(actions: tuple[JugAction, ...]) -> str:
    return "\n".join(_render_action(action) for action in actions)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    example = "Fill 0\nPour 0 1\nEmpty 1"

    if language == "da":
        return (
            "I din svarblok skal du skrive én handling per linje.\n"
            "Brug kun kommandoerne `Fill i`, `Empty i` og `Pour i j`.\n"
            "Brug jug-numrene præcis som i opgaven.\n"
            f"Format:\n{example}"
        )

    return (
        "In your answer block, write one action per line.\n"
        "Use only the commands `Fill i`, `Empty i`, and `Pour i j`.\n"
        "Use the jug indices exactly as shown in the puzzle.\n"
        f"Format:\n{example}"
    )


def is_correct(parsed: tuple[JugAction, ...], hidden: dict[str, object]) -> bool:
    capacities = tuple(int(value) for value in hidden["capacities"])
    target = int(hidden["target"])
    expected_steps = int(hidden["minimal_steps"])
    if len(parsed) != expected_steps:
        return False

    state = tuple(0 for _ in capacities)
    for action in parsed:
        next_state = _apply_action(state, capacities, action)
        if next_state is None:
            return False
        state = next_state

    return target in state


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    capacities = tuple(int(value) for value in hidden["capacities"])
    target = int(hidden["target"])
    solution = tuple(_action_from_dict(payload) for payload in hidden["solution"])
    search = _search(capacities)
    summary = _target_summary(search, target)
    if summary is None:
        return False
    if summary["solution_count"] != 1:
        return False
    return summary["actions"] == solution


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "prompt_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("prompt_style",), observed_getter=_meta_getter),
        "jug_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("jug_count",), observed_getter=_meta_getter),
        "surface_intro_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_intro",), observed_getter=_meta_getter),
        "surface_instruction_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_instruction",), observed_getter=_meta_getter),
        "surface_answer_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_answer",), observed_getter=_meta_getter),
        "surface_clue_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_clue",), observed_getter=_meta_getter),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def _generate_puzzle(
    rng: Random,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> JugPuzzle:
    jug_count = int(recipe["jug_count"])
    capacity_min = int(recipe["capacity_min"])
    capacity_max = int(recipe["capacity_max"])
    min_steps = int(recipe["min_steps"])
    max_steps = int(recipe["max_steps"])
    max_state_space = int(recipe["max_state_space"])
    capacity_options = list(combinations(range(capacity_min, capacity_max + 1), jug_count))
    rng.shuffle(capacity_options)

    for choice in capacity_options:
        capacities = tuple(rng.sample(choice, len(choice)))
        if _state_space(capacities) > max_state_space:
            continue
        search = _search(capacities)
        target_candidates = _target_candidates(search, capacities, min_steps, max_steps)
        if not target_candidates:
            continue

        target, solution = rng.choice(target_candidates)
        prompt = _format_prompt(capacities, target, language=language, recipe=recipe, surface_plan=surface_plan)
        return JugPuzzle(
            capacities=capacities,
            target=target,
            solution=solution,
            prompt=prompt,
        )

    raise AssertionError("failed to generate jug puzzle")


def _target_candidates(
    search: dict[str, object],
    capacities: tuple[int, ...],
    min_steps: int,
    max_steps: int,
) -> list[tuple[int, tuple[JugAction, ...]]]:
    candidates: list[tuple[int, tuple[JugAction, ...]]] = []
    for target in range(1, max(capacities) + 1):
        summary = _target_summary(search, target)
        if summary is None:
            continue
        if summary["solution_count"] != 1:
            continue
        steps = int(summary["distance"])
        if not min_steps <= steps <= max_steps:
            continue
        candidates.append((target, summary["actions"]))
    return candidates


def _state_space(capacities: tuple[int, ...]) -> int:
    total = 1
    for capacity in capacities:
        total *= capacity + 1
    return total


def _target_summary(search: dict[str, object], target: int) -> dict[str, object] | None:
    distances: dict[tuple[int, ...], int] = search["distances"]
    path_counts: dict[tuple[int, ...], int] = search["path_counts"]
    parents: dict[tuple[int, ...], tuple[int, ...] | None] = search["parents"]
    parent_actions: dict[tuple[int, ...], JugAction | None] = search["parent_actions"]

    goal_states = [state for state in distances if target in state]
    if not goal_states:
        return None

    distance = min(distances[state] for state in goal_states)
    shortest_states = [state for state in goal_states if distances[state] == distance]
    solution_count = min(2, sum(path_counts[state] for state in shortest_states))
    if solution_count != 1:
        return {
            "distance": distance,
            "solution_count": solution_count,
            "actions": (),
        }

    goal_state = shortest_states[0]
    return {
        "distance": distance,
        "solution_count": solution_count,
        "actions": _reconstruct_actions(goal_state, parents, parent_actions),
    }


def _search(capacities: tuple[int, ...]) -> dict[str, object]:
    start = tuple(0 for _ in capacities)
    distances: dict[tuple[int, ...], int] = {start: 0}
    path_counts: dict[tuple[int, ...], int] = {start: 1}
    parents: dict[tuple[int, ...], tuple[int, ...] | None] = {start: None}
    parent_actions: dict[tuple[int, ...], JugAction | None] = {start: None}
    queue: deque[tuple[int, ...]] = deque([start])

    while queue:
        state = queue.popleft()
        for action in _ordered_actions(state, capacities):
            next_state = _apply_action(state, capacities, action)
            assert next_state is not None
            next_distance = distances[state] + 1

            if next_state not in distances:
                distances[next_state] = next_distance
                path_counts[next_state] = path_counts[state]
                parents[next_state] = state
                parent_actions[next_state] = action
                queue.append(next_state)
                continue

            if distances[next_state] != next_distance:
                continue

            path_counts[next_state] = min(2, path_counts[next_state] + path_counts[state])

    return {
        "distances": distances,
        "path_counts": path_counts,
        "parents": parents,
        "parent_actions": parent_actions,
    }


def _ordered_actions(state: tuple[int, ...], capacities: tuple[int, ...]) -> list[JugAction]:
    actions: list[JugAction] = []
    for index, amount in enumerate(state):
        if amount < capacities[index]:
            actions.append(JugAction("fill", index))
    for index, amount in enumerate(state):
        if amount > 0:
            actions.append(JugAction("empty", index))
    for source, amount in enumerate(state):
        if amount == 0:
            continue
        for target, target_amount in enumerate(state):
            if source == target:
                continue
            if target_amount >= capacities[target]:
                continue
            actions.append(JugAction("pour", source, target))
    return actions


def _apply_action(
    state: tuple[int, ...],
    capacities: tuple[int, ...],
    action: JugAction,
) -> tuple[int, ...] | None:
    values = list(state)

    if action.kind == "fill":
        if values[action.source] >= capacities[action.source]:
            return None
        values[action.source] = capacities[action.source]
        return tuple(values)

    if action.kind == "empty":
        if values[action.source] == 0:
            return None
        values[action.source] = 0
        return tuple(values)

    if action.kind == "pour":
        assert action.target is not None
        if values[action.source] == 0:
            return None
        space = capacities[action.target] - values[action.target]
        if space == 0:
            return None
        transferred = min(values[action.source], space)
        values[action.source] -= transferred
        values[action.target] += transferred
        return tuple(values)

    raise AssertionError(f"unsupported jug action: {action.kind}")


def _reconstruct_actions(
    state: tuple[int, ...],
    parents: dict[tuple[int, ...], tuple[int, ...] | None],
    parent_actions: dict[tuple[int, ...], JugAction | None],
) -> tuple[JugAction, ...]:
    actions: list[JugAction] = []
    current = state
    while parents[current] is not None:
        action = parent_actions[current]
        assert action is not None
        actions.append(action)
        current = parents[current]
    actions.reverse()
    return tuple(actions)


def _format_prompt(
    capacities: tuple[int, ...],
    target: int,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    prompt_style = str(recipe["prompt_style"])
    surface_spec = SURFACE_SPECS[prompt_style]
    blocks = {
        "intro": _intro_block(len(capacities), language, recipe, surface_plan),
        "capacities": _capacities_block(capacities, language, surface_plan),
        "rules": _rules_block(target, language, surface_plan),
        "instruction": _instruction_block(target, language, surface_plan),
        "answer": _answer_block(language, surface_plan),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(
    jug_count: int,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_intro"])
    last_index = jug_count - 1

    if language == "da":
        if variant == "context":
            lines = (
                f"Du får {jug_count} kander med numrene 0 til {last_index}, og de starter alle tomme.",
            )
        elif variant == "assignment":
            lines = (
                f"Bestem en korteste handlingssekvens for {jug_count} tomme kander med numrene 0 til {last_index}.",
            )
        else:
            lines = (
                f"Løs en kande-opgave med {jug_count} tomme kander.",
                f"Kanderne er nummereret 0 til {last_index}.",
            )
        return prompt_surface.PromptBlock(key="intro", lines=lines)

    if variant == "context":
        lines = (
            f"You are given {jug_count} jugs numbered 0 to {last_index}, all initially empty.",
        )
    elif variant == "assignment":
        lines = (
            f"Determine a shortest action sequence for {jug_count} empty jugs numbered 0 to {last_index}.",
        )
    else:
        lines = (
            f"Solve a jug puzzle with {jug_count} empty jugs.",
            f"The jug indices run from 0 to {last_index}.",
        )
    return prompt_surface.PromptBlock(key="intro", lines=lines)


def _capacities_block(
    capacities: tuple[int, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])

    if language == "da":
        if clue_style == "compact":
            lines = (
                f"Kapaciteter i liter: [{', '.join(str(capacity) for capacity in capacities)}]",
                "Positionerne i listen svarer til kande-numrene.",
            )
        elif clue_style == "deductive":
            lines = (
                "Kapaciteter efter kande-nummer:",
                ", ".join(
                    f"{index}->{capacity}"
                    for index, capacity in enumerate(capacities)
                ),
                "Alle kander starter med 0 liter.",
            )
        else:
            lines = tuple(
                f"Kande {index} kan rumme {capacity} liter."
                for index, capacity in enumerate(capacities)
            )
        return prompt_surface.PromptBlock(key="capacities", heading="Kapaciteter", lines=lines)

    if clue_style == "compact":
        lines = (
            f"Capacities in liters: [{', '.join(str(capacity) for capacity in capacities)}]",
            "The list positions match the jug indices.",
        )
    elif clue_style == "deductive":
        lines = (
            "Capacities by jug index:",
            ", ".join(
                f"{index}->{capacity}"
                for index, capacity in enumerate(capacities)
            ),
            "Each jug starts with 0 liters.",
        )
    else:
        lines = tuple(
            f"Jug {index} has capacity {capacity} liters."
            for index, capacity in enumerate(capacities)
        )
    return prompt_surface.PromptBlock(key="capacities", heading="Capacities", lines=lines)


def _rules_block(target: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    target_text = _liters_text(target, language)

    if language == "da":
        if clue_style == "compact":
            lines = (
                f"Få præcis {target_text} i én valgfri kande.",
                "Tilladte kommandoer: `Fill i`, `Empty i`, `Pour i j`.",
            )
        elif clue_style == "deductive":
            lines = (
                "Starttilstand: alle kander indeholder 0 liter.",
                f"Målet er at ende med præcis {target_text} i mindst én kande.",
                "De eneste tilladte kommandoer er `Fill i`, `Empty i` og `Pour i j`.",
            )
        else:
            lines = (
                f"Du skal ende med præcis {target_text} vand i en valgfri kande.",
                "`Fill i` fylder kande i helt op.",
                "`Empty i` tømmer kande i helt.",
                "`Pour i j` hælder fra kande i til kande j, indtil i er tom eller j er fuld.",
            )
        return prompt_surface.PromptBlock(key="rules", lines=lines)

    if clue_style == "compact":
        lines = (
            f"Get exactly {target_text} in any jug.",
            "Allowed commands: `Fill i`, `Empty i`, `Pour i j`.",
        )
    elif clue_style == "deductive":
        lines = (
            "Initial state: every jug contains 0 liters.",
            f"The goal is to finish with exactly {target_text} in at least one jug.",
            "The only allowed commands are `Fill i`, `Empty i`, and `Pour i j`.",
        )
    else:
        lines = (
            f"You must finish with exactly {target_text} of water in any one jug.",
            "`Fill i` fills jug i to capacity.",
            "`Empty i` empties jug i completely.",
            "`Pour i j` pours from jug i to jug j until i is empty or j is full.",
        )
    return prompt_surface.PromptBlock(key="rules", lines=lines)


def _instruction_block(target: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_instruction"])
    target_text = _liters_text(target, language)

    if language == "da":
        if variant == "solve":
            line = f"Find en korteste gyldig sekvens af handlinger, der giver præcis {target_text}."
        elif variant == "unique":
            line = "Den korteste gyldige løsning er entydig."
        else:
            line = "Alle oplysninger skal passe med den samme korteste handlingssekvens."
        return prompt_surface.PromptBlock(key="instruction", lines=(line,))

    if variant == "solve":
        line = f"Find a shortest valid sequence of actions that produces exactly {target_text}."
    elif variant == "unique":
        line = "The shortest valid solution is unique."
    else:
        line = "All information must fit the same shortest action sequence."
    return prompt_surface.PromptBlock(key="instruction", lines=(line,))


def _answer_block(language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_answer"])
    example = "Fill 0\nPour 0 1\nEmpty 1"

    if language == "da":
        if variant == "respond":
            lines = (
                "Inde i det endelige svar skal du skrive én kommando per linje.",
                "Brug præcis kommandoerne `Fill i`, `Empty i` og `Pour i j`.",
            )
        elif variant == "write":
            lines = (
                "Inde i det endelige svar skal hver linje være en rå kommando.",
                "Brug kun kande-numrene fra opgaven.",
            )
        else:
            lines = (
                "Selve svarindholdet skal være hele handlingssekvensen med én kommando per linje.",
                "Inde i svarindholdet må der ikke være bullets, nummerering eller ekstra tekst.",
            )
        lines += (f"Format:\n{example}",)
        return prompt_surface.PromptBlock(key="answer", lines=lines)

    if variant == "respond":
        lines = (
            "Inside the final response, write one command per line.",
            "Use exactly the commands `Fill i`, `Empty i`, and `Pour i j`.",
        )
    elif variant == "write":
        lines = (
            "Inside the final response, each line should be one raw command.",
            "Use only jug indices from the puzzle.",
        )
    else:
        lines = (
            "The answer content should be the full action sequence with one command per line.",
            "Use only the allowed commands.",
        )
    lines += (f"Format:\n{example}",)
    return prompt_surface.PromptBlock(key="answer", lines=lines)


def _render_action(action: JugAction) -> str:
    if action.kind == "fill":
        return f"Fill {action.source}"
    if action.kind == "empty":
        return f"Empty {action.source}"
    if action.kind == "pour":
        assert action.target is not None
        return f"Pour {action.source} {action.target}"
    raise AssertionError(f"unsupported jug action: {action.kind}")


def _action_from_dict(payload: dict[str, object]) -> JugAction:
    return JugAction(
        kind=str(payload["kind"]),
        source=int(payload["source"]),
        target=None if payload["target"] is None else int(payload["target"]),
    )


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]


def _liters_text(amount: int, language: str) -> str:
    if language == "da":
        return f"{amount} liter"
    if amount == 1:
        return "1 liter"
    return f"{amount} liters"
