from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Any

from sdg.commons import diversity, prompt_surface


@dataclass(frozen=True)
class SetSplittingPuzzle:
    element_count: int
    constraints: tuple[tuple[int, ...], ...]
    solution: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class SetSplittingSurfaceSpec:
    plan: prompt_surface.SurfacePlan


SURFACE_SPECS = {
    "formal": SetSplittingSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="formal",
            block_order=("intro", "rules", "constraints", "instruction", "answer"),
        ),
    ),
    "instruction_first": SetSplittingSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="instruction_first",
            block_order=("intro", "instruction", "rules", "constraints", "answer"),
        ),
    ),
    "constraints_first": SetSplittingSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="constraints_first",
            block_order=("intro", "constraints", "rules", "instruction", "answer"),
        ),
    ),
}


RECIPES = (
    {
        "recipe_id": "easy_formal_5_pairs_triples",
        "difficulty": "easy",
        "prompt_style": "formal",
        "element_count": 5,
        "subset_sizes": (2, 3),
        "min_constraints": 4,
        "max_constraints": 5,
        "sample_attempts": 120,
    },
    {
        "recipe_id": "easy_instruction_first_6_triples",
        "difficulty": "easy",
        "prompt_style": "instruction_first",
        "element_count": 6,
        "subset_sizes": (2, 3),
        "min_constraints": 4,
        "max_constraints": 6,
        "sample_attempts": 140,
    },
    {
        "recipe_id": "medium_constraints_first_6_mixed",
        "difficulty": "medium",
        "prompt_style": "constraints_first",
        "element_count": 6,
        "subset_sizes": (2, 3, 4),
        "min_constraints": 5,
        "max_constraints": 7,
        "sample_attempts": 160,
    },
    {
        "recipe_id": "medium_formal_7_triples_quads",
        "difficulty": "medium",
        "prompt_style": "formal",
        "element_count": 7,
        "subset_sizes": (2, 3, 4),
        "min_constraints": 5,
        "max_constraints": 8,
        "sample_attempts": 180,
    },
    {
        "recipe_id": "hard_instruction_first_7_mixed",
        "difficulty": "hard",
        "prompt_style": "instruction_first",
        "element_count": 7,
        "subset_sizes": (2, 3, 4),
        "min_constraints": 7,
        "max_constraints": 9,
        "sample_attempts": 220,
    },
    {
        "recipe_id": "hard_constraints_first_8_mixed",
        "difficulty": "hard",
        "prompt_style": "constraints_first",
        "element_count": 8,
        "subset_sizes": (3, 4),
        "min_constraints": 7,
        "max_constraints": 10,
        "sample_attempts": 260,
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
            "element_count": puzzle.element_count,
            "constraints": [list(constraint) for constraint in puzzle.constraints],
            "solution": list(puzzle.solution),
        },
        "sources": [{"kind": "dolci_subset", "value": "setsplitting"}],
        "meta": {
            "family": "setsplitting_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": len(puzzle.constraints),
            "given_count": puzzle.element_count,
            "element_count": puzzle.element_count,
            "constraint_count": len(puzzle.constraints),
            "subset_size_profile": _subset_size_profile(puzzle.constraints),
            "output_format": "group_assignment",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[str, ...] | None:
    element_count = int(hidden["element_count"])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != element_count:
        return None

    assignments: dict[int, str] = {}
    for line in lines:
        normalized = line.strip().lstrip("*•- ").strip()
        left, sep, right = normalized.partition(":")
        if not sep:
            left, sep, right = normalized.partition("=")
        if not sep:
            pieces = normalized.split()
            if len(pieces) != 2:
                return None
            left, right = pieces
        index_text = left.strip()
        group_text = right.strip().upper()
        if not index_text.isdigit():
            return None
        if group_text not in {"A", "B"}:
            return None
        index = int(index_text)
        if not 1 <= index <= element_count:
            return None
        if index in assignments:
            return None
        assignments[index] = group_text

    if len(assignments) != element_count:
        return None
    if assignments[1] != "A":
        return None

    return tuple(assignments[index] for index in range(1, element_count + 1))


def canonical_target(parsed: tuple[str, ...], hidden: dict[str, object]) -> str:
    return format_target(parsed)


def format_target(solution: tuple[str, ...]) -> str:
    return "\n".join(f"{index}: {group}" for index, group in enumerate(solution, start=1))


def answer_contract(hidden: dict[str, object], language: str) -> str:
    element_count = int(hidden["element_count"])
    example = "\n".join(
        f"{index}: {'A' if index % 2 else 'B'}"
        for index in range(1, min(element_count, 4) + 1)
    )

    if language == "da":
        return (
            "I din svarblok skal du skrive én linje per element i rækkefølge.\n"
            "Brug kun etiketterne `A` og `B`, og hold element 1 i gruppe A.\n"
            f"Format:\n{example}"
        )

    return (
        "In your answer block, write one line per element in order.\n"
        "Use only the labels `A` and `B`, and keep element 1 in group A.\n"
        f"Format:\n{example}"
    )


def is_correct(parsed: tuple[str, ...], hidden: dict[str, object]) -> bool:
    expected = tuple(str(group) for group in hidden["solution"])
    return parsed == expected


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    element_count = int(hidden["element_count"])
    constraints = tuple(tuple(int(value) for value in constraint) for constraint in hidden["constraints"])
    solution = tuple(str(group) for group in hidden["solution"])
    satisfying = _satisfying_assignments(element_count, constraints)
    if len(satisfying) != 1:
        return False
    return satisfying[0] == solution


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "prompt_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("prompt_style",), observed_getter=_meta_getter),
        "element_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("element_count",), observed_getter=_meta_getter),
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
) -> SetSplittingPuzzle:
    element_count = int(recipe["element_count"])
    subset_sizes = tuple(int(size) for size in recipe["subset_sizes"])
    min_constraints = int(recipe["min_constraints"])
    max_constraints = int(recipe["max_constraints"])

    assignments = _all_assignments(element_count)
    target_assignments = [assignment for assignment in assignments if "B" in assignment]

    for _ in range(int(recipe["sample_attempts"])):
        solution = rng.choice(target_assignments)
        candidate_constraints = _candidate_constraints(solution, subset_sizes)
        rng.shuffle(candidate_constraints)
        constraints = _select_constraints(
            solution,
            assignments,
            candidate_constraints,
            min_constraints=min_constraints,
            max_constraints=max_constraints,
            rng=rng,
        )
        if constraints is None:
            continue
        if not _all_elements_referenced(element_count, constraints):
            continue

        prompt = _format_prompt(
            element_count,
            constraints,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return SetSplittingPuzzle(
            element_count=element_count,
            constraints=constraints,
            solution=solution,
            prompt=prompt,
        )

    raise AssertionError("failed to generate set splitting puzzle")


def _all_assignments(element_count: int) -> list[tuple[str, ...]]:
    assignments: list[tuple[str, ...]] = []
    for mask in range(1 << (element_count - 1)):
        groups = ["A"]
        for offset in range(element_count - 1):
            groups.append("B" if (mask >> offset) & 1 else "A")
        assignments.append(tuple(groups))
    return assignments


def _candidate_constraints(solution: tuple[str, ...], subset_sizes: tuple[int, ...]) -> list[tuple[int, ...]]:
    constraints: list[tuple[int, ...]] = []
    for size in subset_sizes:
        for subset in combinations(range(1, len(solution) + 1), size):
            if _constraint_satisfied(solution, subset):
                constraints.append(subset)
    return constraints


def _select_constraints(
    solution: tuple[str, ...],
    assignments: list[tuple[str, ...]],
    candidate_constraints: list[tuple[int, ...]],
    *,
    min_constraints: int,
    max_constraints: int,
    rng: Random,
) -> tuple[tuple[int, ...], ...] | None:
    chosen: list[tuple[int, ...]] = []
    current = list(assignments)
    remaining = list(candidate_constraints)

    while len(chosen) < max_constraints and len(current) > 1:
        best_constraint: tuple[int, ...] | None = None
        best_filtered: list[tuple[str, ...]] | None = None
        best_count = len(current)

        for constraint in remaining:
            filtered = [assignment for assignment in current if _constraint_satisfied(assignment, constraint)]
            if solution not in filtered:
                continue
            if len(filtered) >= best_count:
                continue
            best_constraint = constraint
            best_filtered = filtered
            best_count = len(filtered)

        if best_constraint is None or best_filtered is None:
            break

        chosen.append(best_constraint)
        remaining.remove(best_constraint)
        current = best_filtered

    if len(current) != 1:
        return None

    while len(chosen) < min_constraints:
        preserved = [constraint for constraint in remaining if _constraint_satisfied(solution, constraint)]
        if not preserved:
            return None
        extra = rng.choice(preserved)
        chosen.append(extra)
        remaining.remove(extra)

    chosen.sort()
    return tuple(chosen)


def _satisfying_assignments(
    element_count: int,
    constraints: tuple[tuple[int, ...], ...],
) -> list[tuple[str, ...]]:
    return [
        assignment
        for assignment in _all_assignments(element_count)
        if all(_constraint_satisfied(assignment, constraint) for constraint in constraints)
    ]


def _constraint_satisfied(assignment: tuple[str, ...], constraint: tuple[int, ...]) -> bool:
    groups = {assignment[element - 1] for element in constraint}
    return len(groups) == 2


def _all_elements_referenced(element_count: int, constraints: tuple[tuple[int, ...], ...]) -> bool:
    seen = {element for constraint in constraints for element in constraint}
    return len(seen) == element_count


def _format_prompt(
    element_count: int,
    constraints: tuple[tuple[int, ...], ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    prompt_style = str(recipe["prompt_style"])
    surface_spec = SURFACE_SPECS[prompt_style]
    blocks = {
        "intro": _intro_block(element_count, language, prompt_style, surface_plan),
        "rules": _rules_block(language, prompt_style, surface_plan),
        "constraints": _constraints_block(constraints, language, prompt_style, surface_plan),
        "instruction": _instruction_block(language, prompt_style, surface_plan),
        "answer": _answer_block(element_count, language, prompt_style, surface_plan),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(
    element_count: int,
    language: str,
    prompt_style: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_intro"])

    if language == "da":
        element_span = f"elementerne 1 til {element_count}"
        if prompt_style == "constraints_first":
            if variant == "context":
                lines = (f"Nedenfor får du delmængder af {element_span}.",)
            elif variant == "assignment":
                lines = (f"Ud fra de opgivne delmængder skal {element_span} deles mellem A og B.",)
            else:
                lines = (f"Brug delmængderne nedenfor til at dele {element_span} mellem A og B.",)
        elif prompt_style == "instruction_first":
            if variant == "context":
                lines = (f"Du får {element_count} nummererede elementer, som skal fordeles mellem A og B.",)
            elif variant == "assignment":
                lines = (f"Fordel {element_span} mellem gruppe A og gruppe B.",)
            else:
                lines = (f"Løs en opdeling af {element_span} i grupperne A og B.",)
        elif variant == "context":
            lines = (f"{element_span.capitalize()} skal deles mellem gruppe A og B.",)
        elif variant == "assignment":
            lines = (f"Bestem en entydig opdeling af {element_span} i grupperne A og B.",)
        else:
            lines = (f"Del {element_span} i to grupper, A og B.",)
        return prompt_surface.PromptBlock(key="intro", lines=lines)

    element_span = f"elements 1 to {element_count}"
    if prompt_style == "constraints_first":
        if variant == "context":
            lines = (f"Below is a collection of subsets over {element_span}.",)
        elif variant == "assignment":
            lines = (f"Use the listed subsets to split {element_span} between groups A and B.",)
        else:
            lines = (f"Split {element_span} between groups A and B using the subsets below.",)
    elif prompt_style == "instruction_first":
        if variant == "context":
            lines = (f"You are given {element_count} numbered elements that must be split between A and B.",)
        elif variant == "assignment":
            lines = (f"Split {element_span} between groups A and B.",)
        else:
            lines = (f"Solve a split of {element_span} into groups A and B.",)
    elif variant == "context":
        lines = (f"The {element_span} must be split between groups A and B.",)
    elif variant == "assignment":
        lines = (f"Determine a unique split of {element_span} into groups A and B.",)
    else:
        lines = (f"Partition {element_span} into two groups, A and B.",)
    return prompt_surface.PromptBlock(key="intro", lines=lines)


def _rules_block(
    language: str,
    prompt_style: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])

    if language == "da":
        if clue_style == "compact":
            lines = (
                "Hver opført mængde skal deles mellem A og B.",
                "Element 1 er fast i A.",
            )
        elif clue_style == "deductive":
            lines = (
                "Ingen af de opgivne delmængder må ende helt i den samme gruppe.",
                "Element 1 låser navnene på grupperne ved altid at høre til A.",
            )
        else:
            lines = (
                "Hver opført delmængde skal have mindst ét element i A og mindst ét element i B.",
                "Element 1 skal altid stå i A, så gruppenavnene er entydige.",
            )
        heading = "Regler" if prompt_style != "constraints_first" else "Opdelingsregler"
        return prompt_surface.PromptBlock(key="rules", heading=heading, lines=lines)

    if clue_style == "compact":
        lines = (
            "Every listed set must be split across A and B.",
            "Element 1 is fixed in A.",
        )
    elif clue_style == "deductive":
        lines = (
            "No listed subset may end up entirely inside one group.",
            "Element 1 stays in A to make the labeling canonical.",
        )
    else:
        lines = (
            "Each listed subset must contain at least one element in A and at least one element in B.",
            "Element 1 must always stay in A, which fixes the group labels.",
        )
    heading = "Rules" if prompt_style != "constraints_first" else "Split Rules"
    return prompt_surface.PromptBlock(key="rules", heading=heading, lines=lines)


def _constraints_block(
    constraints: tuple[tuple[int, ...], ...],
    language: str,
    prompt_style: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])

    if clue_style == "compact":
        lines = tuple(_render_set(constraint) for constraint in constraints)
        numbered = False
    elif clue_style == "deductive":
        lines = tuple(_deductive_constraint_text(constraint, language) for constraint in constraints)
        numbered = True
    else:
        lines = tuple(_render_set(constraint) for constraint in constraints)
        numbered = True

    if language == "da":
        if clue_style == "deductive":
            heading = "Krav"
        elif prompt_style == "constraints_first":
            heading = "Opgivne delmængder"
        else:
            heading = "Delmængder"
    else:
        if clue_style == "deductive":
            heading = "Constraints"
        elif prompt_style == "constraints_first":
            heading = "Given Subsets"
        else:
            heading = "Subsets"

    return prompt_surface.PromptBlock(
        key="constraints",
        heading=heading,
        lines=lines,
        numbered=numbered,
    )


def _instruction_block(
    language: str,
    prompt_style: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_instruction"])

    if language == "da":
        if variant == "solve":
            if prompt_style == "constraints_first":
                line = "Find den eneste opdeling, der passer til alle de opgivne delmængder."
            else:
                line = "Find den eneste opdeling, der opfylder alle delmængderne."
        elif variant == "unique":
            line = "Med element 1 i A er der præcis én gyldig opdeling."
        else:
            if prompt_style == "instruction_first":
                line = "Alle oplysninger skal passe med den samme opdeling af elementerne."
            else:
                line = "Alle krav skal passe med den samme opdeling af elementerne."
        return prompt_surface.PromptBlock(key="instruction", lines=(line,))

    if variant == "solve":
        if prompt_style == "constraints_first":
            line = "Find the only split that fits all listed subsets."
        else:
            line = "Find the only split that satisfies all listed subsets."
    elif variant == "unique":
        line = "Once element 1 is fixed in A, there is exactly one valid split."
    else:
        if prompt_style == "instruction_first":
            line = "All information must fit the same split of the elements."
        else:
            line = "All constraints must fit the same split of the elements."
    return prompt_surface.PromptBlock(key="instruction", lines=(line,))


def _answer_block(
    element_count: int,
    language: str,
    prompt_style: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_answer"])
    example = "\n".join(
        f"{index}: {'A' if index % 2 else 'B'}"
        for index in range(1, min(element_count, 4) + 1)
    )

    if language == "da":
        if variant == "respond":
            lines = (
                "Inde i det endelige svar skal du skrive én linje per element i stigende orden.",
                "Brug kun `A` og `B` som gruppemærker.",
            )
        elif variant == "write":
            lines = (
                "Inde i det endelige svar skal hver linje have formen `element: gruppe`.",
                "Inde i svarindholdet må der ikke være ekstra tekst eller punkttegn.",
            )
        else:
            lines = (
                "Selve svarindholdet skal være hele opdelingen med én rå linje per element.",
                "Element 1 skal stå i A i det endelige svar.",
            )
        lines += (f"Format:\n{example}",)
        heading = "Svarformat" if prompt_style == "formal" else None
        return prompt_surface.PromptBlock(key="answer", heading=heading, lines=lines)

    if variant == "respond":
        lines = (
            "Inside the final response, write one line per element in ascending order.",
            "Use only `A` and `B` as the group labels.",
        )
    elif variant == "write":
        lines = (
            "Inside the final response, each line should have the form `element: group`.",
            "Inside the answer content, do not add extra text or bullets.",
        )
    else:
        lines = (
            "The answer content should be the full split with one raw line per element.",
            "Element 1 must appear in A in the final answer.",
        )
    lines += (f"Format:\n{example}",)
    heading = "Answer Format" if prompt_style == "formal" else None
    return prompt_surface.PromptBlock(key="answer", heading=heading, lines=lines)


def _render_set(constraint: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(value) for value in constraint) + "}"


def _deductive_constraint_text(constraint: tuple[int, ...], language: str) -> str:
    listed = _format_element_list(constraint, language)

    if len(constraint) == 2:
        if language == "da":
            return f"Parret {listed} må ikke ende i samme gruppe."
        return f"The pair {listed} may not end up in the same group."

    if language == "da":
        return f"Elementerne {listed} må ikke alle ende i samme gruppe."
    return f"The elements {listed} may not all end up in the same group."


def _format_element_list(values: tuple[int, ...], language: str) -> str:
    items = [str(value) for value in values]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        connector = " og " if language == "da" else " and "
        return f"{items[0]}{connector}{items[1]}"

    connector = " og " if language == "da" else " and "
    return f"{', '.join(items[:-1])}{connector}{items[-1]}"


def _subset_size_profile(constraints: tuple[tuple[int, ...], ...]) -> str:
    sizes = sorted({len(constraint) for constraint in constraints})
    return "+".join(str(size) for size in sizes)


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]
