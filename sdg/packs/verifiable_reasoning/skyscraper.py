from collections import Counter
from dataclasses import dataclass
from functools import cache
from itertools import permutations
from random import Random
from typing import Any

import z3

from sdg.commons import diversity, prompt_surface, z3_solver


@dataclass(frozen=True)
class SkyscraperClue:
    kind: str
    index: int
    value: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "index": self.index,
            "value": self.value,
        }


@dataclass(frozen=True)
class SkyscraperSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class SkyscraperPuzzle:
    size: int
    solution_grid: tuple[tuple[int, ...], ...]
    clues: tuple[SkyscraperClue, ...]
    prompt: str


SURFACE_SPECS = {
    "board": SkyscraperSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="board",
            block_order=("intro", "rules", "clues", "instruction", "answer"),
        ),
    ),
    "briefing": SkyscraperSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "instruction", "rules", "clues", "answer"),
        ),
    ),
    "deduce": SkyscraperSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "rules", "answer", "instruction", "clues"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_board_4x4",
        "difficulty": "easy",
        "prompt_style": "board",
        "size": 4,
        "clue_profile": "balanced_dense",
        "target_clue_count": 10,
        "max_clue_count": 10,
        "required_kind_counts": {"top": 2, "bottom": 2, "left": 3, "right": 3},
        "kind_priority": {"top": 0, "right": 1, "bottom": 2, "left": 3},
        "selection_attempts": 48,
    },
    {
        "recipe_id": "easy_briefing_4x4",
        "difficulty": "easy",
        "prompt_style": "briefing",
        "size": 4,
        "clue_profile": "row_heavy",
        "target_clue_count": 9,
        "max_clue_count": 9,
        "required_kind_counts": {"top": 2, "bottom": 2, "left": 2, "right": 3},
        "kind_priority": {"left": 0, "right": 1, "top": 2, "bottom": 3},
        "selection_attempts": 48,
    },
    {
        "recipe_id": "medium_board_4x4",
        "difficulty": "medium",
        "prompt_style": "board",
        "size": 4,
        "clue_profile": "balanced",
        "target_clue_count": 8,
        "max_clue_count": 8,
        "required_kind_counts": {"top": 2, "bottom": 2, "left": 2, "right": 2},
        "kind_priority": {"top": 0, "bottom": 1, "left": 2, "right": 3},
        "selection_attempts": 64,
    },
    {
        "recipe_id": "hard_deduce_4x4",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "size": 4,
        "clue_profile": "sparse",
        "target_clue_count": 7,
        "max_clue_count": 8,
        "required_kind_counts": {"top": 2, "bottom": 2, "left": 1, "right": 1},
        "kind_priority": {"top": 0, "bottom": 1, "left": 2, "right": 3},
        "selection_attempts": 80,
    },
    {
        "recipe_id": "medium_briefing_5x5",
        "difficulty": "medium",
        "prompt_style": "briefing",
        "size": 5,
        "clue_profile": "balanced_dense",
        "target_clue_count": 12,
        "max_clue_count": 13,
        "required_kind_counts": {"top": 3, "bottom": 3, "left": 3, "right": 3},
        "kind_priority": {"top": 0, "right": 1, "bottom": 2, "left": 3},
        "selection_attempts": 96,
    },
    {
        "recipe_id": "hard_board_5x5",
        "difficulty": "hard",
        "prompt_style": "board",
        "size": 5,
        "clue_profile": "column_heavy",
        "target_clue_count": 11,
        "max_clue_count": 12,
        "required_kind_counts": {"top": 3, "bottom": 3, "left": 2, "right": 2},
        "kind_priority": {"top": 0, "bottom": 1, "left": 2, "right": 3},
        "selection_attempts": 128,
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
    clue_counts = Counter(clue.kind for clue in puzzle.clues)

    return {
        "id": f"verifiable-reasoning-{index:05d}",
        "prompt": puzzle.prompt,
        "hidden": {
            "size": puzzle.size,
            "solution_grid": [list(row) for row in puzzle.solution_grid],
            "clues": [clue.to_dict() for clue in puzzle.clues],
        },
        "sources": [{"kind": "dolci_subset", "value": "skyscraperpuzzle"}],
        "meta": {
            "family": "skyscraper_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": len(puzzle.clues),
            "size": puzzle.size,
            "output_format": "number_grid",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            "clue_profile": recipe["clue_profile"],
            "top_clues": clue_counts["top"],
            "right_clues": clue_counts["right"],
            "bottom_clues": clue_counts["bottom"],
            "left_clues": clue_counts["left"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[tuple[int, ...], ...] | None:
    size = int(hidden["size"])
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if len(rows) != size:
        return None

    parsed_rows: list[tuple[int, ...]] = []
    for line in rows:
        tokens = line.replace(",", " ").split()
        if len(tokens) != size:
            return None
        if not all(token.isdigit() for token in tokens):
            return None
        numbers = tuple(int(token) for token in tokens)
        if any(number < 1 or number > size for number in numbers):
            return None
        parsed_rows.append(numbers)

    return tuple(parsed_rows)


def canonical_target(parsed: tuple[tuple[int, ...], ...], hidden: dict[str, object]) -> str:
    return format_target(parsed)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    size = int(hidden["size"])
    example = " ".join(str(number) for number in range(1, size + 1))

    if language == "da":
        return (
            "I din svarblok skal den endelige løsning være det udfyldte gitter med én række per linje.\n"
            f"Brug præcis {size} tal på hver linje, adskilt af mellemrum.\n"
            f"Format:\n{example}"
        )

    return (
        "In your answer block, the final solution should be the completed grid with one row per line.\n"
        f"Use exactly {size} digits on each line, separated by spaces.\n"
        f"Format:\n{example}"
    )


def format_target(grid: tuple[tuple[int, ...], ...]) -> str:
    return "\n".join(" ".join(str(value) for value in row) for row in grid)


def is_correct(parsed: tuple[tuple[int, ...], ...], hidden: dict[str, object]) -> bool:
    return [list(row) for row in parsed] == hidden["solution_grid"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    size = int(hidden["size"])
    clues = [clue_from_dict(payload) for payload in hidden["clues"]]
    solutions = _solve_grids(size, clues, limit=2)
    if len(solutions) != 1:
        return False
    return [list(row) for row in solutions[0]] == hidden["solution_grid"]


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    clue_kind_counts: Counter[str] = Counter()
    required_minimums: Counter[str] = Counter()

    for row in rows:
        for clue in row["hidden"]["clues"]:
            clue_kind_counts[str(clue["kind"])] += 1

    for item in planned:
        recipe = _recipe_by_id(str(item["recipe_id"]))
        for kind, minimum in recipe["required_kind_counts"].items():
            required_minimums[kind] += minimum

    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "prompt_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("prompt_style",), observed_getter=_meta_getter),
        "surface_intro_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_intro",), observed_getter=_meta_getter),
        "surface_instruction_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_instruction",), observed_getter=_meta_getter),
        "surface_answer_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_answer",), observed_getter=_meta_getter),
        "surface_clue_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_clue",), observed_getter=_meta_getter),
        "size_coverage": diversity.compare_planned_to_observed(planned, rows, ("size",), observed_getter=_meta_getter),
        "clue_kind_minimums": diversity.counter_minimum_check(clue_kind_counts, dict(required_minimums)),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def clue_from_dict(payload: dict[str, str | int]) -> SkyscraperClue:
    return SkyscraperClue(
        kind=str(payload["kind"]),
        index=int(payload["index"]),
        value=int(payload["value"]),
    )


def _generate_puzzle(
    rng: Random,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> SkyscraperPuzzle:
    size = int(recipe["size"])

    for _ in range(200):
        solution_grid = _sample_solution(size, rng)
        clues = _select_clues(solution_grid, recipe, rng)
        if clues is None:
            continue
        prompt = _format_prompt(
            size,
            clues,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return SkyscraperPuzzle(
            size=size,
            solution_grid=solution_grid,
            clues=clues,
            prompt=prompt,
        )

    raise AssertionError("failed to generate skyscraper puzzle")


def _sample_solution(size: int, rng: Random) -> tuple[tuple[int, ...], ...]:
    base = [
        [((row + col) % size) + 1 for col in range(size)]
        for row in range(size)
    ]
    row_perm = rng.sample(range(size), k=size)
    col_perm = rng.sample(range(size), k=size)
    symbol_perm = rng.sample(range(1, size + 1), k=size)

    grid = []
    for row in range(size):
        values = []
        for col in range(size):
            base_value = base[row_perm[row]][col_perm[col]]
            values.append(symbol_perm[base_value - 1])
        grid.append(tuple(values))
    return tuple(grid)


def _select_clues(
    solution_grid: tuple[tuple[int, ...], ...],
    recipe: dict[str, Any],
    rng: Random,
) -> tuple[SkyscraperClue, ...] | None:
    size = len(solution_grid)
    candidates = _ordered_clues(_build_candidates(solution_grid), recipe["kind_priority"], rng)
    if not _has_unique_solution(size, candidates):
        return None

    target_clue_count = int(recipe["target_clue_count"])
    max_clue_count = int(recipe["max_clue_count"])
    required_kind_counts = dict(recipe["required_kind_counts"])
    selection_attempts = int(recipe.get("selection_attempts", 48))

    for _ in range(selection_attempts):
        selected = list(candidates)
        removal_order = list(candidates)
        rng.shuffle(removal_order)

        for clue in removal_order:
            if len(selected) <= target_clue_count:
                break
            if _would_break_minimum(selected, clue, required_kind_counts):
                continue
            reduced = [item for item in selected if item != clue]
            if _has_unique_solution(size, reduced):
                selected = reduced

        if len(selected) > max_clue_count:
            continue
        if not _meets_kind_minimums(selected, required_kind_counts):
            continue
        return tuple(_ordered_clues(selected, recipe["kind_priority"], rng))

    return None


def _build_candidates(solution_grid: tuple[tuple[int, ...], ...]) -> list[SkyscraperClue]:
    size = len(solution_grid)
    candidates: list[SkyscraperClue] = []

    for row in range(size):
        values = solution_grid[row]
        candidates.append(SkyscraperClue(kind="left", index=row + 1, value=_visible_count(values)))
        candidates.append(SkyscraperClue(kind="right", index=row + 1, value=_visible_count(tuple(reversed(values)))))

    for col in range(size):
        values = tuple(solution_grid[row][col] for row in range(size))
        candidates.append(SkyscraperClue(kind="top", index=col + 1, value=_visible_count(values)))
        candidates.append(SkyscraperClue(kind="bottom", index=col + 1, value=_visible_count(tuple(reversed(values)))))

    return candidates


def _solve_grids(
    size: int,
    clues: list[SkyscraperClue] | tuple[SkyscraperClue, ...],
    *,
    limit: int | None = None,
) -> list[tuple[tuple[int, ...], ...]]:
    solver, variables, _ = _build_solver(size, list(clues))
    assignments = z3_solver.enumerate_int_models(solver, variables, limit=limit)
    return [_grid_from_assignment(size, assignment) for assignment in assignments]


def _solve_grid_count(
    size: int,
    clues: list[SkyscraperClue],
    *,
    limit: int | None = None,
) -> tuple[int, bool]:
    solver, variables, _ = _build_solver(size, clues)
    return z3_solver.count_int_models(solver, variables, limit=limit)


def _has_unique_solution(size: int, clues: list[SkyscraperClue]) -> bool:
    count, complete = _solve_grid_count(size, clues, limit=2)
    return count == 1 and complete


def _build_solver(
    size: int,
    clues: list[SkyscraperClue],
) -> tuple[z3.Solver, dict[str, z3.ArithRef], dict[tuple[int, int], z3.ArithRef]]:
    solver = z3.Solver()
    variables: dict[str, z3.ArithRef] = {}
    cell_vars: dict[tuple[int, int], z3.ArithRef] = {}

    for row in range(1, size + 1):
        row_vars: list[z3.ArithRef] = []
        for col in range(1, size + 1):
            name = _var_key(row, col)
            variable = z3.Int(name)
            variables[name] = variable
            cell_vars[(row, col)] = variable
            row_vars.append(variable)
            solver.add(variable >= 1, variable <= size)
        solver.add(z3.Distinct(row_vars))

    for col in range(1, size + 1):
        solver.add(z3.Distinct([cell_vars[(row, col)] for row in range(1, size + 1)]))

    for clue in clues:
        solver.add(_clue_constraint(cell_vars, size, clue))

    return solver, variables, cell_vars


def _clue_constraint(
    cell_vars: dict[tuple[int, int], z3.ArithRef],
    size: int,
    clue: SkyscraperClue,
) -> z3.BoolRef:
    line = _line_vars(cell_vars, size, clue)
    patterns = _visibility_patterns(size, clue.value)
    clauses = [
        z3.And([variable == value for variable, value in zip(line, pattern, strict=True)])
        for pattern in patterns
    ]
    if len(clauses) == 1:
        return clauses[0]
    return z3.Or(clauses)


def _line_vars(
    cell_vars: dict[tuple[int, int], z3.ArithRef],
    size: int,
    clue: SkyscraperClue,
) -> tuple[z3.ArithRef, ...]:
    if clue.kind == "left":
        return tuple(cell_vars[(clue.index, col)] for col in range(1, size + 1))
    if clue.kind == "right":
        return tuple(cell_vars[(clue.index, col)] for col in range(size, 0, -1))
    if clue.kind == "top":
        return tuple(cell_vars[(row, clue.index)] for row in range(1, size + 1))
    if clue.kind == "bottom":
        return tuple(cell_vars[(row, clue.index)] for row in range(size, 0, -1))
    raise AssertionError(f"unsupported skyscraper clue kind: {clue.kind}")


def _grid_from_assignment(
    size: int,
    assignment: dict[str, int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(assignment[_var_key(row, col)] for col in range(1, size + 1))
        for row in range(1, size + 1)
    )


@cache
def _visibility_patterns(size: int, visible_count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        pattern
        for pattern in permutations(range(1, size + 1))
        if _visible_count(pattern) == visible_count
    )


def _visible_count(values: tuple[int, ...]) -> int:
    highest = 0
    count = 0
    for value in values:
        if value > highest:
            highest = value
            count += 1
    return count


def _ordered_clues(
    clues: list[SkyscraperClue],
    kind_priority: dict[str, int],
    rng: Random,
) -> list[SkyscraperClue]:
    ordered = list(clues)
    rng.shuffle(ordered)
    ordered.sort(key=lambda clue: (kind_priority[clue.kind], clue.index, clue.value))
    return ordered


def _would_break_minimum(
    selected: list[SkyscraperClue],
    clue: SkyscraperClue,
    required_kind_counts: dict[str, int],
) -> bool:
    current = sum(item.kind == clue.kind for item in selected)
    return current <= required_kind_counts.get(clue.kind, 0)


def _meets_kind_minimums(
    clues: list[SkyscraperClue],
    required_kind_counts: dict[str, int],
) -> bool:
    counts = Counter(clue.kind for clue in clues)
    for kind, minimum in required_kind_counts.items():
        if counts[kind] < minimum:
            return False
    return True


def _format_prompt(
    size: int,
    clues: tuple[SkyscraperClue, ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    clue_lines, clue_heading, clue_numbered = _clue_block(
        clues,
        size=size,
        language=language,
        clue_style=str(surface_plan["surface_clue"]),
    )

    blocks = {
        "intro": prompt_surface.PromptBlock(
            "intro",
            _intro_lines(
                language=language,
                size=size,
                intro_variant=str(surface_plan["surface_intro"]),
                prompt_style=recipe["prompt_style"],
            ),
        ),
        "rules": prompt_surface.PromptBlock(
            "rules",
            _rule_lines(language=language, size=size),
            heading="Regler" if language == "da" else "Rules",
        ),
        "clues": prompt_surface.PromptBlock(
            "clues",
            clue_lines,
            heading=clue_heading,
            numbered=clue_numbered,
        ),
        "instruction": prompt_surface.PromptBlock(
            "instruction",
            (_instruction_line(language=language, instruction_variant=str(surface_plan["surface_instruction"])),),
        ),
        "answer": prompt_surface.PromptBlock(
            "answer",
            _answer_lines(language=language, size=size, answer_variant=str(surface_plan["surface_answer"])),
        ),
    }
    return prompt_surface.render_prompt(blocks, SURFACE_SPECS[recipe["prompt_style"]].plan)


def _intro_lines(
    *,
    language: str,
    size: int,
    intro_variant: str,
    prompt_style: str,
) -> tuple[str, ...]:
    if language == "da":
        if intro_variant == "context":
            first = f"Dette er en skyskraber-opgave med et {size}x{size}-gitter."
        elif intro_variant == "assignment":
            first = f"Udfyld et skyskraber-gitter på {size}x{size} felter."
        else:
            first = f"Løs en skyskraber-opgave på et {size}x{size}-gitter."
        indexing = "Rækker tælles oppefra og ned, og kolonner tælles fra venstre mod højre."
        if prompt_style == "deduce":
            return ("Brug kanttallene til at udlede hele gitteret.", first, indexing)
        return (first, indexing)

    if intro_variant == "context":
        first = f"This is a skyscraper puzzle on a {size}x{size} grid."
    elif intro_variant == "assignment":
        first = f"Fill a {size}x{size} skyscraper grid."
    else:
        first = f"Solve a skyscraper puzzle on a {size}x{size} grid."
    indexing = "Rows are counted from top to bottom, and columns from left to right."
    if prompt_style == "deduce":
        return ("Use the border clues to work out the full grid.", first, indexing)
    return (first, indexing)


def _rule_lines(*, language: str, size: int) -> tuple[str, ...]:
    if language == "da":
        return (
            f"Brug tallene 1 til {size}.",
            "Hver række skal indeholde hvert tal præcis én gang.",
            "Hver kolonne skal indeholde hvert tal præcis én gang.",
            "Et større tal betyder en højere bygning.",
            "Et kanttal angiver, hvor mange bygninger der kan ses fra den side.",
        )

    return (
        f"Use the digits 1 to {size}.",
        "Each row must contain each digit exactly once.",
        "Each column must contain each digit exactly once.",
        "A larger number represents a taller building.",
        "A border clue tells you how many buildings are visible from that side.",
    )


def _instruction_line(*, language: str, instruction_variant: str) -> str:
    if language == "da":
        if instruction_variant == "solve":
            return "Der er præcis én gyldig løsning."
        if instruction_variant == "unique":
            return "Kanttallene bestemmer én entydig løsning."
        return "Alle regler og kanttal skal passe med den samme løsning."

    if instruction_variant == "solve":
        return "There is exactly one valid solution."
    if instruction_variant == "unique":
        return "The border clues determine one unique solution."
    return "The rules and border clues must all fit the same solution."


def _answer_lines(*, language: str, size: int, answer_variant: str) -> tuple[str, ...]:
    example_row = " ".join(str(number) for number in range(1, size + 1))
    example_grid = tuple(example_row for _ in range(size))

    if language == "da":
        first_line = {
            "respond": f"Inde i det endelige svar skal gitteret have præcis {size} linjer i dette format:",
            "write": f"Inde i det endelige svar skal du bruge præcis {size} linjer i dette format:",
            "complete": f"Selve svarindholdet skal være det udfyldte gitter med præcis {size} linjer i dette format:",
        }[answer_variant]
        return (first_line, *example_grid)

    first_line = {
        "respond": f"Inside the final response, the grid should use exactly {size} lines in this format:",
        "write": f"Inside the final response, use exactly {size} lines in this format:",
        "complete": f"The answer content should be the completed grid with exactly {size} lines in this format:",
    }[answer_variant]
    return (first_line, *example_grid)


def _clue_block(
    clues: tuple[SkyscraperClue, ...],
    *,
    size: int,
    language: str,
    clue_style: str,
) -> tuple[tuple[str, ...], str, bool]:
    if clue_style != "compact":
        return (
            tuple(_format_clue(clue, language=language, clue_style=clue_style) for clue in clues),
            "Kanttal" if language == "da" else "Border clues",
            True,
        )

    values_by_kind = {
        kind: ["."] * size
        for kind in ("top", "right", "bottom", "left")
    }
    for clue in clues:
        values_by_kind[clue.kind][clue.index - 1] = str(clue.value)

    if language == "da":
        labels = {
            "top": "Top",
            "right": "Højre",
            "bottom": "Bund",
            "left": "Venstre",
        }
    else:
        labels = {
            "top": "Top",
            "right": "Right",
            "bottom": "Bottom",
            "left": "Left",
        }

    lines = tuple(
        f"{labels[kind]}: {' '.join(values_by_kind[kind])}"
        for kind in ("top", "right", "bottom", "left")
    )
    return lines, "Kanttal" if language == "da" else "Border clues", False


def _format_clue(
    clue: SkyscraperClue,
    *,
    language: str,
    clue_style: str,
) -> str:
    building_word = _building_word(clue.value, language=language)

    if clue_style == "compact":
        return _format_compact_clue(clue, language=language)

    if language == "da":
        if clue.kind == "top":
            if clue_style == "deductive":
                return f"Set oppefra kan man se {clue.value} {building_word} i kolonne {clue.index}."
            return f"Fra toppen kan man se {clue.value} {building_word} i kolonne {clue.index}."
        if clue.kind == "bottom":
            if clue_style == "deductive":
                return f"Set nedefra kan man se {clue.value} {building_word} i kolonne {clue.index}."
            return f"Fra bunden kan man se {clue.value} {building_word} i kolonne {clue.index}."
        if clue.kind == "left":
            if clue_style == "deductive":
                return f"Set fra venstre kan man se {clue.value} {building_word} i række {clue.index}."
            return f"Fra venstre kan man se {clue.value} {building_word} i række {clue.index}."
        if clue_style == "deductive":
            return f"Set fra højre kan man se {clue.value} {building_word} i række {clue.index}."
        return f"Fra højre kan man se {clue.value} {building_word} i række {clue.index}."

    if clue.kind == "top":
        if clue_style == "deductive":
            return f"Looking from the top, column {clue.index} shows {clue.value} {building_word}."
        return f"From the top, column {clue.index} shows {clue.value} {building_word}."
    if clue.kind == "bottom":
        if clue_style == "deductive":
            return f"Looking from the bottom, column {clue.index} shows {clue.value} {building_word}."
        return f"From the bottom, column {clue.index} shows {clue.value} {building_word}."
    if clue.kind == "left":
        if clue_style == "deductive":
            return f"Looking from the left, row {clue.index} shows {clue.value} {building_word}."
        return f"From the left, row {clue.index} shows {clue.value} {building_word}."
    if clue_style == "deductive":
        return f"Looking from the right, row {clue.index} shows {clue.value} {building_word}."
    return f"From the right, row {clue.index} shows {clue.value} {building_word}."


def _format_compact_clue(clue: SkyscraperClue, *, language: str) -> str:
    if language == "da":
        labels = {"top": "top", "right": "højre", "bottom": "bund", "left": "venstre"}
    else:
        labels = {"top": "top", "right": "right", "bottom": "bottom", "left": "left"}
    return f"{labels[clue.kind]} {clue.index} = {clue.value}"


def _building_word(value: int, *, language: str) -> str:
    if language == "da":
        return "bygning" if value == 1 else "bygninger"
    return "building" if value == 1 else "buildings"


def _recipe_by_id(recipe_id: str) -> dict[str, Any]:
    for recipe in RECIPES:
        if recipe["recipe_id"] == recipe_id:
            return recipe
    raise AssertionError(f"unknown skyscraper recipe: {recipe_id}")


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]


def _var_key(row: int, col: int) -> str:
    return f"r{row}c{col}"
