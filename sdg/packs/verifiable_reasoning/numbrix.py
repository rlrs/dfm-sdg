from dataclasses import dataclass
from random import Random
from typing import Any

import z3

from sdg.commons import diversity, prompt_surface, z3_solver


@dataclass(frozen=True)
class NumbrixClue:
    row: int
    col: int
    value: int

    def to_dict(self) -> dict[str, int]:
        return {
            "row": self.row,
            "col": self.col,
            "value": self.value,
        }


@dataclass(frozen=True)
class NumbrixSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class NumbrixPuzzle:
    rows: int
    cols: int
    solution_grid: tuple[tuple[int, ...], ...]
    clues: tuple[NumbrixClue, ...]
    prompt: str
    path_style: str


SURFACE_SPECS = {
    "board": NumbrixSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="board",
            block_order=("intro", "rules", "givens", "instruction", "answer"),
        ),
    ),
    "briefing": NumbrixSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "instruction", "rules", "givens", "answer"),
        ),
    ),
    "deduce": NumbrixSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "rules", "answer", "instruction", "givens"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_rect_sparse_2x3",
        "difficulty": "easy",
        "prompt_style": "board",
        "rows": 2,
        "cols": 3,
        "clue_profile": "spread",
        "target_given_count": 3,
        "max_given_count": 4,
        "anchor_ratios": (0.0, 0.5, 1.0),
        "sample_attempts": 48,
        "path_styles": ("row_snake", "col_snake", "spiral"),
    },
    {
        "recipe_id": "easy_square_sparse_3x3",
        "difficulty": "easy",
        "prompt_style": "briefing",
        "rows": 3,
        "cols": 3,
        "clue_profile": "spread",
        "target_given_count": 4,
        "max_given_count": 5,
        "anchor_ratios": (0.0, 0.5, 1.0),
        "sample_attempts": 56,
        "path_styles": ("row_snake", "col_snake", "spiral"),
    },
    {
        "recipe_id": "medium_rect_balanced_3x4",
        "difficulty": "medium",
        "prompt_style": "briefing",
        "rows": 3,
        "cols": 4,
        "clue_profile": "balanced",
        "target_given_count": 5,
        "max_given_count": 6,
        "anchor_ratios": (0.0, 0.33, 0.66, 1.0),
        "sample_attempts": 72,
        "path_styles": ("row_snake", "col_snake", "spiral"),
    },
    {
        "recipe_id": "medium_tall_balanced_5x3",
        "difficulty": "medium",
        "prompt_style": "board",
        "rows": 5,
        "cols": 3,
        "clue_profile": "balanced",
        "target_given_count": 6,
        "max_given_count": 7,
        "anchor_ratios": (0.0, 0.25, 0.5, 0.75, 1.0),
        "sample_attempts": 84,
        "path_styles": ("row_snake", "col_snake", "spiral"),
    },
    {
        "recipe_id": "hard_square_balanced_4x4",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "rows": 4,
        "cols": 4,
        "clue_profile": "balanced",
        "target_given_count": 7,
        "max_given_count": 9,
        "anchor_ratios": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "sample_attempts": 96,
        "path_styles": ("row_snake", "col_snake", "spiral"),
    },
    {
        "recipe_id": "hard_rect_near_complete_3x6",
        "difficulty": "hard",
        "prompt_style": "board",
        "rows": 3,
        "cols": 6,
        "clue_profile": "near_complete",
        "target_given_count": 11,
        "max_given_count": 12,
        "anchor_ratios": (0.0, 0.25, 0.5, 0.75, 1.0),
        "sample_attempts": 120,
        "path_styles": ("row_snake", "col_snake", "spiral"),
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
            "rows": puzzle.rows,
            "cols": puzzle.cols,
            "solution_grid": [list(row) for row in puzzle.solution_grid],
            "clues": [clue.to_dict() for clue in puzzle.clues],
        },
        "sources": [{"kind": "dolci_subset", "value": "numbrix"}],
        "meta": {
            "family": "numbrix_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": len(puzzle.clues),
            "given_count": len(puzzle.clues),
            "rows": puzzle.rows,
            "cols": puzzle.cols,
            "cell_count": puzzle.rows * puzzle.cols,
            "output_format": "number_grid",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            "clue_profile": recipe["clue_profile"],
            "path_style": puzzle.path_style,
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[tuple[int, ...], ...] | None:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    max_value = rows * cols - 1
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != rows:
        return None

    parsed_rows: list[tuple[int, ...]] = []
    for line in lines:
        tokens = line.replace(",", " ").split()
        if len(tokens) != cols:
            return None
        if not all(token.lstrip("-").isdigit() for token in tokens):
            return None
        numbers = tuple(int(token) for token in tokens)
        if any(number < 0 or number > max_value for number in numbers):
            return None
        parsed_rows.append(numbers)

    return tuple(parsed_rows)


def canonical_target(parsed: tuple[tuple[int, ...], ...], hidden: dict[str, object]) -> str:
    return format_target(parsed)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    max_value = rows * cols - 1
    example = "\n".join(_example_grid(rows, cols))

    if language == "da":
        return (
            "I din svarblok skal den endelige løsning være det udfyldte gitter med én række per linje.\n"
            f"Brug præcis {cols} tal på hver linje. Tallene skal ligge mellem 0 og {max_value}.\n"
            f"Format:\n{example}"
        )

    return (
        "In your answer block, the final solution should be the completed grid with one row per line.\n"
        f"Use exactly {cols} numbers on each line. The numbers should range from 0 to {max_value}.\n"
        f"Format:\n{example}"
    )


def format_target(grid: tuple[tuple[int, ...], ...]) -> str:
    return "\n".join(" ".join(str(value) for value in row) for row in grid)


def is_correct(parsed: tuple[tuple[int, ...], ...], hidden: dict[str, object]) -> bool:
    return [list(row) for row in parsed] == hidden["solution_grid"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    clues = [clue_from_dict(payload) for payload in hidden["clues"]]
    solutions = _solve_grids(rows, cols, clues, limit=2)
    if len(solutions) != 1:
        return False
    return [list(row) for row in solutions[0]] == hidden["solution_grid"]


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "prompt_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("prompt_style",), observed_getter=_meta_getter),
        "surface_intro_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_intro",), observed_getter=_meta_getter),
        "surface_instruction_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_instruction",), observed_getter=_meta_getter),
        "surface_answer_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_answer",), observed_getter=_meta_getter),
        "surface_clue_coverage": diversity.compare_planned_to_observed(planned, rows, ("surface_clue",), observed_getter=_meta_getter),
        "row_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("rows",), observed_getter=_meta_getter),
        "col_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("cols",), observed_getter=_meta_getter),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def clue_from_dict(payload: dict[str, int]) -> NumbrixClue:
    return NumbrixClue(
        row=int(payload["row"]),
        col=int(payload["col"]),
        value=int(payload["value"]),
    )


def _generate_puzzle(
    rng: Random,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> NumbrixPuzzle:
    rows = int(recipe["rows"])
    cols = int(recipe["cols"])

    for _ in range(200):
        path_style, solution_grid = _sample_solution(rows, cols, recipe, rng)
        clues = _select_clues(solution_grid, recipe, rng)
        if clues is None:
            continue
        prompt = _format_prompt(
            rows,
            cols,
            clues,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return NumbrixPuzzle(
            rows=rows,
            cols=cols,
            solution_grid=solution_grid,
            clues=clues,
            prompt=prompt,
            path_style=path_style,
        )

    raise AssertionError("failed to generate numbrix puzzle")


def _sample_solution(
    rows: int,
    cols: int,
    recipe: dict[str, Any],
    rng: Random,
) -> tuple[str, tuple[tuple[int, ...], ...]]:
    path_style = rng.choice(tuple(recipe["path_styles"]))
    path = _path_for_style(rows, cols, path_style)

    if rng.choice((False, True)):
        path = list(reversed(path))
    if rng.choice((False, True)):
        path = [(row, cols + 1 - col) for row, col in path]
    if rng.choice((False, True)):
        path = [(rows + 1 - row, col) for row, col in path]

    return path_style, _grid_from_path(rows, cols, path)


def _path_for_style(rows: int, cols: int, path_style: str) -> list[tuple[int, int]]:
    if path_style == "row_snake":
        path: list[tuple[int, int]] = []
        for row in range(1, rows + 1):
            values = range(1, cols + 1) if row % 2 == 1 else range(cols, 0, -1)
            path.extend((row, col) for col in values)
        return path

    if path_style == "col_snake":
        path = []
        for col in range(1, cols + 1):
            values = range(1, rows + 1) if col % 2 == 1 else range(rows, 0, -1)
            path.extend((row, col) for row in values)
        return path

    if path_style == "spiral":
        path = []
        top = 1
        bottom = rows
        left = 1
        right = cols
        while top <= bottom and left <= right:
            path.extend((top, col) for col in range(left, right + 1))
            top += 1
            path.extend((row, right) for row in range(top, bottom + 1))
            right -= 1
            if top <= bottom:
                path.extend((bottom, col) for col in range(right, left - 1, -1))
                bottom -= 1
            if left <= right:
                path.extend((row, left) for row in range(bottom, top - 1, -1))
                left += 1
        return path

    raise AssertionError(f"unsupported numbrix path style: {path_style}")


def _grid_from_path(
    rows: int,
    cols: int,
    path: list[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    grid = [[-1 for _ in range(cols)] for _ in range(rows)]
    for value, (row, col) in enumerate(path):
        grid[row - 1][col - 1] = value
    return tuple(tuple(row) for row in grid)


def _select_clues(
    solution_grid: tuple[tuple[int, ...], ...],
    recipe: dict[str, Any],
    rng: Random,
) -> tuple[NumbrixClue, ...] | None:
    rows = len(solution_grid)
    cols = len(solution_grid[0])
    total = rows * cols
    candidates = _build_candidates(solution_grid)
    clue_by_value = {clue.value: clue for clue in candidates}
    anchors = _anchor_values(total, tuple(recipe["anchor_ratios"]))
    sample_attempts = int(recipe["sample_attempts"])

    for given_count in range(int(recipe["target_given_count"]), int(recipe["max_given_count"]) + 1):
        for _ in range(sample_attempts):
            values = _sample_values(
                total,
                given_count,
                anchors=anchors,
                clue_profile=str(recipe["clue_profile"]),
                rng=rng,
            )
            clues = [clue_by_value[value] for value in values]
            if _has_unique_solution(rows, cols, clues):
                return tuple(sorted(clues, key=lambda clue: clue.value))

    return None


def _build_candidates(solution_grid: tuple[tuple[int, ...], ...]) -> list[NumbrixClue]:
    clues: list[NumbrixClue] = []
    for row_index, row in enumerate(solution_grid, start=1):
        for col_index, value in enumerate(row, start=1):
            clues.append(NumbrixClue(row=row_index, col=col_index, value=value))
    return clues


def _anchor_values(total: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    values: list[int] = []
    max_value = total - 1
    for ratio in ratios:
        candidate = round(max_value * ratio)
        if candidate not in values:
            values.append(candidate)
    return tuple(values)


def _sample_values(
    total: int,
    count: int,
    *,
    anchors: tuple[int, ...],
    clue_profile: str,
    rng: Random,
) -> tuple[int, ...]:
    selected = list(anchors[:count])
    if len(selected) == count:
        return tuple(sorted(selected))

    remaining = [value for value in range(total) if value not in selected]
    needed = count - len(selected)

    if clue_profile == "spread":
        extras = _spread_values(total, remaining, needed, rng)
    elif clue_profile == "near_complete":
        pool = list(remaining)
        rng.shuffle(pool)
        extras = pool[:needed]
    else:
        pool = list(remaining)
        rng.shuffle(pool)
        extras = pool[:needed]

    selected.extend(extras)
    return tuple(sorted(selected))


def _spread_values(total: int, remaining: list[int], needed: int, rng: Random) -> list[int]:
    pool = list(remaining)
    rng.shuffle(pool)
    chosen: list[int] = []
    for index in range(needed):
        target = round((index + 1) * (total - 1) / (needed + 1))
        candidate = min(pool, key=lambda value: (abs(value - target), value))
        chosen.append(candidate)
        pool.remove(candidate)
    return chosen


def _solve_grids(
    rows: int,
    cols: int,
    clues: list[NumbrixClue] | tuple[NumbrixClue, ...],
    *,
    limit: int | None = None,
) -> list[tuple[tuple[int, ...], ...]]:
    solver, variables, row_vars, col_vars = _build_solver(rows, cols, list(clues))
    assignments = z3_solver.enumerate_int_models(solver, variables, limit=limit)
    return [_grid_from_assignment(rows, cols, row_vars, col_vars, assignment) for assignment in assignments]


def _solve_grid_count(
    rows: int,
    cols: int,
    clues: list[NumbrixClue],
    *,
    limit: int | None = None,
) -> tuple[int, bool]:
    solver, variables, _row_vars, _col_vars = _build_solver(rows, cols, clues)
    return z3_solver.count_int_models(solver, variables, limit=limit)


def _has_unique_solution(rows: int, cols: int, clues: list[NumbrixClue]) -> bool:
    count, complete = _solve_grid_count(rows, cols, clues, limit=2)
    return count == 1 and complete


def _build_solver(
    rows: int,
    cols: int,
    clues: list[NumbrixClue],
) -> tuple[z3.Solver, dict[str, z3.ArithRef], dict[int, z3.ArithRef], dict[int, z3.ArithRef]]:
    solver = z3.Solver()
    variables: dict[str, z3.ArithRef] = {}
    row_vars: dict[int, z3.ArithRef] = {}
    col_vars: dict[int, z3.ArithRef] = {}
    total = rows * cols

    for value in range(total):
        row_var = z3.Int(f"r{value}")
        col_var = z3.Int(f"c{value}")
        variables[f"r{value}"] = row_var
        variables[f"c{value}"] = col_var
        row_vars[value] = row_var
        col_vars[value] = col_var
        solver.add(row_var >= 1, row_var <= rows)
        solver.add(col_var >= 1, col_var <= cols)

    for first in range(total):
        for second in range(first + 1, total):
            solver.add(
                z3.Or(
                    row_vars[first] != row_vars[second],
                    col_vars[first] != col_vars[second],
                )
            )

    for value in range(total - 1):
        solver.add(
            z3.Abs(row_vars[value] - row_vars[value + 1]) +
            z3.Abs(col_vars[value] - col_vars[value + 1]) == 1
        )

    for clue in clues:
        solver.add(row_vars[clue.value] == clue.row)
        solver.add(col_vars[clue.value] == clue.col)

    return solver, variables, row_vars, col_vars


def _grid_from_assignment(
    rows: int,
    cols: int,
    row_vars: dict[int, z3.ArithRef],
    col_vars: dict[int, z3.ArithRef],
    assignment: dict[str, int],
) -> tuple[tuple[int, ...], ...]:
    grid = [[-1 for _ in range(cols)] for _ in range(rows)]
    for value in row_vars:
        row = assignment[f"r{value}"] - 1
        col = assignment[f"c{value}"] - 1
        grid[row][col] = value
    return tuple(tuple(row) for row in grid)


def _format_prompt(
    rows: int,
    cols: int,
    clues: tuple[NumbrixClue, ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    given_lines, given_heading, given_numbered = _given_block(
        rows,
        cols,
        clues,
        language=language,
    )

    blocks = {
        "intro": prompt_surface.PromptBlock(
            "intro",
            _intro_lines(
                language=language,
                rows=rows,
                cols=cols,
                intro_variant=str(surface_plan["surface_intro"]),
                prompt_style=recipe["prompt_style"],
            ),
        ),
        "rules": prompt_surface.PromptBlock(
            "rules",
            _rule_lines(language=language, rows=rows, cols=cols),
            heading="Regler" if language == "da" else "Rules",
        ),
        "givens": prompt_surface.PromptBlock(
            "givens",
            given_lines,
            heading=given_heading,
            numbered=given_numbered,
        ),
        "instruction": prompt_surface.PromptBlock(
            "instruction",
            (_instruction_line(language=language, instruction_variant=str(surface_plan["surface_instruction"])),),
        ),
        "answer": prompt_surface.PromptBlock(
            "answer",
            _answer_lines(language=language, rows=rows, cols=cols, answer_variant=str(surface_plan["surface_answer"])),
        ),
    }
    return prompt_surface.render_prompt(blocks, SURFACE_SPECS[recipe["prompt_style"]].plan)


def _intro_lines(
    *,
    language: str,
    rows: int,
    cols: int,
    intro_variant: str,
    prompt_style: str,
) -> tuple[str, ...]:
    if language == "da":
        if intro_variant == "context":
            first = f"Dette er en numbrix-opgave med et {rows}x{cols}-gitter."
        elif intro_variant == "assignment":
            first = f"Udfyld et numbrix-gitter på {rows}x{cols} felter."
        else:
            first = f"Løs en numbrix-opgave på et {rows}x{cols}-gitter."
        indexing = "Rækker tælles oppefra og ned, og kolonner tælles fra venstre mod højre."
        if prompt_style == "deduce":
            return ("Brug de givne tal til at udlede hele stien.", first, indexing)
        return (first, indexing)

    if intro_variant == "context":
        first = f"This is a numbrix puzzle on a {rows}x{cols} grid."
    elif intro_variant == "assignment":
        first = f"Fill a {rows}x{cols} numbrix grid."
    else:
        first = f"Solve a numbrix puzzle on a {rows}x{cols} grid."
    indexing = "Rows are counted from top to bottom, and columns from left to right."
    if prompt_style == "deduce":
        return ("Use the given numbers to work out the full path.", first, indexing)
    return (first, indexing)


def _rule_lines(*, language: str, rows: int, cols: int) -> tuple[str, ...]:
    max_value = rows * cols - 1
    if language == "da":
        return (
            f"Brug tallene 0 til {max_value}.",
            "Hvert tal skal forekomme præcis én gang.",
            "Hvert tal skal stå vandret eller lodret ved siden af det næste tal.",
            "Tomme felter er markeret med -1.",
        )

    return (
        f"Use the numbers 0 to {max_value}.",
        "Each number must appear exactly once.",
        "Each number must be horizontally or vertically adjacent to the next number.",
        "Empty cells are marked with -1.",
    )


def _instruction_line(*, language: str, instruction_variant: str) -> str:
    if language == "da":
        if instruction_variant == "solve":
            return "Der er præcis én gyldig løsning."
        if instruction_variant == "unique":
            return "De givne tal bestemmer én entydig løsning."
        return "Alle regler og givne tal skal passe med den samme sammenhængende sti."

    if instruction_variant == "solve":
        return "There is exactly one valid solution."
    if instruction_variant == "unique":
        return "The given numbers determine one unique solution."
    return "The rules and given numbers must all fit the same connected path."


def _answer_lines(
    *,
    language: str,
    rows: int,
    cols: int,
    answer_variant: str,
) -> tuple[str, ...]:
    example_grid = _example_grid(rows, cols)

    if language == "da":
        first_line = {
            "respond": f"Inde i det endelige svar skal gitteret have præcis {rows} linjer i dette format:",
            "write": f"Inde i det endelige svar skal du bruge præcis {rows} linjer i dette format:",
            "complete": f"Selve svarindholdet skal være det udfyldte gitter med præcis {rows} linjer i dette format:",
        }[answer_variant]
        return (first_line, *example_grid)

    first_line = {
        "respond": f"Inside the final response, the grid should use exactly {rows} lines in this format:",
        "write": f"Inside the final response, use exactly {rows} lines in this format:",
        "complete": f"The answer content should be the completed grid with exactly {rows} lines in this format:",
    }[answer_variant]
    return (first_line, *example_grid)


def _given_block(
    rows: int,
    cols: int,
    clues: tuple[NumbrixClue, ...],
    *,
    language: str,
) -> tuple[tuple[str, ...], str, bool]:
    grid = [["-1" for _ in range(cols)] for _ in range(rows)]
    for clue in clues:
        grid[clue.row - 1][clue.col - 1] = str(clue.value)

    lines = tuple(" ".join(values) for values in grid)
    heading = "Startgitter" if language == "da" else "Starting grid"
    return lines, heading, False


def _recipe_by_id(recipe_id: str) -> dict[str, Any]:
    for recipe in RECIPES:
        if recipe["recipe_id"] == recipe_id:
            return recipe
    raise AssertionError(f"unknown numbrix recipe: {recipe_id}")


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]


def _example_grid(rows: int, cols: int) -> tuple[str, ...]:
    return tuple(
        " ".join(str(row * cols + col) for col in range(cols))
        for row in range(rows)
    )
