from dataclasses import dataclass
from random import Random
from typing import Any

import z3

from sdg.commons import diversity, prompt_surface, z3_solver


@dataclass(frozen=True)
class StarBattleSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class StarBattlePuzzle:
    rows: int
    cols: int
    board_grid: tuple[str, ...]
    solution_grid: tuple[str, ...]
    prompt: str
    blocked_count: int
    open_count: int


SURFACE_SPECS = {
    "board": StarBattleSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="board",
            block_order=("intro", "rules", "grid", "answer"),
        ),
    ),
    "briefing": StarBattleSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "grid", "rules", "answer"),
        ),
    ),
    "deduce": StarBattleSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "rules", "answer", "grid"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_rect_4x6",
        "difficulty": "easy",
        "prompt_style": "board",
        "rows": 4,
        "cols": 6,
        "clue_profile": "blocked_sparse",
        "min_open_cells": 10,
        "min_rows_with_choice": 3,
        "sample_attempts": 80,
    },
    {
        "recipe_id": "easy_square_5x5",
        "difficulty": "easy",
        "prompt_style": "briefing",
        "rows": 5,
        "cols": 5,
        "clue_profile": "blocked_balanced",
        "min_open_cells": 12,
        "min_rows_with_choice": 4,
        "sample_attempts": 96,
    },
    {
        "recipe_id": "medium_rect_5x6",
        "difficulty": "medium",
        "prompt_style": "board",
        "rows": 5,
        "cols": 6,
        "clue_profile": "blocked_dense",
        "min_open_cells": 15,
        "min_rows_with_choice": 5,
        "sample_attempts": 120,
    },
    {
        "recipe_id": "hard_square_6x6",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "rows": 6,
        "cols": 6,
        "clue_profile": "open_heavy",
        "min_open_cells": 20,
        "min_rows_with_choice": 6,
        "sample_attempts": 160,
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
            "board_grid": list(puzzle.board_grid),
            "solution_grid": list(puzzle.solution_grid),
        },
        "sources": [{"kind": "dolci_subset", "value": "starbattle"}],
        "meta": {
            "family": "blocked_star_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": puzzle.rows * puzzle.cols,
            "given_count": puzzle.rows * puzzle.cols,
            "rows": puzzle.rows,
            "cols": puzzle.cols,
            "blocked_count": puzzle.blocked_count,
            "open_count": puzzle.open_count,
            "output_format": "annotated_grid",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            "clue_profile": recipe["clue_profile"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[str, ...] | None:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    lines = [line.strip().replace(" ", "") for line in text.splitlines() if line.strip()]
    if len(lines) != rows:
        return None
    if any(len(line) != cols for line in lines):
        return None
    if any(any(char not in {"X", ".", "*"} for char in line) for line in lines):
        return None
    return tuple(lines)


def canonical_target(parsed: tuple[str, ...], hidden: dict[str, object]) -> str:
    return "\n".join(parsed)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    example = "X" + "." * max(0, cols - 2) + "*"
    if language == "da":
        return (
            "I din svarblok skal den endelige løsning bruge det samme gitterformat som inputtet.\n"
            f"Brug præcis {rows} linjer med {cols} tegn per linje og ingen mellemrum.\n"
            "Hvert tegn skal være `X`, `.` eller `*`.\n"
            f"Eksempellinje:\n{example}"
        )
    return (
        "In your answer block, the final solution should use the same grid format as the input.\n"
        f"Use exactly {rows} lines with {cols} characters per line and no spaces.\n"
        "Each character should be `X`, `.`, or `*`.\n"
        f"Example line:\n{example}"
    )


def is_correct(parsed: tuple[str, ...], hidden: dict[str, object]) -> bool:
    return list(parsed) == hidden["solution_grid"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    board_grid = tuple(str(line) for line in hidden["board_grid"])
    solutions = _solve_grids(rows, cols, board_grid, limit=2)
    if len(solutions) != 1:
        return False
    return list(solutions[0]) == hidden["solution_grid"]


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


def _generate_puzzle(
    rng: Random,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> StarBattlePuzzle:
    rows = int(recipe["rows"])
    cols = int(recipe["cols"])

    for _ in range(int(recipe["sample_attempts"])):
        star_positions = _sample_star_positions(rows, cols, rng)
        if star_positions is None:
            continue
        board_grid = _build_board(rows, cols, star_positions, rng)
        if not _board_is_interesting(board_grid, recipe):
            continue
        solutions = _solve_grids(rows, cols, board_grid, limit=2)
        if len(solutions) != 1:
            continue
        solution_grid = _solution_grid(rows, cols, board_grid, star_positions)
        if solutions[0] != solution_grid:
            continue
        prompt = _format_prompt(
            rows,
            cols,
            board_grid,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        blocked_count = sum(line.count("X") for line in board_grid)
        open_count = rows * cols - blocked_count
        return StarBattlePuzzle(
            rows=rows,
            cols=cols,
            board_grid=board_grid,
            solution_grid=solution_grid,
            prompt=prompt,
            blocked_count=blocked_count,
            open_count=open_count,
        )

    raise AssertionError("failed to generate blocked-star puzzle")


def _sample_star_positions(rows: int, cols: int, rng: Random) -> tuple[int, ...] | None:
    choices = list(range(cols))
    for _ in range(200):
        rng.shuffle(choices)
        selected = tuple(choices[:rows])
        if len(set(selected)) != rows:
            continue
        if any(abs(selected[row] - selected[row - 1]) <= 1 for row in range(1, rows)):
            continue
        return selected
    return None


def _build_board(
    rows: int,
    cols: int,
    star_positions: tuple[int, ...],
    rng: Random,
) -> tuple[str, ...]:
    open_cells = {(row, star_positions[row]) for row in range(rows)}
    candidates = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if (row, col) not in open_cells
    ]
    rng.shuffle(candidates)

    current = open_cells.copy()
    for row, col in candidates:
        proposal = current | {(row, col)}
        board_grid = _board_from_open_cells(rows, cols, proposal)
        solutions = _solve_grids(rows, cols, board_grid, limit=2)
        if len(solutions) == 1:
            current = proposal

    return _board_from_open_cells(rows, cols, current)


def _board_from_open_cells(rows: int, cols: int, open_cells: set[tuple[int, int]]) -> tuple[str, ...]:
    return tuple(
        "".join("." if (row, col) in open_cells else "X" for col in range(cols))
        for row in range(rows)
    )


def _board_is_interesting(board_grid: tuple[str, ...], recipe: dict[str, Any]) -> bool:
    rows = len(board_grid)
    cols = len(board_grid[0])
    open_count = sum(line.count(".") for line in board_grid)
    if open_count < int(recipe["min_open_cells"]):
        return False

    rows_with_choice = sum(line.count(".") >= 2 for line in board_grid)
    if rows_with_choice < int(recipe["min_rows_with_choice"]):
        return False

    open_by_col = [
        sum(board_grid[row][col] == "." for row in range(rows))
        for col in range(cols)
    ]
    if sum(count >= 2 for count in open_by_col) < rows - 1:
        return False

    return True


def _solve_grids(
    rows: int,
    cols: int,
    board_grid: tuple[str, ...],
    *,
    limit: int | None = None,
) -> list[tuple[str, ...]]:
    solver = z3.Solver()
    variables: dict[str, z3.ArithRef] = {}
    cells: list[list[z3.ArithRef]] = []

    for row in range(rows):
        solver_row: list[z3.ArithRef] = []
        for col in range(cols):
            var = z3.Int(f"cell_{row}_{col}")
            solver.add(z3.Or(var == 0, var == 1))
            if board_grid[row][col] == "X":
                solver.add(var == 0)
            variables[f"r{row}c{col}"] = var
            solver_row.append(var)
        cells.append(solver_row)

    for row in range(rows):
        solver.add(z3.Sum(cells[row]) == 1)

    for col in range(cols):
        solver.add(z3.Sum([cells[row][col] for row in range(rows)]) <= 1)

    for row in range(rows):
        for col in range(cols):
            for next_row in range(max(0, row - 1), min(rows, row + 2)):
                for next_col in range(max(0, col - 1), min(cols, col + 2)):
                    if (row, col) >= (next_row, next_col):
                        continue
                    solver.add(cells[row][col] + cells[next_row][next_col] <= 1)

    models = z3_solver.enumerate_int_models(solver, variables, limit=limit)
    solutions: list[tuple[str, ...]] = []
    for model in models:
        solution_lines: list[str] = []
        for row in range(rows):
            chars: list[str] = []
            for col in range(cols):
                if board_grid[row][col] == "X":
                    chars.append("X")
                elif model[f"r{row}c{col}"] == 1:
                    chars.append("*")
                else:
                    chars.append(".")
            solution_lines.append("".join(chars))
        solutions.append(tuple(solution_lines))
    return solutions


def _solution_grid(
    rows: int,
    cols: int,
    board_grid: tuple[str, ...],
    star_positions: tuple[int, ...],
) -> tuple[str, ...]:
    return tuple(
        "".join(
            "X"
            if board_grid[row][col] == "X"
            else "*"
            if star_positions[row] == col
            else "."
            for col in range(cols)
        )
        for row in range(rows)
    )


def _format_prompt(
    rows: int,
    cols: int,
    board_grid: tuple[str, ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    prompt_style = str(recipe["prompt_style"])
    surface_spec = SURFACE_SPECS[prompt_style]
    blocks = {
        "intro": _intro_block(rows, cols, language, surface_plan),
        "rules": _rules_block(language, surface_plan),
        "grid": _grid_block(board_grid, language),
        "answer": _answer_block(rows, cols, language, surface_plan),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(
    rows: int,
    cols: int,
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    intro_style = str(surface_plan["surface_intro"])
    if language == "da":
        if intro_style == "context":
            lines = (
                f"Du får et {rows}x{cols}-gitter.",
                "Hvert felt indeholder enten `X` eller `.`. Vælg nogle `.`-felter og udfyld dem med `*` sådan at:",
            )
        elif intro_style == "assignment":
            lines = (
                f"Her er en stjerneplaceringsopgave på et {rows}x{cols}-gitter.",
                "Placer `*` i nogle af `.`-felterne sådan at alle regler er opfyldt:",
            )
        else:
            lines = (
                f"Løs en stjerneplaceringsopgave på et {rows}x{cols}-gitter.",
                "Udvælg `.`-felter til `*` under følgende regler:",
            )
        return prompt_surface.PromptBlock(key="intro", lines=lines)

    if intro_style == "context":
        lines = (
            f"You are given a {rows} x {cols} grid.",
            "Each cell contains either `X` or `.`. Select some `.` cells to fill with `*` such that:",
        )
    elif intro_style == "assignment":
        lines = (
            f"This is a blocked-star placement puzzle on a {rows} x {cols} grid.",
            "Place `*` in some of the `.` cells so that all rules are satisfied:",
        )
    else:
        lines = (
            f"Solve a blocked-star placement puzzle on a {rows} x {cols} grid.",
            "Choose `.` cells to turn into `*` under the following rules:",
        )
    return prompt_surface.PromptBlock(key="intro", lines=lines)


def _rules_block(language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    if language == "da":
        if clue_style == "compact":
            lines = (
                "Hver række indeholder præcis én `*`.",
                "Hver kolonne indeholder højst én `*`.",
                "Ingen to `*`-felter må være naboer, heller ikke diagonalt.",
            )
        else:
            lines = (
                "Hver række skal indeholde præcis én `*`.",
                "Hver kolonne må indeholde højst én `*`.",
                "Ingen to `*`-felter må være naboer, heller ikke diagonalt.",
            )
        return prompt_surface.PromptBlock(key="rules", lines=lines, numbered=True)

    if clue_style == "compact":
        lines = (
            "Each row contains exactly one `*`.",
            "Each column contains at most one `*`.",
            "No two `*` cells may be adjacent, including diagonally.",
        )
    else:
        lines = (
            "Each row must contain exactly one `*`.",
            "Each column may contain at most one `*`.",
            "No two `*` cells may be adjacent, including diagonally.",
        )
    return prompt_surface.PromptBlock(key="rules", lines=lines, numbered=True)


def _grid_block(board_grid: tuple[str, ...], language: str) -> prompt_surface.PromptBlock:
    if language == "da":
        lines = (
            "Gitteret er givet rækkevis, med én streng per række:",
            *board_grid,
        )
    else:
        lines = (
            "The grid is given in row-major order, with one string per row:",
            *board_grid,
        )
    return prompt_surface.PromptBlock(key="grid", lines=lines)


def _answer_block(
    rows: int,
    cols: int,
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    answer_style = str(surface_plan["surface_answer"])
    example = "X" + "." * max(0, cols - 2) + "*"
    if language == "da":
        if answer_style == "respond":
            lines = (
                f"Svarformat: Skriv præcis {rows} linjer med {cols} tegn per linje.",
                "Hvert tegn skal være `X`, `.`, eller `*`.",
                "Outputtet skal have samme format som inputgitteret.",
            )
        elif answer_style == "write":
            lines = (
                f"Outputformat: Brug præcis {rows} linjer med {cols} tegn.",
                "Brug kun `X`, `.`, og `*`.",
                "Bevar inputgitterets format.",
            )
        else:
            lines = (
                f"Det endelige svar skal bestå af {rows} linjer med {cols} tegn hver.",
                "Brug kun `X`, `.`, og `*`, og match inputformatet.",
            )
        lines += (f"Eksempellinje: {example}",)
        return prompt_surface.PromptBlock(key="answer", lines=lines)

    if answer_style == "respond":
        lines = (
            f"Output Format: Write exactly {rows} lines with {cols} characters per line.",
            "Each character should be `X`, `.`, or `*`.",
            "The output should match the input grid format.",
        )
    elif answer_style == "write":
        lines = (
            f"Output Format: Use exactly {rows} lines with {cols} characters.",
            "Use only `X`, `.`, and `*`.",
            "Preserve the input grid format.",
        )
    else:
        lines = (
            f"Your final answer should consist of {rows} lines with {cols} characters each.",
            "Use only `X`, `.`, and `*`, matching the input format.",
        )
    lines += (f"Example line: {example}",)
    return prompt_surface.PromptBlock(key="answer", lines=lines)


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"].get(key)
