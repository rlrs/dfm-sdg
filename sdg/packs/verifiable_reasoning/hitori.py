from dataclasses import dataclass
from random import Random
from typing import Any

import z3

from sdg.commons import diversity, prompt_surface, z3_solver


@dataclass(frozen=True)
class HitoriSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class HitoriPuzzle:
    rows: int
    cols: int
    board_grid: tuple[tuple[int, ...], ...]
    solution_mask: tuple[str, ...]
    prompt: str
    black_count: int


SURFACE_SPECS = {
    "board": HitoriSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="board",
            block_order=("intro", "rules", "grid", "instruction", "answer"),
        ),
    ),
    "briefing": HitoriSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "instruction", "rules", "grid", "answer"),
        ),
    ),
    "deduce": HitoriSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "rules", "answer", "instruction", "grid"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_rect_3x5",
        "difficulty": "easy",
        "prompt_style": "board",
        "rows": 3,
        "cols": 5,
        "clue_profile": "light",
        "min_black_count": 4,
        "max_black_count": 5,
        "sample_attempts": 80,
    },
    {
        "recipe_id": "easy_square_4x4",
        "difficulty": "easy",
        "prompt_style": "briefing",
        "rows": 4,
        "cols": 4,
        "clue_profile": "light",
        "min_black_count": 4,
        "max_black_count": 5,
        "sample_attempts": 96,
    },
    {
        "recipe_id": "medium_rect_4x5",
        "difficulty": "medium",
        "prompt_style": "board",
        "rows": 4,
        "cols": 5,
        "clue_profile": "balanced",
        "min_black_count": 5,
        "max_black_count": 7,
        "sample_attempts": 128,
    },
    {
        "recipe_id": "hard_square_5x5",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "rows": 5,
        "cols": 5,
        "clue_profile": "dense",
        "min_black_count": 6,
        "max_black_count": 8,
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
            "board_grid": [list(row) for row in puzzle.board_grid],
            "solution_mask": list(puzzle.solution_mask),
        },
        "sources": [{"kind": "dolci_subset", "value": "hitoripuzzle"}],
        "meta": {
            "family": "hitori_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": puzzle.rows * puzzle.cols,
            "given_count": puzzle.rows * puzzle.cols,
            "rows": puzzle.rows,
            "cols": puzzle.cols,
            "cell_count": puzzle.rows * puzzle.cols,
            "black_count": puzzle.black_count,
            "output_format": "mask_grid",
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
    if any(any(char not in {".", "*"} for char in line) for line in lines):
        return None
    return tuple(lines)


def canonical_target(parsed: tuple[str, ...], hidden: dict[str, object]) -> str:
    return "\n".join(parsed)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    example = "." * cols

    if language == "da":
        return (
            "I din svarblok skal den endelige løsning være en maske med én række per linje.\n"
            f"Brug præcis {rows} linjer med {cols} tegn per linje og ingen mellemrum.\n"
            "Brug '.' for et felt, der bliver stående, og '*' for et sort felt.\n"
            f"Eksempel på en linje:\n{example}"
        )

    return (
        "In your answer block, the final solution should be a mask with one row per line.\n"
        f"Use exactly {rows} lines with {cols} characters per line and no spaces.\n"
        "Use '.' for a remaining cell and '*' for a blacked-out cell.\n"
        f"Example line:\n{example}"
    )


def is_correct(parsed: tuple[str, ...], hidden: dict[str, object]) -> bool:
    return list(parsed) == hidden["solution_mask"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    board_grid = tuple(tuple(value for value in row) for row in hidden["board_grid"])
    solutions = _solve_masks(rows, cols, board_grid, limit=2)
    if len(solutions) != 1:
        return False
    return list(solutions[0]) == hidden["solution_mask"]


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
) -> HitoriPuzzle:
    rows = int(recipe["rows"])
    cols = int(recipe["cols"])

    for _ in range(200):
        black_count = rng.randint(int(recipe["min_black_count"]), int(recipe["max_black_count"]))
        solution_mask = _sample_mask(rows, cols, black_count, rng)
        if solution_mask is None:
            continue
        board_grid = _build_board(solution_mask, rows, cols, rng)
        if not _mask_is_valid(board_grid, solution_mask):
            continue
        solutions = _solve_masks(rows, cols, board_grid, limit=2)
        if len(solutions) != 1:
            continue
        if solutions[0] != solution_mask:
            continue
        prompt = _format_prompt(
            rows,
            cols,
            board_grid,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return HitoriPuzzle(
            rows=rows,
            cols=cols,
            board_grid=board_grid,
            solution_mask=solution_mask,
            prompt=prompt,
            black_count=black_count,
        )

    raise AssertionError("failed to generate hitori puzzle")


def _sample_mask(
    rows: int,
    cols: int,
    black_count: int,
    rng: Random,
) -> tuple[str, ...] | None:
    cells = [(row, col) for row in range(rows) for col in range(cols)]

    for _ in range(200):
        rng.shuffle(cells)
        black: set[tuple[int, int]] = set()
        for row, col in cells:
            if len(black) >= black_count:
                break
            if any(neighbor in black for neighbor in _neighbors(row, col, rows, cols)):
                continue
            candidate = black | {(row, col)}
            if not _white_connected(rows, cols, candidate):
                continue
            black = candidate
        if len(black) != black_count:
            continue
        return tuple(
            "".join("*" if (row, col) in black else "." for col in range(cols))
            for row in range(rows)
        )

    return None


def _build_board(
    solution_mask: tuple[str, ...],
    rows: int,
    cols: int,
    rng: Random,
) -> tuple[tuple[int, ...], ...]:
    value_range = max(rows, cols)
    base = [
        [((row + col) % value_range) for col in range(cols)]
        for row in range(rows)
    ]
    board = [list(row) for row in base]

    for row in range(rows):
        for col in range(cols):
            if solution_mask[row][col] != "*":
                continue

            choices: list[int] = []
            for other_col in range(cols):
                if other_col == col or solution_mask[row][other_col] == "*":
                    continue
                choices.append(base[row][other_col])
            for other_row in range(rows):
                if other_row == row or solution_mask[other_row][col] == "*":
                    continue
                choices.append(base[other_row][col])

            assert choices, "black cell must duplicate a visible number in its row or column"
            board[row][col] = rng.choice(choices)

    return tuple(tuple(row) for row in board)


def _mask_is_valid(
    board_grid: tuple[tuple[int, ...], ...],
    solution_mask: tuple[str, ...],
) -> bool:
    rows = len(board_grid)
    cols = len(board_grid[0])

    black = {(row, col) for row in range(rows) for col in range(cols) if solution_mask[row][col] == "*"}
    for row, col in black:
        if any(neighbor in black for neighbor in _neighbors(row, col, rows, cols)):
            return False

    if not _white_connected(rows, cols, black):
        return False

    for row in range(rows):
        seen: set[int] = set()
        for col in range(cols):
            if solution_mask[row][col] == "*":
                continue
            value = board_grid[row][col]
            if value in seen:
                return False
            seen.add(value)

    for col in range(cols):
        seen = set()
        for row in range(rows):
            if solution_mask[row][col] == "*":
                continue
            value = board_grid[row][col]
            if value in seen:
                return False
            seen.add(value)

    return True


def _white_connected(
    rows: int,
    cols: int,
    black: set[tuple[int, int]],
) -> bool:
    white = [(row, col) for row in range(rows) for col in range(cols) if (row, col) not in black]
    if not white:
        return False

    seen = {white[0]}
    stack = [white[0]]
    while stack:
        row, col = stack.pop()
        for neighbor in _neighbors(row, col, rows, cols):
            if neighbor in black or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)

    return len(seen) == len(white)


def _solve_masks(
    rows: int,
    cols: int,
    board_grid: tuple[tuple[int, ...], ...],
    *,
    limit: int | None = None,
) -> list[tuple[str, ...]]:
    solver, black_vars = _build_solver(rows, cols, board_grid)
    assignments = z3_solver.enumerate_int_models(solver, black_vars, limit=limit)
    return [_mask_from_assignment(rows, cols, assignment) for assignment in assignments]


def _solve_mask_count(
    rows: int,
    cols: int,
    board_grid: tuple[tuple[int, ...], ...],
    *,
    limit: int | None = None,
) -> tuple[int, bool]:
    solver, black_vars = _build_solver(rows, cols, board_grid)
    return z3_solver.count_int_models(solver, black_vars, limit=limit)


def _build_solver(
    rows: int,
    cols: int,
    board_grid: tuple[tuple[int, ...], ...],
) -> tuple[z3.Solver, dict[str, z3.ArithRef]]:
    solver = z3.Solver()
    black_vars: dict[str, z3.ArithRef] = {}
    order_vars: dict[tuple[int, int], z3.ArithRef] = {}
    total = rows * cols

    for row in range(rows):
        for col in range(cols):
            name = _var_key(row, col)
            black_var = z3.Int(name)
            order_var = z3.Int(f"o_{row}_{col}")
            black_vars[name] = black_var
            order_vars[(row, col)] = order_var
            solver.add(z3.Or(black_var == 0, black_var == 1))
            solver.add(
                z3.If(
                    black_var == 1,
                    order_var == 0,
                    z3.And(order_var >= 1, order_var <= total),
                )
            )

    for row in range(rows):
        for col in range(cols):
            value = board_grid[row][col]
            for other_col in range(col + 1, cols):
                if board_grid[row][other_col] != value:
                    continue
                solver.add(black_vars[_var_key(row, col)] + black_vars[_var_key(row, other_col)] >= 1)
            for other_row in range(row + 1, rows):
                if board_grid[other_row][col] != value:
                    continue
                solver.add(black_vars[_var_key(row, col)] + black_vars[_var_key(other_row, col)] >= 1)

    for row in range(rows):
        for col in range(cols):
            for other_row, other_col in _neighbors(row, col, rows, cols):
                if (other_row, other_col) <= (row, col):
                    continue
                solver.add(black_vars[_var_key(row, col)] + black_vars[_var_key(other_row, other_col)] <= 1)

    white_cells = [1 - black_vars[_var_key(row, col)] for row in range(rows) for col in range(cols)]
    solver.add(z3.Sum(white_cells) >= 1)
    solver.add(
        z3.Sum(
            [
                z3.If(
                    z3.And(black_vars[_var_key(row, col)] == 0, order_vars[(row, col)] == 1),
                    1,
                    0,
                )
                for row in range(rows)
                for col in range(cols)
            ]
        ) == 1
    )

    cells = [(row, col) for row in range(rows) for col in range(cols)]
    for index, (row, col) in enumerate(cells):
        for other_row, other_col in cells[index + 1:]:
            solver.add(
                z3.Implies(
                    z3.And(
                        black_vars[_var_key(row, col)] == 0,
                        black_vars[_var_key(other_row, other_col)] == 0,
                    ),
                    order_vars[(row, col)] != order_vars[(other_row, other_col)],
                )
            )

    for row in range(rows):
        for col in range(cols):
            smaller_neighbor = z3.Or(
                [
                    z3.And(
                        black_vars[_var_key(other_row, other_col)] == 0,
                        order_vars[(other_row, other_col)] < order_vars[(row, col)],
                    )
                    for other_row, other_col in _neighbors(row, col, rows, cols)
                ]
            )
            solver.add(
                z3.Implies(
                    z3.And(
                        black_vars[_var_key(row, col)] == 0,
                        order_vars[(row, col)] > 1,
                    ),
                    smaller_neighbor,
                )
            )

    return solver, black_vars


def _mask_from_assignment(
    rows: int,
    cols: int,
    assignment: dict[str, int],
) -> tuple[str, ...]:
    return tuple(
        "".join("*" if assignment[_var_key(row, col)] == 1 else "." for col in range(cols))
        for row in range(rows)
    )


def _format_prompt(
    rows: int,
    cols: int,
    board_grid: tuple[tuple[int, ...], ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
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
            _rule_lines(language=language),
            heading="Regler" if language == "da" else "Rules",
        ),
        "grid": prompt_surface.PromptBlock(
            "grid",
            tuple(" ".join(str(value) for value in row) for row in board_grid),
            heading="Matrix",
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
            first = f"Dette er en hitori-opgave på et {rows}x{cols}-gitter."
        elif intro_variant == "assignment":
            first = f"Løs en hitori-opgave på et {rows}x{cols}-gitter."
        else:
            first = f"Markér nogle felter sorte i et {rows}x{cols}-gitter."
        if prompt_style == "deduce":
            return ("Brug reglerne til at finde de sorte felter.", first)
        return (first,)

    if intro_variant == "context":
        first = f"This is a hitori puzzle on a {rows}x{cols} grid."
    elif intro_variant == "assignment":
        first = f"Solve a hitori puzzle on a {rows}x{cols} grid."
    else:
        first = f"Black out some cells in a {rows}x{cols} grid."
    if prompt_style == "deduce":
        return ("Use the rules to determine which cells must be blacked out.", first)
    return (first,)


def _rule_lines(*, language: str) -> tuple[str, ...]:
    if language == "da":
        return (
            "I hver række og kolonne må intet tal forekomme mere end én gang blandt de felter, der bliver stående.",
            "To sorte felter må ikke ligge vandret eller lodret op ad hinanden.",
            "Alle felter, der bliver stående, skal danne ét sammenhængende område.",
        )

    return (
        "In each row and each column, no number may appear more than once among the remaining cells.",
        "No two blacked-out cells may be horizontally or vertically adjacent.",
        "All remaining cells must form a single connected region.",
    )


def _instruction_line(*, language: str, instruction_variant: str) -> str:
    if language == "da":
        if instruction_variant == "solve":
            return "Der er præcis én gyldig måde at markere felterne på."
        if instruction_variant == "unique":
            return "Reglerne bestemmer én entydig løsning."
        return "Alle tre regler skal passe samtidig for den samme maske."

    if instruction_variant == "solve":
        return "There is exactly one valid way to black out the cells."
    if instruction_variant == "unique":
        return "The rules determine one unique solution."
    return "All three rules must hold at the same time for the same mask."


def _answer_lines(
    *,
    language: str,
    rows: int,
    cols: int,
    answer_variant: str,
) -> tuple[str, ...]:
    example_line = "." * cols

    if language == "da":
        first_line = {
            "respond": f"Når du giver dit endelige svar, skal du bruge præcis {rows} linjer med {cols} tegn per linje og ingen mellemrum.",
            "write": f"Når du skriver det endelige svar, skal du bruge præcis {rows} linjer med {cols} tegn per linje og ingen mellemrum.",
            "complete": f"Giv den endelige maske med præcis {rows} linjer og {cols} tegn per linje uden mellemrum.",
        }[answer_variant]
        second_line = "Brug '.' for et felt, der bliver stående, og '*' for et sort felt."
        return (first_line, second_line, f"Eksempellinje: {example_line}")

    first_line = {
        "respond": f"When you give your final answer, use exactly {rows} lines with {cols} characters per line and no spaces.",
        "write": f"When you write the final answer, use exactly {rows} lines with {cols} characters per line and no spaces.",
        "complete": f"Give the final mask using exactly {rows} lines with {cols} characters per line and no spaces.",
    }[answer_variant]
    second_line = "Use '.' for a remaining cell and '*' for a blacked-out cell."
    return (first_line, second_line, f"Example line: {example_line}")


def _neighbors(row: int, col: int, rows: int, cols: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    if row > 0:
        neighbors.append((row - 1, col))
    if row + 1 < rows:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col + 1 < cols:
        neighbors.append((row, col + 1))
    return neighbors


def _var_key(row: int, col: int) -> str:
    return f"b_{row}_{col}"


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]
