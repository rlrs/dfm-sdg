from dataclasses import dataclass
from random import Random
from typing import Any

import z3

from sdg.commons import diversity, prompt_surface, z3_solver


@dataclass(frozen=True)
class LightUpSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class LightUpPuzzle:
    rows: int
    cols: int
    board_grid: tuple[str, ...]
    solution_grid: tuple[str, ...]
    prompt: str
    wall_count: int
    numbered_wall_count: int
    lamp_count: int


SURFACE_SPECS = {
    "board": LightUpSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="board",
            block_order=("intro", "rules", "grid", "instruction", "answer"),
        ),
    ),
    "briefing": LightUpSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "grid", "rules", "instruction", "answer"),
        ),
    ),
    "deduce": LightUpSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "instruction", "rules", "answer", "grid"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_square_4x4",
        "difficulty": "easy",
        "prompt_style": "board",
        "rows": 4,
        "cols": 4,
        "clue_profile": "numbered_local",
        "wall_prob": 0.30,
        "min_walls": 4,
        "max_walls": 7,
        "min_numbered_walls": 4,
        "max_numbered_walls": 7,
        "min_positive_numbered_walls": 2,
        "max_zero_numbered_walls": 2,
        "max_visibility_segment": 3,
        "require_heuristic_solution": True,
        "sample_attempts": 600,
    },
    {
        "recipe_id": "medium_square_5x5",
        "difficulty": "medium",
        "prompt_style": "briefing",
        "rows": 5,
        "cols": 5,
        "clue_profile": "mixed",
        "wall_prob": 0.24,
        "min_walls": 5,
        "max_walls": 8,
        "min_numbered_walls": 4,
        "max_numbered_walls": 6,
        "sample_attempts": 140,
    },
    {
        "recipe_id": "medium_square_6x6",
        "difficulty": "medium",
        "prompt_style": "board",
        "rows": 6,
        "cols": 6,
        "clue_profile": "mixed",
        "wall_prob": 0.28,
        "min_walls": 7,
        "max_walls": 11,
        "min_numbered_walls": 5,
        "max_numbered_walls": 8,
        "sample_attempts": 160,
    },
    {
        "recipe_id": "hard_square_6x6",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "rows": 6,
        "cols": 6,
        "clue_profile": "sparse_numbered",
        "wall_prob": 0.30,
        "min_walls": 8,
        "max_walls": 12,
        "min_numbered_walls": 4,
        "max_numbered_walls": 6,
        "sample_attempts": 180,
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
        "sources": [{"kind": "dolci_subset", "value": "lightuppuzzle"}],
        "meta": {
            "family": "lightuppuzzle_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": puzzle.wall_count,
            "given_count": puzzle.wall_count,
            "rows": puzzle.rows,
            "cols": puzzle.cols,
            "wall_count": puzzle.wall_count,
            "numbered_wall_count": puzzle.numbered_wall_count,
            "lamp_count": puzzle.lamp_count,
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
    allowed = set(".#*01234")
    if any(any(char not in allowed for char in line) for line in lines):
        return None
    return tuple(lines)


def canonical_target(parsed: tuple[str, ...], hidden: dict[str, object]) -> str:
    return "\n".join(parsed)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    rows = int(hidden["rows"])
    cols = int(hidden["cols"])
    example = "#2.*."[:cols].ljust(cols, ".")
    if language == "da":
        return (
            "I din svarblok skal den endelige løsning bruge samme gitterformat som inputtet.\n"
            f"Brug præcis {rows} linjer med {cols} tegn per linje og ingen mellemrum.\n"
            "Behold `#` og talfelter som de er, brug `*` for en lampe og `.` for en tom celle.\n"
            f"Eksempellinje:\n{example}"
        )
    return (
        "In your answer block, the final solution should use the same grid format as the input.\n"
        f"Use exactly {rows} lines with {cols} characters per line and no spaces.\n"
        "Keep `#` and digit cells unchanged, use `*` for a lamp, and `.` for an empty cell.\n"
        f"Example line:\n{example}"
    )


def is_correct(parsed: tuple[str, ...], hidden: dict[str, object]) -> bool:
    return list(parsed) == hidden["solution_grid"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    board_grid = tuple(str(line) for line in hidden["board_grid"])
    models = _solve_models(board_grid, limit=2)
    if len(models) != 1:
        return False
    solution_grid = _render_solution_grid(board_grid, models[0])
    return list(solution_grid) == hidden["solution_grid"]


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
) -> LightUpPuzzle:
    rows = int(recipe["rows"])
    cols = int(recipe["cols"])

    for _ in range(int(recipe["sample_attempts"])):
        wall_board = _sample_wall_board(rows, cols, rng, recipe)
        wall_count = sum(row.count("#") for row in wall_board)
        if wall_count < int(recipe["min_walls"]) or wall_count > int(recipe["max_walls"]):
            continue

        plain_models = _solve_models(wall_board, limit=2)
        if not plain_models:
            continue

        fully_numbered = _number_walls(wall_board, plain_models[0])
        numbered_models = _solve_models(fully_numbered, limit=2)
        if len(numbered_models) != 1:
            continue

        puzzle_board = _strip_numbers(
            fully_numbered,
            rng,
            min_numbered=int(recipe["min_numbered_walls"]),
            max_numbered=int(recipe["max_numbered_walls"]),
        )
        solved_models = _solve_models(puzzle_board, limit=2)
        if len(solved_models) != 1:
            continue

        numbered_wall_count = sum(char.isdigit() for line in puzzle_board for char in line)
        if numbered_wall_count < int(recipe["min_numbered_walls"]):
            continue
        if numbered_wall_count > int(recipe["max_numbered_walls"]):
            continue
        positive_numbered_wall_count = sum(
            char in "1234"
            for line in puzzle_board
            for char in line
        )
        if positive_numbered_wall_count < int(recipe.get("min_positive_numbered_walls", 0)):
            continue
        zero_numbered_wall_count = sum(char == "0" for line in puzzle_board for char in line)
        if zero_numbered_wall_count > int(recipe.get("max_zero_numbered_walls", rows * cols)):
            continue
        if _longest_visibility_segment(puzzle_board) > int(recipe.get("max_visibility_segment", max(rows, cols))):
            continue

        solution_grid = _render_solution_grid(puzzle_board, solved_models[0])
        if recipe.get("require_heuristic_solution"):
            heuristic_solution = _heuristic_solution_grid(puzzle_board)
            if heuristic_solution != solution_grid:
                continue
        prompt = _format_prompt(
            rows,
            cols,
            puzzle_board,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        lamp_count = sum(line.count("*") for line in solution_grid)
        return LightUpPuzzle(
            rows=rows,
            cols=cols,
            board_grid=puzzle_board,
            solution_grid=solution_grid,
            prompt=prompt,
            wall_count=wall_count,
            numbered_wall_count=numbered_wall_count,
            lamp_count=lamp_count,
        )

    raise AssertionError("failed to generate light-up puzzle")


def _sample_wall_board(
    rows: int,
    cols: int,
    rng: Random,
    recipe: dict[str, Any],
) -> tuple[str, ...]:
    wall_prob = float(recipe["wall_prob"])
    board_rows: list[str] = []
    for _ in range(rows):
        chars = []
        for _ in range(cols):
            chars.append("#" if rng.random() < wall_prob else ".")
        board_rows.append("".join(chars))
    return tuple(board_rows)


def _solve_models(board_grid: tuple[str, ...], *, limit: int) -> list[dict[str, int]]:
    solver, variables = _build_solver(board_grid)
    return z3_solver.enumerate_int_models(solver, variables, limit=limit)


def _build_solver(board_grid: tuple[str, ...]) -> tuple[z3.Solver, dict[str, z3.ArithRef]]:
    rows = len(board_grid)
    cols = len(board_grid[0])
    row_segments, col_segments, row_lookup, col_lookup = _visibility_segments(board_grid)

    solver = z3.Solver()
    variables: dict[str, z3.ArithRef] = {}
    lamps: dict[tuple[int, int], z3.ArithRef] = {}

    for row in range(rows):
        for col in range(cols):
            if _is_wall(board_grid[row][col]):
                continue
            var = z3.Int(f"lamp_{row}_{col}")
            solver.add(z3.Or(var == 0, var == 1))
            variables[f"r{row}c{col}"] = var
            lamps[(row, col)] = var

    for segment in row_segments:
        solver.add(z3.Sum([lamps[cell] for cell in segment]) <= 1)
    for segment in col_segments:
        solver.add(z3.Sum([lamps[cell] for cell in segment]) <= 1)

    for cell in lamps:
        visible = set(row_segments[row_lookup[cell]]) | set(col_segments[col_lookup[cell]])
        solver.add(z3.Sum([lamps[pos] for pos in sorted(visible)]) >= 1)

    for row in range(rows):
        for col in range(cols):
            cell = board_grid[row][col]
            if not cell.isdigit():
                continue
            adjacent = [
                lamps[(next_row, next_col)]
                for next_row, next_col in _neighbors(row, col, rows, cols)
                if (next_row, next_col) in lamps
            ]
            solver.add(z3.Sum(adjacent) == int(cell))

    return solver, variables


def _visibility_segments(
    board_grid: tuple[str, ...],
) -> tuple[
    list[tuple[tuple[int, int], ...]],
    list[tuple[tuple[int, int], ...]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], int],
]:
    rows = len(board_grid)
    cols = len(board_grid[0])
    row_segments: list[tuple[tuple[int, int], ...]] = []
    col_segments: list[tuple[tuple[int, int], ...]] = []
    row_lookup: dict[tuple[int, int], int] = {}
    col_lookup: dict[tuple[int, int], int] = {}

    for row in range(rows):
        col = 0
        while col < cols:
            if _is_wall(board_grid[row][col]):
                col += 1
                continue
            segment: list[tuple[int, int]] = []
            while col < cols and not _is_wall(board_grid[row][col]):
                segment.append((row, col))
                col += 1
            row_segments.append(tuple(segment))
            segment_index = len(row_segments) - 1
            for cell in segment:
                row_lookup[cell] = segment_index

    for col in range(cols):
        row = 0
        while row < rows:
            if _is_wall(board_grid[row][col]):
                row += 1
                continue
            segment = []
            while row < rows and not _is_wall(board_grid[row][col]):
                segment.append((row, col))
                row += 1
            col_segments.append(tuple(segment))
            segment_index = len(col_segments) - 1
            for cell in segment:
                col_lookup[cell] = segment_index

    return row_segments, col_segments, row_lookup, col_lookup


def _number_walls(
    board_grid: tuple[str, ...],
    lamp_assignment: dict[str, int],
) -> tuple[str, ...]:
    rows = len(board_grid)
    cols = len(board_grid[0])
    rendered: list[str] = []
    for row, line in enumerate(board_grid):
        chars: list[str] = []
        for col, cell in enumerate(line):
            if cell != "#":
                chars.append(cell)
                continue
            adjacent_lamps = sum(
                lamp_assignment.get(f"r{next_row}c{next_col}", 0)
                for next_row, next_col in _neighbors(row, col, rows, cols)
            )
            chars.append(str(adjacent_lamps))
        rendered.append("".join(chars))
    return tuple(rendered)


def _strip_numbers(
    board_grid: tuple[str, ...],
    rng: Random,
    *,
    min_numbered: int,
    max_numbered: int,
) -> tuple[str, ...]:
    board_rows = [list(row) for row in board_grid]
    numbered_positions = [
        (row, col)
        for row in range(len(board_grid))
        for col in range(len(board_grid[0]))
        if board_grid[row][col].isdigit()
    ]
    rng.shuffle(numbered_positions)
    numbered_count = len(numbered_positions)

    for row, col in numbered_positions:
        if numbered_count <= min_numbered:
            break
        if numbered_count <= max_numbered:
            break
        original = board_rows[row][col]
        board_rows[row][col] = "#"
        candidate = tuple("".join(line) for line in board_rows)
        if len(_solve_models(candidate, limit=2)) == 1:
            numbered_count -= 1
            continue
        board_rows[row][col] = original

    for row, col in numbered_positions:
        if numbered_count <= min_numbered:
            break
        original = board_rows[row][col]
        if original == "#":
            continue
        board_rows[row][col] = "#"
        candidate = tuple("".join(line) for line in board_rows)
        if len(_solve_models(candidate, limit=2)) == 1:
            numbered_count -= 1
            continue
        board_rows[row][col] = original

    return tuple("".join(line) for line in board_rows)


def _render_solution_grid(
    board_grid: tuple[str, ...],
    lamp_assignment: dict[str, int],
) -> tuple[str, ...]:
    rendered: list[str] = []
    for row, line in enumerate(board_grid):
        chars: list[str] = []
        for col, cell in enumerate(line):
            if _is_wall(cell):
                chars.append(cell)
                continue
            chars.append("*" if lamp_assignment[f"r{row}c{col}"] == 1 else ".")
        rendered.append("".join(chars))
    return tuple(rendered)


def _longest_visibility_segment(board_grid: tuple[str, ...]) -> int:
    row_segments, col_segments, _, _ = _visibility_segments(board_grid)
    lengths = [len(segment) for segment in row_segments]
    lengths.extend(len(segment) for segment in col_segments)
    if not lengths:
        return 0
    return max(lengths)


def _heuristic_solution_grid(board_grid: tuple[str, ...]) -> tuple[str, ...] | None:
    rows = len(board_grid)
    cols = len(board_grid[0])
    row_segments, col_segments, row_lookup, col_lookup = _visibility_segments(board_grid)
    empty_cells = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if board_grid[row][col] == "."
    ]
    statuses = {cell: "unknown" for cell in empty_cells}
    visible_map = {
        cell: tuple(
            sorted(
                set(row_segments[row_lookup[cell]]) | set(col_segments[col_lookup[cell]])
            )
        )
        for cell in empty_cells
    }

    changed = True
    while changed:
        changed = False

        for cell, status in list(statuses.items()):
            if status != "lamp":
                continue
            for visible in visible_map[cell]:
                if visible == cell:
                    continue
                assigned, did_change = _assign_empty(statuses, visible, board_grid)
                if not assigned:
                    return None
                changed = changed or did_change

        if not _numbered_walls_feasible(board_grid, statuses):
            return None

        for cell, status in list(statuses.items()):
            if status != "unknown":
                continue
            if _can_place_lamp(board_grid, statuses, visible_map, cell):
                continue
            assigned, did_change = _assign_empty(statuses, cell, board_grid)
            if not assigned:
                return None
            changed = changed or did_change

        if not _numbered_walls_feasible(board_grid, statuses):
            return None

        for row in range(rows):
            for col in range(cols):
                if not board_grid[row][col].isdigit():
                    continue
                required = int(board_grid[row][col])
                adjacent = [
                    pos
                    for pos in _neighbors(row, col, rows, cols)
                    if pos in statuses
                ]
                lamp_count = sum(statuses[pos] == "lamp" for pos in adjacent)
                unknown = [pos for pos in adjacent if statuses[pos] == "unknown"]

                if lamp_count == required:
                    for pos in unknown:
                        assigned, did_change = _assign_empty(statuses, pos, board_grid)
                        if not assigned:
                            return None
                        changed = changed or did_change
                    continue

                if lamp_count + len(unknown) == required:
                    for pos in unknown:
                        assigned, did_change = _assign_lamp(
                            statuses,
                            pos,
                            board_grid,
                            visible_map,
                        )
                        if not assigned:
                            return None
                        changed = changed or did_change

        if not _numbered_walls_feasible(board_grid, statuses):
            return None

        for cell in empty_cells:
            if statuses[cell] == "lamp":
                continue
            if _is_lit(statuses, visible_map, cell):
                continue

            providers = [
                pos
                for pos in visible_map[cell]
                if statuses[pos] != "empty"
                and _can_place_lamp(board_grid, statuses, visible_map, pos)
            ]
            if not providers:
                return None
            if len(providers) == 1:
                assigned, did_change = _assign_lamp(
                    statuses,
                    providers[0],
                    board_grid,
                    visible_map,
                )
                if not assigned:
                    return None
                changed = changed or did_change

    if any(status == "unknown" for status in statuses.values()):
        return None
    if not _numbered_walls_feasible(board_grid, statuses):
        return None
    if not all(_is_lit(statuses, visible_map, cell) for cell in empty_cells):
        return None

    rendered = []
    for row, line in enumerate(board_grid):
        chars = []
        for col, cell in enumerate(line):
            if cell != ".":
                chars.append(cell)
                continue
            chars.append("*" if statuses[(row, col)] == "lamp" else ".")
        rendered.append("".join(chars))
    return tuple(rendered)


def _assign_lamp(
    statuses: dict[tuple[int, int], str],
    cell: tuple[int, int],
    board_grid: tuple[str, ...],
    visible_map: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> tuple[bool, bool]:
    status = statuses[cell]
    if status == "lamp":
        return True, False
    if status == "empty":
        return False, False
    if not _can_place_lamp(board_grid, statuses, visible_map, cell):
        return False, False
    statuses[cell] = "lamp"
    return True, True


def _assign_empty(
    statuses: dict[tuple[int, int], str],
    cell: tuple[int, int],
    board_grid: tuple[str, ...],
) -> tuple[bool, bool]:
    status = statuses[cell]
    if status == "empty":
        return True, False
    if status == "lamp":
        return False, False
    statuses[cell] = "empty"
    if not _numbered_walls_feasible(board_grid, statuses):
        statuses[cell] = "unknown"
        return False, False
    return True, True


def _can_place_lamp(
    board_grid: tuple[str, ...],
    statuses: dict[tuple[int, int], str],
    visible_map: dict[tuple[int, int], tuple[tuple[int, int], ...]],
    cell: tuple[int, int],
) -> bool:
    if statuses[cell] == "empty":
        return False

    for visible in visible_map[cell]:
        if visible == cell:
            continue
        if statuses[visible] == "lamp":
            return False

    row, col = cell
    rows = len(board_grid)
    cols = len(board_grid[0])
    for next_row, next_col in _neighbors(row, col, rows, cols):
        clue = board_grid[next_row][next_col]
        if not clue.isdigit():
            continue
        adjacent = [
            pos
            for pos in _neighbors(next_row, next_col, rows, cols)
            if pos in statuses
        ]
        lamp_count = sum(
            (statuses[pos] == "lamp") or (pos == cell and statuses[pos] != "lamp")
            for pos in adjacent
        )
        unknown_count = sum(
            statuses[pos] == "unknown" and pos != cell
            for pos in adjacent
        )
        required = int(clue)
        if lamp_count > required:
            return False
        if lamp_count + unknown_count < required:
            return False

    return True


def _numbered_walls_feasible(
    board_grid: tuple[str, ...],
    statuses: dict[tuple[int, int], str],
) -> bool:
    rows = len(board_grid)
    cols = len(board_grid[0])
    for row in range(rows):
        for col in range(cols):
            clue = board_grid[row][col]
            if not clue.isdigit():
                continue
            adjacent = [
                pos
                for pos in _neighbors(row, col, rows, cols)
                if pos in statuses
            ]
            lamp_count = sum(statuses[pos] == "lamp" for pos in adjacent)
            unknown_count = sum(statuses[pos] == "unknown" for pos in adjacent)
            required = int(clue)
            if lamp_count > required:
                return False
            if lamp_count + unknown_count < required:
                return False
    return True


def _is_lit(
    statuses: dict[tuple[int, int], str],
    visible_map: dict[tuple[int, int], tuple[tuple[int, int], ...]],
    cell: tuple[int, int],
) -> bool:
    return any(statuses[pos] == "lamp" for pos in visible_map[cell])


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
        "intro": _intro_block(rows, cols, language, recipe, surface_plan),
        "rules": _rules_block(language, surface_plan),
        "grid": _grid_block(board_grid, language, surface_plan),
        "instruction": _instruction_block(language, surface_plan),
        "answer": _answer_block(rows, cols, language, surface_plan),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(
    rows: int,
    cols: int,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    intro_style = str(surface_plan["surface_intro"])
    prompt_style = str(recipe["prompt_style"])
    if language == "da":
        if intro_style == "context":
            first = f"Du får et {rows}x{cols}-gitter til en Akari-opgave."
        elif intro_style == "assignment":
            first = f"Udfyld et {rows}x{cols}-gitter i en Akari-opgave."
        else:
            first = f"Løs en Akari-opgave på et {rows}x{cols}-gitter."
        if prompt_style == "deduce":
            return prompt_surface.PromptBlock(
                key="intro",
                lines=("Brug vægge og talfelter til at finde alle lamper.", first),
            )
        return prompt_surface.PromptBlock(key="intro", lines=(first,))

    if intro_style == "context":
        first = f"You are given a {rows} x {cols} Akari / Light Up grid."
    elif intro_style == "assignment":
        first = f"Fill a {rows} x {cols} Akari / Light Up grid."
    else:
        first = f"Solve an Akari / Light Up puzzle on a {rows} x {cols} grid."
    if prompt_style == "deduce":
        return prompt_surface.PromptBlock(
            key="intro",
            lines=("Use the walls and number clues to place all lamps.", first),
        )
    return prompt_surface.PromptBlock(key="intro", lines=(first,))


def _rules_block(language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    if language == "da":
        if clue_style == "compact":
            lines = (
                "`.` er et tomt felt. Kun tomme felter må indeholde en lampe.",
                "`#` og talfelter er vægge, og vægge blokerer lys.",
                "En lampe lyser sit eget felt samt tomme felter i samme række eller kolonne, indtil lyset rammer en væg.",
                "Hvert tomt felt skal være oplyst af mindst én lampe.",
                "To lamper må ikke lyse på hinanden.",
                "Et talfelt angiver præcis hvor mange lamper der står vandret eller lodret lige ved siden af.",
            )
        else:
            lines = (
                "`.` markerer et tomt felt, og lamper må kun placeres i tomme felter.",
                "`#` og talfelter er vægge, og alle vægge blokerer lys.",
                "En lampe lyser sit eget felt samt de tomme felter i samme række eller kolonne, indtil lyset rammer en væg.",
                "Hvert tomt felt skal være oplyst af mindst én lampe.",
                "To lamper må ikke kunne se hinanden i samme række eller kolonne.",
                "Et talfelt angiver præcis hvor mange lamper der står ortogonalt op ad væggen.",
            )
        return prompt_surface.PromptBlock(key="rules", lines=lines, numbered=True)

    if clue_style == "compact":
        lines = (
            "`.` is an empty cell. Lamps may only be placed in empty cells.",
            "`#` and digit cells are walls, and walls block light.",
            "A lamp lights its own cell and all empty cells in the same row or column until a wall blocks the light.",
            "Every empty cell must be lit by at least one lamp.",
            "No two lamps may shine on each other.",
            "A numbered wall gives the exact number of orthogonally adjacent lamps.",
        )
    else:
        lines = (
            "`.` marks an empty cell, and lamps may only be placed in empty cells.",
            "`#` and digit cells are walls, and every wall blocks light.",
            "A lamp lights its own cell and the empty cells in the same row or column until the light hits a wall.",
            "Every empty cell must be lit by at least one lamp.",
            "No two lamps may see each other in the same row or column.",
            "A numbered wall tells you exactly how many lamps are orthogonally adjacent to that wall.",
        )
    return prompt_surface.PromptBlock(key="rules", lines=lines, numbered=True)


def _grid_block(
    board_grid: tuple[str, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    if language == "da":
        first_line = {
            "plain": "Gitteret er givet rækkevis, med én streng per række:",
            "compact": "Startgitter:",
            "deductive": "Her er startgitteret, hvor vægge og talfelter allerede er markeret:",
        }[clue_style]
    else:
        first_line = {
            "plain": "The grid is given in row-major order, with one string per row:",
            "compact": "Starting grid:",
            "deductive": "Here is the starting grid with walls and numbered walls already marked:",
        }[clue_style]
    return prompt_surface.PromptBlock(key="grid", lines=(first_line, *board_grid))


def _instruction_block(language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    return prompt_surface.PromptBlock(
        key="instruction",
        lines=(_instruction_line(language=language, instruction_variant=str(surface_plan["surface_instruction"])),),
    )


def _instruction_line(*, language: str, instruction_variant: str) -> str:
    if language == "da":
        if instruction_variant == "solve":
            return "Der er præcis én gyldig løsning."
        if instruction_variant == "unique":
            return "Væggene og talfelterne bestemmer én entydig løsning."
        return "Alle regler, vægge og talfelter skal passe med den samme løsning."

    if instruction_variant == "solve":
        return "There is exactly one valid solution."
    if instruction_variant == "unique":
        return "The walls and number clues determine one unique solution."
    return "The rules, walls, and number clues must all fit the same solution."


def _answer_block(
    rows: int,
    cols: int,
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    answer_style = str(surface_plan["surface_answer"])
    example = "#2.*."[:cols].ljust(cols, ".")
    if language == "da":
        if answer_style == "respond":
            lines = (
                f"Inde i det endelige svar skal gitteret have præcis {rows} linjer med {cols} tegn per linje.",
                "Behold `#` og talfelter som de er, brug `*` for en lampe og `.` for et tomt felt.",
            )
        elif answer_style == "write":
            lines = (
                f"Inde i det endelige svar skal du bruge præcis {rows} linjer med {cols} tegn.",
                "Brug samme gitterformat som inputtet.",
            )
        else:
            lines = (
                f"Selve svarindholdet skal være det endelige gitter med præcis {rows} linjer og {cols} tegn per linje.",
                "Vægge og talfelter skal stå uændret.",
            )
        lines += (f"Eksempellinje: {example}",)
        return prompt_surface.PromptBlock(key="answer", lines=lines)

    if answer_style == "respond":
        lines = (
            f"Inside the final response, the grid should use exactly {rows} lines with {cols} characters per line.",
            "Keep `#` and digit cells unchanged, use `*` for a lamp, and `.` for an empty cell.",
        )
    elif answer_style == "write":
        lines = (
            f"Inside the final response, use exactly {rows} lines with {cols} characters.",
            "Use the same grid format as the input.",
        )
    else:
        lines = (
            f"The answer content should be the final grid with exactly {rows} lines and {cols} characters per line.",
            "Walls and digit cells should remain unchanged.",
        )
    lines += (f"Example line: {example}",)
    return prompt_surface.PromptBlock(key="answer", lines=lines)


def _is_wall(cell: str) -> bool:
    return cell != "."


def _neighbors(row: int, col: int, rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    neighbors: list[tuple[int, int]] = []
    for next_row, next_col in (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ):
        if 0 <= next_row < rows and 0 <= next_col < cols:
            neighbors.append((next_row, next_col))
    return tuple(neighbors)


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"].get(key)
