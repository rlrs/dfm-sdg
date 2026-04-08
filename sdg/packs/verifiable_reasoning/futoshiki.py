from collections import Counter
from dataclasses import dataclass
from random import Random
from typing import Any

import z3

from sdg.commons import diversity, prompt_surface, z3_solver


@dataclass(frozen=True)
class FutoshikiClue:
    kind: str
    row: int
    col: int
    value: int | None = None
    operator: str | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "kind": self.kind,
            "row": self.row,
            "col": self.col,
            "value": self.value,
            "operator": self.operator,
        }


@dataclass(frozen=True)
class FutoshikiSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class FutoshikiPuzzle:
    size: int
    solution_grid: tuple[tuple[int, ...], ...]
    clues: tuple[FutoshikiClue, ...]
    prompt: str


SURFACE_SPECS = {
    "board": FutoshikiSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="board",
            block_order=("intro", "rules", "givens", "inequalities", "instruction", "answer"),
        ),
    ),
    "briefing": FutoshikiSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "instruction", "rules", "givens", "inequalities", "answer"),
        ),
    ),
    "deduce": FutoshikiSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "rules", "answer", "instruction", "givens", "inequalities"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_board_4x4",
        "difficulty": "easy",
        "prompt_style": "board",
        "size": 4,
        "clue_profile": "balanced",
        "target_clue_count": 7,
        "max_clue_count": 9,
        "required_kinds": ("given", "horizontal", "vertical"),
        "max_kind_counts": {"given": 4},
        "kind_priority": {"given": 0, "horizontal": 1, "vertical": 2},
    },
    {
        "recipe_id": "easy_briefing_4x4",
        "difficulty": "easy",
        "prompt_style": "briefing",
        "size": 4,
        "clue_profile": "given_heavy",
        "target_clue_count": 8,
        "max_clue_count": 10,
        "required_kinds": ("given", "horizontal", "vertical"),
        "max_kind_counts": {"given": 5},
        "kind_priority": {"given": 0, "horizontal": 1, "vertical": 2},
    },
    {
        "recipe_id": "medium_briefing_4x4",
        "difficulty": "medium",
        "prompt_style": "briefing",
        "size": 4,
        "clue_profile": "inequality_heavy",
        "target_clue_count": 8,
        "max_clue_count": 10,
        "required_kinds": ("given", "horizontal", "vertical"),
        "max_kind_counts": {"given": 3},
        "kind_priority": {"horizontal": 0, "vertical": 1, "given": 2},
    },
    {
        "recipe_id": "medium_board_4x4",
        "difficulty": "medium",
        "prompt_style": "board",
        "size": 4,
        "clue_profile": "vertical_heavy",
        "target_clue_count": 8,
        "max_clue_count": 10,
        "required_kinds": ("given", "horizontal", "vertical"),
        "max_kind_counts": {"given": 2},
        "kind_priority": {"vertical": 0, "horizontal": 1, "given": 2},
    },
    {
        "recipe_id": "medium_deduce_5x5",
        "difficulty": "medium",
        "prompt_style": "deduce",
        "size": 5,
        "clue_profile": "given_heavy",
        "target_clue_count": 13,
        "max_clue_count": 13,
        "required_kinds": ("given", "horizontal", "vertical"),
        "required_kind_counts": {"given": 5, "horizontal": 4, "vertical": 4},
        "max_kind_counts": {"given": 5},
        "kind_priority": {"given": 0, "horizontal": 1, "vertical": 2},
        "selection_strategy": "sample",
        "sample_attempts": 80,
    },
    {
        "recipe_id": "hard_deduce_4x4",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "size": 4,
        "clue_profile": "sparse",
        "target_clue_count": 9,
        "max_clue_count": 11,
        "required_kinds": ("horizontal", "vertical"),
        "max_kind_counts": {"given": 2},
        "kind_priority": {"horizontal": 0, "vertical": 1, "given": 2},
    },
    {
        "recipe_id": "hard_briefing_4x4",
        "difficulty": "hard",
        "prompt_style": "briefing",
        "size": 4,
        "clue_profile": "constraint_heavy",
        "target_clue_count": 10,
        "max_clue_count": 12,
        "required_kinds": ("horizontal", "vertical"),
        "max_kind_counts": {"given": 1},
        "kind_priority": {"vertical": 0, "horizontal": 1, "given": 2},
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
    given_count = sum(clue.kind == "given" for clue in puzzle.clues)

    return {
        "id": f"verifiable-reasoning-{index:05d}",
        "prompt": puzzle.prompt,
        "hidden": {
            "size": puzzle.size,
            "solution_grid": [list(row) for row in puzzle.solution_grid],
            "clues": [clue.to_dict() for clue in puzzle.clues],
        },
        "sources": [{"kind": "dolci_subset", "value": "futoshikipuzzle"}],
        "meta": {
            "family": "futoshiki_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": len(puzzle.clues),
            "given_count": given_count,
            "inequality_count": len(puzzle.clues) - given_count,
            "size": puzzle.size,
            "output_format": "number_grid",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            "clue_profile": recipe["clue_profile"],
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
        "size_coverage": diversity.compare_planned_to_observed(planned, rows, ("size",), observed_getter=_meta_getter),
        "clue_kind_minimums": diversity.counter_minimum_check(clue_kind_counts, dict(required_minimums)),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def clue_from_dict(payload: dict[str, str | int | None]) -> FutoshikiClue:
    return FutoshikiClue(
        kind=str(payload["kind"]),
        row=int(payload["row"]),
        col=int(payload["col"]),
        value=int(payload["value"]) if payload.get("value") is not None else None,
        operator=str(payload["operator"]) if payload.get("operator") is not None else None,
    )


def _generate_puzzle(
    rng: Random,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> FutoshikiPuzzle:
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
        return FutoshikiPuzzle(
            size=size,
            solution_grid=solution_grid,
            clues=clues,
            prompt=prompt,
        )

    raise AssertionError("failed to generate futoshiki puzzle")


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
) -> tuple[FutoshikiClue, ...] | None:
    if recipe.get("selection_strategy") == "sample":
        return _sample_clue_set(solution_grid, recipe, rng)

    candidates = _build_candidates(solution_grid)
    ordered_candidates = _ordered_candidates(candidates, recipe["kind_priority"], rng)
    selected: list[FutoshikiClue] = []
    size = len(solution_grid)
    max_clue_count = int(recipe["max_clue_count"])

    for kind in recipe["required_kinds"]:
        choice = _pick_narrowing_clue_of_kind(
            kind,
            size,
            ordered_candidates,
            selected,
            recipe["max_kind_counts"],
            recipe["kind_priority"],
        )
        if choice is None:
            return None
        selected.append(choice)

    is_unique = _has_unique_solution(size, selected)
    while not is_unique:
        if len(selected) >= max_clue_count:
            return None
        choice = _pick_narrowing_clue(
            size,
            ordered_candidates,
            selected,
            recipe["max_kind_counts"],
            recipe["kind_priority"],
        )
        if choice is None:
            return None
        selected.append(choice)
        is_unique = _has_unique_solution(size, selected)

    while len(selected) < int(recipe["target_clue_count"]):
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


def _sample_clue_set(
    solution_grid: tuple[tuple[int, ...], ...],
    recipe: dict[str, Any],
    rng: Random,
) -> tuple[FutoshikiClue, ...] | None:
    candidates = _build_candidates(solution_grid)
    by_kind = {
        "given": [clue for clue in candidates if clue.kind == "given"],
        "horizontal": [clue for clue in candidates if clue.kind == "horizontal"],
        "vertical": [clue for clue in candidates if clue.kind == "vertical"],
    }
    required_kind_counts = dict(recipe["required_kind_counts"])
    target_clue_count = int(recipe["target_clue_count"])
    sample_attempts = int(recipe.get("sample_attempts", 40))

    for _ in range(sample_attempts):
        selected: list[FutoshikiClue] = []
        for kind, count in required_kind_counts.items():
            pool = list(by_kind[kind])
            rng.shuffle(pool)
            selected.extend(pool[:count])

        if len(selected) < target_clue_count:
            extras = [clue for clue in candidates if clue not in selected]
            rng.shuffle(extras)
            selected.extend(extras[: target_clue_count - len(selected)])

        if not _has_unique_solution(len(solution_grid), selected):
            continue
        ordered = _ordered_candidates(selected, recipe["kind_priority"], rng)
        return tuple(ordered)

    return None


def _build_candidates(solution_grid: tuple[tuple[int, ...], ...]) -> list[FutoshikiClue]:
    size = len(solution_grid)
    candidates: list[FutoshikiClue] = []

    for row in range(size):
        for col in range(size):
            candidates.append(
                FutoshikiClue(
                    "given",
                    row=row + 1,
                    col=col + 1,
                    value=solution_grid[row][col],
                )
            )

    for row in range(size):
        for col in range(size - 1):
            operator = "<" if solution_grid[row][col] < solution_grid[row][col + 1] else ">"
            candidates.append(
                FutoshikiClue(
                    "horizontal",
                    row=row + 1,
                    col=col + 1,
                    operator=operator,
                )
            )

    for row in range(size - 1):
        for col in range(size):
            operator = "<" if solution_grid[row][col] < solution_grid[row + 1][col] else ">"
            candidates.append(
                FutoshikiClue(
                    "vertical",
                    row=row + 1,
                    col=col + 1,
                    operator=operator,
                )
            )

    return candidates


def _pick_narrowing_clue_of_kind(
    kind: str,
    size: int,
    candidates: list[FutoshikiClue],
    selected: list[FutoshikiClue],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> FutoshikiClue | None:
    base_solver, _variables, cell_vars = _build_solver(size, selected)
    best_choice: FutoshikiClue | None = None
    best_score: tuple[int, int, int, int] | None = None

    for clue in _candidate_window(
        candidates,
        selected,
        max_kind_counts,
        kind=kind,
        per_kind_limit=12,
        total_limit=12,
    ):
        constraint = _clue_constraint(cell_vars, clue)
        if not z3_solver.has_model_with(base_solver, z3.Not(constraint)):
            continue
        score = _selection_score(clue, selected, kind_priority)
        if best_score is None or score < best_score:
            best_score = score
            best_choice = clue

    return best_choice


def _pick_narrowing_clue(
    size: int,
    candidates: list[FutoshikiClue],
    selected: list[FutoshikiClue],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> FutoshikiClue | None:
    base_solver, _variables, cell_vars = _build_solver(size, selected)
    best_choice: FutoshikiClue | None = None
    best_score: tuple[int, int, int, int] | None = None

    for clue in _candidate_window(candidates, selected, max_kind_counts):
        constraint = _clue_constraint(cell_vars, clue)
        if not z3_solver.has_model_with(base_solver, z3.Not(constraint)):
            continue
        score = _selection_score(clue, selected, kind_priority)
        if best_score is None or score < best_score:
            best_score = score
            best_choice = clue

    return best_choice


def _pick_extra_clue(
    candidates: list[FutoshikiClue],
    selected: list[FutoshikiClue],
    max_kind_counts: dict[str, int],
    kind_priority: dict[str, int],
) -> FutoshikiClue | None:
    kind_counts = Counter(clue.kind for clue in selected)
    best_clue: FutoshikiClue | None = None
    best_score: tuple[int, int, int] | None = None

    for clue in _candidate_window(candidates, selected, max_kind_counts):
        score = (
            kind_counts[clue.kind],
            _cell_overlap_count(clue, selected),
            kind_priority[clue.kind],
        )
        if best_score is None or score < best_score:
            best_score = score
            best_clue = clue

    return best_clue


def _candidate_window(
    candidates: list[FutoshikiClue],
    selected: list[FutoshikiClue],
    max_kind_counts: dict[str, int],
    *,
    kind: str | None = None,
    per_kind_limit: int = 4,
    total_limit: int = 16,
) -> list[FutoshikiClue]:
    window: list[FutoshikiClue] = []
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


def _selection_score(
    clue: FutoshikiClue,
    selected: list[FutoshikiClue],
    kind_priority: dict[str, int],
) -> tuple[int, int, int, int]:
    kind_count = sum(item.kind == clue.kind for item in selected)
    overlap = _cell_overlap_count(clue, selected)
    return (
        kind_count,
        overlap,
        clue.row,
        kind_priority[clue.kind],
    )


def _cell_overlap_count(clue: FutoshikiClue, selected: list[FutoshikiClue]) -> int:
    cells = set(_clue_cells(clue))
    return sum(bool(cells & set(_clue_cells(item))) for item in selected)


def _clue_cells(clue: FutoshikiClue) -> tuple[tuple[int, int], ...]:
    base = [(clue.row, clue.col)]
    if clue.kind == "horizontal":
        base.append((clue.row, clue.col + 1))
    if clue.kind == "vertical":
        base.append((clue.row + 1, clue.col))
    return tuple(base)


def _solve_grids(
    size: int,
    clues: list[FutoshikiClue] | tuple[FutoshikiClue, ...],
    *,
    limit: int | None = None,
) -> list[tuple[tuple[int, ...], ...]]:
    solver, variables, _ = _build_solver(size, list(clues))
    assignments = z3_solver.enumerate_int_models(solver, variables, limit=limit)
    return [_grid_from_assignment(size, assignment) for assignment in assignments]


def _solve_grid_count(
    size: int,
    clues: list[FutoshikiClue],
    *,
    limit: int | None = None,
) -> tuple[int, bool]:
    solver, variables, _ = _build_solver(size, clues)
    return z3_solver.count_int_models(solver, variables, limit=limit)


def _has_unique_solution(size: int, clues: list[FutoshikiClue]) -> bool:
    count, complete = _solve_grid_count(size, clues, limit=2)
    return count == 1 and complete


def _build_solver(
    size: int,
    clues: list[FutoshikiClue],
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
        solver.add(_clue_constraint(cell_vars, clue))

    return solver, variables, cell_vars


def _clue_constraint(
    cell_vars: dict[tuple[int, int], z3.ArithRef],
    clue: FutoshikiClue,
) -> z3.BoolRef:
    cell = cell_vars[(clue.row, clue.col)]

    if clue.kind == "given":
        assert clue.value is not None
        return cell == clue.value

    assert clue.operator in {"<", ">"}
    if clue.kind == "horizontal":
        other = cell_vars[(clue.row, clue.col + 1)]
    elif clue.kind == "vertical":
        other = cell_vars[(clue.row + 1, clue.col)]
    else:
        raise AssertionError(f"unsupported futoshiki clue kind: {clue.kind}")

    if clue.operator == "<":
        return cell < other
    return cell > other


def _grid_from_assignment(
    size: int,
    assignment: dict[str, int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(assignment[_var_key(row, col)] for col in range(1, size + 1))
        for row in range(1, size + 1)
    )


def _ordered_candidates(
    candidates: list[FutoshikiClue],
    kind_priority: dict[str, int],
    rng: Random,
) -> list[FutoshikiClue]:
    ordered = list(candidates)
    rng.shuffle(ordered)
    ordered.sort(key=lambda clue: (kind_priority[clue.kind], clue.row, clue.col))
    return ordered


def _format_prompt(
    size: int,
    clues: tuple[FutoshikiClue, ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    clue_style = str(surface_plan["surface_clue"])
    givens = tuple(clue for clue in clues if clue.kind == "given")
    inequalities = tuple(clue for clue in clues if clue.kind != "given")
    given_lines, given_heading, given_numbered = _given_block(
        size,
        givens,
        language=language,
        clue_style=clue_style,
    )
    inequality_lines, inequality_heading, inequality_numbered = _inequality_block(
        inequalities,
        language=language,
        clue_style=clue_style,
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
        "givens": prompt_surface.PromptBlock(
            "givens",
            given_lines,
            heading=given_heading,
            numbered=given_numbered,
        ),
        "inequalities": prompt_surface.PromptBlock(
            "inequalities",
            inequality_lines,
            heading=inequality_heading,
            numbered=inequality_numbered,
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
            first = f"Dette er en futoshiki-opgave med et {size}x{size}-gitter."
        elif intro_variant == "assignment":
            first = f"Udfyld et futoshiki-gitter på {size}x{size} felter."
        else:
            first = f"Løs en futoshiki-opgave på et {size}x{size}-gitter."
        indexing = "Rækker tælles oppefra og ned, og kolonner tælles fra venstre mod højre."
        if prompt_style == "deduce":
            return ("Brug reglerne og ulighederne til at udlede hele gitteret.", first, indexing)
        return (first, indexing)

    if intro_variant == "context":
        first = f"This is a futoshiki puzzle on a {size}x{size} grid."
    elif intro_variant == "assignment":
        first = f"Fill a {size}x{size} futoshiki grid."
    else:
        first = f"Solve a futoshiki puzzle on a {size}x{size} grid."
    indexing = "Rows are counted from top to bottom, and columns from left to right."
    if prompt_style == "deduce":
        return ("Use the rules and inequalities to work out the full grid.", first, indexing)
    return (first, indexing)


def _rule_lines(*, language: str, size: int) -> tuple[str, ...]:
    if language == "da":
        return (
            f"Brug tallene 1 til {size}.",
            "Hver række skal indeholde hvert tal præcis én gang.",
            "Hver kolonne skal indeholde hvert tal præcis én gang.",
            "Alle uligheder skal være opfyldt.",
        )

    return (
        f"Use the digits 1 to {size}.",
        "Each row must contain each digit exactly once.",
        "Each column must contain each digit exactly once.",
        "All inequalities must be satisfied.",
    )


def _instruction_line(*, language: str, instruction_variant: str) -> str:
    if language == "da":
        if instruction_variant == "solve":
            return "Der er præcis én gyldig løsning."
        if instruction_variant == "unique":
            return "De faste felter og ulighederne bestemmer én entydig løsning."
        return "Alle regler, faste felter og uligheder skal passe med den samme løsning."

    if instruction_variant == "solve":
        return "There is exactly one valid solution."
    if instruction_variant == "unique":
        return "The givens and inequalities determine one unique solution."
    return "The rules, givens, and inequalities must all fit the same solution."


def _answer_lines(*, language: str, size: int, answer_variant: str) -> tuple[str, ...]:
    example_row = " ".join(str(number) for number in range(1, size + 1))
    example_grid = tuple(example_row for _ in range(size))

    if language == "da":
        first_line = {
            "respond": f"Når du giver dit endelige svar, skal gitteret have præcis {size} linjer i dette format:",
            "write": f"Når du skriver det endelige svar, skal du bruge præcis {size} linjer i dette format:",
            "complete": f"Giv det endelige gitter med præcis {size} linjer i dette format:",
        }[answer_variant]
        return (first_line, *example_grid)

    first_line = {
        "respond": f"When you give your final answer, the grid should use exactly {size} lines in this format:",
        "write": f"When you write the final answer, use exactly {size} lines in this format:",
        "complete": f"Give the final grid using exactly {size} lines in this format:",
    }[answer_variant]
    return (first_line, *example_grid)


def _given_block(
    size: int,
    givens: tuple[FutoshikiClue, ...],
    *,
    language: str,
    clue_style: str,
) -> tuple[tuple[str, ...], str, bool]:
    if clue_style != "compact":
        return (
            tuple(_format_clue(clue, language=language, clue_style=clue_style) for clue in givens),
            "Faste felter" if language == "da" else "Given cells",
            True,
        )

    grid = [["." for _ in range(size)] for _ in range(size)]
    for clue in givens:
        assert clue.value is not None
        grid[clue.row - 1][clue.col - 1] = str(clue.value)

    lines = tuple(
        f"{_row_label(row + 1, language=language)}: {' '.join(values)}"
        for row, values in enumerate(grid)
    )
    heading = "Startgitter" if language == "da" else "Starting grid"
    return lines, heading, False


def _inequality_block(
    inequalities: tuple[FutoshikiClue, ...],
    *,
    language: str,
    clue_style: str,
) -> tuple[tuple[str, ...], str, bool]:
    if clue_style != "compact":
        return (
            tuple(_format_clue(clue, language=language, clue_style=clue_style) for clue in inequalities),
            "Uligheder" if language == "da" else "Inequalities",
            True,
        )

    horizontal = [
        _format_clue(clue, language=language, clue_style=clue_style)
        for clue in inequalities
        if clue.kind == "horizontal"
    ]
    vertical = [
        _format_clue(clue, language=language, clue_style=clue_style)
        for clue in inequalities
        if clue.kind == "vertical"
    ]

    lines: list[str] = []
    if horizontal:
        lines.append("Vandret:" if language == "da" else "Horizontal:")
        lines.extend(horizontal)
    if vertical:
        lines.append("Lodret:" if language == "da" else "Vertical:")
        lines.extend(vertical)
    return tuple(lines), "Uligheder" if language == "da" else "Inequalities", False


def _format_clue(
    clue: FutoshikiClue,
    *,
    language: str,
    clue_style: str,
) -> str:
    if clue.kind == "given":
        assert clue.value is not None
        return _format_given(clue.row, clue.col, clue.value, language=language, clue_style=clue_style)
    assert clue.operator is not None
    return _format_relation(clue, language=language, clue_style=clue_style)


def _format_given(
    row: int,
    col: int,
    value: int,
    *,
    language: str,
    clue_style: str,
) -> str:
    compact_ref = _cell_ref(row, col, language=language, compact=True)
    long_ref = _cell_ref(row, col, language=language, compact=False)

    if language == "da":
        if clue_style == "compact":
            return f"{compact_ref} = {value}"
        if clue_style == "deductive":
            return f"Tallet i {long_ref} er {value}."
        return f"Feltet i {long_ref} er {value}."

    if clue_style == "compact":
        return f"{compact_ref} = {value}"
    if clue_style == "deductive":
        return f"The value in {long_ref} is {value}."
    return f"The cell in {long_ref} is {value}."


def _format_relation(
    clue: FutoshikiClue,
    *,
    language: str,
    clue_style: str,
) -> str:
    first_compact = _cell_ref(clue.row, clue.col, language=language, compact=True)
    first_long = _cell_ref(clue.row, clue.col, language=language, compact=False)
    if clue.kind == "horizontal":
        second_row, second_col = clue.row, clue.col + 1
    else:
        second_row, second_col = clue.row + 1, clue.col
    second_compact = _cell_ref(second_row, second_col, language=language, compact=True)
    second_long = _cell_ref(second_row, second_col, language=language, compact=False)
    operator_word = _operator_word(str(clue.operator), language=language)

    if clue_style == "compact":
        return f"{first_compact} {clue.operator} {second_compact}"

    if language == "da":
        if clue_style == "deductive":
            return f"Tallet i {first_long} skal være {operator_word} tallet i {second_long}."
        return f"Feltet i {first_long} er {operator_word} feltet i {second_long}."

    if clue_style == "deductive":
        return f"The value in {first_long} must be {operator_word} the value in {second_long}."
    return f"The cell in {first_long} is {operator_word} the cell in {second_long}."


def _cell_ref(row: int, col: int, *, language: str, compact: bool) -> str:
    if compact:
        if language == "da":
            return f"r{row}k{col}"
        return f"r{row}c{col}"
    if language == "da":
        return f"række {row}, kolonne {col}"
    return f"row {row}, column {col}"


def _row_label(row: int, *, language: str) -> str:
    if language == "da":
        return f"r{row}"
    return f"r{row}"


def _operator_word(operator: str, *, language: str) -> str:
    if language == "da":
        return "mindre end" if operator == "<" else "større end"
    return "less than" if operator == "<" else "greater than"


def _recipe_by_id(recipe_id: str) -> dict[str, Any]:
    for recipe in RECIPES:
        if recipe["recipe_id"] == recipe_id:
            return recipe
    raise AssertionError(f"unknown futoshiki recipe: {recipe_id}")


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]


def _var_key(row: int, col: int) -> str:
    return f"r{row}c{col}"


def _exceeds_kind_limit(selected: list[FutoshikiClue], clue: FutoshikiClue, max_kind_counts: dict[str, int]) -> bool:
    if clue.kind not in max_kind_counts:
        return False
    count = sum(item.kind == clue.kind for item in selected)
    return count >= max_kind_counts[clue.kind]
