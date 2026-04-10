from dataclasses import dataclass
from random import Random
from typing import Any

import z3

from sdg.commons import diversity, prompt_surface, z3_solver


@dataclass(frozen=True)
class KakurasuSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class KakurasuPuzzle:
    size: int
    row_sums: tuple[int, ...]
    col_sums: tuple[int, ...]
    solution_mask: tuple[str, ...]
    prompt: str
    filled_count: int


SURFACE_SPECS = {
    "board": KakurasuSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="board",
            block_order=("intro", "rules", "clues", "instruction", "answer"),
        ),
    ),
    "briefing": KakurasuSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "instruction", "rules", "clues", "answer"),
        ),
    ),
    "deduce": KakurasuSurfaceSpec(
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
        "clue_profile": "balanced",
        "min_filled": 5,
        "max_filled": 7,
        "min_partial_rows": 3,
        "min_partial_cols": 3,
    },
    {
        "recipe_id": "easy_briefing_5x5",
        "difficulty": "easy",
        "prompt_style": "briefing",
        "size": 5,
        "clue_profile": "light",
        "min_filled": 6,
        "max_filled": 9,
        "min_partial_rows": 4,
        "min_partial_cols": 4,
    },
    {
        "recipe_id": "medium_board_5x5",
        "difficulty": "medium",
        "prompt_style": "board",
        "size": 5,
        "clue_profile": "balanced_dense",
        "min_filled": 9,
        "max_filled": 13,
        "min_partial_rows": 5,
        "min_partial_cols": 5,
    },
    {
        "recipe_id": "medium_deduce_6x6",
        "difficulty": "medium",
        "prompt_style": "deduce",
        "size": 6,
        "clue_profile": "balanced",
        "min_filled": 10,
        "max_filled": 15,
        "min_partial_rows": 5,
        "min_partial_cols": 5,
    },
    {
        "recipe_id": "hard_board_6x6",
        "difficulty": "hard",
        "prompt_style": "board",
        "size": 6,
        "clue_profile": "dense",
        "min_filled": 14,
        "max_filled": 20,
        "min_partial_rows": 6,
        "min_partial_cols": 6,
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
            "size": puzzle.size,
            "row_sums": list(puzzle.row_sums),
            "col_sums": list(puzzle.col_sums),
            "solution_mask": list(puzzle.solution_mask),
        },
        "sources": [{"kind": "dolci_subset", "value": "kakurasu"}],
        "meta": {
            "family": "kakurasu_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": puzzle.size * 2,
            "given_count": puzzle.size * 2,
            "size": puzzle.size,
            "filled_count": puzzle.filled_count,
            "output_format": "binary_grid",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            "clue_profile": recipe["clue_profile"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[str, ...] | None:
    size = int(hidden["size"])
    lines = [line.strip().replace(" ", "") for line in text.splitlines() if line.strip()]
    if len(lines) != size:
        return None
    if any(len(line) != size for line in lines):
        return None
    if all(all(char in {"0", "1"} for char in line) for line in lines):
        return tuple(
            "".join("*" if char == "1" else "." for char in line)
            for line in lines
        )
    if all(all(char in {".", "*"} for char in line) for line in lines):
        return tuple(lines)
    return None


def canonical_target(parsed: tuple[str, ...], hidden: dict[str, object]) -> str:
    return "\n".join(
        "".join("1" if char == "*" else "0" for char in line)
        for line in parsed
    )


def answer_contract(hidden: dict[str, object], language: str) -> str:
    size = int(hidden["size"])
    example = "0" * max(1, size // 2) + "1" * (size - max(1, size // 2))
    if language == "da":
        return (
            "I din svarblok skal den endelige løsning være et 0/1-gitter med én række per linje.\n"
            f"Brug præcis {size} linjer med {size} tegn per linje og ingen mellemrum.\n"
            "Brug '1' for et markeret felt og '0' for et tomt felt.\n"
            f"Eksempellinje:\n{example}"
        )
    return (
        "In your answer block, the final solution should be a 0/1 grid with one row per line.\n"
        f"Use exactly {size} lines with {size} characters per line and no spaces.\n"
        "Use '1' for a filled cell and '0' for an empty cell.\n"
        f"Example line:\n{example}"
    )


def is_correct(parsed: tuple[str, ...], hidden: dict[str, object]) -> bool:
    return list(parsed) == hidden["solution_mask"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    size = int(hidden["size"])
    row_sums = tuple(int(value) for value in hidden["row_sums"])
    col_sums = tuple(int(value) for value in hidden["col_sums"])
    solutions = _solve_masks(size, row_sums, col_sums, limit=2)
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
        "size_coverage": diversity.compare_planned_to_observed(planned, rows, ("size",), observed_getter=_meta_getter),
        "unique_prompts": diversity.unique_count_check([str(row["prompt"]) for row in rows], len(rows)),
    }


def _generate_puzzle(
    rng: Random,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> KakurasuPuzzle:
    size = int(recipe["size"])

    for _ in range(300):
        solution_mask = _sample_mask(size, rng, recipe)
        if solution_mask is None:
            continue
        row_sums, col_sums = _weighted_sums(solution_mask)
        solutions = _solve_masks(size, row_sums, col_sums, limit=2)
        if len(solutions) != 1:
            continue
        if solutions[0] != solution_mask:
            continue
        prompt = _format_prompt(
            size,
            row_sums,
            col_sums,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return KakurasuPuzzle(
            size=size,
            row_sums=row_sums,
            col_sums=col_sums,
            solution_mask=solution_mask,
            prompt=prompt,
            filled_count=sum(line.count("*") for line in solution_mask),
        )

    raise AssertionError("failed to generate kakurasu puzzle")


def _sample_mask(
    size: int,
    rng: Random,
    recipe: dict[str, Any],
) -> tuple[str, ...] | None:
    min_filled = int(recipe["min_filled"])
    max_filled = int(recipe["max_filled"])

    for _ in range(240):
        filled_count = rng.randint(min_filled, max_filled)
        cells = [(row, col) for row in range(size) for col in range(size)]
        chosen = set(rng.sample(cells, k=filled_count))
        mask = tuple(
            "".join("*" if (row, col) in chosen else "." for col in range(size))
            for row in range(size)
        )
        if _interesting_mask(mask, recipe):
            return mask

    return None


def _interesting_mask(mask: tuple[str, ...], recipe: dict[str, Any]) -> bool:
    size = len(mask)
    row_counts = [line.count("*") for line in mask]
    col_counts = [sum(mask[row][col] == "*" for row in range(size)) for col in range(size)]
    partial_rows = sum(0 < count < size for count in row_counts)
    partial_cols = sum(0 < count < size for count in col_counts)

    if partial_rows < int(recipe["min_partial_rows"]):
        return False
    if partial_cols < int(recipe["min_partial_cols"]):
        return False
    return True


def _weighted_sums(mask: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    size = len(mask)
    row_sums = tuple(
        sum(col + 1 for col, cell in enumerate(mask[row]) if cell == "*")
        for row in range(size)
    )
    col_sums = tuple(
        sum(row + 1 for row in range(size) if mask[row][col] == "*")
        for col in range(size)
    )
    return row_sums, col_sums


def _solve_masks(
    size: int,
    row_sums: tuple[int, ...],
    col_sums: tuple[int, ...],
    *,
    limit: int | None = None,
) -> list[tuple[str, ...]]:
    solver = z3.Solver()
    variables: dict[str, z3.ArithRef] = {}
    cells: list[list[z3.ArithRef]] = []

    for row in range(size):
        solver_row: list[z3.ArithRef] = []
        for col in range(size):
            var = z3.Int(f"cell_{row}_{col}")
            solver.add(z3.Or(var == 0, var == 1))
            variables[f"r{row}c{col}"] = var
            solver_row.append(var)
        cells.append(solver_row)

    for row in range(size):
        solver.add(z3.Sum([(col + 1) * cells[row][col] for col in range(size)]) == row_sums[row])
    for col in range(size):
        solver.add(z3.Sum([(row + 1) * cells[row][col] for row in range(size)]) == col_sums[col])

    models = z3_solver.enumerate_int_models(solver, variables, limit=limit)
    solutions: list[tuple[str, ...]] = []
    for model in models:
        solution = tuple(
            "".join("*" if model[f"r{row}c{col}"] == 1 else "." for col in range(size))
            for row in range(size)
        )
        solutions.append(solution)
    return solutions


def _format_prompt(
    size: int,
    row_sums: tuple[int, ...],
    col_sums: tuple[int, ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    prompt_style = str(recipe["prompt_style"])
    surface_spec = SURFACE_SPECS[prompt_style]

    blocks = {
        "intro": _intro_block(size, language, surface_plan),
        "rules": _rules_block(size, language),
        "clues": _clues_block(row_sums, col_sums, language, surface_plan),
        "instruction": _instruction_block(language, surface_plan),
        "answer": _answer_block(size, language, surface_plan),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(size: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    intro_style = str(surface_plan["surface_intro"])
    if language == "da":
        if intro_style == "context":
            lines = (
                f"Du får et {size}x{size}-gitter (1-indekseret). Udfyld gitteret med 0'er og 1-taller sådan at:",
            )
        elif intro_style == "assignment":
            lines = (
                f"Her er en kakurasu-opgave på et {size}x{size}-gitter (1-indekseret). Bestem et 0/1-gitter sådan at:",
            )
        else:
            lines = (
                f"Løs en kakurasu-opgave på et {size}x{size}-gitter (1-indekseret). Indsæt 0'er og 1-taller sådan at:",
            )
        return prompt_surface.PromptBlock(key="intro", lines=lines)

    if intro_style == "context":
        lines = (
            f"You are given a {size} x {size} grid (1-indexed). Fill the grid with `0`s and `1`s such that:",
        )
    elif intro_style == "assignment":
        lines = (
            f"Consider a {size} x {size} kakurasu grid (1-indexed). Determine a `0/1` grid such that:",
        )
    else:
        lines = (
            f"Solve a {size} x {size} kakurasu grid (1-indexed). Place `0`s and `1`s so that:",
        )
    return prompt_surface.PromptBlock(key="intro", lines=lines)


def _rules_block(size: int, language: str) -> prompt_surface.PromptBlock:
    if language == "da":
        lines = (
            "For hver række i er summen af kolonneindekserne for felter med 1 lig A[i].",
            "For hver kolonne j er summen af rækkeindekserne for felter med 1 lig B[j].",
        )
        return prompt_surface.PromptBlock(key="rules", lines=lines)

    lines = (
        "For each row `i`, the sum of the column indices where there are `1`s is equal to `A[i]`.",
        "For each column `j`, the sum of the row indices where there are `1`s is equal to `B[j]`.",
    )
    return prompt_surface.PromptBlock(key="rules", lines=lines)


def _clues_block(
    row_sums: tuple[int, ...],
    col_sums: tuple[int, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])

    if language == "da":
        if clue_style == "plain":
            lines = (
                "A = [" + ", ".join(str(value) for value in row_sums) + "]",
                "B = [" + ", ".join(str(value) for value in col_sums) + "]",
            )
        elif clue_style == "compact":
            lines = (
                "Rækkearray A: [" + ", ".join(str(value) for value in row_sums) + "]",
                "Kolonnearray B: [" + ", ".join(str(value) for value in col_sums) + "]",
            )
        else:
            lines = (
                "Array A er givet ved: [" + ", ".join(str(value) for value in row_sums) + "]",
                "Array B er givet ved: [" + ", ".join(str(value) for value in col_sums) + "]",
            )
        return prompt_surface.PromptBlock(key="clues", lines=lines)

    if clue_style == "plain":
        lines = (
            "A = [" + ", ".join(str(value) for value in row_sums) + "]",
            "B = [" + ", ".join(str(value) for value in col_sums) + "]",
        )
    elif clue_style == "compact":
        lines = (
            "Array A is: [" + ", ".join(str(value) for value in row_sums) + "]",
            "Array B is: [" + ", ".join(str(value) for value in col_sums) + "]",
        )
    else:
        lines = (
            "The row array is given by A = [" + ", ".join(str(value) for value in row_sums) + "]",
            "The column array is given by B = [" + ", ".join(str(value) for value in col_sums) + "]",
        )
    return prompt_surface.PromptBlock(key="clues", lines=lines)


def _instruction_block(language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    instruction_style = str(surface_plan["surface_instruction"])
    if language == "da":
        if instruction_style == "solve":
            lines = ("Find hele 0/1-gitteret.",)
        elif instruction_style == "unique":
            lines = ("A og B bestemmer én entydig løsning.",)
        else:
            lines = ("Bestem det eneste 0/1-gitter, der opfylder begge arrays.",)
        return prompt_surface.PromptBlock(key="instruction", lines=lines)

    if instruction_style == "solve":
        lines = ("Determine the full `0/1` grid.",)
    elif instruction_style == "unique":
        lines = ("The arrays `A` and `B` determine one unique solution.",)
    else:
        lines = ("Find the unique `0/1` grid that satisfies both arrays together.",)
    return prompt_surface.PromptBlock(key="instruction", lines=lines)


def _answer_block(size: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    answer_style = str(surface_plan["surface_answer"])
    example = "0" * max(1, size // 2) + "1" * (size - max(1, size // 2))

    if language == "da":
        if answer_style == "respond":
            lines = (
                f"Svarformat: Skriv præcis {size} linjer med {size} tegn per linje, kun `0` eller `1`, uden mellemrum.",
            )
        elif answer_style == "write":
            lines = (
                f"Outputformat: Brug præcis {size} linjer med {size} tegn (`0` eller `1`) uden mellemrum.",
            )
        else:
            lines = (
                f"Det endelige svar skal bestå af præcis {size} linjer med {size} tegn, kun `0` eller `1`.",
            )
        lines += (f"Eksempel: {example}",)
        return prompt_surface.PromptBlock(key="answer", lines=lines)

    if answer_style == "respond":
        lines = (
            f"Output Format: Write exactly {size} lines with {size} characters per line, using only `0` or `1`, with no separators.",
        )
    elif answer_style == "write":
        lines = (
            f"Output Format: Use exactly {size} lines with {size} characters (`0` or `1`) and no spaces.",
        )
    else:
        lines = (
            f"Your final answer should consist of exactly {size} lines with {size} characters, using only `0` or `1`.",
        )
    lines += (f"Example: {example}",)
    return prompt_surface.PromptBlock(key="answer", lines=lines)


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"].get(key)
