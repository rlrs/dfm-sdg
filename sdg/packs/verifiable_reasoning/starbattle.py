from dataclasses import dataclass
from functools import cache
from itertools import permutations
from random import Random
from string import ascii_uppercase
from typing import Any

from sdg.commons import diversity, prompt_surface


@dataclass(frozen=True)
class StarBattleSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class StarBattlePuzzle:
    size: int
    region_grid: tuple[str, ...]
    solution_mask: tuple[str, ...]
    prompt: str


SURFACE_SPECS = {
    "board": StarBattleSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="board",
            block_order=("intro", "rules", "grid", "instruction", "answer"),
        ),
    ),
    "briefing": StarBattleSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "grid", "rules", "instruction", "answer"),
        ),
    ),
    "deduce": StarBattleSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "instruction", "rules", "grid", "answer"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_square_5x5",
        "difficulty": "easy",
        "prompt_style": "board",
        "size": 5,
        "region_profile": "compact",
        "sample_attempts": 32,
        "restart_attempts": 20,
        "local_search_steps": 80,
    },
    {
        "recipe_id": "medium_square_5x5",
        "difficulty": "medium",
        "prompt_style": "briefing",
        "size": 5,
        "region_profile": "spread",
        "sample_attempts": 40,
        "restart_attempts": 24,
        "local_search_steps": 100,
    },
    {
        "recipe_id": "medium_square_6x6",
        "difficulty": "medium",
        "prompt_style": "board",
        "size": 6,
        "region_profile": "compact",
        "sample_attempts": 36,
        "restart_attempts": 20,
        "local_search_steps": 100,
    },
    {
        "recipe_id": "hard_square_6x6",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "size": 6,
        "region_profile": "spread",
        "sample_attempts": 48,
        "restart_attempts": 28,
        "local_search_steps": 120,
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
            "region_grid": list(puzzle.region_grid),
            "solution_mask": list(puzzle.solution_mask),
        },
        "sources": [{"kind": "dolci_subset", "value": "starbattle"}],
        "meta": {
            "family": "starbattle_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": puzzle.size * puzzle.size,
            "given_count": puzzle.size * puzzle.size,
            "size": puzzle.size,
            "output_format": "mask_grid",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            "region_profile": recipe["region_profile"],
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
    if any(any(char not in {".", "*"} for char in line) for line in lines):
        return None
    return tuple(lines)


def canonical_target(parsed: tuple[str, ...], hidden: dict[str, object]) -> str:
    return "\n".join(parsed)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    size = int(hidden["size"])
    example = "." * max(0, size - 1) + "*"
    if language == "da":
        return (
            "I din svarblok skal den endelige løsning være en maske med én række per linje.\n"
            f"Brug præcis {size} linjer med {size} tegn per linje og ingen mellemrum.\n"
            "Brug '*' for en stjerne og '.' for et tomt felt.\n"
            f"Eksempellinje:\n{example}"
        )
    return (
        "In your answer block, the final solution should be a mask with one row per line.\n"
        f"Use exactly {size} lines with {size} characters per line and no spaces.\n"
        "Use '*' for a star and '.' for an empty cell.\n"
        f"Example line:\n{example}"
    )


def is_correct(parsed: tuple[str, ...], hidden: dict[str, object]) -> bool:
    return list(parsed) == hidden["solution_mask"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    size = int(hidden["size"])
    region_grid = tuple(str(line) for line in hidden["region_grid"])
    solutions = _solve_masks(size, region_grid)
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
) -> StarBattlePuzzle:
    size = int(recipe["size"])
    candidate_solutions = _candidate_solutions(size)

    for _ in range(int(recipe["sample_attempts"])):
        solution = rng.choice(candidate_solutions)
        region_ids = _search_region_partition(size, solution, candidate_solutions, rng, recipe)
        if region_ids is None:
            continue
        if _valid_solution_count(region_ids, candidate_solutions) != 1:
            continue

        region_grid = _label_region_grid(region_ids, rng)
        solution_mask = _solution_mask(size, solution)
        prompt = _format_prompt(
            size,
            region_grid,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return StarBattlePuzzle(
            size=size,
            region_grid=region_grid,
            solution_mask=solution_mask,
            prompt=prompt,
        )

    raise AssertionError("failed to generate starbattle puzzle")


@cache
def _candidate_solutions(size: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    candidates: list[tuple[tuple[int, int], ...]] = []
    for columns in permutations(range(size)):
        if not all(abs(columns[row] - columns[row - 1]) > 1 for row in range(1, size)):
            continue
        candidates.append(tuple((row, columns[row]) for row in range(size)))
    return tuple(candidates)


def _search_region_partition(
    size: int,
    solution: tuple[tuple[int, int], ...],
    candidate_solutions: tuple[tuple[tuple[int, int], ...], ...],
    rng: Random,
    recipe: dict[str, Any],
) -> tuple[tuple[int, ...], ...] | None:
    for _ in range(int(recipe["restart_attempts"])):
        region_ids = _grow_seeded_partition(size, solution, rng, region_profile=str(recipe["region_profile"]))
        if region_ids is None:
            continue

        current = _valid_solution_count(region_ids, candidate_solutions)
        if current == 1:
            return region_ids

        for _ in range(int(recipe["local_search_steps"])):
            moves = _legal_boundary_moves(region_ids, solution)
            if not moves:
                break

            best_grid = None
            best_key = None
            for move in moves:
                candidate_grid = _apply_move(region_ids, move)
                candidate_count = _valid_solution_count(candidate_grid, candidate_solutions)
                key = (
                    candidate_count,
                    _size_penalty(candidate_grid),
                    _shape_penalty(candidate_grid),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_grid = candidate_grid

            if best_grid is None or best_key is None or best_key[0] >= current:
                break

            region_ids = best_grid
            current = best_key[0]
            if current == 1:
                return region_ids

    return None


def _grow_seeded_partition(
    size: int,
    solution: tuple[tuple[int, int], ...],
    rng: Random,
    *,
    region_profile: str,
) -> tuple[tuple[int, ...], ...] | None:
    region_ids = [[-1] * size for _ in range(size)]
    frontiers = [set() for _ in range(size)]
    sizes = [1] * size

    for region, (row, col) in enumerate(solution):
        region_ids[row][col] = region

    for region, (row, col) in enumerate(solution):
        for neighbor in _neighbors(row, col, size):
            next_row, next_col = neighbor
            if region_ids[next_row][next_col] == -1:
                frontiers[region].add(neighbor)

    while sum(sizes) < size * size:
        growable = [region for region in range(size) if frontiers[region]]
        if not growable:
            return None

        min_size = min(sizes[region] for region in growable)
        candidate_regions = [region for region in growable if sizes[region] <= min_size + 1]
        region = rng.choice(candidate_regions)

        scored: list[tuple[float, tuple[int, int]]] = []
        for row, col in list(frontiers[region]):
            if region_ids[row][col] != -1:
                continue
            score = _growth_score(
                region_ids,
                region,
                row,
                col,
                solution,
                sizes,
                size,
                rng,
                region_profile=region_profile,
            )
            scored.append((score, (row, col)))

        if not scored:
            frontiers[region].clear()
            continue

        best_score = max(score for score, _ in scored)
        best_cells = [cell for score, cell in scored if score >= best_score - 1e-9]
        row, col = rng.choice(best_cells)
        region_ids[row][col] = region
        sizes[region] += 1

        for other in range(size):
            frontiers[other].discard((row, col))
        for next_row, next_col in _neighbors(row, col, size):
            if region_ids[next_row][next_col] == -1:
                frontiers[region].add((next_row, next_col))

    return tuple(tuple(row) for row in region_ids)


def _growth_score(
    region_ids: list[list[int]],
    region: int,
    row: int,
    col: int,
    solution: tuple[tuple[int, int], ...],
    sizes: list[int],
    size: int,
    rng: Random,
    *,
    region_profile: str,
) -> float:
    same_neighbors = sum(1 for next_row, next_col in _neighbors(row, col, size) if region_ids[next_row][next_col] == region)
    seed_row, seed_col = solution[region]
    distance = abs(row - seed_row) + abs(col - seed_col)
    open_neighbors = sum(1 for next_row, next_col in _neighbors(row, col, size) if region_ids[next_row][next_col] == -1)

    score = 0.0
    score += 2.0 * same_neighbors
    score += 0.25 * open_neighbors
    score -= 0.35 * max(0, sizes[region] + 1 - size)
    score += rng.random() * 0.1

    if region_profile == "spread":
        score += 0.18 * distance
    else:
        score -= 0.30 * distance

    return score


def _valid_solution_count(
    region_ids: tuple[tuple[int, ...], ...],
    candidate_solutions: tuple[tuple[tuple[int, int], ...], ...],
) -> int:
    size = len(region_ids)
    valid = 0
    for solution in candidate_solutions:
        counts = [0] * size
        for row, col in solution:
            counts[region_ids[row][col]] += 1
        if counts == [1] * size:
            valid += 1
    return valid


def _solve_masks(size: int, region_grid: tuple[str, ...]) -> list[tuple[str, ...]]:
    label_lookup = {label: index for index, label in enumerate(sorted(set("".join(region_grid))))}
    region_ids = tuple(
        tuple(label_lookup[label] for label in row)
        for row in region_grid
    )
    solutions: list[tuple[str, ...]] = []
    for solution in _candidate_solutions(size):
        counts = [0] * size
        for row, col in solution:
            counts[region_ids[row][col]] += 1
        if counts == [1] * size:
            solutions.append(_solution_mask(size, solution))
    return solutions


def _solution_mask(size: int, solution: tuple[tuple[int, int], ...]) -> tuple[str, ...]:
    star_columns = {row: col for row, col in solution}
    return tuple(
        "".join("*" if star_columns[row] == col else "." for col in range(size))
        for row in range(size)
    )


def _legal_boundary_moves(
    region_ids: tuple[tuple[int, ...], ...],
    solution: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[int, int], int]]:
    size = len(region_ids)
    seed_cells = set(solution)
    regions = [_region_cells(region_ids, region) for region in range(size)]
    moves: list[tuple[tuple[int, int], int]] = []

    for row in range(size):
        for col in range(size):
            if (row, col) in seed_cells:
                continue

            source = region_ids[row][col]
            source_cells = regions[source]
            if len(source_cells) <= 1:
                continue

            remaining = source_cells - {(row, col)}
            if not _is_connected(remaining, size):
                continue

            targets = {
                region_ids[next_row][next_col]
                for next_row, next_col in _neighbors(row, col, size)
                if region_ids[next_row][next_col] != source
            }
            for target in targets:
                moves.append(((row, col), target))

    return moves


def _apply_move(
    region_ids: tuple[tuple[int, ...], ...],
    move: tuple[tuple[int, int], int],
) -> tuple[tuple[int, ...], ...]:
    (row, col), target = move
    rows = [list(line) for line in region_ids]
    rows[row][col] = target
    return tuple(tuple(line) for line in rows)


def _size_penalty(region_ids: tuple[tuple[int, ...], ...]) -> int:
    size = len(region_ids)
    region_sizes = [len(_region_cells(region_ids, region)) for region in range(size)]
    return max(region_sizes) - min(region_sizes)


def _shape_penalty(region_ids: tuple[tuple[int, ...], ...]) -> int:
    size = len(region_ids)
    penalty = 0
    for region in range(size):
        cells = _region_cells(region_ids, region)
        rows = [row for row, _ in cells]
        cols = [col for _, col in cells]
        penalty += (max(rows) - min(rows)) + (max(cols) - min(cols))
    return penalty


def _region_cells(region_ids: tuple[tuple[int, ...], ...], region: int) -> set[tuple[int, int]]:
    size = len(region_ids)
    return {
        (row, col)
        for row in range(size)
        for col in range(size)
        if region_ids[row][col] == region
    }


def _is_connected(cells: set[tuple[int, int]], size: int) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}

    while stack:
        row, col = stack.pop()
        for neighbor in _neighbors(row, col, size):
            if neighbor in cells and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return len(seen) == len(cells)


def _label_region_grid(region_ids: tuple[tuple[int, ...], ...], rng: Random) -> tuple[str, ...]:
    size = len(region_ids)
    labels = list(ascii_uppercase[:size])
    rng.shuffle(labels)
    return tuple(
        "".join(labels[region_ids[row][col]] for col in range(size))
        for row in range(size)
    )


def _format_prompt(
    size: int,
    region_grid: tuple[str, ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    prompt_style = str(recipe["prompt_style"])
    surface_spec = SURFACE_SPECS[prompt_style]
    blocks = {
        "intro": _intro_block(size, language, surface_plan),
        "rules": _rules_block(language, surface_plan),
        "grid": _grid_block(region_grid, language, surface_plan),
        "instruction": _instruction_block(language, surface_plan),
        "answer": _answer_block(size, language, surface_plan),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(size: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    intro_style = str(surface_plan["surface_intro"])
    if language == "da":
        if intro_style == "context":
            lines = (
                f"Du får et {size}x{size}-gitter, hvor bogstaver markerer regioner.",
                "Placér stjerner sådan at følgende regler er opfyldt:",
            )
        elif intro_style == "assignment":
            lines = (
                f"Her er en star battle-opgave på et {size}x{size}-gitter med regioner.",
                "Indsæt stjerner i gitteret under disse regler:",
            )
        else:
            lines = (
                f"Løs en star battle-opgave på et {size}x{size}-gitter.",
                "Hver celle tilhører en bogstavregion. Placér stjerner sådan at:",
            )
        return prompt_surface.PromptBlock(key="intro", lines=lines)

    if intro_style == "context":
        lines = (
            f"You are given a {size} x {size} grid where letters mark regions.",
            "Place stars so that the following rules hold:",
        )
    elif intro_style == "assignment":
        lines = (
            f"This is a star battle puzzle on a {size} x {size} grid with regions.",
            "Place stars in the grid under these rules:",
        )
    else:
        lines = (
            f"Solve a star battle puzzle on a {size} x {size} grid.",
            "Each cell belongs to a lettered region. Place stars so that:",
        )
    return prompt_surface.PromptBlock(key="intro", lines=lines)


def _rules_block(language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    if language == "da":
        if clue_style == "compact":
            lines = (
                "Hver række indeholder præcis én stjerne.",
                "Hver kolonne indeholder præcis én stjerne.",
                "Hver region indeholder præcis én stjerne.",
                "Ingen to stjerner må være naboer, heller ikke diagonalt.",
            )
        else:
            lines = (
                "Hver række skal indeholde præcis én stjerne.",
                "Hver kolonne skal indeholde præcis én stjerne.",
                "Hver region skal indeholde præcis én stjerne.",
                "Ingen to stjerner må være naboer, heller ikke diagonalt.",
            )
        return prompt_surface.PromptBlock(key="rules", lines=lines, numbered=True)

    if clue_style == "compact":
        lines = (
            "Each row contains exactly one star.",
            "Each column contains exactly one star.",
            "Each region contains exactly one star.",
            "No two stars may be adjacent, including diagonally.",
        )
    else:
        lines = (
            "Each row must contain exactly one star.",
            "Each column must contain exactly one star.",
            "Each region must contain exactly one star.",
            "No two stars may be adjacent, including diagonally.",
        )
    return prompt_surface.PromptBlock(key="rules", lines=lines, numbered=True)


def _grid_block(
    region_grid: tuple[str, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    if language == "da":
        first_line = {
            "plain": "Regionerne er givet rækkevis, med én streng per række:",
            "compact": "Regionsgitter:",
            "deductive": "Bogstaverne viser regionerne, som stjernerne skal fordeles på:",
        }[clue_style]
    else:
        first_line = {
            "plain": "The regions are given in row-major order, with one string per row:",
            "compact": "Region grid:",
            "deductive": "The letters show the regions the stars must be distributed across:",
        }[clue_style]
    lines = (first_line, *region_grid)
    return prompt_surface.PromptBlock(key="grid", lines=lines)


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
            return "Regionerne og reglerne bestemmer én entydig løsning."
        return "Alle regler og regioner skal passe med den samme løsning."

    if instruction_variant == "solve":
        return "There is exactly one valid solution."
    if instruction_variant == "unique":
        return "The regions and rules determine one unique solution."
    return "The rules and regions must all fit the same solution."


def _answer_block(size: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    answer_style = str(surface_plan["surface_answer"])
    example = "." * max(0, size - 1) + "*"
    if language == "da":
        if answer_style == "respond":
            lines = (
                f"Svarformat: Skriv præcis {size} linjer med {size} tegn per linje.",
                "Brug `*` for en stjerne og `.` for et tomt felt.",
            )
        elif answer_style == "write":
            lines = (
                f"Outputformat: Brug præcis {size} linjer med {size} tegn.",
                "Brug kun `*` og `.`.",
            )
        else:
            lines = (
                f"Det endelige svar skal bestå af {size} linjer med {size} tegn hver.",
                "Brug `*` for stjerne og `.` for tomt felt.",
            )
        lines += (f"Eksempellinje: {example}",)
        return prompt_surface.PromptBlock(key="answer", lines=lines)

    if answer_style == "respond":
        lines = (
            f"Output Format: Write exactly {size} lines with {size} characters per line.",
            "Use `*` for a star and `.` for an empty cell.",
        )
    elif answer_style == "write":
        lines = (
            f"Output Format: Use exactly {size} lines with {size} characters.",
            "Use only `*` and `.`.",
        )
    else:
        lines = (
            f"Your final answer should consist of {size} lines with {size} characters each.",
            "Use `*` for stars and `.` for empty cells.",
        )
    lines += (f"Example line: {example}",)
    return prompt_surface.PromptBlock(key="answer", lines=lines)


def _neighbors(row: int, col: int, size: int) -> tuple[tuple[int, int], ...]:
    neighbors: list[tuple[int, int]] = []
    for next_row, next_col in (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ):
        if 0 <= next_row < size and 0 <= next_col < size:
            neighbors.append((next_row, next_col))
    return tuple(neighbors)


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"].get(key)
