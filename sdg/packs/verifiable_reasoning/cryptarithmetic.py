from dataclasses import dataclass
from itertools import permutations
from random import Random
from typing import Any

from sdg.commons import diversity, prompt_surface


@dataclass(frozen=True)
class CryptarithmeticSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class CryptarithmeticPuzzle:
    base: int
    addends: tuple[tuple[int, ...], ...]
    result: tuple[int, ...]
    solution: tuple[int, ...]
    prompt: str


SURFACE_SPECS = {
    "formal": CryptarithmeticSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="formal",
            block_order=("intro", "rules", "equation", "instruction", "answer"),
        ),
    ),
    "equation_first": CryptarithmeticSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="equation_first",
            block_order=("intro", "equation", "rules", "instruction", "answer"),
        ),
    ),
    "briefing": CryptarithmeticSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "instruction", "rules", "equation", "answer"),
        ),
    ),
}


RECIPES = (
    {
        "recipe_id": "easy_base3_pair",
        "difficulty": "easy",
        "prompt_style": "formal",
        "base": 3,
        "addend_lengths": (6, 6),
        "addend_count": 2,
        "result_lengths": (6, 7),
        "min_symbol_repetitions": 5,
        "sample_attempts": 240,
    },
    {
        "recipe_id": "easy_base4_pair",
        "difficulty": "easy",
        "prompt_style": "equation_first",
        "base": 4,
        "addend_lengths": (7, 7),
        "addend_count": 2,
        "result_lengths": (7, 8),
        "min_symbol_repetitions": 6,
        "sample_attempts": 260,
    },
    {
        "recipe_id": "medium_base4_pair_long",
        "difficulty": "medium",
        "prompt_style": "formal",
        "base": 4,
        "addend_lengths": (8, 8),
        "addend_count": 2,
        "result_lengths": (8, 9),
        "min_symbol_repetitions": 8,
        "sample_attempts": 320,
    },
    {
        "recipe_id": "medium_base5_pair_long",
        "difficulty": "medium",
        "prompt_style": "briefing",
        "base": 5,
        "addend_lengths": (9, 9),
        "addend_count": 2,
        "result_lengths": (9, 10),
        "min_symbol_repetitions": 8,
        "sample_attempts": 360,
    },
    {
        "recipe_id": "hard_base6_pair_long",
        "difficulty": "hard",
        "prompt_style": "equation_first",
        "base": 6,
        "addend_lengths": (10, 10),
        "addend_count": 2,
        "result_lengths": (10, 11),
        "min_symbol_repetitions": 10,
        "sample_attempts": 480,
    },
    {
        "recipe_id": "hard_base6_triple",
        "difficulty": "hard",
        "prompt_style": "briefing",
        "base": 6,
        "addend_lengths": (7, 7, 7),
        "addend_count": 3,
        "result_lengths": (7, 8),
        "min_symbol_repetitions": 10,
        "sample_attempts": 520,
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
            "base": puzzle.base,
            "addends": [list(term) for term in puzzle.addends],
            "result": list(puzzle.result),
            "solution": list(puzzle.solution),
        },
        "sources": [{"kind": "dolci_subset", "value": "cryptarithmetic"}],
        "meta": {
            "family": "cryptarithmetic_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": 1,
            "given_count": puzzle.base,
            "base": puzzle.base,
            "symbol_count": puzzle.base,
            "addend_count": len(puzzle.addends),
            "max_term_length": max(len(term) for term in (*puzzle.addends, puzzle.result)),
            "output_format": "digit_sequence",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[int, ...] | None:
    base = int(hidden["base"])
    stripped = text.strip()
    if not stripped:
        return None

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    candidates = [lines[-1]] if lines else []
    if len(lines) == 1:
        candidates.insert(0, lines[0])

    for candidate in candidates:
        pieces = candidate.split()
        if len(pieces) != base:
            continue
        if not all(piece.isdigit() for piece in pieces):
            continue
        digits = tuple(int(piece) for piece in pieces)
        if set(digits) != set(range(base)):
            continue
        return digits

    return None


def canonical_target(parsed: tuple[int, ...], hidden: dict[str, object]) -> str:
    return format_target(parsed)


def format_target(digits: tuple[int, ...]) -> str:
    return " ".join(str(digit) for digit in digits)


def answer_contract(hidden: dict[str, object], language: str) -> str:
    base = int(hidden["base"])
    example = " ".join(str(digit) for digit in range(base))

    if language == "da":
        return (
            "I din svarblok skal du skrive præcis én linje med decimalværdierne for d[0], d[1], ..., i rækkefølge.\n"
            f"Linjen skal indeholde præcis {base} tal, adskilt af mellemrum.\n"
            f"Format:\n{example}"
        )

    return (
        "In your answer block, write exactly one line with the decimal values of d[0], d[1], ..., in order.\n"
        f"The line must contain exactly {base} numbers separated by spaces.\n"
        f"Format:\n{example}"
    )


def is_correct(parsed: tuple[int, ...], hidden: dict[str, object]) -> bool:
    return list(parsed) == hidden["solution"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    base = int(hidden["base"])
    addends = tuple(tuple(int(index) for index in term) for term in hidden["addends"])
    result = tuple(int(index) for index in hidden["result"])
    solutions = _solve_assignments(base, addends, result, limit=2)
    if len(solutions) != 1:
        return False
    return list(solutions[0]) == hidden["solution"]


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "prompt_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("prompt_style",), observed_getter=_meta_getter),
        "base_coverage": diversity.compare_planned_to_observed(planned, rows, ("base",), observed_getter=_meta_getter),
        "addend_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("addend_count",), observed_getter=_meta_getter),
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
) -> CryptarithmeticPuzzle:
    base = int(recipe["base"])
    addend_lengths = tuple(int(length) for length in recipe["addend_lengths"])
    result_lengths = tuple(int(length) for length in recipe["result_lengths"])

    for _ in range(int(recipe["sample_attempts"])):
        solution = tuple(rng.sample(tuple(range(base)), base))
        addends = _sample_addends(base, solution, addend_lengths, int(recipe["min_symbol_repetitions"]), rng)
        if addends is None:
            continue

        result = _build_result(base, solution, addends)
        if result is None:
            continue
        if len(result) not in result_lengths:
            continue

        if set(_flatten_terms((*addends, result))) != set(range(base)):
            continue

        solutions = _solve_assignments(base, addends, result, limit=2)
        if solutions != [solution]:
            continue

        prompt = _format_prompt(
            base,
            addends,
            result,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return CryptarithmeticPuzzle(
            base=base,
            addends=addends,
            result=result,
            solution=solution,
            prompt=prompt,
        )

    raise AssertionError("failed to generate cryptarithmetic puzzle")


def _sample_addends(
    base: int,
    solution: tuple[int, ...],
    lengths: tuple[int, ...],
    min_symbol_repetitions: int,
    rng: Random,
) -> tuple[tuple[int, ...], ...] | None:
    nonzero_symbols = tuple(index for index, value in enumerate(solution) if value != 0)
    if not nonzero_symbols:
        return None

    for _ in range(64):
        addends = tuple(
            _sample_term(base, length, nonzero_symbols, rng)
            for length in lengths
        )
        if len(set(addends)) != len(addends):
            continue

        flattened = _flatten_terms(addends)
        repetition_count = len(flattened) - len(set(flattened))
        if repetition_count < min_symbol_repetitions:
            continue
        return addends

    return None


def _sample_term(base: int, length: int, nonzero_symbols: tuple[int, ...], rng: Random) -> tuple[int, ...]:
    first = rng.choice(nonzero_symbols)
    tail = tuple(rng.randrange(base) for _ in range(length - 1))
    return (first, *tail)


def _build_result(base: int, solution: tuple[int, ...], addends: tuple[tuple[int, ...], ...]) -> tuple[int, ...] | None:
    total = sum(_term_value(base, solution, term) for term in addends)
    digits = _to_base_digits(total, base)
    inverse = {value: index for index, value in enumerate(solution)}
    if any(digit not in inverse for digit in digits):
        return None
    return tuple(inverse[digit] for digit in digits)


def _term_value(base: int, solution: tuple[int, ...], term: tuple[int, ...]) -> int:
    value = 0
    for index in term:
        value = (value * base) + solution[index]
    return value


def _to_base_digits(value: int, base: int) -> tuple[int, ...]:
    if value == 0:
        return (0,)

    digits: list[int] = []
    working = value
    while working > 0:
        digits.append(working % base)
        working //= base
    return tuple(reversed(digits))


def _solve_assignments(
    base: int,
    addends: tuple[tuple[int, ...], ...],
    result: tuple[int, ...],
    *,
    limit: int,
) -> list[tuple[int, ...]]:
    solutions: list[tuple[int, ...]] = []
    for assignment in permutations(range(base)):
        if _equation_holds(base, assignment, addends, result):
            solutions.append(tuple(int(value) for value in assignment))
            if len(solutions) >= limit:
                break
    return solutions


def _equation_holds(
    base: int,
    assignment: tuple[int, ...],
    addends: tuple[tuple[int, ...], ...],
    result: tuple[int, ...],
) -> bool:
    return sum(_term_value(base, assignment, term) for term in addends) == _term_value(base, assignment, result)


def _format_prompt(
    base: int,
    addends: tuple[tuple[int, ...], ...],
    result: tuple[int, ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    prompt_style = str(recipe["prompt_style"])
    surface_spec = SURFACE_SPECS[prompt_style]
    blocks = {
        "intro": _intro_block(base, language, surface_plan),
        "rules": _rules_block(base, language, surface_plan),
        "equation": _equation_block(base, addends, result, language, surface_plan),
        "instruction": _instruction_block(base, language, surface_plan),
        "answer": _answer_block(base, language, surface_plan),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(base: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_intro"])
    digits = _digit_names(base)

    if language == "da":
        if variant == "context":
            lines = (
                f"Overvej et talsystem med base {base}, som bruger cifrene {digits}.",
                f"Hvert d[i] er et unikt heltal i intervallet [0, {base - 1}], men de konkrete værdier er ukendte.",
            )
        elif variant == "assignment":
            lines = (
                f"Bestem decimalværdien af cifrene {digits} i et ukendt base-{base}-system.",
                f"Hvert d[i] er forskelligt og ligger i intervallet [0, {base - 1}].",
            )
        else:
            lines = (
                f"Du får et ukendt talsystem med base {base} og cifrene {digits}.",
                f"Cifrene d[0] til d[{base - 1}] er en permutation af {base} forskellige decimalværdier.",
            )
        return prompt_surface.PromptBlock(key="intro", lines=lines)

    if variant == "context":
        lines = (
            f"Now consider a number system with base {base}, which uses digits {digits}.",
            f"Each d[i] is a unique integer in the range [0, {base - 1}], but the actual values are unknown.",
        )
    elif variant == "assignment":
        lines = (
            f"Determine the decimal value of digits {digits} in an unknown base-{base} system.",
            f"Each d[i] is distinct and lies in the range [0, {base - 1}].",
        )
    else:
        lines = (
            f"You are given an unknown base-{base} number system with digits {digits}.",
            f"The symbols d[0] through d[{base - 1}] form a permutation of {base} distinct decimal values.",
        )
    return prompt_surface.PromptBlock(key="intro", lines=lines)


def _rules_block(base: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])

    if language == "da":
        if clue_style == "compact":
            lines = (
                f"Tallet `d[i0]d[i1]...d[ik]` betyder d[i0] * {base}^k + ... + d[ik].",
            )
        else:
            lines = (
                f"Tallet `d[i0]d[i1]...d[ik]` repræsenterer værdien d[i0] * {base}^k + d[i1] * {base}^(k-1) + ... + d[ik] * {base}^0.",
            )
        return prompt_surface.PromptBlock(key="rules", lines=lines)

    if clue_style == "compact":
        lines = (
            f"The number `d[i0]d[i1]...d[ik]` means d[i0] * {base}^k + ... + d[ik].",
        )
    else:
        lines = (
            f"The number `d[i0]d[i1]...d[ik]` represents the value d[i0] * {base}^k + d[i1] * {base}^(k-1) + ... + d[ik] * {base}^0.",
        )
    return prompt_surface.PromptBlock(key="rules", lines=lines)


def _equation_block(
    base: int,
    addends: tuple[tuple[int, ...], ...],
    result: tuple[int, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_clue"])
    if language == "da":
        if variant == "compact":
            heading = "Ligning"
            lines = _render_equation_lines(addends, result)
        elif variant == "deductive":
            heading = "Du får følgende ligning i det ukendte base-system"
            lines = _render_equation_lines(addends, result)
        else:
            heading = f"Du får følgende ligning i dette ukendte base-{base}-system"
            lines = _render_equation_lines(addends, result)
        return prompt_surface.PromptBlock(key="equation", heading=heading, lines=lines)

    if variant == "compact":
        heading = "Equation"
        lines = _render_equation_lines(addends, result)
    elif variant == "deductive":
        heading = "You are given the following equation"
        lines = _render_equation_lines(addends, result)
    else:
        heading = f"You are given the following equation in this unknown base-{base} digit system"
        lines = _render_equation_lines(addends, result)
    return prompt_surface.PromptBlock(key="equation", heading=heading, lines=lines)


def _instruction_block(base: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_instruction"])
    digits = _digit_names(base)

    if language == "da":
        if variant == "solve":
            line = f"Find én mulig decimal-tildeling for {digits}, så ligningen går op."
        elif variant == "unique":
            line = "Ligningen bestemmer én entydig løsning."
        else:
            line = f"Bestem decimalværdierne for {digits}, så alle betingelser passer samtidig."
        return prompt_surface.PromptBlock(key="instruction", lines=(line,))

    if variant == "solve":
        line = f"Find one possible decimal assignment for {digits} such that the equation holds."
    elif variant == "unique":
        line = "The equation determines one unique solution."
    else:
        line = f"Determine the decimal values of {digits} so that all constraints hold at once."
    return prompt_surface.PromptBlock(key="instruction", lines=(line,))


def _answer_block(base: int, language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_answer"])
    example = " ".join(str(digit) for digit in range(base))

    if language == "da":
        if variant == "respond":
            lines = (
                "Inde i det endelige svar skal der stå én enkelt linje med decimalværdierne for d[0], d[1], ..., i rækkefølge.",
                f"Brug præcis {base} tal adskilt af mellemrum.",
            )
        elif variant == "write":
            lines = (
                "Inde i det endelige svar skal tallene for d[0], d[1], ... stå på én linje i rækkefølge.",
                "Brug kun decimaltal og mellemrum.",
            )
        else:
            lines = (
                "Selve svarindholdet skal være én enkelt linje med decimalværdierne i rækkefølge.",
                "Brug kun tal og mellemrum.",
            )
        lines += (f"Format:\n{example}",)
        return prompt_surface.PromptBlock(key="answer", lines=lines)

    if variant == "respond":
        lines = (
            "Inside the final response, it should be a single line with the decimal values of d[0], d[1], ..., in order.",
            f"Use exactly {base} numbers separated by spaces.",
        )
    elif variant == "write":
        lines = (
            "Inside the final response, put the values for d[0], d[1], ..., on one line in order.",
            "Use only decimal numbers and spaces.",
        )
    else:
        lines = (
            "The answer content should be one line with the decimal values in order.",
            "Use only numbers and spaces.",
        )
    lines += (f"Format:\n{example}",)
    return prompt_surface.PromptBlock(key="answer", lines=lines)


def _render_equation_lines(addends: tuple[tuple[int, ...], ...], result: tuple[int, ...]) -> tuple[str, ...]:
    lines = [ _render_term(addends[0]) ]
    for term in addends[1:]:
        lines.append(f"+\n{_render_term(term)}")
    lines.append(f"=\n{_render_term(result)}")
    return tuple(lines)


def _render_term(term: tuple[int, ...]) -> str:
    return "".join(f"d[{index}]" for index in term)


def _digit_names(base: int) -> str:
    return ", ".join(f"d[{index}]" for index in range(base))


def _flatten_terms(terms: tuple[tuple[int, ...], ...] | tuple[tuple[int, ...], ..., tuple[int, ...]]) -> tuple[int, ...]:
    flattened: list[int] = []
    for term in terms:
        flattened.extend(term)
    return tuple(flattened)


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]
