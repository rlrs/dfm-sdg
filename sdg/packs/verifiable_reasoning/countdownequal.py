import ast
from dataclasses import dataclass
from itertools import combinations
from random import Random
from typing import Any

from sdg.commons import diversity, prompt_surface


@dataclass(frozen=True)
class CountDownPuzzle:
    numbers: tuple[int, ...]
    target: int
    solution_expr: str
    minimal_ops: int
    allowed_ops: tuple[str, ...]
    use_all_numbers: bool
    prompt: str


@dataclass(frozen=True)
class CountDownSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class ParsedExpression:
    value: int
    expr: str
    ops: int
    numbers: tuple[int, ...]
    operators: frozenset[str]


SURFACE_SPECS = {
    "formal": CountDownSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="formal",
            block_order=("intro", "numbers", "target", "rules", "instruction", "answer"),
        ),
    ),
    "target_first": CountDownSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="target_first",
            block_order=("intro", "target", "numbers", "rules", "instruction", "answer"),
        ),
    ),
    "briefing": CountDownSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "instruction", "numbers", "target", "rules", "answer"),
        ),
    ),
}

OPERATOR_PROFILES = {
    "add_mul": ("+", "*"),
    "add_sub_mul": ("+", "-", "*"),
    "full": ("+", "-", "*", "/"),
}


RECIPES = (
    {
        "recipe_id": "easy_formal_4_add_mul_subset",
        "difficulty": "easy",
        "prompt_style": "formal",
        "number_count": 4,
        "number_min": 1,
        "number_max": 10,
        "target_min": 10,
        "target_max": 40,
        "min_ops": 2,
        "max_ops": 2,
        "operator_profile": "add_mul",
        "use_all_numbers": False,
        "sample_attempts": 180,
    },
    {
        "recipe_id": "easy_target_first_4_add_sub_mul_all",
        "difficulty": "easy",
        "prompt_style": "target_first",
        "number_count": 4,
        "number_min": 1,
        "number_max": 12,
        "target_min": 12,
        "target_max": 60,
        "min_ops": 3,
        "max_ops": 3,
        "operator_profile": "add_sub_mul",
        "use_all_numbers": True,
        "sample_attempts": 220,
    },
    {
        "recipe_id": "medium_briefing_5_add_sub_mul_subset",
        "difficulty": "medium",
        "prompt_style": "briefing",
        "number_count": 5,
        "number_min": 1,
        "number_max": 18,
        "target_min": 20,
        "target_max": 120,
        "min_ops": 3,
        "max_ops": 4,
        "operator_profile": "add_sub_mul",
        "use_all_numbers": False,
        "sample_attempts": 260,
    },
    {
        "recipe_id": "medium_target_first_5_full_all",
        "difficulty": "medium",
        "prompt_style": "target_first",
        "number_count": 5,
        "number_min": 1,
        "number_max": 18,
        "target_min": 30,
        "target_max": 180,
        "min_ops": 4,
        "max_ops": 4,
        "operator_profile": "full",
        "use_all_numbers": True,
        "sample_attempts": 300,
    },
    {
        "recipe_id": "medium_formal_6_full_subset",
        "difficulty": "medium",
        "prompt_style": "formal",
        "number_count": 6,
        "number_min": 1,
        "number_max": 20,
        "target_min": 30,
        "target_max": 180,
        "min_ops": 3,
        "max_ops": 4,
        "operator_profile": "full",
        "use_all_numbers": False,
        "sample_attempts": 320,
    },
    {
        "recipe_id": "hard_formal_6_full_all",
        "difficulty": "hard",
        "prompt_style": "formal",
        "number_count": 6,
        "number_min": 1,
        "number_max": 25,
        "target_min": 40,
        "target_max": 250,
        "min_ops": 5,
        "max_ops": 5,
        "operator_profile": "full",
        "use_all_numbers": True,
        "sample_attempts": 340,
    },
    {
        "recipe_id": "hard_briefing_6_add_sub_mul_all",
        "difficulty": "hard",
        "prompt_style": "briefing",
        "number_count": 6,
        "number_min": 1,
        "number_max": 30,
        "target_min": 50,
        "target_max": 400,
        "min_ops": 5,
        "max_ops": 5,
        "operator_profile": "add_sub_mul",
        "use_all_numbers": True,
        "sample_attempts": 380,
    },
    {
        "recipe_id": "hard_target_first_6_full_subset",
        "difficulty": "hard",
        "prompt_style": "target_first",
        "number_count": 6,
        "number_min": 1,
        "number_max": 30,
        "target_min": 60,
        "target_max": 400,
        "min_ops": 4,
        "max_ops": 5,
        "operator_profile": "full",
        "use_all_numbers": False,
        "sample_attempts": 400,
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
            "numbers": list(puzzle.numbers),
            "target": puzzle.target,
            "solution_expr": puzzle.solution_expr,
            "minimal_ops": puzzle.minimal_ops,
            "allowed_ops": list(puzzle.allowed_ops),
            "use_all_numbers": puzzle.use_all_numbers,
        },
        "sources": [{"kind": "dolci_subset", "value": "countdownequal"}],
        "meta": {
            "family": "countdownequal_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": len(puzzle.numbers) + 1,
            "given_count": len(puzzle.numbers),
            "number_count": len(puzzle.numbers),
            "target_value": puzzle.target,
            "minimal_ops": puzzle.minimal_ops,
            "operator_profile": recipe["operator_profile"],
            "use_all_numbers": puzzle.use_all_numbers,
            "usage_policy": "all_numbers" if recipe["use_all_numbers"] else "subset_allowed",
            "output_format": "expression_string",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> ParsedExpression | None:
    numbers = tuple(int(value) for value in hidden["numbers"])
    allowed_ops = tuple(str(value) for value in hidden["allowed_ops"])
    use_all_numbers = bool(hidden["use_all_numbers"])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    candidates = [lines[-1]]
    if len(lines) == 1:
        candidates.append(lines[0])

    for candidate in candidates:
        normalized = candidate.strip()
        for prefix in ("Answer:", "Svar:", "Expression:", "Udtryk:"):
            if normalized.lower().startswith(prefix.lower()):
                normalized = normalized[len(prefix):].strip()
                break
        parsed = _parse_expression(normalized)
        if parsed is None:
            continue
        if not _uses_allowed_operators(parsed.operators, allowed_ops):
            continue
        if not _uses_allowed_numbers(parsed.numbers, numbers):
            continue
        if use_all_numbers and not _uses_all_numbers(parsed.numbers, numbers):
            continue
        return parsed

    return None


def canonical_target(parsed: ParsedExpression, hidden: dict[str, object]) -> str:
    return parsed.expr


def format_target(expression: str) -> str:
    return expression


def answer_contract(hidden: dict[str, object], language: str) -> str:
    allowed_ops = tuple(str(value) for value in hidden["allowed_ops"])
    use_all_numbers = bool(hidden["use_all_numbers"])
    operator_text = _render_operator_text(allowed_ops)
    example = _example_expression(allowed_ops, use_all_numbers)
    usage_line_da = "Du skal bruge alle de givne tal præcis én gang." if use_all_numbers else "Du må bruge en delmængde af de givne tal, men hvert tal højst én gang."
    usage_line_en = "You must use all given numbers exactly once." if use_all_numbers else "You may use a subset of the given numbers, but each number at most once."

    if language == "da":
        return (
            "I din svarblok skal du skrive præcis én rå aritmetisk udtryk-linje.\n"
            f"Brug kun tal fra opgaven samt operatorerne {operator_text}.\n"
            f"{usage_line_da}\n"
            "Brug parenteser, når det hjælper læsbarheden.\n"
            f"Format:\n{example}"
        )

    return (
        "In your answer block, write exactly one raw arithmetic expression.\n"
        f"Use only numbers from the puzzle and the operators {operator_text}.\n"
        f"{usage_line_en}\n"
        "Use parentheses when they help readability.\n"
        f"Format:\n{example}"
    )


def is_correct(parsed: ParsedExpression, hidden: dict[str, object]) -> bool:
    allowed_ops = tuple(str(value) for value in hidden["allowed_ops"])
    use_all_numbers = bool(hidden["use_all_numbers"])
    numbers = tuple(int(value) for value in hidden["numbers"])
    target = int(hidden["target"])
    minimal_ops = int(hidden["minimal_ops"])
    solution_expr = str(hidden["solution_expr"])
    if parsed.value != target:
        return False
    if not _uses_allowed_operators(parsed.operators, allowed_ops):
        return False
    if not _uses_allowed_numbers(parsed.numbers, numbers):
        return False
    if use_all_numbers and not _uses_all_numbers(parsed.numbers, numbers):
        return False
    if parsed.ops != minimal_ops:
        return False
    return parsed.expr == solution_expr


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    numbers = tuple(int(value) for value in hidden["numbers"])
    allowed_ops = tuple(str(value) for value in hidden["allowed_ops"])
    use_all_numbers = bool(hidden["use_all_numbers"])
    target = int(hidden["target"])
    minimal_ops = int(hidden["minimal_ops"])
    solution_expr = str(hidden["solution_expr"])
    summary = _target_summaries(numbers, allowed_ops, require_all_numbers=use_all_numbers).get(target)
    if summary is None:
        return False
    if summary["count"] != 1:
        return False
    if summary["ops"] != minimal_ops:
        return False
    return summary["expr"] == solution_expr


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "prompt_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("prompt_style",), observed_getter=_meta_getter),
        "number_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("number_count",), observed_getter=_meta_getter),
        "operator_profile_coverage": diversity.compare_planned_to_observed(planned, rows, ("operator_profile",), observed_getter=_meta_getter),
        "usage_policy_coverage": diversity.compare_planned_to_observed(planned, rows, ("use_all_numbers",), observed_getter=_meta_getter),
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
) -> CountDownPuzzle:
    number_count = int(recipe["number_count"])
    number_min = int(recipe["number_min"])
    number_max = int(recipe["number_max"])
    target_min = int(recipe["target_min"])
    target_max = int(recipe["target_max"])
    min_ops = int(recipe["min_ops"])
    max_ops = int(recipe["max_ops"])
    allowed_ops = OPERATOR_PROFILES[str(recipe["operator_profile"])]
    use_all_numbers = bool(recipe["use_all_numbers"])

    number_sets = list(combinations(range(number_min, number_max + 1), number_count))
    rng.shuffle(number_sets)

    for numbers in number_sets[: int(recipe["sample_attempts"])]:
        ordered_numbers = tuple(rng.sample(numbers, len(numbers)))
        summaries = _target_summaries(ordered_numbers, allowed_ops, require_all_numbers=use_all_numbers)
        candidates = [
            (target, summary)
            for target, summary in summaries.items()
            if target_min <= target <= target_max
            and min_ops <= int(summary["ops"]) <= max_ops
            and int(summary["count"]) == 1
        ]
        if not candidates:
            continue

        target, summary = rng.choice(candidates)
        prompt = _format_prompt(
            ordered_numbers,
            target,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return CountDownPuzzle(
            numbers=ordered_numbers,
            target=target,
            solution_expr=str(summary["expr"]),
            minimal_ops=int(summary["ops"]),
            allowed_ops=allowed_ops,
            use_all_numbers=use_all_numbers,
            prompt=prompt,
        )

    raise AssertionError("failed to generate countdownequal puzzle")


def _target_summaries(
    numbers: tuple[int, ...],
    allowed_ops: tuple[str, ...],
    *,
    require_all_numbers: bool,
) -> dict[int, dict[str, object]]:
    reachable = _reachable_by_subset(numbers, allowed_ops)
    summaries: dict[int, dict[str, object]] = {}
    full_mask = (1 << len(numbers)) - 1

    for mask, values in reachable.items():
        if require_all_numbers and mask != full_mask:
            continue
        ops = mask.bit_count() - 1
        if ops == 0:
            continue
        for value, entry in values.items():
            current = summaries.get(value)
            if current is None or ops < int(current["ops"]):
                summaries[value] = {
                    "ops": ops,
                    "count": int(entry["count"]),
                    "expr": str(entry["expr"]),
                    "alt_expr": entry.get("alt_expr"),
                }
                continue
            if ops != int(current["ops"]):
                continue
            summaries[value] = _merge_summary(current, entry)

    return summaries


def _reachable_by_subset(
    numbers: tuple[int, ...],
    allowed_ops: tuple[str, ...],
) -> dict[int, dict[int, dict[str, object]]]:
    reachable: dict[int, dict[int, dict[str, object]]] = {}
    full_mask = 1 << len(numbers)

    for index, number in enumerate(numbers):
        reachable[1 << index] = {
            number: {"count": 1, "expr": str(number), "alt_expr": None},
        }

    for size in range(2, len(numbers) + 1):
        for mask in range(1, full_mask):
            if mask.bit_count() != size:
                continue
            entries: dict[int, dict[str, object]] = {}
            submask = (mask - 1) & mask
            while submask:
                other = mask ^ submask
                if submask < other:
                    _merge_partition(entries, reachable[submask], reachable[other], allowed_ops)
                submask = (submask - 1) & mask
            reachable[mask] = entries

    return reachable


def _merge_partition(
    target_entries: dict[int, dict[str, object]],
    left_entries: dict[int, dict[str, object]],
    right_entries: dict[int, dict[str, object]],
    allowed_ops: tuple[str, ...],
) -> None:
    for left_value, left_entry in left_entries.items():
        for right_value, right_entry in right_entries.items():
            left_expr = str(left_entry["expr"])
            right_expr = str(right_entry["expr"])

            for value, expr in _combine_pair(left_value, left_expr, right_value, right_expr, allowed_ops):
                _record_expression(target_entries, value, expr)


def _combine_pair(
    left_value: int,
    left_expr: str,
    right_value: int,
    right_expr: str,
    allowed_ops: tuple[str, ...],
) -> tuple[tuple[int, str], ...]:
    results: list[tuple[int, str]] = []
    first_value, first_expr, second_value, second_expr = _ordered_commutative(
        left_value,
        left_expr,
        right_value,
        right_expr,
    )

    if "+" in allowed_ops:
        results.append((first_value + second_value, f"({first_expr}+{second_expr})"))

    if "*" in allowed_ops and first_value != 1 and second_value != 1:
        results.append((first_value * second_value, f"({first_expr}*{second_expr})"))

    larger_value = left_value
    larger_expr = left_expr
    smaller_value = right_value
    smaller_expr = right_expr
    if (right_value, right_expr) > (left_value, left_expr):
        larger_value = right_value
        larger_expr = right_expr
        smaller_value = left_value
        smaller_expr = left_expr

    if "-" in allowed_ops and larger_value > smaller_value:
        results.append((larger_value - smaller_value, f"({larger_expr}-{smaller_expr})"))

    if "/" in allowed_ops and smaller_value > 1 and larger_value % smaller_value == 0:
        results.append((larger_value // smaller_value, f"({larger_expr}/{smaller_expr})"))

    return tuple((value, expr) for value, expr in results if value > 0)


def _ordered_commutative(
    left_value: int,
    left_expr: str,
    right_value: int,
    right_expr: str,
) -> tuple[int, str, int, str]:
    if (left_value, left_expr) <= (right_value, right_expr):
        return left_value, left_expr, right_value, right_expr
    return right_value, right_expr, left_value, left_expr


def _record_expression(entries: dict[int, dict[str, object]], value: int, expr: str) -> None:
    current = entries.get(value)
    if current is None:
        entries[value] = {"count": 1, "expr": expr, "alt_expr": None}
        return
    if expr == current["expr"] or expr == current["alt_expr"]:
        return
    if int(current["count"]) == 1:
        current["count"] = 2
        current["alt_expr"] = expr
        return
    current["count"] = 2


def _merge_summary(summary: dict[str, object], entry: dict[str, object]) -> dict[str, object]:
    expr = str(entry["expr"])
    if expr == summary["expr"] or expr == summary["alt_expr"]:
        return summary
    if int(summary["count"]) == 1:
        return {
            "ops": summary["ops"],
            "count": 2,
            "expr": summary["expr"],
            "alt_expr": expr,
        }
    return {
        "ops": summary["ops"],
        "count": 2,
        "expr": summary["expr"],
        "alt_expr": summary["alt_expr"],
    }


def _parse_expression(text: str) -> ParsedExpression | None:
    if not text:
        return None

    try:
        node = ast.parse(text, mode="eval")
    except SyntaxError:
        return None

    parsed = _parse_ast_node(node.body)
    if parsed is None:
        return None
    return parsed


def _parse_ast_node(node: ast.AST) -> ParsedExpression | None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, int):
            return None
        if node.value <= 0:
            return None
        value = int(node.value)
        return ParsedExpression(
            value=value,
            expr=str(value),
            ops=0,
            numbers=(value,),
            operators=frozenset(),
        )

    if not isinstance(node, ast.BinOp):
        return None

    left = _parse_ast_node(node.left)
    right = _parse_ast_node(node.right)
    if left is None or right is None:
        return None

    numbers = left.numbers + right.numbers
    ops = left.ops + right.ops + 1
    operators = left.operators | right.operators

    if isinstance(node.op, ast.Add):
        first_value, first_expr, second_value, second_expr = _ordered_commutative(
            left.value,
            left.expr,
            right.value,
            right.expr,
        )
        return ParsedExpression(
            value=first_value + second_value,
            expr=f"({first_expr}+{second_expr})",
            ops=ops,
            numbers=numbers,
            operators=operators | {"+"},
        )

    if isinstance(node.op, ast.Mult):
        first_value, first_expr, second_value, second_expr = _ordered_commutative(
            left.value,
            left.expr,
            right.value,
            right.expr,
        )
        return ParsedExpression(
            value=first_value * second_value,
            expr=f"({first_expr}*{second_expr})",
            ops=ops,
            numbers=numbers,
            operators=operators | {"*"},
        )

    if isinstance(node.op, ast.Sub):
        if left.value <= right.value:
            return None
        return ParsedExpression(
            value=left.value - right.value,
            expr=f"({left.expr}-{right.expr})",
            ops=ops,
            numbers=numbers,
            operators=operators | {"-"},
        )

    if isinstance(node.op, ast.Div):
        if right.value <= 0:
            return None
        if left.value <= right.value:
            return None
        if left.value % right.value != 0:
            return None
        return ParsedExpression(
            value=left.value // right.value,
            expr=f"({left.expr}/{right.expr})",
            ops=ops,
            numbers=numbers,
            operators=operators | {"/"},
        )

    return None


def _uses_allowed_numbers(used_numbers: tuple[int, ...], available_numbers: tuple[int, ...]) -> bool:
    available = dict.fromkeys(available_numbers, 0)
    for value in available_numbers:
        available[value] = available.get(value, 0) + 1

    used = dict.fromkeys(used_numbers, 0)
    for value in used_numbers:
        used[value] = used.get(value, 0) + 1

    for value, count in used.items():
        if count > available.get(value, 0):
            return False
    return True


def _uses_all_numbers(used_numbers: tuple[int, ...], available_numbers: tuple[int, ...]) -> bool:
    used = sorted(used_numbers)
    available = sorted(available_numbers)
    return used == available


def _uses_allowed_operators(used_ops: frozenset[str], allowed_ops: tuple[str, ...]) -> bool:
    return used_ops.issubset(set(allowed_ops))


def _format_prompt(
    numbers: tuple[int, ...],
    target: int,
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    prompt_style = str(recipe["prompt_style"])
    surface_spec = SURFACE_SPECS[prompt_style]
    blocks = {
        "intro": _intro_block(len(numbers), language, surface_plan),
        "numbers": _numbers_block(numbers, language, surface_plan),
        "target": _target_block(target, language, surface_plan),
        "rules": _rules_block(language, surface_plan, OPERATOR_PROFILES[str(recipe["operator_profile"])], bool(recipe["use_all_numbers"])),
        "instruction": _instruction_block(language, surface_plan),
        "answer": _answer_block(
            language,
            surface_plan,
            OPERATOR_PROFILES[str(recipe["operator_profile"])],
            bool(recipe["use_all_numbers"]),
        ),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(
    number_count: int,
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_intro"])

    if language == "da":
        if variant == "context":
            lines = (f"Du får {number_count} tal til en countdownequal-opgave.",)
        elif variant == "assignment":
            lines = (f"Bestem et kortest muligt regneudtryk ud fra {number_count} givne tal.",)
        else:
            lines = (f"Løs en aritmetisk opgave med {number_count} givne tal.",)
        return prompt_surface.PromptBlock(key="intro", lines=lines)

    if variant == "context":
        lines = (f"You are given {number_count} numbers for a countdownequal puzzle.",)
    elif variant == "assignment":
        lines = (f"Determine a shortest arithmetic expression from {number_count} given numbers.",)
    else:
        lines = (f"Solve an arithmetic puzzle with {number_count} given numbers.",)
    return prompt_surface.PromptBlock(key="intro", lines=lines)


def _numbers_block(
    numbers: tuple[int, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    rendered_numbers = ", ".join(str(number) for number in numbers)

    if language == "da":
        if clue_style == "compact":
            lines = (f"[{rendered_numbers}]",)
            heading = "Tal"
        elif clue_style == "deductive":
            lines = (
                rendered_numbers,
                "Hvert tal må bruges højst én gang.",
            )
            heading = "Talbank"
        else:
            lines = (f"{rendered_numbers}",)
            heading = "Givne tal"
        return prompt_surface.PromptBlock(key="numbers", heading=heading, lines=lines)

    if clue_style == "compact":
        lines = (f"[{rendered_numbers}]",)
        heading = "Numbers"
    elif clue_style == "deductive":
        lines = (
            rendered_numbers,
            "Each number may be used at most once.",
        )
        heading = "Number Bank"
    else:
        lines = (f"{rendered_numbers}",)
        heading = "Available Numbers"
    return prompt_surface.PromptBlock(key="numbers", heading=heading, lines=lines)


def _target_block(
    target: int,
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])

    if language == "da":
        if clue_style == "compact":
            lines = (str(target),)
            heading = "Mål"
        elif clue_style == "deductive":
            lines = (f"Ram præcis {target}.",)
            heading = "Måltal"
        else:
            lines = (f"{target}",)
            heading = "Måltallet"
        return prompt_surface.PromptBlock(key="target", heading=heading, lines=lines)

    if clue_style == "compact":
        lines = (str(target),)
        heading = "Target"
    elif clue_style == "deductive":
        lines = (f"Hit exactly {target}.",)
        heading = "Goal"
    else:
        lines = (f"{target}",)
        heading = "Target Value"
    return prompt_surface.PromptBlock(key="target", heading=heading, lines=lines)


def _rules_block(
    language: str,
    surface_plan: dict[str, object],
    allowed_ops: tuple[str, ...],
    use_all_numbers: bool,
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    operator_text = _render_operator_text(allowed_ops)
    if language == "da":
        usage_line = "Du skal bruge alle de givne tal præcis én gang."
        if not use_all_numbers:
            usage_line = "Du må bruge en delmængde af de givne tal, men hvert tal højst én gang."
    else:
        usage_line = "You must use all given numbers exactly once."
        if not use_all_numbers:
            usage_line = "You may use a subset of the given numbers, but each number at most once."

    if language == "da":
        division_line_da = "Division er kun tilladt, når resultatet stadig er et heltal."
        if "/" not in allowed_ops:
            division_line_da = "Ingen andre operatorer end de nævnte er tilladt."

        if clue_style == "compact":
            lines = (
                f"Tilladte operatorer: {operator_text}.",
                usage_line,
                "Alle mellemregninger skal være positive heltal.",
            )
        elif clue_style == "deductive":
            lines = (
                f"Du må bruge operatorerne {operator_text}.",
                division_line_da,
                usage_line,
            )
        else:
            final_line = "Hold alle mellemresultater som positive heltal."
            if "/" in allowed_ops:
                final_line = "Hold alle mellemresultater som positive heltal, og brug kun division, når den går op."
            lines = (
                f"Brug kun operatorerne {operator_text}.",
                usage_line,
                final_line,
            )
        return prompt_surface.PromptBlock(key="rules", lines=lines)

    division_line_en = "Division is allowed only when it still gives an integer result."
    if "/" not in allowed_ops:
        division_line_en = "No operators other than the listed ones are allowed."

    if clue_style == "compact":
        lines = (
            f"Allowed operators: {operator_text}.",
            usage_line,
            "All intermediate results must stay positive integers.",
        )
    elif clue_style == "deductive":
        lines = (
            f"You may use the operators {operator_text}.",
            division_line_en,
            usage_line,
        )
    else:
        final_line = "Keep every intermediate result as a positive integer."
        if "/" in allowed_ops:
            final_line = "Keep every intermediate result as a positive integer, and divide only when it divides exactly."
        lines = (
            f"Use only the operators {operator_text}.",
            usage_line,
            final_line,
        )
    return prompt_surface.PromptBlock(key="rules", lines=lines)


def _instruction_block(
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_instruction"])

    if language == "da":
        if variant == "solve":
            line = "Find et kortest muligt gyldigt regneudtryk, der rammer måltallet præcist."
        elif variant == "unique":
            line = "Det korteste gyldige regneudtryk er entydigt."
        else:
            line = "Alle oplysninger skal passe med det samme korteste regneudtryk."
        return prompt_surface.PromptBlock(key="instruction", lines=(line,))

    if variant == "solve":
        line = "Find a shortest valid arithmetic expression that hits the target exactly."
    elif variant == "unique":
        line = "The shortest valid arithmetic expression is unique."
    else:
        line = "All information must fit the same shortest arithmetic expression."
    return prompt_surface.PromptBlock(key="instruction", lines=(line,))


def _answer_block(
    language: str,
    surface_plan: dict[str, object],
    allowed_ops: tuple[str, ...],
    use_all_numbers: bool,
) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_answer"])
    example = _example_expression(allowed_ops, use_all_numbers)

    if language == "da":
        if variant == "respond":
            lines = (
                "Inde i det endelige svar skal du skrive præcis én udtryk-linje.",
                "Inde i svarindholdet må der ikke være forklaring, nummerering eller ekstra tekst.",
            )
        elif variant == "write":
            lines = (
                "Inde i det endelige svar skal det være ét råt regneudtryk.",
                "Brug parenteser, når det hjælper læsbarheden.",
            )
        else:
            lines = (
                "Inde i svarindholdet skal der kun stå det færdige regneudtryk på én linje.",
                "Brug kun tal fra opgaven og de tilladte operatorer.",
            )
        lines += (f"Format:\n{example}",)
        return prompt_surface.PromptBlock(key="answer", lines=lines)

    if variant == "respond":
        lines = (
            "Inside the final response, write exactly one expression line.",
            "Inside the answer content, do not add explanation, numbering, or extra text.",
        )
    elif variant == "write":
        lines = (
            "Inside the final response, it should be one raw arithmetic expression.",
            "Use parentheses when they help readability.",
        )
    else:
        lines = (
            "Inside the answer content, write only the finished arithmetic expression on one line.",
            "Use only puzzle numbers and the allowed operators.",
        )
    lines += (f"Format:\n{example}",)
    return prompt_surface.PromptBlock(key="answer", lines=lines)


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]


def _render_operator_text(allowed_ops: tuple[str, ...]) -> str:
    return ", ".join(f"`{operator}`" for operator in allowed_ops)


def _example_expression(allowed_ops: tuple[str, ...], use_all_numbers: bool) -> str:
    if use_all_numbers:
        if "/" in allowed_ops:
            return "(((8/2)+3)*4)"
        if "-" in allowed_ops:
            return "(((9-4)*3)+2)"
        return "((2+3)*(4+5))"

    if "/" in allowed_ops:
        return "((8/2)+(3*4))"
    if "-" in allowed_ops:
        return "((9-4)*3)"
    return "(2+(3*4))"
