from dataclasses import dataclass
from itertools import combinations, product
from random import Random
from typing import Any

from sdg.commons import diversity, prompt_surface


@dataclass(frozen=True)
class Statement:
    kind: str
    args: tuple[object, ...]


@dataclass(frozen=True)
class KnightsAndKnavesSurfaceSpec:
    plan: prompt_surface.SurfacePlan


@dataclass(frozen=True)
class KnightsAndKnavesPuzzle:
    speakers: tuple[str, ...]
    solution_roles: tuple[bool, ...]
    statements: tuple[Statement, ...]
    prompt: str


SURFACE_SPECS = {
    "dialogue": KnightsAndKnavesSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="dialogue",
            block_order=("intro", "rules", "dialogue", "instruction", "answer"),
        ),
    ),
    "briefing": KnightsAndKnavesSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="briefing",
            block_order=("intro", "instruction", "rules", "dialogue", "answer"),
        ),
    ),
    "deduce": KnightsAndKnavesSurfaceSpec(
        plan=prompt_surface.SurfacePlan(
            key="deduce",
            block_order=("intro", "rules", "answer", "instruction", "dialogue"),
        ),
    ),
}

RECIPES = (
    {
        "recipe_id": "easy_pair_direct_2",
        "difficulty": "easy",
        "prompt_style": "dialogue",
        "speaker_count": 2,
        "statement_style": "direct",
        "statement_kinds": ("role", "same_kind", "different_kind", "count_total"),
        "sample_attempts": 80,
    },
    {
        "recipe_id": "easy_trio_direct_3",
        "difficulty": "easy",
        "prompt_style": "briefing",
        "speaker_count": 3,
        "statement_style": "direct",
        "statement_kinds": ("role", "same_kind", "different_kind", "count_total", "subset_exact"),
        "sample_attempts": 96,
    },
    {
        "recipe_id": "medium_trio_count_3",
        "difficulty": "medium",
        "prompt_style": "dialogue",
        "speaker_count": 3,
        "statement_style": "count",
        "statement_kinds": (
            "role",
            "same_kind",
            "different_kind",
            "count_total",
            "subset_exact",
            "exactly_one_of_pair",
            "would_say_role",
        ),
        "sample_attempts": 128,
    },
    {
        "recipe_id": "hard_quartet_mixed_4",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "speaker_count": 4,
        "statement_style": "mixed",
        "statement_kinds": (
            "role",
            "same_kind",
            "different_kind",
            "count_total",
            "subset_exact",
            "exactly_one_of_pair",
            "conditional_role",
            "would_say_role",
            "would_say_same_kind",
        ),
        "sample_attempts": 180,
    },
    {
        "recipe_id": "hard_quintet_mixed_5",
        "difficulty": "hard",
        "prompt_style": "deduce",
        "speaker_count": 5,
        "statement_style": "mixed",
        "statement_kinds": (
            "role",
            "same_kind",
            "different_kind",
            "count_total",
            "subset_exact",
            "exactly_one_of_pair",
            "conditional_role",
            "would_say_role",
            "would_say_same_kind",
        ),
        "sample_attempts": 260,
    },
)

NAME_POOLS = {
    "en": ("Alice", "Ben", "Clara", "Daniel", "Emma", "Felix", "Grace", "Henry", "Iris", "Jonas"),
    "da": ("Anna", "Bo", "Clara", "Emil", "Ida", "Karl", "Maja", "Niels", "Sofie", "Troels"),
}

ROLE_WORDS = {
    "en": {True: "knight", False: "knave"},
    "da": {True: "ridder", False: "skurk"},
}

PARSE_ROLE_WORDS = {
    "knight": True,
    "knave": False,
    "ridder": True,
    "skurk": False,
}

COUNT_WORDS = {
    "en": {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"},
    "da": {0: "nul", 1: "én", 2: "to", 3: "tre", 4: "fire", 5: "fem"},
}


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
            "speakers": list(puzzle.speakers),
            "solution_roles": list(puzzle.solution_roles),
            "statements": [_statement_to_dict(statement) for statement in puzzle.statements],
        },
        "sources": [{"kind": "dolci_subset", "value": "knightsandknaves"}],
        "meta": {
            "family": "knightsandknaves_logic",
            "domain": "logic_puzzles",
            "prompt_language": language,
            "target_language": language,
            "clue_count": len(puzzle.statements),
            "given_count": len(puzzle.statements),
            "speaker_count": len(puzzle.speakers),
            "output_format": "role_assignment",
            "recipe_id": recipe["recipe_id"],
            "difficulty": recipe["difficulty"],
            "prompt_style": recipe["prompt_style"],
            "statement_style": recipe["statement_style"],
            **chosen_surface,
        },
    }


def parse_target(text: str, hidden: dict[str, object]) -> tuple[bool, ...] | None:
    speakers = tuple(str(name) for name in hidden["speakers"])
    speaker_lookup = {speaker.lower(): index for index, speaker in enumerate(speakers)}
    parsed_roles: dict[int, bool] = {}

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    for line in lines:
        normalized = line.strip()
        normalized = normalized.lstrip("*•- ").strip()
        normalized = normalized.replace("=", ":")
        if ":" not in normalized:
            return None
        name_text, role_text = normalized.split(":", 1)
        speaker_key = name_text.strip().lower()
        role_key = role_text.strip().lower().strip("`'\". ")
        if speaker_key not in speaker_lookup:
            return None
        if role_key not in PARSE_ROLE_WORDS:
            return None
        parsed_roles[speaker_lookup[speaker_key]] = PARSE_ROLE_WORDS[role_key]

    if set(parsed_roles) != set(range(len(speakers))):
        return None
    return tuple(parsed_roles[index] for index in range(len(speakers)))


def canonical_target(parsed: tuple[bool, ...], hidden: dict[str, object]) -> str:
    language = str(hidden["language"])
    speakers = tuple(str(name) for name in hidden["speakers"])
    role_words = ROLE_WORDS[language]
    return "\n".join(
        f"{speaker}: {role_words[role]}"
        for speaker, role in zip(speakers, parsed, strict=True)
    )


def format_target(speakers: tuple[str, ...], roles: tuple[bool, ...], language: str) -> str:
    role_words = ROLE_WORDS[language]
    return "\n".join(
        f"{speaker}: {role_words[role]}"
        for speaker, role in zip(speakers, roles, strict=True)
    )


def answer_contract(hidden: dict[str, object], language: str) -> str:
    speakers = tuple(str(name) for name in hidden["speakers"])
    role_words = ROLE_WORDS[language]
    example = "\n".join(f"{speaker}: {role_words[True]}" for speaker in speakers)

    if language == "da":
        return (
            "I din svarblok skal du skrive præcis én rå linje per person i den viste rækkefølge.\n"
            "Skriv kun linjer i formatet `Navn: rolle`.\n"
            f"Tilladte roller er `{role_words[True]}` og `{role_words[False]}`.\n"
            f"Format:\n{example}"
        )

    return (
        "In your answer block, write exactly one raw line per speaker in the shown order.\n"
        "Write only lines in the format `Name: role`.\n"
        f"The allowed roles are `{role_words[True]}` and `{role_words[False]}`.\n"
        f"Format:\n{example}"
    )


def is_correct(parsed: tuple[bool, ...], hidden: dict[str, object]) -> bool:
    return list(parsed) == hidden["solution_roles"]


def clues_resolve_uniquely(hidden: dict[str, object]) -> bool:
    statements = tuple(_statement_from_dict(payload) for payload in hidden["statements"])
    solutions = _solve_assignments(len(hidden["speakers"]), statements, limit=2)
    if len(solutions) != 1:
        return False
    return list(solutions[0]) == hidden["solution_roles"]


def dataset_checks(rows: list[dict[str, object]], planned: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        "recipe_coverage": diversity.compare_planned_to_observed(planned, rows, ("recipe_id",), observed_getter=_meta_getter),
        "difficulty_coverage": diversity.compare_planned_to_observed(planned, rows, ("difficulty",), observed_getter=_meta_getter),
        "prompt_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("prompt_style",), observed_getter=_meta_getter),
        "statement_style_coverage": diversity.compare_planned_to_observed(planned, rows, ("statement_style",), observed_getter=_meta_getter),
        "speaker_count_coverage": diversity.compare_planned_to_observed(planned, rows, ("speaker_count",), observed_getter=_meta_getter),
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
) -> KnightsAndKnavesPuzzle:
    speaker_count = int(recipe["speaker_count"])

    for _ in range(int(recipe["sample_attempts"])):
        speakers = tuple(rng.sample(NAME_POOLS[language], speaker_count))
        solution_roles = _sample_roles(speaker_count, rng)
        candidates = _candidate_statements(
            speakers,
            solution_roles,
            tuple(str(kind) for kind in recipe["statement_kinds"]),
        )
        if any(not options for options in candidates):
            continue

        statements = _select_statements(candidates, solution_roles, rng)
        if statements is None:
            continue

        prompt = _format_prompt(
            speakers,
            statements,
            language=language,
            recipe=recipe,
            surface_plan=surface_plan,
        )
        return KnightsAndKnavesPuzzle(
            speakers=speakers,
            solution_roles=solution_roles,
            statements=statements,
            prompt=prompt,
        )

    raise AssertionError("failed to generate knights and knaves puzzle")


def _sample_roles(count: int, rng: Random) -> tuple[bool, ...]:
    while True:
        roles = tuple(bool(rng.randint(0, 1)) for _ in range(count))
        if any(roles) and not all(roles):
            return roles


def _candidate_statements(
    speakers: tuple[str, ...],
    solution_roles: tuple[bool, ...],
    allowed_kinds: tuple[str, ...],
) -> tuple[tuple[Statement, ...], ...]:
    all_statements = _statement_inventory(len(speakers), allowed_kinds)
    candidates: list[tuple[Statement, ...]] = []

    for speaker_index, role in enumerate(solution_roles):
        desired_truth = role
        options = [
            statement
            for statement in all_statements
            if not _statement_is_awkward_for_speaker(statement, speaker_index)
            if _statement_truth(statement, solution_roles) is desired_truth
        ]
        deduped = tuple(_dedupe_statements(options))
        candidates.append(deduped)

    return tuple(candidates)


def _statement_inventory(speaker_count: int, allowed_kinds: tuple[str, ...]) -> tuple[Statement, ...]:
    inventory: list[Statement] = []
    speaker_indices = tuple(range(speaker_count))

    if "role" in allowed_kinds:
        for target in speaker_indices:
            inventory.append(Statement("role", (target, True)))
            inventory.append(Statement("role", (target, False)))

    if "same_kind" in allowed_kinds or "different_kind" in allowed_kinds:
        for left, right in combinations(speaker_indices, 2):
            if "same_kind" in allowed_kinds:
                inventory.append(Statement("same_kind", (left, right)))
            if "different_kind" in allowed_kinds:
                inventory.append(Statement("different_kind", (left, right)))

    if "count_total" in allowed_kinds:
        for count in range(speaker_count + 1):
            inventory.append(Statement("count_total", (count,)))

    if "subset_exact" in allowed_kinds:
        for subset_size in (2, 3, 4):
            if subset_size > speaker_count:
                continue
            for subset in combinations(speaker_indices, subset_size):
                for count in range(subset_size + 1):
                    inventory.append(Statement("subset_exact", (tuple(subset), count)))

    if "exactly_one_of_pair" in allowed_kinds:
        for left, right in combinations(speaker_indices, 2):
            inventory.append(Statement("exactly_one_of_pair", (left, right)))

    if "conditional_role" in allowed_kinds:
        for premise in speaker_indices:
            for consequence in speaker_indices:
                if premise == consequence:
                    continue
                inventory.append(Statement("conditional_role", (premise, True, consequence, True)))
                inventory.append(Statement("conditional_role", (premise, False, consequence, True)))
                inventory.append(Statement("conditional_role", (premise, True, consequence, False)))
                inventory.append(Statement("conditional_role", (premise, False, consequence, False)))

    if "would_say_role" in allowed_kinds:
        for quoted_speaker in speaker_indices:
            for target in speaker_indices:
                if quoted_speaker == target:
                    continue
                inventory.append(Statement("would_say_role", (quoted_speaker, target, True)))
                inventory.append(Statement("would_say_role", (quoted_speaker, target, False)))

    if "would_say_same_kind" in allowed_kinds:
        for quoted_speaker in speaker_indices:
            for left, right in combinations(speaker_indices, 2):
                if quoted_speaker in {left, right}:
                    continue
                inventory.append(Statement("would_say_same_kind", (quoted_speaker, left, right, True)))
                inventory.append(Statement("would_say_same_kind", (quoted_speaker, left, right, False)))

    return tuple(inventory)


def _dedupe_statements(statements: list[Statement]) -> list[Statement]:
    seen: set[tuple[str, tuple[object, ...]]] = set()
    deduped: list[Statement] = []
    for statement in statements:
        key = (statement.kind, statement.args)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(statement)
    return deduped


def _statement_is_awkward_for_speaker(statement: Statement, speaker_index: int) -> bool:
    if statement.kind == "would_say_role":
        quoted_speaker = int(statement.args[0])
        return quoted_speaker == speaker_index
    if statement.kind == "would_say_same_kind":
        quoted_speaker = int(statement.args[0])
        return quoted_speaker == speaker_index
    return False


def _select_statements(
    candidates: tuple[tuple[Statement, ...], ...],
    solution_roles: tuple[bool, ...],
    rng: Random,
) -> tuple[Statement, ...] | None:
    assignments = tuple(product((False, True), repeat=len(solution_roles)))
    intended = solution_roles

    def search(
        chosen: dict[int, Statement],
        remaining_assignments: tuple[tuple[bool, ...], ...],
    ) -> dict[int, Statement] | None:
        if len(chosen) == len(solution_roles):
            if remaining_assignments == (intended,):
                return chosen
            return None

        next_speaker, ordered_candidates = _next_speaker_choices(chosen, candidates, remaining_assignments, intended, rng)
        if next_speaker is None:
            return None

        for statement, next_assignments in ordered_candidates:
            updated = dict(chosen)
            updated[next_speaker] = statement
            result = search(updated, next_assignments)
            if result is not None:
                return result
        return None

    result = search({}, assignments)
    if result is None:
        return None
    return tuple(result[index] for index in range(len(solution_roles)))


def _next_speaker_choices(
    chosen: dict[int, Statement],
    candidates: tuple[tuple[Statement, ...], ...],
    remaining_assignments: tuple[tuple[bool, ...], ...],
    intended: tuple[bool, ...],
    rng: Random,
) -> tuple[int | None, list[tuple[Statement, tuple[tuple[bool, ...], ...]]]]:
    best_speaker: int | None = None
    best_options: list[tuple[Statement, tuple[tuple[bool, ...], ...]]] = []
    best_size: int | None = None

    for speaker_index, options in enumerate(candidates):
        if speaker_index in chosen:
            continue
        scored: list[tuple[Statement, tuple[tuple[bool, ...], ...]]] = []
        for statement in options:
            next_assignments = tuple(
                assignment
                for assignment in remaining_assignments
                if _speaker_consistent(speaker_index, statement, assignment)
            )
            if intended not in next_assignments:
                continue
            scored.append((statement, next_assignments))

        if not scored:
            return None, []

        rng.shuffle(scored)
        scored.sort(key=lambda item: len(item[1]))
        size = len(scored[0][1])
        if best_size is None or size < best_size:
            best_size = size
            best_speaker = speaker_index
            best_options = scored

    return best_speaker, best_options


def _speaker_consistent(speaker_index: int, statement: Statement, assignment: tuple[bool, ...]) -> bool:
    return assignment[speaker_index] is _statement_truth(statement, assignment)


def _statement_truth(statement: Statement, assignment: tuple[bool, ...]) -> bool:
    if statement.kind == "role":
        target, role = statement.args
        return assignment[int(target)] is bool(role)

    if statement.kind == "same_kind":
        left, right = statement.args
        return assignment[int(left)] is assignment[int(right)]

    if statement.kind == "different_kind":
        left, right = statement.args
        return assignment[int(left)] is not assignment[int(right)]

    if statement.kind == "count_total":
        count = int(statement.args[0])
        return sum(assignment) == count

    if statement.kind == "subset_exact":
        subset, count = statement.args
        total = sum(assignment[int(index)] for index in tuple(subset))
        return total == int(count)

    if statement.kind == "exactly_one_of_pair":
        left, right = statement.args
        return assignment[int(left)] is not assignment[int(right)]

    if statement.kind == "conditional_role":
        premise, premise_role, consequence, consequence_role = statement.args
        if assignment[int(premise)] is not bool(premise_role):
            return True
        return assignment[int(consequence)] is bool(consequence_role)

    if statement.kind == "would_say_role":
        quoted_speaker, target, role = statement.args
        quoted_truth = assignment[int(target)] is bool(role)
        return assignment[int(quoted_speaker)] is quoted_truth

    if statement.kind == "would_say_same_kind":
        quoted_speaker, left, right, same_kind = statement.args
        relation_is_true = assignment[int(left)] is assignment[int(right)]
        quoted_truth = relation_is_true if bool(same_kind) else not relation_is_true
        return assignment[int(quoted_speaker)] is quoted_truth

    raise AssertionError(f"unsupported statement kind: {statement.kind}")


def _solve_assignments(
    speaker_count: int,
    statements: tuple[Statement, ...],
    *,
    limit: int,
) -> list[tuple[bool, ...]]:
    solutions: list[tuple[bool, ...]] = []
    for assignment in product((False, True), repeat=speaker_count):
        if all(_speaker_consistent(index, statement, assignment) for index, statement in enumerate(statements)):
            solutions.append(tuple(bool(value) for value in assignment))
            if len(solutions) >= limit:
                break
    return solutions


def _format_prompt(
    speakers: tuple[str, ...],
    statements: tuple[Statement, ...],
    *,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> str:
    prompt_style = str(recipe["prompt_style"])
    surface_spec = SURFACE_SPECS[prompt_style]
    blocks = {
        "intro": _intro_block(len(speakers), language, recipe, surface_plan),
        "rules": _rules_block(speakers, language, surface_plan),
        "dialogue": _dialogue_block(speakers, statements, language, surface_plan),
        "instruction": _instruction_block(language, surface_plan),
        "answer": _answer_block(speakers, language, surface_plan),
    }
    return prompt_surface.render_prompt(blocks, surface_spec.plan)


def _intro_block(
    speaker_count: int,
    language: str,
    recipe: dict[str, Any],
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    intro_style = str(surface_plan["surface_intro"])
    statement_style = str(recipe["statement_style"])

    if language == "da":
        if intro_style == "context":
            lines = (
                f"På en ø møder du {speaker_count} personer, som hver enten er ridder eller skurk.",
            )
        elif intro_style == "assignment":
            lines = (
                f"Bestem rollerne for {speaker_count} personer i en riddere-og-skurke-opgave.",
            )
        else:
            lines = (
                f"Løs en riddere-og-skurke-opgave med {speaker_count} personer.",
            )
        if statement_style == "mixed":
            lines += ("Nogle udsagn handler direkte om roller, mens andre taler om, hvad en anden person ville sige.",)
        return prompt_surface.PromptBlock(key="intro", lines=lines)

    if intro_style == "context":
        lines = (f"On an island you meet {speaker_count} inhabitants, each of whom is either a knight or a knave.",)
    elif intro_style == "assignment":
        lines = (f"Determine the roles of {speaker_count} speakers in a knights-and-knaves puzzle.",)
    else:
        lines = (f"Solve a knights-and-knaves puzzle with {speaker_count} speakers.",)
    if statement_style == "mixed":
        lines += ("Some statements are direct role claims, while others describe what another speaker would say.",)
    return prompt_surface.PromptBlock(key="intro", lines=lines)


def _rules_block(
    speakers: tuple[str, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    speaker_list = ", ".join(speakers)

    if language == "da":
        if clue_style == "compact":
            lines = (
                "En ridder taler altid sandt.",
                "En skurk lyver altid.",
                f"Personerne er: {speaker_list}.",
            )
        else:
            lines = (
                "Hver person er enten ridder eller skurk.",
                "Riddere siger altid sandheden, og skurke siger altid noget falsk.",
                f"Du skal bestemme rollerne for: {speaker_list}.",
            )
        return prompt_surface.PromptBlock(key="rules", lines=lines, numbered=True)

    if clue_style == "compact":
        lines = (
            "A knight always tells the truth.",
            "A knave always lies.",
            f"The speakers are: {speaker_list}.",
        )
    else:
        lines = (
            "Each speaker is either a knight or a knave.",
            "Knights always make true statements, and knaves always make false statements.",
            f"You must determine the roles of: {speaker_list}.",
        )
    return prompt_surface.PromptBlock(key="rules", lines=lines, numbered=True)


def _dialogue_block(
    speakers: tuple[str, ...],
    statements: tuple[Statement, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    clue_style = str(surface_plan["surface_clue"])
    lines = tuple(
        _render_statement_line(
            speaker,
            statement,
            speakers,
            language=language,
            clue_style=clue_style,
        )
        for speaker, statement in zip(speakers, statements, strict=True)
    )
    heading = "Udsagn" if language == "da" else "Statements"
    return prompt_surface.PromptBlock(key="dialogue", heading=heading, lines=lines)


def _instruction_block(language: str, surface_plan: dict[str, object]) -> prompt_surface.PromptBlock:
    variant = str(surface_plan["surface_instruction"])
    if language == "da":
        if variant == "solve":
            line = "Udsagnene bestemmer præcis én løsning."
        elif variant == "unique":
            line = "Der er én entydig fordeling af riddere og skurke."
        else:
            line = "Alle udsagn skal passe med den samme rollefordeling."
    else:
        if variant == "solve":
            line = "The statements determine exactly one solution."
        elif variant == "unique":
            line = "There is one unique assignment of knights and knaves."
        else:
            line = "All statements must fit the same role assignment."
    return prompt_surface.PromptBlock(key="instruction", lines=(line,))


def _answer_block(
    speakers: tuple[str, ...],
    language: str,
    surface_plan: dict[str, object],
) -> prompt_surface.PromptBlock:
    answer_style = str(surface_plan["surface_answer"])
    role_words = ROLE_WORDS[language]
    example = "\n".join(f"{speaker}: {role_words[True]}" for speaker in speakers)

    if language == "da":
        if answer_style == "respond":
            lines = (
                "Inde i det endelige svar skal du bruge én linje per person i den viste rækkefølge.",
                f"Brug kun rollerne `{role_words[True]}` og `{role_words[False]}`.",
            )
        elif answer_style == "write":
            lines = (
                "Inde i det endelige svar skal hver linje have formen `Navn: rolle`.",
                "Behold navnene i samme rækkefølge som i opgaven.",
            )
        else:
            lines = (
                "Selve svarindholdet skal være den fulde rollefordeling med én linje per person.",
                "Brug formatet `Navn: rolle`.",
            )
        lines += (f"Format:\n{example}",)
        return prompt_surface.PromptBlock(key="answer", lines=lines)

    if answer_style == "respond":
        lines = (
            "Inside the final response, use one line per speaker in the shown order.",
            f"Use only the roles `{role_words[True]}` and `{role_words[False]}`.",
        )
    elif answer_style == "write":
        lines = (
            "Inside the final response, each line should have the form `Name: role`.",
            "Keep the speakers in the same order as in the puzzle.",
        )
    else:
        lines = (
            "The answer content should be the full role assignment with one line per speaker.",
            "Use the format `Name: role`.",
        )
    lines += (f"Format:\n{example}",)
    return prompt_surface.PromptBlock(key="answer", lines=lines)


def _render_statement_line(
    speaker: str,
    statement: Statement,
    speakers: tuple[str, ...],
    *,
    language: str,
    clue_style: str,
) -> str:
    content = _statement_text(statement, speakers, language)
    if clue_style == "compact":
        return f'{speaker}: "{content}"'
    if clue_style == "deductive":
        if language == "da":
            return f'"{content}" siger {speaker}.'
        return f'"{content}" says {speaker}.'
    if language == "da":
        variants = (
            f'{speaker} siger: "{content}"',
            f'"{content}" siger {speaker}.',
            f'{speaker} påstår: "{content}"',
        )
        return variants[_variant_index(speaker, content) % len(variants)]
    variants = (
        f'{speaker} says: "{content}"',
        f'"{content}" says {speaker}.',
        f'{speaker} claims: "{content}"',
    )
    return variants[_variant_index(speaker, content) % len(variants)]


def _statement_text(statement: Statement, speakers: tuple[str, ...], language: str) -> str:
    role_words = ROLE_WORDS[language]

    if statement.kind == "role":
        target, role = statement.args
        if language == "da":
            return f"{speakers[int(target)]} er en {role_words[bool(role)]}."
        article = "a"
        return f"{speakers[int(target)]} is {article} {role_words[bool(role)]}."

    if statement.kind == "same_kind":
        left, right = statement.args
        if language == "da":
            variants = (
                f"{speakers[int(left)]} og {speakers[int(right)]} er af samme slags.",
                f"{speakers[int(left)]} og {speakers[int(right)]} er begge riddere eller begge skurke.",
            )
            return variants[_variant_index(str(left), str(right)) % len(variants)]
        variants = (
            f"{speakers[int(left)]} and {speakers[int(right)]} are the same kind.",
            f"{speakers[int(left)]} and {speakers[int(right)]} are either both knights or both knaves.",
        )
        return variants[_variant_index(str(left), str(right)) % len(variants)]

    if statement.kind == "different_kind":
        left, right = statement.args
        if language == "da":
            variants = (
                f"{speakers[int(left)]} og {speakers[int(right)]} er af forskellig slags.",
                f"Den ene af {speakers[int(left)]} og {speakers[int(right)]} er ridder, og den anden er skurk.",
            )
            return variants[_variant_index(str(left), str(right)) % len(variants)]
        variants = (
            f"{speakers[int(left)]} and {speakers[int(right)]} are different kinds.",
            f"One of {speakers[int(left)]} and {speakers[int(right)]} is a knight, and the other is a knave.",
        )
        return variants[_variant_index(str(left), str(right)) % len(variants)]

    if statement.kind == "count_total":
        count = int(statement.args[0])
        word = COUNT_WORDS[language][count]
        if count == 0:
            if language == "da":
                return "Ingen af os er riddere."
            return "None of us are knights."
        if language == "da":
            noun = "ridder" if count == 1 else "riddere"
            return f"Præcis {word} af os er {noun}."
        if count == 1:
            return "Exactly one of us is a knight."
        noun = "knights"
        return f"Exactly {word} of us are {noun}."

    if statement.kind == "subset_exact":
        subset, count = statement.args
        names = [speakers[int(index)] for index in tuple(subset)]
        joined = _join_names(names, language)
        word = COUNT_WORDS[language][int(count)]
        if int(count) == 0 and len(names) == 2:
            if language == "da":
                return f"Hverken {names[0]} eller {names[1]} er ridder."
            return f"Neither {names[0]} nor {names[1]} is a knight."
        if language == "da":
            noun = "ridder" if int(count) == 1 else "riddere"
            return f"Præcis {word} af {joined} er {noun}."
        if int(count) == 1:
            return f"Exactly one of {joined} is a knight."
        noun = "knights"
        return f"Exactly {word} of {joined} are {noun}."

    if statement.kind == "exactly_one_of_pair":
        left, right = statement.args
        if language == "da":
            variants = (
                f"Præcis én af {speakers[int(left)]} og {speakers[int(right)]} er ridder.",
                f"Én af {speakers[int(left)]} og {speakers[int(right)]} er ridder, og den anden er skurk.",
            )
            return variants[_variant_index(str(left), str(right)) % len(variants)]
        variants = (
            f"Exactly one of {speakers[int(left)]} and {speakers[int(right)]} is a knight.",
            f"One of {speakers[int(left)]} and {speakers[int(right)]} is a knight and the other is a knave.",
        )
        return variants[_variant_index(str(left), str(right)) % len(variants)]

    if statement.kind == "conditional_role":
        premise, premise_role, consequence, consequence_role = statement.args
        premise_name = speakers[int(premise)]
        consequence_name = speakers[int(consequence)]
        premise_role_word = role_words[bool(premise_role)]
        consequence_role_word = role_words[bool(consequence_role)]
        if language == "da":
            if bool(premise_role):
                return f"Hvis {premise_name} er ridder, så er {consequence_name} {consequence_role_word}."
            return f"Hvis {premise_name} er skurk, så er {consequence_name} {consequence_role_word}."
        article = "a"
        if bool(premise_role):
            return f"If {premise_name} is {article} {premise_role_word}, then {consequence_name} is {article} {consequence_role_word}."
        return f"If {premise_name} is {article} {premise_role_word}, then {consequence_name} is {article} {consequence_role_word}."

    if statement.kind == "would_say_role":
        quoted_speaker, target, role = statement.args
        if language == "da":
            variants = (
                f"{speakers[int(quoted_speaker)]} ville sige, at {speakers[int(target)]} er {role_words[bool(role)]}.",
                f"Hvis man spurgte {speakers[int(quoted_speaker)]}, ville {speakers[int(quoted_speaker)]} sige, at {speakers[int(target)]} er {role_words[bool(role)]}.",
            )
            return variants[_variant_index(str(quoted_speaker), str(target)) % len(variants)]
        article = "a"
        variants = (
            f"{speakers[int(quoted_speaker)]} would say that {speakers[int(target)]} is {article} {role_words[bool(role)]}.",
            f"If you asked {speakers[int(quoted_speaker)]}, {speakers[int(quoted_speaker)]} would say that {speakers[int(target)]} is {article} {role_words[bool(role)]}.",
        )
        return variants[_variant_index(str(quoted_speaker), str(target)) % len(variants)]

    if statement.kind == "would_say_same_kind":
        quoted_speaker, left, right, same_kind = statement.args
        quoted_name = speakers[int(quoted_speaker)]
        left_name = speakers[int(left)]
        right_name = speakers[int(right)]
        if language == "da":
            if bool(same_kind):
                return f"{quoted_name} ville sige, at {left_name} og {right_name} er af samme slags."
            return f"{quoted_name} ville sige, at {left_name} og {right_name} er af forskellig slags."
        if bool(same_kind):
            return f"{quoted_name} would say that {left_name} and {right_name} are the same kind."
        return f"{quoted_name} would say that {left_name} and {right_name} are different kinds."

    raise AssertionError(f"unsupported statement kind: {statement.kind}")


def _join_names(names: list[str], language: str) -> str:
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        conjunction = "og" if language == "da" else "and"
        return f"{names[0]} {conjunction} {names[1]}"
    conjunction = "og" if language == "da" else "and"
    if language == "da":
        return ", ".join(names[:-1]) + f" {conjunction} {names[-1]}"
    return ", ".join(names[:-1]) + f", {conjunction} {names[-1]}"


def _variant_index(*parts: str) -> int:
    return sum(ord(char) for part in parts for char in part)


def _statement_to_dict(statement: Statement) -> dict[str, object]:
    serialized_args: list[object] = []
    for value in statement.args:
        if isinstance(value, tuple):
            serialized_args.append(list(value))
            continue
        serialized_args.append(value)
    return {
        "kind": statement.kind,
        "args": serialized_args,
    }


def _statement_from_dict(payload: dict[str, object]) -> Statement:
    args: list[object] = []
    for value in payload["args"]:
        if isinstance(value, list):
            args.append(tuple(value))
            continue
        args.append(value)
    return Statement(kind=str(payload["kind"]), args=tuple(args))


def _meta_getter(row: dict[str, object], key: str) -> object:
    return row["meta"][key]
