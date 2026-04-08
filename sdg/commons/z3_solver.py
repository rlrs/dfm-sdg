from collections.abc import Mapping

import z3


def enumerate_int_models(
    solver: z3.Solver,
    variables: Mapping[str, z3.ArithRef],
    *,
    limit: int | None = None,
) -> list[dict[str, int]]:
    working = z3.Solver()
    working.add(solver.assertions())

    models: list[dict[str, int]] = []
    while working.check() == z3.sat:
        model = working.model()
        assignment = {
            name: model.eval(variable, model_completion=True).as_long()
            for name, variable in variables.items()
        }
        models.append(assignment)

        working.add(
            z3.Or(
                [
                    variable != assignment[name]
                    for name, variable in variables.items()
                ]
            )
        )
        if limit is not None and len(models) >= limit:
            break

    return models


def count_int_models(
    solver: z3.Solver,
    variables: Mapping[str, z3.ArithRef],
    *,
    limit: int | None = None,
) -> tuple[int, bool]:
    working = z3.Solver()
    working.add(solver.assertions())

    count = 0
    while working.check() == z3.sat:
        model = working.model()
        assignment = {
            name: model.eval(variable, model_completion=True).as_long()
            for name, variable in variables.items()
        }
        count += 1

        working.add(
            z3.Or(
                [
                    variable != assignment[name]
                    for name, variable in variables.items()
                ]
            )
        )
        if limit is not None and count >= limit:
            return count, working.check() != z3.sat

    return count, True


def count_int_models_in_place(
    solver: z3.Solver,
    variables: Mapping[str, z3.ArithRef],
    *,
    extra_constraints: tuple[z3.BoolRef, ...] = (),
    limit: int | None = None,
) -> tuple[int, bool]:
    solver.push()
    if extra_constraints:
        solver.add(*extra_constraints)

    count = 0
    while solver.check() == z3.sat:
        model = solver.model()
        assignment = {
            name: model.eval(variable, model_completion=True).as_long()
            for name, variable in variables.items()
        }
        count += 1

        solver.add(
            z3.Or(
                [
                    variable != assignment[name]
                    for name, variable in variables.items()
                ]
            )
        )
        if limit is not None and count >= limit:
            complete = solver.check() != z3.sat
            solver.pop()
            return count, complete

    solver.pop()
    return count, True


def has_model_with(
    solver: z3.Solver,
    *extra_constraints: z3.BoolRef,
) -> bool:
    solver.push()
    if extra_constraints:
        solver.add(*extra_constraints)
    satisfiable = solver.check() == z3.sat
    solver.pop()
    return satisfiable
